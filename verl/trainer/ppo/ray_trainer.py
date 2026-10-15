# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import math
import os
import tempfile
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_think_split_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
    resolve_think_end_token_id,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import deprecated, load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


# How to handle rollouts that reach max_response_length: use, zero reward, or drop.
_CLIPPED_ROLLOUT_MODES = {"use", "zero", "drop"}
_IF_PPL_REWARD_STRATEGIES = {"rank", "anchor", "inverse_ppl", "raw_likelihood"}
_IF_PPL_RANK_REWARD_MODES = {"both", "high_only"}
_IF_PPL_ANCHOR_REWARD_MODES = {"both", "no_lower_zero", "lower_zero_only"}


def normalize_if_ppl_reward_strategy(strategy: Any) -> str:
    """Reject unknown reward strategies instead of silently applying rank shaping."""
    normalized = str(strategy).strip().lower()
    if normalized not in _IF_PPL_REWARD_STRATEGIES:
        raise ValueError(f"if_ppl_reward_strategy must be one of {_IF_PPL_REWARD_STRATEGIES}, got: {strategy}")
    return normalized


def normalize_clipped_rollout_mode(mode: Any) -> str:
    # Keep a few aliases so shell/env values are forgiving.
    normalized = str(mode or "use").strip().lower()
    aliases = {"keep": "use", "original": "use", "reward_zero": "zero", "zero_reward": "zero"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in _CLIPPED_ROLLOUT_MODES:
        raise ValueError(f"clipped_rollout_mode must be one of {_CLIPPED_ROLLOUT_MODES}, got: {mode}")
    return normalized


def normalize_if_ppl_rank_reward_mode(mode: Any) -> str:
    normalized = str(mode or "both").strip().lower().replace("-", "_")
    aliases = {"symmetric": "both", "bidirectional": "both", "penalty_only": "high_only", "high": "high_only"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in _IF_PPL_RANK_REWARD_MODES:
        raise ValueError(f"if_ppl_rank_reward_mode must be one of {_IF_PPL_RANK_REWARD_MODES}, got: {mode}")
    return normalized


def normalize_if_ppl_anchor_reward_mode(mode: Any) -> str:
    normalized = str(mode or "both").strip().lower().replace("-", "_")
    aliases = {
        "full": "both",
        "default": "both",
        "no_low_gate": "no_lower_zero",
        "no_lower_gate": "no_lower_zero",
        "upper_only": "no_lower_zero",
        "no_upper_bonus": "lower_zero_only",
        "zero_only": "lower_zero_only",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _IF_PPL_ANCHOR_REWARD_MODES:
        raise ValueError(f"if_ppl_anchor_reward_mode must be one of {_IF_PPL_ANCHOR_REWARD_MODES}, got: {mode}")
    return normalized


def normalize_bool_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def get_clipped_rollout_mask(batch: DataProto) -> torch.Tensor:
    # metric_utils treats response_length == max_response_length as clipped.
    response_mask = batch.batch["response_mask"]
    max_response_length = batch.batch["responses"].shape[-1]
    response_lengths = response_mask.sum(dim=-1)
    return torch.eq(response_lengths, max_response_length)


def apply_clipped_rollout_batch_mode(batch: DataProto, mode: Any) -> tuple[DataProto, dict[str, float]]:
    # Drop clipped rollouts before logprob/ref/actor work when requested.
    mode = normalize_clipped_rollout_mode(mode)
    clipped_mask = get_clipped_rollout_mask(batch)
    clipped_count = int(clipped_mask.sum().detach().cpu().item())
    original_count = len(batch)
    metrics = {
        "clipped_rollout/count": float(clipped_count),
        "clipped_rollout/ratio": float(clipped_count / max(original_count, 1)),
        "clipped_rollout/mode_use": float(mode == "use"),
        "clipped_rollout/mode_zero": float(mode == "zero"),
        "clipped_rollout/mode_drop": float(mode == "drop"),
        "clipped_rollout/dropped_count": 0.0,
        "clipped_rollout/kept_count": float(original_count),
    }
    if mode != "drop" or clipped_count == 0:
        return batch, metrics

    keep_mask = torch.logical_not(clipped_mask).detach().cpu()
    kept_count = int(keep_mask.sum().item())
    if kept_count == 0:
        raise ValueError("clipped_rollout_mode=drop would remove every rollout in the batch")
    metrics["clipped_rollout/dropped_count"] = float(clipped_count)
    metrics["clipped_rollout/kept_count"] = float(kept_count)
    return batch[keep_mask], metrics


def apply_clipped_rollout_reward_mode(
    batch: DataProto, reward_tensor: torch.Tensor, mode: Any
) -> tuple[torch.Tensor, dict[str, float]]:
    # In zero mode, clipped rollouts stay in-batch but receive no constraint/PPL reward.
    mode = normalize_clipped_rollout_mode(mode)
    clipped_mask = get_clipped_rollout_mask(batch)
    zero_count = int(clipped_mask.sum().detach().cpu().item()) if mode == "zero" else 0
    metrics = {"clipped_rollout/zero_reward_count": float(zero_count)}
    if mode != "zero" or zero_count == 0:
        return reward_tensor, metrics

    updated_reward = reward_tensor.clone()
    updated_reward[clipped_mask.to(updated_reward.device)] = 0.0
    return updated_reward, metrics


def _apply_if_likelihood_reward(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    ppl_reward_coeff: float = 1.0,
    clipped_rollout_mode: Any = "use",
    *,
    length_normalized: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Add raw or length-normalized reference likelihood, including IF=0 responses.

    NLL and N come from the frozen reference scoring the continuation given x.
    Neither variant uses anchors, ranks, thresholds, or a constraint-success gate.
    Empty or invalid sequences contribute zero. The trainer handles the constraint part
    of clipped-rollout zeroing; this function excludes its likelihood term too.
    """
    strategy = "inverse_ppl" if length_normalized else "raw_likelihood"
    coeff = float(ppl_reward_coeff)
    if not math.isfinite(coeff) or coeff < 0:
        raise ValueError(f"{strategy} requires a finite, nonnegative if_py_given_x_reward_coeff")
    required = ("p_y_given_x_nll", "p_y_given_x_token_count")
    missing = [key for key in required if key not in batch.non_tensor_batch]
    if missing:
        raise ValueError(f"{strategy} requires reference likelihood scores; missing fields: {missing}")

    size = reward_tensor.shape[0]
    nlls, token_counts = (batch.non_tensor_batch[key] for key in required)
    if len(nlls) != size or len(token_counts) != size:
        raise ValueError(f"{strategy} NLL and token counts must match the reward batch size")

    response_lengths = batch.batch["response_mask"].sum(dim=-1).detach().cpu().long()
    likelihoods = np.zeros(size, dtype=np.float64)
    valid = np.zeros(size, dtype=bool)
    for idx, (nll, count, length) in enumerate(zip(nlls, token_counts, response_lengths.tolist(), strict=True)):
        try:
            nll, count = float(nll), int(count)
        except (TypeError, ValueError, OverflowError):
            continue
        if length <= 0 or count <= 0 or not math.isfinite(nll) or nll < 0:
            continue
        valid[idx] = True
        likelihoods[idx] = math.exp(-nll / count) if length_normalized else math.exp(-nll)

    excluded = np.zeros(size, dtype=bool)
    if normalize_clipped_rollout_mode(clipped_rollout_mode) == "zero":
        excluded = get_clipped_rollout_mask(batch).detach().cpu().numpy().astype(bool)
    terms = torch.as_tensor(likelihoods, dtype=reward_tensor.dtype, device=reward_tensor.device) * coeff
    terms[torch.as_tensor(excluded, device=terms.device)] = 0.0
    updated_reward = reward_tensor.clone()
    rows = torch.nonzero(terms, as_tuple=True)[0]
    last_positions = (response_lengths - 1).clamp_min(0).to(updated_reward.device)
    updated_reward[rows, last_positions[rows]] += terms[rows]

    # Keep raw per-answer values in the batch for diagnostics as well as aggregates.
    field = "p_y_given_x_likelihood" if length_normalized else "p_y_given_x_raw_likelihood"
    batch.non_tensor_batch[field] = likelihoods.astype(object)
    prefix = f"if_{field}"
    base_scores = reward_tensor.sum(dim=-1)
    composed_scores = updated_reward.sum(dim=-1)
    metrics = {
        f"{prefix}/reward_coeff": coeff,
        f"{prefix}/valid_count": float(valid.sum()),
        f"{prefix}/invalid_count": float((~valid).sum()),
        f"{prefix}/clipped_excluded_count": float(excluded.sum()),
    }
    if not length_normalized:
        # Diagnose tiny probabilities without flooring, rescaling, or normalizing them.
        positive = valid & ~excluded & (likelihoods > 0) & (coeff > 0)
        metrics[f"{prefix}/zero_likelihood_count"] = float((valid & (likelihoods == 0)).sum())
        metrics[f"{prefix}/term_underflow_count"] = float((positive & (terms.detach().cpu().numpy() == 0)).sum())
        metrics[f"{prefix}/rounded_away_count"] = float(
            ((terms > 0) & (composed_scores == base_scores)).sum().detach().cpu().item()
        )
    distributions = {
        "likelihood": likelihoods[valid],
        "likelihood_reward": terms.detach().cpu().double().numpy(),
        "base_reward": base_scores.detach().cpu().double().numpy(),
        "composed_reward": composed_scores.detach().cpu().double().numpy(),
    }
    for name, values in distributions.items():
        for stat in ("mean", "std", "min", "max"):
            metrics[f"{prefix}/{name}/{stat}"] = float(getattr(values, stat)()) if values.size else 0.0
    return updated_reward, metrics


def apply_if_inverse_ppl_reward(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    ppl_reward_coeff: float = 1.0,
    clipped_rollout_mode: Any = "use",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Add coeff * exp(-NLL / N), the geometric mean reference token probability."""
    return _apply_if_likelihood_reward(
        batch, reward_tensor, ppl_reward_coeff, clipped_rollout_mode, length_normalized=True
    )


def apply_if_raw_likelihood_reward(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    ppl_reward_coeff: float = 1.0,
    clipped_rollout_mode: Any = "use",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Add coeff * exp(-summed NLL), the reference probability of the final answer given x."""
    return _apply_if_likelihood_reward(
        batch, reward_tensor, ppl_reward_coeff, clipped_rollout_mode, length_normalized=False
    )


def apply_if_ppl_rank_reward(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    ppl_reward_coeff: float = 1.0,
    rollout_n: int | None = None,
    metric_prefix: str = "if_p_y_given_x",
    ppl_key: str = "p_y_given_x_ppl",
    nll_key: str = "p_y_given_x_nll",
    token_count_key: str = "p_y_given_x_token_count",
    gate_scores: torch.Tensor | None = None,
    rank_reward_mode: Any = "both",
    ref_nll_key: str | None = None,
    ref_token_count_key: str | None = None,
    ref_ppl_gate_enabled: bool = False,
    ref_ppl_gate_margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    # Reward low-PPL and penalize high-PPL samples per prompt.
    # In high_only mode, keep the high-PPL penalty while suppressing the low-PPL bonus.
    # ``gate_scores`` is the original constraint reward snapshot (taken before any PPL shaping); when
    # bidirectional rewards are chained it keeps each direction from gating on the other's bonus.
    # Only rows with positive IF reward receive the PPL rank reward, preserving the IF gate.
    if ppl_key not in batch.non_tensor_batch or "uid" not in batch.non_tensor_batch:
        return reward_tensor, {}
    rank_reward_mode = normalize_if_ppl_rank_reward_mode(rank_reward_mode)

    ppls = []
    for value in batch.non_tensor_batch[ppl_key]:
        try:
            ppls.append(float(value))
        except (TypeError, ValueError):
            ppls.append(float("inf"))

    # Log corpus PPL as exp(total NLL / total continuation tokens), not mean(sample PPL).
    nlls = []
    for value in batch.non_tensor_batch.get(nll_key, []):
        try:
            nlls.append(float(value))
        except (TypeError, ValueError):
            nlls.append(float("inf"))
    token_counts = []
    for value in batch.non_tensor_batch.get(token_count_key, []):
        try:
            token_counts.append(int(value))
        except (TypeError, ValueError):
            token_counts.append(0)

    ref_nlls = []
    if ref_nll_key:
        for value in batch.non_tensor_batch.get(ref_nll_key, []):
            try:
                ref_nlls.append(float(value))
            except (TypeError, ValueError):
                ref_nlls.append(float("inf"))
    ref_token_counts = []
    if ref_token_count_key:
        for value in batch.non_tensor_batch.get(ref_token_count_key, []):
            try:
                ref_token_counts.append(int(value))
            except (TypeError, ValueError):
                ref_token_counts.append(0)

    base_scores = reward_tensor.sum(dim=-1)
    # Gate on the original constraint reward (snapshot) so chained directions don't see
    # each other's bonus. Falls back to the running reward when no snapshot is given (single-direction
    # use is unaffected). Materialized to a CPU float list once to avoid per-row device syncs.
    gate_source = base_scores if gate_scores is None else gate_scores
    gate_values = gate_source.detach().to(device="cpu", dtype=torch.float64).tolist()
    finite_ppl_mask = np.isfinite(np.asarray(ppls, dtype=np.float64))
    corpus_ppl_terms = [
        (nll, token_count)
        for nll, token_count, is_finite in zip(nlls, token_counts, finite_ppl_mask, strict=False)
        if is_finite and np.isfinite(nll) and token_count > 0
    ]
    total_nll = sum(nll for nll, _ in corpus_ppl_terms)
    total_tokens = sum(token_count for _, token_count in corpus_ppl_terms)
    corpus_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else 0.0
    bonus = torch.zeros_like(base_scores, dtype=reward_tensor.dtype)
    uids = batch.non_tensor_batch["uid"]
    top_k = 0
    high_k = max(int(rollout_n or 0) // 4, 1)
    low_ppl_candidate_count = 0
    high_ppl_candidate_count = 0
    ref_ppl_gate_blocked_mask = np.zeros(len(ppls), dtype=bool)
    if ref_ppl_gate_enabled:
        for idx in range(len(ppls)):
            policy_nll = nlls[idx] if idx < len(nlls) else float("inf")
            policy_tokens = token_counts[idx] if idx < len(token_counts) else 0
            ref_nll = ref_nlls[idx] if idx < len(ref_nlls) else float("inf")
            ref_tokens = ref_token_counts[idx] if idx < len(ref_token_counts) else 0
            ref_ppl_gate_blocked_mask[idx] = (
                float(gate_values[idx]) > 0.0
                and np.isfinite(policy_nll)
                and np.isfinite(ref_nll)
                and policy_tokens > 0
                and ref_tokens > 0
                and (policy_nll / policy_tokens) < (ref_nll / ref_tokens) - ref_ppl_gate_margin
            )
    ref_ppl_gate_blocked_count = int(ref_ppl_gate_blocked_mask.sum())

    for uid in np.unique(uids):
        indices = [idx for idx, row_uid in enumerate(uids) if row_uid == uid and finite_ppl_mask[idx]]
        indices.sort(key=lambda idx: ppls[idx])
        low_ppl_indices = indices[:top_k]
        high_ppl_indices = indices[-high_k:]
        low_ppl_candidate_count += len(low_ppl_indices)
        high_ppl_candidate_count += len(high_ppl_indices)
        for idx in low_ppl_indices:
            if rank_reward_mode == "both" and float(gate_values[idx]) > 0.0:
                bonus[idx] += float(ppl_reward_coeff)
        for idx in high_ppl_indices:
            if float(gate_values[idx]) > 0.0:
                bonus[idx] -= float(ppl_reward_coeff)

    if ref_ppl_gate_blocked_count > 0:
        blocked = torch.from_numpy(ref_ppl_gate_blocked_mask).to(device=bonus.device)
        bonus[blocked] = -base_scores[blocked]



    base_reward_scores = base_scores.detach().float()
    ppl_rank_reward_scores = bonus.detach().float()
    composed_reward_scores = base_reward_scores + ppl_rank_reward_scores
    reward_split_metrics = {
        f"{metric_prefix}/base_reward/mean": float(base_reward_scores.mean().cpu().item()),
        f"{metric_prefix}/base_reward/max": float(base_reward_scores.max().cpu().item()),
        f"{metric_prefix}/base_reward/min": float(base_reward_scores.min().cpu().item()),
        f"{metric_prefix}/reward_coeff": float(ppl_reward_coeff),
        f"{metric_prefix}/reward_mode_both": float(rank_reward_mode == "both"),
        f"{metric_prefix}/reward_mode_high_only": float(rank_reward_mode == "high_only"),
        f"{metric_prefix}/ref_ppl_gate_enabled": float(ref_ppl_gate_enabled),
        f"{metric_prefix}/ref_ppl_gate_margin": float(ref_ppl_gate_margin),
        f"{metric_prefix}/ref_ppl_gate_blocked_count": float(ref_ppl_gate_blocked_count),
        f"{metric_prefix}/ref_ppl_gate_zeroed_count": float(ref_ppl_gate_blocked_count),
        f"{metric_prefix}/rank_reward/mean": float(ppl_rank_reward_scores.mean().cpu().item()),
        f"{metric_prefix}/rank_reward/max": float(ppl_rank_reward_scores.max().cpu().item()),
        f"{metric_prefix}/rank_reward/min": float(ppl_rank_reward_scores.min().cpu().item()),
        f"{metric_prefix}/composed_reward/mean": float(composed_reward_scores.mean().cpu().item()),
        f"{metric_prefix}/composed_reward/max": float(composed_reward_scores.max().cpu().item()),
        f"{metric_prefix}/composed_reward/min": float(composed_reward_scores.min().cpu().item()),
    }

    if torch.count_nonzero(bonus).item() == 0:
        metrics = {
            f"{metric_prefix}/finite_count": float(finite_ppl_mask.sum()),
            f"{metric_prefix}/rank_k": float(top_k),
        f"{metric_prefix}/high_rank_k": float(high_k),
            f"{metric_prefix}/low_ppl_candidate_count": float(low_ppl_candidate_count),
            f"{metric_prefix}/high_ppl_candidate_count": float(high_ppl_candidate_count),
            f"{metric_prefix}/positive_bonus_count": 0.0,
            f"{metric_prefix}/negative_bonus_count": 0.0,
            f"{metric_prefix}/corpus_ppl": float(corpus_ppl),
            **reward_split_metrics,
        }
        return reward_tensor, metrics

    updated_reward = reward_tensor.clone()
    response_mask = batch.batch["response_mask"]
    last_response_positions = (response_mask.sum(dim=-1).long() - 1).clamp_min(0)
    row_indices = torch.arange(updated_reward.size(0), device=updated_reward.device)
    updated_reward[row_indices, last_response_positions] += bonus.to(updated_reward.device)

    finite_ppls = [ppl for ppl, is_finite in zip(ppls, finite_ppl_mask, strict=True) if is_finite]
    metrics = {
        f"{metric_prefix}/finite_count": float(len(finite_ppls)),
        f"{metric_prefix}/rank_k": float(top_k),
        f"{metric_prefix}/high_rank_k": float(high_k),
        f"{metric_prefix}/low_ppl_candidate_count": float(low_ppl_candidate_count),
        f"{metric_prefix}/high_ppl_candidate_count": float(high_ppl_candidate_count),
        f"{metric_prefix}/positive_bonus_count": float(torch.count_nonzero(bonus > 0).item()),
        f"{metric_prefix}/negative_bonus_count": float(torch.count_nonzero(bonus < 0).item()),
        f"{metric_prefix}/corpus_ppl": float(corpus_ppl),
        **reward_split_metrics,
    }
    return updated_reward, metrics


def apply_if_ppl_anchor_reward(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    ppl_reward_coeff: float = 1.0,
    metric_prefix: str = "if_p_y_given_x_anchor",
    policy_nll_key: str = "p_y_given_x_nll",
    policy_token_count_key: str = "p_y_given_x_token_count",
    ref0_nll_key: str = "p_y_ref_given_x_nll",
    ref0_token_count_key: str = "p_y_ref_given_x_token_count",
    ref1_nll_key: str = "p_y_ref_xc_given_x_nll",
    ref1_token_count_key: str = "p_y_ref_xc_given_x_token_count",
    gate_scores: torch.Tensor | None = None,
    anchor_reward_mode: Any = "both",
) -> tuple[torch.Tensor, dict[str, float]]:
    # Anchor reward: zero reward below x-only ref PPL, bonus inside [ref0, ref1].
    anchor_reward_mode = normalize_if_ppl_anchor_reward_mode(anchor_reward_mode)
    # Inverted-pair and below-floor behavior are configured independently:
    #   IF_PPL_ANCHOR_FLIP_HANDLING: "abstain" (default; inverted pairs keep IF untouched)
    #     | "strict" (the floor fires before the band-validity
    #       check, so inverted rows with P < A receive the floor action).
    #   IF_PPL_ANCHOR_FLOOR_ACTION: "zero" (default; cancel the whole IF reward)
    #     | "ignore" (below-floor rows keep IF and forfeit the bonus)
    #     | "penalty" (below-floor rows get IF - IF_PPL_ANCHOR_FLOOR_PENALTY, clamped at 0).
    flip_handling = os.getenv("IF_PPL_ANCHOR_FLIP_HANDLING", "abstain").strip().lower()
    if flip_handling not in {"abstain", "strict"}:
        raise ValueError(f"IF_PPL_ANCHOR_FLIP_HANDLING must be abstain|strict, got: {flip_handling}")
    floor_action = os.getenv("IF_PPL_ANCHOR_FLOOR_ACTION", "zero").strip().lower()
    if floor_action not in {"zero", "ignore", "penalty"}:
        raise ValueError(f"IF_PPL_ANCHOR_FLOOR_ACTION must be zero|ignore|penalty, got: {floor_action}")
    floor_penalty = float(os.getenv("IF_PPL_ANCHOR_FLOOR_PENALTY", "0.1"))
    #   IF_PPL_ANCHOR_FLOOR_MARGIN (nats, default 0.0): the floor fires at
    #   P < A - margin, treating the single-draw anchor A as noisy. Rows landing
    #   in [A - margin, A) fall through to the bonus branch by design (the band's
    #   effective lower edge moves down with the floor).
    floor_margin = float(os.getenv("IF_PPL_ANCHOR_FLOOR_MARGIN", "0.0"))
    if floor_margin < 0:
        raise ValueError(f"IF_PPL_ANCHOR_FLOOR_MARGIN must be >= 0, got: {floor_margin}")
    required = [
        policy_nll_key,
        policy_token_count_key,
        ref0_nll_key,
        ref0_token_count_key,
        ref1_nll_key,
        ref1_token_count_key,
    ]
    if any(key not in batch.non_tensor_batch for key in required):
        return reward_tensor, {}

    def to_float_list(key: str) -> list[float]:
        values = []
        for value in batch.non_tensor_batch.get(key, []):
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(float("inf"))
        return values

    def to_int_list(key: str) -> list[int]:
        values = []
        for value in batch.non_tensor_batch.get(key, []):
            try:
                values.append(int(value))
            except (TypeError, ValueError):
                values.append(0)
        return values

    policy_nlls = to_float_list(policy_nll_key)
    ref0_nlls = to_float_list(ref0_nll_key)
    ref1_nlls = to_float_list(ref1_nll_key)
    policy_tokens = to_int_list(policy_token_count_key)
    ref0_tokens = to_int_list(ref0_token_count_key)
    ref1_tokens = to_int_list(ref1_token_count_key)

    base_scores = reward_tensor.sum(dim=-1)
    gate_source = base_scores if gate_scores is None else gate_scores
    gate_values = gate_source.detach().to(device="cpu", dtype=torch.float64).tolist()
    bonus = torch.zeros_like(base_scores, dtype=reward_tensor.dtype)

    eligible_count = 0
    lower_zero_count = 0
    floor_penalty_count = 0
    floor_ignored_count = 0
    interval_bonus_count = 0
    above_ref1_count = 0
    invalid_anchor_count = 0
    finite_policy_count = 0
    finite_ref0_count = 0
    finite_ref1_count = 0

    for idx in range(len(gate_values)):
        policy_nll = policy_nlls[idx] if idx < len(policy_nlls) else float("inf")
        ref0_nll = ref0_nlls[idx] if idx < len(ref0_nlls) else float("inf")
        ref1_nll = ref1_nlls[idx] if idx < len(ref1_nlls) else float("inf")
        policy_count = policy_tokens[idx] if idx < len(policy_tokens) else 0
        ref0_count = ref0_tokens[idx] if idx < len(ref0_tokens) else 0
        ref1_count = ref1_tokens[idx] if idx < len(ref1_tokens) else 0
        has_policy = np.isfinite(policy_nll) and policy_count > 0
        has_ref0 = np.isfinite(ref0_nll) and ref0_count > 0
        has_ref1 = np.isfinite(ref1_nll) and ref1_count > 0
        finite_policy_count += int(has_policy)
        finite_ref0_count += int(has_ref0)
        finite_ref1_count += int(has_ref1)
        if float(gate_values[idx]) <= 0.0 or not (has_policy and has_ref0 and has_ref1):
            continue
        eligible_count += 1
        policy_mean_nll = policy_nll / policy_count
        ref0_mean_nll = ref0_nll / ref0_count
        ref1_mean_nll = ref1_nll / ref1_count
        inverted = ref1_mean_nll < ref0_mean_nll
        if inverted and flip_handling == "abstain":
            # An inverted pair does not define the intended [x-only, x+c] interval.
            # Preserve the original IF reward instead of letting the lower-bound
            # branch cancel it. (flip_handling=strict reproduces the pre-2026-08-09
            # order where the floor fired before this validity check.)
            invalid_anchor_count += 1
            continue
        if anchor_reward_mode != "no_lower_zero" and policy_mean_nll < ref0_mean_nll - floor_margin:
            # Below the floor. The row is consumed here under EVERY floor_action:
            # "ignore" must not fall through to the bonus branch, or it would
            # silently become no_lower_zero (below-floor rows earning the bonus).
            if floor_action == "zero":
                bonus[idx] = -base_scores[idx]
                lower_zero_count += 1
            elif floor_action == "penalty":
                bonus[idx] = -min(floor_penalty, float(base_scores[idx]))
                floor_penalty_count += 1
            else:  # ignore
                floor_ignored_count += 1
        elif (not inverted) and anchor_reward_mode != "lower_zero_only" and policy_mean_nll <= ref1_mean_nll:
            bonus[idx] += float(ppl_reward_coeff)
            interval_bonus_count += 1
        elif inverted:
            # strict mode only: inverted row not consumed by the floor branch.
            invalid_anchor_count += 1
        else:
            above_ref1_count += int(policy_mean_nll > ref1_mean_nll)

    updated_reward = reward_tensor
    if torch.count_nonzero(bonus).item() > 0:
        updated_reward = reward_tensor.clone()
        response_mask = batch.batch["response_mask"]
        last_response_positions = (response_mask.sum(dim=-1).long() - 1).clamp_min(0)
        row_indices = torch.arange(updated_reward.size(0), device=updated_reward.device)
        updated_reward[row_indices, last_response_positions] += bonus.to(updated_reward.device)

    base_reward_scores = base_scores.detach().float()
    anchor_reward_scores = bonus.detach().float()
    composed_reward_scores = base_reward_scores + anchor_reward_scores
    metrics = {
        f"{metric_prefix}/reward_coeff": float(ppl_reward_coeff),
        f"{metric_prefix}/reward_mode_both": float(anchor_reward_mode == "both"),
        f"{metric_prefix}/reward_mode_no_lower_zero": float(anchor_reward_mode == "no_lower_zero"),
        f"{metric_prefix}/reward_mode_lower_zero_only": float(anchor_reward_mode == "lower_zero_only"),
        f"{metric_prefix}/eligible_count": float(eligible_count),
        f"{metric_prefix}/finite_policy_count": float(finite_policy_count),
        f"{metric_prefix}/finite_ref0_count": float(finite_ref0_count),
        f"{metric_prefix}/finite_ref1_count": float(finite_ref1_count),
        f"{metric_prefix}/lower_zero_count": float(lower_zero_count),
        f"{metric_prefix}/floor_penalty_count": float(floor_penalty_count),
        f"{metric_prefix}/floor_ignored_count": float(floor_ignored_count),
        f"{metric_prefix}/flip_handling_strict": float(flip_handling == "strict"),
        f"{metric_prefix}/floor_action_zero": float(floor_action == "zero"),
        f"{metric_prefix}/floor_action_ignore": float(floor_action == "ignore"),
        f"{metric_prefix}/floor_action_penalty": float(floor_action == "penalty"),
        f"{metric_prefix}/floor_penalty_value": float(floor_penalty),
        f"{metric_prefix}/floor_margin": float(floor_margin),
        f"{metric_prefix}/interval_bonus_count": float(interval_bonus_count),
        f"{metric_prefix}/above_ref1_count": float(above_ref1_count),
        f"{metric_prefix}/invalid_anchor_count": float(invalid_anchor_count),
        f"{metric_prefix}/positive_bonus_count": float(torch.count_nonzero(bonus > 0).item()),
        f"{metric_prefix}/negative_bonus_count": float(torch.count_nonzero(bonus < 0).item()),
        f"{metric_prefix}/base_reward/mean": float(base_reward_scores.mean().cpu().item()),
        f"{metric_prefix}/base_reward/max": float(base_reward_scores.max().cpu().item()),
        f"{metric_prefix}/base_reward/min": float(base_reward_scores.min().cpu().item()),
        f"{metric_prefix}/anchor_reward/mean": float(anchor_reward_scores.mean().cpu().item()),
        f"{metric_prefix}/anchor_reward/max": float(anchor_reward_scores.max().cpu().item()),
        f"{metric_prefix}/anchor_reward/min": float(anchor_reward_scores.min().cpu().item()),
        f"{metric_prefix}/composed_reward/mean": float(composed_reward_scores.mean().cpu().item()),
        f"{metric_prefix}/composed_reward/max": float(composed_reward_scores.max().cpu().item()),
        f"{metric_prefix}/composed_reward/min": float(composed_reward_scores.min().cpu().item()),
    }
    return updated_reward, metrics


def compute_spec_decode_metrics(
    spec_drafts,
    spec_accepts,
    spec_verifies,
    non_padding_mask=None,
) -> dict:
    """Aggregate per-request speculative decoding stats.

    Ratios are computed per request and then averaged, so long and short
    responses have equal metric weight.

    The three inputs come from the rollout engine (vLLM request spec-decode
    stats or sglang ``meta_info["spec_*"]`` keys). Either all three are ``None``
    (caller didn't fetch them, e.g. spec rollout disabled) and the function
    is a no-op, or all three are populated; mixed state is a programmer error.

    ``non_padding_mask`` is a numpy bool array used by sync PPO to drop padded
    placeholder samples; pass ``None`` for async PPO.
    """
    if spec_drafts is None and spec_accepts is None and spec_verifies is None:
        return {}
    assert spec_drafts is not None and spec_accepts is not None and spec_verifies is not None, (
        "spec_decode metrics require all three of spec_num_draft_tokens / "
        "spec_num_accepted_tokens / spec_num_verify_steps; got partial inputs"
    )

    drafts = spec_drafts.tolist() if hasattr(spec_drafts, "tolist") else list(spec_drafts)
    accepts = spec_accepts.tolist() if hasattr(spec_accepts, "tolist") else list(spec_accepts)
    verifies = spec_verifies.tolist() if hasattr(spec_verifies, "tolist") else list(spec_verifies)

    if non_padding_mask is not None:
        drafts = [d for d, keep in zip(drafts, non_padding_mask, strict=True) if keep]
        accepts = [a for a, keep in zip(accepts, non_padding_mask, strict=True) if keep]
        verifies = [v for v, keep in zip(verifies, non_padding_mask, strict=True) if keep]

    if len(drafts) == 0:
        return {}

    # Treat zero-denominator samples as 0.0 and keep them in the mean.
    per_sample_accept_rate = [(a / d) if d > 0 else 0.0 for a, d in zip(accepts, drafts, strict=True)]
    per_sample_accept_length = [(1.0 + a / v) if v > 0 else 0.0 for a, v in zip(accepts, verifies, strict=True)]

    n = len(drafts)
    return {
        "rollout/spec_accept_rate": float(sum(per_sample_accept_rate) / n),
        "rollout/spec_accept_length": float(sum(per_sample_accept_length) / n),
    }


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # old_log_probs needed for path-variance proxy: w_t = 1 - 2*exp(old_log_probs) + sum_pi_squared
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


@deprecated(
    "main_ppo.py is deprecated, and wil be replaced by main_ppo_sync.py in v0.8.0, please use main_ppo_sync.py instead."
)
class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.if_ppl_reward_strategy = normalize_if_ppl_reward_strategy(
            self.config.get("if_ppl_reward_strategy", "rank")
        )
        if self.if_ppl_reward_strategy in {"inverse_ppl", "raw_likelihood"}:
            strategy = self.if_ppl_reward_strategy
            if not self.use_reference_policy or not normalize_bool_config(
                self.config.get("if_ref_policy_anchor_ppl", False)
            ):
                raise ValueError(f"{strategy} requires an enabled reference policy and if_ref_policy_anchor_ppl=true")
            if normalize_bool_config(self.config.get("if_ref_anchor_precompute", False)):
                raise ValueError(f"{strategy} requires if_ref_anchor_precompute=false; it does not use anchors")
            if float(self.config.get("if_px_given_y_reward_coeff", 0.0)) != 0.0:
                raise ValueError(f"{strategy} requires if_px_given_y_reward_coeff=0; it scores only y given x")
        self.use_teacher_policy = need_teacher_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None
        self._init_dump_executor()

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = {
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in reward_extra_infos_dict.items()
            }
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "index", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if (self.use_rm or self.compute_reward_after_rollout) and (
                "rm_scores" not in test_output_gen_batch_padded.batch.keys()
            ):
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # convert critic_cfg into TrainingWorkerConfig for the unified model engine worker
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
                extra_context=getattr(self, "_critic_extra_context", {}),
            )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/verl-project/verl/blob/master/examples/tutorial/ray/tutorial.ipynb
        # for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        master_port_range = OmegaConf.select(self.config.trainer, "ray_worker_group_master_port_range")
        if master_port_range is not None:
            master_port_range = OmegaConf.to_container(master_port_range, resolve=True)
            if not isinstance(master_port_range, list) or len(master_port_range) != 2:
                raise ValueError(
                    "trainer.ray_worker_group_master_port_range must be a two-item list, "
                    f"got: {master_port_range}"
                )
            wg_kwargs["master_port_range"] = [int(master_port_range[0]), int(master_port_range[1])]
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )
        self.compute_reward_after_rollout = bool(
            OmegaConf.select(self.config, "reward.compute_after_rollout", default=False)
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # initialize teacher loop manager
        if self.use_teacher_policy:
            from verl.experimental.teacher_loop import MultiTeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = (
            not self.compute_reward_after_rollout
            and (not self.use_rm or self.config.reward.reward_model.enable_resource_pool)
        )

        self.llm_server_manager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        # To stream teacher computation with actor rollout, we instead pass the full manager so that the
        # teacher loop workers can sleep/wake together with rollout workers
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        # Support custom CheckpointEngineManager via config
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            steps_per_epoch = len(self.train_dataloader)
            at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
            if at_epoch_boundary:
                print(
                    f"Skipping dataloader state restore: global_steps={self.global_steps} "
                    f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                    f"The saved state marks the dataloader as exhausted. "
                    f"Next epoch will iterate from scratch."
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        # step 4. No padding to padding
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)

        return ref_log_prob

    def _coerce_token_ids(self, value) -> list[int]:
        if value is None:
            return []
        if isinstance(value, str):
            return self.tokenizer.encode(value, add_special_tokens=False)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, int):
            return [int(value)]

        result = []
        values = list(value)
        if values and all(isinstance(item, str) for item in values):
            return self.tokenizer.encode("".join(values), add_special_tokens=False)
        for item in values:
            if item is None:
                continue
            if isinstance(item, str):
                result.extend(self.tokenizer.encode(item, add_special_tokens=False))
            elif hasattr(item, "tolist"):
                result.extend(self._coerce_token_ids(item.tolist()))
            elif isinstance(item, (list, tuple)):
                result.extend(self._coerce_token_ids(item))
            else:
                result.append(int(item))
        return result

    def _validate_ppl_x_prefix_ids(self, prefixes: list[list[int]]) -> None:
        if str(self.config.get("if_ppl_prefix_mode", "standard")).strip().lower() != "empty_think":
            return
        expected_suffix = list(self.tokenizer.encode("<think>", add_special_tokens=False)) + list(
            self.tokenizer.encode("</think>\n\n", add_special_tokens=False)
        )
        invalid_rows = [
            index
            for index, prefix in enumerate(prefixes)
            if len(prefix) < len(expected_suffix) or prefix[-len(expected_suffix) :] != expected_suffix
        ]
        if invalid_rows:
            raise RuntimeError(
                "empty-think PPL scoring received prefixes that do not end in "
                f"'<think></think>\\n\\n'; invalid rows: {invalid_rows[:8]}"
            )

    def _build_ref_policy_scoring_batch(
        self, prefixes: list[list[int]], continuations: list[list[int]]
    ) -> tuple[DataProto | None, torch.Tensor | None]:
        valid_rows = [idx for idx, (prefix, continuation) in enumerate(zip(prefixes, continuations, strict=False)) if prefix and continuation]
        if not valid_rows:
            return None, None
        selected_prefixes = [prefixes[idx] for idx in valid_rows]
        selected_continuations = [continuations[idx] for idx in valid_rows]
        if self.ref_in_actor:
            ref_dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        else:
            ref_dp_size = self._get_dp_size(self.ref_policy_wg, "ref")
        if ref_dp_size > 1:
            pad_count = (-len(selected_prefixes)) % ref_dp_size
            if pad_count:
                selected_prefixes.extend([selected_prefixes[-1]] * pad_count)
                selected_continuations.extend([selected_continuations[-1]] * pad_count)
        max_prompt = int(self.config.actor_rollout_ref.rollout.prompt_length)
        max_response = int(self.config.actor_rollout_ref.rollout.response_length)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Write each row directly into preallocated tensors instead of building
        # 4*N python lists (via list-concat padding) + torch.tensor() per list +
        # torch.stack() over N tensors, 4 times. Byte-identical output (verified
        # against the previous implementation via torch.equal at N=8192), but
        # ~13x faster there (7.06s -> 0.51s) - this runs single-threaded on the
        # driver, ahead of the GPU forward pass, so every second here was pure
        # added latency on top of it, not overlapped with anything.
        n = len(selected_prefixes)
        prompts = torch.full((n, max_prompt), pad_id, dtype=torch.long)
        responses = torch.full((n, max_response), pad_id, dtype=torch.long)
        prompt_attention = torch.zeros((n, max_prompt), dtype=torch.long)
        response_mask = torch.zeros((n, max_response), dtype=torch.long)
        for i, (prefix, continuation) in enumerate(zip(selected_prefixes, selected_continuations, strict=True)):
            prefix = self._coerce_token_ids(prefix)[-max_prompt:]
            continuation = self._coerce_token_ids(continuation)[:max_response]
            if prefix:
                prompts[i, max_prompt - len(prefix) :] = torch.tensor(prefix, dtype=torch.long)
                prompt_attention[i, max_prompt - len(prefix) :] = 1
            if continuation:
                responses[i, : len(continuation)] = torch.tensor(continuation, dtype=torch.long)
                response_mask[i, : len(continuation)] = 1

        attention_mask = torch.cat([prompt_attention, response_mask], dim=-1)
        input_ids = torch.cat([prompts, responses], dim=-1)
        position_ids = compute_position_id_with_mask(attention_mask)
        scoring_batch = DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask,
                "temperature": torch.ones(prompts.shape[0], dtype=torch.float32),
            }
        )
        return scoring_batch, torch.tensor(valid_rows, dtype=torch.long)

    def _score_continuations_with_ref_policy(
        self, prefixes: list[list[int]], continuations: list[list[int]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = len(prefixes)
        nlls = np.full(size, np.inf, dtype=np.float64)
        token_counts = np.zeros(size, dtype=np.int64)
        ppls = np.full(size, np.inf, dtype=np.float64)
        scoring_batch, valid_rows = self._build_ref_policy_scoring_batch(prefixes, continuations)
        if scoring_batch is None or valid_rows is None:
            return nlls, token_counts, ppls
        ref_log_prob = self._compute_ref_log_prob(scoring_batch)
        log_probs = ref_log_prob.batch["ref_log_prob"].detach().cpu()
        response_mask = scoring_batch.batch["response_mask"].detach().cpu().bool()
        row_nlls = -(log_probs * response_mask.float()).sum(dim=-1).numpy()
        row_counts = response_mask.sum(dim=-1).numpy().astype(np.int64)
        for local_idx, original_idx in enumerate(valid_rows.tolist()):
            count = int(row_counts[local_idx])
            if count <= 0:
                continue
            nll = float(row_nlls[local_idx])
            nlls[original_idx] = nll
            token_counts[original_idx] = count
            ppls[original_idx] = math.exp(nll / count)
        return nlls, token_counts, ppls

    def _compute_anchor_ppl_with_ref_policy(
        self,
        batch: DataProto,
        eligible_mask: np.ndarray | None = None,
        include_anchors: bool = True,
    ) -> dict[str, float]:
        if not normalize_bool_config(self.config.get("if_ref_policy_anchor_ppl", False)):
            return {}
        required = ["ppl_x_prompt_ids", "ppl_y_final_answer_ids"]
        if any(key not in batch.non_tensor_batch for key in required):
            if normalize_bool_config(self.config.get("if_ref_anchor_require_complete_cache", False)):
                raise RuntimeError("CALIF reference scoring fields are missing; refusing to use constraint-only rewards")
            return {"if_ref_policy_anchor_ppl/missing_fields": 1.0}

        prefixes = [list(ids or []) for ids in batch.non_tensor_batch["ppl_x_prompt_ids"]]
        self._validate_ppl_x_prefix_ids(prefixes)
        scoring_mask = None
        if eligible_mask is not None:
            scoring_mask = np.asarray(eligible_mask, dtype=bool)
            if len(scoring_mask) != len(prefixes):
                raise ValueError(
                    "Anchor PPL eligibility mask must match the batch: "
                    f"{len(scoring_mask)} != {len(prefixes)}"
                )
        # Both likelihood baselines score the same extracted final answer as anchors.
        policy_y = [
            list(ids or []) if scoring_mask is None or scoring_mask[idx] else []
            for idx, ids in enumerate(batch.non_tensor_batch["ppl_y_final_answer_ids"])
        ]
        metrics: dict[str, float] = {"if_ref_policy_anchor_ppl/enabled": 1.0}
        if scoring_mask is not None:
            metrics["if_ref_policy_anchor_ppl/eligible_count"] = float(scoring_mask.sum())

        def has_cached_scores(prefix: str) -> bool:
            return (
                f"{prefix}_nll" in batch.non_tensor_batch
                and f"{prefix}_token_count" in batch.non_tensor_batch
                and f"{prefix}_ppl" in batch.non_tensor_batch
            )

        def log_cached_score_metrics(prefix: str) -> None:
            nlls = np.asarray([float(value) for value in batch.non_tensor_batch.get(f"{prefix}_nll", [])], dtype=np.float64)
            token_counts = np.asarray(
                [int(value) for value in batch.non_tensor_batch.get(f"{prefix}_token_count", [])], dtype=np.int64
            )
            finite = np.isfinite(nlls) & (token_counts > 0)
            metrics[f"if_ref_policy_anchor_ppl/{prefix}/finite_count"] = float(finite.sum())
            metrics[f"if_ref_policy_anchor_ppl/{prefix}/from_cache"] = 1.0
            if finite.any():
                metrics[f"if_ref_policy_anchor_ppl/{prefix}/corpus_ppl"] = float(
                    math.exp(float(nlls[finite].sum()) / int(token_counts[finite].sum()))
                )

        metric_specs = [("p_y_given_x", policy_y)]
        if include_anchors:
            if has_cached_scores("p_y_ref_given_x"):
                log_cached_score_metrics("p_y_ref_given_x")
            elif "ppl_y_ref0_ids" in batch.non_tensor_batch:
                ref0_y = [
                    list(ids or []) if scoring_mask is None or scoring_mask[idx] else []
                    for idx, ids in enumerate(batch.non_tensor_batch["ppl_y_ref0_ids"])
                ]
                metric_specs.append(("p_y_ref_given_x", ref0_y))
            else:
                metrics["if_ref_policy_anchor_ppl/p_y_ref_given_x/missing_cache"] = 1.0

            if has_cached_scores("p_y_ref_xc_given_x"):
                log_cached_score_metrics("p_y_ref_xc_given_x")
            elif "ppl_y_ref1_ids" in batch.non_tensor_batch:
                ref1_y = [
                    list(ids or []) if scoring_mask is None or scoring_mask[idx] else []
                    for idx, ids in enumerate(batch.non_tensor_batch["ppl_y_ref1_ids"])
                ]
                metric_specs.append(("p_y_ref_xc_given_x", ref1_y))
            else:
                metrics["if_ref_policy_anchor_ppl/p_y_ref_xc_given_x/missing_cache"] = 1.0

        for prefix, continuations in metric_specs:
            nlls, token_counts, ppls = self._score_continuations_with_ref_policy(prefixes, continuations)
            batch.non_tensor_batch[f"{prefix}_nll"] = nlls.astype(object)
            batch.non_tensor_batch[f"{prefix}_token_count"] = token_counts.astype(object)
            batch.non_tensor_batch[f"{prefix}_ppl"] = ppls.astype(object)
            finite = np.isfinite(ppls)
            metrics[f"if_ref_policy_anchor_ppl/{prefix}/finite_count"] = float(finite.sum())
            metrics[f"if_ref_policy_anchor_ppl/{prefix}/from_cache"] = 0.0
            finite_for_corpus = np.isfinite(nlls) & (token_counts > 0)
            if finite_for_corpus.any():
                metrics[f"if_ref_policy_anchor_ppl/{prefix}/corpus_ppl"] = float(
                    math.exp(float(nlls[finite_for_corpus].sum()) / int(token_counts[finite_for_corpus].sum()))
                )
        return metrics

    def _extract_anchor_answer_ids(self, output: DataProto) -> list[list[int]]:
        if "if_final_answer_ids" in output.non_tensor_batch:
            result = []
            for ids in output.non_tensor_batch["if_final_answer_ids"]:
                if ids is None:
                    result.append([])
                elif isinstance(ids, str):
                    result.append(self.tokenizer.encode(ids, add_special_tokens=False))
                else:
                    result.append([int(token_id) for token_id in list(ids)])
            return result
        responses = output.batch["responses"].detach().cpu()
        response_mask = output.batch["response_mask"].detach().cpu().bool()
        result = []
        for row, mask in zip(responses, response_mask, strict=True):
            result.append(row[mask].tolist())
        return result

    def _extract_anchor_vllm_scores(
        self, output: DataProto, size: int, metric_prefix: str = "p_y_given_x"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        required = [f"{metric_prefix}_nll", f"{metric_prefix}_token_count", f"{metric_prefix}_ppl"]
        missing = [key for key in required if key not in output.non_tensor_batch]
        if missing:
            raise RuntimeError(
                "IF ref anchor precompute expected vLLM prompt-logprob scores from the agent loop, "
                f"but missing fields: {missing}"
            )

        # A per-row scoring failure inside the agent loop (it catches and logs,
        # leaving these fields unset -> None after batching) must not take the
        # whole precompute batch down: coerce to the "unusable" sentinels so the
        # row simply fails _if_ref_anchor_cache_item_complete and is retried on
        # the next run, instead of raising TypeError on float(None) and losing
        # every already-generated row in the batch.
        def _as_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("inf")

        def _as_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        nlls = np.asarray(
            [_as_float(value) for value in output.non_tensor_batch[f"{metric_prefix}_nll"][:size]], dtype=np.float64
        )
        token_counts = np.asarray(
            [_as_int(value) for value in output.non_tensor_batch[f"{metric_prefix}_token_count"][:size]], dtype=np.int64
        )
        ppls = np.asarray(
            [_as_float(value) for value in output.non_tensor_batch[f"{metric_prefix}_ppl"][:size]], dtype=np.float64
        )
        return nlls, token_counts, ppls

    def _make_anchor_generation_batch(self, rows: list[dict], prompt_key: str) -> DataProto:
        dummy = torch.stack([row["dummy_tensor"] for row in rows], dim=0)
        raw_prompts = []
        ppl_prompts = []
        indices = []
        for row in rows:
            raw_prompts.append(row[prompt_key])
            ppl_prompts.append(row.get("ppl_prompt"))
            indices.append(row.get("index", 0))
        return DataProto.from_single_dict(
            {
                "dummy_tensor": dummy,
                "raw_prompt": np.array(raw_prompts, dtype=object),
                "ppl_prompt": np.array(ppl_prompts, dtype=object),
                "index": np.array(indices, dtype=object),
                "__if_anchor_precompute__": np.array([True] * len(rows), dtype=object),
                "__skip_compute_score__": np.array([True] * len(rows), dtype=object),
            }
        )

    @property
    def think_end_token_id(self) -> int | None:
        """Cached id of the reasoning-end token, or ``None`` when this tokenizer has no single
        such token (non-reasoning model families), which disables the reasoning/answer split
        metrics instead of reporting a bogus split."""
        if not hasattr(self, "_think_end_token_id"):
            self._think_end_token_id = resolve_think_end_token_id(self.tokenizer)
            if self._think_end_token_id is None:
                print(
                    "[think metrics] no single reasoning-end token in this tokenizer "
                    f"(marker={os.getenv('IF_THINK_END_TOKEN', '</think>')!r}); "
                    "think/* reasoning-vs-answer length metrics are disabled"
                )
            else:
                print(f"[think metrics] reasoning-end token id = {self._think_end_token_id}")
        return self._think_end_token_id

    def _if_ref_anchor_cache_metadata(self) -> dict[str, Any]:
        data_cfg = self.config.data
        ppl_prefix_mode = str(self.config.get("if_ppl_prefix_mode", "standard")).strip().lower()
        return {
            "version": 3,
            "model_path": str(self.config.actor_rollout_ref.model.path),
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "ppl_prefix_mode": ppl_prefix_mode,
            "ppl_nll_scope": "final_answer_tokens_only",
            "train_sample_count": int(len(self.train_dataset)),
            "train_files": list(data_cfg.get("train_files", [])),
            "if_dataset_hf": str(data_cfg.get("if_dataset_hf", "")),
            "if_dataset_revision": str(data_cfg.get("if_dataset_revision", "")),
            "if_dataset_seed": int(data_cfg.get("if_dataset_seed", 1)),
            "if_dataset_val_size": int(data_cfg.get("if_dataset_val_size", 512)),
            "max_prompt_length": int(data_cfg.get("max_prompt_length", 0)),
            "max_response_length": int(data_cfg.get("max_response_length", 0)),
            "apply_chat_template_kwargs": OmegaConf.to_container(
                data_cfg.get("apply_chat_template_kwargs", {}), resolve=True
            ),
            "rollout_temperature": float(self.config.actor_rollout_ref.rollout.get("temperature", 1.0)),
            "rollout_top_p": float(self.config.actor_rollout_ref.rollout.get("top_p", 1.0)),
            "rollout_top_k": int(self.config.actor_rollout_ref.rollout.get("top_k", -1)),
            "response_length": int(self.config.actor_rollout_ref.rollout.response_length),
        }

    def _if_ref_anchor_cache_path(self) -> str | None:
        path = self.config.get("if_ref_anchor_cache_path", None)
        if path is None:
            return None
        path = str(path).strip()
        return path or None

    def _if_ref_anchor_train_indices(self) -> set[int]:
        """Return stable source IDs, without tokenizing every training example.

        Prompt-length filtering and cached-only filtering can leave a non-contiguous
        set of IDs. A cache with the right *size* is therefore not enough.
        """
        dataframe = getattr(self.train_dataset, "dataframe", None)
        if dataframe is None:
            raise RuntimeError("Complete anchor-cache validation requires an indexed training dataframe")
        try:
            indices = [int(info["index"]) for info in dataframe["extra_info"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Every training row needs a stable extra_info.index for anchor-cache validation") from exc
        if len(indices) != len(self.train_dataset) or len(set(indices)) != len(indices):
            raise RuntimeError("Training anchor-cache indices are missing or duplicated")
        return set(indices)

    def _load_if_ref_anchor_cache(self) -> bool:
        path = self._if_ref_anchor_cache_path()
        if not path or not os.path.exists(path):
            return False
        expected_metadata = self._if_ref_anchor_cache_metadata()
        release_cache_setting = self.config.get("if_release_cache_setting", None)
        release_cache_verified = bool(release_cache_setting)
        if release_cache_verified:
            # The release validator checks the pinned Hub SHA, source metadata,
            # reference model/tokenizer identity, and active training-row coverage.
            # A failure is fatal: never replace a bad cache by freshly sampled anchors.
            from calif_training.cache import validate_for_training

            payload = validate_for_training(
                path, str(release_cache_setting), expected_metadata, self._if_ref_anchor_train_indices()
            )
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:  # noqa: BLE001
                print(f"[IF ref anchor] failed to load cache {path}: {exc}; regenerating")
                return False

        actual_metadata = payload.get("metadata", {})
        semantic_keys = ("ppl_prefix_mode", "ppl_nll_scope")
        semantic_mismatches = {
            key: {"expected": expected_metadata.get(key), "actual": actual_metadata.get(key)}
            for key in semantic_keys
            if actual_metadata.get(key) != expected_metadata.get(key)
        }
        allow_legacy_final_answer_cache = normalize_bool_config(
            self.config.get("if_ref_anchor_allow_legacy_final_answer_cache", False), False
        )
        try:
            actual_cache_version = int(actual_metadata.get("version", -1))
        except (TypeError, ValueError):
            actual_cache_version = -1
        legacy_final_answer_cache_compatible = (
            allow_legacy_final_answer_cache
            and actual_cache_version == 2
            and expected_metadata.get("ppl_prefix_mode") == "standard"
            and all(key not in actual_metadata for key in semantic_keys)
            and (
                actual_metadata.get("model_path") == expected_metadata.get("model_path")
                or actual_metadata.get("model_path") == self.config.get("if_ref_anchor_verified_legacy_model_id", None)
            )
        )
        if semantic_mismatches and not legacy_final_answer_cache_compatible and not release_cache_verified:
            print(
                f"[IF ref anchor] cache PPL semantics mismatch for {path}: {semantic_mismatches}; "
                "regenerating"
            )
            return False
        if semantic_mismatches and not release_cache_verified:
            print(
                f"[IF ref anchor] accepting legacy v2 final-answer cache {path} as "
                "ppl_prefix_mode=standard, ppl_nll_scope=final_answer_tokens_only because "
                "if_ref_anchor_allow_legacy_final_answer_cache=true"
            )
        if actual_metadata != expected_metadata and not release_cache_verified:
            metadata_strict = normalize_bool_config(self.config.get("if_ref_anchor_cache_metadata_strict", True), True)
            if not metadata_strict:
                print(
                    f"[IF ref anchor] cache metadata mismatch for {path}; "
                    "reusing because if_ref_anchor_cache_metadata_strict=false"
                )
            else:
                print(f"[IF ref anchor] cache metadata mismatch for {path}; regenerating")
                print(f"[IF ref anchor] expected metadata: {expected_metadata}")
                print(f"[IF ref anchor] actual metadata: {actual_metadata}")
                return False

        cache = {}
        for key, item in payload.get("items", {}).items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            cache[index] = {
                "y0": list(item.get("y0") or []),
                "y1": list(item.get("y1") or []),
                "ref0_nll": float(item.get("ref0_nll", float("inf"))),
                "ref0_token_count": int(item.get("ref0_token_count", 0)),
                "ref0_ppl": float(item.get("ref0_ppl", float("inf"))),
                "ref1_nll": float(item.get("ref1_nll", float("inf"))),
                "ref1_token_count": int(item.get("ref1_token_count", 0)),
                "ref1_ppl": float(item.get("ref1_ppl", float("inf"))),
            }
        self.if_ref_anchor_cache = cache
        print(f"[IF ref anchor] loaded {len(cache)} cached train samples from {path}")
        return bool(cache)

    def _save_if_ref_anchor_cache(self) -> None:
        path = self._if_ref_anchor_cache_path()
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "metadata": self._if_ref_anchor_cache_metadata(),
            "items": {str(index): item for index, item in sorted(self.if_ref_anchor_cache.items())},
        }
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_if_ref_anchor_", suffix=".json", dir=os.path.dirname(path) or ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp_path, path)
            print(f"[IF ref anchor] saved cache to {path}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _tokenize_anchor_ppl_prompt(self, row: dict) -> list[int]:
        apply_kwargs = dict(**self.config.data.get("apply_chat_template_kwargs", {}))
        apply_kwargs.pop("tokenize", None)
        apply_kwargs.pop("return_dict", None)
        apply_kwargs.pop("return_tensors", None)
        tokenized = self.tokenizer.apply_chat_template(
            row.get("ppl_prompt") or [],
            add_generation_prompt=True,
            tokenize=True,
            **apply_kwargs,
        )
        if isinstance(tokenized, dict):
            tokenized = tokenized.get("input_ids", [])
        if hasattr(tokenized, "tolist"):
            tokenized = tokenized.tolist()
        if tokenized and isinstance(tokenized[0], list):
            tokenized = tokenized[0]
        prefix_ids = list(tokenized)
        if str(self.config.get("if_ppl_prefix_mode", "standard")).strip().lower() == "empty_think":
            open_think_ids = list(self.tokenizer.encode("<think>", add_special_tokens=False))
            close_think_ids = list(self.tokenizer.encode("</think>\n\n", add_special_tokens=False))
            closed_suffix = open_think_ids + close_think_ids
            if prefix_ids[-len(closed_suffix) :] != closed_suffix:
                if not open_think_ids or prefix_ids[-len(open_think_ids) :] != open_think_ids:
                    raise RuntimeError(
                        "if_ppl_prefix_mode=empty_think requires a chat template ending in '<think>'"
                    )
                prefix_ids.extend(close_think_ids)
        self._validate_ppl_x_prefix_ids([prefix_ids])
        return prefix_ids

    def _if_ref_anchor_cache_item_complete(self, item: dict | None) -> bool:
        if not item:
            return False
        return (
            bool(item.get("y0"))
            and bool(item.get("y1"))
            and int(item.get("ref0_token_count", 0) or 0) > 0
            and int(item.get("ref1_token_count", 0) or 0) > 0
            and math.isfinite(float(item.get("ref0_nll", float("inf"))))
            and math.isfinite(float(item.get("ref1_nll", float("inf"))))
        )

    def _precompute_if_ref_anchor_cache(self) -> None:
        if not normalize_bool_config(self.config.get("if_ref_anchor_precompute", False)):
            self.if_ref_anchor_cache = {}
            return

        total = len(self.train_dataset)
        if getattr(self, "if_ref_anchor_cache", None) is None:
            self.if_ref_anchor_cache = {}
        if not self.if_ref_anchor_cache:
            self._load_if_ref_anchor_cache()

        self.if_ref_anchor_cache = {
            int(index): item
            for index, item in self.if_ref_anchor_cache.items()
            if self._if_ref_anchor_cache_item_complete(item)
        }
        initial_cached_count = len(self.if_ref_anchor_cache)
        require_complete = normalize_bool_config(self.config.get("if_ref_anchor_require_complete_cache", False))
        if require_complete:
            required_indices = self._if_ref_anchor_train_indices()
            missing_indices = sorted(required_indices.difference(self.if_ref_anchor_cache))
            if missing_indices:
                raise RuntimeError(
                    "Required complete anchor cache; "
                    f"{len(missing_indices)}/{total} active training IDs are missing or invalid "
                    f"(first IDs: {missing_indices[:8]}); refusing to train"
                )
            print(f"[IF ref anchor] verified complete cache coverage: {total}/{total} active train samples")
            return
        if initial_cached_count >= total:
            print(f"[IF ref anchor] using complete cached train samples: {initial_cached_count}/{total}")
            return

        if normalize_bool_config(self.config.get("if_ref_anchor_skip_missing_precompute", False)):
            print(
                f"[IF ref anchor] using cached train samples only: {initial_cached_count}/{total}; "
                f"skipping {max(total - initial_cached_count, 0)} missing anchor precompute samples"
            )
            return

        batch_size = int(self.config.get("if_ref_anchor_precompute_batch_size", self.config.data.get("gen_batch_size", self.config.data.train_batch_size)))
        save_interval = int(self.config.get("if_ref_anchor_cache_save_interval", 50000))
        agent_workers = int(self.config.actor_rollout_ref.rollout.agent.get("num_workers", 1))
        # Actor weights are never updated inside precompute, so the per-batch
        # sleep + full weight resync below is a no-op that still pays for a KV
        # cache teardown and a whole-model reload on every replica. Skip it for
        # standalone precompute jobs (`if_ref_anchor_precompute_only=true`), and
        # keep the legacy cycle when precompute runs as a prologue to training.
        resync_between_batches = not normalize_bool_config(
            self.config.get("if_ref_anchor_precompute_only", False)
        )
        next_save_count = ((initial_cached_count // save_interval) + 1) * save_interval if save_interval > 0 else total + 1
        last_saved_count = initial_cached_count
        print(
            f"[IF ref anchor] precomputing y0/y1 and anchor PPL for {total} train samples "
            f"with initial rollout/ref policy; resuming from {initial_cached_count} cached samples"
        )

        try:
            with tqdm(total=total * 4, initial=initial_cached_count * 4, desc="IF ref anchor precompute", unit="sample") as pbar:
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    rows = [self.train_dataset[idx] for idx in range(start, end)]
                    rows = [
                        row
                        for row in rows
                        if not self._if_ref_anchor_cache_item_complete(
                            self.if_ref_anchor_cache.get(int(row.get("index", 0)))
                        )
                    ]
                    if not rows:
                        continue

                    generation_rows = rows
                    if rows and agent_workers > 1:
                        pad_count = (-len(rows)) % agent_workers
                        if pad_count:
                            generation_rows = rows + [rows[-1]] * pad_count
                    y0_batch = self._make_anchor_generation_batch(generation_rows, "ppl_prompt")
                    y1_batch = self._make_anchor_generation_batch(generation_rows, "prompt")
                    y0_batch.meta_info["global_steps"] = self.global_steps
                    y1_batch.meta_info["global_steps"] = self.global_steps
                    batch_label = f"{start}-{end}" if len(rows) == end - start else f"{start}-{end} missing={len(rows)}"
                    pbar.set_postfix_str(f"y0 {batch_label}")
                    y0_output = self.async_rollout_manager.generate_sequences(y0_batch)
                    pbar.update(len(rows))
                    pbar.set_postfix_str(f"y1 {batch_label}")
                    y1_output = self.async_rollout_manager.generate_sequences(y1_batch)
                    pbar.update(len(rows))
                    y0_ids = self._extract_anchor_answer_ids(y0_output)[: len(rows)]
                    y1_ids = self._extract_anchor_answer_ids(y1_output)[: len(rows)]
                    pbar.set_postfix_str(f"score y0 {batch_label}")
                    y0_nlls, y0_token_counts, y0_ppls = self._extract_anchor_vllm_scores(y0_output, len(rows))
                    pbar.update(len(rows))
                    pbar.set_postfix_str(f"score y1 {batch_label}")
                    y1_nlls, y1_token_counts, y1_ppls = self._extract_anchor_vllm_scores(y1_output, len(rows))
                    pbar.update(len(rows))
                    for local_idx, (row, ref0_ids, ref1_ids) in enumerate(zip(rows, y0_ids, y1_ids, strict=True)):
                        self.if_ref_anchor_cache[int(row.get("index", 0))] = {
                            "y0": list(ref0_ids),
                            "y1": list(ref1_ids),
                            "ref0_nll": float(y0_nlls[local_idx]),
                            "ref0_token_count": int(y0_token_counts[local_idx]),
                            "ref0_ppl": float(y0_ppls[local_idx]),
                            "ref1_nll": float(y1_nlls[local_idx]),
                            "ref1_token_count": int(y1_token_counts[local_idx]),
                            "ref1_ppl": float(y1_ppls[local_idx]),
                        }

                    completed_count = len(self.if_ref_anchor_cache)
                    if save_interval > 0 and completed_count >= next_save_count and completed_count < total:
                        self._save_if_ref_anchor_cache()
                        last_saved_count = completed_count
                        while next_save_count <= completed_count:
                            next_save_count += save_interval
                    if end < total and resync_between_batches:
                        self.checkpoint_manager.sleep_replicas()
                        self.checkpoint_manager.update_weights(self.global_steps)
        finally:
            if len(getattr(self, "if_ref_anchor_cache", {})) > last_saved_count:
                self._save_if_ref_anchor_cache()
        print(f"[IF ref anchor] cached {len(self.if_ref_anchor_cache)} train samples")

    def _attach_if_ref_anchor_cache(self, batch: DataProto) -> dict[str, float]:
        if not normalize_bool_config(self.config.get("if_ref_anchor_precompute", False)):
            return {}
        cache = getattr(self, "if_ref_anchor_cache", {})
        require_complete = normalize_bool_config(self.config.get("if_ref_anchor_require_complete_cache", False))
        if not cache or "index" not in batch.non_tensor_batch:
            if require_complete:
                raise RuntimeError("Required anchor cache or rollout indices are missing; refusing to train")
            return {"if_ref_anchor_cache/missing": 1.0}
        if require_complete:
            missing_indices = sorted(
                {
                    int(value)
                    for value in batch.non_tensor_batch["index"]
                    if not self._if_ref_anchor_cache_item_complete(cache.get(int(value)))
                }
            )
            if missing_indices:
                raise RuntimeError(
                    "Rollout batch has missing or invalid required anchors "
                    f"(first IDs: {missing_indices[:8]}); refusing to train"
                )
        ref0 = []
        ref1 = []
        ref0_nlls = []
        ref0_token_counts = []
        ref0_ppls = []
        ref1_nlls = []
        ref1_token_counts = []
        ref1_ppls = []
        hit = 0
        for value in batch.non_tensor_batch["index"]:
            item = cache.get(int(value), {})
            y0 = list(item.get("y0") or [])
            y1 = list(item.get("y1") or [])
            ref0_count = int(item.get("ref0_token_count", 0) or 0)
            ref1_count = int(item.get("ref1_token_count", 0) or 0)
            hit += int(bool(y0) and bool(y1) and ref0_count > 0 and ref1_count > 0)
            ref0.append(y0)
            ref1.append(y1)
            ref0_nlls.append(float(item.get("ref0_nll", float("inf"))))
            ref0_token_counts.append(ref0_count)
            ref0_ppls.append(float(item.get("ref0_ppl", float("inf"))))
            ref1_nlls.append(float(item.get("ref1_nll", float("inf"))))
            ref1_token_counts.append(ref1_count)
            ref1_ppls.append(float(item.get("ref1_ppl", float("inf"))))
        batch.non_tensor_batch["ppl_y_ref0_ids"] = np.array(ref0, dtype=object)
        batch.non_tensor_batch["ppl_y_ref1_ids"] = np.array(ref1, dtype=object)
        batch.non_tensor_batch["p_y_ref_given_x_nll"] = np.array(ref0_nlls, dtype=object)
        batch.non_tensor_batch["p_y_ref_given_x_token_count"] = np.array(ref0_token_counts, dtype=object)
        batch.non_tensor_batch["p_y_ref_given_x_ppl"] = np.array(ref0_ppls, dtype=object)
        batch.non_tensor_batch["p_y_ref_xc_given_x_nll"] = np.array(ref1_nlls, dtype=object)
        batch.non_tensor_batch["p_y_ref_xc_given_x_token_count"] = np.array(ref1_token_counts, dtype=object)
        batch.non_tensor_batch["p_y_ref_xc_given_x_ppl"] = np.array(ref1_ppls, dtype=object)
        return {
            "if_ref_anchor_cache/hit_count": float(hit),
            "if_ref_anchor_cache/size": float(len(cache)),
            "if_ref_anchor_cache/using_cached_ppl": 1.0,
        }

    def _compute_old_log_prob(self, batch: DataProto):
        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        calculate_sum_pi_squared = self.config.actor_rollout_ref.actor.get("calculate_sum_pi_squared", False)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            compute_loss=False,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        # gather output
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        sum_pi_squared = tu.get(output, "sum_pi_squared") if calculate_sum_pi_squared else None

        old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
        # step 4. No padding to padding
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        if sum_pi_squared is not None:
            sum_pi_squared = no_padding_2_padding(sum_pi_squared, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        result = {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
        if routed_experts is not None:
            result["routed_experts"] = routed_experts
        if sum_pi_squared is not None:
            result["sum_pi_squared"] = sum_pi_squared.float()
        old_log_prob = tu.get_tensordict(result)
        old_log_prob = DataProto.from_tensordict(old_log_prob)
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        # modify key name
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        if normalize_bool_config(self.config.get("if_ref_anchor_precompute", False)):
            self.checkpoint_manager.update_weights(self.global_steps)
            self._precompute_if_ref_anchor_cache()
            if normalize_bool_config(self.config.get("if_ref_anchor_precompute_only", False)):
                self._shutdown_dump_executor()
                return
            # Precompute leaves rollout replicas asleep; wake them for the first training rollout.
            self.checkpoint_manager.update_weights(self.global_steps)

        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_n = self.config.actor_rollout_ref.rollout.n
                gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # NOTE: REMAX needs one sampled rollout plus one greedy baseline per prompt.
                    # Keep them in a single agent-loop/vLLM request to avoid sending a second
                    # rollout after replicas have been put to sleep, which can leave async vLLM
                    # engines in an invalid state for multi-turn agent workloads.
                    gen_batch_output.non_tensor_batch["__do_sample__"] = np.ones(len(gen_batch_output), dtype=bool)
                    gen_baseline_batch = gen_batch.slice(0, None)
                    gen_baseline_batch.non_tensor_batch["__do_sample__"] = np.zeros(len(gen_baseline_batch), dtype=bool)
                    combined_gen_batch = DataProto.concat([gen_batch_output, gen_baseline_batch])
                    num_sampled_prompts = len(gen_batch_output)
                else:
                    combined_gen_batch = gen_batch_output
                    num_sampled_prompts = len(gen_batch_output)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.llm_server_manager.start_profile()
                        combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.llm_server_manager.stop_profile()

                        timing_raw.update(combined_gen_output.meta_info["timing"])
                        combined_gen_output.meta_info.pop("timing", None)

                    gen_batch_output = combined_gen_output.slice(0, num_sampled_prompts)
                    if "__do_sample__" in gen_batch_output.non_tensor_batch:
                        gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"])

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        gen_baseline_output = combined_gen_output.slice(num_sampled_prompts, None)
                        if "__do_sample__" in gen_baseline_output.non_tensor_batch:
                            gen_baseline_output.pop(non_tensor_batch_keys=["__do_sample__"])

                        if self.use_rm and "rm_scores" not in gen_baseline_output.batch.keys():
                            baseline_reward = self._compute_reward_colocate(gen_baseline_output)
                            gen_baseline_output = gen_baseline_output.union(baseline_reward)

                        reward_baseline_tensor = gen_baseline_output.batch["rm_scores"].sum(dim=-1)
                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_output
                    del combined_gen_batch, combined_gen_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    metrics.update(self._attach_if_ref_anchor_cache(batch))

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Optional handling for max_response_length-clipped rollouts.
                    clipped_rollout_mode = self.config.get("clipped_rollout_mode", "use")
                    batch, clipped_rollout_metrics = apply_clipped_rollout_batch_mode(batch, clipped_rollout_mode)
                    metrics.update(clipped_rollout_metrics)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if (self.use_rm or self.compute_reward_after_rollout) and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)
                        if "acc" in reward_extra_infos_dict:
                            if_constraint = np.asarray(reward_extra_infos_dict["acc"], dtype=np.float32)
                            metrics.update(
                                {
                                    "if_constraint/acc/mean": float(np.mean(if_constraint)),
                                    "if_constraint/acc/max": float(np.max(if_constraint)),
                                    "if_constraint/acc/min": float(np.min(if_constraint)),
                                }
                            )
                        # Optional zero-reward handling for max_response_length-clipped rollouts.
                        reward_tensor, clipped_reward_metrics = apply_clipped_rollout_reward_mode(
                            batch, reward_tensor, clipped_rollout_mode
                        )
                        metrics.update(clipped_reward_metrics)
                        # Snapshot the constraint-only reward BEFORE any PPL shaping so both
                        # directions gate on the original IF reward, not a PPL-polluted running total.
                        constraint_reward_scores = reward_tensor.sum(dim=-1).detach().clone()
                        # Rank rewards are applied after old IF reward exists and before advantage computation.
                        py_given_x_reward_coeff = float(
                            self.config.get("if_py_given_x_reward_coeff", self.config.get("if_ppl_reward_coeff", 1.0))
                        )
                        px_given_y_reward_coeff = float(self.config.get("if_px_given_y_reward_coeff", 0.0))
                        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
                        ppl_rank_reward_mode = self.config.get("if_ppl_rank_reward_mode", "both")
                        ppl_anchor_reward_mode = self.config.get("if_ppl_anchor_reward_mode", "both")
                        ref_ppl_gate_enabled = normalize_bool_config(self.config.get("if_ref_ppl_gate", False))
                        ref_ppl_gate_margin = float(self.config.get("if_ref_ppl_gate_margin", 0.0))
                        ppl_reward_strategy = self.if_ppl_reward_strategy
                        metrics.update(
                            self._compute_anchor_ppl_with_ref_policy(
                                batch,
                                include_anchors=ppl_reward_strategy not in {"inverse_ppl", "raw_likelihood"},
                            )
                        )

                        if ppl_reward_strategy == "anchor":
                            reward_tensor, py_given_x_metrics = apply_if_ppl_anchor_reward(
                                batch,
                                reward_tensor,
                                ppl_reward_coeff=py_given_x_reward_coeff,
                                metric_prefix="if_p_y_given_x_anchor",
                                gate_scores=constraint_reward_scores,
                                anchor_reward_mode=ppl_anchor_reward_mode,
                            )
                        elif ppl_reward_strategy == "inverse_ppl":
                            reward_tensor, py_given_x_metrics = apply_if_inverse_ppl_reward(
                                batch,
                                reward_tensor,
                                ppl_reward_coeff=py_given_x_reward_coeff,
                                clipped_rollout_mode=clipped_rollout_mode,
                            )
                        elif ppl_reward_strategy == "raw_likelihood":
                            reward_tensor, py_given_x_metrics = apply_if_raw_likelihood_reward(
                                batch,
                                reward_tensor,
                                ppl_reward_coeff=py_given_x_reward_coeff,
                                clipped_rollout_mode=clipped_rollout_mode,
                            )
                        else:
                            reward_tensor, py_given_x_metrics = apply_if_ppl_rank_reward(
                                batch,
                                reward_tensor,
                                ppl_reward_coeff=py_given_x_reward_coeff,
                                rollout_n=rollout_n,
                                metric_prefix="if_p_y_given_x",
                                ppl_key="p_y_given_x_ppl",
                                nll_key="p_y_given_x_nll",
                                token_count_key="p_y_given_x_token_count",
                                gate_scores=constraint_reward_scores,
                                rank_reward_mode=ppl_rank_reward_mode,
                                ref_nll_key="p_y_ref_given_x_nll",
                                ref_token_count_key="p_y_ref_given_x_token_count",
                                ref_ppl_gate_enabled=ref_ppl_gate_enabled,
                                ref_ppl_gate_margin=ref_ppl_gate_margin,
                            )
                        metrics.update(py_given_x_metrics)
                        if px_given_y_reward_coeff != 0.0:
                            reward_tensor, px_given_y_metrics = apply_if_ppl_rank_reward(
                                batch,
                                reward_tensor,
                                ppl_reward_coeff=px_given_y_reward_coeff,
                                rollout_n=rollout_n,
                                metric_prefix="if_p_x_given_y",
                                ppl_key="p_x_given_y_ppl",
                                nll_key="p_x_given_y_nll",
                                token_count_key="p_x_given_y_token_count",
                                gate_scores=constraint_reward_scores,
                                rank_reward_mode=ppl_rank_reward_mode,
                            )
                            metrics.update(px_given_y_metrics)
                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup > self.global_steps:
                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        self.checkpoint_manager.update_weights(self.global_steps)
                    else:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # Reasoning vs. answer token split (tokens before/after the first `</think>`).
                metrics.update(compute_think_split_metrics(batch=batch, think_end_token_id=self.think_end_token_id))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # Per-request spec decode metrics.
                metrics.update(
                    compute_spec_decode_metrics(
                        batch.non_tensor_batch.get("spec_num_draft_tokens", None),
                        batch.non_tensor_batch.get("spec_num_accepted_tokens", None),
                        batch.non_tensor_batch.get("spec_num_verify_steps", None),
                    )
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    self._shutdown_dump_executor()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()
