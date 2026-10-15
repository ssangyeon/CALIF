# Copyright 2024 Bytedance Ltd.; 2024 AllenAI (open-instruct); 2025 verl IF-RLVR migration.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Optional instruction-verification reward manager with truncation handling.

The standard training launchers use ``if_reward_fn.compute_score`` instead.
This manager integrates with the bundled verl reward-loop API and can replace
the verification reward with a penalty when a response has no stop token.

Configure ``reward.reward_manager.source=importlib``,
``reward.reward_manager.name=IFRewardManager``, and
``reward.reward_manager.module.path`` to select it. Reward options are read
from ``reward.reward_kwargs``: verification_reward (10.0),
apply_non_stop_penalty (True), and non_stop_penalty_value (0.0).
"""

from __future__ import annotations

import logging
import os
import sys

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

# Self-contained verifier (no verl dependency); located relative to this file.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from ifeval_oi.verifier import score_ifeval  # noqa: E402

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Defaults for this optional reward manager.
_DEFAULT_VERIFICATION_REWARD = 10.0
_DEFAULT_APPLY_NON_STOP_PENALTY = True
_DEFAULT_NON_STOP_PENALTY_VALUE = 0.0

_CANDIDATE_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>")

_DEPS_OK = False


def _verify_runtime_deps() -> None:
    """Fail fast at construction if a worker node lacks the verifier's runtime data."""
    global _DEPS_OK
    if _DEPS_OK:
        return
    try:
        import langdetect  # noqa: F401

        from ifeval_oi import instructions_util

        instructions_util.nltk.word_tokenize("This is a sentence. Here is another one.")
        instructions_util.count_words("two words")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "IFRewardManager runtime dependency check failed on this node: "
            f"{type(e).__name__}: {e}. Install `langdetect immutabledict nltk` and the nltk "
            "'punkt'/'punkt_tab' data on EVERY ray worker node (or set NLTK_DATA to shared storage). "
            "See requirements-calif.txt."
        ) from e
    _DEPS_OK = True


class IFRewardManager(RewardManagerBase):
    """Instruction-following (ifeval) reward manager with non-stop penalty (optional, coupled)."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        _verify_runtime_deps()

        reward_kwargs = {}
        try:
            reward_kwargs = config.reward.get("reward_kwargs", None) or {}
        except Exception:  # noqa: BLE001
            reward_kwargs = {}

        def _get(key, default):
            try:
                if key in reward_kwargs and reward_kwargs[key] is not None:
                    return reward_kwargs[key]
            except Exception:  # noqa: BLE001
                pass
            return default

        self.verification_reward = float(_get("verification_reward", _DEFAULT_VERIFICATION_REWARD))
        self.require_think_end = os.getenv("IF_REQUIRE_THINK_END_FOR_REWARD", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.apply_non_stop_penalty = bool(_get("apply_non_stop_penalty", _DEFAULT_APPLY_NON_STOP_PENALTY))
        self.non_stop_penalty_value = float(_get("non_stop_penalty_value", _DEFAULT_NON_STOP_PENALTY_VALUE))
        try:
            self.reward_fn_key = config.data.get("reward_fn_key", "data_source")
        except Exception:  # noqa: BLE001
            self.reward_fn_key = "data_source"

        self.stop_token_ids = self._resolve_stop_token_ids(tokenizer)
        logger.warning(
            "IFRewardManager: verification_reward=%s apply_non_stop_penalty=%s non_stop_penalty_value=%s "
            "stop_token_ids=%s",
            self.verification_reward,
            self.apply_non_stop_penalty,
            self.non_stop_penalty_value,
            sorted(self.stop_token_ids),
        )

    @staticmethod
    def _resolve_stop_token_ids(tokenizer) -> set[int]:
        stop_ids: set[int] = set()
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_id, int) and eos_id >= 0:
            stop_ids.add(eos_id)
        for tok in _CANDIDATE_STOP_TOKENS:
            try:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0 and tokenizer.convert_ids_to_tokens(tid) == tok:
                    stop_ids.add(tid)
            except Exception:  # noqa: BLE001
                pass
        return stop_ids

    def _is_non_stop(self, valid_response_ids, valid_response_length, response_length) -> bool:
        used_full_budget = int(valid_response_length) >= int(response_length)
        if not used_full_budget:
            return False
        if int(valid_response_length) == 0:
            return True
        last_token = int(valid_response_ids[-1].item())
        return last_token not in self.stop_token_ids

    async def run_single(self, data: DataProto) -> dict:
        data = data[-1:]
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "ifeval")
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        verifier_score = await self.loop.run_in_executor(
            None, lambda: score_ifeval(response_str, ground_truth, require_think_end=self.require_think_end)
        )

        is_non_stop = self._is_non_stop(valid_response_ids, valid_response_length, response_length)
        if self.apply_non_stop_penalty and is_non_stop:
            reward = self.non_stop_penalty_value
        else:
            reward = self.verification_reward * verifier_score

        return {
            "reward_score": float(reward),
            "reward_extra_info": {
                "acc": float(verifier_score),
                "verifier_score": float(verifier_score),
                "scaled_reward": float(reward),
                "is_non_stop": float(is_non_stop),
                "valid_response_length": int(valid_response_length),
                "data_source": str(data_source),
            },
        }
