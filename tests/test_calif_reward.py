"""CPU regression tests for CALIF, independent of Ray, CUDA, and model downloads.

The production functions are AST-loaded rather than copied: assertions exercise
the actual trainer and rollout helpers while keeping distributed imports out of
the unit-test process. Run: python -m unittest discover -s tests -p test_calif_reward.py
"""

import ast
import asyncio
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_helpers(relative_path, names, constants=()):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        + assignments
        + functions,
        type_ignores=[],
    )
    namespace = {"np": np, "torch": torch, "math": math, "os": os, "json": json}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


HELPERS = load_helpers(
    "verl/trainer/ppo/ray_trainer.py",
    {
        "normalize_bool_config",
        "normalize_if_ppl_anchor_reward_mode",
        "apply_if_ppl_anchor_reward",
        "_if_ref_anchor_train_indices",
        "_if_ref_anchor_cache_item_complete",
        "_precompute_if_ref_anchor_cache",
        "_attach_if_ref_anchor_cache",
        "_load_if_ref_anchor_cache",
        "_compute_anchor_ppl_with_ref_policy",
    },
    {"_IF_PPL_ANCHOR_REWARD_MODES"},
)
NEED_REFERENCE = load_helpers("verl/trainer/ppo/utils.py", {"need_reference_policy"})["need_reference_policy"]
ANSWER_HELPERS = load_helpers(
    "verl/experimental/agent_loop/single_turn_agent_loop.py",
    {"_strip_leading_whitespace_ids", "_extract_final_answer_ids", "_build_ppl_x_prompt_ids"},
)


class Config(dict):
    def __getattr__(self, name):
        return self[name]


class Dataset:
    def __init__(self, indices):
        self.dataframe = {"extra_info": [{"index": index} for index in indices]}

    def __len__(self):
        return len(self.dataframe["extra_info"])


class CharacterTokenizer:
    """No pretrained model is needed to test delimiter/extraction semantics."""

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


def complete_anchor():
    return {
        "y0": [4, 5], "y1": [6],
        "ref0_nll": 2.0, "ref0_token_count": 2, "ref0_ppl": math.e,
        "ref1_nll": 2.0, "ref1_token_count": 1, "ref1_ppl": math.exp(2),
    }


def cache_trainer(indices, cache, strict=True):
    trainer = SimpleNamespace(
        config=Config(if_ref_anchor_precompute=True, if_ref_anchor_require_complete_cache=strict),
        train_dataset=Dataset(indices),
        if_ref_anchor_cache=cache,
        _load_if_ref_anchor_cache=lambda: True,
    )
    for name in ("_if_ref_anchor_train_indices", "_if_ref_anchor_cache_item_complete"):
        setattr(trainer, name, MethodType(HELPERS[name], trainer))
    return trainer


def reward_batch(policy, ref0, ref1, counts=None):
    counts = [1] * len(policy) if counts is None else counts
    return SimpleNamespace(
        batch={"response_mask": torch.tensor([[1, 1, 0]] * len(policy))},
        non_tensor_batch={
            "p_y_given_x_nll": policy,
            "p_y_given_x_token_count": counts,
            "p_y_ref_given_x_nll": ref0,
            "p_y_ref_given_x_token_count": [1] * len(policy),
            "p_y_ref_xc_given_x_nll": ref1,
            "p_y_ref_xc_given_x_token_count": [1] * len(policy),
        },
    )


class CALIFRewardTests(unittest.TestCase):
    def setUp(self):
        # Isolate the published reward from developer ablation environment knobs.
        self.environment = patch.dict(os.environ, {
            "IF_PPL_ANCHOR_FLIP_HANDLING": "strict",
            "IF_PPL_ANCHOR_FLOOR_ACTION": "zero",
            "IF_PPL_ANCHOR_FLOOR_MARGIN": "0.0",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_published_reward_branches_and_closed_interval(self):
        # NLL increases when reference log-likelihood decreases. The interval
        # endpoints are inclusive. With inverted bounds, likelihood above the
        # x-only upper bound is still suppressed, as specified in the paper.
        batch = reward_batch([0.5, 1, 1.5, 2, 3, 1.5, 1.5, 2.5], [1] * 6 + [2, 2], [2] * 6 + [1, 1])
        rewards = torch.tensor([[0, 1, 0]] * 5 + [[0, 0, 0]] + [[0, 1, 0]] * 2, dtype=torch.float32)
        original = rewards.clone()
        updated, metrics = HELPERS["apply_if_ppl_anchor_reward"](batch, rewards, ppl_reward_coeff=0.1)
        self.assertTrue(torch.allclose(updated.sum(-1), torch.tensor([0, 1.1, 1.1, 1.1, 1, 0, 0, 1])))
        self.assertTrue(torch.equal(rewards, original))
        self.assertTrue(torch.equal(updated[:, 0], torch.zeros(8)))
        self.assertTrue(torch.equal(updated[:, 2], torch.zeros(8)))
        self.assertEqual(metrics["if_p_y_given_x_anchor/lower_zero_count"], 2)
        self.assertEqual(metrics["if_p_y_given_x_anchor/interval_bonus_count"], 3)
        self.assertEqual(metrics["if_p_y_given_x_anchor/invalid_anchor_count"], 1)

    def test_legacy_inverted_abstention_remains_explicitly_opt_in(self):
        batch = reward_batch([1.5], [2], [1])
        with patch.dict(os.environ, {"IF_PPL_ANCHOR_FLIP_HANDLING": "abstain"}):
            updated, _ = HELPERS["apply_if_ppl_anchor_reward"](batch, torch.tensor([[0.0, 1.0, 0.0]]), 0.1)
        self.assertEqual(updated.sum().item(), 1.0)

    def test_materialized_inverted_scores_use_existing_strict_reward(self):
        item = dict(complete_anchor(), y1=[4, 5], ref1_nll=1.0,
                    ref1_token_count=2, ref1_ppl=math.exp(0.5))
        trainer = cache_trainer([9], {9: item})
        HELPERS["_precompute_if_ref_anchor_cache"](trainer)
        batch = reward_batch([0.25, 0.75, 1, 1.5, 3], [0] * 5, [0] * 5)
        batch.non_tensor_batch["index"] = [9] * 5
        metrics = HELPERS["_attach_if_ref_anchor_cache"](trainer, batch)
        self.assertEqual(metrics["if_ref_anchor_cache/hit_count"], 5)
        updated, metrics = HELPERS["apply_if_ppl_anchor_reward"](
            batch, torch.tensor([[0., 1., 0.]] * 5), 0.1)
        self.assertEqual(updated.sum(-1).tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(metrics["if_p_y_given_x_anchor/interval_bonus_count"], 0)
        self.assertEqual(metrics["if_p_y_given_x_anchor/lower_zero_count"], 2)

    def test_reward_uses_mean_nll_not_summed_nll(self):
        batch = reward_batch([3.0], [1.0], [2.0], counts=[2])
        updated, _ = HELPERS["apply_if_ppl_anchor_reward"](batch, torch.tensor([[0.0, 1.0, 0.0]]), 0.1)
        self.assertAlmostEqual(updated.sum().item(), 1.1, places=6)

    def test_zero_verifier_score_never_receives_anchor_bonus(self):
        batch = reward_batch([1.5], [1], [2])
        updated, _ = HELPERS["apply_if_ppl_anchor_reward"](batch, torch.zeros(1, 3), 0.1)
        self.assertEqual(updated.sum().item(), 0.0)

    def test_kl_zero_still_keeps_frozen_calif_reference(self):
        cfg = Config(
            algorithm=Config(use_kl_in_reward=False),
            actor_rollout_ref=Config(actor=Config(use_kl_loss=False)),
            if_ref_policy_anchor_ppl=True,
            if_py_given_x_reward_coeff=0.1,
        )
        self.assertTrue(NEED_REFERENCE(cfg))
        cfg["if_py_given_x_reward_coeff"] = 0.0
        self.assertFalse(NEED_REFERENCE(cfg))
        cfg.actor_rollout_ref.actor["use_kl_loss"] = True
        self.assertTrue(NEED_REFERENCE(cfg))


class AnchorCoverageTests(unittest.TestCase):
    def test_same_cache_size_with_wrong_ids_is_rejected(self):
        trainer = cache_trainer([10, 30], {10: complete_anchor(), 20: complete_anchor()})
        with self.assertRaisesRegex(RuntimeError, "active training IDs"):
            HELPERS["_precompute_if_ref_anchor_cache"](trainer)

    def test_complete_active_subset_accepts_unused_cache_rows(self):
        trainer = cache_trainer([10, 30], {10: complete_anchor(), 20: complete_anchor(), 30: complete_anchor()})
        HELPERS["_precompute_if_ref_anchor_cache"](trainer)

    def test_nonfinite_anchor_is_treated_as_missing(self):
        item = complete_anchor()
        item["ref0_nll"] = math.inf
        trainer = cache_trainer([10], {10: item})
        with self.assertRaisesRegex(RuntimeError, "missing or invalid"):
            HELPERS["_precompute_if_ref_anchor_cache"](trainer)

    def test_duplicate_source_indices_are_rejected(self):
        trainer = cache_trainer([10, 10], {10: complete_anchor()})
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            HELPERS["_precompute_if_ref_anchor_cache"](trainer)

    def test_every_repeated_rollout_gets_same_cached_anchors(self):
        trainer = cache_trainer([10, 30], {10: complete_anchor(), 30: complete_anchor()})
        batch = SimpleNamespace(non_tensor_batch={"index": [10, 30, 10, 30]})
        metrics = HELPERS["_attach_if_ref_anchor_cache"](trainer, batch)
        self.assertEqual(metrics["if_ref_anchor_cache/hit_count"], 4)
        self.assertEqual(batch.non_tensor_batch["p_y_ref_given_x_nll"].tolist(), [2.0] * 4)

    def test_missing_batch_anchor_is_fatal_before_reward(self):
        trainer = cache_trainer([10], {10: complete_anchor()})
        with self.assertRaisesRegex(RuntimeError, "missing or invalid required anchors"):
            HELPERS["_attach_if_ref_anchor_cache"](trainer, SimpleNamespace(non_tensor_batch={"index": [11]}))

    def test_missing_batch_indices_are_fatal(self):
        trainer = cache_trainer([10], {10: complete_anchor()})
        with self.assertRaisesRegex(RuntimeError, "rollout indices"):
            HELPERS["_attach_if_ref_anchor_cache"](trainer, SimpleNamespace(non_tensor_batch={}))

    def test_constraint_only_does_not_require_or_attach_anchors(self):
        trainer = cache_trainer([10], {})
        trainer.config["if_ref_anchor_precompute"] = False
        self.assertEqual(HELPERS["_attach_if_ref_anchor_cache"](trainer, SimpleNamespace(non_tensor_batch={})), {})

    def test_release_loader_uses_validated_payload_and_active_ids(self):
        trainer = cache_trainer([10], {})
        trainer.config["if_release_cache_setting"] = "qwen3_4b_thinking"
        trainer._if_ref_anchor_cache_metadata = lambda: {"ppl_prefix_mode": "standard"}
        payload = {"metadata": {"version": 2}, "items": {"10": complete_anchor()}}
        calls = []

        def validate(path, setting, expected_metadata, required_indices):
            calls.append((setting, expected_metadata, required_indices))
            return payload

        with tempfile.NamedTemporaryFile() as handle:
            trainer._if_ref_anchor_cache_path = lambda: handle.name
            with patch.dict(sys.modules, {"calif_training.cache": SimpleNamespace(validate_for_training=validate)}):
                self.assertTrue(HELPERS["_load_if_ref_anchor_cache"](trainer))
        self.assertEqual(calls, [("qwen3_4b_thinking", {"ppl_prefix_mode": "standard"}, {10})])
        self.assertEqual(set(trainer.if_ref_anchor_cache), {10})

    def test_release_validator_failure_is_not_silently_ignored(self):
        trainer = cache_trainer([10], {})
        trainer.config["if_release_cache_setting"] = "qwen3_4b_thinking"
        trainer._if_ref_anchor_cache_metadata = lambda: {}

        def reject(*args):
            raise ValueError("checksum mismatch")

        with tempfile.NamedTemporaryFile() as handle:
            trainer._if_ref_anchor_cache_path = lambda: handle.name
            with patch.dict(sys.modules, {"calif_training.cache": SimpleNamespace(validate_for_training=reject)}):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    HELPERS["_load_if_ref_anchor_cache"](trainer)


class FinalAnswerScoringTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.loop = SimpleNamespace(tokenizer=self.tokenizer, allow_missing_think_final_answer=False,
                                    _extract_gemma_final_answer_ids=lambda ids: None)
        self.loop._strip_leading_whitespace_ids = MethodType(ANSWER_HELPERS["_strip_leading_whitespace_ids"], self.loop)

    def extract(self, response, prompt="x+c<think>"):
        ids, valid = ANSWER_HELPERS["_extract_final_answer_ids"](
            self.loop, self.tokenizer.encode(response), self.tokenizer.encode(prompt)
        )
        return self.tokenizer.decode(ids), valid

    def test_thinking_answer_excludes_reasoning_and_delimiter(self):
        self.assertEqual(self.extract("reasoning text</think>\n\nFinal answer"), ("Final answer", True))

    def test_missing_or_empty_final_answer_is_not_marked_eligible(self):
        self.assertEqual(self.extract("unfinished reasoning"), ("", False))
        self.assertEqual(self.extract("reasoning</think>\n\n"), ("", False))

    def test_non_thinking_qwen_closed_prefix_uses_entire_response(self):
        self.assertEqual(self.extract("Final answer", "x+c<think>\n\n</think>\n\n"), ("Final answer", True))

    def test_tulu_without_think_delimiters_uses_entire_response(self):
        self.loop.allow_missing_think_final_answer = True
        self.assertEqual(self.extract("Final answer", "x+c<assistant>"), ("Final answer", True))

    def test_reference_prompt_is_x_only_and_non_thinking(self):
        calls = []

        async def template(messages, apply_chat_template_kwargs):
            calls.append((messages, apply_chat_template_kwargs))
            return self.tokenizer.encode(messages[0]["content"] + "<think>\n\n</think>\n\n")

        self.loop.apply_chat_template = template
        self.loop._nonreason_chat_template_kwargs = lambda: {"enable_thinking": False}
        self.loop.ppl_prefix_mode = "standard"
        prompt = [{"role": "user", "content": "original x"}]
        ids = asyncio.run(ANSWER_HELPERS["_build_ppl_x_prompt_ids"](self.loop, prompt))
        self.assertEqual(calls, [(prompt, {"enable_thinking": False})])
        self.assertEqual(self.tokenizer.decode(ids), "original x<think>\n\n</think>\n\n")

    def test_precomputed_anchor_scores_are_not_recomputed(self):
        calls = []

        def score(prefixes, answers):
            calls.append((prefixes, answers))
            return np.array([3.0]), np.array([2]), np.array([math.exp(1.5)])

        trainer = SimpleNamespace(
            config=Config(if_ref_policy_anchor_ppl=True, if_ref_anchor_require_complete_cache=True),
            _validate_ppl_x_prefix_ids=lambda prefixes: None,
            _score_continuations_with_ref_policy=score,
        )
        batch = SimpleNamespace(non_tensor_batch={
            "ppl_x_prompt_ids": [[1, 2]], "ppl_y_final_answer_ids": [[3, 4]],
            "p_y_ref_given_x_nll": [1.0], "p_y_ref_given_x_token_count": [1], "p_y_ref_given_x_ppl": [math.e],
            "p_y_ref_xc_given_x_nll": [2.0], "p_y_ref_xc_given_x_token_count": [1],
            "p_y_ref_xc_given_x_ppl": [math.exp(2)],
        })
        HELPERS["_compute_anchor_ppl_with_ref_policy"](trainer, batch)
        self.assertEqual(calls, [([[1, 2]], [[3, 4]])])
        self.assertEqual(batch.non_tensor_batch["p_y_given_x_nll"].tolist(), [3.0])

    def test_missing_reference_fields_cannot_silently_disable_calif(self):
        trainer = SimpleNamespace(config=Config(if_ref_policy_anchor_ppl=True, if_ref_anchor_require_complete_cache=True))
        with self.assertRaisesRegex(RuntimeError, "constraint-only rewards"):
            HELPERS["_compute_anchor_ppl_with_ref_policy"](trainer, SimpleNamespace(non_tensor_batch={}))


if __name__ == "__main__":
    unittest.main()
