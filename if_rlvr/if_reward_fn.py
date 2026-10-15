# Copyright 2024 AllenAI (open-instruct); 2025 verl IF-RLVR migration.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Instruction-verification reward for verl's custom reward function interface.

The training launchers use ``compute_score`` to return the fraction of
constraints satisfied, scaled by ``verification_reward``. Reasoning text
and answer tags are handled by the shared verifier before scoring.

The optional ``IFRewardManager`` provides truncation-penalty handling;
it is not selected by the standard training launchers.
"""

from __future__ import annotations

import os
import sys

# Make the self-contained verifier package importable regardless of where this file is loaded
# from (verl loads custom_reward_function by file path, without package context).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from ifeval_oi.verifier import score_ifeval  # noqa: E402  (after sys.path bootstrap)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, verification_reward=1.0, **kwargs):
    """verl ``custom_reward_function`` entry point.

    Args:
        data_source: dataset tag (``"ifeval"``); unused for dispatch (this fn is IF-specific).
        solution_str: decoded model response (thinking section stripped inside ``score_ifeval``).
        ground_truth: the string-encoded IFEval constraint spec
            ``"[{'instruction_id': [...], 'kwargs': [...]}]"``.
        extra_info: unused.
        verification_reward: optional scalar multiplier (default 1.0 -> raw fraction in [0,1],
            which scales the verifier component, not the CALIF bonus). Override via
            ``custom_reward_function.reward_kwargs.verification_reward=...``.

    Returns:
        dict with mandatory ``"score"`` key (+ ``"acc"`` = the raw fraction, for logging).
    """
    require_think_end = _env_bool("IF_REQUIRE_THINK_END_FOR_REWARD", False)
    score = score_ifeval(solution_str, ground_truth, require_think_end=require_think_end)
    return {"score": float(verification_reward) * score, "acc": score}
