# Copyright 2024 The Google Research Authors.
# Copyright 2024 AllenAI (open-instruct).
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
"""Instruction-following (IFEval) verifier, vendored from AllenAI open-instruct.

Adapted for the bundled verl runtime, including reasoning-section parsing.

The ground-truth ``label`` is a string-encoded one-element list of dicts:

    "[{'instruction_id': ['detectable_format:title', ...], 'kwargs': [None, ...]}]"

``score_ifeval`` returns the fraction of constraints satisfied, in ``[0.0, 1.0]``.
"""

from __future__ import annotations

import ast
import json
import logging

from . import instructions_registry

logger = logging.getLogger(__name__)


def _strip_gemma_thought_channel(prediction: str) -> str:
    # Gemma thinking uses: <|channel>thought\n...<channel|>[final answer].
    # When disabled, some variants may emit an empty thought block before the answer.
    for start_marker in ("<|channel>thought\n", "<|channel>thought"):
        start = prediction.find(start_marker)
        if start < 0:
            continue
        end = prediction.find("<channel|>", start + len(start_marker))
        if end >= 0:
            return prediction[end + len("<channel|>") :]
    return prediction


def _has_gemma_thought_channel(prediction: str) -> bool:
    return _strip_gemma_thought_channel(prediction) != prediction


def remove_thinking_section(prediction: str, require_think_end: bool = False) -> str | None:
    """Strip a reasoning/thinking section and answer tags before verification.

    Uses the final segment after a Qwen reasoning-end marker and removes Gemma
    thought channels when present. With ``require_think_end``, a response
    without a recognized reasoning boundary is rejected.
    """
    prediction = prediction.replace("<|assistant|>", "").strip()
    has_qwen_think_end = "</think>" in prediction
    has_gemma_thought = _has_gemma_thought_channel(prediction)
    if require_think_end and not (has_qwen_think_end or has_gemma_thought):
        return None
    # remove thinking section from the prediction
    prediction = prediction.split("</think>")[-1]
    prediction = _strip_gemma_thought_channel(prediction)
    # remove answer tags from the prediction
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    prediction = prediction.replace("<|think|>", "")
    return prediction.strip()


def score_ifeval(prediction: str, label, require_think_end: bool = False) -> float:
    """Score one instruction-following response against its constraint set.

    Based on ``IFEvalVerifier.__call__`` (open-instruct). Returns a
    float in ``[0.0, 1.0]`` equal to the fraction of constraints the (thinking-
    stripped) prediction satisfies.

    Args:
        prediction: The decoded model output (special tokens already stripped, as
            in open-instruct ``batch_decode(..., skip_special_tokens=True)``).
        label: The ground-truth constraint spec. A string-encoded list of dicts
            (as stored in the dataset); a pre-parsed list/dict is also accepted.
    """
    instruction_dict = instructions_registry.INSTRUCTION_DICT
    # Parse the ground truth. open-instruct stores it as a string and does
    # ``ast.literal_eval(label)[0]``; we accept an already-parsed list/dict too,
    # which is behaviourally identical for the string inputs used in training.
    if isinstance(label, str):
        constraint_dict = ast.literal_eval(label)
    else:
        constraint_dict = label
    constraint_dict = constraint_dict[0]
    if isinstance(constraint_dict, str):
        constraint_dict = json.loads(constraint_dict)
    answer = remove_thinking_section(prediction, require_think_end=require_think_end)
    instruction_keys = constraint_dict["instruction_id"]
    args_list = constraint_dict["kwargs"]
    rewards = []
    if answer is None:
        logger.warning("Missing </think> in reasoning response received for IFEvalVerifier.")
        return 0.0
    if len(prediction) == 0 or len(answer) == 0:
        logger.warning("Empty prediction received for IFEvalVerifier.")
        return 0.0
    for instruction_key, args in zip(instruction_keys, args_list):
        if args is None:
            args = {}
        args = {k: v for k, v in args.items() if v is not None}
        # Never let one constraint checker's exception (raised on adversarial /
        # degenerate rollouts) propagate: the reward loop has no per-sample
        # isolation up to the driver, so a single raise would crash the whole
        # training step. An unsatisfiable/erroring constraint simply scores 0.0.
        try:
            instruction_cls = instruction_dict[instruction_key]
            instruction_instance = instruction_cls(instruction_key)
            instruction_instance.build_description(**args)
            satisfied = bool(prediction.strip() and instruction_instance.check_following(answer))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "IFEval constraint %r raised %s: %s; scoring it 0.0",
                instruction_key,
                type(exc).__name__,
                exc,
            )
            satisfied = False
        rewards.append(1.0 if satisfied else 0.0)
    return sum(rewards) / max(len(rewards), 1)
