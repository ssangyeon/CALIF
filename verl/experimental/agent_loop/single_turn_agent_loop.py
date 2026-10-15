# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import math
import os
from urllib.parse import urljoin
from typing import Any

import aiohttp
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

GEMMA_THOUGHT_START_MARKERS = ("<|channel>thought\n", "<|channel>thought")
GEMMA_THOUGHT_END_MARKER = "<channel|>"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.enable_ref_ppl_baseline = os.getenv("IF_REF_PPL_BASELINE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            self.enable_px_given_y_ppl = float(
                self.config.get("if_px_given_y_reward_coeff", os.getenv("PX_GIVEN_Y_REWARD_COEFF", "0.0"))
            ) != 0.0
        except (TypeError, ValueError):
            self.enable_px_given_y_ppl = False
        try:
            self.enable_py_given_x_ppl = float(
                self.config.get(
                    "if_py_given_x_reward_coeff",
                    self.config.get("if_ppl_reward_coeff", os.getenv("PY_GIVEN_X_REWARD_COEFF", "0.0")),
                )
            ) != 0.0
        except (TypeError, ValueError):
            self.enable_py_given_x_ppl = False
        try:
            self.ref_ppl_baseline_temperature = float(
                self.config.get("if_ref_ppl_baseline_temperature", os.getenv("IF_REF_PPL_BASELINE_TEMPERATURE", "1.0"))
            )
        except (TypeError, ValueError):
            self.ref_ppl_baseline_temperature = 1.0
        self.ref_verifier_base_url = os.getenv("IF_REF_VLLM_BASE_URL", "").strip()
        self.ref_verifier_model = os.getenv("IF_REF_VLLM_MODEL", "Qwen/Qwen3-4B").strip()
        self.use_external_ref_verifier = bool(self.ref_verifier_base_url)
        if self.ref_verifier_base_url and not self.ref_verifier_base_url.endswith("/"):
            self.ref_verifier_base_url += "/"
        self.enable_ref_ppl_anchor = os.getenv("IF_REF_PPL_ANCHOR", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.enable_if_ppl_scoring = (
            self.enable_py_given_x_ppl
            or self.enable_px_given_y_ppl
            or self.enable_ref_ppl_baseline
            or self.enable_ref_ppl_anchor
        )
        self.defer_ppl_to_ref_policy = str(
            self.config.get("if_ref_policy_anchor_ppl", os.getenv("IF_REF_POLICY_ANCHOR_PPL", "0"))
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.ppl_prefix_mode = str(
            self.config.get("if_ppl_prefix_mode", os.getenv("IF_PPL_PREFIX_MODE", "standard"))
        ).strip().lower()
        if self.ppl_prefix_mode not in {"standard", "empty_think"}:
            raise ValueError(
                f"Unsupported if_ppl_prefix_mode={self.ppl_prefix_mode!r}; "
                "expected 'standard' or 'empty_think'"
            )
        self.apply_enable_thinking_kwarg = _env_bool("IF_APPLY_ENABLE_THINKING_KWARG", True)
        self.allow_missing_think_final_answer = _env_bool("IF_ALLOW_MISSING_THINK_FINAL_ANSWER", False)
        self.anchor_precompute_final_answer_only = _env_bool("IF_ANCHOR_PRECOMPUTE_FINAL_ANSWER_ONLY", False)
        self.anchor_precompute_empty_response_retries = int(os.getenv("IF_ANCHOR_PRECOMPUTE_EMPTY_RESPONSE_RETRIES", "2"))
        self.anchor_precompute_min_tokens = int(os.getenv("IF_ANCHOR_PRECOMPUTE_MIN_TOKENS", "0"))
        if self.anchor_precompute_min_tokens < 0:
            raise ValueError("IF_ANCHOR_PRECOMPUTE_MIN_TOKENS must be non-negative")
        self._ref_ppl_cache: dict[tuple[int, ...], dict[str, Any]] = {}
        self._ref_xc_ppl_cache: dict[tuple[int, ...], dict[str, Any]] = {}

    def _nonreason_chat_template_kwargs(self) -> dict[str, Any]:
        return {"enable_thinking": False} if self.apply_enable_thinking_kwarg else {}

    async def _build_ppl_x_prompt_ids(self, ppl_prompt) -> list[int]:
        """Build the x-side PPL prefix, optionally forcing an empty reasoning block.

        The Think chat template already ends in a tokenized ``<think>``.  The
        closing marker is appended as a forced continuation so the scorer sees
        ``<think></think>\n\n`` while all of those tokens remain in the prompt
        (only the separately supplied final-answer tokens receive NLL).
        """
        prefix_ids = list(
            await self.apply_chat_template(
                list(ppl_prompt),
                apply_chat_template_kwargs=self._nonreason_chat_template_kwargs(),
            )
        )
        if self.ppl_prefix_mode == "standard":
            return prefix_ids

        open_think_ids = list(self.tokenizer.encode("<think>", add_special_tokens=False))
        close_think_ids = list(self.tokenizer.encode("</think>\n\n", add_special_tokens=False))
        closed_suffix = open_think_ids + close_think_ids
        if prefix_ids[-len(closed_suffix) :] == closed_suffix:
            return prefix_ids
        if not open_think_ids or prefix_ids[-len(open_think_ids) :] != open_think_ids:
            raise RuntimeError(
                "if_ppl_prefix_mode=empty_think requires a chat template ending in '<think>'"
            )

        prefix_ids.extend(close_think_ids)
        try:
            decoded_suffix = self.tokenizer.decode(
                prefix_ids[-len(closed_suffix) :],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            decoded_suffix = self.tokenizer.decode(prefix_ids[-len(closed_suffix) :], skip_special_tokens=False)
        if decoded_suffix != "<think></think>\n\n":
            raise RuntimeError(
                "empty-think PPL prefix tokenization did not decode to '<think></think>\\n\\n'"
            )
        return prefix_ids

    @staticmethod
    def _find_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
        # Locate a marker in token space so y can be final answer only.
        if not pattern or len(pattern) > len(sequence):
            return None
        for idx in range(len(sequence) - len(pattern) + 1):
            if sequence[idx : idx + len(pattern)] == pattern:
                return idx
        return None

    @staticmethod
    def _strip_gemma_thought_text(response_text: str) -> str:
        for start_marker in GEMMA_THOUGHT_START_MARKERS:
            start = response_text.find(start_marker)
            if start < 0:
                continue
            end = response_text.find(GEMMA_THOUGHT_END_MARKER, start + len(start_marker))
            if end >= 0:
                return response_text[end + len(GEMMA_THOUGHT_END_MARKER) :]
        return response_text

    def _extract_gemma_final_answer_ids(self, response_ids: list[int]) -> tuple[list[int], bool] | None:
        end_ids = self.tokenizer.encode(GEMMA_THOUGHT_END_MARKER, add_special_tokens=False)
        for start_marker in GEMMA_THOUGHT_START_MARKERS:
            start_ids = self.tokenizer.encode(start_marker, add_special_tokens=False)
            marker_pos = self._find_subsequence(response_ids, start_ids)
            if marker_pos is None:
                continue
            search_from = marker_pos + len(start_ids)
            end_pos = self._find_subsequence(response_ids[search_from:], end_ids)
            if end_pos is None:
                continue
            final_answer_ids = self._strip_leading_whitespace_ids(
                response_ids[search_from + end_pos + len(end_ids) :]
            )
            return final_answer_ids, len(final_answer_ids) > 0

        try:
            response_text = self.tokenizer.decode(
                response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        except TypeError:
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        final_text = self._strip_gemma_thought_text(response_text)
        if final_text != response_text:
            final_answer_ids = self._strip_leading_whitespace_ids(
                self.tokenizer.encode(final_text, add_special_tokens=False)
            )
            return final_answer_ids, len(final_answer_ids) > 0
        return None

    def _strip_leading_whitespace_ids(self, token_ids: list[int]) -> list[int]:
        # Qwen3 often emits </think> followed by blank lines.
        # Score the final answer text itself, matching non-reasoning PPL.
        if not token_ids:
            return []
        try:
            text = self.tokenizer.decode(
                token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except TypeError:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        stripped = text.lstrip()
        if not stripped:
            return []
        if stripped == text:
            return list(token_ids)
        return list(self.tokenizer.encode(stripped, add_special_tokens=False))

    def _extract_final_answer_ids(self, response_ids: list[int], prompt_ids: list[int]) -> tuple[list[int], bool]:
        # Qwen uses </think>; Gemma uses <|channel>thought\n...<channel|>.
        # If thinking is disabled and no marker exists, configured non-reasoning runs can use
        # the whole response as y via IF_ALLOW_MISSING_THINK_FINAL_ANSWER=true.
        gemma_final_answer = self._extract_gemma_final_answer_ids(response_ids)
        if gemma_final_answer is not None:
            return gemma_final_answer

        # The closing think marker is ordinary byte-BPE text for tokenizers such
        # as OLMo's, so its token ids change when adjacent whitespace is merged.
        # Split the decoded response instead of matching isolated token ids.
        marker = "</think>"
        try:
            response_text = self.tokenizer.decode(
                response_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        marker_pos = response_text.find(marker)
        if marker_pos != -1:
            final_answer_text = response_text[marker_pos + len(marker) :]
            final_answer_ids = self._strip_leading_whitespace_ids(
                self.tokenizer.encode(final_answer_text, add_special_tokens=False)
            )
            return final_answer_ids, len(final_answer_ids) > 0

        try:
            prompt_text = self.tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        prompt_marker_pos = prompt_text.rfind(marker)
        prompt_after_marker = prompt_text[prompt_marker_pos + len(marker) :] if prompt_marker_pos != -1 else ""
        if prompt_marker_pos != -1 and not prompt_after_marker.strip():
            final_answer_ids = self._strip_leading_whitespace_ids(response_ids)
            return final_answer_ids, len(final_answer_ids) > 0
        if self.allow_missing_think_final_answer:
            final_answer_ids = self._strip_leading_whitespace_ids(response_ids)
            return final_answer_ids, len(final_answer_ids) > 0
        return [], False

    @staticmethod
    def _empty_ppl_fields(prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_ppl": float("inf"),
            f"{prefix}_mean_logprob": None,
            f"{prefix}_nll": None,
            f"{prefix}_token_count": 0,
            f"{prefix}_eligible": False,
        }

    @staticmethod
    def _constraint_free_prompt_text(ppl_prompt) -> str:
        if not ppl_prompt:
            return ""
        user_parts = [str(message.get("content", "")) for message in ppl_prompt if message.get("role") == "user"]
        if user_parts:
            return "\n".join(part for part in user_parts if part).strip()
        return "\n".join(str(message.get("content", "")) for message in ppl_prompt).strip()

    @staticmethod
    def _extract_final_answer_text(response_text: str, prompt_text: str = "") -> tuple[str, bool]:
        marker = "</think>"
        if marker in response_text:
            final_answer = response_text.split(marker, 1)[1]
            return final_answer, bool(final_answer.strip())
        gemma_final_answer = SingleTurnAgentLoop._strip_gemma_thought_text(response_text)
        if gemma_final_answer != response_text:
            return gemma_final_answer, bool(gemma_final_answer.strip())
        if marker in prompt_text:
            return response_text, bool(response_text.strip())
        return response_text, bool(response_text.strip())

    async def _external_vllm_completion(self, prompt_text: str, sampling_params: dict[str, Any]) -> dict[str, Any]:
        if not self.ref_verifier_base_url:
            raise RuntimeError("IF_REF_VLLM_BASE_URL is required for external frozen ref verifier")
        url = urljoin(self.ref_verifier_base_url, "v1/completions")
        payload = {
            "model": self.ref_verifier_model,
            "prompt": prompt_text,
            **sampling_params,
        }
        timeout = aiohttp.ClientTimeout(total=float(os.getenv("IF_REF_VLLM_TIMEOUT", "600")))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise RuntimeError(f"external ref vLLM error {response.status}: {text[:500]}")
                return await response.json()

    async def _external_generate_answer_ids(self, prompt_ids: list[int]) -> tuple[list[int], bool]:
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        data = await self._external_vllm_completion(
            prompt_text,
            {"max_tokens": self.response_length, "temperature": self.ref_ppl_baseline_temperature},
        )
        response_text = (data.get("choices") or [{}])[0].get("text", "")
        final_text, eligible = self._extract_final_answer_text(response_text, prompt_text)
        if not eligible:
            return [], False
        return self.tokenizer.encode(final_text, add_special_tokens=False), True

    @staticmethod
    def _extract_prompt_logprob(row: Any, token_id: int) -> float | None:
        if row is None:
            return None
        if isinstance(row, (int, float)):
            return float(row)
        if isinstance(row, list):
            values = [SingleTurnAgentLoop._extract_prompt_logprob(item, token_id) for item in row]
            values = [value for value in values if value is not None]
            return max(values) if values else None
        if isinstance(row, dict):
            direct = row.get(str(token_id), row.get(token_id))
            if direct is not None:
                if isinstance(direct, (int, float)):
                    return float(direct)
                if isinstance(direct, dict) and direct.get("logprob") is not None:
                    return float(direct["logprob"])
            candidates = []
            for value in row.values():
                if isinstance(value, (int, float)):
                    candidates.append(float(value))
                elif isinstance(value, dict) and value.get("logprob") is not None:
                    candidates.append(float(value["logprob"]))
            return max(candidates) if candidates else None
        return None

    async def _score_continuation_ppl_external(
        self, prefix_ids: list[int], continuation_ids: list[int], metric_prefix: str
    ) -> dict[str, Any]:
        if not prefix_ids or not continuation_ids:
            return self._empty_ppl_fields(metric_prefix)
        sequence_ids = prefix_ids + continuation_ids
        prompt_text = self.tokenizer.decode(sequence_ids, skip_special_tokens=False)
        data = await self._external_vllm_completion(
            prompt_text,
            {"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 1},
        )
        choice = (data.get("choices") or [{}])[0]
        logprobs = choice.get("logprobs") or {}
        prompt_logprobs = logprobs.get("prompt_logprobs") or choice.get("prompt_logprobs") or []
        start = max(len(prefix_ids) - 1, 0)
        rows = prompt_logprobs[start : start + len(continuation_ids)]
        token_logprobs = []
        for row, token_id in zip(rows, continuation_ids, strict=False):
            logprob = self._extract_prompt_logprob(row, token_id)
            if logprob is not None:
                token_logprobs.append(float(logprob))
        if len(token_logprobs) != len(continuation_ids):
            raise ValueError(
                f"{metric_prefix} external PPL logprob length mismatch: "
                f"got {len(token_logprobs)}, expected {len(continuation_ids)}"
            )
        mean_logprob = sum(token_logprobs) / max(len(token_logprobs), 1)
        nll = -sum(token_logprobs)
        return {
            f"{metric_prefix}_ppl": float(math.exp(-mean_logprob)),
            f"{metric_prefix}_mean_logprob": float(mean_logprob),
            f"{metric_prefix}_nll": float(nll),
            f"{metric_prefix}_token_count": len(token_logprobs),
            f"{metric_prefix}_eligible": True,
        }

    async def _score_continuation_ppl(
        self, prefix_ids: list[int], continuation_ids: list[int], metric_prefix: str
    ) -> dict[str, Any]:
        if not prefix_ids or not continuation_ids:
            return self._empty_ppl_fields(metric_prefix)
        if self.use_external_ref_verifier:
            return await self._score_continuation_ppl_external(prefix_ids, continuation_ids, metric_prefix)

        sequence_ids = prefix_ids + continuation_ids
        score_output: TokenOutput = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params={"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0},
        )
        prompt_logprobs = score_output.extra_fields.get("prompt_logprobs") or []
        start = max(len(prefix_ids) - 1, 0)
        rows = prompt_logprobs[start : start + len(continuation_ids)]
        token_logprobs = [float(row[0]) for row in rows if row and row[0] is not None]
        if len(token_logprobs) != len(continuation_ids):
            raise ValueError(
                f"{metric_prefix} PPL logprob length mismatch: "
                f"got {len(token_logprobs)}, expected {len(continuation_ids)}"
            )
        mean_logprob = sum(token_logprobs) / max(len(token_logprobs), 1)
        nll = -sum(token_logprobs)
        return {
            f"{metric_prefix}_ppl": float(math.exp(-mean_logprob)),
            f"{metric_prefix}_mean_logprob": float(mean_logprob),
            f"{metric_prefix}_nll": float(nll),
            f"{metric_prefix}_token_count": len(token_logprobs),
            f"{metric_prefix}_eligible": True,
        }

    async def _compute_bidirectional_ppl(
        self, ppl_prompt, response_ids: list[int], prompt_ids: list[int]
    ) -> dict[str, Any]:
        # Score p(y|x) and optionally p(x|y), where y is the final answer and x is the constraint-free prompt.
        final_answer_ids, eligible = self._extract_final_answer_ids(response_ids, prompt_ids)
        defaults = {
            **self._empty_ppl_fields("p_y_given_x"),
            **self._empty_ppl_fields("p_x_given_y"),
            **self._empty_ppl_fields("p_y_ref_given_x"),
            **self._empty_ppl_fields("p_y_ref_xc_given_x"),
            "ppl_x_prompt_ids": [],
            "ppl_y_final_answer_ids": [],
        }
        if not eligible or not ppl_prompt:
            return defaults

        extra_fields = dict(defaults)
        x_prompt_ids = None
        try:
            x_prompt_ids = await self._build_ppl_x_prompt_ids(ppl_prompt)
            extra_fields["ppl_x_prompt_ids"] = list(x_prompt_ids)
            extra_fields["ppl_y_final_answer_ids"] = list(final_answer_ids)
            if self.enable_py_given_x_ppl:
                extra_fields.update(await self._score_continuation_ppl(x_prompt_ids, final_answer_ids, "p_y_given_x"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("##6/3 ppl## p(y|x) PPL scoring failed: %s", exc)

        if self.enable_ref_ppl_baseline and x_prompt_ids:
            cache_key = tuple(x_prompt_ids)
            try:
                cached_ref = self._ref_ppl_cache.get(cache_key)
                if cached_ref is None:
                    if self.use_external_ref_verifier:
                        ref_answer_ids, ref_eligible = await self._external_generate_answer_ids(x_prompt_ids)
                    else:
                        ref_output: TokenOutput = await self.server_manager.generate(
                            request_id=uuid4().hex,
                            prompt_ids=x_prompt_ids,
                            sampling_params={"max_tokens": self.response_length, "temperature": self.ref_ppl_baseline_temperature},
                        )
                        ref_response_ids = ref_output.token_ids[: self.response_length]
                        ref_answer_ids, ref_eligible = self._extract_final_answer_ids(ref_response_ids, x_prompt_ids)
                    if ref_eligible:
                        cached_ref = await self._score_continuation_ppl(
                            x_prompt_ids, ref_answer_ids, "p_y_ref_given_x"
                        )
                    else:
                        cached_ref = self._empty_ppl_fields("p_y_ref_given_x")
                    self._ref_ppl_cache[cache_key] = dict(cached_ref)
                extra_fields.update(dict(cached_ref))
            except Exception as exc:  # noqa: BLE001
                logger.warning("##6/3 ppl## reference p(y_ref|x) PPL scoring failed: %s", exc)

        if self.enable_ref_ppl_anchor and x_prompt_ids:
            cache_key = tuple(prompt_ids)
            try:
                cached_ref_xc = self._ref_xc_ppl_cache.get(cache_key)
                if cached_ref_xc is None:
                    if self.use_external_ref_verifier:
                        ref_xc_answer_ids, ref_xc_eligible = await self._external_generate_answer_ids(prompt_ids)
                    else:
                        ref_xc_output: TokenOutput = await self.server_manager.generate(
                            request_id=uuid4().hex,
                            prompt_ids=prompt_ids,
                            sampling_params={"max_tokens": self.response_length, "temperature": self.ref_ppl_baseline_temperature},
                        )
                        ref_xc_response_ids = ref_xc_output.token_ids[: self.response_length]
                        ref_xc_answer_ids, ref_xc_eligible = self._extract_final_answer_ids(
                            ref_xc_response_ids, prompt_ids
                        )
                    if ref_xc_eligible:
                        cached_ref_xc = await self._score_continuation_ppl(
                            x_prompt_ids, ref_xc_answer_ids, "p_y_ref_xc_given_x"
                        )
                    else:
                        cached_ref_xc = self._empty_ppl_fields("p_y_ref_xc_given_x")
                    self._ref_xc_ppl_cache[cache_key] = dict(cached_ref_xc)
                extra_fields.update(dict(cached_ref_xc))
            except Exception as exc:  # noqa: BLE001
                logger.warning("##6/3 ppl## reference p(y_ref_xc|x) PPL scoring failed: %s", exc)

        if not self.enable_px_given_y_ppl:
            return extra_fields

        try:
            final_answer_text = self.tokenizer.decode(final_answer_ids, skip_special_tokens=True).strip()
            x_text = self._constraint_free_prompt_text(ppl_prompt)
            if not final_answer_text or not x_text:
                return extra_fields
            y_prompt_ids = await self.apply_chat_template(
                [{"role": "user", "content": final_answer_text}],
                apply_chat_template_kwargs=self._nonreason_chat_template_kwargs(),
            )
            x_response_ids = self.tokenizer.encode(x_text, add_special_tokens=False)
            extra_fields.update(await self._score_continuation_ppl(y_prompt_ids, x_response_ids, "p_x_given_y"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("##6/3 ppl## p(x|y) PPL scoring failed: %s", exc)

        return extra_fields

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        is_anchor_precompute = bool(kwargs.get("__if_anchor_precompute__", False))
        if is_anchor_precompute and self.anchor_precompute_min_tokens:
            # The cache validator requires a non-empty continuation. Setting a
            # one-token floor is equivalent to rejecting an immediate EOS and
            # retrying, without altering ordinary training rollouts.
            sampling_params = dict(sampling_params)
            sampling_params["min_tokens"] = max(
                int(sampling_params.get("min_tokens", 0) or 0), self.anchor_precompute_min_tokens
            )

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            token_output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                audio_data=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = token_output.num_preempted if token_output.num_preempted is not None else -1
        response_ids = token_output.token_ids[: self.response_length]
        response_mask = [1] * len(response_ids)

        # Score all rollouts while vLLM is still awake; trainer later gates by old reward.
        final_answer_ids, final_answer_eligible = self._extract_final_answer_ids(response_ids, prompt_ids)
        if is_anchor_precompute:
            retry_count = 0
            score_answer_ids = list(final_answer_ids) if final_answer_eligible else []
            if not score_answer_ids and not self.anchor_precompute_final_answer_only:
                score_answer_ids = self._strip_leading_whitespace_ids(response_ids)
            while not score_answer_ids and retry_count < self.anchor_precompute_empty_response_retries:
                retry_count += 1
                logger.warning("IF anchor precompute got an empty answer; retrying generation (%d/%d)", retry_count, self.anchor_precompute_empty_response_retries)
                with simple_timer(f"generate_sequences_retry_{retry_count}", metrics):
                    token_output = await self.server_manager.generate(
                        request_id=uuid4().hex,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=images,
                        video_data=videos,
                        audio_data=audios,
                        mm_processor_kwargs=mm_processor_kwargs,
                    )
                response_ids = token_output.token_ids[: self.response_length]
                response_mask = [1] * len(response_ids)
                final_answer_ids, final_answer_eligible = self._extract_final_answer_ids(response_ids, prompt_ids)
                score_answer_ids = list(final_answer_ids) if final_answer_eligible else []
                if not score_answer_ids and not self.anchor_precompute_final_answer_only:
                    score_answer_ids = self._strip_leading_whitespace_ids(response_ids)
            if retry_count:
                metrics["anchor_precompute_empty_response_retries"] = retry_count
            if not score_answer_ids:
                metrics["anchor_precompute_missing_final_answer"] = 1
        ppl_extra_fields = {
            "if_final_answer_ids": list(final_answer_ids) if final_answer_eligible else [],
        }
        if (
            not kwargs.get("__if_anchor_precompute__", False)
            and not self.defer_ppl_to_ref_policy
            and self.enable_if_ppl_scoring
        ):
            ppl_extra_fields.update(
                await self._compute_bidirectional_ppl(kwargs.get("ppl_prompt"), response_ids, prompt_ids)
            )
        elif kwargs.get("ppl_prompt") and (kwargs.get("__if_anchor_precompute__", False) or self.defer_ppl_to_ref_policy):
            try:
                x_prompt_ids = await self._build_ppl_x_prompt_ids(kwargs.get("ppl_prompt"))
                ppl_extra_fields["ppl_x_prompt_ids"] = list(x_prompt_ids)
                score_answer_ids = list(final_answer_ids) if final_answer_eligible else []
                # Missing final answers must not reward unfinished reasoning in either likelihood baseline.
                likelihood_reward = str(self.config.get("if_ppl_reward_strategy", "rank")).strip().lower() in {
                    "inverse_ppl",
                    "raw_likelihood",
                }
                if not score_answer_ids and not likelihood_reward and not (
                    kwargs.get("__if_anchor_precompute__", False) and self.anchor_precompute_final_answer_only
                ):
                    score_answer_ids = self._strip_leading_whitespace_ids(response_ids)
                ppl_extra_fields["ppl_y_final_answer_ids"] = list(score_answer_ids)
                if kwargs.get("__if_anchor_precompute__", False):
                    if score_answer_ids:
                        ppl_extra_fields.update(
                            await self._score_continuation_ppl(x_prompt_ids, score_answer_ids, "p_y_given_x")
                        )
                    else:
                        ppl_extra_fields.update(
                            {
                                "p_y_given_x_ppl": float("inf"),
                                "p_y_given_x_mean_logprob": None,
                                "p_y_given_x_nll": float("inf"),
                                "p_y_given_x_token_count": 0,
                                "p_y_given_x_eligible": False,
                            }
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning("##6/3 ppl## ref-policy PPL token prep failed: %s", exc)
        extra_fields = dict(token_output.extra_fields)
        extra_fields.update(ppl_extra_fields)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=token_output.log_probs[: self.response_length] if token_output.log_probs else None,
            routed_experts=(
                token_output.routed_experts[: len(prompt_ids) + self.response_length]
                if token_output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
