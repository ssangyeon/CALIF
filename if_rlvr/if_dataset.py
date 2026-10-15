#!/usr/bin/env python
"""Load instruction-following data from the Hugging Face Hub for verl training.

Each row is converted into chat messages, verifier constraints, and a
constraint-free request for reference-policy scoring. The dataset is shuffled
with a fixed seed before creating disjoint training and validation splits.

The launchers select this class through ``data.custom_cls``. Split tokens
containing val/eval/valid/test select validation; other tokens select training.
Dataset ID, revision, validation size, and seed are configured under
``config.data``.
"""

from __future__ import annotations

import json
import math
import os
import re

import datasets
import numpy as np

from verl.utils.dataset.rl_dataset import RLHFDataset

HF_DATASET = "allenai/IF_multi_constraints_upto5"
DATA_SOURCE = "ifeval"
_VAL_MARKERS = ("val", "eval", "valid", "test")


def _split_constraints(raw) -> list[str]:
    # Constraints are tab-separated in the dataset but space-separated in prompts.
    # Match each constraint separately when extracting the underlying request.
    return [part.strip() for part in (raw or "").split("\t") if part.strip()]


def _strip_constraints_from_text(content: str, parts: list[str]) -> str | None:
    # Extract the prefix before the first constraint span and trim surrounding whitespace.
    # Return None when no constraint is present; use span removal as a fallback.
    starts = [content.find(part) for part in parts]
    found = [pos for pos in starts if pos >= 0]
    if not found:
        return None
    base = content[: min(found)].strip()
    if base and not any(part in base for part in parts):
        return base
    # Fallback (constraints not a clean suffix here, e.g. base is empty or a span repeats): remove
    # every constraint occurrence in place, longest first so nested spans can't partially match.
    base = content
    for part in sorted(parts, key=len, reverse=True):
        base = base.replace(part, " ")
    return re.sub(r"[ \t]{2,}", " ", base).strip()


def _make_constraint_free_messages(example: dict) -> list[dict]:
    # Build x for p(y|x) and p(x|y): drop the constraint text c from the user prompt.
    parts = _split_constraints(example.get("constraint"))
    messages = [{"role": m["role"], "content": m["content"]} for m in example["messages"]]
    if not parts:
        return messages
    for message in messages:
        if message.get("role") != "user":
            continue
        stripped = _strip_constraints_from_text(message.get("content", ""), parts)
        if stripped is not None:
            message["content"] = stripped
    return messages


def _to_verl_row(example: dict, idx: int) -> dict:
    """Map one dataset row to verl's RLVR schema."""
    messages = [{"role": m["role"], "content": m["content"]} for m in example["messages"]]
    return {
        "data_source": DATA_SOURCE,
        "prompt": messages,
        # Carried to the agent loop for vLLM scoring of final-answer p(y|x).
        "ppl_prompt": _make_constraint_free_messages(example),
        "ability": "instruction_following",
        "reward_model": {
            "style": "rule",
            "ground_truth": example["ground_truth"],  # string-encoded IFEval constraint spec
        },
        "extra_info": {
            "index": idx,
            "key": example.get("key", ""),
            "constraint": example.get("constraint", ""),
            "constraint_type": example.get("constraint_type", ""),
        },
    }


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _complete_anchor_cache_indices(path: str) -> set[int] | None:
    path = (path or "").strip()
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load IF ref anchor cache for dataset filtering: {path}: {exc}") from exc

    expected_prefix_mode = os.getenv("IF_PPL_PREFIX_MODE", "").strip().lower()
    if expected_prefix_mode:
        metadata = payload.get("metadata", {})
        actual_prefix_mode = str(metadata.get("ppl_prefix_mode", "")).strip().lower()
        actual_nll_scope = str(metadata.get("ppl_nll_scope", "")).strip().lower()
        expected_model_path = os.getenv("MODEL_PATH", "").strip()
        try:
            cache_version = int(metadata.get("version", -1))
        except (TypeError, ValueError):
            cache_version = -1
        legacy_final_answer_cache_compatible = (
            _env_bool("IF_REF_ANCHOR_ALLOW_LEGACY_FINAL_ANSWER_CACHE", False)
            and cache_version == 2
            and expected_prefix_mode == "standard"
            and "ppl_prefix_mode" not in metadata
            and "ppl_nll_scope" not in metadata
            and (not expected_model_path or metadata.get("model_path") == expected_model_path)
        )
        if legacy_final_answer_cache_compatible:
            actual_prefix_mode = "standard"
            actual_nll_scope = "final_answer_tokens_only"
            print(
                f"[IFMultiConstraintsDataset] accepting legacy v2 final-answer cache semantics: {path}"
            )
        if actual_prefix_mode != expected_prefix_mode or actual_nll_scope != "final_answer_tokens_only":
            raise RuntimeError(
                "IF ref anchor cache PPL semantics mismatch: "
                f"expected prefix_mode={expected_prefix_mode!r}, nll_scope='final_answer_tokens_only'; "
                f"got prefix_mode={actual_prefix_mode!r}, nll_scope={actual_nll_scope!r}: {path}"
            )

    indices: set[int] = set()
    for key, item in payload.get("items", {}).items():
        try:
            index = int(key)
            ref0_count = int(item.get("ref0_token_count", 0) or 0)
            ref1_count = int(item.get("ref1_token_count", 0) or 0)
            ref0_nll = float(item.get("ref0_nll", float("inf")))
            ref1_nll = float(item.get("ref1_nll", float("inf")))
        except (TypeError, ValueError):
            continue
        if (
            item.get("y0")
            and item.get("y1")
            and ref0_count > 0
            and ref1_count > 0
            and math.isfinite(ref0_nll)
            and math.isfinite(ref1_nll)
        ):
            indices.add(index)
    return indices


class IFMultiConstraintsDataset(RLHFDataset):
    """RLHFDataset that sources its rows from the HF Hub instead of local parquet files."""

    def _download(self, use_origin_parquet=False):
        # Data comes from the HF Hub (handled in _read_files_and_tokenize); nothing to copy.
        return

    def _read_files_and_tokenize(self):
        cfg = self.config
        hf_name = cfg.get("if_dataset_hf", HF_DATASET)
        val_size = int(cfg.get("if_dataset_val_size", 512))
        seed = int(cfg.get("if_dataset_seed", 1))

        # Which split does THIS instance want? verl passes data.train_files for train and
        # data.val_files for val; we read the token to decide (see module docstring).
        tokens = " ".join(str(f) for f in self.original_data_files).lower()
        want_val = any(m in tokens for m in _VAL_MARKERS)

        revision = cfg.get("if_dataset_revision", None)
        ds = datasets.load_dataset(hf_name, split="train", revision=revision)
        ds = ds.shuffle(seed=seed)  # deterministic train/validation split
        if want_val:
            ds = ds.select(range(val_size))
        else:
            ds = ds.select(range(val_size, len(ds)))

        rows = [_to_verl_row(ds[i], i) for i in range(len(ds))]
        if not want_val and _env_bool("IF_REF_ANCHOR_TRAIN_CACHED_ONLY", False):
            cache_path = os.getenv("IF_REF_ANCHOR_CACHE_PATH", "").strip()
            cached_indices = _complete_anchor_cache_indices(cache_path)
            if cached_indices is None:
                raise RuntimeError("IF_REF_ANCHOR_TRAIN_CACHED_ONLY=true requires IF_REF_ANCHOR_CACHE_PATH")
            before = len(rows)
            rows = [row for row in rows if int(row["extra_info"]["index"]) in cached_indices]
            if not rows:
                raise RuntimeError(f"No train rows matched complete IF ref anchor cache entries: {cache_path}")
            print(
                f"[IFMultiConstraintsDataset] filtered train split to {len(rows)} cached-anchor rows "
                f"from {before} rows using {cache_path}"
            )

        # Optional explicit index filter (targeted anchor regeneration / subset runs).
        # File format: JSON list of ints, or an object with an "indices" list.
        index_filter_path = os.getenv("IF_TRAIN_INDEX_FILTER", "").strip()
        if not want_val and index_filter_path:
            with open(index_filter_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            wanted = payload.get("indices") if isinstance(payload, dict) else payload
            wanted = {int(i) for i in (wanted or [])}
            if not wanted:
                raise RuntimeError(f"IF_TRAIN_INDEX_FILTER has no indices: {index_filter_path}")
            before = len(rows)
            rows = [row for row in rows if int(row["extra_info"]["index"]) in wanted]
            if not rows:
                raise RuntimeError(
                    f"IF_TRAIN_INDEX_FILTER matched no train rows: {index_filter_path}"
                )
            print(
                f"[IFMultiConstraintsDataset] IF_TRAIN_INDEX_FILTER kept {len(rows)} of {before} rows "
                f"({len(wanted)} indices requested) from {index_filter_path}"
            )

        self.dataframe: datasets.Dataset = datasets.Dataset.from_list(rows)

        total = len(self.dataframe)
        print(
            f"[IFMultiConstraintsDataset] {'val' if want_val else 'train'} split: "
            f"{total} rows from {hf_name} (seed={seed}, val_size={val_size})"
        )

        # Preserve RLHFDataset's max_samples sub-sampling + long-prompt filtering behavior.
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(*((self.seed,) if self.seed is not None else ()))
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"[IFMultiConstraintsDataset] selected {self.max_samples} of {total} samples")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
