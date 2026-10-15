"""Download and verify the five published CALIF training caches.

No GPU or model dependency is needed here. The original cache bytes are kept
unchanged. Runtime compatibility permits cached-response generation settings to differ
from policy-training settings, but never changes dataset, mode, or model identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "calif"

MANIFEST_PATH = Path(__file__).with_name("cache_manifest.json")


def _manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def setting_spec(setting: str) -> dict:
    """Return a fresh manifest record; safe to call offline for IF-RLVR."""
    manifest = _manifest()
    if setting not in manifest["settings"]:
        raise ValueError(f"Unknown CALIF setting {setting!r}; choose {list(manifest['settings'])}")
    result = dict(manifest["settings"][setting])
    result.update(repo_id=manifest["repo_id"], revision=manifest["revision"],
                  dataset_id=manifest["dataset_id"], dataset_revision=manifest["dataset_revision"])
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete(item: object, suffixes=("0", "1")) -> bool:
    if not isinstance(item, dict):
        return False
    for suffix in suffixes:
        tokens = item.get("y" + suffix)
        if not isinstance(tokens, list) or not tokens:
            return False
        if not all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in tokens):
            return False
        count = item.get(f"ref{suffix}_token_count")
        nll = item.get(f"ref{suffix}_nll")
        ppl = item.get(f"ref{suffix}_ppl")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            return False
        if not isinstance(nll, (float, int)) or not math.isfinite(nll) or nll < 0:
            return False
        if not isinstance(ppl, (float, int)) or not math.isfinite(ppl) or ppl <= 0:
            return False
    return True


def _read_verified(path: Path, spec: dict) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing training cache: {path}")
    if path.stat().st_size != spec["size_bytes"] or _sha256(path) != spec["sha256"]:
        raise ValueError(f"Training cache checksum mismatch: {path}. Do not train with modified cache bytes.")
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("Training cache metadata must be an object")
    items = payload.get("items")
    if not isinstance(items, dict) or len(items) != spec["entry_count"]:
        raise ValueError("Incorrect training cache item count")
    incomplete = set()
    for key, item in items.items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ValueError(f"Malformed training cache index: {key!r}")
        if not _complete(item):
            incomplete.add(int(key))
    if incomplete:
        raise ValueError(f"Unexpected incomplete cache rows: {sorted(incomplete)[:10]}")
    return payload


def prepare_cache(setting: str, cache_dir: str | Path) -> Path:
    """Fetch the pinned file with ``hf download`` and verify before returning it.

    Authentication is inherited from HF_TOKEN or ``hf auth login``. Credentials
    are never command arguments or printed. Existing valid files are reused.
    """
    spec = setting_spec(setting)
    destination = Path(cache_dir).expanduser().resolve() / spec["revision"]
    path = destination / spec["filename"]
    if not path.exists():
        hf = shutil.which("hf")
        if hf is None:
            raise RuntimeError("Install huggingface_hub[cli] so the 'hf' command is available")
        destination.mkdir(parents=True, exist_ok=True)
        help_text = subprocess.run([hf, "download", "--help"], check=True,
                                   capture_output=True, text=True).stdout
        repo_type_option = "--repo-type" if "--repo-type" in help_text else "--type"
        subprocess.run([hf, "download", spec["repo_id"], spec["filename"], repo_type_option, "dataset",
                        "--revision", spec["revision"], "--local-dir", str(destination)], check=True)
    _read_verified(path, spec)
    return path


def validate_for_training(path: str | Path, setting: str, expected_metadata: dict,
                          required_indices: Iterable[int]) -> dict:
    """Verify identity and active rows without changing published cache contents.

    ``required_indices`` must come from the actual dataset after the declared
    tokenizer-based prompt-length filtering, not from the
    cache itself. Counting rows is insufficient because equal-size sets can differ.
    """
    spec = setting_spec(setting)
    payload = _read_verified(Path(path), spec)
    source = payload["metadata"]
    runtime_model = expected_metadata.get("model_path")
    from .model import validate_model_path
    # A floating Hub ID or arbitrary local directory is never trusted by the
    # cache validation: canonical identity is derived only from verified bytes.
    try:
        canonical_model = validate_model_path(runtime_model, setting)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Cache/training mismatch for model_path: {exc}") from exc
    expected_metadata = dict(expected_metadata, model_path=canonical_model)
    invariants = {
        "model_path": spec["base_model"],
        "tokenizer_class": source["tokenizer_class"],
        "if_dataset_seed": source["if_dataset_seed"],
        "if_dataset_val_size": source["if_dataset_val_size"],
        "train_files": source["train_files"],
        "max_prompt_length": source["max_prompt_length"],
        "apply_chat_template_kwargs": source["apply_chat_template_kwargs"],
        "ppl_prefix_mode": "standard",
        "ppl_nll_scope": "final_answer_tokens_only",
        "if_dataset_revision": spec["dataset_revision"],
    }
    for field, expected in invariants.items():
        if expected_metadata.get(field) != expected:
            raise ValueError(f"Cache/training mismatch for {field}: expected {expected!r}, "
                             f"got {expected_metadata.get(field)!r}")
    if expected_metadata.get("if_dataset_hf") not in ("", spec["dataset_id"]):
        raise ValueError("Training cache belongs to a different training dataset")
    required = list(required_indices)
    if not required or len(required) != len(set(required)):
        raise ValueError("Training indices must be nonempty and unique")
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in required):
        raise ValueError("Training indices must be nonnegative integers")
    missing = [index for index in required if not _complete(payload["items"].get(str(index)))]
    if missing:
        raise ValueError(f"Incomplete CALIF cache coverage: {len(missing)}/{len(required)} rows missing; "
                         f"first indices={missing[:10]}. Refusing to start training.")
    declared_count = expected_metadata.get("train_sample_count")
    if declared_count != len(required):
        raise ValueError("Dataset train_sample_count and active index count disagree")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setting", choices=list(_manifest()["settings"]))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)
    print(prepare_cache(args.setting, args.cache_dir))


if __name__ == "__main__":
    main()
