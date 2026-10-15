"""Download and authenticate the exact base-model snapshots used for training.

Uses the normal Hugging Face cache (HF_HOME/HF_HUB_CACHE are respected). The
manifest pins both the repository revision and every inference file's digest.
No model weights or authentication credentials are part of this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .cache import setting_spec

MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")


def model_spec(setting: str) -> dict:
    """Read the pinned model manifest without importing HF or contacting it."""
    profile = setting_spec(setting)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    result = dict(manifest["models"][profile["base_model"]])
    result["repo_id"] = profile["base_model"]
    if result["revision"] != profile["base_model_revision"]:
        raise ValueError("Model and cache manifests disagree on the base-model revision")
    return result


def _file_digest(path: Path, record: dict) -> str:
    if "sha256" in record:
        digest = hashlib.sha256()
    elif "git_blob_sha1" in record:
        digest = hashlib.sha1()
        digest.update(f"blob {record['size_bytes']}\0".encode("ascii"))
    else:
        raise ValueError("Model file has no pinned digest")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_path(path: str | Path, setting: str) -> str:
    """Return the canonical model ID only after offline verification succeeds.

    A directory merely named after a revision is not trusted: its repository
    location, each file's size, every weight shard, configuration and tokenizer
    digest must match. Symlinks into the standard HF blobs cache are supported.
    """
    spec = model_spec(setting)
    snapshot = Path(path).expanduser().resolve(strict=True)
    repo_directory = "models--" + spec["repo_id"].replace("/", "--")
    if (not snapshot.is_dir() or snapshot.name != spec["revision"] or
            snapshot.parent.name != "snapshots" or snapshot.parent.parent.name != repo_directory):
        raise ValueError("Model path is not the pinned repository's Hugging Face snapshot")
    allowed = {item["filename"] for item in spec["files"]}
    # Alternate tokenizer/configuration or unindexed safetensors can change what
    # from_pretrained loads, even when the expected files are all present.
    for candidate in snapshot.iterdir():
        if candidate.is_file() and candidate.suffix in {".json", ".safetensors", ".model", ".jinja"}:
            if candidate.name not in allowed:
                raise ValueError(f"Unpinned inference file in model snapshot: {candidate.name}")
    for record in spec["files"]:
        file = snapshot / record["filename"]
        if not file.is_file() or file.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Model file missing or wrong size: {record['filename']}")
        expected = record.get("sha256", record.get("git_blob_sha1"))
        if _file_digest(file, record) != expected:
            raise ValueError(f"Model file checksum mismatch: {record['filename']}")
    return spec["repo_id"]


def prepare_model(setting: str) -> Path:
    """Download pinned inference files via HF CLI and verify their local bytes."""
    spec = model_spec(setting)
    hf = shutil.which("hf")
    if hf is None:
        raise RuntimeError("Install huggingface_hub[cli] so the 'hf' command is available")
    subprocess.run([hf, "download", spec["repo_id"],
                    *[record["filename"] for record in spec["files"]],
                    "--revision", spec["revision"]], check=True)
    # Resolves HF_HOME/HF_HUB_CACHE without parsing a human-readable CLI message;
    # local_files_only ensures this resolution is not an extra network request.
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(spec["repo_id"], revision=spec["revision"],
                                     allow_patterns=[item["filename"] for item in spec["files"]],
                                     local_files_only=True)).resolve()
    validate_model_path(snapshot, setting)
    return snapshot


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setting")
    args = parser.parse_args(argv)
    print(prepare_model(args.setting))


if __name__ == "__main__":
    main()
