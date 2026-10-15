#!/usr/bin/env bash
# Run from a Linux NVIDIA CUDA environment with a compatible driver.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
command -v uv >/dev/null || { echo 'Install uv: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-calif.txt
uv pip install --python .venv/bin/python --no-build-isolation 'flash-attn==2.8.3'
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m nltk.downloader punkt punkt_tab
echo 'Activate with: source .venv/bin/activate'
