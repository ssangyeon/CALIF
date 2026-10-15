#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec "${PYTHON:-python}" -m calif_training.launch --setting qwen3_4b_think --method constraint_only "$@"
