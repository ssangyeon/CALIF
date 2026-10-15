#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec "${PYTHON:-python}" -m calif_training.launch --setting tulu3_8b_dpo --method constraint_only "$@"
