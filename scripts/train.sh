#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec .venv/bin/python -m mlx_lm lora --config configs/qwen3.5-4b/lora.yaml "$@"
