#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
# Do not load the training config: its train:true would restart training.
exec .venv/bin/python -m mlx_lm lora \
  --model models/qwen3.5-4b/weights --data data/example \
  --test --test-batches 2 --batch-size 1 --max-seq-length 256 \
  --mask-prompt --adapter-path "" "$@"
