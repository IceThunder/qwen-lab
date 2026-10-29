#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec .venv/bin/python -m mlx_lm server \
  --model "$PROJECT_ROOT/models/qwen3.5-4b/weights" \
  --host 127.0.0.1 --port 8080 --temp 0 --max-tokens 256 \
  --decode-concurrency 1 --prompt-concurrency 1 \
  --prefill-step-size 256 --prompt-cache-size 1 \
  --chat-template-args '{"enable_thinking":false}' "$@"
