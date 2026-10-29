#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'Qwen Lab currently requires Apple Silicon macOS for its MLX environment.' >&2
  exit 1
fi
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/'; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.11 .venv
fi
uv pip install --python .venv/bin/python -r requirements.lock.txt
printf '\nReady. Start: bash scripts/web.sh\nDownload the default MLX model: bash scripts/download.sh\n'
