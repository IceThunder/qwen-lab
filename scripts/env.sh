#!/usr/bin/env bash
# Source this file from a project shell script.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.runtime/python"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export PYTHONUNBUFFERED=1
export TMPDIR="$PROJECT_ROOT/work/tmp"
mkdir -p "$TMPDIR" "$PROJECT_ROOT/models" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/adapters" "$PROJECT_ROOT/outputs"
cd "$PROJECT_ROOT"
