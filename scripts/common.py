"""Shared local paths and conservative inference defaults."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
MODEL = ROOT / "models/qwen3.5-4b/weights"


def load_model(adapter=None):
    from mlx_lm import load
    if not (MODEL / "config.json").is_file() or not list(MODEL.glob("*.safetensors")):
        raise SystemExit("模型尚未下载完成，请先运行 bash scripts/download.sh")
    if adapter and not (Path(adapter) / "adapters.safetensors").exists():
        raise SystemExit(f"找不到适配器：{adapter}/adapters.safetensors")
    return load(str(MODEL), adapter_path=adapter)


def encode_prompt(tokenizer, prompt, thinking=False):
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=thinking,
    )
    if len(tokens) > 2048:
        raise ValueError("本实验默认限制输入为 2048 tokens，请缩短输入。")
    return tokens
