"""Compare baseline with reversible MLP post-SwiGLU channel scaling."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import common
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler


class ScaledInput(nn.Module):
    """Wrap down_proj: change its input, i.e. true MLP intermediate channels."""
    def __init__(self, original, width, neurons, scale):
        super().__init__()
        self.original = original
        mask = [1.0] * width
        for index in neurons:
            mask[index] = scale
        self.mask = mx.array(mask)

    def __call__(self, x):
        return self.original(x * self.mask.astype(x.dtype))


@contextmanager
def intervention(model, layer, neurons, scale):
    mlp = model.layers[layer].mlp
    original = mlp.down_proj
    width = model.args.text_config["intermediate_size"]
    mlp.down_proj = ScaledInput(original, width, neurons, scale)
    try:
        yield
    finally:
        mlp.down_proj = original


def next_logits(model, tokens):
    # No persistent KV/recurrent cache: each condition starts from identical input.
    logits = model(mx.array([tokens]))[0, -1].astype(mx.float32)
    mx.eval(logits)
    return logits


def top_tokens(logits, tokenizer, k=5):
    logp = logits - mx.logsumexp(logits)
    ids = mx.argsort(logits)[-k:][::-1].tolist()
    return [{"id": i, "text": tokenizer.decode([i]),
             "probability": float(mx.exp(logp[i]).item())} for i in ids]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default="法国的首都是哪里？请简短回答。")
    p.add_argument("--layer", type=int, default=16, help="从 0 开始")
    p.add_argument("--neurons", default="0,1,2,3", help="逗号分隔的中间通道编号")
    p.add_argument("--scales", default="0,2", help="如 0,2；1 为不改变激活的对照")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--output", default="outputs/qwen3.5-4b/intervention.json")
    args = p.parse_args()
    try:
        neurons = sorted(set(int(x) for x in args.neurons.split(",")))
        scales = [float(x) for x in args.scales.split(",")]
    except ValueError:
        p.error("neurons 应为整数列表，scales 应为数值列表")
    if not scales or any(not math.isfinite(s) or abs(s) > 10 for s in scales):
        p.error("scale 必须为 -10 到 10 之间的有限数值")
    if not 1 <= args.max_tokens <= 512:
        p.error("max-tokens 必须在 1–512 之间")
    # Validate against config before allocating GPU memory.
    cfg = json.loads((common.MODEL / "config.json").read_text())["text_config"]
    if not 0 <= args.layer < cfg["num_hidden_layers"]:
        p.error(f"layer 范围：0–{cfg['num_hidden_layers'] - 1}")
    if not neurons or any(i < 0 or i >= cfg["intermediate_size"] for i in neurons):
        p.error(f"neuron 范围：0–{cfg['intermediate_size'] - 1}")
    model, tokenizer = common.load_model()
    tokens = common.encode_prompt(tokenizer, args.prompt)
    def reply():
        return generate(model, tokenizer, prompt=tokens, max_tokens=args.max_tokens,
                        sampler=make_sampler(temp=0), verbose=False)
    baseline = next_logits(model, tokens)
    base_text = reply()
    print(f"原始：{base_text}", flush=True)
    report = {"prompt": args.prompt, "layer": args.layer, "neurons": neurons,
        "position": "post-SwiGLU, before down_proj; all prompt/generated tokens",
        "model_source": json.loads((common.ROOT / "models/qwen3.5-4b/source.json").read_text()),
        "baseline": {"text": base_text, "top_next_tokens": top_tokens(baseline, tokenizer)},
        "conditions": []}
    for scale in scales:
        with intervention(model, args.layer, neurons, scale):
            logits = next_logits(model, tokens)
            text = reply()
            diff = logits - baseline
            row = {"scale": scale, "text": text,
                "max_abs_logit_delta": float(mx.max(mx.abs(diff)).item()),
                "rms_logit_delta": float(mx.sqrt(mx.mean(diff * diff)).item()),
                "top_next_tokens": top_tokens(logits, tokenizer)}
            report["conditions"].append(row)
            print(f"scale={scale}: {text}\n最大 logit 变化={row['max_abs_logit_delta']:.6g}", flush=True)
    restored = next_logits(model, tokens)
    report["restored_max_abs_logit_delta"] = float(mx.max(mx.abs(restored - baseline)).item())
    report["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"报告：{out.resolve()}")


if __name__ == "__main__":
    main()
