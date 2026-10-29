"""Single-turn text inference; optionally load a trained LoRA adapter."""
import argparse
import common


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default="请用中文简要介绍你自己。")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--adapter", help="LoRA 适配器目录")
    p.add_argument("--thinking", action="store_true")
    args = p.parse_args()
    if not 1 <= args.max_tokens <= 1024:
        p.error("max-tokens 必须在 1–1024 之间")
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = common.load_model(args.adapter)
    tokens = common.encode_prompt(tokenizer, args.prompt, args.thinking)
    generate(model, tokenizer, prompt=tokens, max_tokens=args.max_tokens,
             sampler=make_sampler(temp=0), verbose=True)


if __name__ == "__main__":
    main()
