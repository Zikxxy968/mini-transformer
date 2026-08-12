import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer import decode_generated_text, load_model


DEFAULT_PROMPTS = [
    "Once upon a time",
    "Lily found a little cat",
    "Tom wanted to help his friend",
]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample outputs from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("ckpt_tinystories_300mb_1024d6l/epoch_1.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("bpe_tokenizer/tokenizer.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("eval_outputs"))
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer))
    model = load_model(args.checkpoint, tokenizer, device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for i, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        text = decode_generated_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        sample_path = args.output_dir / f"sample_{i}.txt"
        sample_path.write_text(text, encoding="utf-8")
        records.append({"prompt": prompt, "output_file": str(sample_path), "text": text})
        print(f"[{i}] prompt: {prompt}")
        print(text)
        print()

    metadata = {
        "checkpoint": str(args.checkpoint),
        "device": device,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "samples": records,
    }
    (args.output_dir / "samples.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved: {args.output_dir}")


if __name__ == "__main__":
    main()
