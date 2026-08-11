import argparse
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from model import TransformerLM, decode_generated_text


def load_model(checkpoint_path: Path, tokenizer, device: str) -> TransformerLM:
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["context_length"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Run text generation from a trained mini-transformer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("ckpt/epoch_1.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("bpe_tokenizer/tokenizer.json"))
    parser.add_argument("--prompt", type=str, default="Lily likes")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer))
    model = load_model(args.checkpoint, tokenizer, device)

    text = decode_generated_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
