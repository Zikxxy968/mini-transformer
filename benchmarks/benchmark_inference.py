import argparse
import time

import psutil
import torch

from model import TransformerLM


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Benchmark mini-transformer forward and generation speed.")
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--context_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--new_tokens", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.context_length,
    ).to(device)
    model.eval()
    idx = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    for _ in range(3):
        _ = model(idx)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    _ = model(idx)
    if device == "cuda":
        torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - start) * 1000

    prompt = idx[:, : min(8, args.context_length)]
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    _ = model.generate(prompt, max_new_tokens=args.new_tokens, top_k=20)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    rss_mb = psutil.Process().memory_info().rss / 1024**2
    print(f"device: {device}")
    print(f"parameters: {count_parameters(model) / 1e6:.2f}M")
    print(f"forward_latency_ms: {forward_ms:.2f}")
    print(f"generation_tokens_per_second: {args.new_tokens / max(elapsed, 1e-9):.2f}")
    print(f"process_memory_mb: {rss_mb:.2f}")
    if device == "cuda":
        print(f"gpu_memory_allocated_mb: {torch.cuda.memory_allocated() / 1024**2:.2f}")


if __name__ == "__main__":
    main()
