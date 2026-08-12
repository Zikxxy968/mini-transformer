import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer import decode_generated_text, load_model


def cuda_sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def peak_cuda_memory_mb(device: str) -> float:
    if device != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a trained Mini-Transformer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("ckpt_tinystories_300mb_1024d6l/epoch_1.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("bpe_tokenizer/tokenizer.json"))
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmark_outputs/inference_benchmark.json"))
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer))
    model = load_model(args.checkpoint, tokenizer, device)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    parameter_count = count_parameters(model)

    print("=== Mini-Transformer Checkpoint Benchmark ===")
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"parameters: {parameter_count / 1e6:.2f}M")
    print(f"prompt tokens: {len(prompt_ids)}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"runs: {args.runs}")

    for _ in range(args.warmup):
        _ = decode_generated_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    run_results = []
    last_text = ""

    for i in range(args.runs):
        cuda_sync(device)
        start = time.perf_counter()
        last_text = decode_generated_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        cuda_sync(device)
        elapsed = time.perf_counter() - start

        output_ids = tokenizer.encode(last_text, add_special_tokens=False)
        generated_tokens_estimate = max(1, len(output_ids) - len(prompt_ids))
        tokens_per_second = generated_tokens_estimate / elapsed
        run = {
            "run": i + 1,
            "time_sec": elapsed,
            "generated_tokens_estimate": generated_tokens_estimate,
            "tokens_per_second": tokens_per_second,
        }
        run_results.append(run)
        print(
            f"run {i + 1}: time={elapsed:.3f}s | "
            f"generated_tokens~={generated_tokens_estimate} | "
            f"tokens/s={tokens_per_second:.2f}"
        )

    avg_time = sum(item["time_sec"] for item in run_results) / len(run_results)
    avg_tokens_per_second = sum(item["tokens_per_second"] for item in run_results) / len(run_results)
    peak_memory = peak_cuda_memory_mb(device)

    summary = {
        "device": device,
        "checkpoint": str(args.checkpoint),
        "parameter_count": parameter_count,
        "parameter_count_millions": parameter_count / 1e6,
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "runs": run_results,
        "avg_time_sec": avg_time,
        "avg_tokens_per_second": avg_tokens_per_second,
        "peak_cuda_memory_mb": peak_memory,
        "sample_output": last_text,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"avg_time: {avg_time:.3f}s")
    print(f"avg_tokens/s: {avg_tokens_per_second:.2f}")
    print(f"peak_cuda_memory: {peak_memory:.2f} MB")
    print(f"saved_json: {args.output}")
    print("\n=== Sample Output ===")
    print(last_text)


if __name__ == "__main__":
    main()
