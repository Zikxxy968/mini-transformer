import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def encode_file(input_path: Path, tokenizer_path: Path, output_path: Path, dtype=np.int32) -> int:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<|eos|>")

    ids = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            encoded = tokenizer.encode(text).ids
            ids.extend(encoded)
            if eos_id is not None:
                ids.append(eos_id)

    if not ids:
        raise ValueError(f"No tokens were produced from {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(output_path)
    return int(arr.size)


def main():
    parser = argparse.ArgumentParser(description="Prepare a tiny token-id binary file for smoke tests.")
    parser.add_argument("--input", type=Path, default=Path("data/tiny_corpus.txt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("bpe_tokenizer/tokenizer.json"))
    parser.add_argument("--output", type=Path, default=Path("data/data.bin"))
    args = parser.parse_args()

    n_tokens = encode_file(args.input, args.tokenizer, args.output)
    print(f"Wrote {n_tokens:,} token ids to {args.output}")


if __name__ == "__main__":
    main()
