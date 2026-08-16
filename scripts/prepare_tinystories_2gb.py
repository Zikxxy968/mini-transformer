from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "TinyStoriesV2-GPT4-train.txt"
SUBSET_FILE = ROOT / "data" / "raw" / "TinyStories_2GB.txt"
OUTPUT_BIN = ROOT / "data" / "tinystories_2gb.bin"
TOKENIZER_FILE = ROOT / "bpe_tokenizer" / "tokenizer.json"

LIMIT_BYTES = 2 * 1024 * 1024 * 1024


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    eos_id = tokenizer.token_to_id("<|eos|>")

    SUBSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_BIN.parent.mkdir(parents=True, exist_ok=True)

    written_bytes = 0
    total_tokens = 0
    total_lines = 0

    with RAW_FILE.open("r", encoding="utf-8") as src, \
         SUBSET_FILE.open("w", encoding="utf-8") as subset, \
         OUTPUT_BIN.open("wb") as out:

        for line in src:
            encoded_bytes = line.encode("utf-8")

            if written_bytes + len(encoded_bytes) > LIMIT_BYTES:
                break

            subset.write(line)
            written_bytes += len(encoded_bytes)

            text = line.strip()
            if not text:
                continue

            ids = tokenizer.encode(text).ids
            if eos_id is not None:
                ids.append(eos_id)

            arr = np.asarray(ids, dtype=np.int32)
            arr.tofile(out)

            total_tokens += len(ids)
            total_lines += 1

            if total_lines % 100000 == 0:
                print(
                    f"processed {total_lines:,} lines | "
                    f"{written_bytes / 1024 / 1024:.1f} MB | "
                    f"{total_tokens:,} tokens"
                )

    print("Done.")
    print(f"subset text: {SUBSET_FILE}")
    print(f"token bin:   {OUTPUT_BIN}")
    print(f"text size:   {written_bytes / 1024 / 1024:.1f} MB")
    print(f"tokens:      {total_tokens:,}")


if __name__ == "__main__":
    main()