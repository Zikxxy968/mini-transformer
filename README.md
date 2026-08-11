# mini-transformer

Mini Transformer is a from-scratch educational implementation of a small GPT-style decoder-only language model. The goal is not to train a production model, but to make the complete language-model training pipeline reproducible:

```text
raw text -> BPE tokenizer -> token ids -> data.bin -> TransformerLM training -> checkpoint -> inference -> tests -> benchmark
```

The project is based on Datawhale DIY-LLM Assignment 1 and extends the original coursework with a cleaner engineering workflow: data preparation, training, inference, unit tests, benchmark scripts, and reproducible commands.

## Features

- From-scratch `Linear` and `Embedding`
- From-scratch `RMSNorm`
- RoPE rotary positional embedding
- Multi-head causal self-attention
- `SwiGLU` feed-forward network
- `TransformerBlock` and `TransformerLM`
- Custom `AdamW` optimizer
- Next-token prediction training
- Cosine learning-rate schedule with warmup
- Top-k and top-p sampling generation
- Unit tests with `pytest`
- Inference benchmark script

## Project Structure

```text
mini-transformer/
鈹溾攢鈹€ model.py                         # model architecture, RoPE, attention, SwiGLU, generation
鈹溾攢鈹€ train.py                         # training loop, dataset, optimizer, checkpoint, PPL curves
鈹溾攢鈹€ prepare_data.py                  # raw text -> token ids -> data.bin
鈹溾攢鈹€ infer.py                         # load checkpoint and generate text
鈹溾攢鈹€ requirements.txt                 # Python dependencies
鈹溾攢鈹€ README.md
鈹溾攢鈹€ bpe_tokenizer/
鈹?  鈹斺攢鈹€ tokenizer.json               # BPE tokenizer vocabulary and merge rules
鈹溾攢鈹€ data/
鈹?  鈹斺攢鈹€ tiny_corpus.txt              # small smoke-test corpus
鈹溾攢鈹€ tests/
鈹?  鈹溾攢鈹€ test_model.py                # model forward and generation tests
鈹?  鈹溾攢鈹€ test_dataset.py              # next-token dataset test
鈹?  鈹溾攢鈹€ test_train_step.py           # one-step training test
鈹?  鈹斺攢鈹€ test_checkpoint.py           # checkpoint save/load test
鈹斺攢鈹€ benchmarks/
    鈹斺攢鈹€ benchmark_inference.py       # inference performance benchmark
```

## Environment

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
torch
transformers
tokenizers
numpy
pytest
matplotlib
psutil
```

## Quick Start

### 1. Run tests

```bash
python -m pytest tests -q
```

The tests check:

- `TransformerLM` logits shape
- sampling generation sequence length
- `CausalMemmapDataset` input/target construction
- one training step with backward and optimizer update
- checkpoint save and reload

### 2. Prepare demo data

```bash
python prepare_data.py --input data/tiny_corpus.txt --output data/data.bin
```

`prepare_data.py` reads raw text, encodes it with `bpe_tokenizer/tokenizer.json`, and writes `int32` token ids to `data.bin`.

### 3. Train a small demo model

```bash
python train.py \
  --data_path data/data.bin \
  --epochs 1 \
  --batch_size 2 \
  --context_length 32 \
  --d_model 64 \
  --num_heads 4 \
  --num_layers 2 \
  --checkpoint_dir ckpt
```

Generated artifacts:

```text
ckpt/epoch_1.pt
ckpt/train_ppl.png
ckpt/val_ppl.png
```

### 4. Inference

```bash
python infer.py \
  --checkpoint ckpt/epoch_1.pt \
  --prompt "Lily likes" \
  --max_new_tokens 40
```

`infer.py` reads the model config from the checkpoint, so the model dimensions do not need to be specified manually during inference.

### 5. Benchmark

```bash
python benchmarks/benchmark_inference.py
```

The benchmark reports:

- parameter count
- forward latency
- generation tokens per second
- CPU/GPU memory usage

## Training With TinyStories

Large datasets are not stored in this Git repository. To train with TinyStories:

```bash
mkdir -p data/raw
wget -O data/raw/TinyStoriesV2-GPT4-train.txt \
https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
```

For a first stable run, use a smaller subset instead of the full file:

```bash
python - <<'PY'
from pathlib import Path

src = Path("data/raw/TinyStoriesV2-GPT4-train.txt")
dst = Path("data/raw/TinyStories_150MB.txt")
limit = 150 * 1024 * 1024

written = 0
with src.open("r", encoding="utf-8") as f, dst.open("w", encoding="utf-8") as g:
    for line in f:
        b = line.encode("utf-8")
        if written + len(b) > limit:
            break
        g.write(line)
        written += len(b)

print(f"wrote {written / 1024 / 1024:.1f} MB to {dst}")
PY
```

Encode it:

```bash
python prepare_data.py \
  --input data/raw/TinyStories_150MB.txt \
  --output data/tinystories_150mb.bin
```

Example training command:

```bash
python train.py \
  --data_path data/tinystories_150mb.bin \
  --epochs 1 \
  --batch_size 1 \
  --context_length 256 \
  --d_model 1024 \
  --num_heads 16 \
  --num_layers 8 \
  --checkpoint_dir ckpt_tinystories_204m_150mb
```

## Model Architecture

The model is a decoder-only Transformer:

```text
input_ids
  -> token embedding
  -> TransformerBlock x N
      -> RMSNorm
      -> causal self-attention + RoPE
      -> residual add
      -> RMSNorm
      -> SwiGLU FFN
      -> residual add
  -> final RMSNorm
  -> LM head
  -> logits
```

Training objective:

```text
x = [t0, t1, t2, ..., t127]
y = [t1, t2, t3, ..., t128]
```

The model predicts the next token at every position and is optimized with cross-entropy loss.

## Large Files

This repository intentionally does not track large datasets or model checkpoints. The following are ignored:

```text
data/raw/
data/*.bin
*.bin
ckpt*/
checkpoints/
*.pt
*.pth
TinyStoriesV2-GPT4-train.txt
TinyStoriesV2-GPT4-valid.txt
```

For large checkpoints, use one of:

- GitHub Releases
- Git LFS
- Hugging Face Hub

## Current Limitations

- `data/tiny_corpus.txt` is only for smoke tests.
- The training loop is written for readability and learning, not maximum throughput.
- Mixed precision, gradient accumulation, distributed training, and advanced data streaming are not implemented yet.
- Model quality depends on data size, model size, and training time.

## References

- Datawhale DIY-LLM Assignment 1
- TinyStories dataset
- GPT / decoder-only Transformer
- RoPE, RMSNorm, SwiGLU, AdamW