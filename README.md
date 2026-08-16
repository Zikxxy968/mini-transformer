# mini-transformer

A from-scratch Mini GPT / decoder-only Transformer implementation for learning language model pretraining.

This project is based on Datawhale DIY-LLM Assignment 1, but has been extended into a more complete training and evaluation pipeline.

## Features

- BPE tokenizer loading
- Decoder-only Transformer language model
- RMSNorm
- RoPE positional embedding
- Multi-head causal self-attention
- SwiGLU feed-forward network
- Custom AdamW optimizer
- Next-token prediction training
- Checkpoint saving and resuming
- Text generation with top-k / top-p sampling
- Evaluation sample generation
- Inference benchmark
- Lightweight training logs and result artifacts

## Project Structure

```text
mini-transformer/
├── model.py                         # model architecture and generation utilities
├── train.py                         # training loop, checkpointing, validation, PPL curves
├── infer.py                         # load checkpoint and generate text
├── prepare_data.py                  # generic text-to-token-bin preprocessing
├── requirements.txt                 # Python dependencies
├── README.md
├── EVAL_COMMANDS.md                 # evaluation command examples
├── bpe_tokenizer/
│   └── tokenizer.json               # BPE tokenizer vocabulary and merge rules
├── data/
│   ├── .gitkeep
│   ├── tiny_corpus.txt              # small smoke-test corpus
│   └── raw/
│       └── .gitkeep                 # raw large datasets are not tracked
├── scripts/
│   ├── prepare_tinystories_300mb.py # prepare 300MB TinyStories subset
│   ├── prepare_tinystories_1gb.py   # prepare 1GB TinyStories subset
│   ├── prepare_tinystories_2gb.py   # prepare 2GB TinyStories subset
│   ├── generate_samples.py          # generate evaluation samples
│   └── run_eval.sh                  # evaluation helper script
├── tests/
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_train_step.py
│   └── test_checkpoint.py
├── benchmarks/
│   └── benchmark_inference.py       # inference speed and memory benchmark
├── eval_outputs/
│   ├── tinystories_300mb_1024d6l_epoch1/
│   └── tinystories_2gb_1280d10l_latest/
├── benchmark_outputs/
│   ├── tinystories_300mb_1024d6l_epoch1/
│   └── tinystories_2gb_1280d10l_latest/
└── logs/
    ├── train_300mb_180m.log
    ├── train_1gb_1536d12l.log
    └── train_2gb_1280d10l_2ep_fixed.log
```

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
torch
numpy
matplotlib
tokenizers
transformers
pytest
```

Recommended GPU environment:

- CUDA-enabled GPU
- 16GB+ VRAM for small experiments
- 32GB+ VRAM recommended for larger runs

## Dataset

The main dataset used in the larger experiments is TinyStories:

```bash
mkdir -p data/raw

curl -L -C - \
  "https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" \
  -o data/raw/TinyStoriesV2-GPT4-train.txt
```

Large raw data files are ignored by Git and should be kept locally or on the training machine.

## Data Preparation

Prepare a 300MB TinyStories subset:

```bash
python scripts/prepare_tinystories_300mb.py
```

Prepare a 2GB TinyStories subset:

```bash
python scripts/prepare_tinystories_2gb.py
```

The preprocessing script reads raw text, tokenizes it with `bpe_tokenizer/tokenizer.json`, and saves token IDs as a binary `.bin` file.

Example outputs:

```text
data/raw/TinyStories_300MB.txt
data/tinystories_300mb.bin

data/raw/TinyStories_2GB.txt
data/tinystories_2gb.bin
```

These files are not uploaded to GitHub.

## Training

### 300MB Baseline Run

```bash
PYTHONUNBUFFERED=1 python train.py \
  --data_path data/tinystories_300mb.bin \
  --epochs 1 \
  --batch_size 8 \
  --context_length 512 \
  --d_model 1024 \
  --num_heads 16 \
  --num_layers 6 \
  --checkpoint_dir ckpt_tinystories_300mb_1024d6l \
  2>&1 | tee logs/train_300mb_180m.log
```

### 2GB Larger Run

```bash
PYTHONUNBUFFERED=1 python train.py \
  --data_path data/tinystories_2gb.bin \
  --epochs 2 \
  --batch_size 12 \
  --context_length 512 \
  --d_model 1280 \
  --num_heads 16 \
  --num_layers 10 \
  --save_every_steps 5000 \
  --max_val_batches 1000 \
  --checkpoint_dir ckpt_tinystories_2gb_1280d10l_2ep_fixed \
  2>&1 | tee logs/train_2gb_1280d10l_2ep_fixed.log
```

The training script saves:

- `latest.pt`
- `epoch_*.pt`
- `interrupted.pt` if interrupted
- `train_ppl.png`
- `val_ppl.png`

Checkpoints are ignored by Git because they are large.

## Inference

Generate text from a checkpoint:

```bash
python infer.py \
  --checkpoint ckpt_tinystories_2gb_1280d10l_2ep_fixed/latest.pt \
  --prompt "Once upon a time" \
  --max_new_tokens 120 \
  --temperature 0.7 \
  --top_p 0.9
```

Another example:

```bash
python infer.py \
  --checkpoint ckpt_tinystories_2gb_1280d10l_2ep_fixed/latest.pt \
  --prompt "Lily found a little cat" \
  --max_new_tokens 120 \
  --temperature 0.7 \
  --top_p 0.9
```

## Evaluation

Generate multiple fixed samples:

```bash
python scripts/generate_samples.py \
  --checkpoint ckpt_tinystories_2gb_1280d10l_2ep_fixed/latest.pt \
  --output_dir eval_outputs/tinystories_2gb_1280d10l_latest
```

Run inference benchmark:

```bash
PYTHONPATH=. python benchmarks/benchmark_inference.py \
  --checkpoint ckpt_tinystories_2gb_1280d10l_2ep_fixed/latest.pt \
  --runs 5 \
  --max_new_tokens 120 \
  --output benchmark_outputs/tinystories_2gb_1280d10l_latest/inference_benchmark.json
```

Run unit tests:

```bash
python -m pytest tests -q
```

## Experiment Results

This repository currently contains evaluation artifacts for two TinyStories training runs.

### 1. TinyStories 300MB Baseline

- Checkpoint: `ckpt_tinystories_300mb_1024d6l/epoch_1.pt`
- Model size: about 178M parameters
- Training data: 300MB TinyStories subset
- Epochs: 1
- Train PPL: about 6.67
- Val PPL: about 4.34
- Evaluation outputs:
  - `eval_outputs/tinystories_300mb_1024d6l_epoch1/`
  - `benchmark_outputs/tinystories_300mb_1024d6l_epoch1/`

### 2. TinyStories 2GB Larger Run

- Checkpoint: `ckpt_tinystories_2gb_1280d10l_2ep_fixed/latest.pt`
- Model size: 326.93M parameters
- Training data: 2GB TinyStories subset
- Epochs: close to 2 epochs
- Evaluation outputs:
  - `eval_outputs/tinystories_2gb_1280d10l_latest/`
  - `benchmark_outputs/tinystories_2gb_1280d10l_latest/`

Inference benchmark:

| Model | Params | Avg tokens/s | Avg time for 120 tokens | Peak CUDA memory |
|---|---:|---:|---:|---:|
| 300MB baseline | about 178M | about 171 tokens/s | about 0.70s | about 716 MB |
| 2GB larger run | 326.93M | 130.59 tokens/s | 0.973s | 1282.45 MB |

Compared with the 300MB baseline, the 2GB model produces more coherent TinyStories-style outputs, with better story structure and fewer abrupt breaks. The trade-off is slower inference and higher GPU memory usage.

## Large File Policy

The following files are intentionally not uploaded to GitHub:

- model checkpoints: `*.pt`, `*.pth`
- tokenized datasets: `*.bin`
- raw datasets: `data/raw/*.txt`
- large TinyStories subset files

Tracked files include:

- source code
- tokenizer JSON
- small test corpus
- training scripts
- evaluation scripts
- training logs
- generated sample outputs
- benchmark JSON files
- lightweight training curves

Before committing, check that no large files are staged:

```bash
git diff --cached --name-only | grep -E '\.pt$|\.bin$|TinyStories.*\.txt$' || echo "OK: no large model/data files staged"
```

## Notes

This project is mainly for learning how GPT-style language models work internally.

It is not an instruction-tuned model. The generated text is expected to follow the TinyStories style, but it may still contain repeated phrases, unusual names, or unstable long-range coherence. This is normal for a small Transformer trained from scratch on a limited dataset.
