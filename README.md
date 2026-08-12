# mini-transformer

A hand-written Mini GPT / decoder-only Transformer language model for learning and experimentation.

This repository implements a compact language-modeling pipeline from tokenization and data preparation to model training, checkpoint loading, text generation, unit tests, and inference benchmarking. The project is based on the Datawhale DIY-LLM Assignment 1 learning task, with extra training scripts, evaluation scripts, benchmark outputs, and lightweight experiment artifacts.

## Current Status

The current trained checkpoint is not tracked by Git because it is about 2GB. It was trained separately on AutoDL and can be placed at:

```text
ckpt_tinystories_300mb_1024d6l/epoch_1.pt
```

Current checkpoint summary:

```text
Dataset subset: TinyStories, about 300MB raw text
Training epoch: 1
Model size: 178.83M parameters
Checkpoint size: about 2.0GB
Benchmark device: CUDA GPU
Average inference speed: about 171 tokens/s
Peak CUDA memory during benchmark: about 716 MB
```

The model can generate short TinyStories-style English text. The output is readable and usually follows the prompt, but it still contains repeated phrases, unusual words, and occasional entity drift because it was trained from scratch on a small subset for only one epoch.

## Features

- Custom `Linear` layer and embedding layer
- `RMSNorm`
- RoPE rotary positional embedding
- Multi-head causal self-attention
- SwiGLU feed-forward network
- Pre-norm Transformer block
- Decoder-only `TransformerLM`
- Custom `AdamW` optimizer
- Next-token prediction training loop
- Cosine learning-rate schedule with warmup
- Checkpoint save/load
- Top-k / top-p sampling inference
- Unit tests with `pytest`
- Real checkpoint text-sample generation
- Real checkpoint inference benchmark

## Project Structure

```text
mini-transformer/
|-- model.py                              # model architecture: RoPE, attention, SwiGLU, TransformerLM
|-- train.py                              # training loop, dataset, optimizer, checkpoint, PPL curves
|-- prepare_data.py                       # raw text -> token ids -> data.bin
|-- infer.py                              # load checkpoint and generate text
|-- requirements.txt                      # Python dependencies
|-- README.md                             # project documentation
|-- EVAL_COMMANDS.md                      # evaluation and benchmark commands
|-- bpe_tokenizer/
|   `-- tokenizer.json                    # BPE tokenizer vocabulary and merge rules
|-- data/
|   |-- .gitkeep
|   |-- tiny_corpus.txt                   # small smoke-test corpus
|   `-- raw/
|       `-- .gitkeep                      # raw datasets are ignored by Git
|-- ckpt_tinystories_300mb_1024d6l/
|   |-- .gitkeep
|   |-- train_ppl.png                     # lightweight training curve
|   |-- val_ppl.png                       # lightweight validation curve
|   `-- epoch_1.pt                        # external checkpoint, not tracked by Git
|-- logs/
|   |-- .gitkeep
|   |-- train_300mb_180m.log              # 300MB training log
|   `-- train_1gb_1536d12l.log            # 1GB interrupted training log
|-- scripts/
|   |-- prepare_tinystories_300mb.py      # build 300MB TinyStories token bin
|   |-- prepare_tinystories_1gb.py        # build 1GB TinyStories token bin
|   |-- generate_samples.py               # generate sample outputs from checkpoint
|   `-- run_eval.sh                       # run tests, samples, benchmark
|-- tests/
|   |-- test_model.py                     # model forward and generation-related tests
|   |-- test_dataset.py                   # causal dataset test
|   |-- test_train_step.py                # one-step training/backward test
|   `-- test_checkpoint.py                # checkpoint save/load test
|-- benchmarks/
|   `-- benchmark_inference.py            # real checkpoint inference benchmark
|-- eval_outputs/                         # generated text samples, optional tracked small artifacts
|-- benchmark_outputs/                    # benchmark JSON result, optional tracked small artifacts
`-- test_outputs/                         # pytest output text, optional tracked small artifact
```

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
torch
transformers
numpy
matplotlib
pytest
```

Check CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## Data Preparation

Raw and tokenized datasets are not tracked by Git. Large files should be placed under `data/raw/` or `data/*.bin` locally.

Example: prepare a 300MB TinyStories subset:

```bash
python scripts/prepare_tinystories_300mb.py
```

Example: prepare a 1GB TinyStories subset:

```bash
python scripts/prepare_tinystories_1gb.py
```

Expected outputs:

```text
data/raw/TinyStories_300MB.txt
data/tinystories_300mb.bin

data/raw/TinyStories_1GB.txt
data/tinystories_1gb.bin
```

These generated files are ignored by `.gitignore`.

## Training

Example training command for the 300MB / 178M-parameter run:

```bash
mkdir -p logs
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

The resulting checkpoint is large and is intentionally ignored by Git:

```text
ckpt_tinystories_300mb_1024d6l/epoch_1.pt
```

Training curves are lightweight and can be tracked:

```text
ckpt_tinystories_300mb_1024d6l/train_ppl.png
ckpt_tinystories_300mb_1024d6l/val_ppl.png
```

## Inference

Run text generation from a trained checkpoint:

```bash
python infer.py \
  --checkpoint ckpt_tinystories_300mb_1024d6l/epoch_1.pt \
  --prompt "Once upon a time" \
  --max_new_tokens 120 \
  --temperature 0.7 \
  --top_p 0.9
```

Another prompt:

```bash
python infer.py \
  --checkpoint ckpt_tinystories_300mb_1024d6l/epoch_1.pt \
  --prompt "Lily found a little cat" \
  --max_new_tokens 120 \
  --temperature 0.7 \
  --top_p 0.9
```

## Tests

Run unit tests:

```bash
python -m pytest tests -q
```

Current result:

```text
5 passed, 1 warning
```

The warning comes from PyTorch `torch.load(weights_only=False)` future behavior and does not affect current correctness.

## Evaluation Samples

Generate multiple sample outputs from the checkpoint:

```bash
python scripts/generate_samples.py \
  --checkpoint ckpt_tinystories_300mb_1024d6l/epoch_1.pt
```

Expected outputs:

```text
eval_outputs/sample_1.txt
eval_outputs/sample_2.txt
eval_outputs/sample_3.txt
eval_outputs/samples.json
```

These files are small and can be used as qualitative evidence that the checkpoint can generate text.

## Inference Benchmark

Run real-checkpoint benchmark:

```bash
python benchmarks/benchmark_inference.py \
  --checkpoint ckpt_tinystories_300mb_1024d6l/epoch_1.pt \
  --runs 5 \
  --max_new_tokens 120 \
  --output benchmark_outputs/inference_benchmark.json
```

Current benchmark result:

```text
Device: CUDA
Parameters: 178.83M
Prompt: Once upon a time
Max new tokens: 120
Average time: about 0.70 seconds
Average speed: about 171 tokens/s
Peak CUDA memory: about 716 MB
```

Benchmark output file:

```text
benchmark_outputs/inference_benchmark.json
```

## Run All Evaluation Steps

```bash
bash scripts/run_eval.sh
```

This runs:

```text
1. pytest unit tests
2. sample generation
3. checkpoint inference benchmark
```

## Git Tracking Policy

Tracked:

```text
source code
unit tests
benchmark scripts
sample-generation scripts
README / command docs
small logs
small PNG training curves
small JSON/text evaluation outputs
```

Ignored:

```text
*.pt / *.pth checkpoint files
*.bin tokenized datasets
raw TinyStories text files
cache directories
large generated datasets
```

Reason: GitHub is not suitable for multi-GB model weights or datasets. Keep those files on the training machine, local disk, cloud storage, or a model/dataset hosting service.

## Interpretation of Current Model Quality

The current 300MB checkpoint is useful as an engineering proof of a complete Mini GPT training pipeline. It demonstrates that the model can be trained, saved, loaded, benchmarked, and used for generation.

However, it is not a high-quality language model yet. The model was trained from scratch on a limited subset for one epoch, so generated stories may include:

```text
unusual words
repeated phrases
weak long-range consistency
entity/name drift
```

This is expected for the current training scale. Improving quality would require more data, more training steps, better checkpointing, and more systematic evaluation.
