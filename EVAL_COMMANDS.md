# Evaluation and Benchmark Commands

Run unit tests:

```bash
python -m pytest tests -q
```

Generate text samples from the trained 300MB checkpoint:

```bash
python scripts/generate_samples.py \
  --checkpoint ckpt_tinystories_300mb_1024d6l/epoch_1.pt
```

Run real checkpoint inference benchmark:

```bash
python benchmarks/benchmark_inference.py \
  --checkpoint ckpt_tinystories_300mb_1024d6l/epoch_1.pt \
  --runs 5 \
  --max_new_tokens 120
```

Or run all evaluation steps:

```bash
bash scripts/run_eval.sh
```

Expected artifacts:

```text
eval_outputs/sample_1.txt
eval_outputs/sample_2.txt
eval_outputs/sample_3.txt
eval_outputs/samples.json
benchmark_outputs/inference_benchmark.json
```

The checkpoint file `epoch_1.pt` is intentionally not tracked by Git because it is about 2GB.
