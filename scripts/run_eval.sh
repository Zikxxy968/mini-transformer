#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pytest tests -q
python scripts/generate_samples.py
python benchmarks/benchmark_inference.py --runs 5 --max_new_tokens 120
