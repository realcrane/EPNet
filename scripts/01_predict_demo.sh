#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python epnet_cli.py predict-proxy \
  --config configs/epnet_demo.json \
  --gpu-id "${GPU_ID:-0}" \
  --checkpoint-results-dir results/demo_enet \
  --pnet-ckpt results/demo_pnet/checkpoints/pnet/pnet.weights.h5 \
  --out-dir results/demo_prediction
