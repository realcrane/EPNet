#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python epnet_cli.py predict-proxy \
  --config configs/epnet_demo.json \
  --gpu-id "${GPU_ID:-0}" \
  --checkpoint-results-dir results/demo_enet \
  --pnet-ckpt results/demo_pnet/checkpoints/pnet/pnet.weights.h5 \
  --out-dir results/demo_prediction \
  --proxy-iteration 3 \
  --elasticity-iteration 3 \
  --alpha-cutoff 0.005 \
  --drive-threshold 0.05 \
  --drive-window 5 \
  --drive-min-frames 3 \
  --rest-scale 1.0 \
  --warmup-frames 20 \
  --collision-projection-threshold 0.004 \
  --stage-feature-mode scalar \
  --stage-normalizer 3 \
  --topology-cache results/cache/topology_epnet_tshirt.npz \
  --feature-cache-dir results/feature_cache/demo
