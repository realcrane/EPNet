$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

python epnet_cli.py predict-proxy `
  --config configs/epnet_demo.json `
  --gpu-id 0 `
  --checkpoint-results-dir results/demo_enet `
  --pnet-ckpt results/demo_pnet/checkpoints/pnet/pnet.weights.h5 `
  --out-dir results/demo_prediction
