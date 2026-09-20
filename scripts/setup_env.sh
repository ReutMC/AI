#!/usr/bin/env bash
# ARION ALPHA 1 — environment setup (CPU-only torch, ML stack, base model download)
set -x
export MALLOC_ARENA_MAX=2
PIP="pip3"

echo "=== [1/3] Installing Python ML stack (CPU torch) ==="
$PIP install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
$PIP install --no-cache-dir transformers peft huggingface_hub[hf_transfer] gguf tinycss2 html5lib psutil cmake safetensors

echo "=== [2/3] Downloading base model Qwen3-0.6B (Apache-2.0) ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 /home/z/my-project/arion-alpha-1/scripts/download_base.py

echo "=== [3/3] Setup DONE ==="
df -h / | tail -1
