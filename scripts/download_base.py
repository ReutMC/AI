#!/usr/bin/env python3
"""ARION ALPHA 1 — download base model Qwen3-0.6B (Apache-2.0)."""
import os, sys

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen3-0.6B"
DEST = "/home/z/my-project/arion-alpha-1/model/base"

print(f"downloading {REPO} -> {DEST}", flush=True)
p = snapshot_download(
    REPO,
    local_dir=DEST,
    allow_patterns=[
        "config.json", "generation_config.json", "model.safetensors",
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "LICENSE", "README.md",
    ],
)
print("done:", p, flush=True)
for f in sorted(os.listdir(p)):
    fp = os.path.join(p, f)
    print(f"  {f}: {os.path.getsize(fp)/1e6:.1f} MB", flush=True)
