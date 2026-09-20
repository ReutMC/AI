#!/usr/bin/env python3
"""ARION ALPHA 1 — merge LoRA adapter into the base model.

Produces model/arion-alpha-1-merged/ (full fine-tuned weights, bf16),
ready for GGUF conversion via llama.cpp.
"""
import os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "model", "base")
ADAPTER = os.path.join(ROOT, "model", "arion-alpha-1-lora")
OUT = os.path.join(ROOT, "model", "arion-alpha-1-merged")

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("loading base:", BASE, flush=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    print("attaching adapter:", ADAPTER, flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER)
    print("merging weights ...", flush=True)
    model = model.merge_and_unload()
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(ADAPTER if os.path.exists(
        os.path.join(ADAPTER, "tokenizer_config.json")) else BASE)
    tok.save_pretrained(OUT)
    n = sum(p.numel() for p in model.parameters())
    json.dump({"merged_from": BASE, "adapter": ADAPTER, "params": n},
              open(os.path.join(OUT, "merge_info.json"), "w"), indent=2)
    print(f"merged model saved → {OUT} ({n/1e9:.3f}B params)", flush=True)

if __name__ == "__main__":
    main()
