#!/usr/bin/env python3
"""ARION ALPHA 1 — merge LoRA adapter into the base model.

Produces model/arion-alpha-1-merged/ (full fine-tuned weights, bf16),
ready for GGUF conversion via llama.cpp.
"""
import os, sys, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "model", "base")
ADAPTER = os.path.join(ROOT, "model", "arion-alpha-1-lora")
OUT = os.path.join(ROOT, "model", "arion-alpha-1-merged")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--lora", default=ADAPTER)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    base, adapter, out = args.base, args.lora, args.out

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("loading base:", base, flush=True)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    print("attaching adapter:", adapter, flush=True)
    model = PeftModel.from_pretrained(model, adapter)
    print("merging weights ...", flush=True)
    model = model.merge_and_unload()
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(adapter if os.path.exists(
        os.path.join(adapter, "tokenizer_config.json")) else base)
    tok.save_pretrained(out)
    n = sum(p.numel() for p in model.parameters())
    json.dump({"merged_from": base, "adapter": adapter, "params": n},
              open(os.path.join(out, "merge_info.json"), "w"), indent=2)
    print(f"merged model saved → {out} ({n/1e9:.3f}B params)", flush=True)

if __name__ == "__main__":
    main()
