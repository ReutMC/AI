#!/usr/bin/env python3
"""KAGGLE KERNEL A — ARION Persian phase: env report + seq-len benchmark +
BASELINE Persian benchmark (base Qwen3-0.6B, BEFORE training) + tiny smoke
train → merge → GGUF → llama-cli smoke. Saves everything to /kaggle/working.

Attached dataset: reutmc/arion-persian-data → /kaggle/input/arion-persian-data
"""
import json, os, subprocess, sys, time, glob

W = "/kaggle/working"
IN = "/kaggle/input"
DATA = IN + "/arion-persian-data"  # flattened bundle (see kaggle_pipeline.py bundle)
os.makedirs(W, exist_ok=True)
os.makedirs(f"{W}/baseline", exist_ok=True)
os.chdir(W)

def sh(cmd, timeout=None, check=True):
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    print(r.stdout[-4000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-4000:], flush=True)
        if check:
            raise RuntimeError(f"command failed: {cmd}")
    print(f"  (exit={r.returncode}, {round(time.time()-t0,1)}s)", flush=True)
    return r

# ---------------- 0. env ----------------
# GPU diagnostics FIRST (nvidia-smi tells us if the MACHINE has a GPU at all)
sh("nvidia-smi || echo NO-NVIDIA-SMI", check=False, timeout=60)
import torch
print("GPU:", torch.cuda.get_device_properties(0) if torch.cuda.is_available() else "NONE",
      flush=True)
assert torch.cuda.is_available(), "FATAL: no GPU — enable GPU accelerator for this kernel"
print("peft available check...", flush=True)
try:
    import peft  # noqa
    print("peft OK", flush=True)
except ImportError:
    sh("pip install -q peft", timeout=300)

# ---------------- 1. bundle paths (flattened layout) ----------------
train_jsonl = f"{DATA}/persian_train.jsonl"
val_jsonl = f"{DATA}/persian_validation.jsonl"
bench_jsonl = f"{DATA}/persian_fluency_benchmark.jsonl"
for p in (train_jsonl, bench_jsonl):
    assert os.path.exists(p), f"missing bundle file: {p}"
print("bundle OK:", flush=True)
sh(f"ls -la {DATA}/datasets/processed {DATA}/evaluation", check=False)

# ---------------- 2. seq-length benchmark (2048/3072/4096) ----------------
cfg = {
    "base_model_path": "Qwen/Qwen3-0.6B",
    "train_path": train_jsonl,
    "val_path": val_jsonl,
    "output_dir": f"{W}/smoke_lora",
    "max_seq_len": 3072,
    "batch_size": 4,
    "gradient_accumulation": 8,
    "dtype": "auto",
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "learning_rate": 1e-4, "weight_decay": 0.01, "warmup_ratio": 0.05,
    "max_epochs": 1, "eval_every_steps": 25, "log_every_steps": 5,
    "seed": 42, "gradient_checkpointing": True,
}
json.dump(cfg, open(f"{W}/config_smoke.json", "w"), indent=2)
sh(f"python3 {DATA}/training/train_kaggle.py --config {W}/config_smoke.json "
   f"--benchmark-seq-lens 2048,3072,4096 --benchmark-steps 3 "
   f"--data-train {train_jsonl} --base-model Qwen/Qwen3-0.6B "
   f"--output {W}/smoke_lora", timeout=3600)
# seq_benchmark.json written next to output dir

# ---------------- 3. BASELINE Persian benchmark (BEFORE training) ----------------
sh(f"python3 {DATA}/evaluation/run_persian_benchmark.py --model Qwen/Qwen3-0.6B "
   f"--benchmark {bench_jsonl} --out {W}/baseline/baseline_responses.jsonl "
   f"--max-new 256", timeout=7200)
sh(f"python3 {DATA}/evaluation/persian_metrics.py "
   f"--responses {W}/baseline/baseline_responses.jsonl "
   f"--benchmark {bench_jsonl} --label BASE-Qwen3-0.6B "
   f"--out {W}/baseline/baseline_metrics.json")

# ---------------- 4. smoke training (tiny, proves full path) ----------------
# 1 epoch over a small slice is enough to validate adapter/merge/gguf
small_train = f"{W}/train_smoke_slice.jsonl"
with open(train_jsonl, encoding="utf-8") as f, open(small_train, "w", encoding="utf-8") as g:
    for i, line in enumerate(f):
        if i >= 1200:
            break
        g.write(line)
cfg["train_path"] = small_train
cfg["max_epochs"] = 1
cfg["eval_every_steps"] = 25
json.dump(cfg, open(f"{W}/config_smoke.json", "w"), indent=2)
sh(f"python3 {DATA}/training/train_kaggle.py --config {W}/config_smoke.json "
   f"--data-train {small_train} --data-val {val_jsonl} "
   f"--base-model Qwen/Qwen3-0.6B --output {W}/smoke_lora", timeout=5400)

# ---------------- 5. merge → GGUF → llama-cli smoke ----------------
sh(f"python3 {DATA}/conversion/merge_lora.py --base Qwen/Qwen3-0.6B "
   f"--lora {W}/smoke_lora --out {W}/smoke_merged", timeout=1800)
# llama.cpp (shallow clone, build quantize + cli only)
sh("git clone --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/temp/llama.cpp",
   timeout=600)
sh("cmake -S /kaggle/temp/llama.cpp -B /kaggle/temp/llama.cpp/build -DLLAMA_CURL=OFF "
   "-DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=OFF",
   timeout=900)
sh("cmake --build /kaggle/temp/llama.cpp/build --target llama-quantize llama-cli -j4",
   timeout=1800)
sh(f"python3 /kaggle/temp/llama.cpp/convert_hf_to_gguf.py {W}/smoke_merged "
   f"--outfile {W}/smoke-f16.gguf --outtype f16", timeout=3600)
sh(f"/kaggle/temp/llama.cpp/build/bin/llama-quantize {W}/smoke-f16.gguf "
   f"{W}/smoke-q4_k_m.gguf Q4_K_M", timeout=900)
smoke_prompt = "به فارسی جواب بده: هوش مصنوعی چیست؟"
sh(f"/kaggle/temp/llama.cpp/build/bin/llama-cli -m {W}/smoke-q4_k_m.gguf "
   f"-p '{smoke_prompt}' --single-turn -n 96 -no-warmup 2>&1 | tail -20",
   timeout=600, check=False)

# ---------------- 6. inventory ----------------
files = []
for p in glob.glob(f"{W}/**/*", recursive=True):
    if os.path.isfile(p) and "/.cache" not in p:
        files.append((p, os.path.getsize(p)))
print("\n=== /kaggle/working inventory ===", flush=True)
for p, s in sorted(files):
    print(f"  {s/1e6:9.1f} MB  {p}", flush=True)
print("KERNEL A DONE", flush=True)
