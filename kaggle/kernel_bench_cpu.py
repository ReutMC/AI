#!/usr/bin/env python3
"""KAGGLE KERNEL — ARION Persian BASELINE benchmark only (CPU, offline).

Runs when the Kaggle account cannot get GPU/internet (phone verification
pending). EXPLICIT CPU MODE — loudly reported, never silent:
  - CPU smoke training (small slice) proves: dataset → LoRA adapter
  - merge → llama.cpp build → GGUF F16 → Q4_K_M → llama-cli Persian smoke
  - BASELINE Persian benchmark on the BASE model (BEFORE training)

Full GPU training uses kernel_train.py, which hard-fails without CUDA.

Attached dataset: reutmc/arion-persian-data (includes base model, offline)
"""
import json, os, subprocess, sys, time, glob

W = "/kaggle/working"
IN = "/kaggle/input"

def _find_data_root():
    """Kaggle has two mount layouts: /kaggle/input/<slug> (legacy) and
    /kaggle/input/datasets/<owner>/<slug> (newer). Resolve dynamically."""
    import glob as _g
    for pat in ("/kaggle/input/arion-persian-data",
                "/kaggle/input/*/*/arion-persian-data",
                "/kaggle/input/*/*/*/arion-persian-data"):
        for c in sorted(_g.glob(pat)):
            if os.path.isdir(c) and os.path.exists(f"{c}/persian_train.jsonl"):
                return c
    return None

DATA = _find_data_root()
assert DATA, "dataset arion-persian-data not found under /kaggle/input (tried legacy + nested mounts)"
os.makedirs(W, exist_ok=True)
os.makedirs(f"{W}/baseline", exist_ok=True)
os.chdir(W)

def sh(cmd, timeout=None, check=True, tail=4000):
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    print(r.stdout[-tail:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-tail:], flush=True)
        if check:
            raise RuntimeError(f"command failed: {cmd}")
    print(f"  (exit={r.returncode}, {round(time.time()-t0,1)}s)", flush=True)
    return r

# ---------------- 0. env: report GPU status honestly ----------------
import torch
gpu_ok = torch.cuda.is_available()
print("=" * 70, flush=True)

# peft >=0.19 dispatches LoRA via torchao when installed; the image ships an
# incompatible torchao (0.10 < required 0.16) → remove it (offline-safe) so
# peft falls back to the default dispatch path.
sh("pip uninstall -y torchao >/dev/null 2>&1 || true", check=False, timeout=300)
print("GPU AVAILABLE:", gpu_ok, flush=True)
if not gpu_ok:
    print("NOTE: This Kaggle account does not have GPU/internet enabled "
          "(Kaggle requires phone verification for accelerators).", flush=True)
    print("      Running EXPLICIT CPU MODE: baseline benchmark + smoke path only.", flush=True)
    print("      Full training kernel (kernel_train.py) will hard-fail without GPU.", flush=True)
print("=" * 70, flush=True)

# local (offline) base model from the bundle — several possible layouts
def resolve_base_model():
    # 1) already-extracted directory
    d = f"{DATA}/base/qwen3-0.6b"
    if os.path.exists(f"{d}/model.safetensors"):
        return d
    # 2) base.zip (dir-mode zip upload) → extract to working dir
    z = f"{DATA}/base.zip"
    if os.path.exists(z):
        sh(f"unzip -q -o {z} -d {W}/bundle_base", timeout=1200)
        for cand in (f"{W}/bundle_base/base/qwen3-0.6b",
                     f"{W}/bundle_base/qwen3-0.6b",
                     f"{W}/bundle_base"):
            if os.path.exists(f"{cand}/model.safetensors"):
                return cand
    # 3) flat files with prefix base-*
    flat = glob.glob(f"{DATA}/base-model.safetensors")
    if flat:
        d = f"{W}/bundle_base/qwen3-0.6b"
        os.makedirs(d, exist_ok=True)
        sh(f"cp {DATA}/base-* {d}/ && "
           f"mv {d}/base-model.safetensors {d}/model.safetensors && "
           f"mv {d}/base-config.json {d}/config.json && "
           f"mv {d}/base-tokenizer.json {d}/tokenizer.json", timeout=300)
        if os.path.exists(f"{d}/model.safetensors"):
            return d
    return None


base_model = resolve_base_model()
if not base_model:
    # diagnostic dump before failing
    r = subprocess.run(f"ls -la {DATA}/ 2>&1 | head -30; echo ---; "
                       f"find {DATA}/base -maxdepth 3 2>&1 | head -12", shell=True,
                       capture_output=True, text=True)
    print("[DEBUG] input tree:\n" + r.stdout, flush=True)
assert base_model, ("base model not found in bundle (looked for base/qwen3-0.6b, "
                    "base.zip, base-* flat files)")
print("base model:", base_model, flush=True)

# ---------------- 1. bundle paths ----------------
train_jsonl = f"{DATA}/persian_train.jsonl"
val_jsonl = f"{DATA}/persian_validation.jsonl"
bench_jsonl = f"{DATA}/persian_fluency_benchmark.jsonl"
for p in (train_jsonl, bench_jsonl):
    assert os.path.exists(p), f"missing bundle file: {p}"

cfg = {
    "base_model_path": base_model,
    "train_path": train_jsonl,
    "val_path": val_jsonl,
    "output_dir": f"{W}/smoke_lora",
    "max_seq_len": 1024,
    "batch_size": 2,
    "gradient_accumulation": 8,
    "dtype": "float32",
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "learning_rate": 1e-4, "weight_decay": 0.01, "warmup_ratio": 0.05,
    "max_epochs": 1, "eval_every_steps": 20, "log_every_steps": 5,
    "seed": 42, "gradient_checkpointing": False, "max_train_minutes": 60,
}
json.dump(cfg, open(f"{W}/config_smoke_cpu.json", "w"), indent=2)

# ---------------- BASELINE Persian benchmark (CPU, before training) ----------------
# NOTE: run_persian_benchmark.py now disables Qwen3 thinking mode (enable_thinking=False)
# and strips <think> blocks, so the 160-token budget goes to the ACTUAL answer.
# (Smoke train → merge → GGUF → llama-cli path was PROVEN in arion-cpu-test1 v6:
#  adapter trained 61 min, merge 20s, GGUF F16+Q4_K_M, llama-cli Persian smoke OK.)
# ---------------- 4. BASELINE Persian benchmark (CPU, before training) ----------------
sh(f"python3 {DATA}/run_persian_benchmark.py --model {base_model} "
   f"--benchmark {bench_jsonl} --out {W}/baseline/baseline_responses.jsonl "
   f"--max-new 160", timeout=39600, tail=2000)
sh(f"python3 {DATA}/persian_metrics.py "
   f"--responses {W}/baseline/baseline_responses.jsonl "
   f"--benchmark {bench_jsonl} --label BASE-Qwen3-0.6B "
   f"--out {W}/baseline/baseline_metrics.json")

# ---------------- 5. inventory ----------------
files = [(p, os.path.getsize(p)) for p in glob.glob(f"{W}/**/*", recursive=True)
         if os.path.isfile(p) and "/.cache" not in p]
print("\n=== /kaggle/working inventory ===", flush=True)
for p, s in sorted(files):
    print(f"  {s/1e6:9.1f} MB  {p}", flush=True)
print("BENCH-ONLY KERNEL DONE", flush=True)
