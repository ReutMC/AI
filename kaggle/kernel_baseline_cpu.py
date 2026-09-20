#!/usr/bin/env python3
"""KAGGLE KERNEL A (CPU fallback) — ARION Persian phase.

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

# ---------------- 2. CPU smoke training (small slice) ----------------
small_train = f"{W}/train_smoke_slice.jsonl"
with open(train_jsonl, encoding="utf-8") as f, open(small_train, "w", encoding="utf-8") as g:
    for i, line in enumerate(f):
        if i >= 600:
            break
        g.write(line)
sh(f"python3 {DATA}/train_kaggle.py --config {W}/config_smoke_cpu.json "
   f"--data-train {small_train} --data-val {val_jsonl} "
   f"--base-model {base_model} --output {W}/smoke_lora --allow-cpu",
   timeout=7200, tail=6000)
assert os.path.exists(f"{W}/smoke_lora/adapter_model.safetensors"), "no adapter produced"

# ---------------- 3. merge → GGUF → llama-cli smoke ----------------
sh(f"python3 {DATA}/merge_lora.py --base {base_model} "
   f"--lora {W}/smoke_lora --out {W}/smoke_merged", timeout=3600)
# llama.cpp source: bundled (Kaggle auto-extracts tar.gz inside datasets)
LL_SRC = f"{DATA}/llama.cpp"
if not os.path.exists(f"{LL_SRC}/convert_hf_to_gguf.py"):
    tar = f"{DATA}/llama.cpp.tar.gz"
    assert os.path.exists(tar), "llama.cpp source not found in bundle"
    os.makedirs("/kaggle/temp", exist_ok=True)
    sh(f"tar xzf {tar} -C /kaggle/temp/llama.cpp --one-top-level")
    LL_SRC = "/kaggle/temp/llama.cpp"
sh(f"cmake -S {LL_SRC} -B {LL_SRC}/build -DLLAMA_CURL=OFF "
   "-DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=OFF",
   timeout=1800)
sh(f"cmake --build {LL_SRC}/build --target llama-quantize llama-cli -j4",
   timeout=3600)
BIN = f"{LL_SRC}/build/bin"
sh(f"python3 {LL_SRC}/convert_hf_to_gguf.py {W}/smoke_merged "
   f"--outfile {W}/smoke-f16.gguf --outtype f16", timeout=7200)
sh(f"{BIN}/llama-quantize {W}/smoke-f16.gguf "
   f"{W}/smoke-q4_k_m.gguf Q4_K_M", timeout=1800)
smoke_prompt = "به فارسی جواب بده: هوش مصنوعی چیست؟"
r = sh(f"{BIN}/llama-cli -m {W}/smoke-q4_k_m.gguf "
       f"-p '{smoke_prompt}' --single-turn -n 64 -t 4 -no-warmup 2>&1 | tail -24",
       timeout=1800, check=False, tail=3000)
open(f"{W}/gguf_persian_smoke.txt", "w").write(r.stdout)

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
print("KERNEL A-CPU DONE", flush=True)
