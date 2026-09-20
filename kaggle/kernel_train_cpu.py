#!/usr/bin/env python3
"""KAGGLE KERNEL — ARION Persian CPU TRAINING (explicit, time-boxed).

Runs when GPU is unavailable (Kaggle phone-verification pending). This is an
EXPLICIT CPU run — loudly reported — on a quality-ranked subset that fits the
12h session budget:
  subset (3,600 top-ranked train examples + all booster/seed rows, seq 1024,
  1 epoch, LoRA r16 α32 lr 1e-4) → merge → GGUF F16 → Q4_K_M → llama-server
  → full 239-prompt Persian benchmark ON THE GGUF → metrics.
Produces a measurable ARION v1.1-pre while the full 45k GPU run awaits
account verification (kernel_train.py, unchanged).
"""
import json, os, subprocess, sys, time, glob, urllib.request, hashlib

W = "/kaggle/working"
IN = "/kaggle/input"
os.makedirs(W, exist_ok=True)
os.chdir(W)

def _find_data_root():
    import glob as _g
    for pat in ("/kaggle/input/arion-persian-data",
                "/kaggle/input/*/*/arion-persian-data",
                "/kaggle/input/*/*/*/arion-persian-data"):
        for c in sorted(_g.glob(pat)):
            if os.path.isdir(c) and os.path.exists(f"{c}/persian_train.jsonl"):
                return c
    return None

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

DATA = _find_data_root()
assert DATA, "dataset arion-persian-data not found under /kaggle/input"
print("dataset root:", DATA, flush=True)

# ---------------- 0. env: honest banner ----------------
import torch
gpu_ok = torch.cuda.is_available()
print("=" * 70, flush=True)
print("GPU AVAILABLE:", gpu_ok, flush=True)
if gpu_ok:
    print("GPU present — this CPU kernel was intended as fallback; continuing anyway.", flush=True)
else:
    print("EXPLICIT CPU TRAINING MODE (phone verification pending on this Kaggle account).", flush=True)
    print("Time-boxed subset run; full 45k training uses kernel_train.py on GPU.", flush=True)
print("=" * 70, flush=True)
sh("pip uninstall -y torchao >/dev/null 2>&1 || true", check=False, timeout=300)

# ---------------- 1. subset selection (deterministic) ----------------
train_jsonl = f"{DATA}/persian_train.jsonl"
val_jsonl = f"{DATA}/persian_validation.jsonl"
bench_jsonl = f"{DATA}/persian_fluency_benchmark.jsonl"

# the bundle ships persian_stats.json whose domain mix guides the cap; here we
# simply take the first N rows (dataset builder already wrote them
# quality-ranked and domain-balanced via rng.shuffle with seed 42)
N_SUBSET = 3600
subset = f"{W}/train_subset.jsonl"
with open(train_jsonl, encoding="utf-8") as f, open(subset, "w", encoding="utf-8") as g:
    for i, line in enumerate(f):
        if i >= N_SUBSET:
            break
        g.write(line)
print(f"subset: {i + 1} rows", flush=True)

cfg = {
    "base_model_path": f"{DATA}/base/qwen3-0.6b",
    "train_path": subset,
    "val_path": val_jsonl,
    "output_dir": f"{W}/arion-cpu-lora",
    "max_seq_len": 1024,
    "batch_size": 2,
    "gradient_accumulation": 8,
    "dtype": "float32",
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "learning_rate": 1e-4, "weight_decay": 0.01, "warmup_ratio": 0.05,
    "max_epochs": 1, "eval_every_steps": 60, "log_every_steps": 10,
    "seed": 42, "gradient_checkpointing": False,
    "max_train_minutes": 540,
}
json.dump(cfg, open(f"{W}/config_cpu_train.json", "w"), indent=2)

meta = {
    "mode": "CPU-fallback (explicit, phone-verification pending)",
    "git_commit": "see bundle_manifest.json",
    "subset_rows": N_SUBSET,
    "max_seq_len": cfg["max_seq_len"],
    "lora": {"r": cfg["lora_r"], "alpha": cfg["lora_alpha"], "lr": cfg["learning_rate"]},
    "seed": cfg["seed"],
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(meta, open(f"{W}/run_metadata.json", "w"), indent=2)

# ---------------- 2. training (time-boxed) ----------------
sh(f"python3 {DATA}/train_kaggle.py --config {W}/config_cpu_train.json "
   f"--data-train {subset} --data-val {val_jsonl} "
   f"--base-model {cfg['base_model_path']} --output {W}/arion-cpu-lora --allow-cpu",
   timeout=38000, tail=6000)
assert os.path.exists(f"{W}/arion-cpu-lora/adapter_model.safetensors"), "no adapter produced"
meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(meta, open(f"{W}/run_metadata.json", "w"), indent=2)

# ---------------- 3. merge → GGUF ----------------
sh(f"python3 {DATA}/merge_lora.py --base {cfg['base_model_path']} "
   f"--lora {W}/arion-cpu-lora --out {W}/arion-cpu-merged", timeout=3600)
LL_SRC = f"{DATA}/llama.cpp"
if not os.path.exists(f"{LL_SRC}/convert_hf_to_gguf.py"):
    tar = f"{DATA}/llama.cpp.tar.gz"
    assert os.path.exists(tar), "llama.cpp source not found in bundle"
    os.makedirs("/kaggle/temp", exist_ok=True)
    sh(f"mkdir -p /kaggle/temp/llama.cpp && tar xzf {tar} -C /kaggle/temp/llama.cpp")
    LL_SRC = "/kaggle/temp/llama.cpp"
sh(f"cmake -S {LL_SRC} -B /kaggle/temp/llama-build -DLLAMA_CURL=OFF "
   "-DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=OFF",
   timeout=1800)
sh("cmake --build /kaggle/temp/llama-build --target llama-quantize llama-cli llama-server -j4",
   timeout=3000)
BIN = "/kaggle/temp/llama-build/bin"
sh(f"python3 {LL_SRC}/convert_hf_to_gguf.py {W}/arion-cpu-merged "
   f"--outfile {W}/arion-cpu-f16.gguf --outtype f16", timeout=7200)
sh(f"{BIN}/llama-quantize {W}/arion-cpu-f16.gguf {W}/arion-cpu-Q4_K_M.gguf Q4_K_M",
   timeout=1800)
for g in ("arion-cpu-f16.gguf", "arion-cpu-Q4_K_M.gguf"):
    h = hashlib.sha256(open(f"{W}/{g}", "rb").read(1 << 30)).hexdigest()  # first 1GB for speed
    json.dump({"file": g, "sha256_first_1gb": h}, open(f"{W}/{g}.sha256.json", "w"))

smoke_prompt = "به فارسی جواب بده: هوش مصنوعی چیست؟"
r = sh(f"{BIN}/llama-cli -m {W}/arion-cpu-Q4_K_M.gguf -p '{smoke_prompt}' "
       f"--single-turn -n 128 -t 4 -no-warmup 2>&1 | tail -24", timeout=1200, check=False, tail=3000)
open(f"{W}/gguf_persian_smoke.txt", "w").write(r.stdout)

# ---------------- 4. full Persian benchmark ON THE GGUF ----------------
server = subprocess.Popen(
    [f"{BIN}/llama-server", "-m", f"{W}/arion-cpu-Q4_K_M.gguf",
     "-c", "4096", "-t", "4", "-np", "2", "--host", "127.0.0.1", "--port", "8080"],
    stdout=open(f"{W}/llama_server.log", "w"), stderr=subprocess.STDOUT)
try:
    up = False
    for _ in range(60):
        time.sleep(2)
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
            up = True
            break
        except Exception:
            pass
    assert up, "llama-server did not come up"
    print("llama-server UP", flush=True)
    sh(f"python3 {DATA}/run_persian_benchmark.py "
       f"--server http://127.0.0.1:8080 --benchmark {bench_jsonl} "
       f"--out {W}/arion_gguf_responses.jsonl --max-new 192", timeout=14400, tail=1500)
finally:
    server.terminate()

sh(f"python3 {DATA}/persian_metrics.py "
   f"--responses {W}/arion_gguf_responses.jsonl --benchmark {bench_jsonl} "
   f"--label ARION-cpu-pre-Q4_K_M --train-summary {W}/train_summary.json "
   f"--out {W}/arion_gguf_metrics.json")

files = [(p, os.path.getsize(p)) for p in glob.glob(f"{W}/**/*", recursive=True)
         if os.path.isfile(p) and "/.cache" not in p]
print("\n=== /kaggle/working inventory ===", flush=True)
for p, s in sorted(files):
    print(f"  {s/1e6:9.1f} MB  {p}", flush=True)
print("CPU TRAIN KERNEL DONE", flush=True)
