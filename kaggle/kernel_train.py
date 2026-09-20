#!/usr/bin/env python3
"""KAGGLE KERNEL B — ARION Persian FULL training on GPU, then:
LoRA adapter → merge → GGUF (F16 → Q8_0 → Q4_K_M) → llama-server →
full Persian benchmark on the GGUF → metrics. All saved to /kaggle/working.

Attached dataset: reutmc/arion-persian-data
Optional Kaggle Secret: SEQ_LENS override is read from working/config.json
"""
import json, os, subprocess, sys, time, glob, urllib.request, hashlib

W = "/kaggle/working"
IN = "/kaggle/input"
DATA = IN + "/arion-persian-data"  # flattened bundle (see kaggle_pipeline.py bundle)
os.makedirs(W, exist_ok=True)
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

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------- 0. env ----------------
import torch
assert torch.cuda.is_available(), "FATAL: no GPU"
print("GPU:", torch.cuda.get_device_properties(0), flush=True)
try:
    import peft  # noqa
except ImportError:
    sh("pip install -q peft", timeout=300)

# ---------------- 1. bundle paths (flattened layout) ----------------
train_jsonl = f"{DATA}/persian_train.jsonl"
val_jsonl = f"{DATA}/persian_validation.jsonl"
bench_jsonl = f"{DATA}/persian_fluency_benchmark.jsonl"
# seq benchmark result from Kernel A, re-uploaded inside the bundle by the launcher
chosen_seq = 3072  # preferred target; overridden by benchmark outcome if available
seqbench = f"{DATA}/seq_benchmark.json"
if os.path.exists(seqbench):
    sb = json.load(open(seqbench)).get("seq_benchmark", {})
    # choose largest len with real measurements, no error, and VRAM < 90% of 16GB
    ok = [(int(k), v) for k, v in sb.items() if "error" not in v and v.get("peak_vram_gb", 99) < 14.5]
    if ok:
        ok.sort()
        chosen_seq = ok[-1][0]
        print(f"[seq] benchmark-driven choice: max_seq_len={chosen_seq}", flush=True)
    else:
        chosen_seq = 2048
        print("[seq] no usable benchmark entry → fallback 2048", flush=True)
else:
    print("[seq] no seq_benchmark.json in bundle → default 3072", flush=True)

cfg = {
    "base_model_path": "Qwen/Qwen3-0.6B",
    "train_path": train_jsonl,
    "val_path": val_jsonl,
    "output_dir": f"{W}/arion-persian-lora",
    "max_seq_len": chosen_seq,
    "batch_size": 4,
    "gradient_accumulation": 8,
    "dtype": "auto",
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "learning_rate": 1e-4, "weight_decay": 0.01, "warmup_ratio": 0.05,
    "max_epochs": 2, "eval_every_steps": 100, "log_every_steps": 10,
    "seed": 42, "gradient_checkpointing": True,
}
json.dump(cfg, open(f"{W}/config_full.json", "w"), indent=2)

# reproducibility metadata
git_commit = "unknown"
try:
    manifest = json.load(open(f"{DATA}/bundle_manifest.json"))
    git_commit = manifest.get("git_commit", "unknown")
except Exception:
    pass
meta = {
    "git_commit": git_commit,
    "dataset_files": {
        os.path.basename(p): {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
        for p in (train_jsonl, val_jsonl, bench_jsonl) if os.path.exists(p)
    },
    "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
    "model_name": "arion-alpha-1-persian",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "max_seq_len": chosen_seq, "seed": cfg["seed"],
    "lora": {"r": cfg["lora_r"], "alpha": cfg["lora_alpha"], "targets": cfg["lora_targets"]},
    "learning_rate": cfg["learning_rate"], "batch_size": cfg["batch_size"],
    "gradient_accumulation": cfg["gradient_accumulation"], "epochs": cfg["max_epochs"],
}
json.dump(meta, open(f"{W}/run_metadata.json", "w"), indent=2)

# ---------------- 2. FULL TRAINING ----------------
sh(f"python3 {DATA}/training/train_kaggle.py --config {W}/config_full.json "
   f"--data-train {train_jsonl} --data-val {val_jsonl} "
   f"--base-model Qwen/Qwen3-0.6B --output {W}/arion-persian-lora",
   timeout=36000, tail=6000)
summary = json.load(open(f"{W}/train_summary.json"))
meta["train_summary"] = summary
meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(meta, open(f"{W}/run_metadata.json", "w"), indent=2)

# ---------------- 3. merge ----------------
sh(f"python3 {DATA}/conversion/merge_lora.py --base Qwen/Qwen3-0.6B "
   f"--lora {W}/arion-persian-lora --out {W}/arion-persian-merged", timeout=2400)

# ---------------- 4. llama.cpp + GGUF F16/Q8_0/Q4_K_M ----------------
sh("git clone --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/temp/llama.cpp",
   timeout=600)
sh("cmake -S /kaggle/temp/llama.cpp -B /kaggle/temp/llama.cpp/build -DLLAMA_CURL=OFF "
   "-DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=OFF",
   timeout=900)
sh("cmake --build /kaggle/temp/llama.cpp/build --target llama-quantize llama-cli llama-server -j4",
   timeout=2400)
BIN = "/kaggle/temp/llama.cpp/build/bin"
sh(f"python3 /kaggle/temp/llama.cpp/convert_hf_to_gguf.py {W}/arion-persian-merged "
   f"--outfile {W}/arion-persian-f16.gguf --outtype f16", timeout=7200)
sh(f"{BIN}/llama-quantize {W}/arion-persian-f16.gguf {W}/arion-persian-Q8_0.gguf Q8_0",
   timeout=1800)
sh(f"{BIN}/llama-quantize {W}/arion-persian-f16.gguf {W}/arion-persian-Q4_K_M.gguf Q4_K_M",
   timeout=1800)
for g in ("arion-persian-f16.gguf", "arion-persian-Q8_0.gguf", "arion-persian-Q4_K_M.gguf"):
    p = f"{W}/{g}"
    json.dump({"file": g, "sha256": sha256_file(p), "bytes": os.path.getsize(p)},
              open(f"{W}/{g}.sha256.json", "w"), indent=2)

# ---------------- 5. inference smoke on Q4_K_M (in Persian) ----------------
smoke_prompt = "به فارسی جواب بده: هوش مصنوعی چیست؟"
r = sh(f"{BIN}/llama-cli -m {W}/arion-persian-Q4_K_M.gguf -p '{smoke_prompt}' "
       f"--single-turn -n 128 -no-warmup 2>&1 | tail -24", timeout=600, check=False, tail=3000)
open(f"{W}/gguf_persian_smoke.txt", "w").write(r.stdout)

# ---------------- 6. full Persian benchmark ON GGUF (llama-server) ----------------
server = subprocess.Popen(
    [f"{BIN}/llama-server", "-m", f"{W}/arion-persian-Q4_K_M.gguf",
     "-c", "4096", "-np", "4", "--host", "127.0.0.1", "--port", "8080"],
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
    sh(f"python3 {DATA}/evaluation/run_persian_benchmark.py "
       f"--server http://127.0.0.1:8080 --benchmark {bench_jsonl} "
       f"--out {W}/arion_gguf_responses.jsonl --max-new 256", timeout=7200)
finally:
    server.terminate()

sh(f"python3 {DATA}/evaluation/persian_metrics.py "
   f"--responses {W}/arion_gguf_responses.jsonl --benchmark {bench_jsonl} "
   f"--label ARION-alpha1-persian-Q4_K_M --train-summary {W}/train_summary.json "
   f"--out {W}/arion_gguf_metrics.json")

# ---------------- 7. adapter + summaries inventory ----------------
# merged model dir is large (~1.2GB) — include it, quota is 20GB
files = [(p, os.path.getsize(p)) for p in glob.glob(f"{W}/**/*", recursive=True)
         if os.path.isfile(p) and "/.cache" not in p]
print("\n=== /kaggle/working inventory ===", flush=True)
total = 0
for p, s in sorted(files):
    total += s
    print(f"  {s/1e6:9.1f} MB  {p}", flush=True)
print(f"TOTAL: {total/1e9:.2f} GB", flush=True)
print("KERNEL B DONE", flush=True)
