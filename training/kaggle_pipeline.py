#!/usr/bin/env python3
"""ARION ALPHA 1 — Kaggle training pipeline launcher.

Orchestrates the Persian-phase pipeline:
  bundle         — assemble the flattened data bundle (jsonl + scripts + configs)
  dataset-push   — create or version the private Kaggle dataset
  kernel-push    — push baseline or train kernel (GPU) for execution
  status         — one-shot kernel status
  monitor        — poll kernel until completion
  fetch          — download kernel outputs
  run-baseline   — bundle → dataset-push → push Kernel A → monitor → fetch
  run-train      — update bundle with Kernel A results → dataset-push →
                   push Kernel B → monitor → fetch
  run-all        — run-baseline then run-train

Credentials: read ONLY from KAGGLE_API_TOKEN env var (never hardcoded, never
logged). GitHub Actions passes it as a secret; locally export it manually.
"""
import argparse, json, os, shutil, subprocess, sys, time, hashlib

_HERE = os.path.dirname(os.path.abspath(__file__))            # .../training
ARION = os.path.dirname(_HERE)                                # repo root (flat) or parent
if os.path.isdir(os.path.join(os.path.dirname(ARION), "arion-alpha-1")):
    ARION = os.path.join(os.path.dirname(ARION), "arion-alpha-1")  # nested legacy layout
ROOT = ARION
BUNDLE_DIR = os.path.join(ARION, "output", "kaggle_bundle")
USER = os.environ.get("KAGGLE_USERNAME", "reutmc")
DATASET_SLUG = "arion-persian-data"
KERNEL_BASELINE = "arion-persian-baseline"
KERNEL_TRAIN = "arion-persian-train"
KERNEL_BASELINE_CPU = "arion-persian-baseline-cpu"


def die(msg):
    print(f"ERROR: {msg}", flush=True)
    sys.exit(1)


def require_token():
    tok = os.environ.get("KAGGLE_API_TOKEN")
    if not tok:
        die("KAGGLE_API_TOKEN env var is not set — refusing to proceed "
            "(credentials must never be hardcoded).")
    return tok


def sh(cmd, timeout=None):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "") + (("\n[stderr] " + r.stderr) if r.returncode != 0 else "")
    print(out[-3000:], flush=True)
    return r


def git_commit():
    for repo in (os.path.join(ROOT, "gh-reutmc-ai"), ARION):
        try:
            r = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return r.stdout.strip()
        except Exception:
            pass
    return "unknown"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- bundle ----------------
def bundle(extra_files=None, with_model=False):
    if os.path.isdir(BUNDLE_DIR):
        shutil.rmtree(BUNDLE_DIR)
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    files = [
        (f"{ARION}/datasets/processed/persian_train.jsonl", "persian_train.jsonl"),
        (f"{ARION}/datasets/processed/persian_validation.jsonl", "persian_validation.jsonl"),
        (f"{ARION}/datasets/processed/persian_stats.json", "persian_stats.json"),
        (f"{ARION}/evaluation/persian_fluency_benchmark.jsonl", "persian_fluency_benchmark.jsonl"),
        (f"{ARION}/evaluation/persian_metrics.py", "persian_metrics.py"),
        (f"{ARION}/evaluation/persian_langid.py", "persian_langid.py"),
        (f"{ARION}/evaluation/run_persian_benchmark.py", "run_persian_benchmark.py"),
        (f"{ARION}/training/train_kaggle.py", "train_kaggle.py"),
        (f"{ARION}/conversion/merge_lora.py", "merge_lora.py"),
        (f"{ARION}/kaggle/kernel_baseline.py", "kernel_baseline.py"),
        (f"{ARION}/kaggle/kernel_baseline_cpu.py", "kernel_baseline_cpu.py"),
        (f"{ARION}/kaggle/kernel_train.py", "kernel_train.py"),
    ]
    if with_model:
        # offline base model (for CPU-mode kernels / no-internet sessions)
        bdir = os.path.join(BUNDLE_DIR, "base", "qwen3-0.6b")
        os.makedirs(bdir, exist_ok=True)
        for f in os.listdir(f"{ARION}/model/base"):
            if f.endswith((".json", ".txt", ".safetensors", ".model")):
                shutil.copy2(os.path.join(f"{ARION}/model/base", f), os.path.join(bdir, f))
        # llama.cpp source tarball (offline build when internet is disabled)
        lsrc = f"{ARION}/conversion/llama.cpp"
        if os.path.isdir(lsrc):
            exclude = "--exclude=.git --exclude=build --exclude=*.gguf"
            sh(f"tar czf {BUNDLE_DIR}/llama.cpp.tar.gz {exclude} -C {lsrc} .", timeout=600)
    for src, dst in files:
        if not os.path.exists(src):
            die(f"bundle source missing: {src}")
        shutil.copy2(src, os.path.join(BUNDLE_DIR, dst))
    json.dump({
        "title": "arion-persian-data",
        "id": f"{USER}/{DATASET_SLUG}",
        "licenses": [{"name": "apache-2.0"}],
        "keywords": ["persian", "arion", "sft"],
    }, open(os.path.join(BUNDLE_DIR, "dataset-metadata.json"), "w"), indent=2)
    manifest = {
        "git_commit": git_commit(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {},
    }
    if extra_files:
        for src in extra_files:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(BUNDLE_DIR, os.path.basename(src)))
    for root, _dirs, fnames in os.walk(BUNDLE_DIR):
        for f in sorted(fnames):
            if f in ("bundle_manifest.json", "dataset-metadata.json"):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, BUNDLE_DIR)
            manifest["files"][rel] = {"sha256": file_sha256(full)}
    json.dump(manifest, open(os.path.join(BUNDLE_DIR, "bundle_manifest.json"), "w"), indent=2)
    print(f"[bundle] {len(manifest['files'])} files assembled in {BUNDLE_DIR}", flush=True)
    return BUNDLE_DIR


# ---------------- dataset ----------------
def dataset_push():
    require_token()
    meta = {
        "title": "arion-persian-data",
        "id": f"{USER}/{DATASET_SLUG}",
        "licenses": [{"name": "apache-2.0"}],
        "keywords": ["persian", "arion", "sft"],
    }
    json.dump(meta, open(os.path.join(BUNDLE_DIR, "dataset-metadata.json"), "w"), indent=2)
    exists = sh(f"kaggle datasets status {USER}/{DATASET_SLUG}")
    if exists.returncode == 0:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        r = sh(f'kaggle datasets version -p {BUNDLE_DIR} -m "pipeline sync {stamp}" -q',
               timeout=3600)
        if r.returncode != 0:
            die("dataset version push failed")
        print("[dataset] version pushed; waiting for processing...", flush=True)
        for _ in range(60):
            time.sleep(10)
            st = sh(f"kaggle datasets status {USER}/{DATASET_SLUG}")
            if "complete" in (st.stdout or "").lower():
                break
    else:
        r = sh(f"kaggle datasets create -p {BUNDLE_DIR} -q", timeout=3600)
        if r.returncode != 0:
            die("dataset create failed")
        print("[dataset] created; waiting for processing...", flush=True)
        time.sleep(60)
    print("[dataset] ready", flush=True)


# ---------------- kernels ----------------
def kernel_push(kind):
    require_token()
    slug = KERNEL_BASELINE if kind == "baseline" else KERNEL_TRAIN
    code = "kernel_baseline.py" if kind == "baseline" else "kernel_train.py"
    src_code = os.path.join(ARION, "kaggle", code)  # source of truth
    kd = os.path.join(ARION, "output", f"kaggle_kernel_{kind}")
    if os.path.isdir(kd):
        shutil.rmtree(kd)
    os.makedirs(kd, exist_ok=True)
    shutil.copy2(src_code, os.path.join(kd, code))
    meta = {
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": code,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{USER}/{DATASET_SLUG}"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    json.dump(meta, open(os.path.join(kd, "kernel-metadata.json"), "w"), indent=2)
    r = sh(f"kaggle kernels push -p {kd}", timeout=1800)
    if r.returncode != 0:
        die(f"kernel push failed for {slug}: {r.stdout or r.stderr}")
    print(f"[kernel] {slug} pushed & queued", flush=True)


def kernel_push_cpu():
    require_token()
    slug = KERNEL_BASELINE_CPU
    code = "kernel_baseline_cpu.py"
    src_code = os.path.join(ARION, "kaggle", code)
    kd = os.path.join(ARION, "output", "kaggle_kernel_baseline_cpu")
    if os.path.isdir(kd):
        shutil.rmtree(kd)
    os.makedirs(kd, exist_ok=True)
    shutil.copy2(src_code, os.path.join(kd, code))
    meta = {
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": code,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [f"{USER}/{DATASET_SLUG}"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    json.dump(meta, open(os.path.join(kd, "kernel-metadata.json"), "w"), indent=2)
    r = sh(f"kaggle kernels push -p {kd}", timeout=1800)
    if r.returncode != 0:
        die(f"kernel push failed for {slug}: {r.stdout or r.stderr}")
    print(f"[kernel] {slug} pushed & queued (CPU mode)", flush=True)


def kernel_status(slug):
    r = sh(f"kaggle kernels status {USER}/{slug}")
    if r.returncode != 0:
        return "unknown"
    out = (r.stdout or "") + (r.stderr or "")
    low = out.lower()
    if "complete" in low:
        return "complete"
    if "error" in low:
        return "error"
    if "cancel" in low:
        return "cancelled"
    if "running" in low or "queued" in low:
        return "running"
    return out.strip()[-120:]


def monitor(slug, poll_s=90, max_hours=14):
    print(f"[monitor] {slug} ...", flush=True)
    t0 = time.time()
    while time.time() - t0 < max_hours * 3600:
        st = kernel_status(slug)
        print(f"  [{time.strftime('%H:%M:%S')}] {slug}: {st}", flush=True)
        if st in ("complete", "error", "cancelled"):
            return st
        time.sleep(poll_s)
    return "timeout"


def fetch(slug, outdir):
    os.makedirs(outdir, exist_ok=True)
    r = sh(f"kaggle kernels output {USER}/{slug} -p {outdir}", timeout=3600)
    if r.returncode != 0:
        die(f"fetch failed for {slug}")
    print(f"[fetch] {slug} → {outdir}", flush=True)


# ---------------- high-level runs ----------------
def run_baseline(outdir):
    bundle()
    dataset_push()
    kernel_push("baseline")
    st = monitor(KERNEL_BASELINE)
    fetch(KERNEL_BASELINE, outdir)
    return st


def run_train(outdir, baseline_outdir=None):
    extra = []
    if baseline_outdir and os.path.isdir(baseline_outdir):
        for f in ("seq_benchmark.json", "baseline_metrics.json", "baseline_responses.jsonl"):
            p = os.path.join(baseline_outdir, f)
            if os.path.exists(p):
                extra.append(p)
    bundle(extra)
    dataset_push()
    kernel_push("train")
    st = monitor(KERNEL_TRAIN)
    fetch(KERNEL_TRAIN, outdir)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["bundle", "dataset-push", "kernel-push",
                                        "status", "monitor", "fetch",
                                        "run-baseline", "run-train", "run-all"])
    ap.add_argument("--kind", choices=["baseline", "baseline-cpu", "train"],
                    default="baseline")
    ap.add_argument("--with-model", action="store_true",
                    help="include base model + llama.cpp in the bundle (offline mode)")
    ap.add_argument("--slug", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-hours", type=float, default=14)
    args = ap.parse_args()

    outdir = args.out or os.path.join(ARION, "output", f"kaggle_{args.kind}_out")
    if args.command == "bundle":
        bundle(with_model=args.with_model)
    elif args.command == "dataset-push":
        bundle(with_model=args.with_model)
        dataset_push()
    elif args.command == "kernel-push":
        if args.kind == "baseline-cpu":
            kernel_push_cpu()
        else:
            kernel_push(args.kind)
    elif args.command == "status":
        slug = args.slug or {"baseline": KERNEL_BASELINE,
                             "baseline-cpu": KERNEL_BASELINE_CPU,
                             "train": KERNEL_TRAIN}[args.kind]
        print(kernel_status(slug))
    elif args.command == "monitor":
        slug = args.slug or {"baseline": KERNEL_BASELINE,
                             "baseline-cpu": KERNEL_BASELINE_CPU,
                             "train": KERNEL_TRAIN}[args.kind]
        print(monitor(slug, max_hours=args.max_hours))
    elif args.command == "fetch":
        slug = args.slug or {"baseline": KERNEL_BASELINE,
                             "baseline-cpu": KERNEL_BASELINE_CPU,
                             "train": KERNEL_TRAIN}[args.kind]
        fetch(slug, outdir)
    elif args.command == "run-baseline":
        run_baseline(outdir)
    elif args.command == "run-train":
        bl_out = os.path.join(ARION, "output", "kaggle_baseline_out")
        run_train(outdir, bl_out)
    elif args.command == "run-all":
        st1 = run_baseline(os.path.join(ARION, "output", "kaggle_baseline_out"))
        print(f"[run-all] baseline status: {st1}", flush=True)
        if st1 != "complete":
            die("baseline kernel did not complete — aborting train kernel")
        st2 = run_train(outdir, os.path.join(ARION, "output", "kaggle_baseline_out"))
        print(f"[run-all] train status: {st2}", flush=True)


if __name__ == "__main__":
    main()
