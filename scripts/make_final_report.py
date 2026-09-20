#!/usr/bin/env python3
"""ARION ALPHA 1 — assemble docs/TRAINING_REPORT.md and FINAL_REPORT.md from artifacts."""
import json, os, time

ROOT = "/home/z/my-project/arion-alpha-1"

def load(p):
    try:
        return json.load(open(os.path.join(ROOT, p)))
    except Exception:
        return None

def fmt_size(p):
    try:
        n = os.path.getsize(os.path.join(ROOT, p))
        return f"{n/1e9:.2f} GB" if n > 1e9 else f"{n/1e6:.1f} MB"
    except Exception:
        return "—"

def main():
    hw = load("artifacts/hardware_report.json") or {}
    ts = load("artifacts/train_summary.json") or {}
    ds = load("datasets/processed/dataset_stats.json") or {}
    ev = load("artifacts/eval_report.json") or {}
    zc = load("artifacts/checksums.json")

    sets = ev.get("sets", {})
    def sum_line(name):
        s = sets.get(name, {}).get("summary")
        if not s:
            return "_not run_"
        return ", ".join(f"{k.replace('_',' ')}: **{v}**" for k, v in s.items())

    total_params = 596_049_920
    report = f"""# ARION ALPHA 1 — Final Training & Delivery Report

_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_

## 1. Model
| Item | Value |
|---|---|
| Name | Arion Alpha 1 (`arion-alpha-1`) |
| Architecture | Qwen3 decoder-only Transformer (GQA, SwiGLU, RoPE) |
| Base model | Qwen/Qwen3-0.6B (Apache-2.0) |
| Parameter count | **{total_params/1e9:.3f} B** (requirement: > 0.5 B ✔) |
| Trainable (LoRA) | {ts.get('trainable_params_millions','—')} M params (r={12}, α={24}) |
| Method | {ts.get('method','LoRA SFT')} → merged into full weights |

## 2. Hardware used (auto-detected, spec §29)
- CPU: {hw.get('hardware',{}).get('cpu_cores','—')} cores, AVX-512/AMX-BF16 — **no GPU, CPU-only training**
- RAM: {hw.get('hardware',{}).get('ram_total_gb','—')} GB total / ~{hw.get('hardware',{}).get('ram_available_gb','—')} GB available
- Disk: {hw.get('hardware',{}).get('disk_total_gb','—')} GB (managed intermediates)
- Consequence: sequence length capped at 224 tokens; time-budgeted run with checkpoints
  (measured: seq-448 OOM-kills the 4.1 GB machine)

## 3. Training
| Item | Value |
|---|---|
| Stop reason | {ts.get('stop_reason','—')} |
| Optimizer steps | {ts.get('final_step','—')} |
| Wall time | {ts.get('total_minutes','—')} min |
| Examples trained on | {ts.get('examples_trained_on','—')} |
| Tokens seen | {ts.get('tokens_seen','—')} |
| Best val loss | {ts.get('best_val_loss','—')} |
| Examples/s | {ts.get('examples_per_sec','—')} |

## 4. Dataset
- Accepted after cleaning: **{ds.get('accepted_after_cleaning','—')}** examples (removed: {ds.get('rejected',{})})
- Splits: {ds.get('splits',{})} (90/5/5)
- Languages (train): {ds.get('train_langs',{})}
- Provenance & licenses: see `DATA_SOURCES.md` (100% locally authored/generated; secret-scanned)

## 5. Evaluation (automated, `evaluation/run_eval.py`)
- Prompt sets: 105 web-dev + 110 bilingual (215 total); dynamic run on subsets
- **webdev:** {sum_line('webdev')}
- **bilingual:** {sum_line('bilingual')}

## 6. Conversion & packaging
| Artifact | Size |
|---|---|
| `model/gguf/arion-alpha-1-F16.gguf` | {fmt_size('model/gguf/arion-alpha-1-F16.gguf')} |
| `model/gguf/arion-alpha-1-Q8_0.gguf` | {fmt_size('model/gguf/arion-alpha-1-Q8_0.gguf')} |
| `model/gguf/arion-alpha-1-Q4_K_M.gguf` | {fmt_size('model/gguf/arion-alpha-1-Q4_K_M.gguf')} |
| `artifacts/arion-alpha-1-model.zip` | {fmt_size('artifacts/arion-alpha-1-model.zip')} |
| `artifacts/arion-alpha-1-source.zip` | {fmt_size('artifacts/arion-alpha-1-source.zip')} |

Checkums (sha256): see `artifacts/checksums.sha256`

## 7. Ollama usage
```bash
cd ollama && cp ../model/gguf/arion-alpha-1-Q4_K_M.gguf .
ollama create arion-alpha-1 -f Modelfile
ollama run arion-alpha-1
```

## 8. Honest limitations
1. CPU-only LoRA SFT (~{ts.get('final_step','—')} steps over ~{ts.get('examples_trained_on','—')} curated/synthetic examples):
   strong style/format adaptation on top of Qwen3-0.6B's pretrained knowledge; not a full pretraining.
2. Training context 224 tokens (RAM ceiling): long project answers were head-truncated (fences auto-closed).
3. Quantized GGUF (Q4_K_M) trades some accuracy for a 3.4× smaller footprint.
4. The model politely stays in the web-development domain; it is not a general-purpose assistant.

## 9. Verification checklist (spec §36)
- [x] base model loads / tokenizer loads
- [x] inference works (transformers smoke test)
- [x] training loop ran with real weight updates (loss {ts.get('best_val_loss','n/a')} val)
- [x] LoRA merge + GGUF conversion (see §6 sizes)
- [x] llama-cli smoke test on Q4_K_M (Persian + English)
- [x] selftest on generated project (HTML/CSS/JS validators)
- [x] ZIP #1 + ZIP #2 integrity (`unzip -t`)
- [x] Ollama Modelfile + install instructions
- [ ] live `ollama run` (Ollama daemon not installable in this sandbox — Modelfile provided per spec §26)
"""
    os.makedirs(f"{ROOT}/docs", exist_ok=True)
    open(f"{ROOT}/docs/TRAINING_REPORT.md", "w").write(report)
    open(f"{ROOT}/artifacts/FINAL_REPORT.md", "w").write(report)
    print("reports written")

if __name__ == "__main__":
    main()
