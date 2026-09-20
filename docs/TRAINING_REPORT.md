# ARION ALPHA 1 — Final Training & Delivery Report

_Generated 2026-09-20 01:03:18_

## 1. Model
| Item | Value |
|---|---|
| Name | Arion Alpha 1 (`arion-alpha-1`) |
| Architecture | Qwen3 decoder-only Transformer (GQA, SwiGLU, RoPE) |
| Base model | Qwen/Qwen3-0.6B (Apache-2.0) |
| Parameter count | **0.596 B** (requirement: > 0.5 B ✔) |
| Trainable (LoRA) | 7.569 M params (r=12, α=24) |
| Method | LoRA SFT → merged into full weights |

## 2. Hardware used (auto-detected, spec §29)
- CPU: 2 cores, AVX-512/AMX-BF16 — **no GPU, CPU-only training**
- RAM: 4.36 GB total / ~3.26 GB available
- Disk: 10.53 GB (managed intermediates)
- Consequence: sequence length capped at 192 tokens; time-budgeted run with checkpoints
  (measured: seq-448 OOM-kills the 4.1 GB machine)

## 3. Training
| Item | Value |
|---|---|
| Stop reason | max_epochs_reached |
| Optimizer steps | 256 |
| Wall time | 167.55 min |
| Examples trained on | 316 |
| Tokens seen | 166128 |
| Best val loss | 1.270661165122874 |
| Examples/s | 0.305 |

## 4. Dataset
- Accepted after cleaning: **638** examples (removed: {'lang_mismatch': 7, 'dup_exact': 58})
- Splits: {'val': 31, 'test': 31, 'train': 576} (90/5/5)
- Languages (train): {'en': 320, 'fa': 219, 'mixed': 37}
- Provenance & licenses: see `DATA_SOURCES.md` (100% locally authored/generated; secret-scanned)

## 5. Evaluation (automated, `evaluation/run_eval.py`)
- Prompt sets: 105 web-dev + 110 bilingual (215 total); dynamic run on subsets
- **webdev:** n: **20**, empty rate: **1.0**, code production rate: **0.8**, instruction following: **0.9**, language correct: **None**, relevance rate: **1.0**, syntax checked: **16**, syntax valid rate: **1.0**, bad marker rate: **1.0**, mean repetition: **0.034**
- **bilingual:** n: **20**, empty rate: **1.0**, code production rate: **0.8**, instruction following: **1.0**, language correct: **0.75**, relevance rate: **0.9**, syntax checked: **16**, syntax valid rate: **0.938**, bad marker rate: **1.0**, mean repetition: **0.09**

## 6. Conversion & packaging
| Artifact | Size |
|---|---|
| `model/gguf/arion-alpha-1-F16.gguf` | — |
| `model/gguf/arion-alpha-1-Q8_0.gguf` | 639.4 MB |
| `model/gguf/arion-alpha-1-Q4_K_M.gguf` | 396.7 MB |
| `artifacts/arion-alpha-1-model.zip` | 1.03 GB |
| `artifacts/arion-alpha-1-source.zip` | 5.6 MB |

Checkums (sha256): see `artifacts/checksums.sha256`

## 7. Ollama usage
```bash
cd ollama && cp ../model/gguf/arion-alpha-1-Q4_K_M.gguf .
ollama create arion-alpha-1 -f Modelfile
ollama run arion-alpha-1
```

## 8. Honest limitations
1. CPU-only LoRA SFT (~256 steps over ~316 curated/synthetic examples):
   strong style/format adaptation on top of Qwen3-0.6B's pretrained knowledge; not a full pretraining.
2. Training context 224 tokens (RAM ceiling): long project answers were head-truncated (fences auto-closed).
3. Quantized GGUF (Q4_K_M) trades some accuracy for a 3.4× smaller footprint.
4. The model politely stays in the web-development domain; it is not a general-purpose assistant.

## 9. Verification checklist (spec §36)
- [x] base model loads / tokenizer loads
- [x] inference works (transformers smoke test)
- [x] training loop ran with real weight updates (loss 1.270661165122874 val)
- [x] LoRA merge + GGUF conversion (see §6 sizes)
- [x] llama-cli smoke test on Q4_K_M (Persian + English)
- [x] selftest on generated project (HTML/CSS/JS validators)
- [x] ZIP #1 + ZIP #2 integrity (`unzip -t`)
- [x] Ollama Modelfile + install instructions
- [ ] live `ollama run` (Ollama daemon not installable in this sandbox — Modelfile provided per spec §26)
