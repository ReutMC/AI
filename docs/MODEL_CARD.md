# Model Card — ARION ALPHA 1

## Model details
- **Name:** Arion Alpha 1 (`arion-alpha-1` on Ollama)
- **Version:** 1.0
- **Type:** Decoder-only causal language model, instruction-tuned (LoRA SFT)
- **Base:** Qwen3-0.6B (Apache-2.0), 28 layers, GQA, SwiGLU, RoPE, 151 936-token BPE
- **Parameters:** 0.596 B total / 7.57 M trainable adapters (merged into full weights at release)
- **Languages:** Persian + English (bidirectional, code-switch aware)
- **Domain:** Web development — HTML, CSS, JavaScript, modern frontend, RTL typography

## Intended use
A bilingual web-development assistant and teacher: generate complete pages and multi-file
projects, explain and debug HTML/CSS/JS, review and refactor code, teach from beginner to expert,
produce Persian (RTL) websites with proper fonts and semantics.

## Out-of-scope use
General-purpose chat, medical/legal/financial advice, backend-only languages (the model may
produce plausible-but-unverified answers outside its specialization), high-stakes production
code without human review.

## Training data
- ~640 curated + synthetic instruction examples (Persian/English web development), authored and
  generated for this project; see `DATA_SOURCES.md`.
- 90/5/5 train/val/test split; test set untouched by training.

## Training procedure
- LoRA SFT (r=12, α=24, all attention+MLP projections), bf16 base, fp32 adapters, AdamW,
  cosine schedule, lr 1.2e-4, effective batch 8, 256-token context (hardware-constrained),
  ≤3 epochs, seed 42, time-budgeted with checkpoint/resume.
- Hardware: 2 vCPU / 4.1 GB RAM cloud VM, no GPU. Details and rationale: README §3.

## Evaluation summary
See `docs/TRAINING_REPORT.md` and `artifacts/eval_report.json` for the run produced with this
release: HTML/CSS/JS syntax validity rates, instruction following, language-following accuracy
on bilingual prompts, repetition/degeneration rates.

## Safety
- Training data secret-scanned (API keys, tokens, passwords).
- The model is instruction-tuned only on web-development content; it should politely decline
  unrelated requests ("I'm a web development assistant").
- No user data was collected or trained on.

## Quantizations shipped
| File | Precision | Size | Use |
|---|---|---|---|
| `arion-alpha-1-Q4_K_M.gguf` | 4-bit | ~0.4 GB | Ollama / low-memory (recommended) |
| `arion-alpha-1-Q8_0.gguf` | 8-bit | ~0.65 GB | better fidelity |
| `arion-alpha-1-F16.gguf` | 16-bit | ~1.2 GB | reference / rebuild quantizations |
