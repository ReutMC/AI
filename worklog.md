# ARION ALPHA 1 — Worklog (recovered)

> NOTE: the sandbox rootfs was recycled at ~2026-09-20 18:20 UTC (fresh rootfs).
> The previous worklog (GH-1, GH-2, GH-2-b, FA-1 sections) was sandbox-local and
> was lost with it. All CODE survived via GitHub (commits up to f914b49).
> This file was rebuilt from the recovery state; history below is summarized.

---

Task ID: FA-1 (recap — completed before sandbox reset)
Agent: main (Z.ai Code)
Task: Persian-phase upgrade — ParsBench corpus pipeline, normalization, 239-prompt
benchmark, Kaggle GPU training pipeline + CPU fallback, GitHub Actions orchestration.

Work Log (summary of what was DONE and survived on GitHub):
- GGUF pipeline verified healthy (Actions smoke run 35505842466 green; Release v1.0.0 assets intact)
- KAGGLE_API_TOKEN stored as GitHub Secret (never committed/printed); account reutmc
- training/persian_norm.py (code-safe Persian normalization, 10/10 tests)
- training/prepare_persian_dataset.py → MEASURED: 99,994 raw → 16 rejected → 98,238
  (exact-dedup) → 64,310 (33,928 near-dups removed, MinHash-LSH) → 44,908 accepted →
  40,382 train / 4,526 val (deterministic, leak-free)
- evaluation/persian_fluency_benchmark.jsonl (239 prompts, 12 categories, fa 230 / ar 6 / en 3)
- evaluation/persian_langid.py (fasttext optional + function-word heuristic), persian_metrics.py,
  run_persian_benchmark.py (transformers + llama-server, CPU-safe auto device)
- training/train_kaggle.py (GPU trainer: CUDA-required unless --allow-cpu, bf16/fp16 auto,
  OOM-adaptive bs/ga, seq-len benchmark mode, reproducibility metadata)
- kaggle/kernel_baseline.py (GPU: seqbench+baseline+smoke), kernel_train.py (GPU: full train
  → merge → GGUF F16/Q8_0/Q4_K_M → llama-server eval; hard-fails w/o CUDA),
  kernel_baseline_cpu.py (explicit CPU fallback), kernel_bench_cpu.py (bench-only)
- training/kaggle_pipeline.py (bundle/dataset-push/kernel-push/monitor/fetch; run-all)
- .github/workflows/persian-kaggle.yml (dispatch: baseline-cpu|baseline|train|all)
- Kaggle private dataset reutmc/arion-persian-data v7 (1.67GB: jsonl + scripts + base model + llama.cpp)
- Kaggle private dataset reutmc/arion-persian-data v7 (1.67GB: jsonl + scripts + base model + llama.cpp)
- DEBUGGED via real runs: GPU+internet gated on phone verification (server silently gives
  CPU-only image torch 2.10.0+cpu, no DNS); dataset mount layout changed to
  /kaggle/input/datasets/<owner>/<slug> → dynamic resolution added; subdirs need
  --dir-mode zip; archives auto-extract server-side; peft 0.19 vs torchao 0.10 conflict →
  offline `pip uninstall torchao` in kernels; /kaggle/input is READ-ONLY → llama.cpp must
  build in /kaggle/temp; smoke path PROVEN end-to-end on CPU (train 61 min → merge 20s →
  GGUF F16+Q4_K_M → llama-cli Persian smoke OK)
- LLM booster: 687 conversations generated before throttling — LOST in sandbox reset
  (regenerable via training/booster_gen.mjs; hand-authored 77 seeds are committed)

Stage Summary:
- All Persian-phase CODE + data pipeline is on GitHub; Kaggle dataset + kernels live
- Only external blocker for GPU training: Kaggle account phone verification
- Baseline benchmark (CPU) was running on arion-cpu-test1 v8 at reset time (Kaggle-side,
  unaffected by reset)

---
Task ID: FA-2
Agent: main (Z.ai Code)
Task: Post-reset recovery + baseline benchmark + booster regeneration + dashboard rebuild

Work Log:
- Fresh rootfs detected (all local files gone); cloned ReutMC/AI into /home/z/my-project
- Reinstalled kaggle CLI (2.2.4) + persisted token to ~/.kaggle/access_token (chmod 600)
- Rebuilt dashboard at dashboard/ (Next.js 16, /api/status reads repo artifacts);
  dev server on :3000 (bun run dev inside dashboard/); committed as f914b49
- arion-cpu-test1 v8 (bench-only kernel, dataset v7 with CPU-safe runner) RUNNING
- Booster regeneration restarted (LLM gateway config lost its token in reset — 401
  "missing X-Token header"; retrying periodically)
- Rebuilt this worklog

Stage Summary:
- Recovery complete: repo + Kaggle + dashboard operational
- Next: (a) fetch baseline metrics when v8 completes → artifacts/persian/baseline_metrics.json
  → dashboard BASE column live; (b) rebuild booster data when LLM works; (c) dataset v9 with
  booster; (d) user verifies Kaggle phone → dispatch GPU train kernel → ARION GGUF + metrics;
  (e) Release v1.1 via scripts/release_persian.sh
