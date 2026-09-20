# DATA_SOURCES.md — ARION ALPHA 1

Every byte of training data in this project is **generated or authored locally** for this project.
No copyrighted websites were scraped, no terms of service violated, no private data used.

| # | Source | Type | URL / Origin | License | Language | Purpose | Approx. size | Preprocessing |
|---|--------|------|--------------|---------|----------|---------|--------------|---------------|
| 1 | `curated_fa.jsonl`, `curated_fa2.jsonl` | Human-curated (authored by dataset engineers for this project) | local authoring | Apache-2.0 (this project) | Persian | Persian HTML/CSS/JS explanations, Persian site building (RTL), Persian debugging Q&A | 90 examples | schema validation, dedup, secret scan |
| 2 | `curated_en.jsonl`, `curated_en2.jsonl` | Human-curated | local authoring | Apache-2.0 | English | English HTML/CSS/JS deep explanations, builds, debugging | 90 examples | same |
| 3 | `curated_bi.jsonl` | Human-curated | local authoring | Apache-2.0 | Persian+English mixed | language-following behavior, explicit switch requests, EN↔FA terminology | 45 examples | same |
| 4 | `curated_pro.jsonl`, `curated_teach.jsonl` | Human-curated | local authoring | Apache-2.0 | mixed | multi-file projects, Google Fonts usage, CSS tricks, JS component patterns, teaching progressions, refactoring | 90 examples | same |
| 5 | `template_data.jsonl` | Programmatic synthesis (`scripts/gen_template_data.py`) | generated locally, seeded (4217) | Apache-2.0 | Persian + English | parameterized complete pages (8 palettes × fonts × copy), CSS tricks, fact-based explanations, debugging templates, glossary, multi-file scaffolds | 377 examples | same |
| 6 | `llm_data.jsonl` | LLM-assisted synthesis | generated locally via an internal LLM API (GLM), strictly task-prompted for this project, human-reviewed schemas | Apache-2.0 (this project) | Persian, English, mixed | additional variety in the same categories; every row generated from a controlled prompt that forbids placeholders and copyrighted content | ~100-400 examples | JSON extraction, schema validation, dedup, secret scan |

## Third-party assets referenced (NOT distributed as data)

- **Qwen3-0.6B** base model weights — Hugging Face `Qwen/Qwen3-0.6B` — Apache-2.0. Used as the
  pretrained starting point; fine-tuning adapts it. The tokenizer ships with the model.
- **Google Fonts** — mentioned inside *code examples* (Vazirmatn, Estedad, Inter, Manrope, …) as
  factual usage patterns (preconnect + stylesheet links). No font binaries are included in the
  dataset or the model package; examples merely show how to link them.
- **llama.cpp** — conversion tooling only (Apache-2.0 / MIT). Not training data.

## Data quality pipeline (training/prepare_dataset.py)

1. **Schema validation** — message roles/order, non-empty content, expected system prompt.
2. **Quality filters** — min answer length, max lengths, trigram repetition < 0.45.
3. **Secret scanning** — regex detection of API keys (OpenAI-style `sk-`, AWS `AKIA`, GitHub
   tokens, `Bearer` headers, generic `api_key=...` literals), PEM private keys; matched rows are
   rejected. (All originals were authored safe; this is a defense-in-depth guarantee.)
4. **Language consistency** — Persian examples must contain Persian script; English examples must
   not be majority-Persian.
5. **Deduplication** — MD5 of normalized (user ‖ answer) pairs; 58 exact duplicates removed.
6. **Splits** — 90 % train / 5 % validation / 5 % test (shuffled, seed 4217). The test set is
   never seen during training or hyperparameter selection.

## Synthetic-data ethics statement

- LLM-assisted rows were generated with prompts that require complete, working, original code and
  forbid placeholders (`TODO`, “implement yourself”), credentials, and copyrighted text.
- Persian text is authored/validated to read as natural Iranian developer Persian (not machine-
  translated boilerplate); technical terms stay in English where real developers would keep them.
