# LICENSES.md — ARION ALPHA 1

## Project code & training data
- **Apache License 2.0** — all scripts, pipeline code, documentation, and all authored/generated
  training datasets in `datasets/` are original work created for this project and are released
  under Apache-2.0.

## Base model
- **Qwen3-0.6B** — Copyright (c) Alibaba Cloud — **Apache License 2.0**.
  Source: https://huggingface.co/Qwen/Qwen3-0.6B
  Fine-tuning (LoRA merge) produces a derivative model; the resulting
  Arion Alpha 1 weights are distributed under Apache-2.0 as well.

## Tooling used (not redistributed in model package)
- **PyTorch** — BSD-style license.
- **Hugging Face Transformers / PEFT** — Apache-2.0.
- **llama.cpp** — MIT (conversion + quantization tooling).
- **Ollama** — MIT (deployment runtime; user-installed).
- **tinycss2 / html5lib / html5 ever / psutil** — BSD/MIT/Apache family (evaluation only).

## Google Fonts
Font names (Vazirmatn, Estedad, Inter, …) appear inside code examples as *usage instructions*
(URL linking patterns). No font software is bundled. Fonts themselves carry their own licenses
(OFL for Vazirmatn/Estedad/Inter) and are loaded from Google Fonts at runtime by the generated
websites.

## Summary for model users
You may use, modify, and redistribute Arion Alpha 1 (including commercially) under Apache-2.0,
provided you include the license notice and state changes. The model is provided "AS IS",
without warranty of any kind.
