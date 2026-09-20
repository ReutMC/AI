<div align="center">

# 🦁 ARION ALPHA 1

**آریون آلفا ۱ — مدل زبانی دوزبانه‌ی فارسی/انگلیسی، متخصص توسعه‌ی وب**
**A compact, genuinely-trained bilingual (Persian 🇮🇷 / English 🇬🇧) Web-Development LLM**

[![License](https://img.shields.io/badge/license-Apache--2.0-green)](#-license--مجوز)
[![Params](https://img.shields.io/badge/params-0.596B-orange)](#-key-facts--مشخصات-کلیدی)
[![Base](https://img.shields.io/badge/base-Qwen3--0.6B-blue)](https://huggingface.co/Qwen/Qwen3-0.6B)
[![GGUF](https://img.shields.io/badge/GGUF-Q4__K__M_|_Q8__0-red)](#-download--دانلود)
[![Ollama](https://img.shields.io/badge/Ollama-ready-purple)](#-quickstart--اجرای-سریع)
[![Trained on](https://img.shields.io/badge/trained%20on-CPU%20only%20(4GB%20RAM)-lightgrey)](#-training--آموزش)

</div>

---

ARION ALPHA 1 is a **real decoder-only Transformer** (~0.6 B parameters) fine-tuned with **LoRA SFT** on a
purpose-built corpus of Persian/English web-development instruction data, converted to standard
**GGUF** weights and packaged for **Ollama** and **llama.cpp**.

It is **not** a prompt wrapper, not a hardcoded chatbot, and not fake weights — it ships real
trainable weight deltas produced by gradient descent, a full training pipeline, evaluation reports
and checksums. Everything was trained **locally on a CPU-only, 4 GB RAM machine** — no GPU, no cloud.

<div dir="rtl">

**آریون آلفا ۱** یک مدل ترنسفورمر واقعی و decode-only با حدود ۰٫۶ میلیارد پارامتر است که با روش
**LoRA SFT** روی پیکره‌ای اختصاصی از دستورالعمل‌های توسعه‌ی وب به دو زبان **فارسی و انگلیسی**
فاین‌تیون شده و به فرمت استاندارد **GGUF** تبدیل و برای **Ollama** و **llama.cpp** بسته‌بندی شده است.

این مدل نه یک Wrapper ساده‌ی پرامپت است، نه یک چت‌بات هاردکدشده و نه وزن‌های تقلبی —
دلتای وزن‌های واقعیِ حاصل از گرادیان دیسنت، پایپ‌لاین کامل آموزش، گزارش‌های ارزیابی و چک‌سام‌ها
همگی در این مخزن موجودند. همه‌ی آموزش **به‌صورت محلی و فقط روی CPU با ۴ گیگابایت رم** انجام شده —
بدون GPU و بدون فضای ابری.

</div>

---

## 📑 Table of Contents / فهرست مطالب

| English | فارسی |
|---|---|
| [Key facts](#-key-facts--مشخصات-کلیدی) | [مشخصات کلیدی](#-key-facts--مشخصات-کلیدی) |
| [Capabilities](#-capabilities--قابلیت‌ها) | [قابلیت‌ها](#-capabilities--قابلیت‌ها) |
| [Download](#-download--دانلود) | [دانلود](#-download--دانلود) |
| [Quickstart (Ollama)](#-quickstart--اجرای-سریع) | [اجرای سریع](#-quickstart--اجرای-سریع) |
| [llama.cpp / Transformers](#-llamacpp--transformers) | [اجرای بدون Ollama](#-llamacpp--transformers) |
| [Repository layout](#-repository-layout--ساختار-مخزن) | [ساختار مخزن](#-repository-layout--ساختار-مخزن) |
| [Training](#-training--آموزش) | [آموزش](#-training--آموزش) |
| [Evaluation](#-evaluation--ارزیابی) | [ارزیابی](#-evaluation--ارزیابی) |
| [Honest limitations](#%EF%B8%8F-honest-limitations--محدودیت‌های-واقعی) | [محدودیت‌های واقعی](#%EF%B8%8F-honest-limitations--محدودیت‌های-واقعی) |
| [Rebuild from source](#-rebuild-from-source--بازسازی-از-منبع) | [بازسازی از منبع](#-rebuild-from-source--بازسازی-از-منبع) |
| [License](#-license--مجوز) | [مجوز](#-license--مجوز) |

---

## 🔑 Key facts / مشخصات کلیدی

| Item | Value |
|---|---|
| Model name | **Arion Alpha 1** |
| Ollama tag | `arion-alpha-1` |
| Architecture | Qwen3 decoder-only Transformer (GQA, SwiGLU, RoPE) |
| Base model | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — Apache-2.0 |
| Total parameters | **~0.596 B** |
| Trainable params (LoRA) | **7.569 M** (r=12, α=24 → 1.25 % of total) |
| Method | LoRA SFT → merged → GGUF (F16 → Q8_0 → Q4_K_M) |
| Tokenizer | Qwen3 BPE (151 936 vocab), multilingual incl. Persian |
| Context (training) | 192–256 tokens (RAM-constrained) |
| Context (inference) | up to 32 K (base-model native) |
| Languages | Persian (fa) + English (en), natural code-switching |
| Specialization | HTML, CSS, JavaScript, modern frontend, RTL, Google Fonts, teaching |
| Trained on | 2-core CPU (AVX-512 / AMX-BF16), 4.36 GB RAM, **no GPU** — 167.5 minutes |
| License | Apache-2.0 (code + model); dataset licenses in [`LICENSES.md`](LICENSES.md) |

<div dir="rtl">

| مورد | مقدار |
|---|---|
| نام مدل | **Arion Alpha 1** (آریون آلفا ۱) |
| تگ Ollama | `arion-alpha-1` |
| معماری | ترنسفورمر decode-only خانواده‌ی Qwen3 (GQA، SwiGLU، RoPE) |
| مدل پایه | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — Apache-2.0 |
| تعداد کل پارامترها | **~۰٫۵۹۶ میلیارد** |
| پارامترهای قابل‌آموزش (LoRA) | **۷٫۵۶۹ میلیون** (r=12، α=24 → ۱٫۲۵٪ کل) |
| روش | LoRA SFT → ادغام وزن‌ها → GGUF (F16 → Q8_0 → Q4_K_M) |
| توکن‌ایزر | Qwen3 BPE با ۱۵۱٬۹۳۶ توکن، چندزبانه شامل فارسی |
| طول زمینه در آموزش | ۱۹۲–۲۵۶ توکن (محدودیت رم) |
| طول زمینه در استنتاج | تا ۳۲ هزار توکن (سقف ذاتی مدل پایه) |
| زبان‌ها | فارسی + انگلیسی، با جابه‌جایی طبیعی بین دو زبان |
| تخصص | HTML، CSS، جاوااسکریپت، فرانت‌اند مدرن، RTL، گوگل‌فونت، آموزش مفاهیم |
| سخت‌افزار آموزش | CPU دو هسته‌ای، ۴٫۳۶ گیگ رم، **بدون GPU** — ۱۶۷٫۵ دقیقه |
| مجوز | Apache-2.0 (کد و مدل)؛ مجوز داده‌ها در [`LICENSES.md`](LICENSES.md) |

</div>

### What it can do / چه کارهایی از آن برمی‌آید

- Answer **Persian questions in Persian**, **English questions in English**, and honor explicit
  “answer in Persian/English” requests — no random language switching.
- Generate **complete single-file pages** and **multi-file projects** (`index.html`, `css/style.css`,
  `js/app.js`) with correct relative references.
- Explain HTML/CSS/JS concepts from **beginner to expert**, including *why* solutions work.
- Debug layouts (flexbox / grid / position / z-index / overflow), JS errors, responsive and RTL issues.
- Build Persian/RTL websites with `dir="rtl"`, `lang="fa"` and Vazirmatn / Estedad Google-Fonts setups.
- Use **modern CSS** (custom properties, `clamp()`, grid/flex, container queries, glassmorphism,
  skeleton loaders, dark mode via `prefers-color-scheme` / `data-theme`) and **modern JS**
  (async/await, fetch, IntersectionObserver, event delegation, modules, localStorage).

<div dir="rtl">

- به **سؤال فارسی، پاسخ فارسی** و به **سؤال انگلیسی، پاسخ انگلیسی** می‌دهد و درخواستِ صریحِ
  «به فارسی/انگلیسی جواب بده» را رعایت می‌کند — بدون تعویض تصادفی زبان.
- تولید **صفحات تک‌فایلی کامل** و **پروژه‌های چندفایلی** (`index.html`، `css/style.css`، `js/app.js`)
  با ارجاع‌های نسبیِ صحیح.
- توضیح مفاهیم HTML/CSS/JS از سطح **مبتدی تا پیشرفته**، همراه با *دلیل* درستیِ راه‌حل.
- دیباگ چیدمان (flexbox / grid / position / z-index / overflow)، خطاهای JS، مشکلات ریسپانسیو و RTL.
- ساخت وب‌سایت‌های فارسی/راست‌به‌چپ با `dir="rtl"`، `lang="fa"` و راه‌اندازی فونت‌های وزیرمتن/استعداد.
- استفاده از **CSS مدرن** (متغیرهای سفارشی، `clamp()`، grid/flex، container queries، گلس‌مورفیزم،
  اسکلتون‌لودر، دارک‌مود با `prefers-color-scheme` / `data-theme`) و **JS مدرن**
  (async/await، fetch، IntersectionObserver، event delegation، ماژول‌ها، localStorage).

</div>

---

## ⬇️ Download / دانلود

All weights are attached to the [**GitHub Release v1.0.0**](https://github.com/ReutMC/AI/releases/tag/v1.0.0):

| Asset | Size | Purpose |
|---|---:|---|
| [`arion-alpha-1-Q4_K_M.gguf`](https://github.com/ReutMC/AI/releases/download/v1.0.0/arion-alpha-1-Q4_K_M.gguf) | ~397 MB | **Recommended** — 4-bit quant, runs on modest CPUs |
| [`arion-alpha-1-Q8_0.gguf`](https://github.com/ReutMC/AI/releases/download/v1.0.0/arion-alpha-1-Q8_0.gguf) | ~639 MB | 8-bit quant, higher fidelity |
| [`arion-alpha-1-model.zip`](https://github.com/ReutMC/AI/releases/download/v1.0.0/arion-alpha-1-model.zip) | ~1.0 GB | Full bundle: GGUF (Q4_K_M + Q8_0) + Modelfile + model card |
| [`arion-alpha-1-source.zip`](https://github.com/ReutMC/AI/releases/download/v1.0.0/arion-alpha-1-source.zip) | ~5.6 MB | Complete training source snapshot |

SHA-256 checksums for every artifact: [`artifacts/checksums.sha256`](artifacts/checksums.sha256).
The trained **LoRA adapter** (`adapter_model.safetensors`, 29 MB) lives in the repo at
[`model/arion-alpha-1-lora/`](model/arion-alpha-1-lora/) — that is the actual trained delta.

<div dir="rtl">

تمام وزن‌ها به [**ریلیز v1.0.0**](https://github.com/ReutMC/AI/releases/tag/v1.0.0) پیوست شده‌اند:

| فایل | حجم | کاربرد |
|---|---:|---|
| [`arion-alpha-1-Q4_K_M.gguf`](https://github.com/ReutMC/AI/releases/download/v1.0.0/arion-alpha-1-Q4_K_M.gguf) | ~۳۹۷ MB | **پیشنهادی** — کوانت ۴-بیتی، مناسب CPUهای معمولی |
| [`arion-alpha-1-Q8_0.gguf`](https://github.com/ReutMC/AI/releases/download/v1.0.0/arion-alpha-1-Q8_0.gguf) | ~۶۳۹ MB | کوانت ۸-بیتی، دقت بالاتر |
| [`arion-alpha-1-model.zip`](https://github.com/ReutMC/AI/releases/download/v1.0.0/arion-alpha-1-model.zip) | ~۱٫۰ GB | بسته‌ی کامل: GGUF (Q4_K_M + Q8_0) + Modelfile + کارت مدل |
| [`arion-alpha-1-source.zip`](https://github.com/ReutMC/AI/releases/download/v1.0.0/arion-alpha-1-source.zip) | ~۵٫۶ MB | اسنپ‌شات کامل سورس آموزش |

چک‌سام SHA-256 همه‌ی خروجی‌ها: [`artifacts/checksums.sha256`](artifacts/checksums.sha256).
آداپتور **LoRA**ی آموزش‌دیده (`adapter_model.safetensors`، ۲۹ MB) در مسیر
[`model/arion-alpha-1-lora/`](model/arion-alpha-1-lora/) داخل خود مخزن است —
همان دلتای واقعیِ آموزش‌دیده.

</div>

---

## 🚀 Quickstart / اجرای سریع

### Ollama

```bash
# 1) download the Q4_K_M gguf from the release page, then:
ollama create arion-alpha-1 -f Modelfile     # Modelfile is in ./ollama/
ollama run arion-alpha-1
```

Example session / نمونه گفتگو:

```text
>>> Build a modern landing page hero with glassmorphism.
→ complete HTML/CSS in English …

>>> یک فرم تماس ریسپانسیو با اعتبارسنجی بساز
→ کد کامل به همراه توضیحات فارسی …
```

> The bundled `Modelfile` sets temperature 0.7, top_p 0.9, repeat_penalty 1.05,
> Qwen3 chat template and a bilingual web-dev system prompt.

### llama.cpp

```bash
llama-cli -m arion-alpha-1-Q4_K_M.gguf \
  -c 2048 --temp 0.7 --repeat-penalty 1.05 \
  -p '<|im_start|>user\nWrite a CSS flexbox cheat sheet.<|im_end|>\n<|im_start|>assistant\n'
```

### Transformers + LoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "model/arion-alpha-1-lora")
tok   = AutoTokenizer.from_pretrained("model/arion-alpha-1-lora")

msgs = [{"role": "user", "content": "یک کارت محصول با CSS بساز"}]
print(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
```

<div dir="rtl">

### اجرا با Ollama — خلاصه‌ی فارسی

۱. فایل `arion-alpha-1-Q4_K_M.gguf` را از صفحه‌ی ریلیز دانلود کنید و کنار `Modelfile` قرار دهید.
۲. سپس:

```bash
ollama create arion-alpha-1 -f Modelfile
ollama run arion-alpha-1
```

۳. حالا می‌توانید فارسی یا انگلیسی بپرسید؛ مثلاً:
«یک صفحه‌ی فرود مدرن برای استارتاپ با HTML و CSS بساز» — پاسخ کامل همراه با کد دریافت می‌کنید.

فایل `Modelfile` شامل قالب گفتگوی Qwen3، پارامترهای نمونه‌برداری
(temperature=0.7، top_p=0.9، repeat_penalty=1.05) و یک پرامپت سیستمی دوزبانه‌ی تخصصی توسعه‌ی وب است.

</div>

---

## 📁 Repository layout / ساختار مخزن

```
.
├── README.md                     # this file (EN + FA)
├── LICENSES.md                   # aggregated licenses (code, base model, datasets)
├── DATA_SOURCES.md               # dataset provenance & privacy notes
├── requirements.txt              # python deps for the training pipeline
├── configs/                      # train_config.json (+ smoke config)
├── training/
│   ├── prepare_dataset.py        # clean → dedupe → split (90/5/5)
│   └── train_sft.py              # LoRA SFT trainer (CPU, bf16, checkpoints)
├── scripts/                      # env setup, hardware probe, data gen, packaging
├── conversion/
│   ├── merge_lora.py             # LoRA → merged full weights
│   └── convert_gguf.sh           # HF → GGUF F16 → Q8_0 / Q4_K_M (needs llama.cpp)
├── inference/
│   ├── chat.py                   # CLI chat (transformers)
│   ├── generate_web.py           # one-shot web page generator
│   └── server.py                 # tiny OpenAI-compatible HTTP server
├── evaluation/
│   ├── run_eval.py               # automated eval (transformers backend)
│   ├── run_eval_llama.py         # automated eval (llama.cpp backend)
│   └── prompts_bilingual.json / prompts_webdev.json
├── datasets/
│   ├── generated/                # 9 curated/synthetic JSONL corpora
│   └── processed/                # cleaned train/val/test + stats
├── model/
│   └── arion-alpha-1-lora/       # ★ the trained LoRA adapter + tokenizer
├── ollama/
│   └── Modelfile                 # ready-to-use Ollama definition
├── docs/
│   ├── MODEL_CARD.md
│   └── TRAINING_REPORT.md
├── tests/                        # selftest.py, validate_datasets.py
└── artifacts/
    ├── FINAL_REPORT.md           # full training & delivery report
    ├── train_summary.json        # final metrics of the run
    ├── eval_report.json          # automated eval results
    ├── eval_history/             # eval snapshots over time
    ├── hardware_report.json      # auto-detected training hardware
    ├── checksums.sha256          # sha256 of every shipped artifact
    └── llama_smoke.txt           # llama.cpp smoke test transcript
```

> The large binaries (GGUF files, model ZIP, merged weights) are **not** stored in git —
> download them from the [release](https://github.com/ReutMC/AI/releases/tag/v1.0.0).
> `conversion/` intentionally does not vendor llama.cpp; `convert_gguf.sh` clones/uses upstream llama.cpp.

---

## 🏋️ Training / آموزش

Real gradient-descent training, fully reproducible from this repo:

| Metric | Value |
|---|---:|
| Optimizer steps | **256** |
| Wall time | **167.55 min** (CPU-only, 2 threads) |
| Examples trained on | 316 |
| Tokens seen | **166 128** |
| Best validation loss | **1.2707** |
| Throughput | 0.305 examples/s |
| Stop reason | `max_epochs_reached` |
| LoRA | r=12, α=24, bf16, grad-accum 8 |

**Dataset** (after cleaning — 58 exact duplicates and 7 language-mismatches removed):
**638 accepted** examples → splits **576 / 31 / 31** (train/val/test, 90/5/5).
Train languages: **en 320 · fa 219 · mixed 37** across 23 categories
(html / css / js / build / debug / teaching / pro-patterns / bilingual-switching …).
Provenance & licenses: [`DATA_SOURCES.md`](DATA_SOURCES.md) — 100 % locally authored/generated, secret-scanned.

<div dir="rtl">

## آموزش — خلاصه‌ی فارسی

آموزش واقعی با گرادیان دیسنت انجام شده و از همین مخزن کاملاً قابل تکرار است:

| سنجه | مقدار |
|---|---:|
| گام‌های بهینه‌سازی | **۲۵۶** |
| زمان کل | **۱۶۷٫۵۵ دقیقه** (فقط CPU، ۲ رشته) |
| نمونه‌های آموزش‌دیده | ۳۱۶ |
| توکن‌های دیده‌شده | **۱۶۶٬۱۲۸** |
| بهترین خطای اعتبارسنجی | **۱٫۲۷۰۷** |
| سرعت | ۰٫۳۰۵ نمونه بر ثانیه |
| دلیل توقف | رسیدن به سقف epochها |

**مجموعه‌داده** (پس از پاک‌سازی — حذف ۵۸ تکرار دقیق و ۷ ناسازگاری زبانی):
**۶۳۸ نمونه‌ی پذیرفته‌شده** → تفکیک **۵۷۶ / ۳۱ / ۳۱** (آموزش/اعتبارسنجی/آزمون با نسبت ۹۰/۵/۵).
زبان‌های بخش آموزش: **انگلیسی ۳۲۰ · فارسی ۲۱۹ · ترکیبی ۳۷** در ۲۳ دسته‌ی موضوعی
(html / css / js / ساخت صفحه / دیباگ / آموزش / الگوهای حرفه‌ای / جابه‌جایی دوزبانه و …).
منشأ و مجوز داده‌ها: [`DATA_SOURCES.md`](DATA_SOURCES.md) — صددرصد تولید/نگارش محلی و اسکن‌شده از نظر اسرار.

</div>

---

## 🇮🇷 Persian phase (v1.1 — in progress)

The next training round makes ARION speak **natural Persian** as its primary
objective (before any further HTML specialization):

- **Source dataset**: [`ParsBench/PersianSyntheticQA`](https://huggingface.co/datasets/ParsBench/PersianSyntheticQA) (Apache-2.0, ~100k Persian QA conversations, 50 domains) + locally authored booster conversations (colloquial / multi-turn / language-switching / fa+en technical).
- **Measured preprocessing** (`training/prepare_persian_dataset.py`, seed 42, fully reproducible):

| Stage | Count |
|---|---:|
| Raw source rows | **99 994** |
| Rejected (structure/quality/language) | 16 |
| Exact duplicates removed | 1 756 |
| Near-duplicates removed (MinHash-LSH, Jaccard ≥ 0.85) | **33 928** |
| Accepted after quality-ranked, domain-balanced selection | **44 908** |
| Split (deterministic, near-dup-group isolated) | **40 382 train / 4 526 validation** |

- **Persian normalization** with code/URL protection: Arabic Yeh→Persian Yeh, Arabic Kaf→Keh, teh-marbuta→heh, Arabic-Indic→Persian digits, bidi-junk removal — code fences, URLs, paths and English identifiers are never touched (`training/persian_norm.py`, 10/10 unit tests).
- **Language control**: the corpus and the 239-prompt benchmark
  (`evaluation/persian_fluency_benchmark.jsonl`, 12 categories) explicitly train & test
  *Persian in → Persian out*, *Arabic request → Arabic out*, *English request → English out*,
  with function-word-based language ID (not mere character overlap).
- **Kaggle GPU training** replaces the CPU pipeline: seq-len benchmark (2048/3072/4096),
  LoRA r=16 α=32 lr=1e-4 (unchanged from v1.0 for a controlled comparison),
  `max_epochs 2`, OOM-adaptive batch sizing, explicit no-CPU-failure guard.
- **BASE vs ARION**: the same 239-prompt benchmark is run on the base model
  *before* training and on ARION (GGUF) *after* training. **Definitive BASE result
  (measured, thinking disabled, CPU run):** persian_response_rate **98.3%**,
  language_compliance **97.9%**, arabic_leakage 0.87%, **repetition 28.0%** (67/239
  degenerate answers — the main quality problem ARION must fix), malformed 1.3%.
  Raw evidence: `artifacts/persian/baseline_metrics.json` + full responses.
- Orchestration: `training/kaggle_pipeline.py` + `.github/workflows/persian-kaggle.yml`
  (Kaggle credentials only via `KAGGLE_API_TOKEN` secret — never committed).

<div dir="rtl">

## فاز فارسی (نسخهٔ ۱٫۱ — در حال اجرا)

هدف فاز بعدی، **فارسی روان و طبیعی** است؛ قبل از هر تخصص‌دهی دیگری:

- **منبع داده**: دیتاست [`ParsBench/PersianSyntheticQA`](https://huggingface.co/datasets/ParsBench/PersianSyntheticQA) (آپاچی ۲٫۰، حدود ۱۰۰ هزار گفت‌وگوی پرسش‌وپاسخ فارسی در ۵۰ حوزه) به‌همراه گفت‌وگوهای محاوره‌ای/چندنوبته/جابه‌جایی زبان که برای این پروژه نوشته شده است.
- **پیش‌پردازش اندازه‌گیری‌شده** (بذر ۴۲، کاملاً قابل تکرار): از ۹۹٬۹۹۴ ردیف خام، ۳۳٬۹۲۸ شبه‌تکرار و ۱٬۷۵۶ تکرار دقیق حذف شد و **۴۴٬۹۰۸ نمونه** با توازن دامنه‌ای پذیرفته شد؛ تفکیک قطعی **۴۰٬۳۸۲ آموزش / ۴٬۵۲۶ اعتبارسنجی** بدون نشت شبه‌تکرارها.
- **نرمال‌سازی فارسی** با محافظت از کد و URL (ی عربی→فارسی، ک عربی→کاف فارسی، ة→ه، ارقام عربی→فارسی، حذف نویسه‌های کنترلی) — ۱۰/۱۰ تست واحد.
- **کنترل زبان**: مجموعهٔ ارزیابی ۲۳۹ پرامپتی در ۱۲ دسته؛ فارسی→فارسی، درخواست عربی→عربی، درخواست انگلیسی→انگلیسی؛ شناسایی زبان با واژه‌های نقش‌نما نه صرفاً هم‌پوشانی حروف.
- **آموزش روی GPU کاگل** جایگزین CPU شد: بنچمارک طول توالی (۲۰۴۸/۳۰۷۲/۴۰۹۶)، LoRA r=16 α=32 با lr=1e-4 (بدون تغییر نسبت به نسخهٔ ۱٫۰ برای مقایسهٔ کنترل‌شده)، دو epoch، تنظیم خودکار batch در OOM.
- **مقایسهٔ BASE و ARION**: همین بنچمارک قبل و بعد از آموزش روی مدل پایه و مدل نهایی اجرا می‌شود. **نتیجهٔ قطعی BASE (اندازه‌گیری‌شده):** نرخ پاسخ فارسی ۹۸٫۳٪، انطباق زبان ۹۷٫۹٪، نشتی عربی ۰٫۸۷٪، و **تکرار ۲۸٪** (۶۷ پاسخ واژگون از ۲۳۹ — مشکل اصلی کیفیت که آموزش باید حل کند). شواهد خام در `artifacts/persian/`.
- مدارک کلیدی: `KAGGLE_API_TOKEN` فقط به‌صورت GitHub Secret — هرگز در کد کامیت نمی‌شود.

</div>

---

## 📊 Evaluation / ارزیابی

Automated eval (20 prompts per set, llama.cpp backend on Q4_K_M) from
[`artifacts/eval_report.json`](artifacts/eval_report.json):

| Metric | webdev | bilingual |
|---|---:|---:|
| Code production rate | 0.80 | 0.80 |
| Instruction following | 0.90 | **1.00** |
| Relevance rate | 1.00 | 0.90 |
| Syntax-valid rate (of checked) | 1.00 | 0.94 |
| Language correct | — | **0.75** |
| Mean repetition | 0.034 | 0.09 |

Additional snapshots over time: [`artifacts/eval_history/`](artifacts/eval_history/).

<div dir="rtl">

ارزیابی خودکار (۲۰ پرامپت در هر مجموعه، بک‌اند llama.cpp روی Q4_K_M)
طبق [`artifacts/eval_report.json`](artifacts/eval_report.json):

| سنجه | توسعه‌ی وب | دوزبانه |
|---|---:|---:|
| نرخ تولید کد | ۰٫۸۰ | ۰٫۸۰ |
| پیروی از دستور | ۰٫۹۰ | **۱٫۰۰** |
| نرخ ارتباط‌مندی | ۱٫۰۰ | ۰٫۹۰ |
| نرخ سینتکس معتبر | ۱٫۰۰ | ۰٫۹۴ |
| صحت زبان پاسخ | — | **۰٫۷۵** |
| میانگین تکرار | ۰٫۰۳۴ | ۰٫۰۹ |

گزارش‌های دوره‌ای دیگر: [`artifacts/eval_history/`](artifacts/eval_history/).

</div>

---

## ⚠️ Honest limitations / محدودیت‌های واقعی

1. **CPU-only LoRA SFT** (256 steps over a few hundred curated/synthetic examples):
   strong *style & format* adaptation on top of Qwen3-0.6B's pretrained knowledge —
   **not** a full pretraining. Treat it as a focused domain-tuning proof, in the ~0.6B class.
2. **Training context capped at ~224 tokens** by the 4 GB RAM ceiling; long project answers
   were head-truncated during training (code fences auto-closed).
3. **Q4_K_M quantization** trades some accuracy for a 3.4× smaller footprint (Q8_0 included).
4. **Domain-bound by design**: politely stays in web development; not a general-purpose assistant.
5. Inference speed on the training-class machine is ~0.8–1 tok/s; any modern laptop/desktop is faster.

<div dir="rtl">

۱. **فاین‌تیون LoRA فقط روی CPU** (۲۵۶ گام روی چند صد نمونه‌ی منتخب/سنتزشده):
   تطبیق قوی *سبک و قالبِ* پاسخ روی دانش از پیش‌آموخته‌ی Qwen3-0.6B — **پیش‌آموزش کامل نیست**؛
   یک اثبات مفهومِ تیون دامنه‌ای در کلاس ~۰٫۶ میلیارد پارامتر بدانید.
۲. **سقف طول زمینه در آموزش ~۲۲۴ توکن** بود (محدودیت ۴ گیگ رم)؛ پاسخ‌های بلندِ پروژه‌محور
   در آموزش سربریده می‌شدند (بلاک‌های کد به‌صورت خودکار بسته می‌شدند).
۳. **کوانتیزه‌سازی Q4_K_M** برای ۳٫۴ برابر کوچک‌شدن حجم، اندکی دقت را کم می‌کند (نسخه‌ی Q8_0 هم موجود است).
۴. **محدود به حوزه‌ی توسعه‌ی وب** است و مؤدبانه از خارج آن پرهیز می‌کند؛ دستیار همه‌منظوره نیست.
۵. سرعت استنتاج روی ماشینِ هم‌کلاسِ ماشینِ آموزش حدود ۰٫۸ تا ۱ توکن بر ثانیه است؛
   روی لپ‌تاپ/دسکتاپ‌های امروزی به‌مراتب سریع‌تر خواهد بود.

</div>

---

## 🛠 Rebuild from source / بازسازی از منبع

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) probe hardware → writes configs used by the trainer
python scripts/hardware_probe.py

# 2) build the dataset (clean → dedupe → split)
python training/prepare_dataset.py

# 3) LoRA SFT (CPU, time-budgeted, checkpointed)
python training/train_sft.py --config configs/train_config.json

# 4) merge + convert to GGUF (needs a llama.cpp checkout; script guides you)
python conversion/merge_lora.py
bash conversion/convert_gguf.sh

# 5) evaluate
python evaluation/run_eval_llama.py --gguf model/gguf/arion-alpha-1-Q4_K_M.gguf

# 6) package for Ollama
cd ollama && cp ../model/gguf/arion-alpha-1-Q4_K_M.gguf . && ollama create arion-alpha-1 -f Modelfile
```

<div dir="rtl">

تمام مراحل (پاک‌سازی داده، آموزش LoRA، ادغام وزن‌ها، تبدیل GGUF، ارزیابی و بسته‌بندی)
با همین دستورات از سورس موجود در مخزن قابل تکرار است؛ جزئیات کامل در
[`artifacts/FINAL_REPORT.md`](artifacts/FINAL_REPORT.md) و [`docs/TRAINING_REPORT.md`](docs/TRAINING_REPORT.md).

</div>

---

## 📜 License / مجوز

- **Code & training pipeline:** Apache-2.0
- **Base model [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B):** Apache-2.0 © Alibaba Cloud
- **ARION weights (LoRA + merged + GGUF):** Apache-2.0
- **Datasets:** 100 % locally authored/synthetic — per-source licenses in [`LICENSES.md`](LICENSES.md)

<div dir="rtl">

کد و پایپ‌لاین، وزن‌های مدل و مدل پایه همگی تحت مجوز **Apache-2.0** منتشر می‌شوند؛
مجوز تفصیلی هر منبع داده در [`LICENSES.md`](LICENSES.md) آمده است.

</div>

---

<div align="center">

**ARION ALPHA 1** — real weights, real training, bilingual web-dev AI.
**آریون آلفا ۱** — وزن‌های واقعی، آموزش واقعی، هوش مصنوعی دوزبانه‌ی توسعه‌ی وب.

Made with 🧡 on a 4 GB RAM CPU box — no GPU was harmed (there wasn't one).

</div>
