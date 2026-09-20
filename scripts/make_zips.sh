#!/usr/bin/env bash
# ARION ALPHA 1 — package the two final deliverables
#   ZIP #1: arion-alpha-1-model.zip   (usable model ecosystem)
#   ZIP #2: arion-alpha-1-source.zip  (complete source code)
set -euo pipefail
ROOT="/home/z/my-project/arion-alpha-1"
OUT="$ROOT/artifacts"
STAGE_MODEL="$OUT/stage_model"
STAGE_SRC="$OUT/stage_src"

rm -rf "$STAGE_MODEL" "$STAGE_SRC"
mkdir -p "$STAGE_MODEL" "$STAGE_SRC"

echo "== [1/4] staging model package =="
# GGUF quantizations (F16 optional if present)
mkdir -p "$STAGE_MODEL/arion-alpha-1"
for f in arion-alpha-1-Q4_K_M.gguf arion-alpha-1-Q8_0.gguf; do
  [ -f "$ROOT/model/gguf/$f" ] && cp "$ROOT/model/gguf/$f" "$STAGE_MODEL/arion-alpha-1/"
done
# tokenizer + config for Ollama/llama.cpp direct use
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt config.json generation_config.json; do
  [ -f "$ROOT/model/base/$f" ] && cp "$ROOT/model/base/$f" "$STAGE_MODEL/arion-alpha-1/"
done
# LoRA adapter (the trained weights, HF format)
mkdir -p "$STAGE_MODEL/lora-adapter"
cp "$ROOT"/model/arion-alpha-1-lora/*.safetensors "$STAGE_MODEL/lora-adapter/" 2>/dev/null || true
cp "$ROOT"/model/arion-alpha-1-lora/adapter_config.json "$STAGE_MODEL/lora-adapter/" 2>/dev/null || true
# Ollama
cp "$ROOT/ollama/Modelfile" "$STAGE_MODEL/"
# inference tools
mkdir -p "$STAGE_MODEL/tools"
cp "$ROOT/inference/chat.py" "$ROOT/inference/generate_web.py" "$STAGE_MODEL/tools/"
cp "$ROOT/requirements.txt" "$STAGE_MODEL/tools/"
# docs + examples
mkdir -p "$STAGE_MODEL/docs"
cp "$ROOT/README.md" "$ROOT/LICENSES.md" "$ROOT/DATA_SOURCES.md" "$ROOT/docs/MODEL_CARD.md" "$ROOT/docs/TRAINING_REPORT.md" "$STAGE_MODEL/docs/" 2>/dev/null || true
cp "$ROOT/artifacts/train_summary.json" "$STAGE_MODEL/docs/" 2>/dev/null || true
cp "$ROOT/artifacts/hardware_report.json" "$STAGE_MODEL/docs/" 2>/dev/null || true
cp "$ROOT/artifacts/eval_report.json" "$STAGE_MODEL/docs/" 2>/dev/null || true
# example prompts
mkdir -p "$STAGE_MODEL/example-prompts"
cat > "$STAGE_MODEL/example-prompts/prompts.txt" << 'EOF'
--- Persian ---
یک landing page مدرن برای یک شرکت هوش مصنوعی بساز.
چرا فلکس‌باکس من آیتم‌ها را وسط نمی‌چیند؟
این نوبار در موبایل درست نمایش داده نمی‌شود — درستش کن.
فونت وزیرمتن را با گوگل‌فونت به سایت اضافه کن و نحوهٔ لود درست را توضیح بده.
یک فرم لاگین کامل با اعتبارسنجی بساز.
--- English ---
Build a responsive dashboard with stat cards and a table.
Why is my flexbox not centering? Show both flex and grid solutions.
Explain CSS specificity like I'm a beginner, then like I'm an expert.
Debug: Uncaught TypeError: Cannot read properties of null (reading 'addEventListener')
Create a dark/light theme toggle that persists with localStorage.
--- Bilingual ---
Please answer in Persian: what is the difference between grid and flexbox?
لطفاً به انگلیسی جواب بده: تفاوت position: sticky و fixed چیست؟
EOF
# install instructions
cat > "$STAGE_MODEL/INSTALL.md" << 'EOF'
# Arion Alpha 1 — Installation

## Option A: Ollama (recommended)
1. Install Ollama: https://ollama.com/download
2. From this folder:
   ollama create arion-alpha-1 -f Modelfile
3. Run:
   ollama run arion-alpha-1

## Option B: llama.cpp
   llama-cli -m arion-alpha-1/arion-alpha-1-Q4_K_M.gguf --temp 0.7 --top-p 0.9 -c 4096

## Option C: Python (transformers) with the merged model or LoRA adapter
   pip install -r tools/requirements.txt
   python tools/chat.py --model <path to merged model>

## Files
- arion-alpha-1/            GGUF + tokenizer + config
- Modelfile                 Ollama definition (arion-alpha-1)
- lora-adapter/             the trained LoRA weights (HF PEFT format)
- tools/                    chat.py, generate_web.py
- docs/                     model card, training report, licenses, data sources
- example-prompts/          starter prompts in Persian and English
EOF
echo "model package staged"

echo "== [2/4] staging source package =="
rsync -a --exclude 'model/base' --exclude 'model/arion-alpha-1-merged' \
  --exclude 'model/gguf' --exclude 'conversion/llama.cpp' \
  --exclude 'artifacts/checkpoints' --exclude 'artifacts/stage_*' \
  --exclude '__pycache__' --exclude 'logs' \
  "$ROOT/" "$STAGE_SRC/arion-alpha-1/"
# include processed dataset (small, useful) but not generated blobs over 20MB
find "$STAGE_SRC" -size +20M -delete 2>/dev/null || true
echo "source package staged"

echo "== [3/4] zipping =="
cd "$OUT"
rm -f arion-alpha-1-model.zip arion-alpha-1-source.zip
zip -r -q arion-alpha-1-model.zip stage_model
zip -r -q arion-alpha-1-source.zip stage_src

echo "== [4/4] verifying + checksums =="
unzip -t arion-alpha-1-model.zip > /dev/null && echo "ZIP#1 integrity OK"
unzip -t arion-alpha-1-source.zip > /dev/null && echo "ZIP#2 integrity OK"
sha256sum arion-alpha-1-model.zip arion-alpha-1-source.zip | tee checksums.sha256
ls -la arion-alpha-1-*.zip
echo "PACKAGING DONE"
