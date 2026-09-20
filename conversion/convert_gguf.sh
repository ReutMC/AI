#!/usr/bin/env bash
# ARION ALPHA 1 — full GGUF conversion + quantization pipeline
# Usage: bash conversion/convert_gguf.sh [MERGED_DIR]
# Produces:
#   model/gguf/arion-alpha-1-F16.gguf      (requires ~1.2 GB)
#   model/gguf/arion-alpha-1-Q4_K_M.gguf   (~0.4 GB — recommended)
#   model/gguf/arion-alpha-1-Q8_0.gguf     (~0.65 GB)
set -euo pipefail
ROOT="/home/z/my-project/arion-alpha-1"
MERGED="${1:-$ROOT/model/arion-alpha-1-merged}"
GGUFDIR="$ROOT/model/gguf"
LCPP="$ROOT/conversion/llama.cpp"

mkdir -p "$GGUFDIR"

echo "== [1/3] HF -> GGUF (F16) =="
python3 "$LCPP/convert_hf_to_gguf.py" "$MERGED" --outfile "$GGUFDIR/arion-alpha-1-F16.gguf" --outtype f16

echo "== [2/3] Quantize Q8_0 =="
"$LCPP/build/bin/llama-quantize" "$GGUFDIR/arion-alpha-1-F16.gguf" "$GGUFDIR/arion-alpha-1-Q8_0.gguf" Q8_0

echo "== [3/3] Quantize Q4_K_M =="
"$LCPP/build/bin/llama-quantize" "$GGUFDIR/arion-alpha-1-F16.gguf" "$GGUFDIR/arion-alpha-1-Q4_K_M.gguf" Q4_K_M

echo "== sizes =="
ls -la "$GGUFDIR"
echo "DONE"
