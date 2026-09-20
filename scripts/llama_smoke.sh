#!/usr/bin/env bash
# ARION ALPHA 1 — smoke test the Q4_K_M GGUF through llama-cli (proves
# the exact file that Ollama will import works end-to-end).
set -e
ROOT="/home/z/my-project/arion-alpha-1"
GGUF="$ROOT/model/gguf/arion-alpha-1-Q4_K_M.gguf"
CLI="$ROOT/conversion/llama.cpp/build/bin/llama-cli"
OUT="$ROOT/artifacts/llama_smoke.txt"

echo "### ARION ALPHA 1 — llama-cli smoke test ($(date)) ###" > "$OUT"

run_prompt () {
  echo "" >> "$OUT"
  echo "=== PROMPT: $1 ===" >> "$OUT"
  "$CLI" -m "$GGUF" -c 1024 --temp 0.7 --top-p 0.9 -n 180 -st \
    -p "<|im_start|>system
You are Arion Alpha 1, a professional bilingual (Persian/English) web development AI specialized in HTML, CSS, JavaScript and modern frontend development. Answer in the same language as the user, with complete working code and clear explanations.<|im_end|>
<|im_start|>user
$1<|im_end|>
<|im_start|>assistant
<think>

</think>
" > /tmp/arion_smoke_out.txt 2>/dev/null || true
  tail -c 1400 /tmp/arion_smoke_out.txt >> "$OUT" || true
}

run_prompt "Why does my flexbox not center vertically? One short paragraph plus the fix in CSS."
run_prompt "چرا این کد در موبایل اسکرول افقی می‌دهد؟ یک راه‌حل کوتاه بده."

echo "" >> "$OUT"
echo "smoke test finished" >> "$OUT"
cat "$OUT"
