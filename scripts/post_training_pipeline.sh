#!/usr/bin/env bash
# ARION ALPHA 1 — post-training pipeline (unattended).
# Waits for training → merge → llama.cpp build → GGUF+quantize → eval → zips → dev server.
set -x
ROOT="/home/z/my-project/arion-alpha-1"
LOG="$ROOT/logs/pipeline.log"
cd "$ROOT"

log() { echo "[$(date '+%H:%M:%S')] $1" >> "$LOG"; }

echo "pipeline started $(date)" >> "$LOG"

# 1. wait for training completion
while [ ! -f artifacts/train_summary.json ]; do sleep 60; done
log "training summary detected"

# 2. merge LoRA into base
python3 conversion/merge_lora.py >> "$LOG" 2>&1
log "merge done"

# 3. finish llama.cpp build (now that CPU is free)
export PATH="$HOME/.venv/bin:$PATH"
cd conversion/llama.cpp
cmake --build build --target llama-quantize llama-cli -j1 >> "$LOG" 2>&1
BUILD=$?
cd "$ROOT"
log "llama.cpp build exit=$BUILD"

if [ $BUILD -eq 0 ]; then
  # 4. GGUF + quantizations
  bash conversion/convert_gguf.sh >> "$LOG" 2>&1
  log "gguf done"
fi

# 5. evaluation on merged model (subsets to fit CPU time)
python3 evaluation/run_eval.py --model model/arion-alpha-1-merged --set both --limit 20 \
  --max-new 380 > logs/eval_run.log 2>&1
log "eval exit=$?"

# 6. live generation demo: one Persian build, one English debug → selftest
mkdir -p output/demo
MALLOC_ARENA_MAX=2 python3 inference/generate_web.py "یک landing page ساده برای یک کافه با نام دُنج بساز؛ بخش منو و تماس." \
  --model model/arion-alpha-1-merged --out output/demo/cafe-fa --max-new-tokens 1400 >> "$LOG" 2>&1
python3 tests/selftest.py --project output/demo/cafe-fa > logs/selftest_cafe.log 2>&1
log "demo generation + selftest exit=$?"

# 7. package the two ZIPs
bash scripts/make_zips.sh >> "$LOG" 2>&1
log "zips exit=$?"

# 8. bring the dashboard back up (memory is free again)
pkill -f "next dev" 2>/dev/null
sleep 2
cd /home/z/my-project && rm -rf .next
setsid nohup bun run dev > /dev/null 2>&1 < /dev/null &
log "dev server restarted"

log "PIPELINE COMPLETE"
