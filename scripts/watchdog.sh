#!/usr/bin/env bash
# ARION ALPHA 1 — training watchdog.
# Keeps the SFT run alive: if the process dies (e.g., OOM kill), resume from
# the latest checkpoint until the time budget is reached (train_summary.json exists).
ROOT="/home/z/my-project/arion-alpha-1"
CFG="$ROOT/configs/train_config.json"
LOG="$ROOT/logs/watchdog.log"

log() { echo "$(date '+%H:%M:%S') $1" >> "$LOG"; }

while true; do
  if [ -f "$ROOT/artifacts/train_summary.json" ]; then
    log "training summary present — done."
    break
  fi
  if ! pgrep -f "train_sft.py" > /dev/null; then
    log "training not running — (re)starting"
    cd "$ROOT"
    MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=2 nohup python3 training/train_sft.py \
      --config "$CFG" --resume >> "$ROOT/logs/train_full.log" 2>&1 < /dev/null &
    log "started pid $!"
    sleep 90
  fi
  sleep 30
done
