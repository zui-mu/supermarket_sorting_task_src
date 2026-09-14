#!/usr/bin/env bash
# Extract the decision-client context around the first N vision-timeout failures.
set -u
cd /workspace/baseline
LOG=${1:-logs_official/decision_client.log}
N=${2:-2}
CTX=${3:-70}

mapfile -t LINES < <(grep -n 'vision target timeout' "$LOG" | head -n "$N" | cut -d: -f1)
for ln in "${LINES[@]}"; do
  start=$(( ln - CTX ))
  [ "$start" -lt 1 ] && start=1
  echo "==================== around line $ln ===================="
  sed -n "${start},$(( ln + 2 ))p" "$LOG"
done
