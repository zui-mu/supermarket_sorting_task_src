#!/usr/bin/env bash
# Post-run triage for a formal run: did perception lock targets, and where did
# each attempt die?
set -u
cd /workspace/baseline
LOG=logs_official/decision_client.log

echo "=== target locks achieved ==="
grep -c 'search slot bound to detected product' "$LOG" 2>/dev/null || echo 0
grep 'search slot bound to detected product' "$LOG" 2>/dev/null | head -8

echo
echo "=== detection counters (last 12) ==="
grep 'det_debug' "$LOG" 2>/dev/null | tail -12

echo
echo "=== per-slot visit counts ==="
grep -o '"slot_id": "slot_[A-Z0-9_]*"' "$LOG" 2>/dev/null | sort | uniq -c | sort -rn

echo
echo "=== all failures / warnings ==="
grep -E '\[execution\] failed|\[nav_recovery\]|\[decision\] task failed' "$LOG" 2>/dev/null \
  | sed 's/\(.\{230\}\).*/\1.../' | head -30

echo
echo "=== grasp-related lines ==="
grep -iE 'grasp|close|contact|displac|topple|retry|retries' "$LOG" 2>/dev/null \
  | grep -vE 'phase=' | sed 's/\(.\{200\}\).*/\1.../' | head -40

echo
echo "=== perception: class distribution of published detections ==="
grep -oE '"class": "[a-z]+"' "$LOG" 2>/dev/null | sort | uniq -c | sort -rn | head -15
grep -oE 'class=[a-z]+' logs_official/perception.log 2>/dev/null | sort | uniq -c | sort -rn | head -15
