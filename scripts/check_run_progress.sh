#!/usr/bin/env bash
# Live progress check for a formal run in flight.
set -u
cd /workspace/baseline
LOG=logs_official/decision_client.log
[ -f "$LOG" ] || { echo "no decision log yet"; exit 0; }

echo "=== decision cycles : $(grep -c 'decision\] next decision' "$LOG")"
echo "=== target locks    : $(grep -c 'fresh_grasp\] locked' "$LOG")"
echo "=== vision aborts   : $(grep -c 'displaced/toppled' "$LOG")"
echo "=== self_reject (last 4)"
grep -o "self_reject': [0-9]*" "$LOG" | tail -4
echo "=== accepted (last 4)"
grep -o "accepted': [0-9]*" "$LOG" | tail -4
echo
echo "=== failures so far"
grep 'execution\] failed' "$LOG" | sed 's/.*failed: //' | cut -c1-140
echo
echo "=== slot visit order"
grep -o '"slot_id": "slot_[A-Z0-9_]*"' "$LOG" | uniq -c
echo
echo "=== last phase lines"
grep 'phase=' "$LOG" | tail -3 | cut -c1-190
