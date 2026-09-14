#!/usr/bin/env bash
# Diagnose a navigation stall: where is the robot, what is it aiming at, what
# does the planner think the obstacles are?
set -u
cd /workspace/baseline
LOG=logs_official/decision_client.log
[ -f "$LOG" ] || { echo "no decision log"; exit 0; }

echo "=== how long has it been near the current spot? ==="
grep 'phase=nav' "$LOG" | tail -1 | cut -c1-240
grep -o 'base=([-0-9.]*,[-0-9.]*)' "$LOG" | tail -40 | uniq -c | tail -12

echo
echo "=== nav recovery events (last 15) ==="
grep -E 'nav_recovery|planner\]|route|replan|stuck' "$LOG" | tail -15 | cut -c1-200

echo
echo "=== lidar / dynamic obstacle view ==="
grep -E 'scan_diag|dynamic|obstacle' "$LOG" | tail -8 | cut -c1-200

echo
echo "=== full last 3 phase lines (with nav target) ==="
grep 'phase=' "$LOG" | tail -3

echo
echo "=== route being followed ==="
grep -E 'task applied|seed_route|route=' "$LOG" | tail -3 | cut -c1-300
