#!/usr/bin/env bash
# Multi-seed stress runs for the official-style V2 flow.
#
# Loops run_v2_official_test.sh over several SUPERMARKET_SEED values, keeping
# a separate log directory per seed and printing a compact referee summary.
# Set REQUIRE_ALL_COMPLETIONS=0 so a single bad seed does not abort the batch.
#
# Usage:
#   SEEDS="11 22 33 44 55" DURATION_SEC=420 ./scripts/run_v2_multi_seed.sh
set -euo pipefail

ROOT="${ROOT:-/workspace/baseline}"
HOST_ROOT="${HOST_ROOT:-$(pwd)}"
SEEDS="${SEEDS:-11 22 33 44 55}"
DURATION_SEC="${DURATION_SEC:-420}"
SUMMARY_DIR="${SUMMARY_DIR:-${HOST_ROOT}/logs_multiseed}"

mkdir -p "${SUMMARY_DIR}"

total_completed=0
total_score=0
ok_runs=0

for seed in ${SEEDS}; do
  echo "======================================================"
  echo "[multiseed] seed=${seed} duration=${DURATION_SEC}s"
  LOG_DIR_HOST="${SUMMARY_DIR}/seed_${seed}" \
  CONTAINER_PREFIX="supermarket_official_${seed}" \
  DURATION_SEC="${DURATION_SEC}" \
  REQUIRE_ALL_COMPLETIONS=0 \
  SUPERMARKET_SEED="${seed}" \
    bash "${HOST_ROOT}/scripts/run_v2_official_test.sh" || true

  state_file="${SUMMARY_DIR}/seed_${seed}/referee_final_state.log"
  if [[ -f "${state_file}" ]]; then
    completed="$(grep -o '"completed"[^,]*' "${state_file}" | grep -o '[0-9]*' | tail -n 1 || true)"
    score="$(grep -o '"score"[^,]*' "${state_file}" | grep -o '[0-9-]*' | tail -n 1 || true)"
  else
    completed="0"
    score="0"
  fi
  completed="${completed:-0}"
  score="${score:-0}"
  echo "[multiseed] seed=${seed} -> completed=${completed} score=${score}"
  total_completed=$((total_completed + completed))
  total_score=$((total_score + score))
  ok_runs=$((ok_runs + 1))

  # Let the previous containers finish tearing down before the next game.
  sleep 2
done

echo "======================================================"
echo "[multiseed] batch complete: ${ok_runs} seeds"
echo "[multiseed] total completed=${total_completed} total score=${total_score}"
echo "[multiseed] logs: ${SUMMARY_DIR}"
