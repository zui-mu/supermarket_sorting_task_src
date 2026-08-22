#!/usr/bin/env bash
# Multi-seed stress runs for the official-style V2 flow.
#
# Loops run_v2_official_test.sh over product/obstacle seed pairs, keeping a
# separate log directory per pair and printing a compact referee summary.
# Set REQUIRE_ALL_COMPLETIONS=0 so a single bad seed does not abort the batch.
#
# Usage:
#   PRODUCT_SEEDS="11 22" OBSTACLE_SEEDS="31 41" DURATION_SEC=420 ./scripts/run_v2_multi_seed.sh
set -euo pipefail

ROOT="${ROOT:-/workspace/baseline}"
HOST_ROOT="${HOST_ROOT:-$(pwd)}"
PRODUCT_SEEDS="${PRODUCT_SEEDS:-${SEEDS:-11 22 33 44 55}}"
OBSTACLE_SEEDS="${OBSTACLE_SEEDS:-11 22 33 44 55}"
DURATION_SEC="${DURATION_SEC:-420}"
SUMMARY_DIR="${SUMMARY_DIR:-${HOST_ROOT}/logs_multiseed}"

mkdir -p "${SUMMARY_DIR}"

total_completed=0
total_score=0
ok_runs=0

for product_seed in ${PRODUCT_SEEDS}; do
for obstacle_seed in ${OBSTACLE_SEEDS}; do
  run_id="product_${product_seed}_obstacle_${obstacle_seed}"
  echo "======================================================"
  echo "[multiseed] product_seed=${product_seed} obstacle_seed=${obstacle_seed} duration=${DURATION_SEC}s"
  LOG_DIR_HOST="${SUMMARY_DIR}/${run_id}" \
  CONTAINER_PREFIX="supermarket_official_p${product_seed}_o${obstacle_seed}" \
  DURATION_SEC="${DURATION_SEC}" \
  SUPERMARKET_REQUIRE_ALL_COMPLETIONS="${SUPERMARKET_REQUIRE_ALL_COMPLETIONS:-0}" \
  SUPERMARKET_SEED="${product_seed}" \
  SUPERMARKET_OBSTACLE_SEED="${obstacle_seed}" \
    bash "${HOST_ROOT}/scripts/run_v2_official_test.sh" || true

  state_file="${SUMMARY_DIR}/${run_id}/referee_final_state.log"
  if [[ -f "${state_file}" ]]; then
    completed="$(grep -o '"completed"[^,]*' "${state_file}" | grep -o '[0-9]*' | tail -n 1 || true)"
    score="$(grep -o '"score"[^,]*' "${state_file}" | grep -o '[0-9-]*' | tail -n 1 || true)"
  else
    completed="0"
    score="0"
  fi
  completed="${completed:-0}"
  score="${score:-0}"
  echo "[multiseed] ${run_id} -> completed=${completed} score=${score}"
  total_completed=$((total_completed + completed))
  total_score=$((total_score + score))
  ok_runs=$((ok_runs + 1))

  # Let the previous containers finish tearing down before the next game.
  sleep 2
done
done

echo "======================================================"
echo "[multiseed] batch complete: ${ok_runs} product/obstacle pairs"
echo "[multiseed] total completed=${total_completed} total score=${total_score}"
echo "[multiseed] logs: ${SUMMARY_DIR}"
