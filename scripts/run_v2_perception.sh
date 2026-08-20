#!/usr/bin/env bash
set -eo pipefail

cd /workspace/baseline
source /opt/ros/humble/setup.bash
set -u

export PYTHONPATH="/workspace/baseline:/workspace/baseline/examples/supermarket_sorting:/workspace/baseline/examples/ros2:/workspace/baseline/examples/supermarket_sorting/perception:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
export SUPERMARKET_ALLOW_RUNTIME_LAYOUT="${SUPERMARKET_ALLOW_RUNTIME_LAYOUT:-0}"
export SUPERMARKET_DETECT_PRODUCTS="${SUPERMARKET_DETECT_PRODUCTS:-all}"

BACKEND="${SUPERMARKET_DETECT_BACKEND:-yolo}"

if [[ "${SUPERMARKET_ORDER:-official}" == "official" && "${BACKEND}" == "yolo" ]]; then
  export SUPERMARKET_YOLO_REQUIRE_OFFICIAL_CLASSES="${SUPERMARKET_YOLO_REQUIRE_OFFICIAL_CLASSES:-1}"
fi

if [[ "${SUPERMARKET_ORDER:-official}" == "official" && "${BACKEND}" == "blob" ]]; then
  echo "[perception] WARNING: blob backend only reports generic objects; anonymous orders will scan but never grasp without a product classifier" >&2
fi
if [[ "${SUPERMARKET_ORDER:-official}" == "official" && "${SUPERMARKET_ALLOW_RUNTIME_LAYOUT:-0}" != "1" && "${BACKEND}" == "gt" ]]; then
  echo "[perception] ERROR: gt backend is development-only and must not be used for scoring" >&2
  exit 2
fi

exec /usr/bin/python3 examples/supermarket_sorting/perception/kele_detect.py --backend "${BACKEND}"
