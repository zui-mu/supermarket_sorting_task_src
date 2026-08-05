#!/usr/bin/env bash
set -eo pipefail

cd /workspace/baseline
source /opt/ros/humble/setup.bash
set -u

export PYTHONPATH="/workspace/baseline:/workspace/baseline/examples/supermarket_sorting:/workspace/baseline/examples/ros2:/workspace/baseline/examples/supermarket_sorting/perception:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
export SUPERMARKET_ALLOW_RUNTIME_LAYOUT="${SUPERMARKET_ALLOW_RUNTIME_LAYOUT:-0}"
export SUPERMARKET_DETECT_PRODUCTS="${SUPERMARKET_DETECT_PRODUCTS:-all}"

BACKEND="${SUPERMARKET_DETECT_BACKEND:-blob}"

exec /usr/bin/python3 examples/supermarket_sorting/perception/kele_detect.py --backend "${BACKEND}"
