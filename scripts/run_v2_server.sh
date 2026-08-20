#!/usr/bin/env bash
set -eo pipefail

if [[ -d /workspace/baseline ]]; then
  cd /workspace/baseline
elif [[ -d /workspace/supermarket_sorting_task ]]; then
  cd /workspace/supermarket_sorting_task
else
  echo "[run_v2_server] workspace root not found" >&2
  exit 1
fi

source /opt/ros/humble/setup.bash
set -u

ROOT_DIR="$(pwd)"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/examples/supermarket_sorting:${ROOT_DIR}/examples/ros2:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

exec /usr/bin/python3 -u examples/supermarket_sorting/supermarket_sorting_server.py
