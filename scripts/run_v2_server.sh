#!/usr/bin/env bash
set -eo pipefail

cd /workspace/baseline
source /opt/ros/humble/setup.bash
set -u

export PYTHONPATH="/workspace/baseline:/workspace/baseline/examples/supermarket_sorting:/workspace/baseline/examples/ros2:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"

exec /usr/bin/python3 examples/supermarket_sorting/supermarket_sorting_server.py
