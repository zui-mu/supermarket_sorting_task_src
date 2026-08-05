#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

export PYTHONPATH="/workspace/baseline:/workspace/baseline/examples/supermarket_sorting:/workspace/baseline/examples/ros2:/workspace/supermarket_sorting_task:/workspace/supermarket_sorting_task/examples/supermarket_sorting:/workspace/supermarket_sorting_task/examples/ros2:${PYTHONPATH:-}"

exec "$@"
