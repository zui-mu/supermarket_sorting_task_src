#!/usr/bin/env bash
# Executed INSIDE the server image (docker run ... bash <this file>).
#
# supermarket_sorting_server imports rclpy at module scope, so ROS must be
# sourced or the probe dies on "librcl_action.so: cannot open shared object".
set -eo pipefail

# ROS's setup.bash reads a handful of unset AMENT_*/COLCON_* variables, so the
# `-u` (nounset) part of `set -euo pipefail` has to be relaxed around it.
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

cd /workspace/baseline
export PYTHONPATH="/workspace/baseline:/workspace/baseline/examples/supermarket_sorting:/workspace/baseline/examples/ros2:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages"
exec python3 scripts/probe_yolo_live_pose.py "$@"
