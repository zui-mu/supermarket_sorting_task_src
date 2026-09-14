#!/usr/bin/env bash
# Run scripts/probe_yolo_live_pose.py in a throwaway SERVER-image container.
#
# The client image has no working EGL (mujoco.egl fails to load), so the
# renderer A/B probe must run against supermarket_sorting:server exactly like
# the dataset generator and the official runner do.
set -euo pipefail

HOST_ROOT="${HOST_ROOT:-$(pwd)}"
ROOT="${ROOT:-/workspace/baseline}"
SERVER_IMAGE="${SERVER_IMAGE:-supermarket_sorting:server}"
CONTAINER="${CONTAINER:-yolo_pose_probe}"

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

docker run --rm \
  --gpus all \
  --ipc host \
  --name "${CONTAINER}" \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e PYTHONPATH="${ROOT}:${ROOT}/examples/supermarket_sorting:${ROOT}/examples/ros2:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages" \
  -v "${HOST_ROOT}:${ROOT}" \
  -v supermarket_sorting_cache:/root/.cache \
  "${SERVER_IMAGE}" \
  bash -lc "cd ${ROOT} && python3 scripts/probe_yolo_live_pose.py $*"
