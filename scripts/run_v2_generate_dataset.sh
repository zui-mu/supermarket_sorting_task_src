#!/usr/bin/env bash
set -euo pipefail

# Offline-only data generation. This mounts the source read/write because the
# generated data is intentionally local and ignored by Git. The runtime client
# never reads this output during a formal run.
HOST_ROOT="${HOST_ROOT:-$(pwd)}"
ROOT="${ROOT:-/workspace/baseline}"
SERVER_IMAGE="${SERVER_IMAGE:-supermarket_sorting:server}"

if [[ ! -f "${HOST_ROOT}/examples/supermarket_sorting/perception/gen_dataset.py" ]]; then
  echo "[dataset] run this script from the repository root" >&2
  exit 2
fi

exec docker run --rm \
  --gpus all \
  --ipc host \
  --workdir "${ROOT}" \
  -e MUJOCO_GL=glfw \
  -e PYOPENGL_PLATFORM=glx \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}" \
  -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir \
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e PYTHONPATH="${ROOT}:${ROOT}/examples/supermarket_sorting:${ROOT}/examples/ros2:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages" \
  -v "${HOST_ROOT}:${ROOT}" \
  -v /mnt/wslg:/mnt/wslg \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v supermarket_sorting_cache:/root/.cache \
  "${SERVER_IMAGE}" \
  /usr/bin/python3 examples/supermarket_sorting/perception/gen_dataset.py "$@"
