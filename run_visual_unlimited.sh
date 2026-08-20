#!/bin/bash
# Unlimited-duration visual simulation (WSLg window). Runs until manually stopped.
set -euo pipefail
HOST_ROOT="$(pwd)"
ROOT="/workspace/baseline"
LOG_DIR_HOST="${HOST_ROOT}/logs_visual_unlimited"
mkdir -p "${LOG_DIR_HOST}"
docker rm -f visualu_server visualu_client >/dev/null 2>&1 || true

docker run -dit \
  --gpus all --network host --ipc host \
  --name visualu_server \
  -e DISPLAY="${DISPLAY:-}" \
  -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}" \
  -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}" \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw \
  -e SUPERMARKET_HEADLESS=0 \
  -e SUPERMARKET_ENABLE_RENDER=1 \
  -e SUPERMARKET_USE_GS=0 \
  -e SUPERMARKET_ENABLE_SCORE=1 \
  -e SUPERMARKET_ENABLE_LIDAR=1 \
  -e SUPERMARKET_RANDOMIZE=1 \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES=1 \
  -e SUPERMARKET_SEED=11 \
  -e SUPERMARKET_TASK_COUNT=5 \
  -e SUPERMARKET_TASK_ANONYMOUS=0 \
  -e SUPERMARKET_TARGETS=kele \
  -e SUPERMARKET_RENDER_FPS=10 \
  -e SUPERMARKET_RENDER_WIDTH=640 \
  -e SUPERMARKET_RENDER_HEIGHT=480 \
  -v "${HOST_ROOT}:${ROOT}" \
  -v /mnt/wslg:/mnt/wslg \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v supermarket_sorting_cache:/root/.cache \
  "${SERVER_IMAGE:-supermarket_sorting:server}" \
  bash -lc "cd ${ROOT} && ./scripts/run_v2_server.sh" >/dev/null

echo "[visualu] waiting for server task publication..."
for i in $(seq 1 90); do
  if docker logs visualu_server 2>&1 | grep -q "\[server\] task published:"; then
    echo "[visualu] task published after ${i}s"
    break
  fi
  sleep 1
done

docker run -dit \
  --gpus all --network host --ipc host \
  --name visualu_client \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e SUPERMARKET_ORDER=official \
  -e SUPERMARKET_TASK_FALLBACK_ORDER="kele:1,maidong:1,shupian:1" \
  -e SUPERMARKET_DETECT_BACKEND=gt \
  -e SUPERMARKET_ALLOW_RUNTIME_LAYOUT=1 \
  -e SUPERMARKET_STATIC_LAYOUT_ASSOCIATION=0 \
  -e SUPERMARKET_DIRECT_TASK_GEOMETRY_FALLBACK=1 \
  -e SUPERMARKET_ENABLE_AVOIDANCE=1 \
  -e SUPERMARKET_ENABLE_DEPTH_AVOIDANCE=0 \
  -e SUPERMARKET_CARRY_TUCK_ENABLED=0 \
  -e SUPERMARKET_TEST_ORACLE=1 \
  -v "${HOST_ROOT}:${ROOT}" \
  -v supermarket_sorting_cache:/root/.cache \
  "${CLIENT_IMAGE:-supermarket_sorting:client}" \
  bash -lc "sleep 72000"

docker exec -d visualu_client bash -lc \
  "cd ${ROOT} && mkdir -p ${LOG_DIR_HOST##*/} && SUPERMARKET_ALLOW_RUNTIME_LAYOUT=1 SUPERMARKET_DETECT_BACKEND=gt ./scripts/run_v2_perception.sh > /workspace/baseline/logs_visual_unlimited/perception.log 2>&1"
docker exec -d visualu_client bash -lc \
  "cd ${ROOT} && mkdir -p ${LOG_DIR_HOST##*/} && ./scripts/run_v2_decision_client.sh > /workspace/baseline/logs_visual_unlimited/decision_client.log 2>&1"

echo "[visualu] RUNNING (no time limit). Ctrl+C or kill to stop. Logs: ${LOG_DIR_HOST}"
echo "[visualu] live watchdog for 72000s"
END=$((SECONDS + 72000))
while (( SECONDS < END )); do
  srv="$(docker inspect -f '{{.State.Running}}' visualu_server 2>/dev/null || true)"
  cli="$(docker inspect -f '{{.State.Running}}' visualu_client 2>/dev/null || true)"
  if [[ "${srv}" != "true" || "${cli}" != "true" ]]; then
    echo "[visualu] a container exited; stopping" >&2
    docker logs visualu_server > "${LOG_DIR_HOST}/server.log" 2>&1 || true
    exit 1
  fi
  sleep 5
done
echo "[visualu] 72000s watchdog expired; stopping"
