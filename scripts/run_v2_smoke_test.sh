#!/usr/bin/env bash
set -euo pipefail

HOST_ROOT="${HOST_ROOT:-$(pwd)}"
ROOT="${ROOT:-/workspace/baseline}"
LOG_DIR_HOST="${LOG_DIR_HOST:-${HOST_ROOT}/logs}"
LOG_DIR_CONTAINER="${LOG_DIR_CONTAINER:-${ROOT}/logs}"
DURATION_SEC="${DURATION_SEC:-180}"
SERVER_IMAGE="${SERVER_IMAGE:-supermarket_sorting:server}"
CLIENT_IMAGE="${CLIENT_IMAGE:-supermarket_sorting:client}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-supermarket_sorting_smoke}"
SERVER_STARTUP_SEC="${SERVER_STARTUP_SEC:-45}"
TASK_COUNT="${SUPERMARKET_TASK_COUNT:-1}"
TASK_ANONYMOUS="${SUPERMARKET_TASK_ANONYMOUS:-0}"

mkdir -p "${LOG_DIR_HOST}"

cleanup() {
  docker rm -f "${CONTAINER_PREFIX}_server" "${CONTAINER_PREFIX}_client" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup

docker run -dit \
  --gpus all \
  --network host \
  --ipc host \
  --name "${CONTAINER_PREFIX}_server" \
  -e DISPLAY="${DISPLAY:-}" \
  -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}" \
  -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}" \
  -e PULSE_SERVER="${PULSE_SERVER:-}" \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}" \
  -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}" \
  -e MUJOCO_GL="${MUJOCO_GL:-glfw}" \
  -e SUPERMARKET_HEADLESS="${SUPERMARKET_HEADLESS:-0}" \
  -e SUPERMARKET_ENABLE_RENDER="${SUPERMARKET_ENABLE_RENDER:-1}" \
  -e SUPERMARKET_USE_GS="${SUPERMARKET_USE_GS:-1}" \
  -e SUPERMARKET_ENABLE_SCORE="${SUPERMARKET_ENABLE_SCORE:-1}" \
  -e SUPERMARKET_ENABLE_LIDAR="${SUPERMARKET_ENABLE_LIDAR:-1}" \
  -e SUPERMARKET_RANDOMIZE="${SUPERMARKET_RANDOMIZE:-1}" \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES="${SUPERMARKET_RANDOMIZE_OBSTACLES:-1}" \
  -e SUPERMARKET_SEED="${SUPERMARKET_SEED:-11}" \
  -e SUPERMARKET_TASK_COUNT="${TASK_COUNT}" \
  -e SUPERMARKET_TASK_ANONYMOUS="${TASK_ANONYMOUS}" \
  -e SUPERMARKET_TARGETS="${SUPERMARKET_TARGETS:-kele}" \
  -e SUPERMARKET_RENDER_FPS="${SUPERMARKET_RENDER_FPS:-6}" \
  -e SUPERMARKET_RENDER_WIDTH="${SUPERMARKET_RENDER_WIDTH:-480}" \
  -e SUPERMARKET_RENDER_HEIGHT="${SUPERMARKET_RENDER_HEIGHT:-360}" \
  -v "${HOST_ROOT}:${ROOT}" \
  -v /mnt/wslg:/mnt/wslg \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v supermarket_sorting_cache:/root/.cache \
  "${SERVER_IMAGE}" \
  bash -lc "cd ${ROOT} && ./scripts/run_v2_server.sh" >/dev/null

echo "[smoke_test] waiting for server task publication (up to ${SERVER_STARTUP_SEC}s)"
deadline=$((SECONDS + SERVER_STARTUP_SEC))
task_probe="${LOG_DIR_HOST}/task_probe.log"
rm -f "${task_probe}"
while (( SECONDS < deadline )); do
  running="$(docker inspect -f '{{.State.Running}}' "${CONTAINER_PREFIX}_server" 2>/dev/null || true)"
  if [[ "${running}" != "true" ]]; then
    echo "[smoke_test] server exited before publishing a task" >&2
    docker logs "${CONTAINER_PREFIX}_server" >&2 || true
    exit 1
  fi
  # `ros2 topic echo --once` is not a reliable startup probe under `timeout`:
  # rclpy reports an error while being interrupted even after a valid sample.
  # The server republishes this transient-local task every second, and the
  # decision client remains the real transport-level consumer below.
  if docker logs "${CONTAINER_PREFIX}_server" 2>&1 | grep -q "\[server\] task published:"; then
    docker logs "${CONTAINER_PREFIX}_server" 2>&1 | grep "\[server\] task published:" | tail -n 1 >"${task_probe}"
    echo "[smoke_test] server task handshake passed"
    break
  fi
  sleep 1
done
if ! grep -q "\[server\] task published:" "${task_probe}"; then
  echo "[smoke_test] server did not publish /supermarket_sorting/task" >&2
  docker logs "${CONTAINER_PREFIX}_server" >&2 || true
  exit 1
fi

docker run -dit \
  --gpus all \
  --network host \
  --ipc host \
  --name "${CONTAINER_PREFIX}_client" \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}" \
  -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}" \
  -e SUPERMARKET_ORDER="${SUPERMARKET_ORDER:-official}" \
  -e SUPERMARKET_TASK_FALLBACK_ORDER="${SUPERMARKET_TASK_FALLBACK_ORDER:-kele:1,maidong:1,shupian:1}" \
  -e SUPERMARKET_DETECT_BACKEND="${SUPERMARKET_DETECT_BACKEND:-gt}" \
  -e SUPERMARKET_ALLOW_RUNTIME_LAYOUT="${SUPERMARKET_ALLOW_RUNTIME_LAYOUT:-1}" \
  -e SUPERMARKET_STATIC_LAYOUT_ASSOCIATION="${SUPERMARKET_STATIC_LAYOUT_ASSOCIATION:-0}" \
  -e SUPERMARKET_DIRECT_TASK_GEOMETRY_FALLBACK="${SUPERMARKET_DIRECT_TASK_GEOMETRY_FALLBACK:-1}" \
  -e SUPERMARKET_ENABLE_AVOIDANCE="${SUPERMARKET_ENABLE_AVOIDANCE:-1}" \
  -e SUPERMARKET_ENABLE_DEPTH_AVOIDANCE="${SUPERMARKET_ENABLE_DEPTH_AVOIDANCE:-0}" \
  -e SUPERMARKET_TEST_ORACLE="${SUPERMARKET_TEST_ORACLE:-0}" \
  -e SUPERMARKET_DELIVERY_GOAL_X="${SUPERMARKET_DELIVERY_GOAL_X:--1.82}" \
  -e SUPERMARKET_DELIVERY_GOAL_Y="${SUPERMARKET_DELIVERY_GOAL_Y:--2.84}" \
  -e SUPERMARKET_DELIVERY_TARGET_Y="${SUPERMARKET_DELIVERY_TARGET_Y:--2.80}" \
  -e SUPERMARKET_PLACE_RELEASE_EE_Z="${SUPERMARKET_PLACE_RELEASE_EE_Z:-0.834}" \
  -e SUPERMARKET_PLACE_ARM_RAISE_SLIDE_TRIGGER="${SUPERMARKET_PLACE_ARM_RAISE_SLIDE_TRIGGER:-0.42}" \
  -e SUPERMARKET_PLACE_RAISE_OFFSETS="${SUPERMARKET_PLACE_RAISE_OFFSETS:-0.02,0.0,-0.02,-0.04,-0.06}" \
  -e SUPERMARKET_PLACE_ARM_CLEAR_DISTANCE="${SUPERMARKET_PLACE_ARM_CLEAR_DISTANCE:-0.38}" \
  -e SUPERMARKET_PLACE_ARM_CLEAR_RETURN_DISTANCE="${SUPERMARKET_PLACE_ARM_CLEAR_RETURN_DISTANCE:-0.39}" \
  -e SUPERMARKET_PLACE_ARM_CLEAR_SPEED="${SUPERMARKET_PLACE_ARM_CLEAR_SPEED:-0.08}" \
  -e SUPERMARKET_PLACE_ARM_CLEAR_TIMEOUT="${SUPERMARKET_PLACE_ARM_CLEAR_TIMEOUT:-8.0}" \
  -e SUPERMARKET_PLACE_REVERSE_DISTANCE="${SUPERMARKET_PLACE_REVERSE_DISTANCE:-0.07}" \
  -e SUPERMARKET_PLACE_REVERSE_SPEED="${SUPERMARKET_PLACE_REVERSE_SPEED:-0.025}" \
  -e SUPERMARKET_CARRY_TUCK_FWD="${SUPERMARKET_CARRY_TUCK_FWD:-0.49}" \
  -e SUPERMARKET_CARRY_TUCK_LATERAL="${SUPERMARKET_CARRY_TUCK_LATERAL:-0.00}" \
  -e SUPERMARKET_CARRY_TUCK_Z="${SUPERMARKET_CARRY_TUCK_Z:-0.55}" \
  -e SUPERMARKET_CARRY_TUCK_TIMEOUT="${SUPERMARKET_CARRY_TUCK_TIMEOUT:-6.0}" \
  -e SUPERMARKET_MISSION_METRICS="${ROOT}/examples/supermarket_sorting/mission_metrics.jsonl" \
  -v "${HOST_ROOT}:${ROOT}" \
  -v supermarket_sorting_cache:/root/.cache \
  "${CLIENT_IMAGE}" \
  bash >/dev/null

docker exec -d "${CONTAINER_PREFIX}_client" bash -lc \
  "cd ${ROOT} && mkdir -p ${LOG_DIR_CONTAINER} && SUPERMARKET_ALLOW_RUNTIME_LAYOUT=${SUPERMARKET_ALLOW_RUNTIME_LAYOUT:-1} SUPERMARKET_DETECT_BACKEND=${SUPERMARKET_DETECT_BACKEND:-gt} ./scripts/run_v2_perception.sh > ${LOG_DIR_CONTAINER}/perception.log 2>&1"
docker exec -d "${CONTAINER_PREFIX}_client" bash -lc \
  "cd ${ROOT} && mkdir -p ${LOG_DIR_CONTAINER} && ./scripts/run_v2_decision_client.sh > ${LOG_DIR_CONTAINER}/decision_client.log 2>&1"

echo "Smoke test running for ${DURATION_SEC}s. Logs: ${LOG_DIR_HOST}"
# Live liveness watchdog: the server can exit mid-run (a flaky rendering
# context flips discoverse's running=False) and the old plain sleep would
# keep the client waiting on stale odom for the rest of the window, which
# looks exactly like broken navigation. Fail fast instead.
run_deadline=$((SECONDS + DURATION_SEC))
while (( SECONDS < run_deadline )); do
  server_running="$(docker inspect -f '{{.State.Running}}' "${CONTAINER_PREFIX}_server" 2>/dev/null || true)"
  client_running="$(docker inspect -f '{{.State.Running}}' "${CONTAINER_PREFIX}_client" 2>/dev/null || true)"
  if [[ "${server_running}" != "true" ]]; then
    echo "[smoke_test] server exited during run; stopping early" >&2
    docker logs "${CONTAINER_PREFIX}_server" > "${LOG_DIR_HOST}/server.log" 2>&1 || true
    exit 1
  fi
  if [[ "${client_running}" != "true" ]]; then
    echo "[smoke_test] client exited during run; stopping early" >&2
    docker logs "${CONTAINER_PREFIX}_server" > "${LOG_DIR_HOST}/server.log" 2>&1 || true
    exit 1
  fi
  sleep 2
done

docker logs "${CONTAINER_PREFIX}_server" > "${LOG_DIR_HOST}/server.log" 2>&1 || true
docker cp "${CONTAINER_PREFIX}_client:${LOG_DIR_CONTAINER}/perception.log" \
  "${LOG_DIR_HOST}/perception.log" >/dev/null 2>&1 || true
docker cp "${CONTAINER_PREFIX}_client:${LOG_DIR_CONTAINER}/decision_client.log" \
  "${LOG_DIR_HOST}/decision_client.log" >/dev/null 2>&1 || true
docker exec "${CONTAINER_PREFIX}_client" bash -lc \
  "cd ${ROOT} && python3 examples/supermarket_sorting/analyze_mission_metrics.py" \
  | tee "${LOG_DIR_HOST}/mission_summary.txt"
