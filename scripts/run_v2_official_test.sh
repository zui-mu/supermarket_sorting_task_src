#!/usr/bin/env bash
set -euo pipefail

HOST_ROOT="${HOST_ROOT:-$(pwd)}"
ROOT="${ROOT:-/workspace/baseline}"
LOG_DIR_HOST="${LOG_DIR_HOST:-${HOST_ROOT}/logs_official}"
LOG_DIR_CONTAINER="${LOG_DIR_CONTAINER:-${ROOT}/logs_official}"
DURATION_SEC="${DURATION_SEC:-600}"
SERVER_IMAGE="${SERVER_IMAGE:-supermarket_sorting:server}"
CLIENT_IMAGE="${CLIENT_IMAGE:-supermarket_sorting:client}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-supermarket_sorting_official}"
SERVER_STARTUP_SEC="${SERVER_STARTUP_SEC:-60}"
SERVER_READY_SEC="${SERVER_READY_SEC:-150}"
HEADLESS="${SUPERMARKET_HEADLESS:-1}"
MUJOCO_BACKEND="${MUJOCO_GL:-egl}"
# This helper is intentionally a formal-style entry point.  Keep its task
# shape deterministic even when a previous smoke test left shell variables set.
TASK_COUNT=5
TASK_ANONYMOUS=1
REQUIRE_ALL_COMPLETIONS="${SUPERMARKET_REQUIRE_ALL_COMPLETIONS:-1}"

if [[ "${SUPERMARKET_ALLOW_RUNTIME_LAYOUT:-0}" == "1" || "${SUPERMARKET_TEST_ORACLE:-0}" == "1" ]]; then
  echo "[official_test] refusing to run with development truth enabled" >&2
  echo "[official_test] use scripts/run_v2_smoke_test.sh for gt/runtime-layout tests" >&2
  exit 2
fi
if [[ "${SUPERMARKET_DETECT_BACKEND:-blob}" == "gt" ]]; then
  echo "[official_test] refusing the development-only gt detector" >&2
  echo "[official_test] use scripts/run_v2_smoke_test.sh for gt/runtime-layout tests" >&2
  exit 2
fi
if [[ "${SUPERMARKET_TASK_ANONYMOUS:-1}" == "1" && "${SUPERMARKET_DETECT_BACKEND:-yolo}" == "blob" ]]; then
  echo "[official_test] refusing anonymous tasks with the generic blob detector" >&2
  echo "[official_test] blob cannot identify product kind; provide a legal multi-class YOLO model" >&2
  echo "[official_test] use scripts/run_v2_smoke_test.sh only for non-formal debugging" >&2
  exit 2
fi

mkdir -p "${LOG_DIR_HOST}"

cleanup() {
  docker rm -f "${CONTAINER_PREFIX}_server" "${CONTAINER_PREFIX}_client" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup

server_mounts=(
  -v "${HOST_ROOT}:${ROOT}"
  -v supermarket_sorting_cache:/root/.cache
)
# Headless tests do not need WSLg.  Only attach its sockets when available so
# the same script also works in a remote/non-GUI WSL distribution.
if [[ -d /mnt/wslg && -d /tmp/.X11-unix ]]; then
  server_mounts+=(
    -v /mnt/wslg:/mnt/wslg
    -v /tmp/.X11-unix:/tmp/.X11-unix
  )
fi

# A detached bare `bash` is not a reliable container init process: on some
# Docker/WSL combinations it exits before the two exec'd ROS nodes start.
# Keep an explicit idle PID 1 for the full test lifecycle.
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
  -e MUJOCO_GL="${MUJOCO_BACKEND}" \
  -e SUPERMARKET_HEADLESS="${HEADLESS}" \
  -e SUPERMARKET_ENABLE_RENDER="${SUPERMARKET_ENABLE_RENDER:-1}" \
  -e SUPERMARKET_USE_GS="${SUPERMARKET_USE_GS:-1}" \
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e SUPERMARKET_ENABLE_SCORE=1 \
  -e SUPERMARKET_ENABLE_LIDAR="${SUPERMARKET_ENABLE_LIDAR:-1}" \
  -e SUPERMARKET_RANDOMIZE=1 \
  -e SUPERMARKET_RANDOMIZE_OBSTACLES=1 \
  -e SUPERMARKET_SEED="${SUPERMARKET_SEED:-11}" \
  -e SUPERMARKET_OBSTACLE_SEED="${SUPERMARKET_OBSTACLE_SEED:-11}" \
  -e SUPERMARKET_TASK_COUNT="${TASK_COUNT}" \
  -e SUPERMARKET_TASK_ANONYMOUS="${TASK_ANONYMOUS}" \
  -e SUPERMARKET_RENDER_FPS="${SUPERMARKET_RENDER_FPS:-6}" \
  -e SUPERMARKET_RENDER_WIDTH="${SUPERMARKET_RENDER_WIDTH:-480}" \
  -e SUPERMARKET_RENDER_HEIGHT="${SUPERMARKET_RENDER_HEIGHT:-360}" \
  "${server_mounts[@]}" \
  "${SERVER_IMAGE}" \
  bash -lc "cd ${ROOT} && ./scripts/run_v2_server.sh"

echo "[official_test] waiting for server task publication (up to ${SERVER_STARTUP_SEC}s)"
deadline=$((SECONDS + SERVER_STARTUP_SEC))
task_probe="${LOG_DIR_HOST}/task_probe.log"
rm -f "${task_probe}"
while (( SECONDS < deadline )); do
  running="$(docker inspect -f '{{.State.Running}}' "${CONTAINER_PREFIX}_server" 2>/dev/null || true)"
  if [[ "${running}" != "true" ]]; then
    echo "[official_test] server exited before publishing a task" >&2
    docker logs "${CONTAINER_PREFIX}_server" >&2 || true
    exit 1
  fi
  # See the smoke runner for why a timeout-wrapped `ros2 topic echo` is not a
  # trustworthy probe here. The decision client below verifies actual delivery.
  if docker logs "${CONTAINER_PREFIX}_server" 2>&1 | grep -q "\[server\] task published:"; then
    docker logs "${CONTAINER_PREFIX}_server" 2>&1 | grep "\[server\] task published:" | tail -n 1 >"${task_probe}"
    echo "[official_test] server task handshake passed"
    break
  fi
  sleep 1
done
if ! grep -q "\[server\] task published:" "${task_probe}"; then
  echo "[official_test] server did not publish /supermarket_sorting/task" >&2
  docker logs "${CONTAINER_PREFIX}_server" >&2 || true
  exit 1
fi

# The task publisher is created before the simulator finishes optional lidar
# extension setup.  Do not launch the client merely because the latched task
# exists: wait until live robot state is available, otherwise its watchdog
# correctly refuses to move and the test produces misleading empty metrics.
echo "[official_test] waiting for live odometry (up to ${SERVER_READY_SEC}s)"
ready_probe="${LOG_DIR_HOST}/odom_probe.log"
rm -f "${ready_probe}"
ready_deadline=$((SECONDS + SERVER_READY_SEC))
while (( SECONDS < ready_deadline )); do
  if docker exec "${CONTAINER_PREFIX}_server" bash -lc \
      "source /opt/ros/humble/setup.bash && timeout 5 ros2 topic echo /slamware_ros_sdk_server_node/odom --once" \
      >"${ready_probe}" 2>&1; then
    if grep -q "pose:" "${ready_probe}"; then
      echo "[official_test] server odometry handshake passed"
      break
    fi
  fi
  sleep 1
done
if ! grep -q "pose:" "${ready_probe}"; then
  echo "[official_test] server did not publish live odometry" >&2
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
  -e SUPERMARKET_ALLOW_RUNTIME_LAYOUT="${SUPERMARKET_ALLOW_RUNTIME_LAYOUT:-0}" \
  -e SUPERMARKET_DETECT_BACKEND="${SUPERMARKET_DETECT_BACKEND:-yolo}" \
  -e SUPERMARKET_YOLO_WEIGHTS="${SUPERMARKET_YOLO_WEIGHTS:-/workspace/baseline/examples/supermarket_sorting/perception/checkpoints/supermarket_multiclass.pt}" \
  -e SUPERMARKET_YOLO_REQUIRE_OFFICIAL_CLASSES=1 \
  -e SUPERMARKET_YOLO_DEVICE="${SUPERMARKET_YOLO_DEVICE:-auto}" \
  -e SUPERMARKET_BASELINE_WEIGHTS="${SUPERMARKET_BASELINE_WEIGHTS:-}" \
  -e SUPERMARKET_ENABLE_AVOIDANCE="${SUPERMARKET_ENABLE_AVOIDANCE:-1}" \
  -e SUPERMARKET_ENABLE_DEPTH_AVOIDANCE="${SUPERMARKET_ENABLE_DEPTH_AVOIDANCE:-1}" \
  -e SUPERMARKET_TASK_FALLBACK_ORDER="${SUPERMARKET_TASK_FALLBACK_ORDER:-}" \
  -e SUPERMARKET_TEST_ORACLE="${SUPERMARKET_TEST_ORACLE:-0}" \
  -e SUPERMARKET_REQUEST_SERVER_RESET="${SUPERMARKET_REQUEST_SERVER_RESET:-0}" \
  -e SUPERMARKET_TASK_WAIT_TIMEOUT="${SUPERMARKET_TASK_WAIT_TIMEOUT:-60}" \
  -e SUPERMARKET_SEARCH_CLASS_MIN_SAMPLES="${SUPERMARKET_SEARCH_CLASS_MIN_SAMPLES:-3}" \
  -e SUPERMARKET_SEARCH_CLASS_MIN_RATIO="${SUPERMARKET_SEARCH_CLASS_MIN_RATIO:-0.67}" \
  -e SUPERMARKET_MISSION_METRICS="${ROOT}/examples/supermarket_sorting/mission_metrics.jsonl" \
  -v "${HOST_ROOT}:${ROOT}" \
  -v supermarket_sorting_cache:/root/.cache \
  "${CLIENT_IMAGE}" \
  bash -lc 'trap "exit 0" TERM INT; while true; do sleep 3600; done' >/dev/null

docker exec -d "${CONTAINER_PREFIX}_client" bash -lc \
  "cd ${ROOT} && mkdir -p ${LOG_DIR_CONTAINER} && SUPERMARKET_ALLOW_RUNTIME_LAYOUT=${SUPERMARKET_ALLOW_RUNTIME_LAYOUT:-0} SUPERMARKET_DETECT_BACKEND=${SUPERMARKET_DETECT_BACKEND:-yolo} ./scripts/run_v2_perception.sh > ${LOG_DIR_CONTAINER}/perception.log 2>&1"

# CUDA model construction is asynchronous.  Wait for one actual detection
# heartbeat before starting the decision node, otherwise a healthy model can
# be mistaken for an empty anonymous slot while it is still warming up.
echo "[official_test] waiting for first perception heartbeat (up to 60s)"
perception_ready=0
perception_deadline=$((SECONDS + 60))
while (( SECONDS < perception_deadline )); do
  if docker exec "${CONTAINER_PREFIX}_client" bash -lc \
      "source /opt/ros/humble/setup.bash && timeout 4 ros2 topic echo /supermarket_sorting/detections --once" \
      >/dev/null 2>&1; then
    perception_ready=1
    echo "[official_test] perception heartbeat passed"
    break
  fi
  sleep 1
done
if (( perception_ready == 0 )); then
  echo "[official_test] perception published no heartbeat; inspect ${LOG_DIR_HOST}/perception.log" >&2
  docker exec "${CONTAINER_PREFIX}_client" bash -lc \
    "cat ${LOG_DIR_CONTAINER}/perception.log" \
    | tee "${LOG_DIR_HOST}/perception_startup_failure.log" >&2 || true
  exit 1
fi
docker exec -d "${CONTAINER_PREFIX}_client" bash -lc \
  "cd ${ROOT} && mkdir -p ${LOG_DIR_CONTAINER} && ./scripts/run_v2_decision_client.sh > ${LOG_DIR_CONTAINER}/decision_client.log 2>&1"

echo "Official test running for ${DURATION_SEC}s. Logs: ${LOG_DIR_HOST}"
run_deadline=$((SECONDS + DURATION_SEC))
while (( SECONDS < run_deadline )); do
  server_running="$(docker inspect -f '{{.State.Running}}' "${CONTAINER_PREFIX}_server" 2>/dev/null || true)"
  client_running="$(docker inspect -f '{{.State.Running}}' "${CONTAINER_PREFIX}_client" 2>/dev/null || true)"
  if [[ "${server_running}" != "true" ]]; then
    echo "[official_test] server exited during run; stopping early" >&2
    docker logs "${CONTAINER_PREFIX}_server" > "${LOG_DIR_HOST}/server.log" 2>&1 || true
    docker cp "${CONTAINER_PREFIX}_client:${LOG_DIR_CONTAINER}/perception.log" \
      "${LOG_DIR_HOST}/perception.log" >/dev/null 2>&1 || true
    docker cp "${CONTAINER_PREFIX}_client:${LOG_DIR_CONTAINER}/decision_client.log" \
      "${LOG_DIR_HOST}/decision_client.log" >/dev/null 2>&1 || true
    exit 1
  fi
  if [[ "${client_running}" != "true" ]]; then
    echo "[official_test] client exited during run; stopping early" >&2
    docker logs "${CONTAINER_PREFIX}_server" > "${LOG_DIR_HOST}/server.log" 2>&1 || true
    exit 1
  fi
  sleep 2
done

docker logs "${CONTAINER_PREFIX}_server" > "${LOG_DIR_HOST}/server.log" 2>&1 || true
# The client container is intentionally removed by cleanup().  Copy its two
# node logs before that happens; otherwise a successful formal run leaves only
# a server log and later diagnosis accidentally reads a previous run.
docker cp "${CONTAINER_PREFIX}_client:${LOG_DIR_CONTAINER}/perception.log" \
  "${LOG_DIR_HOST}/perception.log" >/dev/null 2>&1 || true
docker cp "${CONTAINER_PREFIX}_client:${LOG_DIR_CONTAINER}/decision_client.log" \
  "${LOG_DIR_HOST}/decision_client.log" >/dev/null 2>&1 || true
docker exec "${CONTAINER_PREFIX}_client" bash -lc \
  "cd ${ROOT} && python3 examples/supermarket_sorting/analyze_mission_metrics.py" \
  | tee "${LOG_DIR_HOST}/mission_summary.txt" || true

# A healthy process is not evidence of a successful formal run.  Read the
# referee's authoritative completion count before cleanup and fail closed when
# the requested batch was not delivered.
referee_state_log="${LOG_DIR_HOST}/referee_final_state.log"
docker exec "${CONTAINER_PREFIX}_server" bash -lc \
  "source /opt/ros/humble/setup.bash && timeout 8 ros2 topic echo /referee/state --once" \
  >"${referee_state_log}" 2>&1 || true
completed_count="$(python3 -c '
import re
import sys
match = re.search(r"\"completed\"\\s*:\\s*(\\d+)", sys.stdin.read())
print(match.group(1) if match else "0")
' <"${referee_state_log}")"
echo "[official_test] referee completed=${completed_count}/${TASK_COUNT}"
if [[ "${REQUIRE_ALL_COMPLETIONS}" == "1" ]] && (( completed_count < TASK_COUNT )); then
  echo "[official_test] incomplete formal run; refusing success status" >&2
  exit 1
fi
