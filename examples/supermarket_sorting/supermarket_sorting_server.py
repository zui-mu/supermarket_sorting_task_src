#!/usr/bin/env python3
"""超市分拣任务的 ROS2 server。

加载 retail_competition 场景，复用本仓库 examples/ros2/mmk2_ros2.py 的
MMK2ROS2 发布相机、里程计、关节状态等标准话题，供
supermarket_sorting_client.py 控制机器人完成抓取放置。
"""
import json
import math
import os
import sys
import random
import secrets
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger

# 可迁移:从脚本自身位置推导示例目录和仓库根目录
TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
ROS2_EXAMPLES_DIR = REPO_ROOT / "examples" / "ros2"
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
if str(ROS2_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(ROS2_EXAMPLES_DIR))

ASSETS_DIR = TASK_DIR / "models"
os.environ["DISCOVERSE_ASSETS_DIR"] = str(ASSETS_DIR)

from discoverse.robots_env.mmk2_base import MMK2Cfg
from mmk2_ros2 import MMK2ROS2
from referee import Referee

SOURCE_XML = TASK_DIR / "mjcf" / "retail_competition.xml"
RUNTIME_XML = Path("/tmp/retail_competition_ros2_runtime.xml")
LAYOUT_JSON = TASK_DIR / "retail_competition_layout.json"
RUNTIME_LAYOUT_JSON = TASK_DIR / "runtime_layout.json"
REFEREE_CONFIG_JSON = TASK_DIR / "referee_json" / "retail_referee_config.json"
START_XY = np.array([1.92, -3.17], dtype=float)   # 出发区


def env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default, minimum=1):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        print(f"[server] invalid {name}={value!r}; using {default}")
        return default


def local_robot_gs_model_dict():
    gs_model_dict = {}
    for name, path in MMK2Cfg.gs_model_dict.items():
        if path.startswith("mobile_chassis/mmk2/"):
            gs_model_dict[name] = path.replace("mobile_chassis/mmk2/", "mmk2/")
        elif path.startswith("manipulator/airbot_play/"):
            gs_model_dict[name] = path.replace("manipulator/airbot_play/", "airbot_play/")
        else:
            gs_model_dict[name] = path
    return gs_model_dict


def resolve_background_ply():
    """Background 3DGS model (drawn at world identity; no MuJoCo body).

    SUPERMARKET_BACKGROUND_PLY overrides the path (relative to models/3dgs/, or
    absolute) -- handy while tuning tools_align_background.py knobs. Otherwise use
    the aligned scan retail_background_fit.ply if it has been baked, else fall back
    to the tiny dummy_background.ply placeholder."""
    override = os.getenv("SUPERMARKET_BACKGROUND_PLY")
    if override:
        return override
    fit = ASSETS_DIR / "3dgs" / "shentoon" / "retail_background_fit.ply"
    if fit.exists():
        return "shentoon/retail_background_fit.ply"
    return "shentoon/dummy_background.ply"


# 货架每层台面高度(世界 z)。物体静止 z = 台面 + 该物体自身半高。
SHELF_SURFACE = {"L1": 0.499, "L2": 0.851, "L3": 1.189}
DEFAULT_REFEREE_TARGETS = "all"
OBSTACLE_BODY_NAMES = tuple(
    f"dynamic_obstacle_box_{index:02d}" for index in range(1, 6)
)
OBSTACLE_ZONE_ORIGIN = np.array([-0.96, -1.01], dtype=float)


def write_runtime_xml(pos_overrides=None, obstacle_overrides=None):
    """Render the runtime MJCF. If pos_overrides is given, rewrite each named
    body''s pos="x y z" so the whole body (collision geom + gs ply travel
    together) moves to its randomized shelf slot. obstacle_overrides entries
    are (x, y, z, yaw) tuples: the body pos is rewritten AND a quat="w x y z"
    attribute is appended, randomizing the box yaw like the official scene."""
    text = SOURCE_XML.read_text().replace("__REPO_ROOT__", str(TASK_DIR))
    if pos_overrides or obstacle_overrides:
        import re
        import math
        pos_all = dict(pos_overrides or {})
        quat_all = {}
        for body_name, entry in (obstacle_overrides or {}).items():
            x, y, z, yaw = (float(value) for value in entry[:4])
            pos_all[body_name] = (x, y, z)
            half = yaw / 2.0
            quat_all[body_name] = (math.cos(half), 0.0, 0.0, math.sin(half))
        for body_name, (x, y, z) in pos_all.items():
            pattern = re.compile(
                r'(<body name="' + re.escape(body_name) + r'"[^>]*?pos=")[^"]*(")'
            )
            replacement = rf"\g<1>{x:.5f} {y:.5f} {z:.5f}\g<2>"
            if body_name in quat_all:
                w, qx, qy, qz = quat_all[body_name]
                replacement += f' quat="{w:.6f} {qx:.6f} {qy:.6f} {qz:.6f}"'
            text, n = pattern.subn(replacement, text)
            if n != 1:
                raise RuntimeError(
                    f"randomize: expected exactly 1 body pos for {body_name}, got {n}")
    RUNTIME_XML.write_text(text)
    return str(RUNTIME_XML)


def randomize_obstacle_positions(seed):
    """Sample collision-safe box poses in the open corridor.

    The obstacle bodies are children of ``dynamic_obstacle_corridor``.  The
    returned coordinates are therefore local to that body, while sampling is
    done in world coordinates so the keep-out zones are easy to audit.  The
    shelf, divider, delivery table, start pocket, and delivery goal are kept
    clear.  Like the official scene, each box also gets a random yaw; a coarse
    A* check guarantees that a route from the shelf side to the delivery goal
    still exists (falling back to axis-aligned boxes if necessary).  LaserScan
    remains the runtime source of obstacle observations; this function only
    makes the local simulation match the advertised random obstacle mode.

    Returns {body: (x_local, y_local, z, yaw)}.
    """
    import math
    import random

    rng = random.Random(int(seed))
    xmin, xmax = -1.30, 0.10
    ymin, ymax = -2.55, 1.95
    table_rect = (-2.42, -1.46, -3.63, -3.19)
    start = np.array([1.92, -3.17], dtype=float)
    goal = np.array([-1.88, -2.80], dtype=float)
    # Rotated boxes need a slightly larger centre-to-centre keep-out than the
    # axis-aligned 0.78 m (half-diagonal is ~0.36 m).
    min_separation = 0.82
    # The east-west crossing band north of the centre divider (y > 1.70) and
    # the shelf-clear descent line (y≈2.06) is the staged delivery corridor.
    # Mirror the official "shelf entrance -> delivery entrance path exists"
    # guarantee by keeping that band box-free; a box there pins the loaded
    # robot during the crossing turn (verified in simulation).
    corridor_band = (xmin, xmax, 1.35, 2.70)

    def sample_set(with_yaw):
        placed: list[tuple[np.ndarray, float]] = []
        for _ in range(1200):
            candidate = np.array([
                rng.uniform(xmin, xmax),
                rng.uniform(ymin, ymax),
            ])
            yaw = rng.uniform(-math.pi, math.pi) if with_yaw else 0.0
            if np.linalg.norm(candidate - start) < 0.90 or np.linalg.norm(candidate - goal) < 0.75:
                continue
            if (
                corridor_band[0] <= candidate[0] <= corridor_band[1]
                and corridor_band[2] <= candidate[1] <= corridor_band[3]
            ):
                continue
            if (
                table_rect[0] - 0.35 <= candidate[0] <= table_rect[1] + 0.35
                and table_rect[2] - 0.35 <= candidate[1] <= table_rect[3] + 0.35
            ):
                continue
            if any(np.linalg.norm(candidate - previous) < min_separation
                   for previous, _ in placed):
                continue
            placed.append((candidate, yaw))
            if len(placed) == len(OBSTACLE_BODY_NAMES):
                break
        if len(placed) != len(OBSTACLE_BODY_NAMES):
            return None
        return placed

    chosen = None
    for _ in range(12):
        placed = sample_set(with_yaw=True)
        if placed is None:
            continue
        if obstacle_path_exists([point for point, _ in placed]):
            chosen = placed
            break
    if chosen is None:
        for _ in range(12):
            placed = sample_set(with_yaw=False)
            if placed is None:
                continue
            if obstacle_path_exists([point for point, _ in placed]):
                chosen = placed
                break
    if chosen is None:
        placed = sample_set(with_yaw=False)
        chosen = placed or []
        print(
            "[server] warning: no obstacle layout with a verifiable clear "
            "corridor was found; falling back to axis-aligned boxes "
            "(client A* avoidance must handle it)"
        )
    return {
        body: (
            float(point[0] - OBSTACLE_ZONE_ORIGIN[0]),
            float(point[1] - OBSTACLE_ZONE_ORIGIN[1]),
            0.25,
            float(yaw),
        )
        for body, (point, yaw) in zip(OBSTACLE_BODY_NAMES, chosen)
    }


def obstacle_path_exists(points, shelf_side=(0.85, 2.30), delivery_goal=(-1.88, -2.74)):
    """Coarse connectivity check: a grid A* route from the shelf side to the
    delivery goal must exist around the placed boxes. Returns True when the
    planner is unavailable (then the caller keeps the sampled layout)."""
    try:
        from navigation.grid_planner import SupermarketGridPlanner
    except ImportError:
        return True
    planner = SupermarketGridPlanner(
        resolution=0.20,
        robot_radius=0.22,
        corridor_clearance=0.30,
        # Points are box CENTRES: the loaded chassis needs the box
        # half-diagonal (0.36 m) + chassis half (0.22 m) + margin, so 0.45 m
        # inflation accepted layouts where the boxes formed an impassable
        # diagonal wall (verified: boxes 0.89 m apart, gap for a 0.44 m
        # chassis only 0.03 m).
        dynamic_clearance=0.65,
    )
    dynamic = [[float(p[0]), float(p[1])] for p in points]
    route = planner.plan(shelf_side, delivery_goal, dynamic)
    if not route:
        route = planner.plan(delivery_goal, shelf_side, dynamic)
    return bool(route)


def write_runtime_layout_json(layout):
    """Persist the runtime layout so every node sees the same live world."""
    try:
        RUNTIME_LAYOUT_JSON.write_text(
            json.dumps(layout, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(RUNTIME_LAYOUT_JSON)
    except OSError as exc:
        print(f"[server] failed to export runtime layout: {exc}")
        return None


def randomize_positions(layout, seed=None, anchored_body=None):
    """Shuffle which shelf slot each object body occupies.

    Each body keeps its own collision geom AND its own gs ply (they stay bound);
    only the body's world position moves to another slot.  The new z is the new
    shelf surface plus the body's intrinsic half-height (derived from its
    original z), so the object rests on the shelf instead of clipping/floating.
    By default every product body participates in the shuffle.  An anchored body
    can be supplied explicitly for debugging, but normal competition startup
    should leave it unset.

    Returns (new_layout, pos_overrides) where pos_overrides maps body name ->
    (x, y, z) for the runtime MJCF rewrite.
    """
    import random
    rng = random.Random(seed)
    # target slot positions (x, y, level) taken from the original layout
    slots = [(s["world_position"][0], s["world_position"][1], s["level"]) for s in layout]
    # each body's intrinsic half-height = original z - its original shelf surface
    half_h = [s["world_position"][2] - SHELF_SURFACE[s["level"]] for s in layout]

    anchored_i = None
    if anchored_body:
        for i, slot in enumerate(layout):
            if slot["body"] == anchored_body:
                anchored_i = i
                break
        if anchored_i is None:
            raise RuntimeError(f"randomize: anchored body not found: {anchored_body}")

    body_indices = list(range(len(layout)))
    slot_indices = list(range(len(layout)))
    if anchored_i is not None:
        body_indices.remove(anchored_i)
        slot_indices.remove(anchored_i)

    # Make the shuffle a derangement: if a debug anchor is set it stays put,
    # and every other product is forced into a different slot.
    rng.shuffle(slot_indices)
    fixed = [i for i, (body_i, slot_i) in enumerate(zip(body_indices, slot_indices))
             if body_i == slot_i]
    if len(fixed) == 1 and len(slot_indices) > 1:
        i = fixed[0]
        j = 0 if i != 0 else 1
        slot_indices[i], slot_indices[j] = slot_indices[j], slot_indices[i]
    elif len(fixed) > 1:
        fixed_slots = [slot_indices[i] for i in fixed]
        fixed_slots = fixed_slots[1:] + fixed_slots[:1]
        for i, slot_i in zip(fixed, fixed_slots):
            slot_indices[i] = slot_i
    order = {body_i: slot_i for body_i, slot_i in zip(body_indices, slot_indices)}
    if anchored_i is not None:
        order[anchored_i] = anchored_i

    new_layout, pos_overrides = [], {}
    for body_i in range(len(layout)):
        slot_i = order[body_i]
        x, y, level = slots[slot_i]
        z = SHELF_SURFACE[level] + half_h[body_i]
        body = layout[body_i]["body"]
        pos_overrides[body] = (x, y, z)
        ns = layout[body_i].copy()
        for key in ("shelf", "level", "column", "aruco_id"):
            ns[key] = layout[slot_i][key]
        ns["world_position"] = [x, y, z]
        new_layout.append(ns)
    return new_layout, pos_overrides


def select_targets(layout):
    """Return the referee-selected body ids for this run."""
    spec = os.getenv("SUPERMARKET_TARGETS", DEFAULT_REFEREE_TARGETS).strip()
    if not spec or spec.lower() in {"all", "*"}:
        return [slot["body"] for slot in layout]
    tokens = [t.strip() for t in spec.replace(";", ",").split(",") if t.strip()]
    bodies = {slot["body"] for slot in layout}
    kinds = {}
    for slot in layout:
        kinds.setdefault(slot["object_kind"], []).append(slot["body"])
    out = []
    for tok in tokens:
        if tok in bodies:
            out.append(tok)
        elif tok in kinds:
            out.extend(kinds[tok])
    seen, res = set(), []
    for b in out:
        if b not in seen:
            seen.add(b)
            res.append(b)
    return res


def limit_targets(target_bodies):
    """Optional top-N limit for local official-style runs."""
    count_spec = os.getenv("SUPERMARKET_TASK_COUNT", "").strip()
    if not count_spec or count_spec.lower() in {"all", "*"}:
        return list(target_bodies)
    try:
        count = max(1, int(count_spec))
    except ValueError:
        print(f"[server] invalid SUPERMARKET_TASK_COUNT={count_spec!r}; using all targets")
        return list(target_bodies)
    target_bodies = list(target_bodies)
    if count >= len(target_bodies):
        return target_bodies
    if env_flag("SUPERMARKET_TASK_SAMPLE", False):
        seed_str = os.getenv("SUPERMARKET_TASK_SAMPLE_SEED", os.getenv("SUPERMARKET_SEED", "11"))
        try:
            seed = int(seed_str)
        except ValueError:
            seed = 11
        sampled = random.Random(seed).sample(target_bodies, count)
        print(f"[server] sampled {count} referee targets with seed={seed}")
        return sampled
    return target_bodies[:count]


def build_task_payload(layout, target_bodies):
    """Publish only the referee-selected targets, not the whole shelf layout.

    Matches the official V2.0 message shape: ``id`` is the anonymous
    ``item_<run_prefix>_NN`` string and ``run_prefix`` is a fresh hex token per
    game, so clients can never cache or derive shelf positions from ids.
    """
    body_map = {slot["body"]: slot for slot in layout}
    anonymous = env_flag("SUPERMARKET_TASK_ANONYMOUS", False)
    run_prefix = os.getenv("SUPERMARKET_RUN_PREFIX")
    if not run_prefix:
        run_prefix = "run_" + secrets.token_hex(6)
    targets = []
    missing = []
    for index, body in enumerate(target_bodies, 1):
        slot = body_map.get(body)
        if slot is None:
            missing.append(body)
            continue
        targets.append(
            {
                "id": f"item_{run_prefix}_{index:02d}" if anonymous else body,
                "kind": slot["object_kind"],
            }
        )
    if missing:
        print(f"[server] warning: referee target bodies not found in layout: {missing}")
    return json.dumps(
        {
            "schema_version": 1,
            # A fresh token per payload keeps repeated client processes (and
            # clients restarting after /reset_run) from treating a new game as
            # the previous run's latched task.
            "run_prefix": run_prefix,
            "count": len(targets),
            "targets": targets,
        },
        ensure_ascii=False,
    )


def build_config():
    cfg = MMK2Cfg()
    cfg.use_gaussian_renderer = env_flag("SUPERMARKET_USE_GS", True)
    cfg.enable_render = env_flag("SUPERMARKET_ENABLE_RENDER", True)
    cfg.headless = env_flag("SUPERMARKET_HEADLESS", False)

    # 货架场景的 3DGS 绑定:保留 MMK2Cfg 默认的机器人 link 绑定,追加 background + 货架物体
    layout = json.loads(LAYOUT_JSON.read_text())

    # 随机摆放功能(默认开启,给选手用):整把物体(碰撞geom+3DGS一起)随机搬到别的货架格子
    pos_overrides = None
    if env_flag("SUPERMARKET_RANDOMIZE", True):
        seed_str = os.getenv("SUPERMARKET_SEED")
        try:
            seed = int(seed_str) if seed_str else None
        except ValueError:
            print(f"[server] invalid SUPERMARKET_SEED={seed_str!r}; using random")
            seed = None
        anchored_body = os.getenv("SUPERMARKET_ANCHORED_TARGET", "").strip() or None
        layout, pos_overrides = randomize_positions(layout, seed, anchored_body)
        anchored_msg = anchored_body if anchored_body else "none"
        print(f"[server] randomized object positions (seed={seed}, anchored={anchored_msg})")
    else:
        print("[server] fixed layout (SUPERMARKET_RANDOMIZE=0)")

    obstacle_overrides = None
    if env_flag("SUPERMARKET_RANDOMIZE_OBSTACLES", False):
        seed_str = os.getenv("SUPERMARKET_OBSTACLE_SEED", os.getenv("SUPERMARKET_SEED", "11"))
        try:
            obstacle_seed = int(seed_str) + 1000003
        except ValueError:
            obstacle_seed = 1000014
        obstacle_overrides = randomize_obstacle_positions(obstacle_seed)
        world_positions = {
            body: [
                round(x + OBSTACLE_ZONE_ORIGIN[0], 3),
                round(y + OBSTACLE_ZONE_ORIGIN[1], 3),
                round(math.degrees(yaw), 0),
            ]
            for body, (x, y, _z, yaw) in obstacle_overrides.items()
        }
        print(
            f"[server] randomized corridor obstacles (seed={obstacle_seed}): "
            f"{world_positions}"
        )
    cfg.mjcf_file_path = write_runtime_xml(pos_overrides, obstacle_overrides)
    cfg.runtime_layout_path = write_runtime_layout_json(layout)
    if cfg.runtime_layout_path:
        print(f"[server] runtime layout exported: {cfg.runtime_layout_path}")

    cfg.obj_list = [slot["body"] for slot in layout]
    cfg.gs_model_dict = local_robot_gs_model_dict()
    cfg.gs_model_dict["background"] = resolve_background_ply()
    for slot in layout:
        cfg.gs_model_dict[slot["body"]] = slot["gs_ply"]

    # The sorting baseline consumes only the head RGB-D camera. Rendering all
    # three cameras every frame makes simulation time several times slower.
    cfg.obs_rgb_cam_id = [0, 1, 2] if env_flag("SUPERMARKET_RENDER_ALL_CAMERAS", False) else [0]
    cfg.obs_depth_cam_id = [0]
    # The official image currently omits the optional mujoco_lidar package.
    cfg.lidar_s2_sim = env_flag("SUPERMARKET_ENABLE_LIDAR", False)
    if cfg.lidar_s2_sim:
        try:
            __import__("mujoco_lidar")
        except ImportError:
            print("[server] mujoco_lidar unavailable; falling back to depth-camera safety")
            cfg.lidar_s2_sim = False
    cfg.render_set = {
        "fps": env_int("SUPERMARKET_RENDER_FPS", 12),
        "width": env_int("SUPERMARKET_RENDER_WIDTH", 640, minimum=160),
        "height": env_int("SUPERMARKET_RENDER_HEIGHT", 480, minimum=120),
    }

    # 起始位姿:出发区,朝北(+Y)
    cfg.init_state["base_position"] = [float(START_XY[0]), float(START_XY[1]), 0.0]
    cfg.init_state["base_orientation"] = Rotation.from_euler("z", np.pi / 2.0).as_quat()[[3, 0, 1, 2]].tolist()

    # 裁判系统(默认关闭,SUPERMARKET_ENABLE_SCORE=1 开启)
    cfg.referee_enable = env_flag("SUPERMARKET_ENABLE_SCORE", False)
    cfg.referee_objects = list(cfg.obj_list)
    cfg.referee_targets = limit_targets(select_targets(layout))
    cfg.referee_config_path = str(REFEREE_CONFIG_JSON)
    cfg.supermarket_task_payload = build_task_payload(layout, cfg.referee_targets)
    # 保留布局供 reset_run 时重建带新 run_prefix 的任务消息
    cfg.supermarket_task_layout = layout
    return cfg


class ScoredMMK2ROS2(MMK2ROS2):
    """在 MMK2ROS2 基础上挂裁判:每物理步 update,发布 /referee/* 话题。"""
    def __init__(self, config):
        super().__init__(config)
        self._run_reset_lock = threading.Lock()
        self._run_reset_requested = False
        self._run_reset_done = threading.Event()
        self._run_reset_done.set()
        self.create_service(Trigger, "/supermarket_sorting/reset_run", self._reset_run_cb)
        self.referee = None
        if getattr(config, "referee_enable", False):
            cfg_path = getattr(config, "referee_config_path", None)
            if not (cfg_path and os.path.exists(cfg_path)):
                cfg_path = None
            self.referee = Referee(self.mj_model, config.referee_targets,
                                   config.referee_objects, cfg_path)
            task_qos = QoSProfile(depth=1)
            task_qos.reliability = ReliabilityPolicy.RELIABLE
            task_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self.task_puber = self.create_publisher(String, "/supermarket_sorting/task", task_qos)
            self._task_payload = getattr(config, "supermarket_task_payload", "")
            self._task_payload_logged = False
            self._last_task_publish_wall = 0.0
            # Publish once immediately as well as from the periodic referee
            # timer. TRANSIENT_LOCAL retains it for clients that start later,
            # while the immediate publish removes an executor startup gap.
            self._publish_task_payload()
            self.game_info_puber = self.create_publisher(String, "/referee/gameinfo", 2)
            self.state_puber = self.create_publisher(String, "/referee/state", 2)
            self.score_puber = self.create_publisher(Int32, "/referee/score", 2)
            self.taskinfo_puber = self.create_publisher(String, "/referee/taskinfo", 2)
            self._taskinfo = "targets(%d): %s" % (
                len(config.referee_targets), ", ".join(config.referee_targets))
            self.create_timer(0.5, self._pub_referee)
            print("[server] referee enabled; %s" % self._taskinfo)

    def post_physics_step(self):
        super().post_physics_step()
        if self._take_reset_request():
            self._reset_run_now()
        # On the V2 image, optional lidar/renderer initialization can delay
        # the rclpy executor even while physics and odometry are already live.
        # Republish the latched anonymous task from this loop so a later
        # decision client never waits forever for the one startup publication.
        if time.monotonic() - getattr(self, "_last_task_publish_wall", 0.0) >= 1.0:
            self._publish_task_payload()
        if self.referee is not None:
            self.referee.update(self.mj_data)

    def _publish_task_payload(self):
        if getattr(self, "task_puber", None) is None or not self._task_payload:
            return
        self.task_puber.publish(String(data=self._task_payload))
        self._last_task_publish_wall = time.monotonic()
        if not self._task_payload_logged:
            print(f"[server] task published: {self._task_payload}", flush=True)
            self._task_payload_logged = True

    def _take_reset_request(self):
        with self._run_reset_lock:
            if not self._run_reset_requested:
                return False
            self._run_reset_requested = False
            return True

    def _reset_run_now(self):
        self.reset()
        if self.referee is not None:
            self.referee.reset(self.mj_data)
        # A reset starts a fresh game. Refresh the latched task payload with a
        # new run_prefix so a restarted client cannot deduplicate it as the
        # previous run's task and keep its stale plan.
        layout = getattr(self.config, "supermarket_task_layout", None)
        if layout is not None and getattr(self.config, "referee_targets", None):
            try:
                self._task_payload = build_task_payload(
                    layout, self.config.referee_targets)
                self._task_payload_logged = False
                self._last_task_publish_wall = 0.0
                self._publish_task_payload()
            except Exception as exc:  # noqa: BLE001 - reset must still complete
                print(f"[server] reset_run: task payload refresh failed: {exc}")
        self._run_reset_done.set()
        print("[server] reset_run complete")

    def _reset_run_cb(self, request, response):
        del request
        with self._run_reset_lock:
            self._run_reset_requested = True
            self._run_reset_done.clear()
        if self._run_reset_done.wait(timeout=2.0):
            response.success = True
            response.message = "reset_run complete"
        else:
            # Be honest: the physics loop has not confirmed the reset yet.
            response.success = False
            response.message = "reset_run not confirmed within timeout"
        return response

    def _pub_referee(self):
        if self.referee is None:
            return
        self._publish_task_payload()
        self.taskinfo_puber.publish(String(data=self._taskinfo))
        self.game_info_puber.publish(String(data=self.referee.game_info))
        self.state_puber.publish(String(data=json.dumps(self.referee.state_dict, ensure_ascii=False)))
        self.score_puber.publish(Int32(data=int(self.referee.total_score)))


def spin_node(node):
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, RCLError):
        pass


def main():
    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    exec_node = ScoredMMK2ROS2(build_config())
    try:
        exec_node.reset()
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        # reset() renders one frame (enable_render). A transient 3DGS/EGL
        # context failure there must not prevent the whole match: physics is
        # already reset; render() will retry from the main loop.
        print("[server] initial render failed (continuing physics-only): %r" % exc,
              file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()

    spin_thread = threading.Thread(target=spin_node, args=(exec_node,), daemon=True)
    spin_thread.start()

    topic_hz = env_int("SUPERMARKET_ROS_HZ", 24)
    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(topic_hz,), daemon=True)
    pubtopic_thread.start()

    if exec_node.config.lidar_s2_sim:
        publidar_thread = threading.Thread(target=exec_node.thread_publidartopic, args=(12,), daemon=True)
        publidar_thread.start()

    try:
        pace = env_flag("SUPERMARKET_REALTIME_PACING", True)
        previous_sim = None
        sim_origin = None
        sim_wall_origin = time.monotonic()
        running_guard = 0
        loop_failures = 0
        # Persistent loop: discoverse's renderer flips exec_node.running=False
        # when glfw.window_should_close() returns True. Under WSLg/flaky
        # rendering contexts that can happen spuriously and silently kill the
        # whole simulation mid-test (verified: server Exited(0) right after
        # "referee results saved", leaving the client waiting on stale odom).
        # mmk2_ros2 patches window_should_close; this re-assertion is the
        # second line of defence so the match only ends on rclpy shutdown.
        while rclpy.ok():
            if not exec_node.running:
                exec_node.running = True
                if running_guard % 240 == 0:
                    print("[server] exec_node.running flipped False while ROS is up; "
                          "re-asserting and continuing physics", flush=True)
                running_guard += 1
            try:
                exec_node.physics_step()
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                # A single bad physics/render step must not end the match.
                # On Docker Desktop the 3DGS/EGL renderer can throw transient
                # context errors: physics already advanced (mj_step runs before
                # render in simulator.step()), so the right move is to log
                # rate-limited, keep stepping, and let the client + referee
                # finish.  Only a pathological streak ends the run.
                loop_failures += 1
                if loop_failures == 1 or loop_failures % 240 == 0:
                    print("[server] physics loop step failed (%d consecutive): %r"
                          % (loop_failures, exc), file=sys.stderr, flush=True)
                    if loop_failures <= 240:
                        import traceback
                        traceback.print_exc()
                if loop_failures >= 20000:
                    print("[server] too many consecutive physics failures; ending run",
                          file=sys.stderr, flush=True)
                    break
                continue
            loop_failures = 0
            if pace:
                # The referee's time limit runs in simulation time. Without
                # pacing (headless/no-render), the loop would race far ahead
                # of wall time and end the match early in wall-clock terms.
                # Sleeping only the *excess* makes this a no-op whenever the
                # renderer is already pacing at real time.
                try:
                    sim_t = float(exec_node.mj_data.time)
                except Exception:
                    sim_t = None
                if sim_t is not None:
                    if previous_sim is not None and sim_t < previous_sim:
                        # A reset restarted the simulation clock: re-baseline.
                        sim_origin = sim_t
                        sim_wall_origin = time.monotonic()
                    if sim_origin is None:
                        sim_origin = sim_t
                    previous_sim = sim_t
                    ahead = (sim_t - sim_origin) - (time.monotonic() - sim_wall_origin)
                    if ahead > 0.05:
                        time.sleep(min(ahead, 0.25))
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(exec_node, "referee", None) is not None:
            try:
                exec_node.referee.save_results()
            except Exception as exc:   # noqa: BLE001 - 落盘失败不应掩盖退出
                print("[server] referee save_results failed: %s" % exc)
        exec_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
