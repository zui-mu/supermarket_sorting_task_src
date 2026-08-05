#!/usr/bin/env python3
"""超市分拣任务的 ROS2 server。

加载 retail_competition 场景，复用本仓库 examples/ros2/mmk2_ros2.py 的
MMK2ROS2 发布相机、里程计、关节状态等标准话题，供
supermarket_sorting_client.py 控制机器人完成抓取放置。
"""
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger

# 可迁移:从脚本自身位置推导示例目录和仓库根目录
TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
ROS2_EXAMPLES_DIR = REPO_ROOT / "examples" / "ros2"
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


def write_runtime_xml(pos_overrides=None):
    """Render the runtime MJCF. If pos_overrides is given, rewrite each named
    body''s pos="x y z" so the whole body (collision geom + gs ply travel
    together) moves to its randomized shelf slot."""
    text = SOURCE_XML.read_text().replace("__REPO_ROOT__", str(TASK_DIR))
    if pos_overrides:
        import re
        for body_name, (x, y, z) in pos_overrides.items():
            pattern = re.compile(
                r'(<body name="' + re.escape(body_name) + r'"[^>]*?pos=")[^"]*(")'
            )
            text, n = pattern.subn(rf"\g<1>{x:.5f} {y:.5f} {z:.5f}\g<2>", text)
            if n != 1:
                raise RuntimeError(
                    f"randomize: expected exactly 1 body pos for {body_name}, got {n}")
    RUNTIME_XML.write_text(text)
    return str(RUNTIME_XML)


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
    """本局可计分目标(body 名)。默认全场商品均可计分；
    可用 SUPERMARKET_TARGETS 覆盖(逗号分隔的 object_kind 或具体 body 名)。"""
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
        seed = int(seed_str) if seed_str and seed_str.isdigit() else None
        anchored_body = os.getenv("SUPERMARKET_ANCHORED_TARGET", "").strip() or None
        layout, pos_overrides = randomize_positions(layout, seed, anchored_body)
        anchored_msg = anchored_body if anchored_body else "none"
        print(f"[server] randomized object positions (seed={seed}, anchored={anchored_msg})")
    else:
        print("[server] fixed layout (SUPERMARKET_RANDOMIZE=0)")

    cfg.mjcf_file_path = write_runtime_xml(pos_overrides)
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
    cfg.referee_targets = select_targets(layout)
    cfg.referee_config_path = str(REFEREE_CONFIG_JSON)
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
            self.game_info_puber = self.create_publisher(String, "/referee/gameinfo", 2)
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
        if self.referee is not None:
            self.referee.update(self.mj_data)

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
            response.success = True
            response.message = "reset_run requested"
        return response

    def _pub_referee(self):
        if self.referee is None:
            return
        self.taskinfo_puber.publish(String(data=self._taskinfo))
        self.game_info_puber.publish(String(data=self.referee.game_info))
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
    exec_node.reset()

    spin_thread = threading.Thread(target=spin_node, args=(exec_node,), daemon=True)
    spin_thread.start()

    topic_hz = env_int("SUPERMARKET_ROS_HZ", 24)
    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(topic_hz,), daemon=True)
    pubtopic_thread.start()

    if exec_node.config.lidar_s2_sim:
        publidar_thread = threading.Thread(target=exec_node.thread_publidartopic, args=(12,), daemon=True)
        publidar_thread.start()

    try:
        while rclpy.ok() and exec_node.running:
            exec_node.step(exec_node.target_control)
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
