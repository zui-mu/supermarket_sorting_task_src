#!/usr/bin/env python3
"""ROS2 client for the Supermarket Sorting Task.

Drive MMK2 to the shelf approach pose, pick the visually detected kele bottle,
and place it on the delivery table. The client keeps a 19-d target_control,
computes right-arm joints with
MMK2Kdl, and publishes to the controller command topics exposed by the server.

Subscribes:
  /slamware_ros_sdk_server_node/odom   (nav_msgs/Odometry)  base pose in world
  /joint_states                        (sensor_msgs/JointState) 17 joints
Publishes:
  /cmd_vel, /spine.../commands, /head.../commands,
  /{left,right}_arm_forward_position_controller/commands
"""
import math
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from collections import deque

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, JointState, LaserScan
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray

from discoverse.utils import step_func
from mmk2_kdl import MMK2Kdl

# ---- scene constants (world frame, +X east / +Y north). ----
TABLE_ORIGIN = np.array([-1.940, -3.410, 0.0])   # delivery_table; top surface z~0.77
YAW_NORTH = math.pi / 2.0
YAW_SOUTH = -math.pi / 2.0

YELLOW_MID_Y = 2.475          # 抓取区两条黄线(y=1.70/3.25)正中
APPROACH_BASE_X = 0.852       # shelf approach lane that leaves the target in right-arm reach
GRASP_YAW_EAST_BIAS_DEG = 11.0
GRASP_YAW = math.pi / 2.0 - math.radians(GRASP_YAW_EAST_BIAS_DEG)  # slightly east of north to square the gripper visually
# 直行到黄线中点 -> 左转西行到货架列,停在黄线处部署胳膊,再 creep 进去
ROUTE_TO_SHELF = [[1.92, YELLOW_MID_Y], [APPROACH_BASE_X, YELLOW_MID_Y]]
# 倒车退回黄线中点后,沿旧 baseline 的直角避障路线走,避免斜切时胳膊扫墙.
# 右臂保持抓取姿态不收回,所以这条路线按 base + 夹爪扫掠一起避开当前黄箱.
ROUTE_TO_TABLE = [[-0.50, YELLOW_MID_Y], [-0.50, -0.70],
                  [-0.90, -0.70], [-0.90, -2.80], [-1.88, -2.80]]
SAFE_RIGHT_LANE_X = 1.92
SAFE_STAGING_Y = 1.05
SAFE_X_MIN = float(os.getenv("SUPERMARKET_SAFE_X_MIN", "-2.05"))
SAFE_X_MAX = float(os.getenv("SUPERMARKET_SAFE_X_MAX", "2.05"))
SAFE_Y_MIN = float(os.getenv("SUPERMARKET_SAFE_Y_MIN", "-3.20"))
SAFE_Y_MAX = float(os.getenv("SUPERMARKET_SAFE_Y_MAX", "2.70"))

# ---- manipulation params. ----
GRASP_ARM = "r"
HEAD_PITCH = -0.6
SLIDE_PRE, SLIDE_GRASP, SLIDE_LIFT = 0.11, 0.11, -0.04
GRIP_OPEN, GRIP_CLOSE = 1.0, 0.08   # 可乐瓶较细,夹爪需要闭得更紧
# Gripper orientation in the FOOTPRINT frame. This is the previous visually
# closest pose; the base yaw now supplies the small eastward correction.
GRASP_ROT = np.array([
    [ 1.0, 0.0, 0.0],
    [ 0.0, 1.0, 0.0],
    [ 0.0, 0.0, 1.0],
])
# Open-space deploy pose and creep stop are offsets from the vision-estimated
# bottle center, not from any shelf slot coordinate.
DEPLOY_OFFSET = np.array([-0.011, -0.220, -0.010]) # slightly higher than before, without pushing the wrist out of IK reach
CREEP_STOP_DY = 0.035                             # close 3cm deeper than center so the fingers wrap the bottle

# perception gating: no preset target fallback. The arm is posed only after vision
# gives a stable reachable kele point.
DETECT_DWELL = 1.0                # s to let head settle + detections accumulate before locking
DETECT_TIMEOUT = float(os.getenv("SUPERMARKET_DETECT_TIMEOUT", "10.0"))
DETECT_MIN_SAMPLES = 5            # min detection frames to trust vision
TARGET_ASSOC_MAX_DIST = float(os.getenv("SUPERMARKET_TARGET_ASSOC_MAX_DIST", "0.28"))
NEIGHBOR_CLEARANCE_X = float(os.getenv("SUPERMARKET_NEIGHBOR_CLEARANCE_X", "0.13"))
NEIGHBOR_CLEARANCE_Z = float(os.getenv("SUPERMARKET_NEIGHBOR_CLEARANCE_Z", "0.18"))
REACH_FWD_MIN, REACH_FWD_MAX = 0.3, 1.5   # m ahead of base: plausible shelf depth
REACH_LATERAL_MAX = 0.14         # m sideways: only accept a bottle directly ahead
REACH_Z_MIN, REACH_Z_MAX = 0.70, 1.15     # reachable middle shelf band
VISION_SURFACE_TO_CENTER_FWD = 0.0265     # kele radius: RGB-D point is on the visible front surface
DEPLOY_CART_TOL = 0.065                   # m; allow small joint-controller residuals before creeping
DEPLOY_JOINT_TOL = 0.080                  # rad; deploy is followed by straight base creep, not fine arm motion
DEPLOY_ROT_TOL = 0.35                     # rad; wrist must be close to the upright grasp attitude
CREEP_SPEED = 0.14                                # m/s; overcome chassis static friction
CREEP_FINE_SPEED = 0.09                           # m/s for final approach
CREEP_SLOW_DISTANCE = 0.08                        # m remaining before fine approach
CREEP_CLOSE_TOL = 0.025                           # m endpoint tolerance
CREEP_TIMEOUT = 8.0                               # s; never remain in creep forever
CREEP_YAW_KP = 4.0                                # hold heading firmly so the creep goes dead straight
LIFT_AMOUNT = 0.05                                # 夹住后竖直抬起量(减小 slide),让物体离开隔板再倒车
# Placement: robot faces SOUTH at the table; arm must reach OUT over the table top
# (z~0.77) and set the object down. Offsets are world-frame (TABLE_ORIGIN z=0).
PLACE_LOWER_SLIDE = 0.17                          # 松爪前少降一点,避免夹爪/瓶身底部碰桌

# JointState names (order documented by the server).
JOINT_NAMES = [
    "slide_joint", "head_yaw_joint", "head_pitch_joint",
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4", "left_arm_joint5", "left_arm_joint6", "left_arm_eef_gripper_joint",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_eef_gripper_joint",
]
INIT_ARM_L = [0.0, -0.166, 0.032, 0.0, 1.571, 2.223]
INIT_ARM_R = [0.0, -0.166, 0.032, 0.0, -1.571, -2.223]

# top-level phases
NAV_SHELF, DEPLOY, CREEP, CLOSE, LIFT, RETREAT, NAV_TABLE, PLACE, DONE = range(9)
PHASE_NAME = {NAV_SHELF: "nav->shelf", DEPLOY: "deploy-arm", CREEP: "creep-in", CLOSE: "close",
              LIFT: "lift", RETREAT: "retreat", NAV_TABLE: "nav->table", PLACE: "place", DONE: "done"}
RETREAT_SPEED = 0.20          # reverse speed, m/s

RIGHT_ARM_OBJECT_X_OFFSET = 0.072
OBSTACLE_STOP_DISTANCE = 0.42
OBSTACLE_SLOW_DISTANCE = 0.90
OBSTACLE_TURN_SPEED = 0.55
SCAN_STALE_TIMEOUT = 0.5
STUCK_CHECK_INTERVAL = float(os.getenv("SUPERMARKET_STUCK_CHECK_INTERVAL", "2.5"))
STUCK_MIN_PROGRESS = float(os.getenv("SUPERMARKET_STUCK_MIN_PROGRESS", "0.08"))
STUCK_RECOVERY_TIME = float(os.getenv("SUPERMARKET_STUCK_RECOVERY_TIME", "1.4"))


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PickPlaceClient(Node):
    def __init__(self):
        super().__init__("supermarket_sorting_client")
        self.kdl = MMK2Kdl()   # unified FK + analytical-grade IK (matches MMK2FIK on this mjcf)

        # target_control: [base_lin, base_ang, slide, head_yaw, head_pitch,
        #                  l_arm(6), l_grip, r_arm(6), r_grip]
        self.tc = np.zeros(19)
        self.tc[5:11] = INIT_ARM_L
        self.tc[11] = GRIP_OPEN
        self.tc[12:18] = INIT_ARM_R
        self.tc[18] = GRIP_OPEN

        # smoothed command actually published for slide/head/arms/grippers (idx 2..18):
        # step_func slews `action` toward `tc` so a new IK target never snaps (fixes 瞬移).
        # base velocity (idx 0,1) is NOT smoothed here -- it has its own accel ramp.
        self.action = self.tc.copy()
        self.joint_move_ratio = np.ones(19)
        self.tc_prev = self.tc.copy()
        self.joint_slew = float(os.getenv("SUPERMARKET_JOINT_SLEW", "1.5"))

        # latest feedback
        self.base_xy = None
        self.base_yaw = 0.0
        self.jpos = None          # dict name->pos
        self.jvel = None

        # nav/phase state
        self.phase = NAV_SHELF
        self.nav_idx = 0
        self.nav_mode = "turn"
        self.sub_idx = 0
        self.sub_entered = False
        self.deploy_set = False
        self.place_sub = 0
        self.state_t0 = self.now()
        self.arm_target_set = False

        # ---- perception: target is locked only from /kele/detections. ----
        self.OBJECT_WORLD = None
        self.DEPLOY_WORLD = None
        self.CREEP_STOP_Y = None
        self.det_buf = deque(maxlen=30)   # recent vision detections of the kele directly ahead (world xyz)
        self.target_locked = False
        self.last_wait_log = 0.0
        self.last_deploy_wait_log = 0.0
        self.creep_started_at = None
        self.execution_failed = False
        self.failure_reason = ""
        self.last_nav_progress_xy = None
        self.last_nav_progress_time = self.now()
        self.recovery_until = 0.0

        # Decision-aware approach route; configure_pick_task() replaces this.
        self.route_to_shelf = [list(point) for point in ROUTE_TO_SHELF]
        self.grasp_yaw = GRASP_YAW
        self.active_task = None
        self.expected_object_world = None
        self.runtime_layout_items = self._load_runtime_layout()

        # Lidar safety state for navigation phases.
        self.enable_obstacle_avoidance = os.getenv("SUPERMARKET_ENABLE_AVOIDANCE", "1") != "0"
        self.scan_ranges = None
        self.scan_angle_min = 0.0
        self.scan_angle_increment = 0.0
        self.scan_stamp = 0.0
        self.depth_sectors = None
        self.depth_stamp = 0.0
        self.last_avoidance_log = 0.0
        self.last_nav_progress_xy = None
        self.last_nav_progress_time = self.now()
        self.recovery_until = 0.0
        self.recovery_turn_sign = 1.0

        # gains
        self.pos_tol, self.turn_tol = 0.06, 0.03
        self.max_lin = float(os.getenv("SUPERMARKET_MAX_LINEAR", "0.60"))
        self.max_ang = float(os.getenv("SUPERMARKET_MAX_ANGULAR", "1.35"))
        # velocity ramping (acceleration limits) so /cmd_vel never jumps -> smooth motion
        self.rate_hz = 50.0
        self.dt = 1.0 / self.rate_hz
        self.max_lin_acc = float(os.getenv("SUPERMARKET_LINEAR_ACCEL", "0.9"))
        self.max_ang_acc = float(os.getenv("SUPERMARKET_ANGULAR_ACCEL", "5.0"))
        self.des_lin = self.des_ang = 0.0               # desired (from controller)
        self.cur_lin = self.cur_ang = 0.0               # ramped (actually published)

        # io
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.spine_pub = self.create_publisher(Float64MultiArray, "/spine_forward_position_controller/commands", 5)
        self.head_pub = self.create_publisher(Float64MultiArray, "/head_forward_position_controller/commands", 5)
        self.larm_pub = self.create_publisher(Float64MultiArray, "/left_arm_forward_position_controller/commands", 5)
        self.rarm_pub = self.create_publisher(Float64MultiArray, "/right_arm_forward_position_controller/commands", 5)
        self.reset_cli = self.create_client(Trigger, "/supermarket_sorting/reset_run")
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom", self.odom_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.js_cb, 10)
        self.create_subscription(Detection3DArray, "/kele/detections", self.det_cb, 10)
        self.create_subscription(LaserScan, "/slamware_ros_sdk_server_node/scan", self.scan_cb, 5)
        self.create_subscription(Image, "/head_camera/aligned_depth_to_color/image_raw", self.depth_safety_cb, 5)

        self.request_new_run()
        self.timer = self.create_timer(self.dt, self.tick)
        self.last_log = 0.0
        self.get_logger().info("pick-place client up; waiting for odom + joint_states...")

    # ---- ros helpers ----
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _load_runtime_layout(self):
        layout_path = os.getenv(
            "SUPERMARKET_RUNTIME_LAYOUT_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime_layout.json"),
        )
        if not os.path.exists(layout_path):
            return []
        try:
            with open(layout_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f"[layout] failed to load runtime layout: {exc}")
            return []

    def request_new_run(self):
        if not self.reset_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("reset service not available; continuing without server-side run reset")
            return
        future = self.reset_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if not future.done():
            self.get_logger().warn("reset service timed out; continuing after request")
            return
        result = future.result()
        if result is None:
            self.get_logger().warn("reset service failed; continuing without confirmed reset")
        elif result.success:
            self.get_logger().info(f"server run reset: {result.message}")
        else:
            self.get_logger().warn(f"server refused run reset: {result.message}")

    def reset_for_next_pick(self):
        """Reset local execution state without resetting the simulator run."""
        self.set_twist(0.0, 0.0)
        self.cur_lin = self.cur_ang = 0.0
        self.tc[0], self.tc[1] = 0.0, 0.0
        self.tc[2] = SLIDE_PRE
        self.tc[3], self.tc[4] = 0.0, 0.0
        self.tc[5:11] = INIT_ARM_L
        self.tc[11] = GRIP_OPEN
        self.tc[12:18] = INIT_ARM_R
        self.tc[18] = GRIP_OPEN
        self.phase = NAV_SHELF
        self.reset_nav()
        self.sub_idx = 0
        self.sub_entered = False
        self.deploy_set = False
        self.place_sub = 0
        self.state_t0 = self.now()
        self.arm_target_set = False
        self.OBJECT_WORLD = None
        self.DEPLOY_WORLD = None
        self.CREEP_STOP_Y = None
        self.det_buf.clear()
        self.target_locked = False
        self.creep_started_at = None
        self.execution_failed = False
        self.failure_reason = ""

    def fail_current_execution(self, reason):
        self.execution_failed = True
        self.failure_reason = reason
        self.set_twist(0.0, 0.0)
        self.phase = DONE
        self.get_logger().warn(f"[execution] failed: {reason}")

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_xy = np.array([p.x, p.y])
        self.base_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]

    def js_cb(self, msg):
        self.jpos = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}
        self.jvel = {n: msg.velocity[i] for i, n in enumerate(msg.name) if i < len(msg.velocity)}

    def scan_cb(self, msg):
        self.scan_ranges = np.asarray(msg.ranges, dtype=float)
        self.scan_angle_min = float(msg.angle_min)
        self.scan_angle_increment = float(msg.angle_increment)
        self.scan_stamp = self.now()

    def depth_safety_cb(self, msg):
        """Extract robust left/front/right obstacle ranges from mono16 depth."""
        if msg.height <= 0 or msg.width <= 0 or len(msg.data) < msg.height * msg.width * 2:
            return
        depth = np.frombuffer(msg.data, dtype=np.uint16, count=msg.height * msg.width)
        depth = depth.reshape((msg.height, msg.width)).astype(np.float32) * 1e-3
        y0, y1 = int(msg.height * 0.35), int(msg.height * 0.85)
        x0, x1 = int(msg.width * 0.15), int(msg.width * 0.85)
        roi = depth[y0:y1, x0:x1]
        thirds = np.array_split(roi, 3, axis=1)

        def robust_near(sector):
            valid = sector[(sector > 0.15) & np.isfinite(sector)]
            return float(np.percentile(valid, 10.0)) if valid.size else float("inf")

        left, front, right = (robust_near(sector) for sector in thirds)
        self.depth_sectors = (left, front, right)
        self.depth_stamp = self.now()

    def configure_pick_task(self, task):
        """Apply a TaskManager PickTask to navigation and target selection."""
        object_x = float(task.world_position[0])
        nav_x = object_x - RIGHT_ARM_OBJECT_X_OFFSET
        nav_y = float(task.navigation_target.y)
        self.active_task = task
        self.expected_object_world = np.array(task.world_position, dtype=float)
        self.runtime_layout_items = self._load_runtime_layout()
        # Avoid cutting through the middle boxes: stay in the right aisle first,
        # then enter the shelf approach lane and only slide laterally near it.
        self.route_to_shelf = [
            [SAFE_RIGHT_LANE_X, SAFE_STAGING_Y],
            [SAFE_RIGHT_LANE_X, nav_y],
            [nav_x, nav_y],
        ]
        self.grasp_yaw = float(task.navigation_target.yaw)
        self.get_logger().info(
            f"[execution] task applied: {task.task_id} route={self.route_to_shelf} "
            f"expected_object_x={object_x:.3f}")

    def det_cb(self, msg):
        """Accumulate the kele directly ahead of the parked base (world frame).

        Among all kele detections we keep the reachable one most directly ahead
        of the robot. This is the only source of the grasp target.
        """
        if self.target_locked or self.base_xy is None:
            return
        # Do not accumulate detections while driving to the shelf. With multiple
        # kele bottles visible, stale nav-time detections can otherwise lock a
        # bottle outside the current approach lane before the arm deploys.
        if self.phase != DEPLOY or self.deploy_set:
            return
        best, best_score = None, float("inf")
        for det in msg.detections:
            if not det.results:
                continue
            pos = det.results[0].pose.pose.position
            pw = np.array([pos.x, pos.y, pos.z])
            fp = self.world_to_footprint(pw)   # fp[0]=forward (ahead), fp[1]=lateral (left+)
            fwd, lat = fp[0], abs(fp[1])
            if fwd < REACH_FWD_MIN or fwd > REACH_FWD_MAX:
                continue                       # wrong depth: floor, far shelf, etc.
            if pw[2] < REACH_Z_MIN or pw[2] > REACH_Z_MAX:
                continue
            assoc_dist = 0.0
            if self.expected_object_world is not None:
                assoc_dist = float(np.linalg.norm(pw[[0, 2]] - self.expected_object_world[[0, 2]]))
                if assoc_dist > TARGET_ASSOC_MAX_DIST:
                    continue
            score = assoc_dist * 3.0 + lat
            if score < best_score:
                best_score, best = score, pw
        if best is not None:
            self.det_buf.append(best)

    def _vision_to_object_center(self, p_world):
        """Convert a visible RGB-D surface point into an estimated bottle center."""
        fp = self.world_to_footprint(p_world)
        fp[0] += VISION_SURFACE_TO_CENTER_FWD
        return self.footprint_to_world(fp)

    def _neighbor_clearance_ok(self, object_world):
        if not self.runtime_layout_items or self.active_task is None:
            return True
        body = self.active_task.metadata.get("body") if hasattr(self.active_task, "metadata") else None
        level = getattr(self.active_task, "level", None)
        for item in self.runtime_layout_items:
            if item.get("body") == body or item.get("level") != level:
                continue
            pos = np.array(item.get("world_position", [999.0, 999.0, 999.0]), dtype=float)
            same_depth = abs(float(pos[1]) - float(object_world[1])) < 0.12
            close_x = abs(float(pos[0]) - float(object_world[0])) < NEIGHBOR_CLEARANCE_X
            close_z = abs(float(pos[2]) - float(object_world[2])) < NEIGHBOR_CLEARANCE_Z
            if same_depth and close_x and close_z:
                self.get_logger().warn(
                    f"[grasp_safety] neighbor too close: target={np.round(object_world,3)} "
                    f"neighbor={item.get('body')} pos={np.round(pos,3)}")
                return False
        return True

    def _lock_target(self):
        """Lock a reachable kele target from recent vision detections."""
        if len(self.det_buf) < DETECT_MIN_SAMPLES:
            if self.now() - self.last_wait_log > 1.0:
                self.get_logger().info(
                    f"[perception] waiting for kele detections "
                    f"({len(self.det_buf)}/{DETECT_MIN_SAMPLES})")
                self.last_wait_log = self.now()
            return False

        arr = np.array(list(self.det_buf))
        candidate = np.median(arr, axis=0)
        fp = self.world_to_footprint(candidate)
        if (
            fp[0] < REACH_FWD_MIN or fp[0] > REACH_FWD_MAX
            or abs(fp[1]) > REACH_LATERAL_MAX
            or candidate[2] < REACH_Z_MIN or candidate[2] > REACH_Z_MAX
        ):
            self.get_logger().warn(
                f"[perception] discard unreachable kele candidate: "
                f"world={np.round(candidate,3)} fp={np.round(fp,3)}")
            self.det_buf.clear()
            return False

        object_world = self._vision_to_object_center(candidate)
        if not self._neighbor_clearance_ok(object_world):
            self.det_buf.clear()
            return False

        self.OBJECT_WORLD = object_world
        self.DEPLOY_WORLD  = self.OBJECT_WORLD + DEPLOY_OFFSET
        self.CREEP_STOP_Y  = self.OBJECT_WORLD[1] + CREEP_STOP_DY
        self.target_locked = True
        self.get_logger().info(
            f"[perception] kele locked from vision: "
            f"raw={np.round(candidate,3)} OBJECT={np.round(self.OBJECT_WORLD,3)}  "
            f"CREEP_STOP_Y={self.CREEP_STOP_Y:.4f}  samples={len(self.det_buf)}")
        return True

    @property
    def slide_meas(self):
        return self.jpos.get("slide_joint", self.tc[2])

    @property
    def rarm_meas(self):
        return np.array([self.jpos.get(f"right_arm_joint{i+1}", self.tc[12 + i]) for i in range(6)])

    # ---- frames ----
    def world_to_footprint(self, p_world):
        d = np.array(p_world, dtype=float) - np.array([self.base_xy[0], self.base_xy[1], 0.0])
        c, s = math.cos(-self.base_yaw), math.sin(-self.base_yaw)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])

    def footprint_to_world(self, fp):
        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        return np.array([self.base_xy[0] + c * fp[0] - s * fp[1],
                         self.base_xy[1] + s * fp[0] + c * fp[1], fp[2]])

    def arm_to(self, world_pos, rot=GRASP_ROT):
        """Set right-arm joints so the gripper reaches a world position with the grasp
        orientation, via MMK2Kdl IK (footprint frame). IK failures leave the arm held."""
        fp = self.world_to_footprint(world_pos)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = fp
        ref = np.zeros(7)
        ref[0] = float(self.tc[2])
        ref[1:] = self.rarm_meas
        sols = self.kdl.inverse_kinematics(T_left=None, T_right=T, ref_pos=ref, target_height=float(self.tc[2]))
        if sols:
            self.tc[12:18] = np.asarray(sols[0])[1:7]
            self.arm_target_set = True
            return True
        else:
            self.get_logger().warn(f"IK unreachable: world={np.round(world_pos, 3)} fp={np.round(fp, 3)} (arm holds)")
            return False

    def ee_footprint_pose(self):
        """Actual gripper endpoint pose in footprint via measured joints."""
        _, T = self.kdl.forward_kinematics(np.concatenate([[float(self.slide_meas)], self.rarm_meas]), index="right")
        return T

    def ee_world(self):
        """Actual gripper endpoint in world via MMK2Kdl forward kinematics (measured joints)."""
        T = self.ee_footprint_pose()
        return self.footprint_to_world(T[:3, 3])

    # ---- smoothing ----
    def smooth_step(self):
        """Slew `action[2:19]` toward `tc[2:19]` so a freshly-set joint target ramps in
        instead of snapping (the cause of grasp 瞬移). When the target changes,
        normalize per-joint speed by the largest delta so all joints
        arrive together; then step_func each toward its target every tick."""
        if not np.allclose(self.tc[2:19], self.tc_prev[2:19]):
            dif = np.abs(self.action[2:19] - self.tc[2:19])
            self.joint_move_ratio[2:19] = dif / (np.max(dif) + 1e-6)
            self.joint_move_ratio[2] *= 0.3   # 升降放慢到 1/3: 放置时物体轻放下, 不砸桌面引发晃动
            self.tc_prev[:] = self.tc
        step = self.joint_slew * self.dt
        for i in range(2, 19):
            self.action[i] = step_func(self.action[i], self.tc[i], self.joint_move_ratio[i] * step)

    # ---- publishing ----
    def publish(self):
        tw = Twist()
        tw.linear.x = float(self.tc[0])
        tw.angular.z = float(self.tc[1])
        self.cmd_vel_pub.publish(tw)
        self.spine_pub.publish(Float64MultiArray(data=[float(self.action[2])]))
        self.head_pub.publish(Float64MultiArray(data=[float(self.action[3]), float(self.action[4])]))
        self.larm_pub.publish(Float64MultiArray(data=[float(x) for x in self.action[5:11]] + [float(self.action[11])]))
        self.rarm_pub.publish(Float64MultiArray(data=[float(x) for x in self.action[12:18]] + [float(self.action[18])]))

    # ---- navigation ----
    def set_twist(self, lin, ang):
        self.des_lin = float(np.clip(lin, -self.max_lin, self.max_lin))
        self.des_ang = float(np.clip(ang, -self.max_ang, self.max_ang))

    def ramp_twist(self):
        """Acceleration-limit the published velocity so /cmd_vel changes smoothly."""
        dl = np.clip(self.des_lin - self.cur_lin, -self.max_lin_acc * self.dt, self.max_lin_acc * self.dt)
        da = np.clip(self.des_ang - self.cur_ang, -self.max_ang_acc * self.dt, self.max_ang_acc * self.dt)
        self.cur_lin += dl
        self.cur_ang += da
        self.tc[0], self.tc[1] = self.cur_lin, self.cur_ang

    def apply_obstacle_safety(self):
        """Slow or stop forward navigation using the simulated 2-D lidar."""
        if (
            not self.enable_obstacle_avoidance
            or self.phase not in (NAV_SHELF, NAV_TABLE)
            or self.des_lin <= 0.0
        ):
            return

        if self.scan_ranges is not None and self.now() - self.scan_stamp <= SCAN_STALE_TIMEOUT:
            angles = self.scan_angle_min + np.arange(len(self.scan_ranges)) * self.scan_angle_increment
            valid = np.isfinite(self.scan_ranges) & (self.scan_ranges > 0.05)

            def sector_min(lo, hi):
                mask = valid & (angles >= lo) & (angles <= hi)
                return float(np.min(self.scan_ranges[mask])) if np.any(mask) else float("inf")

            front = sector_min(-0.38, 0.38)
            left = sector_min(0.38, 1.15)
            right = sector_min(-1.15, -0.38)
        elif self.depth_sectors is not None and self.now() - self.depth_stamp <= SCAN_STALE_TIMEOUT:
            left, front, right = self.depth_sectors
        else:
            return
        if front < OBSTACLE_STOP_DISTANCE:
            turn_sign = 1.0 if left >= right else -1.0
            self.des_lin = 0.0
            self.des_ang = turn_sign * OBSTACLE_TURN_SPEED
            if self.now() - self.last_avoidance_log > 0.8:
                self.get_logger().warn(
                    f"[avoidance] hard stop: front={front:.2f} left={left:.2f} right={right:.2f}")
                self.last_avoidance_log = self.now()
        elif front < OBSTACLE_SLOW_DISTANCE:
            scale = (front - OBSTACLE_STOP_DISTANCE) / (OBSTACLE_SLOW_DISTANCE - OBSTACLE_STOP_DISTANCE)
            self.des_lin *= float(np.clip(scale, 0.15, 1.0))

    def clamp_nav_target(self, target):
        return np.array([
            float(np.clip(target[0], SAFE_X_MIN, SAFE_X_MAX)),
            float(np.clip(target[1], SAFE_Y_MIN, SAFE_Y_MAX)),
        ])

    def run_recovery_motion(self):
        if self.now() >= self.recovery_until:
            self.recovery_until = 0.0
            self.last_nav_progress_xy = np.array(self.base_xy, dtype=float)
            self.last_nav_progress_time = self.now()
            self.nav_mode = "turn"
            self.set_twist(0.0, 0.0)
            return False
        self.set_twist(-0.16, self.recovery_turn_sign * 0.75)
        return True

    def maybe_start_stuck_recovery(self, target):
        if self.nav_mode != "drive" or self.base_xy is None:
            return False
        now = self.now()
        if self.last_nav_progress_xy is None:
            self.last_nav_progress_xy = np.array(self.base_xy, dtype=float)
            self.last_nav_progress_time = now
            return False
        if now - self.last_nav_progress_time < STUCK_CHECK_INTERVAL:
            return False
        moved = float(np.linalg.norm(np.array(self.base_xy, dtype=float) - self.last_nav_progress_xy))
        if moved >= STUCK_MIN_PROGRESS:
            self.last_nav_progress_xy = np.array(self.base_xy, dtype=float)
            self.last_nav_progress_time = now
            return False
        lateral = float(target[0] - self.base_xy[0])
        self.recovery_turn_sign = -1.0 if lateral >= 0.0 else 1.0
        self.recovery_until = now + STUCK_RECOVERY_TIME
        self.get_logger().warn(
            f"[nav_recovery] stuck near base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}); "
            f"backing out for {STUCK_RECOVERY_TIME:.1f}s")
        return True

    def follow_route(self, route, final_yaw):
        if self.recovery_until > self.now() and self.run_recovery_motion():
            return False
        if self.nav_idx < len(route):
            target = self.clamp_nav_target(np.array(route[self.nav_idx], dtype=float))
            delta = target - self.base_xy
            dist = float(np.linalg.norm(delta))
            yaw_err = wrap_to_pi(math.atan2(delta[1], delta[0]) - self.base_yaw)
            if self.nav_mode == "turn":
                self.set_twist(0.0, 2.2 * yaw_err)
                if abs(yaw_err) < self.turn_tol:
                    self.nav_mode = "drive"
            else:
                if self.maybe_start_stuck_recovery(target):
                    self.run_recovery_motion()
                    return False
                if dist < self.pos_tol:
                    self.nav_idx += 1
                    self.nav_mode = "turn"
                    self.set_twist(0.0, 0.0)
                    self.last_nav_progress_xy = None
                    self.last_nav_progress_time = self.now()
                else:
                    # Steering: deadband when nearly aligned, and FREEZE near the
                    # waypoint (bearing blows up there) -> long straights stay dead
                    # straight with no angular twitch.
                    if abs(yaw_err) < 0.05 or dist < 0.25:
                        ang = 0.0
                    else:
                        ang = 2.2 * yaw_err
                    align = max(0.0, math.cos(yaw_err))
                    self.set_twist(1.0 * dist * align, ang)
            return False
        yaw_err = wrap_to_pi(final_yaw - self.base_yaw)
        self.set_twist(0.0, 1.8 * yaw_err)
        if abs(yaw_err) < self.turn_tol:
            self.set_twist(0.0, 0.0)
            return True
        return False

    def reset_nav(self):
        self.nav_idx = 0
        self.nav_mode = "turn"
        self.last_nav_progress_xy = None
        self.last_nav_progress_time = self.now()
        self.recovery_until = 0.0

    # ---- manipulation step gating (joint-space convergence + dwell) ----
    def action_done(self, dwell=0.4):
        if self.now() - self.state_t0 < dwell:
            return False
        slide_ok = abs(self.slide_meas - self.tc[2]) < 0.02
        arm_ok = (not self.arm_target_set) or np.max(np.abs(self.rarm_meas - self.tc[12:18])) < 0.05
        return slide_ok and arm_ok

    def deploy_done(self, dwell=0.4):
        if self.now() - self.state_t0 < dwell:
            return False
        slide_err = abs(self.slide_meas - self.tc[2])
        joint_err = np.max(np.abs(self.rarm_meas - self.tc[12:18])) if self.arm_target_set else float("inf")
        cart_err = float("inf")
        rot_err = float("inf")
        if self.DEPLOY_WORLD is not None:
            T = self.ee_footprint_pose()
            cart_err = float(np.linalg.norm(self.footprint_to_world(T[:3, 3]) - self.DEPLOY_WORLD))
            rot_delta = T[:3, :3].T @ GRASP_ROT
            cos_angle = np.clip((np.trace(rot_delta) - 1.0) * 0.5, -1.0, 1.0)
            rot_err = float(math.acos(cos_angle))

        ready = slide_err < 0.025 and (
            joint_err < DEPLOY_JOINT_TOL
            or (cart_err < DEPLOY_CART_TOL and rot_err < DEPLOY_ROT_TOL)
        )
        if not ready and self.now() - self.last_deploy_wait_log > 1.0:
            self.get_logger().info(
                f"[deploy] waiting: slide_err={slide_err:.3f} "
                f"joint_err={joint_err:.3f} cart_err={cart_err:.3f} rot_err={rot_err:.3f}")
            self.last_deploy_wait_log = self.now()
        return ready

    def enter_sub(self):
        self.sub_entered = True
        self.state_t0 = self.now()
        self.arm_target_set = False

    def run_sub(self, setter, n_states):
        if not self.sub_entered:
            setter(self.sub_idx)
            self.enter_sub()
        if self.action_done():
            self.sub_idx += 1
            self.sub_entered = False
        return self.sub_idx >= n_states

    # ---- main 30 Hz tick ----
    def tick(self):
        if self.base_xy is None or self.jpos is None:
            return

        if self.phase == NAV_SHELF:
            if self.follow_route(self.route_to_shelf, self.grasp_yaw):
                self.phase, self.deploy_set = DEPLOY, False
                self.det_buf.clear()
                self.target_locked = False
                self.OBJECT_WORLD = None
                self.DEPLOY_WORLD = None
                self.CREEP_STOP_Y = None
                self.state_t0 = self.now()   # start the detection dwell window fresh
        elif self.phase == DEPLOY:
            # 在黄线附近把胳膊摆成抓取姿态,之后靠底盘直线 creep 完成进给
            self.set_twist(0.0, 0.0)
            if not self.deploy_set:
                # Aim head/slide so the shelf is in view, accumulate
                # /kele/detections, then pose the arm from the vision target.
                self.tc[4] = HEAD_PITCH
                self.tc[2] = SLIDE_GRASP
                self.tc[18] = GRIP_OPEN
                if not self.target_locked and self.now() - self.state_t0 < DETECT_DWELL:
                    pass
                elif self._lock_target():
                    if self.arm_to(self.DEPLOY_WORLD):
                        self.deploy_set = True
                        self.state_t0 = self.now()
                    else:
                        self.target_locked = False
                        self.det_buf.clear()
                elif self.now() - self.state_t0 > DETECT_TIMEOUT:
                    self.fail_current_execution("vision target timeout during deploy")
            if self.deploy_set and self.deploy_done():
                self.phase = CREEP
                self.creep_started_at = self.now()
        elif self.phase == CREEP:
            # 保持胳膊不动,车直着往前开,把整个夹爪平移送到物体处
            ee = self.ee_world()
            remaining = self.CREEP_STOP_Y - ee[1]
            timed_out = self.creep_started_at is not None and self.now() - self.creep_started_at > CREEP_TIMEOUT
            if remaining > CREEP_CLOSE_TOL and not timed_out:
                speed = CREEP_FINE_SPEED if remaining < CREEP_SLOW_DISTANCE else CREEP_SPEED
                self.set_twist(speed, CREEP_YAW_KP * wrap_to_pi(self.grasp_yaw - self.base_yaw))
            else:
                self.set_twist(0.0, 0.0)
                if timed_out:
                    self.get_logger().warn(
                        f"[creep] timeout recovery: remaining={remaining:.3f} m; closing gripper")
                else:
                    self.get_logger().info(f"[creep] target reached: remaining={remaining:.3f} m")
                self.phase = CLOSE
                self.state_t0 = self.now()
        elif self.phase == CLOSE:
            self.set_twist(0.0, 0.0)
            self.tc[18] = GRIP_CLOSE
            if self.now() - self.state_t0 > 0.8:
                self.phase = LIFT
        elif self.phase == LIFT:
            # 竖直抬起(减小 slide,胸部上移),让物体离开隔板,胳膊关节保持不动
            self.set_twist(0.0, 0.0)
            self.tc[2] = SLIDE_GRASP - LIFT_AMOUNT
            if abs(self.slide_meas - self.tc[2]) < 0.02:
                self.phase = RETREAT
        elif self.phase == RETREAT:
            # 倒车(保持抓取朝向)退回黄线中点,object 还夹在手里
            yaw_err = wrap_to_pi(self.grasp_yaw - self.base_yaw)
            if self.base_xy[1] > YELLOW_MID_Y + self.pos_tol:
                self.set_twist(-RETREAT_SPEED, 1.0 * yaw_err)
            else:
                self.set_twist(0.0, 0.0)
                self.state_t0 = self.now()
                self.get_logger().info("retreat done; keeping grasp arm extended for table route")
                self.phase = NAV_TABLE
                self.reset_nav()
        elif self.phase == NAV_TABLE:
            if self.follow_route(ROUTE_TO_TABLE, YAW_SOUTH):
                self.phase, self.place_sub = PLACE, 0
                self.state_t0 = self.now()
        elif self.phase == PLACE:
            self.set_twist(0.0, 0.0)
            if self.place_sub == 0:
                # 先把升降平台整体降下来,物体随之竖直下降到桌面附近(手臂关节不动)
                self.tc[2] = PLACE_LOWER_SLIDE
                if abs(self.slide_meas - PLACE_LOWER_SLIDE) < 0.02:
                    self.place_sub = 1
                    self.state_t0 = self.now()
            else:
                # 到位后松爪,物体落桌
                self.tc[18] = GRIP_OPEN
                if self.now() - self.state_t0 > 1.0:
                    self.phase = DONE
        else:
            self.set_twist(0.0, 0.0)

        self.apply_obstacle_safety()
        self.ramp_twist()
        self.smooth_step()
        self.publish()

        if self.now() - self.last_log > 1.0:
            ee = self.ee_world()
            obj = self.OBJECT_WORLD
            obj_str = (f"({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f})"
                       if obj is not None else "unlocked")
            self.get_logger().info(
                f"phase={PHASE_NAME[self.phase]} sub={self.sub_idx} "
                f"base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) yaw={self.base_yaw:.2f} slide={self.slide_meas:.3f} "
                f"gripper=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                f"obj={obj_str} locked={self.target_locked}")
            self.last_log = self.now()


def main():
    rclpy.init()
    node = PickPlaceClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
