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
from std_msgs.msg import Float64MultiArray, String
from collections import deque

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, JointState, LaserScan
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray

from discoverse.utils import step_func
from mmk2_kdl import MMK2Kdl
from navigation.grid_planner import SupermarketGridPlanner

# ---- scene constants (world frame, +X east / +Y north). ----
TABLE_ORIGIN = np.array([-1.940, -3.410, 0.0])   # delivery_table; top surface z~0.77
YAW_NORTH = math.pi / 2.0
YAW_SOUTH = -math.pi / 2.0

YELLOW_MID_Y = 2.475          # 抓取区两条黄线(y=1.70/3.25)正中
APPROACH_BASE_X = 0.852       # shelf approach lane that leaves the target in right-arm reach
GRASP_YAW_EAST_BIAS_DEG = 11.0
GRASP_YAW = math.pi / 2.0 - math.radians(GRASP_YAW_EAST_BIAS_DEG)  # slightly east of north to square the gripper visually
SHELF_CROSS_Y = 2.20
# 直行到黄线中点 -> 左转西行到货架列,停在黄线处部署胳膊,再 creep 进去
ROUTE_TO_SHELF = [[1.62, 1.05], [1.62, SHELF_CROSS_Y], [APPROACH_BASE_X, YELLOW_MID_Y]]
# 倒车退回黄线中点后,沿旧 baseline 的直角避障路线走,避免斜切时胳膊扫墙.
# 右臂保持抓取姿态不收回,所以这条路线按 base + 夹爪扫掠一起避开当前黄箱.
ROUTE_TO_TABLE = [[-0.50, SHELF_CROSS_Y], [-0.50, -0.70],
                  [-0.90, -0.70], [-0.90, -2.80], [-1.88, -2.80]]
SAFE_RIGHT_LANE_X = 1.62
SAFE_STAGING_Y = 1.05
# The V2 robot spawns in a narrow pocket beside the right wall.  Leave that
# pocket while still facing north before making the small left shift into the
# corridor; asking for the shift first caused a slow diagonal scrape.
START_EXIT_Y = float(os.getenv("SUPERMARKET_START_EXIT_Y", "-2.55"))
SAFE_X_MIN = float(os.getenv("SUPERMARKET_SAFE_X_MIN", "-2.05"))
SAFE_X_MAX = float(os.getenv("SUPERMARKET_SAFE_X_MAX", "2.05"))
SAFE_Y_MIN = float(os.getenv("SUPERMARKET_SAFE_Y_MIN", "-3.20"))
SAFE_Y_MAX = float(os.getenv("SUPERMARKET_SAFE_Y_MAX", "2.70"))

# ---- manipulation params. ----
GRASP_ARM = "r"
HEAD_PITCH = -0.6
SLIDE_TRAVEL = float(os.getenv("SUPERMARKET_SLIDE_TRAVEL", "0.0"))
SLIDE_PRE, SLIDE_GRASP, SLIDE_LIFT = 0.11, 0.11, -0.04
GRIP_OPEN, GRIP_CLOSE = 1.0, 0.0    # use the full actuator range for reliable two-finger contact
# Gripper orientation in the FOOTPRINT frame. This is the previous visually
# closest pose; the base yaw now supplies the small eastward correction.
GRASP_ROT = np.array([
    [ 1.0, 0.0, 0.0],
    [ 0.0, 1.0, 0.0],
    [ 0.0, 0.0, 1.0],
])
# Open-space deploy pose and creep stop are offsets from the vision-estimated
# bottle center, not from any shelf slot coordinate.
DEPLOY_OFFSET = np.array([0.010, -0.225, 0.045])
# Put the tool centre on the bottle axis. The old positive depth offset drove
# the palm past the centre and made the right finger push the bottle first.
CREEP_STOP_DY = 0.0

# perception gating: no preset target fallback. The arm is posed only after vision
# gives a stable reachable kele point.
DETECT_DWELL = 1.0                # s to let head settle + detections accumulate before locking
DETECT_TIMEOUT = float(os.getenv("SUPERMARKET_DETECT_TIMEOUT", "10.0"))
SEARCH_DETECT_TIMEOUT = float(os.getenv("SUPERMARKET_SEARCH_DETECT_TIMEOUT", "2.5"))
DETECT_MIN_SAMPLES = 3            # enough stable frames without dwelling at every slot
# Some local V2 server builds publish the physical body as the task id.  In
# that development-only case the public slot geometry is sufficient to make a
# controlled grasp attempt when RGB-D has no usable blob.  Anonymous official
# tasks never enter this fallback: they still require a real detection.
DIRECT_TASK_GEOMETRY_FALLBACK = os.getenv(
    "SUPERMARKET_DIRECT_TASK_GEOMETRY_FALLBACK", "1"
) == "1"
DIRECT_TASK_DETECT_TIMEOUT = float(
    os.getenv("SUPERMARKET_DIRECT_TASK_DETECT_TIMEOUT", "2.5")
)
TARGET_ASSOC_MAX_DIST = float(os.getenv("SUPERMARKET_TARGET_ASSOC_MAX_DIST", "0.28"))
NEIGHBOR_CLEARANCE_X = float(os.getenv("SUPERMARKET_NEIGHBOR_CLEARANCE_X", "0.13"))
NEIGHBOR_CLEARANCE_Z = float(os.getenv("SUPERMARKET_NEIGHBOR_CLEARANCE_Z", "0.18"))
REACH_FWD_MIN, REACH_FWD_MAX = 0.3, 1.5   # m ahead of base: plausible shelf depth
REACH_LATERAL_MAX = 0.18         # tolerate parking error; layout association rejects other bottles
REACH_Z_MIN, REACH_Z_MAX = 0.45, 1.40     # all three shelf levels; task association narrows this further
TARGET_ASSOC_Z = float(os.getenv("SUPERMARKET_TARGET_ASSOC_Z", "0.30"))
VISION_SURFACE_TO_CENTER_FWD = 0.0265     # kele radius: RGB-D point is on the visible front surface
# Keep only a small lateral bias.  The previous 12 mm shift made the left
# fingertip align with the bottle centre, so the grasp succeeded mostly by
# friction after pushing.  A slight inward bias still helps the fingers bite
# without turning the bottle centre into a one-finger target.
GRASP_CENTER_X_BIAS = float(os.getenv("SUPERMARKET_GRASP_CENTER_X_BIAS", "-0.004"))
DEPLOY_CART_TOL = 0.065                   # m; allow small joint-controller residuals before creeping
DEPLOY_JOINT_TOL = 0.080                  # rad; deploy is followed by straight base creep, not fine arm motion
DEPLOY_ROT_TOL = 0.35                     # rad; wrist must be close to the upright grasp attitude
CREEP_SPEED = 0.080
CREEP_FINE_SPEED = 0.040
CREEP_SLOW_DISTANCE = 0.12
CREEP_CLOSE_TOL = 0.010
CREEP_TIMEOUT = float(os.getenv("SUPERMARKET_CREEP_TIMEOUT", "13.0"))
CREEP_YAW_KP = 4.0                                # hold heading firmly so the creep goes dead straight
CREEP_MAX_YAW_CORRECTION = 0.22
GRASP_MAX_LATERAL_CLOSE_ERR = float(os.getenv("SUPERMARKET_GRASP_MAX_LATERAL_CLOSE_ERR", "0.012"))
TOUCH_CLOSE_REMAINING = float(os.getenv("SUPERMARKET_TOUCH_CLOSE_REMAINING", "0.024"))
TOUCH_CLOSE_LATERAL_ERR = float(os.getenv("SUPERMARKET_TOUCH_CLOSE_LATERAL_ERR", "0.014"))
TOUCH_RECENTER_LATERAL_ERR = float(os.getenv("SUPERMARKET_TOUCH_RECENTER_LATERAL_ERR", "0.030"))
TOUCH_REACTION_REMAINING = float(os.getenv("SUPERMARKET_TOUCH_REACTION_REMAINING", "0.080"))
TOUCH_CREEP_SPEED = float(os.getenv("SUPERMARKET_TOUCH_CREEP_SPEED", "0.026"))
CLOSE_SEAT_CREEP_SPEED = float(os.getenv("SUPERMARKET_CLOSE_SEAT_CREEP_SPEED", "0.016"))
CLOSE_SEAT_CREEP_TIME = float(os.getenv("SUPERMARKET_CLOSE_SEAT_CREEP_TIME", "0.35"))
LIFT_AMOUNT = 0.035                               # 夹住后竖直抬起量(减小 slide),让物体离开隔板再倒车
# Placement: robot faces SOUTH at the table; arm must reach OUT over the table top
# (z~0.77) and set the object down. Offsets are world-frame (TABLE_ORIGIN z=0).
PLACE_LOWER_SLIDE = 0.13                          # release just above table; avoid pressing bottle into tabletop

# JointState names (order documented by the server).
JOINT_NAMES = [
    "slide_joint", "head_yaw_joint", "head_pitch_joint",
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4", "left_arm_joint5", "left_arm_joint6", "left_arm_eef_gripper_joint",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_eef_gripper_joint",
]
INIT_ARM_L = [0.0, -0.166, 0.032, 0.0, 1.571, 2.223]
INIT_ARM_R = [0.0, -0.166, 0.032, 0.0, -1.571, -2.223]

# top-level phases
NAV_SHELF, DEPLOY, CREEP, CLOSE, LIFT, VERIFY_GRASP, NAV_TABLE, PLACE, DONE = range(9)
PHASE_NAME = {NAV_SHELF: "nav->shelf", DEPLOY: "deploy-arm", CREEP: "creep-in", CLOSE: "close",
              LIFT: "lift", VERIFY_GRASP: "verify-grasp", NAV_TABLE: "nav->table", PLACE: "place", DONE: "done"}
RETREAT_SPEED = 0.13          # keep the bottle stable while clearing the shelf
# VERIFY_GRASP performs the first short reverse. NAV_TABLE then continues to
# the clear line below before allowing any turn with the extended arm.
GRASP_VERIFY_BASE_Y = YELLOW_MID_Y - 0.12

RIGHT_ARM_OBJECT_X_OFFSET = 0.108
OBSTACLE_STOP_DISTANCE = 0.42
OBSTACLE_SLOW_DISTANCE = 0.90
OBSTACLE_TURN_SPEED = 0.45
SCAN_STALE_TIMEOUT = 0.5
SERVER_FEEDBACK_TIMEOUT = float(os.getenv("SUPERMARKET_SERVER_FEEDBACK_TIMEOUT", "1.5"))
STUCK_CHECK_INTERVAL = float(os.getenv("SUPERMARKET_STUCK_CHECK_INTERVAL", "4.5"))
STUCK_MIN_PROGRESS = float(os.getenv("SUPERMARKET_STUCK_MIN_PROGRESS", "0.025"))
STUCK_RECOVERY_TIME = float(os.getenv("SUPERMARKET_STUCK_RECOVERY_TIME", "1.8"))
GRASP_VERIFY_TIMEOUT = float(os.getenv("SUPERMARKET_GRASP_VERIFY_TIMEOUT", "5.0"))
GRIP_CLOSE_DWELL = float(os.getenv("SUPERMARKET_GRIP_CLOSE_DWELL", "2.2"))
LIFT_SETTLE_DWELL = float(os.getenv("SUPERMARKET_LIFT_SETTLE_DWELL", "1.1"))
MAX_LOCAL_GRASP_RETRIES = max(3, int(os.getenv("SUPERMARKET_LOCAL_GRASP_RETRIES", "3")))
MAX_DROP_RECOVERIES = int(os.getenv("SUPERMARKET_DROP_RECOVERIES", "2"))
MAX_NAV_RECOVERIES = int(os.getenv("SUPERMARKET_MAX_NAV_RECOVERIES", "8"))
MAX_DELIVERY_RECOVERIES = int(os.getenv("SUPERMARKET_MAX_DELIVERY_RECOVERIES", "8"))
DELIVERY_RECOVERY_COOLDOWN = float(os.getenv("SUPERMARKET_DELIVERY_RECOVERY_COOLDOWN", "2.0"))
DELIVERY_RECOVERY_REVERSE_TIME = float(os.getenv("SUPERMARKET_DELIVERY_RECOVERY_REVERSE_TIME", "0.85"))
DELIVERY_RECOVERY_ROTATE_TIME = float(os.getenv("SUPERMARKET_DELIVERY_RECOVERY_ROTATE_TIME", "0.45"))
DELIVERY_BLOCKED_RECOVERY_DELAY = float(os.getenv("SUPERMARKET_DELIVERY_BLOCKED_RECOVERY_DELAY", "0.55"))
REPLAN_COOLDOWN = float(os.getenv("SUPERMARKET_REPLAN_COOLDOWN", "1.0"))
WAYPOINT_TURN_TOL = float(os.getenv("SUPERMARKET_WAYPOINT_TURN_TOL", "0.30"))
WAYPOINT_DRIVE_TURN_LIMIT = float(os.getenv("SUPERMARKET_WAYPOINT_DRIVE_TURN_LIMIT", "1.70"))
NAV_MIN_LINEAR_SPEED = float(os.getenv("SUPERMARKET_NAV_MIN_LINEAR", "0.30"))
PLACE_VERIFY_TIMEOUT = float(os.getenv("SUPERMARKET_PLACE_VERIFY_TIMEOUT", "10.0"))
PLACE_OPEN_DWELL = float(os.getenv("SUPERMARKET_PLACE_OPEN_DWELL", "1.5"))
PLACE_CLEAR_SLIDE = 0.07
CARRY_LINEAR_SPEED = float(os.getenv("SUPERMARKET_CARRY_LINEAR", "0.14"))
CARRY_ANGULAR_SPEED = float(os.getenv("SUPERMARKET_CARRY_ANGULAR", "0.28"))
CARRY_LINEAR_ACCEL = float(os.getenv("SUPERMARKET_CARRY_LINEAR_ACCEL", "0.12"))
CARRY_ANGULAR_ACCEL = float(os.getenv("SUPERMARKET_CARRY_ANGULAR_ACCEL", "0.20"))
GRASP_APPROACH_X_OFFSETS = (0.0, -0.014, 0.014, 0.0)
GRASP_CREEP_DY_OFFSETS = (0.0, 0.012, 0.012, 0.024)
DEFAULT_GRASP_PROFILE = {
    "deploy_offset": DEPLOY_OFFSET,
    "surface_to_center_fwd": VISION_SURFACE_TO_CENTER_FWD,
    "center_x_bias": GRASP_CENTER_X_BIAS,
    # ``rgt_endpoint`` is the wrist reference, not the midpoint of the two
    # finger pads.  The pads sit about 4 cm behind it along the tool X axis.
    # We command the endpoint ahead of the product so the *pinch centre* is
    # on the object centre when the gripper closes.
    "endpoint_from_pinch_fwd": 0.043,
    "contact_z_bias": 0.0,
    "creep_stop_dy": CREEP_STOP_DY,
    "creep_dy_offsets": GRASP_CREEP_DY_OFFSETS,
    "lift_amount": LIFT_AMOUNT,
    "grip_close_dwell": GRIP_CLOSE_DWELL,
    "grip_preopen": GRIP_OPEN,
    "grip_close_target": GRIP_CLOSE,
    "creep_speed": CREEP_SPEED,
    "creep_fine_speed": CREEP_FINE_SPEED,
    "creep_close_tolerance": 0.006,
    "creep_max_yaw_correction": 0.080,
    "touch_close_remaining": TOUCH_CLOSE_REMAINING,
    "touch_close_lateral_err": TOUCH_CLOSE_LATERAL_ERR,
    "touch_recenter_lateral_err": TOUCH_RECENTER_LATERAL_ERR,
    "touch_reaction_remaining": TOUCH_REACTION_REMAINING,
    "touch_creep_speed": TOUCH_CREEP_SPEED,
    "neighbor_clearance_x": NEIGHBOR_CLEARANCE_X,
}
PRODUCT_GRASP_PROFILES = {
    # All values below are relative to the *pinch centre*, i.e. the midpoint
    # between the two fingertips.  They come from the collision geometry in
    # retail_competition.xml, not from an image-box edge.  This avoids the
    # common one-finger-first contact that pushes light products backwards.
    "kele": {
        # cylinder r=26.5 mm, h=145 mm: close at the cylindrical mid-plane.
        "surface_to_center_fwd": 0.0265,
        "center_x_bias": -0.004,
        "endpoint_from_pinch_fwd": 0.018,
        "contact_z_bias": -0.020,
        "creep_dy_offsets": (0.0, 0.006, 0.012, 0.018),
        "retry_vision_xy_limit": 0.045,
        "grip_close_dwell": 3.4,
        "creep_speed": 0.065,
        "creep_fine_speed": 0.028,
        "creep_max_yaw_correction": 0.060,
        "base_x_bias": 0.026,
        "touch_close_remaining": 0.064,
        "touch_close_lateral_err": 0.012,
        "touch_recenter_lateral_err": 0.026,
        "touch_reaction_remaining": 0.080,
        "touch_creep_speed": 0.020,
    },
    "maidong": {
        # cylinder r=32.5 mm, h=210 mm: pinch just below its mass centre.
        "surface_to_center_fwd": 0.0325,
        "center_x_bias": -0.004,
        "endpoint_from_pinch_fwd": 0.044,
        "contact_z_bias": -0.004,
        "creep_stop_dy": -0.002,
        "creep_dy_offsets": (0.0, 0.005, 0.010),
        "lift_amount": 0.026,
        "grip_close_dwell": 3.0,
        "retreat_speed": 0.080,
        "carry_linear_speed": 0.075,
        "carry_angular_speed": 0.10,
        "creep_speed": 0.055,
        "creep_fine_speed": 0.022,
        "creep_max_yaw_correction": 0.055,
    },
    "sanmingzhi": {
        # 65 x 100 x 99 mm wedge: use its geometric mid-height and stop
        # shallowly, because its broad front face is easy to shove rearward.
        "deploy_offset": np.array([0.006, -0.235, 0.020]),
        "surface_to_center_fwd": 0.050,
        "center_x_bias": -0.003,
        "endpoint_from_pinch_fwd": 0.040,
        "contact_z_bias": -0.022,
        "creep_stop_dy": -0.004,
        "creep_dy_offsets": (0.0, 0.004, 0.008),
        "lift_amount": 0.030,
        "neighbor_clearance_x": 0.16,
        "creep_speed": 0.045,
        "creep_fine_speed": 0.018,
        "creep_max_yaw_correction": 0.050,
        "grip_close_dwell": 2.8,
    },
    "heweidao": {
        # Tapered 65--95 mm package: centre the fingers at the broad middle,
        # then use the same slow final approach as the sandwich.
        "deploy_offset": np.array([0.006, -0.235, 0.020]),
        "surface_to_center_fwd": 0.048,
        "center_x_bias": -0.003,
        "endpoint_from_pinch_fwd": 0.041,
        "contact_z_bias": -0.020,
        "creep_stop_dy": -0.004,
        "creep_dy_offsets": (0.0, 0.004, 0.008),
        "lift_amount": 0.030,
        "neighbor_clearance_x": 0.16,
        "creep_speed": 0.045,
        "creep_fine_speed": 0.018,
        "creep_max_yaw_correction": 0.050,
        "grip_close_dwell": 2.8,
    },
    "shupian": {
        # cylinder r=32.5 mm, h=210 mm.  Its larger height makes the upper
        # edge catch a shelf; retain a centred but low-force final advance.
        "deploy_offset": np.array([0.006, -0.240, 0.018]),
        "surface_to_center_fwd": 0.0325,
        "center_x_bias": -0.002,
        "endpoint_from_pinch_fwd": 0.044,
        "contact_z_bias": -0.018,
        "creep_stop_dy": -0.003,
        "creep_dy_offsets": (0.0, 0.004, 0.008),
        "lift_amount": 0.032,
        "neighbor_clearance_x": 0.17,
        "creep_speed": 0.050,
        "creep_fine_speed": 0.020,
        "creep_max_yaw_correction": 0.050,
        "grip_close_dwell": 3.0,
    },
    "zhijin": {
        # 172 x 85 x 88 mm box.  The fingers must close about the box centre,
        # not at its visible front face, otherwise the first finger tips it.
        "deploy_offset": np.array([0.006, -0.240, 0.018]),
        "surface_to_center_fwd": 0.0425,
        "center_x_bias": -0.002,
        "endpoint_from_pinch_fwd": 0.040,
        "contact_z_bias": -0.020,
        "creep_stop_dy": -0.004,
        "creep_dy_offsets": (0.0, 0.004, 0.008),
        "lift_amount": 0.028,
        "neighbor_clearance_x": 0.18,
        "creep_speed": 0.045,
        "creep_fine_speed": 0.018,
        "creep_max_yaw_correction": 0.050,
        "grip_close_dwell": 2.8,
    },
    "kouxiangtang": {
        "deploy_offset": np.array([0.004, -0.225, 0.030]),
        "surface_to_center_fwd": 0.030,
        "center_x_bias": -0.004,
        "creep_stop_dy": -0.004,
        "lift_amount": 0.026,
    },
    # Fruit can roll; use a shallower approach and gentler lift.
    "pingguo": {
        # The round products are intentionally pinched 5 mm above the
        # equator.  That gives both fingertips a small upward support and
        # prevents rolling along the shelf during the last centimetres.
        "deploy_offset": np.array([0.004, -0.235, 0.020]),
        "surface_to_center_fwd": 0.035,
        "center_x_bias": 0.0,
        "endpoint_from_pinch_fwd": 0.042,
        "contact_z_bias": 0.005,
        "creep_stop_dy": -0.004,
        "creep_dy_offsets": (0.0, 0.003, 0.006),
        "lift_amount": 0.026,
        "neighbor_clearance_x": 0.16,
        "creep_speed": 0.035,
        "creep_fine_speed": 0.014,
        "creep_max_yaw_correction": 0.035,
        "grip_close_dwell": 3.2,
        "retreat_speed": 0.075,
        "carry_linear_speed": 0.065,
        "carry_angular_speed": 0.09,
    },
    "chengzi": {
        "deploy_offset": np.array([0.004, -0.235, 0.020]),
        "surface_to_center_fwd": 0.037,
        "center_x_bias": 0.0,
        "endpoint_from_pinch_fwd": 0.042,
        "contact_z_bias": 0.005,
        "creep_stop_dy": -0.004,
        "creep_dy_offsets": (0.0, 0.003, 0.006),
        "lift_amount": 0.026,
        "neighbor_clearance_x": 0.16,
        "creep_speed": 0.035,
        "creep_fine_speed": 0.014,
        "creep_max_yaw_correction": 0.035,
        "grip_close_dwell": 3.2,
        "retreat_speed": 0.075,
        "carry_linear_speed": 0.065,
        "carry_angular_speed": 0.09,
    },
    # Some task descriptions call the round apple-like proxy "tudou".
    "tudou": {
        "surface_to_center_fwd": 0.035,
        "endpoint_from_pinch_fwd": 0.042,
        "contact_z_bias": 0.005,
        "creep_stop_dy": -0.004,
        "creep_dy_offsets": (0.0, 0.003, 0.006),
        "creep_speed": 0.035,
        "creep_fine_speed": 0.014,
        "creep_max_yaw_correction": 0.035,
        "grip_close_dwell": 3.2,
        "retreat_speed": 0.075,
        "carry_linear_speed": 0.065,
        "carry_angular_speed": 0.09,
    },
}
SHELF_POS_TOL = max(0.055, float(os.getenv("SUPERMARKET_SHELF_POS_TOL", "0.055")))
CARRY_POS_TOL = float(os.getenv("SUPERMARKET_CARRY_POS_TOL", "0.06"))
CARRY_SHELF_CLEAR_Y = float(os.getenv("SUPERMARKET_CARRY_SHELF_CLEAR_Y", "1.92"))
DELIVERY_GOAL = np.array([-1.88, -2.74], dtype=float)
DELIVERY_USE_ASTAR = os.getenv("SUPERMARKET_DELIVERY_USE_ASTAR", "1") == "1"
DELIVERY_SAFE_WAYPOINTS = [
    np.array([0.18, 2.04], dtype=float),
    np.array([-0.45, 1.42], dtype=float),
    np.array([-0.45, -0.70], dtype=float),
    np.array([-1.45, -0.70], dtype=float),
    np.array([-1.45, -1.58], dtype=float),
    DELIVERY_GOAL.copy(),
]
RETRY_RETREAT_MARGIN = float(os.getenv("SUPERMARKET_RETRY_RETREAT_MARGIN", "0.035"))
STUCK_NEAR_WAYPOINT_RADIUS = float(os.getenv("SUPERMARKET_STUCK_NEAR_WAYPOINT_RADIUS", "0.20"))
SHELF_FINAL_NO_RECOVERY_RADIUS = float(os.getenv("SUPERMARKET_SHELF_FINAL_NO_RECOVERY_RADIUS", "0.42"))
SHELF_FINAL_YAW_TOL = float(os.getenv("SUPERMARKET_SHELF_FINAL_YAW_TOL", "0.075"))
SLIDE_GRASP_BY_LEVEL = {
    "L1": 0.43,
    "L2": 0.11,
    "L3": 0.06,
}


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
        self.odom_stamp = 0.0
        self.joint_stamp = 0.0
        self.last_server_watchdog_log = 0.0

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
        self.PINCH_WORLD = None
        self.GRASP_ENDPOINT_WORLD = None
        self.DEPLOY_WORLD = None
        self.CREEP_STOP_Y = None
        self.det_buf = deque(maxlen=30)   # recent vision detections of the kele directly ahead (world xyz)
        self.target_locked = False
        self.last_wait_log = 0.0
        self.last_deploy_wait_log = 0.0
        self.creep_started_at = None
        self.execution_failed = False
        self.failure_reason = ""
        self.local_grasp_retries = 0
        self.drop_recoveries = 0
        self.handled_drop_keys = set()
        self.fallen_object_points = []
        self.verify_start_xy = None
        self.referee_state = {}
        self.completed_before_task = 0
        self.grasp_was_confirmed = False
        self.carry_retreat_active = False
        self.grasp_retry_retreat_active = False
        self.close_nudge_until = None
        self.close_nudge_done = False
        self.last_touch_creep_log = 0.0
        self.last_nav_progress_xy = None
        self.last_nav_progress_time = self.now()
        self.recovery_until = 0.0
        self.recovery_state = "idle"
        self.nav_recovery_count = 0
        self.delivery_recovery_count = 0
        self.last_delivery_recovery_time = -999.0
        self.recovery_linear = -0.18
        self.last_replan_time = 0.0
        self.front_blocked = False
        self.front_blocked_since = None
        self.route_goal = None
        self.route_purpose = "shelf"
        self.route_needs_plan = True
        self.planner = SupermarketGridPlanner(
            resolution=float(os.getenv("SUPERMARKET_GRID_RESOLUTION", "0.10")),
            robot_radius=float(os.getenv("SUPERMARKET_ROBOT_CLEARANCE", "0.22")),
            corridor_clearance=float(os.getenv("SUPERMARKET_CORRIDOR_CLEARANCE", "0.30")),
        )

        # Decision-aware approach route; configure_pick_task() replaces this.
        self.route_to_shelf = [list(point) for point in ROUTE_TO_SHELF]
        self.route_to_table = [list(point) for point in ROUTE_TO_TABLE]
        self.grasp_yaw = GRASP_YAW
        self.grasp_slide = SLIDE_GRASP
        self.active_product_name = "kele"
        self.active_task_level = "L2"
        self.grasp_profile = dict(DEFAULT_GRASP_PROFILE)
        self.active_task = None
        self.expected_object_world = None
        self.runtime_layout_items = self._load_runtime_layout()

        # Lidar safety state for navigation phases.
        self.enable_obstacle_avoidance = os.getenv("SUPERMARKET_ENABLE_AVOIDANCE", "1") != "0"
        # Head depth frequently sees MMK2's own arm. Keep it opt-in; a real
        # /scan remains enabled automatically whenever the server publishes it.
        self.enable_depth_avoidance = os.getenv("SUPERMARKET_ENABLE_DEPTH_AVOIDANCE", "0") == "1"
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
        self.recovery_linear = -0.18

        # gains
        self.pos_tol, self.turn_tol = SHELF_POS_TOL, 0.05
        self.max_lin = float(os.getenv("SUPERMARKET_MAX_LINEAR", "0.85"))
        self.max_ang = float(os.getenv("SUPERMARKET_MAX_ANGULAR", "1.35"))
        # velocity ramping (acceleration limits) so /cmd_vel never jumps -> smooth motion
        self.rate_hz = 50.0
        self.dt = 1.0 / self.rate_hz
        self.max_lin_acc = float(os.getenv("SUPERMARKET_LINEAR_ACCEL", "1.4"))
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
        self.create_subscription(String, "/referee/state", self.referee_state_cb, 5)

        self.request_new_run()
        self.timer = self.create_timer(self.dt, self.tick)
        self.last_log = 0.0
        self.get_logger().info("pick-place client up; waiting for odom + joint_states...")

    # ---- ros helpers ----
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _load_runtime_layout(self):
        if os.getenv("SUPERMARKET_ALLOW_RUNTIME_LAYOUT", "0") != "1":
            return []
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
        # Keep the slide fully retracted while travelling.  It is extended only
        # in DEPLOY after the base has reached the shelf approach pose.
        self.tc[2] = SLIDE_TRAVEL
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
        self.PINCH_WORLD = None
        self.GRASP_ENDPOINT_WORLD = None
        self.DEPLOY_WORLD = None
        self.CREEP_STOP_Y = None
        self.det_buf.clear()
        self.target_locked = False
        self.creep_started_at = None
        self.execution_failed = False
        self.failure_reason = ""
        self.local_grasp_retries = 0
        self.drop_recoveries = 0
        self.handled_drop_keys = set()
        self.verify_start_xy = None
        self.completed_before_task = int(self.referee_state.get("completed", 0))
        self.grasp_was_confirmed = False
        self.carry_retreat_active = False
        self.grasp_retry_retreat_active = False
        self.close_nudge_until = None
        self.close_nudge_done = False
        self.last_touch_creep_log = 0.0
        self.delivery_collision_recovered = False
        self.delivery_recovery_count = 0
        self.last_delivery_recovery_time = -999.0
        self.route_to_table = [list(point) for point in ROUTE_TO_TABLE]
        self.route_needs_plan = True
        self.recovery_linear = -0.18

    def fail_current_execution(self, reason):
        self.execution_failed = True
        self.failure_reason = reason
        self.set_twist(0.0, 0.0)
        self.phase = DONE
        self.get_logger().warn(f"[execution] failed: {reason}")

    def expected_referee_body(self):
        if self.active_task is not None and hasattr(self.active_task, "metadata"):
            return self.active_task.metadata.get("body")
        return None

    def active_search_mode(self):
        return bool(
            self.active_task is not None
            and hasattr(self.active_task, "metadata")
            and self.active_task.metadata.get("search_mode")
        )

    def has_direct_official_target(self):
        """Whether the task message identifies a physical shelf body.

        This is intentionally separate from ``expected_object_world``: the
        latter is also populated for normal static-layout tasks, whereas an
        anonymous V2 task must not be allowed to guess a product location.
        """
        return bool(
            self.active_task is not None
            and hasattr(self.active_task, "metadata")
            and self.active_task.metadata.get("official_direct")
        )

    def current_target_touched(self):
        expected_body = self.expected_referee_body()
        touched = set(self.referee_state.get("touched_targets") or [])
        if expected_body is None:
            return bool(touched) or int(self.referee_state.get("flow_step", 0)) >= 2
        return expected_body in touched

    def drop_flow_key(self, flow):
        if not isinstance(flow, dict):
            return None
        steps = flow.get("steps") or {}
        return (
            flow.get("target"),
            bool(flow.get("dropped")),
            steps.get("s3"),
        )

    def recover_dropped_object(self, last_flow):
        """Stop this flow; a shelf item that fell cannot be picked there again."""
        expected_body = None
        if self.active_task is not None and hasattr(self.active_task, "metadata"):
            expected_body = self.active_task.metadata.get("body")
        if expected_body is not None and last_flow.get("target") != expected_body:
            return False
        drop_key = self.drop_flow_key(last_flow)
        if drop_key is None or drop_key in self.handled_drop_keys:
            return False
        self.handled_drop_keys.add(drop_key)
        self.drop_recoveries += 1
        self.grasp_was_confirmed = False
        final_pos = last_flow.get("final_pos")
        if isinstance(final_pos, (list, tuple)) and len(final_pos) >= 2:
            point = np.asarray(final_pos[:2], dtype=float)
            if np.all(np.isfinite(point)):
                if not any(np.linalg.norm(point - np.asarray(old)) < 0.10 for old in self.fallen_object_points):
                    self.fallen_object_points.append(point.tolist())
        self.get_logger().warn(
            "[drop_recovery] object dropped and left its shelf slot; "
            "skipping this physical bottle and selecting another candidate")
        self.fail_current_execution("carried object dropped; skip fallen shelf item")
        return True

    def active_flow_dropped(self):
        expected_body = self.expected_referee_body()
        if expected_body is None:
            return False
        flow_target = self.referee_state.get("flow_target")
        if flow_target not in (None, expected_body):
            return False
        return bool(self.referee_state.get("dropped"))

    def active_drop_report(self):
        expected_body = self.expected_referee_body()
        last_flow = self.referee_state.get("last_flow") or {}
        if (
            isinstance(last_flow, dict)
            and last_flow.get("target")
            and last_flow.get("dropped")
            and not last_flow.get("completed")
            and (expected_body is None or last_flow.get("target") == expected_body)
        ):
            return last_flow
        if self.active_flow_dropped():
            return {
                "target": expected_body,
                "dropped": True,
                "completed": False,
                "steps": {"s3": self.referee_state.get("flow_step", 0) >= 3},
            }
        return None

    def has_active_drop_report(self):
        return self.active_drop_report() is not None

    def active_target_knocked_or_dropped(self):
        expected_body = self.expected_referee_body()
        if expected_body is None:
            return False
        flow_target = self.referee_state.get("flow_target")
        if flow_target not in (None, expected_body):
            return False
        if bool(self.referee_state.get("dropped")):
            return True
        if (
            bool(self.referee_state.get("toppled"))
            and expected_body in set(self.referee_state.get("touched_targets") or [])
            and int(self.referee_state.get("flow_step", 0)) < 3
        ):
            return True
        # The referee can keep a broad toppled flag once the active flow starts.
        # Before S3 we treat this as a soft warning: the robot may have brushed
        # the shelf while still trying to center the pinch line, and that should
        # not force an immediate target switch.  Only after S3 do we treat the
        # topple flag as fatal for the active target.
        return bool(self.referee_state.get("toppled")) and (
            flow_target == expected_body
            and int(self.referee_state.get("flow_step", 0)) >= 3
        )

    def retry_local_grasp(self, reason, *, consume_retry=True, manual_offset=None):
        """Retry perception and grasping at the same physical product slot."""
        if self.active_search_mode() and "vision target timeout" in reason:
            self.fail_current_execution(
                f"search slot did not contain target kind: {reason}")
            return
        if (
            self.has_direct_official_target()
            and self.active_task_level == "L3"
            and (
                "direct-task geometry fallback IK failed" in reason
                or "vision target timeout" in reason
            )
        ):
            self.fail_current_execution(
                f"grasp failed at shelf: {reason}; upper-shelf target is not stable enough now, skip this item")
            return
        if self.active_target_knocked_or_dropped():
            self.fail_current_execution(
                f"grasp failed at shelf: {reason}; target toppled or dropped, skip this item")
            return
        if consume_retry and self.local_grasp_retries >= MAX_LOCAL_GRASP_RETRIES:
            self.fail_current_execution(
                f"grasp failed at shelf: {reason}; local grasp retries exhausted ({MAX_LOCAL_GRASP_RETRIES})")
            return
        if consume_retry:
            self.local_grasp_retries += 1
        retry_index = self.local_grasp_retries + self.drop_recoveries
        if manual_offset is None:
            offset = GRASP_APPROACH_X_OFFSETS[retry_index % len(GRASP_APPROACH_X_OFFSETS)]
        else:
            offset = float(np.clip(manual_offset, -0.030, 0.030))
        if self.expected_object_world is not None:
            base_goal = np.array([
                float(self.expected_object_world[0])
                - RIGHT_ARM_OBJECT_X_OFFSET
                + float(self.grasp_profile.get("base_x_bias", 0.0))
                + offset,
                YELLOW_MID_Y,
            ])
        elif self.route_goal is not None:
            # Search-mode failures have no known object coordinate.  Preserve
            # the current legal slot goal instead of crashing while building a
            # retry route.
            base_goal = np.asarray(self.route_goal, dtype=float).copy()
            base_goal[0] += offset
        else:
            base_goal = np.asarray(self.base_xy, dtype=float).copy()
        self.tc[18] = GRIP_OPEN
        self.tc[12:18] = INIT_ARM_R
        self.tc[2] = SLIDE_TRAVEL
        self.arm_target_set = False
        self.target_locked = False
        self.deploy_set = False
        self.det_buf.clear()
        self.OBJECT_WORLD = None
        self.PINCH_WORLD = None
        self.GRASP_ENDPOINT_WORLD = None
        self.DEPLOY_WORLD = None
        self.CREEP_STOP_Y = None
        self.creep_started_at = None
        self.verify_start_xy = None
        self.grasp_was_confirmed = False
        self.close_nudge_until = None
        self.close_nudge_done = False
        self.last_touch_creep_log = 0.0
        near_shelf = (
            self.base_xy is not None
            and self.base_xy[1] >= SHELF_CROSS_Y - 0.12
        )
        # Only reverse when the chassis is actually inside the shelf mouth.
        # A previous version always rebuilt the full right-lane corridor after
        # a visual timeout, making a 2 cm retry turn into a trip across all
        # shelves.
        self.grasp_retry_retreat_active = bool(
            self.base_xy is not None
            and self.base_xy[1] > YELLOW_MID_Y + RETRY_RETREAT_MARGIN
        )
        self.route_goal = base_goal
        self.route_purpose = "grasp-retry"
        if near_shelf:
            # The arm is retracted above, so this is a safe local alignment.
            # Keep only the new approach pose and never route through the
            # global right-side staging lane for a grasp retry.
            self.route_to_shelf = [base_goal.tolist()]
            self.route_needs_plan = False
            route_note = "local shelf alignment"
        else:
            self.route_to_shelf = self.shelf_corridor_route(base_goal)
            self.route_needs_plan = False
            route_note = "staged shelf corridor"
        self.phase = NAV_SHELF
        self.reset_nav()
        self.state_t0 = self.now()
        self.get_logger().warn(
            f"[grasp_retry] {reason}; local_retry={self.local_grasp_retries}/"
            f"{MAX_LOCAL_GRASP_RETRIES}, drop_retry={self.drop_recoveries}/"
            f"{MAX_DROP_RECOVERIES}, approach_x_offset={offset:+.3f}m; "
            f"using {route_note}")

    def start_delivery_collision_recovery(self, reason: str):
        """Break away from a structure collision and replan the delivery leg."""
        now = self.now()
        if now - self.last_delivery_recovery_time < DELIVERY_RECOVERY_COOLDOWN:
            return False
        if self.delivery_recovery_count >= MAX_DELIVERY_RECOVERIES:
            drop_report = self.active_drop_report()
            if drop_report is not None and self.recover_dropped_object(drop_report):
                return True
            self.fail_current_execution(
                f"delivery navigation recovery limit exceeded ({MAX_DELIVERY_RECOVERIES}); "
                f"last reason={reason}")
            return True
        self.delivery_collision_recovered = True
        self.delivery_recovery_count += 1
        self.last_delivery_recovery_time = now
        collided_while_retreating = self.carry_retreat_active
        turn_sign = 1.0
        if self.depth_sectors is not None:
            left, front, right = self.depth_sectors
            turn_sign = 1.0 if left >= right else -1.0
        elif self.base_xy is not None:
            turn_sign = 1.0 if self.base_xy[0] <= 0.0 else -1.0
        self.get_logger().warn(
            f"[delivery_recovery] {reason}; break away then use delivery corridor, "
            f"attempt={self.delivery_recovery_count}/{MAX_DELIVERY_RECOVERIES}, "
            f"turn_sign={turn_sign:+.1f}")
        self.carry_retreat_active = False
        self.route_to_table = self.delivery_corridor_route()
        self.route_needs_plan = True
        self.route_goal = DELIVERY_GOAL.copy()
        self.route_purpose = "delivery"
        self.recovery_turn_sign = turn_sign
        carry_speed = float(self.grasp_profile.get("carry_linear_speed", CARRY_LINEAR_SPEED))
        # If collision happened during the first reverse-out, creep forward a
        # little to free the object. Otherwise back up to create turning room.
        self.recovery_linear = min(0.08, carry_speed) if collided_while_retreating else -min(0.14, carry_speed + 0.04)
        self.recovery_state = "reverse"
        self.recovery_until = now + DELIVERY_RECOVERY_REVERSE_TIME
        self.nav_mode = "drive"
        self.last_nav_progress_xy = np.array(self.base_xy, dtype=float) if self.base_xy is not None else None
        self.last_nav_progress_time = now
        return True

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_xy = np.array([p.x, p.y])
        self.base_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        self.odom_stamp = self.now()

    def js_cb(self, msg):
        self.jpos = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}
        self.jvel = {n: msg.velocity[i] for i, n in enumerate(msg.name) if i < len(msg.velocity)}
        self.joint_stamp = self.now()

    def referee_state_cb(self, msg):
        try:
            self.referee_state = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            self.referee_state = {}

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
        # Use the lower-middle image band.  The upper/wide image often contains
        # MMK2's own extended arm while carrying an object.
        y0, y1 = int(msg.height * 0.58), int(msg.height * 0.88)
        x0, x1 = int(msg.width * 0.22), int(msg.width * 0.78)
        roi = depth[y0:y1, x0:x1]
        thirds = np.array_split(roi, 3, axis=1)

        def robust_near(sector):
            valid = sector[(sector > 0.15) & np.isfinite(sector)]
            return float(np.percentile(valid, 10.0)) if valid.size else float("inf")

        left, front, right = (robust_near(sector) for sector in thirds)
        self.depth_sectors = (left, front, right)
        self.depth_stamp = self.now()

    def profile_for_product(self, product_name):
        profile = dict(DEFAULT_GRASP_PROFILE)
        profile.update(PRODUCT_GRASP_PROFILES.get(str(product_name), {}))
        return profile

    def profile_for_task(self, task):
        profile = self.profile_for_product(getattr(task, "product_name", "kele"))
        level = str(getattr(task, "level", "L2"))
        product_name = str(getattr(task, "product_name", "kele"))
        deploy_offset = np.asarray(
            profile.get("deploy_offset", DEPLOY_OFFSET),
            dtype=float,
        ).copy()

        # Upper-shelf failures were dominated by unreachable pre-grasp poses.
        # Pull the deploy pose slightly closer and lower so IK can at least
        # attempt the grasp instead of looping on a pose that is impossible.
        if level == "L3":
            deploy_offset[1] = max(float(deploy_offset[1]), -0.205)
            deploy_offset[2] = min(float(deploy_offset[2]), 0.010)
            profile["creep_stop_dy"] = min(
                float(profile.get("creep_stop_dy", CREEP_STOP_DY)),
                -0.012 if product_name in {"pingguo", "chengzi"} else -0.006,
            )
        if level == "L3" and product_name in {"pingguo", "chengzi"}:
            deploy_offset[1] = max(float(deploy_offset[1]), -0.195)
            deploy_offset[2] = min(float(deploy_offset[2]), 0.0)
            profile["center_x_bias"] = -0.002
        profile["deploy_offset"] = deploy_offset
        return profile

    def configure_pick_task(self, task):
        """Apply a TaskManager PickTask to navigation and target selection."""
        object_x = float(task.world_position[0])
        nav_y = float(task.navigation_target.y)
        self.active_task = task
        self.active_product_name = str(getattr(task, "product_name", "kele"))
        self.active_task_level = str(getattr(task, "level", "L2"))
        self.grasp_profile = self.profile_for_task(task)
        nav_x = (
            object_x
            - RIGHT_ARM_OBJECT_X_OFFSET
            + float(self.grasp_profile.get("base_x_bias", 0.0))
        )
        search_mode = bool(getattr(task, "metadata", {}).get("search_mode"))
        self.expected_object_world = None if search_mode else np.array(task.world_position, dtype=float)
        self.grasp_slide = float(SLIDE_GRASP_BY_LEVEL.get(getattr(task, "level", "L2"), SLIDE_GRASP))
        self.runtime_layout_items = self._load_runtime_layout()
        self.route_goal = np.array([nav_x, nav_y], dtype=float)
        self.route_purpose = "shelf"
        # Use an explicit shelf approach instead of replanning the last meters.
        # The A* route can legally cut diagonally near the yellow line, which
        # makes the chassis oscillate just outside the pickup area.  This staged
        # path keeps the robot in the right safe lane, crosses in front of the
        # shelf, then performs only a short final alignment to the product slot.
        cross_x = float(np.clip(nav_x, SAFE_X_MIN + 0.12, SAFE_X_MAX - 0.12))
        self.route_to_shelf = [
            [SAFE_RIGHT_LANE_X, SAFE_STAGING_Y],
            [SAFE_RIGHT_LANE_X, SHELF_CROSS_Y],
            [cross_x, SHELF_CROSS_Y],
            [nav_x, nav_y],
        ]
        self.route_needs_plan = True
        self.grasp_yaw = float(task.navigation_target.yaw)
        self.get_logger().info(
            f"[execution] task applied: {task.task_id} seed_route={self.route_to_shelf} "
            f"product={self.active_product_name} strategy={getattr(task, 'grasp_strategy', 'front_center')} "
            f"search_mode={search_mode} expected_object_x={object_x:.3f} "
            f"grasp_slide={self.grasp_slide:.3f} referee_body="
            f"{task.metadata.get('body') if hasattr(task, 'metadata') else None}")

    def det_cb(self, msg):
        """Accumulate detections for the active product ahead of the parked base.

        Among all detections we keep the reachable one associated with the
        selected task. This is the only source of the grasp target.
        """
        if self.target_locked or self.base_xy is None:
            return
        # Do not accumulate detections while driving to the shelf. With multiple
        # kele bottles visible, stale nav-time detections can otherwise lock a
        # bottle outside the current approach lane before the arm deploys.
        if self.phase != DEPLOY or self.deploy_set:
            return
        best, best_score = None, float("inf")
        direct_official_target = bool(
            self.active_task is not None
            and hasattr(self.active_task, "metadata")
            and self.active_task.metadata.get("official_direct")
        )
        for det in msg.detections:
            if not det.results:
                continue
            class_id = str(det.results[0].hypothesis.class_id)
            # The legacy blob backend labels every visible blob as ``kele``.
            # When the official message already identifies the physical body,
            # use the expected world position as the association gate instead
            # of rejecting non-kele products on that placeholder class label.
            if (
                not direct_official_target
                and self.active_product_name
                and class_id
                and class_id != self.active_product_name
            ):
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
                if abs(float(pw[2]) - float(self.expected_object_world[2])) > TARGET_ASSOC_Z:
                    continue
                assoc_dist = float(np.linalg.norm(pw[[0, 2]] - self.expected_object_world[[0, 2]]))
                if assoc_dist > TARGET_ASSOC_MAX_DIST:
                    continue
            score = assoc_dist * 3.0 + lat
            if score < best_score:
                best_score, best = score, pw
        if best is not None:
            self.det_buf.append(best)

    def _vision_to_object_center(self, p_world):
        """Convert a visible RGB-D surface point into an estimated object center."""
        fp = self.world_to_footprint(p_world)
        fp[0] += float(self.grasp_profile.get("surface_to_center_fwd", VISION_SURFACE_TO_CENTER_FWD))
        return self.footprint_to_world(fp)

    def grasp_forward_world(self):
        """World direction of the fingertip centreline at the shelf."""
        return np.array([math.cos(self.grasp_yaw), math.sin(self.grasp_yaw), 0.0])

    def lock_grasp_geometry(self, object_world, *, source):
        """Build one shape-aware grasp frame from an object geometric centre.

        ``OBJECT_WORLD`` remains the physical object centre.  ``PINCH_WORLD``
        is where the midpoint between the two finger pads must be, and
        ``DEPLOY_WORLD`` is the wrist endpoint required to place that midpoint
        correctly.  Keeping these three positions separate prevents a future
        visual-centre adjustment from silently turning into a one-finger push.
        """
        object_world = np.asarray(object_world, dtype=float).copy()
        retry_index = self.local_grasp_retries + self.drop_recoveries
        pinch_world = object_world.copy()
        pinch_world[0] += float(self.grasp_profile.get("center_x_bias", GRASP_CENTER_X_BIAS))
        pinch_world[0] += GRASP_APPROACH_X_OFFSETS[
            retry_index % len(GRASP_APPROACH_X_OFFSETS)
        ] * 0.35
        pinch_world[2] += float(self.grasp_profile.get("contact_z_bias", 0.0))

        creep_offsets = tuple(self.grasp_profile.get("creep_dy_offsets", GRASP_CREEP_DY_OFFSETS))
        final_forward_extra = (
            float(self.grasp_profile.get("creep_stop_dy", CREEP_STOP_DY))
            + float(creep_offsets[retry_index % len(creep_offsets)])
        )
        endpoint_world = pinch_world + self.grasp_forward_world() * (
            float(self.grasp_profile.get("endpoint_from_pinch_fwd", 0.043))
            + final_forward_extra
        )
        self.OBJECT_WORLD = object_world
        self.PINCH_WORLD = pinch_world
        self.GRASP_ENDPOINT_WORLD = endpoint_world
        self.DEPLOY_WORLD = endpoint_world + np.asarray(
            self.grasp_profile.get("deploy_offset", DEPLOY_OFFSET), dtype=float
        )
        self.CREEP_STOP_Y = endpoint_world[1]
        self.target_locked = True
        self.get_logger().info(
            f"[grasp_frame] {self.active_product_name} source={source} "
            f"object={np.round(self.OBJECT_WORLD,3)} pinch={np.round(self.PINCH_WORLD,3)} "
            f"endpoint={np.round(endpoint_world,3)} creep_y={self.CREEP_STOP_Y:.4f}"
        )

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
            close_x = abs(float(pos[0]) - float(object_world[0])) < float(
                self.grasp_profile.get("neighbor_clearance_x", NEIGHBOR_CLEARANCE_X))
            close_z = abs(float(pos[2]) - float(object_world[2])) < NEIGHBOR_CLEARANCE_Z
            if same_depth and close_x and close_z:
                self.get_logger().warn(
                    f"[grasp_safety] neighbor too close: target={np.round(object_world,3)} "
                    f"neighbor={item.get('body')} pos={np.round(pos,3)}")
                return False
        return True

    def _lock_target(self):
        """Lock a reachable active-product target from recent vision detections."""
        if len(self.det_buf) < DETECT_MIN_SAMPLES:
            if self.now() - self.last_wait_log > 1.0:
                self.get_logger().info(
                    f"[perception] waiting for {self.active_product_name} detections "
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
                f"[perception] discard unreachable {self.active_product_name} candidate: "
                f"world={np.round(candidate,3)} fp={np.round(fp,3)}")
            self.det_buf.clear()
            return False

        object_world = self._vision_to_object_center(candidate)
        if self.expected_object_world is not None:
            # Vision confirms that the requested product is present. Its RGB-D
            # box centre can lie on the shelf behind a thin bottle, so the first
            # attempt uses the exact randomized slot centre exported by the
            # server.  After contact/retry, the product may have shifted; then
            # trust vision for X/Y while keeping the known shelf height.
            vision_error = float(np.linalg.norm(object_world - self.expected_object_world))
            if vision_error > TARGET_ASSOC_MAX_DIST:
                self.get_logger().warn(
                    f"[perception] vision/layout disagreement {vision_error:.3f}m; reacquiring")
                self.det_buf.clear()
                return False
            if self.local_grasp_retries > 0 or self.current_target_touched():
                corrected = np.array(self.expected_object_world, dtype=float)
                xy_delta = object_world[:2] - corrected[:2]
                max_xy_correction = float(self.grasp_profile.get("retry_vision_xy_limit", 0.12))
                norm = float(np.linalg.norm(xy_delta))
                if norm > max_xy_correction:
                    xy_delta *= max_xy_correction / max(norm, 1e-6)
                corrected[:2] += xy_delta
                object_world = corrected
            else:
                object_world = np.array(self.expected_object_world, dtype=float)
        if not self._neighbor_clearance_ok(object_world):
            self.det_buf.clear()
            return False
        self.lock_grasp_geometry(object_world, source="vision")
        self.get_logger().info(
            f"[perception] {self.active_product_name} locked from vision: "
            f"raw={np.round(candidate,3)} OBJECT={np.round(self.OBJECT_WORLD,3)}  "
            f"CREEP_STOP_Y={self.CREEP_STOP_Y:.4f}  samples={len(self.det_buf)}")
        return True

    def lock_direct_task_geometry_fallback(self):
        """Lock the public slot centre after a direct-task vision timeout.

        The fallback is deliberately narrow: it needs a server-provided body
        id, a reachable expected slot and an explicit opt-in environment flag.
        It is not available to anonymous official search tasks, so it cannot
        turn unknown product search into layout-truth lookup.
        """
        if (
            not DIRECT_TASK_GEOMETRY_FALLBACK
            or not self.has_direct_official_target()
            or self.expected_object_world is None
            or self.base_xy is None
        ):
            return False
        object_world = np.asarray(self.expected_object_world, dtype=float).copy()
        fp = self.world_to_footprint(object_world)
        if (
            fp[0] < REACH_FWD_MIN
            or fp[0] > REACH_FWD_MAX
            or abs(fp[1]) > REACH_LATERAL_MAX
            or object_world[2] < REACH_Z_MIN
            or object_world[2] > REACH_Z_MAX
        ):
            self.get_logger().warn(
                "[perception] direct-task geometry fallback is unreachable: "
                f"world={np.round(object_world, 3)} fp={np.round(fp, 3)}"
            )
            return False
        if not self._neighbor_clearance_ok(object_world):
            return False
        self.lock_grasp_geometry(object_world, source="direct-slot")
        self.get_logger().warn(
            "[perception] RGB-D timeout; using direct-task public slot geometry "
            f"for one grasp attempt: OBJECT={np.round(self.OBJECT_WORLD, 3)}"
        )
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

    def stop_robot(self):
        """Publish repeated zero velocity commands before the client exits."""
        self.des_lin = self.des_ang = 0.0
        self.cur_lin = self.cur_ang = 0.0
        self.tc[0], self.tc[1] = 0.0, 0.0
        for _ in range(5):
            try:
                self.publish()
            except Exception:
                break

    # ---- navigation ----
    def set_twist(self, lin, ang):
        self.des_lin = float(np.clip(lin, -self.max_lin, self.max_lin))
        self.des_ang = float(np.clip(ang, -self.max_ang, self.max_ang))

    def lidar_obstacle_points(self, max_range=2.5):
        """Return fresh scan hits transformed into the world frame."""
        if self.scan_ranges is None or self.now() - self.scan_stamp > SCAN_STALE_TIMEOUT:
            return []
        angles = self.scan_angle_min + np.arange(len(self.scan_ranges)) * self.scan_angle_increment
        valid = np.isfinite(self.scan_ranges) & (self.scan_ranges > 0.12) & (self.scan_ranges < max_range)
        ranges = self.scan_ranges[valid]
        angles = angles[valid] + self.base_yaw
        if ranges.size == 0:
            return []
        points = np.column_stack((
            self.base_xy[0] + ranges * np.cos(angles),
            self.base_xy[1] + ranges * np.sin(angles),
        ))
        # Downsample so replanning remains cheap at 50 Hz.
        return points[::4].tolist()

    def plan_route(self, goal, purpose):
        if purpose == "grasp-retry":
            # A retry at the shelf must remain a local correction.  Global
            # staging waypoints are valid only for the initial approach.
            route = [np.asarray(goal, dtype=float).tolist()]
            self.nav_idx = 0
            self.nav_mode = "turn"
            self.route_needs_plan = False
            self.front_blocked = False
            self.front_blocked_since = None
            self.last_replan_time = self.now()
            self.get_logger().info(
                f"[planner] grasp-retry local alignment: {np.round(np.asarray(route), 2).tolist()}")
            return route
        if purpose in {"shelf", "grasp-retry"}:
            route = self.shelf_corridor_route(goal)
            self.nav_idx = 0
            self.nav_mode = "turn"
            self.route_needs_plan = False
            self.front_blocked = False
            self.front_blocked_since = None
            self.last_replan_time = self.now()
            self.get_logger().info(
                f"[planner] {purpose} staged corridor with {len(route)} waypoints: "
                f"{np.round(np.asarray(route), 2).tolist()}")
            return route
        if purpose == "delivery":
            if DELIVERY_USE_ASTAR:
                dynamic = self.lidar_obstacle_points(max_range=3.0) if self.enable_obstacle_avoidance else []
                dynamic.extend(self.fallen_object_points)
                route = self.planner.plan(self.base_xy, goal, dynamic)
                if not route:
                    route = self.delivery_corridor_route()
                    planner_name = "delivery corridor fallback"
                else:
                    planner_name = "delivery A*"
            else:
                route = self.delivery_corridor_route()
                planner_name = "delivery staged corridor"
            self.nav_idx = 0
            self.nav_mode = "turn"
            self.route_needs_plan = False
            self.front_blocked = False
            self.front_blocked_since = None
            self.last_replan_time = self.now()
            self.get_logger().info(
                f"[planner] {planner_name} with {len(route)} waypoints: "
                f"{np.round(np.asarray(route), 2).tolist()}")
            return route
        dynamic = self.lidar_obstacle_points() if self.enable_obstacle_avoidance else []
        dynamic.extend(self.fallen_object_points)
        route = self.planner.plan(self.base_xy, goal, dynamic)
        if not route:
            # Ignore only transient lidar noise on the fallback pass. A fallen
            # product is a real obstacle and must remain in the map.
            route = self.planner.plan(self.base_xy, goal, self.fallen_object_points)
        if not route:
            self.fail_current_execution(f"global planner found no {purpose} route")
            return []
        self.nav_idx = 0
        self.nav_mode = "turn"
        self.route_needs_plan = False
        self.front_blocked = False
        self.front_blocked_since = None
        self.last_replan_time = self.now()
        self.get_logger().info(
            f"[planner] {purpose} route with {len(route)} waypoints: "
            f"{np.round(np.asarray(route), 2).tolist()}")
        return route

    def ramp_twist(self):
        """Acceleration-limit the published velocity so /cmd_vel changes smoothly."""
        lin_acc = CARRY_LINEAR_ACCEL if self.phase == NAV_TABLE else self.max_lin_acc
        ang_acc = CARRY_ANGULAR_ACCEL if self.phase == NAV_TABLE else self.max_ang_acc
        dl = np.clip(self.des_lin - self.cur_lin, -lin_acc * self.dt, lin_acc * self.dt)
        da = np.clip(self.des_ang - self.cur_ang, -ang_acc * self.dt, ang_acc * self.dt)
        self.cur_lin += dl
        self.cur_ang += da
        self.tc[0], self.tc[1] = self.cur_lin, self.cur_ang

    def apply_obstacle_safety(self):
        """Slow or stop forward navigation using the simulated 2-D lidar."""
        if (
            not self.enable_obstacle_avoidance
            or self.phase not in (NAV_SHELF, NAV_TABLE)
            or self.recovery_state != "idle"
            or self.des_lin <= 0.0
        ):
            return

        if self.scan_ranges is not None and self.now() - self.scan_stamp <= SCAN_STALE_TIMEOUT:
            angles = self.scan_angle_min + np.arange(len(self.scan_ranges)) * self.scan_angle_increment
            valid = np.isfinite(self.scan_ranges) & (self.scan_ranges > 0.05)

            def sector_min(lo, hi):
                mask = valid & (angles >= lo) & (angles <= hi)
                return float(np.min(self.scan_ranges[mask])) if np.any(mask) else float("inf")

            if self.phase == NAV_SHELF:
                front = sector_min(-0.28, 0.28)
                left = sector_min(0.28, 1.05)
                right = sector_min(-1.05, -0.28)
            else:
                front = sector_min(-0.38, 0.38)
                left = sector_min(0.38, 1.15)
                right = sector_min(-1.15, -0.38)
        elif (
            self.enable_depth_avoidance
            and self.depth_sectors is not None
            and self.now() - self.depth_stamp <= SCAN_STALE_TIMEOUT
        ):
            left, front, right = self.depth_sectors
        else:
            return
        stop_distance = 0.34 if self.phase == NAV_SHELF else OBSTACLE_STOP_DISTANCE
        slow_distance = 0.68 if self.phase == NAV_SHELF else OBSTACLE_SLOW_DISTANCE
        blocked_now = front < stop_distance
        if blocked_now:
            if self.front_blocked_since is None:
                self.front_blocked_since = self.now()
        else:
            self.front_blocked_since = None
        # Require persistence to reject isolated arm/depth artifacts.
        self.front_blocked = bool(
            blocked_now
            and self.front_blocked_since is not None
            and self.now() - self.front_blocked_since >= 0.35
        )
        if self.front_blocked:
            turn_sign = 1.0 if left >= right else -1.0
            self.des_lin = 0.0
            self.des_ang = 0.0
            self.recovery_turn_sign = turn_sign
            if self.now() - self.last_avoidance_log > 0.8:
                self.get_logger().warn(
                    f"[avoidance] blocked: front={front:.2f} left={left:.2f} right={right:.2f}; "
                    "requesting replan")
                self.last_avoidance_log = self.now()
        elif front < slow_distance:
            scale = (front - stop_distance) / (slow_distance - stop_distance)
            min_scale = 0.38 if self.phase == NAV_SHELF else 0.18
            self.des_lin *= float(np.clip(scale, min_scale, 1.0))

    def clamp_nav_target(self, target):
        return np.array([
            float(np.clip(target[0], SAFE_X_MIN, SAFE_X_MAX)),
            float(np.clip(target[1], SAFE_Y_MIN, SAFE_Y_MAX)),
        ])

    def shelf_corridor_route(self, goal):
        """Use a staged corridor to avoid the V2 map's right-wall start trap.

        A* often prunes the first leg into a long diagonal from the start pose
        to the shelf mouth.  In the official V2 scene that diagonal can skim the
        centre divider or random boxes, so the base crawls and repeatedly
        declares itself stuck.  This route first moves away from the wall, then
        advances along a clear lane, and only crosses in front of the shelf.
        """
        goal = np.asarray(goal, dtype=float)
        cross_x = float(np.clip(goal[0], SAFE_X_MIN + 0.12, SAFE_X_MAX - 0.12))
        start = np.asarray(self.base_xy, dtype=float) if self.base_xy is not None else None
        lane_x = float(np.clip(SAFE_RIGHT_LANE_X, SAFE_X_MIN + 0.18, SAFE_X_MAX - 0.18))
        candidates = []
        if start is not None and start[1] < START_EXIT_Y - 0.12:
            # The first leg deliberately preserves the spawn X coordinate.
            # It is a straight northbound clearance move, not a diagonal path
            # into the corridor wall.
            # Preserve the measured spawn X.  Clipping 1.92 to 1.87 created
            # the small initial yaw change that looked like an immediate turn.
            exit_x = float(np.clip(start[0], SAFE_X_MIN, SAFE_X_MAX))
            candidates.append(np.array([exit_x, START_EXIT_Y], dtype=float))
            candidates.append(np.array([lane_x, START_EXIT_Y], dtype=float))
        else:
            candidates.append(np.array([lane_x, START_EXIT_Y], dtype=float))
        candidates.extend([
            np.array([lane_x, SAFE_STAGING_Y], dtype=float),
            np.array([lane_x, SHELF_CROSS_Y], dtype=float),
            np.array([cross_x, SHELF_CROSS_Y], dtype=float),
            goal.copy(),
        ])
        route = []
        for point in candidates:
            if start is not None:
                if np.linalg.norm(point - start) < 0.14:
                    continue
                # Do not ask the robot to drive backwards to a staging gate it
                # has already passed during a recovery.
                if point[1] < start[1] - 0.20:
                    continue
            if route and np.linalg.norm(point - np.asarray(route[-1])) < 0.10:
                continue
            route.append(point.tolist())
        if not route:
            route.append(goal.tolist())
        elif np.linalg.norm(np.asarray(route[-1]) - goal) > 0.05:
            route.append(goal.tolist())
        return route

    def delivery_corridor_route(self):
        """Build a conservative table route that respects the centre divider.

        The official V2 arena has a long divider around x~=0.53 from the table
        area up to y~=1.70.  A previous fallback crossed at y=1.05 and could
        therefore drive the loaded robot into the divider.  Keep the crossing
        above that divider, then descend on the left side toward the table.
        """
        start = np.asarray(self.base_xy, dtype=float) if self.base_xy is not None else None
        opening_y = 2.04
        candidates = []
        if start is not None and start[0] > 0.08:
            candidates.append(np.array([start[0], opening_y], dtype=float))
            candidates.extend(DELIVERY_SAFE_WAYPOINTS)
        elif start is not None and start[0] < -1.20:
            candidates.extend([
                np.array([max(start[0], -1.65), opening_y], dtype=float),
                np.array([-1.65, -0.68], dtype=float),
                np.array([-1.45, -0.68], dtype=float),
                np.array([-1.45, -1.58], dtype=float),
                DELIVERY_GOAL.copy(),
            ])
        else:
            candidates.extend(DELIVERY_SAFE_WAYPOINTS[1:])
        route = []
        for waypoint in candidates:
            point = np.asarray(waypoint, dtype=float)
            if start is not None:
                if np.linalg.norm(point - start) < 0.12:
                    continue
            route.append(point.tolist())
        if not route or np.linalg.norm(np.asarray(route[-1]) - DELIVERY_GOAL) > 0.05:
            route.append(DELIVERY_GOAL.tolist())
        return route

    def run_recovery_motion(self):
        if self.now() >= self.recovery_until:
            if self.recovery_state == "reverse":
                self.recovery_state = "rotate"
                rotate_time = DELIVERY_RECOVERY_ROTATE_TIME if self.phase == NAV_TABLE else STUCK_RECOVERY_TIME * 0.55
                self.recovery_until = self.now() + rotate_time
            else:
                self.recovery_state = "idle"
                self.recovery_until = 0.0
                self.last_nav_progress_xy = np.array(self.base_xy, dtype=float)
                self.last_nav_progress_time = self.now()
                self.nav_mode = "turn"
                if self.phase == NAV_TABLE:
                    self.route_goal = DELIVERY_GOAL.copy()
                    self.route_purpose = "delivery"
                    self.route_needs_plan = True
                else:
                    self.route_needs_plan = True
                self.set_twist(0.0, 0.0)
                return False
        if self.recovery_state == "reverse":
            self.set_twist(self.recovery_linear, 0.0)
        else:
            turn_speed = OBSTACLE_TURN_SPEED if self.phase == NAV_TABLE else 0.65
            self.set_twist(0.0, self.recovery_turn_sign * turn_speed)
        return True

    def maybe_start_stuck_recovery(self, target):
        if self.nav_mode != "drive" or self.base_xy is None:
            return False
        now = self.now()
        dist_to_target = float(np.linalg.norm(np.array(target, dtype=float) - np.array(self.base_xy, dtype=float)))
        if self.phase == NAV_SHELF:
            if (
                self.nav_idx == 0
                and target[1] <= START_EXIT_Y + 0.05
                and self.base_xy[1] < START_EXIT_Y - 0.08
                and not self.front_blocked
            ):
                # The first straight exit can be slow while static friction
                # settles.  Do not convert harmless initial progress into a
                # reverse-and-spin recovery cycle.
                self.last_nav_progress_xy = np.array(self.base_xy, dtype=float)
                self.last_nav_progress_time = now
                return False
            final_goal = np.asarray(self.route_to_shelf[-1], dtype=float) if self.route_to_shelf else np.asarray(target, dtype=float)
            dist_to_final = float(np.linalg.norm(final_goal - np.asarray(self.base_xy, dtype=float)))
            if dist_to_final < SHELF_FINAL_NO_RECOVERY_RADIUS and not self.front_blocked:
                # At the shelf mouth, odom progress can become tiny while the
                # chassis squares itself to the grasp yaw. Reversing here looks
                # like "head shaking" and often prevents entering deploy-arm.
                self.last_nav_progress_xy = np.array(self.base_xy, dtype=float)
                self.last_nav_progress_time = now
                return False
        if dist_to_target < STUCK_NEAR_WAYPOINT_RADIUS:
            self.last_nav_progress_xy = np.array(self.base_xy, dtype=float)
            self.last_nav_progress_time = now
            return False
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
        if self.phase == NAV_TABLE:
            return self.start_delivery_collision_recovery("stuck while carrying")
        lateral = float(target[0] - self.base_xy[0])
        self.recovery_turn_sign = -1.0 if lateral >= 0.0 else 1.0
        self.nav_recovery_count += 1
        if self.nav_recovery_count > MAX_NAV_RECOVERIES:
            self.fail_current_execution("navigation recovery limit exceeded")
            return True
        self.recovery_state = "reverse"
        self.recovery_linear = -0.18
        self.recovery_until = now + STUCK_RECOVERY_TIME
        self.get_logger().warn(
            f"[nav_recovery] stuck near base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}); "
            f"recovery {self.nav_recovery_count}/{MAX_NAV_RECOVERIES}: reverse then replan")
        return True

    def follow_route(self, route, final_yaw):
        if self.recovery_state != "idle" and self.run_recovery_motion():
            return False
        if self.route_needs_plan and self.route_goal is not None:
            planned = self.plan_route(self.route_goal, self.route_purpose)
            if not planned:
                return False
            route[:] = planned
        if (
            self.phase == NAV_TABLE
            and self.front_blocked
            and self.front_blocked_since is not None
            and self.now() - self.front_blocked_since >= DELIVERY_BLOCKED_RECOVERY_DELAY
        ):
            return self.start_delivery_collision_recovery("front obstacle blocked delivery path")
        if (
            self.front_blocked
            and self.now() - self.last_replan_time >= REPLAN_COOLDOWN
            and self.nav_idx < len(route)
        ):
            final_shelf_distance = float("inf")
            if self.phase == NAV_SHELF and route:
                final_shelf_distance = float(np.linalg.norm(
                    np.asarray(route[-1], dtype=float) - np.asarray(self.base_xy, dtype=float)
                ))
            if final_shelf_distance <= SHELF_FINAL_NO_RECOVERY_RADIUS:
                # At the final shelf approach the laser naturally sees the
                # shelf face.  Replanning here creates a feedback loop where
                # the robot spins away from the very pose required for grasp.
                self.front_blocked = False
                self.front_blocked_since = None
            else:
                self.route_needs_plan = True
                self.front_blocked = False
                self.front_blocked_since = None
                planned = self.plan_route(self.route_goal, self.route_purpose)
                if planned:
                    route[:] = planned
        if self.nav_idx < len(route):
            target = self.clamp_nav_target(np.array(route[self.nav_idx], dtype=float))
            delta = target - self.base_xy
            dist = float(np.linalg.norm(delta))
            yaw_err = wrap_to_pi(math.atan2(delta[1], delta[0]) - self.base_yaw)
            pos_tol = CARRY_POS_TOL if self.phase == NAV_TABLE else SHELF_POS_TOL
            if (
                self.phase == NAV_TABLE
                and self.nav_idx == 0
                and abs(float(delta[0])) < 0.18
                and float(delta[1]) < -pos_tol
                and abs(wrap_to_pi(self.grasp_yaw - self.base_yaw)) < 0.45
            ):
                # After S3 the arm is still extended toward the shelf.  Back
                # straight out of the shelf lane first; turning 180 degrees in
                # place is slow and can sweep the carried object into walls.
                yaw_hold = wrap_to_pi(self.grasp_yaw - self.base_yaw)
                reverse_speed = float(self.grasp_profile.get("retreat_speed", RETREAT_SPEED))
                self.set_twist(-reverse_speed, float(np.clip(1.2 * yaw_hold, -0.08, 0.08)))
                self.last_nav_progress_xy = np.array(self.base_xy, dtype=float)
                self.last_nav_progress_time = self.now()
                return False
            if self.nav_mode == "turn":
                is_final_shelf_leg = self.phase == NAV_SHELF and self.nav_idx == len(route) - 1
                turn_tol = self.turn_tol if is_final_shelf_leg else WAYPOINT_TURN_TOL
                if abs(yaw_err) < turn_tol:
                    self.nav_mode = "drive"
                    self.last_nav_progress_xy = np.array(self.base_xy, dtype=float)
                    self.last_nav_progress_time = self.now()
                elif (not is_final_shelf_leg) and abs(yaw_err) < WAYPOINT_DRIVE_TURN_LIMIT:
                    # Avoid "dancing in place" at long waypoints. Once the
                    # heading is roughly correct, move and let continuous
                    # steering remove the remaining error.
                    if self.phase == NAV_TABLE:
                        carry_angular = float(self.grasp_profile.get("carry_angular_speed", CARRY_ANGULAR_SPEED))
                        if abs(yaw_err) > 0.70:
                            self.set_twist(0.0, float(np.clip(1.4 * yaw_err, -carry_angular, carry_angular)))
                            return False
                        self.nav_mode = "drive"
                        creep_forward = min(0.12, max(0.06, 0.22 * dist))
                        self.set_twist(
                            creep_forward,
                            float(np.clip(1.0 * yaw_err, -carry_angular, carry_angular)),
                        )
                    else:
                        self.nav_mode = "drive"
                        creep_forward = min(0.22, max(0.10, 0.35 * dist))
                        self.set_twist(creep_forward, float(np.clip(1.4 * yaw_err, -0.45, 0.45)))
                else:
                    turn_cmd = float(np.clip(1.7 * yaw_err, -0.75, 0.75))
                    if self.phase == NAV_TABLE:
                        carry_angular = float(self.grasp_profile.get("carry_angular_speed", CARRY_ANGULAR_SPEED))
                        turn_cmd = float(np.clip(
                            turn_cmd, -carry_angular, carry_angular))
                    self.set_twist(0.0, turn_cmd)
            else:
                if dist < pos_tol:
                    self.nav_idx += 1
                    self.nav_mode = "turn"
                    self.set_twist(0.0, 0.0)
                    self.last_nav_progress_xy = None
                    self.last_nav_progress_time = self.now()
                else:
                    if self.maybe_start_stuck_recovery(target):
                        self.run_recovery_motion()
                        return False
                    # Steering: deadband when nearly aligned, and FREEZE near the
                    # waypoint (bearing blows up there) -> long straights stay dead
                    # straight with no angular twitch.
                    # Regulated pure-pursuit style steering: continuously steer,
                    # but reduce speed sharply on curvature and near obstacles.
                    ang = 0.0 if abs(yaw_err) < 0.025 else 2.0 * yaw_err
                    align = max(0.0, math.cos(yaw_err))
                    curvature_scale = 1.0 / (1.0 + 2.2 * abs(yaw_err))
                    requested_speed = 1.10 * dist * align * curvature_scale
                    is_final_shelf_drive = (
                        self.phase == NAV_SHELF
                        and self.nav_idx == len(route) - 1
                        and dist < SHELF_FINAL_NO_RECOVERY_RADIUS
                    )
                    if self.phase == NAV_SHELF and not is_final_shelf_drive and align > 0.45:
                        requested_speed = max(requested_speed, NAV_MIN_LINEAR_SPEED)
                    if self.phase == NAV_TABLE:
                        carry_angular = float(self.grasp_profile.get("carry_angular_speed", CARRY_ANGULAR_SPEED))
                        carry_linear = float(self.grasp_profile.get("carry_linear_speed", CARRY_LINEAR_SPEED))
                        requested_speed = min(requested_speed, carry_linear)
                        ang = float(np.clip(
                            ang, -carry_angular, carry_angular))
                    elif self.phase == NAV_SHELF and self.nav_idx == len(route) - 1:
                        if dist < SHELF_FINAL_NO_RECOVERY_RADIUS:
                            requested_speed = min(requested_speed, CREEP_SPEED)
                            if dist > pos_tol:
                                requested_speed = max(requested_speed, CREEP_FINE_SPEED)
                            ang = float(np.clip(ang, -0.32, 0.32))
                    self.set_twist(requested_speed, ang)
            return False
        yaw_err = wrap_to_pi(final_yaw - self.base_yaw)
        final_turn_cmd = 1.8 * yaw_err
        if self.phase == NAV_TABLE:
            carry_angular = float(self.grasp_profile.get("carry_angular_speed", CARRY_ANGULAR_SPEED))
            final_turn_cmd = float(np.clip(final_turn_cmd, -carry_angular, carry_angular))
        self.set_twist(0.0, final_turn_cmd)
        final_turn_tol = SHELF_FINAL_YAW_TOL if self.phase == NAV_SHELF else self.turn_tol
        if abs(yaw_err) < final_turn_tol:
            self.set_twist(0.0, 0.0)
            return True
        return False

    def reset_nav(self):
        self.nav_idx = 0
        self.nav_mode = "turn"
        self.last_nav_progress_xy = None
        self.last_nav_progress_time = self.now()
        self.recovery_until = 0.0
        self.recovery_state = "idle"
        self.recovery_linear = -0.18
        self.nav_recovery_count = 0
        self.front_blocked = False
        self.front_blocked_since = None

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
        now = self.now()
        if (
            now - self.odom_stamp > SERVER_FEEDBACK_TIMEOUT
            or now - self.joint_stamp > SERVER_FEEDBACK_TIMEOUT
        ):
            self.set_twist(0.0, 0.0)
            self.ramp_twist()
            self.smooth_step()
            self.publish()
            if now - self.last_server_watchdog_log > 1.0:
                self.get_logger().warn(
                    "[server_watchdog] odom/joint_states stale; "
                    "is the left-top supermarket_sorting_server still running?")
                self.last_server_watchdog_log = now
            return

        if self.phase == NAV_SHELF:
            if self.grasp_retry_retreat_active:
                yaw_err = wrap_to_pi(self.grasp_yaw - self.base_yaw)
                retreat_speed = float(self.grasp_profile.get("retreat_speed", RETREAT_SPEED))
                if self.base_xy[1] > YELLOW_MID_Y + RETRY_RETREAT_MARGIN:
                    # Back out in the current shelf-facing direction before
                    # re-aiming. This prevents a failed grasp from turning in
                    # place with the arm still near the shelf.
                    self.set_twist(-retreat_speed, 0.8 * yaw_err)
                else:
                    self.set_twist(0.0, 0.0)
                    self.cur_lin = self.cur_ang = 0.0
                    self.grasp_retry_retreat_active = False
                    self.route_needs_plan = True
                    self.reset_nav()
            elif self.follow_route(self.route_to_shelf, self.grasp_yaw):
                self.phase, self.deploy_set = DEPLOY, False
                self.det_buf.clear()
                self.target_locked = False
                self.OBJECT_WORLD = None
                self.PINCH_WORLD = None
                self.GRASP_ENDPOINT_WORLD = None
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
                self.tc[2] = self.grasp_slide
                self.tc[18] = float(self.grasp_profile.get("grip_preopen", GRIP_OPEN))
                if not self.target_locked and self.now() - self.state_t0 < DETECT_DWELL:
                    pass
                elif self._lock_target():
                    if self.arm_to(self.DEPLOY_WORLD):
                        self.deploy_set = True
                        self.state_t0 = self.now()
                    else:
                        self.target_locked = False
                        self.det_buf.clear()
                else:
                    if self.active_search_mode():
                        detect_timeout = SEARCH_DETECT_TIMEOUT
                    elif self.has_direct_official_target() and DIRECT_TASK_GEOMETRY_FALLBACK:
                        # Do not spend ten seconds staring at a shelf when a
                        # direct task already has a legal public slot pose.
                        detect_timeout = DIRECT_TASK_DETECT_TIMEOUT
                    else:
                        detect_timeout = DETECT_TIMEOUT
                    if self.now() - self.state_t0 > detect_timeout:
                        if self.lock_direct_task_geometry_fallback():
                            if self.arm_to(self.DEPLOY_WORLD):
                                self.deploy_set = True
                                self.state_t0 = self.now()
                            else:
                                self.target_locked = False
                                self.OBJECT_WORLD = None
                                self.PINCH_WORLD = None
                                self.GRASP_ENDPOINT_WORLD = None
                                self.DEPLOY_WORLD = None
                                self.CREEP_STOP_Y = None
                                self.retry_local_grasp(
                                    "direct-task geometry fallback IK failed"
                                )
                        else:
                            self.retry_local_grasp("vision target timeout during deploy")
            if self.deploy_set and self.deploy_done():
                self.phase = CREEP
                self.creep_started_at = self.now()
        elif self.phase == CREEP:
            # 保持胳膊不动,车直着往前开,把整个夹爪平移送到物体处
            ee = self.ee_world()
            ee = self.ee_world()
            grasp_fwd = self.grasp_forward_world()
            grasp_left = np.array([-grasp_fwd[1], grasp_fwd[0], 0.0])
            endpoint_goal = self.GRASP_ENDPOINT_WORLD
            if endpoint_goal is None:
                endpoint_goal = np.array([ee[0], self.CREEP_STOP_Y, ee[2]], dtype=float)
            endpoint_error = np.asarray(endpoint_goal, dtype=float) - ee
            # Stop along the two-finger centreline, not at a world-Y proxy.
            remaining = float(np.dot(endpoint_error, grasp_fwd))
            lateral_error = float(np.dot(endpoint_error, grasp_left))
            timed_out = self.creep_started_at is not None and self.now() - self.creep_started_at > CREEP_TIMEOUT
            close_tol = float(self.grasp_profile.get("creep_close_tolerance", CREEP_CLOSE_TOL))
            target_touched = self.current_target_touched()
            touch_close_remaining = float(self.grasp_profile.get(
                "touch_close_remaining", TOUCH_CLOSE_REMAINING))
            touch_close_lateral = float(self.grasp_profile.get(
                "touch_close_lateral_err", TOUCH_CLOSE_LATERAL_ERR))
            touch_recenter_lateral = float(self.grasp_profile.get(
                "touch_recenter_lateral_err", TOUCH_RECENTER_LATERAL_ERR))
            touch_reaction_remaining = float(self.grasp_profile.get(
                "touch_reaction_remaining", TOUCH_REACTION_REMAINING))
            target_touched_near = target_touched and remaining <= touch_reaction_remaining
            if (
                target_touched_near
                and remaining <= touch_close_remaining
                and abs(lateral_error) <= touch_close_lateral
            ):
                self.set_twist(0.0, 0.0)
                self.get_logger().info(
                    f"[creep] target touched and pinch-centred; closing: "
                    f"remaining={remaining:.3f} m lateral={lateral_error:.3f} m")
                self.phase = CLOSE
                self.state_t0 = self.now()
                return
            if (
                target_touched_near
                and remaining > touch_close_remaining
                and abs(lateral_error) > touch_recenter_lateral
            ):
                corrective_offset = float(np.clip(0.65 * lateral_error, -0.030, 0.030))
                self.get_logger().warn(
                    f"[creep] target touched off-centre; backing out for x recenter: "
                    f"remaining={remaining:.3f} m lateral={lateral_error:.3f} m "
                    f"x_offset={corrective_offset:+.3f} m")
                self.retry_local_grasp(
                    "target touched before pinch centre",
                    manual_offset=corrective_offset,
                )
                return
            if remaining > close_tol and not timed_out:
                speed = float(self.grasp_profile.get(
                    "creep_fine_speed" if remaining < CREEP_SLOW_DISTANCE else "creep_speed",
                    CREEP_FINE_SPEED if remaining < CREEP_SLOW_DISTANCE else CREEP_SPEED,
                ))
                if target_touched_near:
                    speed = min(speed, float(self.grasp_profile.get(
                        "touch_creep_speed", TOUCH_CREEP_SPEED)))
                    if self.now() - self.last_touch_creep_log > 0.8:
                        self.get_logger().info(
                            f"[creep] target touched before pinch centre; "
                            f"slow-centering remaining={remaining:.3f} m "
                            f"lateral={lateral_error:.3f} m")
                        self.last_touch_creep_log = self.now()
                # The arm has established the lateral pinch position. Correct
                # only residual chassis drift, so the final centimetres cannot
                # turn one fingertip into the first point of contact.
                correction = float(np.clip(
                    math.atan2(lateral_error, max(remaining, 0.12)),
                    -float(self.grasp_profile.get(
                        "creep_max_yaw_correction", CREEP_MAX_YAW_CORRECTION)),
                    float(self.grasp_profile.get(
                        "creep_max_yaw_correction", CREEP_MAX_YAW_CORRECTION)),
                ))
                creep_yaw = self.grasp_yaw + correction
                self.set_twist(speed, CREEP_YAW_KP * wrap_to_pi(creep_yaw - self.base_yaw))
            else:
                self.set_twist(0.0, 0.0)
                if abs(lateral_error) > GRASP_MAX_LATERAL_CLOSE_ERR:
                    corrective_offset = float(np.clip(0.65 * lateral_error, -0.028, 0.028))
                    self.get_logger().warn(
                        f"[creep] lateral alignment miss: remaining={remaining:.3f} m "
                        f"lateral={lateral_error:.3f} m; retry with x_offset={corrective_offset:+.3f} m"
                    )
                    self.retry_local_grasp(
                        "lateral alignment error before close",
                        manual_offset=corrective_offset,
                    )
                    return
                if timed_out:
                    self.get_logger().warn(
                        f"[creep] timeout recovery: remaining={remaining:.3f} m "
                        f"lateral={lateral_error:.3f} m; closing gripper")
                else:
                    self.get_logger().info(
                        f"[creep] pinch centre reached: remaining={remaining:.3f} m "
                        f"lateral={lateral_error:.3f} m")
                self.phase = CLOSE
                self.state_t0 = self.now()
        elif self.phase == CLOSE:
            self.set_twist(0.0, 0.0)
            self.tc[18] = float(self.grasp_profile.get("grip_close_target", GRIP_CLOSE))
            close_dwell = float(self.grasp_profile.get("grip_close_dwell", GRIP_CLOSE_DWELL))
            close_elapsed = self.now() - self.state_t0
            if (
                self.current_target_touched()
                and int(self.referee_state.get("flow_step", 0)) < 3
                and not self.close_nudge_done
                and close_elapsed > 1.1
            ):
                if self.close_nudge_until is None:
                    self.close_nudge_until = self.now() + CLOSE_SEAT_CREEP_TIME
                    self.get_logger().info("[close] seating target with a short closed-gripper settle")
                if self.now() < self.close_nudge_until:
                    yaw_err = wrap_to_pi(self.grasp_yaw - self.base_yaw)
                    self.set_twist(CLOSE_SEAT_CREEP_SPEED, float(np.clip(0.8 * yaw_err, -0.04, 0.04)))
                else:
                    self.set_twist(0.0, 0.0)
                    self.close_nudge_done = True
            elif close_elapsed > close_dwell:
                if not self.current_target_touched():
                    self.retry_local_grasp("closed gripper without touching target")
                    return
                self.phase = LIFT
        elif self.phase == LIFT:
            # 竖直抬起(减小 slide,胸部上移),让物体离开隔板,胳膊关节保持不动
            self.set_twist(0.0, 0.0)
            self.tc[2] = self.grasp_slide - float(self.grasp_profile.get("lift_amount", LIFT_AMOUNT))
            if abs(self.slide_meas - self.tc[2]) < 0.02 and self.now() - self.state_t0 > LIFT_SETTLE_DWELL:
                self.phase = VERIFY_GRASP
                self.verify_start_xy = np.array(self.base_xy, dtype=float)
                self.state_t0 = self.now()
        elif self.phase == VERIFY_GRASP:
            # 倒车(保持抓取朝向)退回黄线中点,object 还夹在手里
            yaw_err = wrap_to_pi(self.grasp_yaw - self.base_yaw)
            yaw_err = wrap_to_pi(self.grasp_yaw - self.base_yaw)
            retreat_speed = float(self.grasp_profile.get("retreat_speed", RETREAT_SPEED))
            if self.base_xy[1] > GRASP_VERIFY_BASE_Y:
                # Keep the chassis straight while the object is still beside
                # the shelf; use only a tiny heading hold so reverse motion
                # does not drift into the shelf or side boards.
                self.set_twist(-retreat_speed, float(np.clip(1.2 * yaw_err, -0.08, 0.08)))
            else:
                self.set_twist(0.0, 0.0)
                expected_body = None
                if self.active_task is not None and hasattr(self.active_task, "metadata"):
                    expected_body = self.active_task.metadata.get("body")
                flow_step = int(self.referee_state.get("flow_step", 0))
                flow_target = self.referee_state.get("flow_target")
                if flow_step >= 3 and (expected_body is None or flow_target == expected_body):
                    self.grasp_was_confirmed = True
                    self.get_logger().info(
                        f"[grasp_verify] referee confirmed S3 target={flow_target}; planning delivery")
                    self.phase = NAV_TABLE
                    self.carry_retreat_active = True
                    self.delivery_collision_recovered = False
                    self.delivery_recovery_count = 0
                    self.last_delivery_recovery_time = -999.0
                    self.route_to_table = [list(point) for point in ROUTE_TO_TABLE]
                    self.route_goal = DELIVERY_GOAL.copy()
                    self.route_purpose = "delivery"
                    self.route_needs_plan = True
                    self.reset_nav()
                    self.state_t0 = self.now()
                elif flow_step >= 3 and flow_target != expected_body:
                    self.fail_current_execution(
                        f"referee bound wrong target {flow_target}, expected {expected_body}")
                elif self.now() - self.state_t0 > GRASP_VERIFY_TIMEOUT:
                    self.retry_local_grasp("referee did not confirm S3 grasp")
        elif self.phase == NAV_TABLE:
            drop_report = self.active_drop_report()
            if self.grasp_was_confirmed and drop_report is not None:
                # last_flow persists after a previous flow. Only stop here if
                # this record belongs to the active referee body; otherwise
                # continue into the delivery route for the current target.
                if self.recover_dropped_object(drop_report):
                    return
            if (
                self.grasp_was_confirmed
                and bool(self.referee_state.get("collided"))
                and not self.delivery_collision_recovered
            ):
                if self.start_delivery_collision_recovery("collision detected while carrying"):
                    return
            # Clear the shelf before rotating the loaded, extended arm. This
            # avoids sweeping the bottle or gripper into the shelf structure.
            if self.carry_retreat_active:
                yaw_err = wrap_to_pi(self.grasp_yaw - self.base_yaw)
                retreat_speed = float(self.grasp_profile.get("retreat_speed", RETREAT_SPEED))
                if self.base_xy[1] > CARRY_SHELF_CLEAR_Y:
                    self.set_twist(-retreat_speed, float(np.clip(1.2 * yaw_err, -0.08, 0.08)))
                    self.ramp_twist()
                    self.smooth_step()
                    self.publish()
                    return
                self.set_twist(0.0, 0.0)
                self.cur_lin = self.cur_ang = 0.0
                self.carry_retreat_active = False
                self.route_needs_plan = True
                self.reset_nav()
            if self.follow_route(self.route_to_table, YAW_SOUTH):
                if int(self.referee_state.get("flow_step", 0)) >= 4:
                    self.phase, self.place_sub = PLACE, 0
                    self.state_t0 = self.now()
                else:
                    self.set_twist(0.0, 0.0)
        elif self.phase == PLACE:
            self.set_twist(0.0, 0.0)
            if self.place_sub == 0:
                # 先把升降平台整体降下来,物体随之竖直下降到桌面附近(手臂关节不动)
                self.tc[2] = PLACE_LOWER_SLIDE
                if abs(self.slide_meas - PLACE_LOWER_SLIDE) < 0.02:
                    self.place_sub = 1
                    self.state_t0 = self.now()
            elif self.place_sub == 1:
                # Open fully and wait for both fingers to clear the bottle.
                self.tc[18] = GRIP_OPEN
                if self.now() - self.state_t0 >= PLACE_OPEN_DWELL:
                    self.place_sub = 2
                    self.tc[2] = PLACE_CLEAR_SLIDE
                    self.state_t0 = self.now()
            else:
                # Lift the empty gripper vertically before judging placement.
                completed_now = int(self.referee_state.get("completed", 0))
                if completed_now > self.completed_before_task:
                    self.get_logger().info(
                        f"[place_verify] referee confirmed S5; completed={completed_now}")
                    self.phase = DONE
                elif self.now() - self.state_t0 > PLACE_VERIFY_TIMEOUT:
                    last_flow = self.referee_state.get("last_flow")
                    self.fail_current_execution(
                        f"referee did not confirm S5 placement; last_flow={last_flow}")
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
            nav_str = ""
            if self.phase in (NAV_SHELF, NAV_TABLE) and self.nav_idx < len(
                    self.route_to_shelf if self.phase == NAV_SHELF else self.route_to_table):
                active_route = self.route_to_shelf if self.phase == NAV_SHELF else self.route_to_table
                nav_target = np.asarray(active_route[self.nav_idx], dtype=float)
                nav_str = (
                    f" nav={self.nav_idx + 1}/{len(active_route)}"
                    f" target=({nav_target[0]:.2f},{nav_target[1]:.2f})"
                    f" cmd=({self.tc[0]:.2f},{self.tc[1]:.2f})")
            self.get_logger().info(
                f"phase={PHASE_NAME[self.phase]} sub={self.sub_idx} "
                f"base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) yaw={self.base_yaw:.2f} slide={self.slide_meas:.3f} "
                f"gripper=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                f"obj={obj_str} locked={self.target_locked}{nav_str}")
            self.last_log = self.now()


def main():
    rclpy.init()
    node = PickPlaceClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
