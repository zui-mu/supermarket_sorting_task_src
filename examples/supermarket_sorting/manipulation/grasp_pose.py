from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

BASE_GRASP_ROT = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

STRATEGY_TOOL_ROLL_DEG = {
    "front_center": 0.0,
    "front_bottle_wrap": 0.0,
    "front_box_clamp": 0.0,
    "front_short_axis_box_clamp": 0.0,
    "front_fruit_cradle": 0.0,
    "low_front_pinching": 0.0,
}


def grasp_rotation_for_strategy(
    strategy: str,
    profile: dict | None = None,
    *,
    base_rot: np.ndarray | None = None,
    strategy_roll_map: dict[str, float] | None = None,
) -> np.ndarray:
    """Build a wrist orientation for a grasp strategy.

    The baseline approach direction stays unchanged. Only the tool roll around
    the forward axis varies so wide boxes can be pinched with a vertical or
    side-on finger presentation instead of the default horizontal one.
    """
    strategy = str(strategy or "").strip()
    profile = profile or {}
    base_rot = BASE_GRASP_ROT if base_rot is None else np.asarray(base_rot, dtype=float)
    strategy_roll_map = STRATEGY_TOOL_ROLL_DEG if strategy_roll_map is None else strategy_roll_map

    pitch_deg = float(profile.get("wrist_pitch_deg", 0.0))
    roll_deg = profile.get("wrist_roll_deg")
    if roll_deg is None:
        roll_deg = strategy_roll_map.get(strategy, 0.0)
    roll_deg = float(roll_deg)
    rot = np.array(base_rot, dtype=float, copy=True)
    if abs(pitch_deg) >= 1e-9:
        rot = rot @ Rotation.from_euler("y", pitch_deg, degrees=True).as_matrix()
    if abs(roll_deg) >= 1e-9:
        rot = rot @ Rotation.from_euler("x", roll_deg, degrees=True).as_matrix()
    # This is a yaw about the footprint/world vertical, after the tool has
    # been pitched down. It swaps which horizontal box dimension the fingers
    # close across without turning the overhead approach into a side swipe.
    yaw_deg = float(profile.get("wrist_yaw_deg", 0.0))
    if abs(yaw_deg) >= 1e-9:
        rot = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ rot
    if abs(pitch_deg) < 1e-9 and abs(roll_deg) < 1e-9 and abs(yaw_deg) < 1e-9:
        return np.array(base_rot, dtype=float, copy=True)
    return rot


def finger_closing_axis(rotation: np.ndarray) -> np.ndarray:
    """Return the gripper's two-finger closing axis in the supplied frame.

    In the official MMK2 MJCF, the two finger slide joints are both aligned to
    the link-6 local Y axis.  Exposing this small piece of robot geometry keeps
    product-specific orientation code and runtime diagnostics tied to the
    actual actuator axis instead of a camera-view convention.
    """
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")
    axis = rotation[:, 1].copy()
    length = float(np.linalg.norm(axis))
    if length <= 1e-9:
        raise ValueError("rotation has an invalid finger-closing axis")
    return axis / length
