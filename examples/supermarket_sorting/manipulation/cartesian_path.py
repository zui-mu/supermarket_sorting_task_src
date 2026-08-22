"""Small Cartesian-path helpers for the competition arm controller.

The official client image does not include MoveIt. These helpers keep the
important safety properties of a Cartesian path stage: bounded SE(3) steps,
continuous IK selection, and an explicit achieved fraction on failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class CartesianIKResult:
    joint_path: tuple[np.ndarray, ...]
    achieved_fraction: float
    reason: str = ""

    @property
    def complete(self) -> bool:
        return self.achieved_fraction >= 1.0 - 1e-9


def interpolate_se3(
    start: np.ndarray,
    goal: np.ndarray,
    *,
    translation_step: float = 0.01,
    rotation_step_rad: float = 0.12,
    segments: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Return SE(3) samples after ``start``, including ``goal``.

    ``segments`` lets synchronized multi-arm callers share one time grid. If
    omitted, the smallest sample count satisfying both step limits is used.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if start.shape != (4, 4) or goal.shape != (4, 4):
        raise ValueError("start and goal must be 4x4 transforms")
    if translation_step <= 0.0 or rotation_step_rad <= 0.0:
        raise ValueError("Cartesian interpolation steps must be positive")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
        raise ValueError("transforms must be finite")

    distance = float(np.linalg.norm(goal[:3, 3] - start[:3, 3]))
    relative = Rotation.from_matrix(start[:3, :3].T @ goal[:3, :3])
    angle = float(relative.magnitude())
    minimum_segments = max(
        1,
        int(math.ceil(distance / translation_step)),
        int(math.ceil(angle / rotation_step_rad)),
    )
    if segments is None:
        segments = minimum_segments
    elif not isinstance(segments, int) or isinstance(segments, bool) or segments < 1:
        raise ValueError("segments must be a positive integer")
    elif segments < minimum_segments:
        raise ValueError(
            "segments is too small to satisfy the Cartesian interpolation limits"
        )

    rotations = Rotation.from_matrix(np.stack((start[:3, :3], goal[:3, :3])))
    slerp = Slerp([0.0, 1.0], rotations)
    result = []
    for fraction in np.linspace(1.0 / segments, 1.0, segments):
        pose = np.eye(4)
        pose[:3, :3] = slerp([float(fraction)]).as_matrix()[0]
        pose[:3, 3] = (
            (1.0 - fraction) * start[:3, 3]
            + fraction * goal[:3, 3]
        )
        result.append(pose)
    return tuple(result)


def plan_continuous_ik_path(
    poses: Iterable[np.ndarray],
    solve_fn: Callable[[np.ndarray, np.ndarray], Sequence[np.ndarray]],
    seed_joints: np.ndarray,
    *,
    max_joint_step: float = 0.30,
) -> CartesianIKResult:
    """Solve each pose while rejecting invalid or discontinuous IK branches."""
    poses = tuple(poses)
    if max_joint_step <= 0.0:
        raise ValueError("max_joint_step must be positive")
    previous = np.asarray(seed_joints, dtype=float).copy()
    if previous.ndim != 1 or not np.all(np.isfinite(previous)):
        raise ValueError("seed_joints must be a finite vector")

    path = []
    for index, pose in enumerate(poses):
        raw_solutions = solve_fn(np.asarray(pose, dtype=float), previous.copy())
        candidates = []
        if raw_solutions is None:
            raw_solutions = ()
        for raw in raw_solutions:
            candidate = np.asarray(raw, dtype=float)
            if candidate.shape != previous.shape or not np.all(np.isfinite(candidate)):
                continue
            delta = np.abs(candidate - previous)
            if float(np.max(delta)) <= max_joint_step:
                candidates.append((float(np.linalg.norm(delta)), candidate.copy()))
        if not candidates:
            fraction = index / len(poses) if poses else 1.0
            reason = f"no continuous IK solution at waypoint {index + 1}/{len(poses)}"
            return CartesianIKResult(tuple(path), fraction, reason)
        _, selected = min(candidates, key=lambda item: item[0])
        path.append(selected)
        previous = selected

    return CartesianIKResult(tuple(path), 1.0, "")
