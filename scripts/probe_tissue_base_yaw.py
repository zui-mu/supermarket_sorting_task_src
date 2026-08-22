#!/usr/bin/env python3
"""Search base yaw offsets for the collision-safe C1 outer-side pinch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "supermarket_sorting"))

from manipulation.cartesian_path import interpolate_se3, plan_continuous_ik_path  # noqa: E402
from mmk2_kdl import MMK2Kdl  # noqa: E402


KDL_TO_LINK6 = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
)
# Shelf-frame columns: local X down, local Y along shelf depth, and local +Z
# from C1's outer side inward across the long box dimension.
SAFE_LINK6_SHELF = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]
)


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pose(position, orientation):
    result = np.eye(4)
    result[:3, :3] = orientation
    result[:3, 3] = position
    return result


def solve_path(kdl, start, goal, orientation, slide, seed):
    poses = interpolate_se3(
        pose(start, orientation), pose(goal, orientation),
        translation_step=0.012, rotation_step_rad=0.08,
    )

    def solve(candidate, previous):
        solutions = kdl.inverse_kinematics(
            T_right=candidate,
            ref_pos=np.concatenate(([slide], previous)),
            target_height=slide,
        )
        return [np.asarray(solution)[1:7] for solution in (solutions or ())]

    return plan_continuous_ik_path(poses, solve, seed, max_joint_step=0.45)


def main():
    kdl = MMK2Kdl()
    centre = np.array([0.60, 0.0, 0.895])
    stow = np.array([0.0, -0.166, 0.032, 0.0, -1.571, -2.223])
    for yaw_degrees in range(-120, 121, 15):
        yaw_offset = np.deg2rad(yaw_degrees)
        shelf_to_fp = rz(-yaw_offset)
        depth = shelf_to_fp @ np.array([1.0, 0.0, 0.0])
        outer = shelf_to_fp @ np.array([0.0, 1.0, 0.0])
        link6_R = shelf_to_fp @ SAFE_LINK6_SHELF
        endpoint_R = link6_R @ KDL_TO_LINK6.T
        grasp = centre + 0.070 * outer
        front = grasp - 0.200 * depth
        row = []
        for slide in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            solutions = kdl.inverse_kinematics(
                T_right=pose(front, endpoint_R),
                ref_pos=np.concatenate(([slide], stow)),
                target_height=slide,
            )
            if not solutions:
                row.append(0.0)
                continue
            path = solve_path(
                kdl, front, grasp, endpoint_R, slide,
                np.asarray(solutions[0])[1:7],
            )
            row.append(path.achieved_fraction)
        print(
            f"base_yaw_offset={yaw_degrees:4d}: slides[.20:.60]={np.round(row, 2)}"
        )

    print("physical side-stance search at yaw_offset=-90")
    yaw_offset = np.deg2rad(-90.0)
    shelf_to_fp = rz(-yaw_offset)
    depth = shelf_to_fp @ np.array([1.0, 0.0, 0.0])
    outer = shelf_to_fp @ np.array([0.0, 1.0, 0.0])
    link6_R = shelf_to_fp @ SAFE_LINK6_SHELF
    endpoint_R = link6_R @ KDL_TO_LINK6.T
    complete = []
    best = []
    for centre_x in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        for centre_y in (-0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25):
            centre = np.array([centre_x, centre_y, 0.895])
            grasp = centre + 0.070 * outer
            front = grasp - 0.180 * depth
            for slide in (0.20, 0.30, 0.40, 0.50, 0.60):
                solutions = kdl.inverse_kinematics(
                    T_right=pose(front, endpoint_R),
                    ref_pos=np.concatenate(([slide], stow)),
                    target_height=slide,
                )
                if not solutions:
                    continue
                path = solve_path(
                    kdl, front, grasp, endpoint_R, slide,
                    np.asarray(solutions[0])[1:7],
                )
                candidate = (path.achieved_fraction, centre_x, centre_y, slide)
                best.append(candidate)
                if path.complete:
                    complete.append(candidate)
    print("  complete configs:", complete[:30])
    print("  best configs:", sorted(best, reverse=True)[:15])


if __name__ == "__main__":
    main()
