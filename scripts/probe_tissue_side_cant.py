#!/usr/bin/env python3
"""Sweep wrist cant between the reachable and collision-safe C1 pinches."""

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
REACHABLE_ENDPOINT = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
)
REACHABLE_LINK6 = REACHABLE_ENDPOINT @ KDL_TO_LINK6


def rotation(theta, symmetric=False):
    c, s = np.cos(theta), np.sin(theta)
    local_y_cant = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    base_link6 = REACHABLE_LINK6
    if symmetric:
        base_link6 = base_link6 @ np.diag([-1.0, -1.0, 1.0])
    desired_link6 = base_link6 @ local_y_cant
    return desired_link6 @ KDL_TO_LINK6.T, desired_link6


def pose(position, orientation):
    result = np.eye(4)
    result[:3, :3] = orientation
    result[:3, 3] = position
    return result


def solve_path(kdl, start, goal, orientation, slide, seed):
    poses = interpolate_se3(
        pose(start, orientation),
        pose(goal, orientation),
        translation_step=0.012,
        rotation_step_rad=0.08,
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
    for symmetric in (False, True):
      print(f"symmetric_roll={symmetric}")
      for degrees in range(0, 181, 15):
        endpoint_R, link6_R = rotation(np.deg2rad(degrees), symmetric)
        # Use the short positive-Z fingertip once the cant points it inward.
        inward = max(0.0, -float(link6_R[1, 2]))
        upward = float(link6_R[2, 2])
        standoff = 0.086 + 0.020 * inward
        grasp = centre + np.array([0.0, standoff, -0.020 * upward])
        front = grasp + np.array([-0.200, 0.0, 0.0])
        row = []
        for slide in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
            solutions = kdl.inverse_kinematics(
                T_right=pose(front, endpoint_R),
                ref_pos=np.concatenate(([slide], stow)),
                target_height=slide,
            )
            if not solutions:
                row.append(0.0)
                continue
            path = solve_path(
                kdl,
                front,
                grasp,
                endpoint_R,
                slide,
                np.asarray(solutions[0])[1:7],
            )
            row.append(path.achieved_fraction)
        print(
            f"cant={degrees:3d} link_z={np.round(link6_R[:, 2], 3)} "
            f"slides[.25:.65]={np.round(row, 2)}"
        )


if __name__ == "__main__":
    main()
