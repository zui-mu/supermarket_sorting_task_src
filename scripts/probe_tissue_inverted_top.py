#!/usr/bin/env python3
"""Verify the calibrated inverted-top tissue pinch and extraction path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "supermarket_sorting"))

from manipulation.cartesian_path import interpolate_se3, plan_continuous_ik_path  # noqa: E402
from mmk2_kdl import MMK2Kdl  # noqa: E402


# KDL endpoint rotation which, after the calibrated fixed endpoint->link6
# transform, makes link6 local +Z point down and its finger-closing Y axis
# span the shelf-depth short dimension.
ROTATION = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
)


def pose(position):
    result = np.eye(4)
    result[:3, :3] = ROTATION
    result[:3, 3] = position
    return result


def solve_path(kdl, start, goal, slide, seed):
    poses = interpolate_se3(
        pose(start), pose(goal), translation_step=0.012, rotation_step_rad=0.08
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
    stow = np.array([0.0, -0.166, 0.032, 0.0, -1.571, -2.223])
    for centre_x in (0.66, 0.68, 0.70, 0.72, 0.74, 0.76):
        for endpoint_z_bias in (0.010, 0.020, 0.030, 0.040):
            centre = np.array([centre_x, 0.0, 0.895])
            grasp = centre + np.array([0.0, 0.0, endpoint_z_bias])
            front = grasp + np.array([-0.200, 0.0, 0.0])
            extracted = grasp + np.array([-0.230, 0.0, 0.0])
            slide = 0.20
            solutions = kdl.inverse_kinematics(
                T_right=pose(front),
                ref_pos=np.concatenate(([slide], stow)),
                target_height=slide,
            )
            if not solutions:
                print(f"x={centre_x:.2f} z={endpoint_z_bias:.3f}: gateway=no-IK")
                continue
            insertion = solve_path(
                kdl, front, grasp, slide, np.asarray(solutions[0])[1:7]
            )
            extraction = None
            if insertion.complete:
                extraction = solve_path(
                    kdl, grasp, extracted, slide, insertion.joint_path[-1]
                )
            print(
                f"x={centre_x:.2f} z={endpoint_z_bias:.3f}: "
                f"in={insertion.achieved_fraction:.2f} "
                f"out={extraction.achieved_fraction if extraction else 0.0:.2f}"
            )


if __name__ == "__main__":
    main()
