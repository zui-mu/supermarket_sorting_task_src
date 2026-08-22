#!/usr/bin/env python3
"""Offline continuous-IK sweep for the C1 positive-Z fingertip pinch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "supermarket_sorting"))

from manipulation.cartesian_path import interpolate_se3, plan_continuous_ik_path  # noqa: E402
from mmk2_kdl import MMK2Kdl  # noqa: E402


ROTATION = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
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
    centre = np.array([0.64, 0.0, 0.895])
    stow = np.array([0.0, -0.166, 0.032, 0.0, -1.571, -2.223])
    for slide in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
        for standoff in (0.060, 0.070, 0.080, 0.090, 0.100, 0.110):
            grasp = centre + np.array([0.0, standoff, 0.0])
            front = grasp + np.array([-0.240, 0.0, 0.0])
            extracted = grasp + np.array([-0.230, 0.0, 0.020])
            solutions = kdl.inverse_kinematics(
                T_right=pose(front),
                ref_pos=np.concatenate(([slide], stow)),
                target_height=slide,
            )
            if not solutions:
                print(f"slide={slide:.2f} stand={standoff:.3f}: gateway=no-IK")
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
                f"slide={slide:.2f} stand={standoff:.3f}: "
                f"in={insertion.achieved_fraction:.2f} "
                f"out={extraction.achieved_fraction if extraction else 0.0:.2f}"
            )

    print("base-distance/retract search at standoff=.070")
    for slide in (0.30, 0.45, 0.50, 0.55, 0.60):
        row = []
        for centre_x in (0.50, 0.54, 0.58, 0.62, 0.66, 0.70):
            candidate_centre = centre.copy()
            candidate_centre[0] = centre_x
            grasp = candidate_centre + np.array([0.0, 0.070, 0.0])
            front = grasp + np.array([-0.200, 0.0, 0.0])
            solutions = kdl.inverse_kinematics(
                T_right=pose(front),
                ref_pos=np.concatenate(([slide], stow)),
                target_height=slide,
            )
            if not solutions:
                row.append(0.0)
                continue
            insertion = solve_path(
                kdl, front, grasp, slide, np.asarray(solutions[0])[1:7]
            )
            row.append(insertion.achieved_fraction)
        print(f"  slide={slide:.2f}: centre_x[.50:.70]={np.round(row, 2)}")


if __name__ == "__main__":
    main()
