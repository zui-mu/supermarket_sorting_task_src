#!/usr/bin/env python3
"""Search the depth-first, short-lateral tissue edge-clamp route."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "supermarket_sorting"))

from manipulation.cartesian_path import interpolate_se3, plan_continuous_ik_path  # noqa: E402
from mmk2_kdl import MMK2Kdl  # noqa: E402

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
        pose(start), pose(goal), translation_step=0.010, rotation_step_rad=0.08
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
    complete = []
    for slide in (0.10, 0.15, 0.20, 0.25):
        for target_x in (0.56, 0.58, 0.60, 0.62, 0.64, 0.66):
            for outer_y in (0.090, 0.105, 0.120, 0.125, 0.130, 0.135, 0.150):
                for edge_move in (0.020, 0.040, 0.050, 0.080):
                    contact_y = outer_y - edge_move
                    gateway = np.array([target_x - 0.200, outer_y, 0.890])
                    aligned = np.array([target_x, outer_y, 0.890])
                    contact = np.array([target_x, contact_y, 0.890])
                    extracted = contact + np.array([-0.230, 0.0, 0.0])
                    solutions = kdl.inverse_kinematics(
                        T_right=pose(gateway),
                        ref_pos=np.concatenate(([slide], stow)),
                        target_height=slide,
                    )
                    if not solutions:
                        continue
                    depth = solve_path(
                        kdl, gateway, aligned, slide, np.asarray(solutions[0])[1:7]
                    )
                    edge = out = None
                    if depth.complete:
                        edge = solve_path(kdl, aligned, contact, slide, depth.joint_path[-1])
                    if edge and edge.complete:
                        out = solve_path(kdl, contact, extracted, slide, edge.joint_path[-1])
                    if depth.complete and edge and edge.complete and out and out.complete:
                        complete.append((slide, target_x, outer_y, contact_y))
    print("complete edge routes:")
    for candidate in complete:
        print(" ", candidate)


if __name__ == "__main__":
    main()
