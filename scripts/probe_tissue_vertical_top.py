#!/usr/bin/env python3
"""Search a high-insert then vertical-descend tissue pinch route."""

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


def solve_path(kdl, points, slide, seed):
    poses = []
    for start, goal in zip(points[:-1], points[1:]):
        segment = interpolate_se3(
            pose(start), pose(goal), translation_step=0.010, rotation_step_rad=0.08
        )
        if poses:
            segment = segment[1:]
        poses.extend(segment)

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
    for slide in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        for pre_z in (0.950, 0.960, 0.970, 0.980, 0.990):
            gateway = np.array([0.40, 0.0, pre_z])
            above = np.array([0.60, 0.0, pre_z])
            grasp = np.array([0.60, 0.0, 0.890])
            extracted = np.array([0.37, 0.0, 0.890])
            solutions = kdl.inverse_kinematics(
                T_right=pose(gateway),
                ref_pos=np.concatenate(([slide], stow)),
                target_height=slide,
            )
            if not solutions:
                print(f"slide={slide:.2f} pre_z={pre_z:.3f}: gateway=no-IK")
                continue
            high = solve_path(
                kdl, [gateway, above], slide, np.asarray(solutions[0])[1:7]
            )
            down = out = None
            if high.complete:
                down = solve_path(kdl, [above, grasp], slide, high.joint_path[-1])
            if down and down.complete:
                out = solve_path(kdl, [grasp, extracted], slide, down.joint_path[-1])
            joints = tuple(high.joint_path)
            if down:
                joints += tuple(down.joint_path)
            max_link4_z = float("nan")
            if joints:
                values = []
                for arm in joints:
                    transform = (
                        kdl.spine.get_transformation_matrix(float(slide))
                        @ kdl.spine2arm.get_transformation_matrix("right")
                    )
                    for index, joint in enumerate(arm[:4], start=1):
                        transform = transform @ kdl.right_arm.dh.adjacent_transform(
                            float(joint), index
                        )
                    values.append(float(transform[2, 3]))
                max_link4_z = max(values)
            print(
                f"slide={slide:.2f} pre_z={pre_z:.3f}: "
                f"high={high.achieved_fraction:.2f} "
                f"down={down.achieved_fraction if down else 0.0:.2f} "
                f"out={out.achieved_fraction if out else 0.0:.2f} "
                f"link4z={max_link4_z:.3f}"
            )


if __name__ == "__main__":
    main()
