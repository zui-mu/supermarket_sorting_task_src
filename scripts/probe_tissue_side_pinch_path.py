#!/usr/bin/env python3
"""Offline IK probe for a C1 tissue side-insertion short-axis pinch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "supermarket_sorting"))

from manipulation.cartesian_path import interpolate_se3, plan_continuous_ik_path  # noqa: E402
from mmk2_kdl import MMK2Kdl  # noqa: E402


# local -X points from the C1 outer aisle towards the box; local Y spans the
# shelf-depth short axis.  Local -Z points down, so the finger pads sit below
# link6 instead of forcing the wrist into the L2 shelf board.
SIDE_ROTATION = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def pose(position):
    result = np.eye(4)
    result[:3, :3] = SIDE_ROTATION
    result[:3, 3] = position
    return result


def solve_path(kdl, start_pose, goal_pose, slide, seed):
    poses = interpolate_se3(
        start_pose,
        goal_pose,
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
    # Match the competition client exactly; iteration=1 can hide branch
    # instability that appears with the production solver's ranked candidates.
    kdl = MMK2Kdl()
    centre = np.array([0.624, 0.0, 0.895])
    # Finger collision centre is [-0.100, +/-gap, -0.038] in endpoint-local
    # coordinates, so endpoint y=centre+0.100 and z=centre+0.038 centres both
    # finger pads on the tissue side face.
    grasp = centre + np.array([0.0, 0.100, 0.038])
    front = grasp + np.array([-0.240, 0.0, 0.0])
    extracted = grasp + np.array([0.0, 0.210, 0.020])
    stow = np.array([0.0, -0.166, 0.032, 0.0, -1.571, -2.223])

    for slide in np.linspace(0.10, 0.65, 12):
        front_solutions = kdl.inverse_kinematics(
            T_right=pose(front),
            ref_pos=np.concatenate(([slide], stow)),
            target_height=slide,
        )
        if not front_solutions:
            print(f"slide={slide:.2f}: front=no IK")
            continue
        front_joints = np.asarray(front_solutions[0])[1:7]
        insertion = solve_path(
            kdl, pose(front), pose(grasp), slide, front_joints
        )
        extraction = None
        if insertion.complete:
            extraction = solve_path(
                kdl,
                pose(grasp),
                pose(extracted),
                slide,
                insertion.joint_path[-1],
            )
        all_joints = tuple(insertion.joint_path) + tuple(
            extraction.joint_path if extraction else ()
        )
        max_jump = max(
            (
                np.max(np.abs(current - previous))
                for previous, current in zip(all_joints[:-1], all_joints[1:])
            ),
            default=0.0,
        )
        print(
            f"slide={slide:.2f}: front={np.round(front_joints, 3)} "
            f"insert={insertion.achieved_fraction:.2f}/{len(insertion.joint_path)} "
            f"extract={extraction.achieved_fraction if extraction else 0.0:.2f}/"
            f"{len(extraction.joint_path) if extraction else 0} max_jump={max_jump:.3f}"
        )
        if abs(slide - 0.45) < 1e-6 and insertion.complete:
            origins = []
            for joints in all_joints:
                transform = (
                    kdl.spine.get_transformation_matrix(float(slide))
                    @ kdl.spine2arm.get_transformation_matrix("right")
                )
                row = []
                for index, joint in enumerate(joints, start=1):
                    transform = transform @ kdl.right_arm.dh.adjacent_transform(
                        float(joint), index
                    )
                    row.append(transform[:3, 3].copy())
                origins.append(row)
            origins = np.asarray(origins)
            for link_index in range(2, 6):
                values = origins[:, link_index]
                print(
                    f"  link{link_index + 1} origin ranges: "
                    f"x=[{values[:, 0].min():.3f},{values[:, 0].max():.3f}] "
                    f"y=[{values[:, 1].min():.3f},{values[:, 1].max():.3f}] "
                    f"z=[{values[:, 2].min():.3f},{values[:, 2].max():.3f}]"
                )

    print("base-standoff robustness at slide=0.45")
    for centre_x in np.linspace(0.54, 0.68, 8):
        candidate_centre = centre.copy()
        candidate_centre[0] = centre_x
        candidate_grasp = candidate_centre + np.array([0.0, 0.100, -0.026])
        candidate_front = candidate_grasp + np.array([-0.240, 0.0, 0.0])
        solutions = kdl.inverse_kinematics(
            T_right=pose(candidate_front),
            ref_pos=np.concatenate(([0.45], stow)),
            target_height=0.45,
        )
        if not solutions:
            print(f"  centre_x={centre_x:.3f}: front=no IK")
            continue
        result = solve_path(
            kdl,
            pose(candidate_front),
            pose(candidate_grasp),
            0.45,
            np.asarray(solutions[0])[1:7],
        )
        print(
            f"  centre_x={centre_x:.3f}: insert={result.achieved_fraction:.2f}/"
            f"{len(result.joint_path)}"
        )

    print("endpoint-height robustness at centre_x=0.60, slide=0.45")
    for z_bias in np.linspace(-0.038, -0.004, 18):
        candidate_centre = centre.copy()
        candidate_centre[0] = 0.60
        candidate_grasp = candidate_centre + np.array([0.0, 0.100, z_bias])
        candidate_front = candidate_grasp + np.array([-0.240, 0.0, 0.0])
        solutions = kdl.inverse_kinematics(
            T_right=pose(candidate_front),
            ref_pos=np.concatenate(([0.45], stow)),
            target_height=0.45,
        )
        if not solutions:
            print(f"  z_bias={z_bias:.3f}: front=no IK")
            continue
        result = solve_path(
            kdl,
            pose(candidate_front),
            pose(candidate_grasp),
            0.45,
            np.asarray(solutions[0])[1:7],
        )
        print(
            f"  z_bias={z_bias:.3f}: insert={result.achieved_fraction:.2f}/"
            f"{len(result.joint_path)}"
        )

    print("raised-endpoint slide/standoff search (z_bias=-0.008)")
    for slide in np.linspace(0.35, 0.60, 6):
        row = []
        for centre_x in (0.60, 0.64, 0.68):
            candidate_centre = centre.copy()
            candidate_centre[0] = centre_x
            candidate_grasp = candidate_centre + np.array([0.0, 0.100, -0.008])
            candidate_front = candidate_grasp + np.array([-0.240, 0.0, 0.0])
            solutions = kdl.inverse_kinematics(
                T_right=pose(candidate_front),
                ref_pos=np.concatenate(([slide], stow)),
                target_height=slide,
            )
            if not solutions:
                row.append(0.0)
                continue
            result = solve_path(
                kdl,
                pose(candidate_front),
                pose(candidate_grasp),
                slide,
                np.asarray(solutions[0])[1:7],
            )
            row.append(result.achieved_fraction)
        print(f"  slide={slide:.2f}: fractions={np.round(row, 2)}")

    print("clearance/height side-channel search (centre_x=0.64, slide=0.40)")
    for clearance in (0.17, 0.18, 0.19, 0.20):
        row = []
        for z_bias in (0.028, 0.038, 0.048, 0.058):
            candidate_centre = centre.copy()
            candidate_centre[0] = 0.64
            candidate_grasp = candidate_centre + np.array(
                [0.0, clearance, z_bias]
            )
            candidate_front = candidate_grasp + np.array([-0.240, 0.0, 0.0])
            solutions = kdl.inverse_kinematics(
                T_right=pose(candidate_front),
                ref_pos=np.concatenate(([0.40], stow)),
                target_height=0.40,
            )
            if not solutions:
                row.append(0.0)
                continue
            result = solve_path(
                kdl,
                pose(candidate_front),
                pose(candidate_grasp),
                0.40,
                np.asarray(solutions[0])[1:7],
            )
            row.append(result.achieved_fraction)
        print(
            f"  clearance={clearance:.3f}: z[.028,.038,.048,.058]="
            f"{np.round(row, 2)}"
        )

    print("downward-pad slide/clearance search (centre_x=0.64, z_bias=.038)")
    for slide in (0.35, 0.40, 0.45, 0.50, 0.55):
        row = []
        for clearance in (0.16, 0.17, 0.18, 0.19, 0.20):
            candidate_centre = centre.copy()
            candidate_centre[0] = 0.64
            candidate_grasp = candidate_centre + np.array(
                [0.0, clearance, 0.038]
            )
            candidate_front = candidate_grasp + np.array([-0.240, 0.0, 0.0])
            solutions = kdl.inverse_kinematics(
                T_right=pose(candidate_front),
                ref_pos=np.concatenate(([slide], stow)),
                target_height=slide,
            )
            if not solutions:
                row.append(0.0)
                continue
            result = solve_path(
                kdl,
                pose(candidate_front),
                pose(candidate_grasp),
                slide,
                np.asarray(solutions[0])[1:7],
            )
            row.append(result.achieved_fraction)
        print(f"  slide={slide:.2f}: clear[.16:.20]={np.round(row, 2)}")


if __name__ == "__main__":
    main()
