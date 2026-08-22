#!/usr/bin/env python3
"""Offline reachability probe for the tissue top-pinch Cartesian route."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "supermarket_sorting"))

from manipulation.cartesian_path import interpolate_se3, plan_continuous_ik_path  # noqa: E402
from mmk2_kdl import MMK2Kdl  # noqa: E402


TOP_ROTATION = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
)


def pose(position: np.ndarray, rotation: np.ndarray = TOP_ROTATION) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def solve_path(kdl, positions, slide, seed, rotation=TOP_ROTATION):
    poses = []
    for start, goal in zip(positions[:-1], positions[1:]):
        segment = interpolate_se3(
            pose(start, rotation), pose(goal, rotation), translation_step=0.015, rotation_step_rad=0.08
        )
        if poses:
            segment = segment[1:]
        poses.extend(segment)

    def solve(candidate, previous):
        ref = np.concatenate(([slide], previous))
        solutions = kdl.inverse_kinematics(
            T_right=candidate, ref_pos=ref, target_height=slide
        )
        return [np.asarray(solution)[1:7] for solution in (solutions or ())]

    return plan_continuous_ik_path(
        poses, solve, np.asarray(seed), max_joint_step=0.45
    )


def main():
    kdl = MMK2Kdl(iteration=1)
    centre = np.array([0.60, 0.0, 0.895])
    slide_travel = 0.0
    slide_grasp = 0.50
    stow_seed = np.array([0.0, -0.25, 0.0, 1.05, 0.0, 0.35])

    print("orientation variants at the selected 0.30-m gateway")
    gateway = centre + np.array([-0.30, 0.0, 0.70])
    rotations = {
        "forward_y": TOP_ROTATION,
        "reverse_y": TOP_ROTATION @ np.diag([1.0, -1.0, -1.0]),
    }
    for name, rotation in rotations.items():
        solutions = kdl.inverse_kinematics(
            T_right=pose(gateway, rotation),
            ref_pos=np.concatenate(([slide_travel], stow_seed)),
            target_height=slide_travel,
        )
        print(
            f"  {name}: "
            + (f"joints={np.round(np.asarray(solutions[0])[1:7], 3)}" if solutions else "no IK")
        )

    rng = np.random.default_rng(7)
    branches = []
    for _ in range(160):
        ref = rng.uniform(-2.6, 2.6, 6)
        solutions = kdl.inverse_kinematics(
            T_right=pose(gateway),
            ref_pos=np.concatenate(([slide_travel], ref)),
            target_height=slide_travel,
        )
        for solution in solutions or ():
            joints = np.asarray(solution)[1:7]
            if not any(np.max(np.abs(joints - known)) < 0.05 for known in branches):
                branches.append(joints)
    print(f"  unique IK branches from random references: {len(branches)}")
    for joints in sorted(branches, key=lambda q: abs(q[4]))[:12]:
        print(f"    joints={np.round(joints, 3)}")

    print("slide-height gateway variants")
    for candidate_slide in np.linspace(0.20, 0.50, 7):
        candidate_gateway = centre + np.array([-0.30, 0.0, 0.20 + candidate_slide])
        solutions = kdl.inverse_kinematics(
            T_right=pose(candidate_gateway),
            ref_pos=np.concatenate(([slide_travel], stow_seed)),
            target_height=slide_travel,
        )
        if not solutions:
            print(f"  slide={candidate_slide:.2f}: no gateway IK")
            continue
        gateway_joints = np.asarray(solutions[0])[1:7]
        high_over = centre + np.array([0.0, 0.0, 0.20 + candidate_slide])
        high_path = solve_path(
            kdl, [candidate_gateway, high_over], slide_travel, gateway_joints
        )
        low_pre = centre + np.array([0.0, 0.0, 0.20])
        low_solutions = kdl.inverse_kinematics(
            T_right=pose(low_pre),
            ref_pos=np.concatenate(([candidate_slide], high_path.joint_path[-1]))
            if high_path.joint_path
            else None,
            target_height=candidate_slide,
        )
        descend = None
        if low_solutions:
            descend = solve_path(
                kdl,
                [low_pre, centre + np.array([0.0, 0.0, 0.07])],
                candidate_slide,
                np.asarray(low_solutions[0])[1:7],
            )
        print(
            f"  slide={candidate_slide:.2f}: gateway={np.round(gateway_joints, 3)} "
            f"high={high_path.achieved_fraction:.2f} "
            f"descend={descend.achieved_fraction if descend else 0.0:.2f}"
        )

    # Replay the measured equilibrium from Probe 44: the proximal joints have
    # reached the outside gateway while wrist joint 5 is torque-limited.  Test
    # whether a Cartesian forward-and-orient segment has a continuous branch.
    stalled = np.array([0.119, -1.572, 0.607, 0.128, 0.480, 1.363])
    _, stalled_pose = kdl.forward_kinematics(
        np.concatenate(([slide_travel], stalled)), index="right"
    )
    high_over = centre + np.array([0.0, 0.0, 0.70])
    recovery_poses = interpolate_se3(
        stalled_pose,
        pose(high_over),
        translation_step=0.015,
        rotation_step_rad=0.08,
    )

    def solve_recovery(candidate, previous):
        solutions = kdl.inverse_kinematics(
            T_right=candidate,
            ref_pos=np.concatenate(([slide_travel], previous)),
            target_height=slide_travel,
        )
        return [np.asarray(solution)[1:7] for solution in (solutions or ())]

    recovery = plan_continuous_ik_path(
        recovery_poses, solve_recovery, stalled, max_joint_step=0.45
    )
    print(
        "stalled-gateway Cartesian recovery: "
        f"fraction={recovery.achieved_fraction:.2f} waypoints={len(recovery.joint_path)} "
        f"max_joint5={max((q[4] for q in recovery.joint_path), default=float('nan')):.3f}"
    )

    _, stalled_low_pose = kdl.forward_kinematics(
        np.concatenate(([slide_grasp], stalled)), index="right"
    )
    low_pre = centre + np.array([0.0, 0.0, 0.20])
    low_insert_poses = interpolate_se3(
        stalled_low_pose,
        pose(low_pre),
        translation_step=0.015,
        rotation_step_rad=0.08,
    )

    def solve_low_insert(candidate, previous):
        solutions = kdl.inverse_kinematics(
            T_right=candidate,
            ref_pos=np.concatenate(([slide_grasp], previous)),
            target_height=slide_grasp,
        )
        return [np.asarray(solution)[1:7] for solution in (solutions or ())]

    low_insert = plan_continuous_ik_path(
        low_insert_poses, solve_low_insert, stalled, max_joint_step=0.45
    )
    print(
        "lower-outside then Cartesian insert: "
        f"fraction={low_insert.achieved_fraction:.2f} waypoints={len(low_insert.joint_path)}"
    )

    print("high retracted endpoint candidates")
    candidates = []
    for retract in np.linspace(0.20, 0.40, 9):
        target = centre + np.array([-retract, 0.0, 0.70])
        solutions = kdl.inverse_kinematics(
            T_right=pose(target),
            ref_pos=np.concatenate(([slide_travel], stow_seed)),
            target_height=slide_travel,
        )
        if not solutions:
            print(f"  retract={retract:.3f}: no IK")
            continue
        joints = np.asarray(solutions[0])[1:7]
        candidates.append((retract, target, joints))
        print(
            f"  retract={retract:.3f}: joints={np.round(joints, 3)} "
            f"jump={np.max(np.abs(joints - stow_seed)):.3f}"
        )

    print("\ncomplete route candidates")
    for retract, high_retracted, seed in candidates:
        high_over = centre + np.array([0.0, 0.0, 0.70])
        high_path = solve_path(
            kdl, [high_retracted, high_over], slide_travel, seed
        )
        low_pre = centre + np.array([0.0, 0.0, 0.20])
        low_solutions = kdl.inverse_kinematics(
            T_right=pose(low_pre),
            ref_pos=np.concatenate(([slide_grasp], high_path.joint_path[-1]))
            if high_path.joint_path
            else None,
            target_height=slide_grasp,
        )
        if not high_path.complete or not low_solutions:
            print(
                f"  retract={retract:.3f}: high={high_path.achieved_fraction:.2f} "
                f"low_pre={'yes' if low_solutions else 'no'}"
            )
            continue
        low_seed = np.asarray(low_solutions[0])[1:7]
        descend = solve_path(
            kdl,
            [low_pre, centre + np.array([0.0, 0.0, 0.07])],
            slide_grasp,
            low_seed,
        )
        extraction = solve_path(
            kdl,
            [
                centre + np.array([0.0, 0.0, 0.15]),
                centre + np.array([-0.28, 0.0, 0.15]),
            ],
            slide_grasp - 0.08,
            descend.joint_path[-1] if descend.joint_path else low_seed,
        )
        print(
            f"  retract={retract:.3f}: high={high_path.achieved_fraction:.2f} "
            f"descend={descend.achieved_fraction:.2f} "
            f"extract={extraction.achieved_fraction:.2f}"
        )


if __name__ == "__main__":
    main()
