#!/usr/bin/env python3
"""Probe the CURRENT tissue_top geometry (matches supermarket_sorting_client.py).

Unlike probe_tissue_vertical_top.py (which still uses the older rotation and a
centre-relative grasp_z of -0.005), this script mirrors the exact parameters of
the live `tick_tissue_top_pinch` state machine:

  * endpoint rotation = [[0,1,0],[1,0,0],[0,0,-1]]   (tissue_top_rotation)
  * above  = centre + [0, 0, tissue_top_pre_z]        (0.200)
  * grasp  = centre + [0, 0, tissue_top_grasp_z]      (0.060)
  * extracted = centre + [-tissue_top_pull, 0, grasp_z]  (pull 0.240)

It chains insertion (above), descent (above->grasp) and extraction
(grasp->extracted) exactly like the client, and scans slide x centre_x x lateral
so we can see where the L2/L3 descent becomes IK-incomplete.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "supermarket_sorting"))

from manipulation.cartesian_path import interpolate_se3, plan_continuous_ik_path  # noqa: E402
from mmk2_kdl import MMK2Kdl  # noqa: E402

# The live client's tissue_top_rotation.
ROTATION = np.array(
    [[0.0, 1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0]]
)

PRE_Z = 0.200      # tissue_top_pre_z
GRASP_Z = 0.060    # tissue_top_grasp_z
PULL = 0.240       # tissue_top_pull
RETRACT = 0.200    # tissue_top_gateway_retract
SLIDE_TRAVEL = 0.0


def pose(position):
    result = np.eye(4)
    result[:3, :3] = ROTATION
    result[:3, 3] = position
    return result


def solve_ik_pose(kdl, target_pose, slide, seed):
    """target_pose is a 4x4 SE(3)."""
    solutions = kdl.inverse_kinematics(
        T_right=target_pose,
        ref_pos=np.concatenate(([slide], seed)),
        target_height=slide,
    )
    return [np.asarray(sol)[1:7] for sol in (solutions or ())]


def solve_ik_pos(kdl, position, slide, seed):
    return solve_ik_pose(kdl, pose(position), slide, seed)


def solve_path(kdl, start, goal, slide, seed):
    poses = interpolate_se3(
        pose(start), pose(goal), translation_step=0.010, rotation_step_rad=0.08
    )

    def solve(candidate, previous):
        return solve_ik_pose(kdl, candidate, slide, previous)

    return plan_continuous_ik_path(poses, solve, seed, max_joint_step=0.45)


def probe_one(kdl, slide, centre_x, lateral, centre_z, seed=None):
    """Return (descent_fraction, extraction_fraction or None)."""
    centre = np.array([centre_x, lateral, centre_z])
    above = centre + np.array([0.0, 0.0, PRE_Z])
    grasp = centre + np.array([0.0, 0.0, GRASP_Z])
    extracted = centre + np.array([-PULL, 0.0, GRASP_Z])

    # Insertion (above) -- solve directly with a seed scan if needed.
    if seed is None:
        seed = np.array([0.0, -0.3, 0.2, 0.0, -1.3, -1.6])
    above_joints = None
    for trial_seed in (seed,):
        sols = solve_ik_pos(kdl, above, slide, trial_seed)
        if sols:
            above_joints = sols[0]
            break

    if above_joints is None:
        return 0.0, None  # above unreachable

    # Descent above -> grasp.
    descent = solve_path(kdl, above, grasp, slide, above_joints)
    out = None
    if descent.complete:
        out = solve_path(kdl, grasp, extracted, slide, descent.joint_path[-1])
    return descent.achieved_fraction, (out.achieved_fraction if out else 0.0)


def main():
    kdl = MMK2Kdl()
    # L2 and L3 actual centre z from retail_competition_layout.json.
    levels = {
        "L2": 0.895,
        "L3": 1.229,
    }
    for level, centre_z in levels.items():
        print(f"== level={level} centre_z={centre_z} ==")
        # L3 uses grasp_slide=-0.030 (team profile); L2 uses 0.253 (default).
        if level == "L3":
            slides = (-0.030, -0.020, 0.0, 0.05, 0.10, 0.20)
        else:
            slides = (0.20, 0.253, 0.30, 0.35)
        for slide in slides:
            for centre_x in (0.52, 0.574, 0.60, 0.66):
                for lateral in (0.0, 0.078, 0.12):
                    dfrac, ofrac = probe_one(kdl, slide, centre_x, lateral, centre_z)
                    out_s = f"{ofrac:.2f}" if ofrac is not None else " n/a"
                    print(
                        f"  slide={slide:+.3f} x={centre_x:.3f} lat={lateral:.3f}: "
                        f"descent={dfrac:.2f} extraction={out_s}"
                    )


if __name__ == "__main__":
    main()
