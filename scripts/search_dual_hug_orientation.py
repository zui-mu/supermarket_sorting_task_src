#!/usr/bin/env python3
"""Search symmetric, inward/downward dual-gripper poses with complete IK."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "supermarket_sorting"))
from mmk2_kdl import MMK2Kdl  # noqa: E402


def pose(rotation, position):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def main():
    kdl = MMK2Kdl(iteration=1)
    centre = np.array([0.60, 0.0, 0.895])
    # Distal collision-mesh proxy measured from the link-6 endpoint.
    pad_local = np.array([-0.100, 0.0, -0.038])
    link6_box_local = np.array([0.0, 0.0, -0.070])
    link6_box_half_size = np.array([0.025, 0.080, 0.015])
    angles = np.linspace(-math.pi, math.pi, 17)
    heights = (0.35, 0.42, 0.50, 0.60, 0.70, 0.80)
    results = []
    pitch_bias = math.pi - 0.0551
    gateway_left_rotation = Rotation.from_euler(
        "zyx", [-math.pi / 2.0, pitch_bias, -math.pi / 8.0]
    ).as_matrix()
    gateway_right_rotation = Rotation.from_euler(
        "zyx", [math.pi / 2.0, pitch_bias, math.pi / 8.0]
    ).as_matrix()

    for yaw, pitch, roll in itertools.product(angles, repeat=3):
        right_rotation = Rotation.from_euler("zyx", [yaw, pitch, roll]).as_matrix()
        right_pad = right_rotation @ pad_local
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            left_rotation = Rotation.from_euler(
                "zyx",
                [signs[0] * yaw, signs[1] * pitch, signs[2] * roll],
            ).as_matrix()
            left_pad = left_rotation @ pad_local
            # Own-side wrists: left is +y and must point inward (-y), while
            # right is -y and must point inward (+y). Both pads point down.
            if not (
                left_pad[1] < -0.040
                and right_pad[1] > 0.040
                and left_pad[2] < -0.025
                and right_pad[2] < -0.025
                and left_pad[0] > 0.010
                and right_pad[0] > 0.010
            ):
                continue

            baseline_retract = 0.5 * (left_pad[0] + right_pad[0])
            left_link_centre = left_rotation @ link6_box_local
            right_link_centre = right_rotation @ link6_box_local
            left_link_x_extent = float(
                np.sum(np.abs(left_rotation[0]) * link6_box_half_size)
            )
            right_link_x_extent = float(
                np.sum(np.abs(right_rotation[0]) * link6_box_half_size)
            )
            link6_front = max(
                -baseline_retract + left_link_centre[0] + left_link_x_extent,
                -baseline_retract + right_link_centre[0] + right_link_x_extent,
            )
            if link6_front > -0.025:
                continue

            lateral = 0.081 + max(-left_pad[1], right_pad[1])
            pre_lateral = lateral + 0.035
            base_z_offset = max(-left_pad[2], -right_pad[2])
            for pad_height in (0.005, 0.020, 0.035, 0.045):
                z_offset = base_z_offset + pad_height
                for height in heights:
                    gateway_left = pose(
                        gateway_left_rotation,
                        centre + np.array([
                            -(baseline_retract + 0.180), pre_lateral, 0.130
                        ]),
                    )
                    gateway_right = pose(
                        gateway_right_rotation,
                        centre + np.array([
                            -(baseline_retract + 0.180), -pre_lateral, 0.130
                        ]),
                    )
                    gateway_solutions = kdl.inverse_kinematics(
                        gateway_left,
                        gateway_right,
                        ref_pos=np.zeros(13),
                        target_height=height,
                    )
                    if not gateway_solutions:
                        continue
                    previous = np.asarray(gateway_solutions[0], dtype=float)
                    stages = (
                        (pre_lateral, baseline_retract + 0.180, z_offset, 0.0),
                        (pre_lateral, baseline_retract, z_offset, 0.0),
                        (lateral, baseline_retract, z_offset, 0.0),
                        (lateral, baseline_retract + 0.300, z_offset + 0.080, 0.080),
                    )
                    feasible = True
                    for stage_lateral, pull, stage_z, lift in stages:
                        left = pose(
                            left_rotation,
                            centre + np.array([-pull, stage_lateral, stage_z]),
                        )
                        right = pose(
                            right_rotation,
                            centre + np.array([-pull, -stage_lateral, stage_z]),
                        )
                        solutions = kdl.inverse_kinematics(
                            left,
                            right,
                            ref_pos=previous,
                            target_height=height - lift,
                        )
                        if not solutions:
                            feasible = False
                            break
                        selected = min(
                            (np.asarray(solution, dtype=float) for solution in solutions),
                            key=lambda solution: float(np.linalg.norm(solution - previous)),
                        )
                        if (
                            stage_lateral == pre_lateral
                            and abs(pull - (baseline_retract + 0.180)) < 1e-9
                        ):
                            transition = float(np.max(np.abs(selected - previous)))
                        previous = selected
                    if feasible:
                        score = 10.0 * transition + height + lateral + z_offset
                        results.append((
                            score,
                            transition,
                            yaw,
                            pitch,
                            roll,
                            signs,
                            height,
                            lateral,
                            pre_lateral,
                            z_offset,
                            baseline_retract,
                            left_pad,
                            right_pad,
                        ))

    results.sort(key=lambda item: item[0])
    print(f"complete solutions: {len(results)}")
    for item in results[:20]:
        (_, transition, yaw, pitch, roll, signs, height, lateral, pre, z, baseline, left_pad, right_pad) = item
        print(
            "angles=[%.4f, %.4f, %.4f] signs=%s transition=%.3f slide=%.3f "
            "lateral=%.3f pre=%.3f z=%.3f baseline=%.3f left_pad=%s right_pad=%s"
            % (
                yaw,
                pitch,
                roll,
                signs,
                transition,
                height,
                lateral,
                pre,
                z,
                baseline,
                np.round(left_pad, 3),
                np.round(right_pad, 3),
            )
        )
    print("largest inward pad offsets:")
    for item in sorted(results, key=lambda candidate: candidate[7], reverse=True)[:12]:
        (_, transition, yaw, pitch, roll, signs, height, lateral, pre, z, baseline, left_pad, right_pad) = item
        print(
            "angles=[%.4f, %.4f, %.4f] signs=%s transition=%.3f slide=%.3f "
            "lateral=%.3f pre=%.3f z=%.3f baseline=%.3f left_pad=%s right_pad=%s"
            % (
                yaw,
                pitch,
                roll,
                signs,
                transition,
                height,
                lateral,
                pre,
                z,
                baseline,
                np.round(left_pad, 3),
                np.round(right_pad, 3),
            )
        )


if __name__ == "__main__":
    main()
