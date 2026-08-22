#!/usr/bin/env python3
"""Contact/dynamics sweep for the collision-safe tissue long-edge route."""

from __future__ import annotations

import itertools
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from probe_tissue_contact_grid import ContactProbe, pose  # noqa: E402


SIDE_ROTATION = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=float
)
CALIBRATION = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float
)


def canted_rotation(cant_deg: float) -> np.ndarray:
    """Cant physical link6 about its down axis to equalise fingertip entry."""
    angle = math.radians(cant_deg)
    c, s = math.cos(angle), math.sin(angle)
    local_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return SIDE_ROTATION @ CALIBRATION @ local_z @ CALIBRATION.T


def run_candidate(
    probe: ContactProbe,
    *,
    slide: float,
    tip: float,
    depth: float,
    cant: float,
) -> dict[str, object] | None:
    rotation = canted_rotation(cant)
    centre = np.array([0.565, 0.0, 0.895])
    start = centre + np.array([depth, 0.155, -0.010])
    grasp = centre + np.array([depth, tip, -0.010])
    post_close_inset = float(os.getenv("PROBE_POST_CLOSE_INSET", "0.0"))
    pull_distance = float(os.getenv("PROBE_PULL", "0.35"))
    inset_grasp = grasp + np.array([0.0, -post_close_inset, 0.0])
    pulled = inset_grasp + np.array([-pull_distance, 0.0, 0.0])
    references = [
        np.array([0.0, -0.166, 0.032, 0.0, -1.571, -2.223]),
        np.array([0.0, -1.0, 1.0, 0.0, -1.0, 0.0]),
    ]
    rng = np.random.default_rng(113)
    references.extend(rng.uniform(-2.6, 2.6, (10, 6)))
    candidates = []
    for reference in references:
        solutions = probe.kdl.inverse_kinematics(
            T_right=pose(start, rotation),
            ref_pos=np.concatenate(([slide], reference)),
            target_height=slide,
        )
        for solution in solutions or ():
            joints = np.asarray(solution)[1:7]
            if not any(np.max(np.abs(joints - known)) < 0.03 for known in candidates):
                candidates.append(joints)
    selected = None
    for initial in candidates:
        approach = probe.solve_path([start, grasp], rotation, slide, initial)
        if not approach:
            continue
        extraction_points = (
            [grasp, inset_grasp, pulled]
            if post_close_inset > 1e-6
            else [grasp, pulled]
        )
        extraction = probe.solve_path(
            extraction_points, rotation, slide, approach[-1]
        )
        if not extraction:
            continue
        probe.reset(slide, initial)
        blockers = set()
        for joints in approach:
            probe.data.qpos[probe.arm_qadrs] = joints
            probe.data.qvel[probe.arm_dadrs] = 0.0
            mujoco.mj_forward(probe.model, probe.data)
            blockers |= probe.arm_environment_contacts()
        if blockers:
            continue
        selected = (initial, approach, extraction)
        break
    if selected is None:
        return None
    initial, approach, extraction = selected

    initial_object = probe.reset(slide, initial)
    approach_hits, approach_shelf = probe.command_path(approach)
    preclose_object = np.asarray(probe.data.body(probe.object_body).xpos).copy()
    probe.data.ctrl[probe.gripper_act] = 0.0
    close_hits, close_shelf = probe.step_for(4.5)
    close_end_hits, _ = probe.contacts()
    closed_gripper = float(probe.data.qpos[probe.finger_qadrs[1]] * 25.0)
    extraction_hits, extract_shelf = probe.command_path(extraction)
    final_hits, _ = probe.contacts()
    final_object = np.asarray(probe.data.body(probe.object_body).xpos).copy()
    return {
        "dual_close": len(close_end_hits) == 2,
        "dual_final": len(final_hits) == 2,
        "ever_dual": len(approach_hits | close_hits | extraction_hits) == 2,
        "shelf": approach_shelf or close_shelf or extract_shelf,
        "blockers": sorted(probe.blockers_seen),
        "pre_shift": float(np.linalg.norm(preclose_object[:2] - initial_object[:2])),
        "extract_xy": float(np.linalg.norm(final_object[:2] - initial_object[:2])),
        "max_dual_shift": probe.max_dual_shift,
        "closed_gripper": closed_gripper,
        "final": final_object,
    }


def values(name: str, default: str) -> tuple[float, ...]:
    return tuple(float(value) for value in os.getenv(name, default).split(","))


def main() -> None:
    probe = ContactProbe()
    results = []
    grid = itertools.product(
        values("PROBE_SLIDES", "0.15"),
        values("PROBE_TIPS", "0.075,0.085,0.095,0.105,0.115"),
        values("PROBE_DEPTHS", "-0.008,-0.004,0.0,0.004,0.008"),
        values("PROBE_CANTS", "-10,-6,-3,0,3,6,10"),
    )
    for slide, tip, depth, cant in grid:
        result = run_candidate(
            probe, slide=slide, tip=tip, depth=depth, cant=cant
        )
        if result is None:
            continue
        score = (
            1000.0 * float(result["max_dual_shift"])
            + 80.0 * float(result["dual_close"])
            - 100.0 * float(result["shelf"])
            - 100.0 * float(bool(result["blockers"]))
            - 1000.0 * max(0.0, float(result["pre_shift"]) - 0.045)
        )
        results.append((score, slide, tip, depth, cant, result))
        if float(result["max_dual_shift"]) >= 0.20:
            print(
                f"S3_VALID slide={slide:.2f} tip={tip:.3f} depth={depth:+.3f} "
                f"cant={cant:+.1f} result={result}",
                flush=True,
            )
    print(f"simulated={len(results)}")
    for score, slide, tip, depth, cant, result in sorted(
        results, reverse=True, key=lambda item: item[0]
    )[:30]:
        print(
            f"score={score:.1f} slide={slide:.2f} tip={tip:.3f} "
            f"depth={depth:+.3f} cant={cant:+.1f} result={result}"
        )


if __name__ == "__main__":
    main()
