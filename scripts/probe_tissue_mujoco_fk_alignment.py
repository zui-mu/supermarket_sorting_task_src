#!/usr/bin/env python3
"""Rank KDL IK candidates by their actual MuJoCo finger-centre pose."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "supermarket_sorting"))

from mmk2_kdl import MMK2Kdl  # noqa: E402

ROTATION = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
)


def pose(position):
    result = np.eye(4)
    result[:3, :3] = ROTATION
    result[:3, 3] = position
    return result


def main():
    xml_path = ROOT / "examples" / "supermarket_sorting" / "mjcf" / "retail_competition.xml"
    xml = xml_path.read_text(encoding="utf-8").replace(
        "__REPO_ROOT__", str(ROOT / "examples" / "supermarket_sorting")
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    kdl = MMK2Kdl()

    def qadr(name):
        return int(model.joint(name).qposadr[0])

    slide_name = "slide_joint" if mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "slide_joint"
    ) >= 0 else "slide"
    arm_names = [f"rgt_arm_joint{i}" for i in range(1, 7)]
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "agv_link")
    link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rgt_arm_link6")
    left_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "rgt_finger_left_link"
    )
    right_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "rgt_finger_right_link"
    )
    rng = np.random.default_rng(19)
    candidates = []
    exact_joints = os.environ.get("PROBE_JOINTS")
    if exact_joints:
        joints = np.fromstring(exact_joints, sep=",")
        if joints.shape != (6,):
            raise ValueError("PROBE_JOINTS must contain six comma-separated radians")
        slide = float(os.environ.get("PROBE_SLIDE", "0.15"))
        data.qpos[qadr(slide_name)] = slide
        for name, value in zip(arm_names, joints):
            data.qpos[qadr(name)] = value
        data.qpos[qadr("rgt_finger_left_joint")] = -0.04
        data.qpos[qadr("rgt_finger_right_joint")] = 0.04
        mujoco.mj_forward(model, data)
        base_p = np.asarray(data.site("base_link").xpos)
        base_R = np.asarray(data.site("base_link").xmat).reshape(3, 3)

        def exact_fp(body_id):
            return base_R.T @ (np.asarray(data.body(body_id).xpos) - base_p)

        print(
            "exact joints: link=%s fingers=%s q=%s"
            % (
                np.round(exact_fp(link_id), 4),
                np.round(0.5 * (exact_fp(left_id) + exact_fp(right_id)), 4),
                np.round(joints, 4),
            )
        )
        return
    specific = os.environ.get("PROBE_TARGET_X") is not None
    desired = np.array([
        float(os.environ.get("PROBE_DESIRED_X", "0.55")),
        float(os.environ.get("PROBE_DESIRED_Y", "0.115")),
        float(os.environ.get("PROBE_DESIRED_Z", "0.895")),
    ])
    slides = (float(os.environ.get("PROBE_SLIDE", "0.15")),) if specific else (
        0.10, 0.15, 0.20, 0.25
    )
    target_xs = (float(os.environ["PROBE_TARGET_X"]),) if specific else np.arange(
        0.54, 0.681, 0.01
    )
    target_ys = (float(os.environ.get("PROBE_TARGET_Y", "0.12")),) if specific else (
        0.09, 0.12, 0.15
    )
    random_refs = int(os.environ.get("PROBE_RANDOM_REFS", "96" if specific else "8"))
    for slide in slides:
        for target_x in target_xs:
            for target_y in target_ys:
                target = np.array([target_x, target_y, 0.890])
                references = [np.array([0.0, -0.166, 0.032, 0.0, -1.571, -2.223])]
                references.extend(rng.uniform(-2.6, 2.6, (random_refs, 6)))
                unique = []
                for reference in references:
                    solutions = kdl.inverse_kinematics(
                        T_right=pose(target),
                        ref_pos=np.concatenate(([slide], reference)),
                        target_height=slide,
                    )
                    for solution in solutions or ():
                        joints = np.asarray(solution)[1:7]
                        if not any(np.max(np.abs(joints - known)) < 0.02 for known in unique):
                            unique.append(joints)
                for joints in unique:
                    data.qpos[qadr(slide_name)] = slide
                    for name, value in zip(arm_names, joints):
                        data.qpos[qadr(name)] = value
                    data.qpos[qadr("rgt_finger_left_joint")] = -0.04
                    data.qpos[qadr("rgt_finger_right_joint")] = 0.04
                    mujoco.mj_forward(model, data)
                    base_p = np.asarray(data.body(base_id).xpos)
                    base_R = np.asarray(data.body(base_id).xmat).reshape(3, 3)
                    def fp(body_id):
                        return base_R.T @ (np.asarray(data.body(body_id).xpos) - base_p)
                    link = fp(link_id)
                    finger_mid = 0.5 * (fp(left_id) + fp(right_id))
                    score = float(np.linalg.norm(finger_mid - desired))
                    candidates.append(
                        (score, slide, target_x, target_y, link, finger_mid, joints)
                    )
    limit = len(candidates) if specific else 30
    for score, slide, target_x, target_y, link, midpoint, joints in sorted(
        candidates, key=lambda item: item[0]
    )[:limit]:
        print(
            f"score={score:.4f} slide={slide:.2f} target=({target_x:.3f},{target_y:.3f}) "
            f"link={np.round(link, 3)} fingers={np.round(midpoint, 3)} "
            f"q={np.round(joints, 3)}"
        )


if __name__ == "__main__":
    main()
