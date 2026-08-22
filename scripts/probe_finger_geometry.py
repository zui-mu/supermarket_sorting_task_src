#!/usr/bin/env python3
"""Measure the MMK2 right gripper finger geometry at GRIP_OPEN/CLOSE."""
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "examples", "supermarket_sorting"))

from mmk2_kdl import MMK2Kdl

XML = os.path.join(
    os.path.dirname(__file__), "..",
    "examples", "supermarket_sorting", "mjcf", "retail_competition.xml")

def main():
    xml = open(XML, encoding="utf-8").read().replace(
        "__REPO_ROOT__",
        os.path.join(os.path.dirname(__file__), "..", "examples", "supermarket_sorting"))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Geoms belonging to either gripper's finger links.
    finger_geoms = []
    for g in range(model.ngeom):
        bname = model.body(model.geom_bodyid[g]).name
        gname = model.geom(g).name
        if "finger" in bname or "finger" in gname:
            finger_geoms.append((bname, gname, g))
    print("finger geoms:", [(b, g, i) for b, g, i in finger_geoms])
    # Joint qpos addresses.
    def qidx(jn):
        for j in range(model.njnt):
            if model.joint(j).name == jn:
                return model.joint(j).qposadr
        return None
    for grip in (1.0, 0.5, 0.0):
        li = qidx("rgt_finger_left_joint")
        ri = qidx("rgt_finger_right_joint")
        if li is not None:
            data.qpos[li] = -0.04 * grip
        if ri is not None:
            data.qpos[ri] = 0.04 * grip
        mujoco.mj_forward(model, data)
        row = ["grip=%.1f" % grip]
        for bname, gname, g in finger_geoms:
            p = data.geom_xpos[g]
            bid = model.geom_bodyid[g]
            body_p = data.body(bid).xpos
            body_R = data.body(bid).xmat.reshape(3, 3)
            local = body_R.T @ (p - body_p)
            link_name = "lft_arm_link6" if bname.startswith("lft_") else "rgt_arm_link6"
            link_id = next(
                b for b in range(model.nbody) if model.body(b).name == link_name
            )
            link_p = data.body(link_id).xpos
            link_R = data.body(link_id).xmat.reshape(3, 3)
            endpoint_local = link_R.T @ (p - link_p)
            row.append(
                "%s=[%.3f %.3f %.3f] local=[%.3f %.3f %.3f] endpoint=[%.3f %.3f %.3f]"
                % (gname or bname, p[0], p[1], p[2], *local, *endpoint_local)
            )
        print(" ".join(row))
        # Also report the finger body origins.
        for bn in ("rgt_finger_left_link", "rgt_finger_right_link"):
            for b in range(model.nbody):
                if model.body(b).name == bn:
                    p = data.body(b).xpos
                    print("  body %s=[%.3f %.3f %.3f]" % (bn, p[0], p[1], p[2]))
                    break

    print("right endpoint-local collision bounds at open grip")
    li = qidx("rgt_finger_left_joint")
    ri = qidx("rgt_finger_right_joint")
    data.qpos[li] = -0.04
    data.qpos[ri] = 0.04
    mujoco.mj_forward(model, data)
    link_id = next(
        b for b in range(model.nbody) if model.body(b).name == "rgt_arm_link6"
    )
    link_p = np.asarray(data.body(link_id).xpos)
    link_R = np.asarray(data.body(link_id).xmat).reshape(3, 3)
    for g in range(model.ngeom):
        body_id = int(model.geom_bodyid[g])
        body_name = model.body(body_id).name
        if body_name not in {
            "rgt_arm_link6", "rgt_finger_left_link", "rgt_finger_right_link"
        }:
            continue
        geom = model.geom(g)
        centre = link_R.T @ (np.asarray(data.geom_xpos[g]) - link_p)
        geom_type = int(model.geom_type[g])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[g])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(model.mesh_vert[start:start + count])
            geom_R = np.asarray(data.geom_xmat[g]).reshape(3, 3)
            world = np.asarray(data.geom_xpos[g]) + vertices @ geom_R.T
            endpoint_vertices = (world - link_p) @ link_R
            bounds = (
                np.min(endpoint_vertices, axis=0),
                np.max(endpoint_vertices, axis=0),
            )
            print(
                f"  {body_name}/{geom.name or g} mesh centre={np.round(centre, 4)} "
                f"min={np.round(bounds[0], 4)} max={np.round(bounds[1], 4)}"
            )
        else:
            print(
                f"  {body_name}/{geom.name or g} type={geom_type} "
                f"centre={np.round(centre, 4)} size={np.round(model.geom_size[g], 4)}"
            )

    # Calibrate the rotation expected by MMK2Kdl against the MuJoCo link6
    # collision-body frame.  They differ by the fixed endpoint-site rotation.
    arm_q = []
    for joint_name in [f"rgt_arm_joint{i}" for i in range(1, 7)]:
        address = qidx(joint_name)
        arm_q.append(float(data.qpos[address]))
    slide_address = qidx("slide_joint")
    if slide_address is None:
        slide_address = qidx("slide")
    slide_q = float(data.qpos[slide_address]) if slide_address is not None else 0.0
    _, endpoint = MMK2Kdl().forward_kinematics(
        np.asarray([slide_q] + arm_q), index="right"
    )
    base_id = next(
        b for b in range(model.nbody) if model.body(b).name == "agv_link"
    )
    base_R = np.asarray(data.body(base_id).xmat).reshape(3, 3)
    base_p = np.asarray(data.body(base_id).xpos)
    link_R_world = np.asarray(data.body(link_id).xmat).reshape(3, 3)
    link_R_fp = base_R.T @ link_R_world
    link_p_fp = base_R.T @ (np.asarray(data.body(link_id).xpos) - base_p)
    print("KDL endpoint R (footprint):\n", np.round(endpoint[:3, :3], 4))
    print("MuJoCo link6 R (footprint):\n", np.round(link_R_fp, 4))
    print(
        "fixed KDL->link6 rotation:\n",
        np.round(endpoint[:3, :3].T @ link_R_fp, 4),
    )
    delta_fp = link_p_fp - endpoint[:3, 3]
    print("KDL endpoint position:", np.round(endpoint[:3, 3], 4))
    print("MuJoCo link6 position:", np.round(link_p_fp, 4))
    print("endpoint->link6 delta footprint:", np.round(delta_fp, 4))
    print(
        "endpoint->link6 delta KDL-local:",
        np.round(endpoint[:3, :3].T @ delta_fp, 4),
    )

if __name__ == "__main__":
    main()
