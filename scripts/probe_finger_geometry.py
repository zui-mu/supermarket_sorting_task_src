#!/usr/bin/env python3
"""Measure the MMK2 right gripper finger geometry at GRIP_OPEN/CLOSE."""
import os
import sys

import mujoco

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
    # Geoms belonging to the right finger links (collision meshes "right"/"left").
    finger_geoms = []
    for g in range(model.ngeom):
        bname = model.body(model.geom_bodyid[g]).name
        gname = model.geom(g).name
        if "rgt_finger" in bname or "finger" in gname:
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
            row.append("%s=[%.3f %.3f %.3f]" % (gname or bname, p[0], p[1], p[2]))
        print(" ".join(row))
        # Also report the finger body origins.
        for bn in ("rgt_finger_left_link", "rgt_finger_right_link"):
            for b in range(model.nbody):
                if model.body(b).name == bn:
                    p = data.body(b).xpos
                    print("  body %s=[%.3f %.3f %.3f]" % (bn, p[0], p[1], p[2]))
                    break

if __name__ == "__main__":
    main()
