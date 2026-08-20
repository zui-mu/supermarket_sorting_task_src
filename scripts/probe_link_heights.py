#!/usr/bin/env python3
"""Probe per-link heights from the DH table for the raise target joints."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "supermarket_sorting"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmk2_kdl import MMK2Kdl
import arm_kdl

RARM = np.array([-1.03, -1.549, 1.062, 0.93, 1.562, -2.284])
ROT = np.eye(3)

def fk_ee(kdl, slide, arm_joints):
    q = np.concatenate([[slide], arm_joints])
    _, T = kdl.forward_kinematics(q, index="right")
    return T[:3, 3]

def ik_wrist(kdl, slide, fp_xy, z):
    T = np.eye(4)
    T[:3, :3] = ROT
    T[:3, 3] = [fp_xy[0], fp_xy[1], z]
    ref = np.zeros(7)
    ref[0] = slide
    ref[1:] = RARM
    sols = kdl.inverse_kinematics(
        T_left=None, T_right=T, ref_pos=ref, target_height=slide
    )
    if not sols:
        return None
    joints = np.asarray(sols[0], dtype=float)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        return None
    return joints

def dh_links(dh_obj, joints):
    """Compute per-link origin z in the arm base frame from the DH table."""
    a = np.asarray(dh_obj.a) if hasattr(dh_obj, "a") else None
    alpha = np.asarray(dh_obj.alpha) if hasattr(dh_obj, "alpha") else None
    d = np.asarray(dh_obj.d) if hasattr(dh_obj, "d") else None
    theta = np.asarray(dh_obj.theta) if hasattr(dh_obj, "theta") else None
    if a is None or alpha is None or d is None or theta is None:
        print("ArmDH fields:", [m for m in dir(dh_obj) if not m.startswith("_")])
        for attr in ("a", "alpha", "d", "theta"):
            try:
                v = getattr(dh_obj, attr)
                print(f"  {attr} = {v}")
            except Exception as exc:
                print(f"  {attr} err: {exc}")
        return None
    T = np.eye(4)
    origins = []
    n = min(len(joints), len(a))
    for i in range(n):
        ai = float(a[i])
        al = float(alpha[i])
        di = float(d[i]) + (float(theta[i]) if theta is not None and i < len(theta) else 0.0)
        qi = float(joints[i])
        c, s = np.cos(qi), np.sin(qi)
        ca, sa = np.cos(al), np.sin(al)
        Ti = np.array([
            [c, -s * ca, s * sa, ai * c],
            [s, c * ca, -c * sa, ai * s],
            [0, sa, ca, di],
            [0, 0, 0, 1],
        ])
        T = T @ Ti
        origins.append(T[:3, 3].copy())
    return origins

def main():
    kdl = MMK2Kdl()
    chain = arm_kdl.ArmKdl()
    ee = fk_ee(kdl, 0.402, RARM)
    fwd, lat = ee[0], ee[1]
    print(f"frozen EE fp=({fwd:.3f},{lat:.3f})")
    dh = getattr(chain, "dh", None)
    if dh is None:
        print("no dh attribute")
        return
    # Frozen pose links.
    links = dh_links(dh, RARM)
    if links is not None:
        print("frozen link origins x/z:", [(f"{l[0]:.3f}", f"{l[2]:.3f}") for l in links])
    for z in (0.854, 0.88, 0.90, 0.92):
        sol = ik_wrist(kdl, 0.402, (fwd, lat), z)
        if sol is None:
            print(f"raise z={z}: IK FAIL")
            continue
        links = dh_links(dh, sol[1:])
        if links is not None:
            print(f"raise z={z}: link x/z:", [(f"{l[0]:.3f}", f"{l[2]:.3f}") for l in links])

if __name__ == "__main__":
    main()
