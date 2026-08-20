#!/usr/bin/env python3
"""Probe the delivery/placement raise geometry with the real arm_kdl.

Answers: at the frozen loaded arm pose (rarm=[-1.03,-1.549,1.062,0.93,1.562,-2.284]),
what is the EE z at slide 0.402 vs slide 0.10 (FK), and is the wrist raise to
z=0.854 at the same footprint x,y reachable (IK) at each slide?
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "supermarket_sorting"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmk2_kdl import MMK2Kdl

RARM = np.array([-1.03, -1.549, 1.062, 0.93, 1.562, -2.284])
ROT = np.eye(3)  # grasp rotation placeholder (world-frame independent for z)

def fk_ee(kdl, slide):
    q = np.concatenate([[slide], RARM])
    _, T = kdl.forward_kinematics(q, index="right")
    return T[:3, 3]

def ik_wrist(kdl, slide, fp_xy, z):
    """Solve the wrist at footprint (fp_xy, z) with the identity rotation."""
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

def main():
    kdl = MMK2Kdl()
    for slide in (0.402, 0.10, 0.30):
        ee = fk_ee(kdl, slide)
        print(f"slide={slide:.3f} FK ee_z={ee[2]:.3f} ee_xy=({ee[0]:.3f},{ee[1]:.3f})")
    # At the frozen pose, what is the EE footprint x,y (fwd)?
    ee = fk_ee(kdl, 0.402)
    fwd, lat = ee[0], ee[1]
    print(f"frozen pose at slide 0.402: ee_fp=({fwd:.3f},{lat:.3f})")
    # Try the raise at various slides and footprint x values.
    for slide in (0.402, 0.10, 0.05):
        for fwd_try in (0.45, 0.50, 0.577, 0.62):
            sol = ik_wrist(kdl, slide, (fwd_try, lat), 0.854)
            if sol is not None:
                delta = np.max(np.abs(sol[1:] - RARM))
                print(f"  slide={slide:.3f} fwd={fwd_try:.3f} z=0.854 -> IK OK joint_delta={delta:.3f}")
            else:
                print(f"  slide={slide:.3f} fwd={fwd_try:.3f} z=0.854 -> IK FAIL")
    # Lower raise targets too.
    for slide in (0.402, 0.10):
        for z in (0.834, 0.80, 0.77, 0.74):
            sol = ik_wrist(kdl, slide, (fwd, lat), z)
            if sol is not None:
                print(f"  slide={slide:.3f} fwd={fwd:.3f} z={z:.3f} -> IK OK delta={np.max(np.abs(sol[1:]-RARM)):.3f}")
            else:
                print(f"  slide={slide:.3f} fwd={fwd:.3f} z={z:.3f} -> IK FAIL")
    # Higher raise targets: link4 hangs at ~0.786 when the wrist is at 0.854,
    # which grazes the table top (C1 in count5_v7).  Test whether a higher
    # wrist lifts link4 above the edge.
    for slide in (0.402, 0.10, 0.05):
        for z in (0.88, 0.90, 0.92, 0.95):
            sol = ik_wrist(kdl, slide, (fwd, lat), z)
            if sol is not None:
                print(f"  slide={slide:.3f} fwd={fwd:.3f} z={z:.3f} -> IK OK delta={np.max(np.abs(sol[1:]-RARM)):.3f}")
            else:
                print(f"  slide={slide:.3f} fwd={fwd:.3f} z={z:.3f} -> IK FAIL")

if __name__ == "__main__":
    main()
