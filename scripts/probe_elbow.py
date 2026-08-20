"""Scan wrist fwd positions at the place height and report the link3 fwd for
each, to find the wrist placement whose elbow stays north of the table edge."""
import sys
import numpy as np

sys.path.insert(0, "/workspace/baseline/examples/supermarket_sorting")
from mmk2_kdl import MMK2Kdl
from arm_kdl import ArmKdl

kdl = MMK2Kdl()
arm = ArmKdl(eef_type="G2")


def arm_ik_all(arm_pose):
    return arm.inverse_kinematics(arm_pose, ref_pos=None)


def frame3_pos(joints):
    res = np.eye(4)
    for i in range(3):
        res = res @ arm.dh.adjacent_transform(joints[i], i + 1)
    return res[0:3, 3]


for slide in (0.05, 0.10):
    print(f"== slide={slide}, wrist z=0.84 ==")
    for fwd in np.arange(0.44, 0.58, 0.01):
        T = np.eye(4)
        T[:3, 3] = [round(fwd, 2), 0.0, 0.84]
        T_spine = kdl.spine.get_transformation_matrix(slide)
        T_spine2arm = kdl.spine2arm.get_transformation_matrix("right")
        T_arm = np.linalg.inv(T_spine @ T_spine2arm) @ T
        sols = arm_ik_all(T_arm)
        if not sols:
            continue
        p3 = frame3_pos(sols[0])
        full = kdl.spine.get_transformation_matrix(slide) @ T_spine2arm
        p3_base = (full @ np.append(p3, 1.0))[0:3]
        print(f"  wrist fwd={fwd:.2f} -> link3 fwd={p3_base[0]:.3f} "
              f"lat={p3_base[1]:.3f} z={p3_base[2]:.3f}")
