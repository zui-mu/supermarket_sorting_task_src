"""Scan the MMK2 right-arm reachable workspace (footprint frame) for the carry
tuck and the place pose. Run inside the client container."""
import sys
import numpy as np

sys.path.insert(0, "/workspace/baseline/examples/supermarket_sorting")
from mmk2_kdl import MMK2Kdl

kdl = MMK2Kdl()


def ik_right(fp, rot, slide, ref6):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = fp
    ref = np.zeros(7)
    ref[0] = float(slide)
    ref[1:] = ref6
    try:
        sols = kdl.inverse_kinematics(T_left=None, T_right=T, ref_pos=ref,
                                      target_height=float(slide))
    except Exception as exc:
        return None
    return sols[0] if sols else None


rot = np.eye(3)
ref6 = np.zeros(6)

print("== reachable z at fwd/lateral grid, slide=0.408 (carry) ==")
for fwd in (0.20, 0.28, 0.36, 0.44, 0.52):
    row = []
    for lat in (-0.12, -0.06, 0.0, 0.06, 0.12):
        reach = []
        for z in np.arange(0.30, 1.15, 0.05):
            if ik_right([fwd, lat, z], rot, 0.408, ref6) is not None:
                reach.append(round(z, 2))
        if reach:
            row.append(f"lat={lat:+5.2f}: z[{min(reach)},{max(reach)}]")
        else:
            row.append(f"lat={lat:+5.2f}: none")
    print(f"fwd={fwd:.2f}  " + "  ".join(row))

print("== reachable at place height z=0.84, slide sweep ==")
for slide in (0.05, 0.10, 0.15, 0.20, 0.30, 0.408):
    found = []
    for fwd in np.arange(0.25, 0.60, 0.02):
        for lat in (-0.10, 0.0, 0.10):
            if ik_right([round(fwd, 2), lat, 0.84], rot, slide, ref6) is not None:
                found.append((round(fwd, 2), lat))
    print(f"slide={slide}: {sorted(set(found))[:12]}{' ...' if len(found) > 12 else ''}")

print("== reachable at tuck-ish z, slide=0.408 ==")
for z in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
    found = []
    for fwd in np.arange(0.15, 0.55, 0.02):
        for lat in (-0.10, -0.06, 0.0, 0.06, 0.10):
            if ik_right([round(fwd, 2), lat, z], rot, 0.408, ref6) is not None:
                found.append(round(fwd, 2))
    print(f"z={z}: fwd reachable {sorted(set(found))}")
