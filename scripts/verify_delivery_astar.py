#!/usr/bin/env python3
"""Prove the delivery-A* deadlock is fixed, without a full simulation run.

Reproduces the exact failing configuration from the live log:
  start = shelf-front picking line, goal = delivery point,
  dynamic set = lidar hits INCLUDING shelf faces (y in [3.173, 3.473]).
Before the fix the loaded planner returned [] (log: "delivery A* empty:
dynamic=1548 ... static_only=OK").  After the fix it must return a route.
"""
import sys

import numpy as np

sys.path.insert(0, "/workspace/baseline/examples/supermarket_sorting")

from navigation.grid_planner import SupermarketGridPlanner  # noqa: E402

# Static rects are inflated by corridor/robot clearance for obstacles, so the
# same values the client uses.
from supermarket_sorting_client import (  # noqa: E402
    LOADED_CORRIDOR_CLEARANCE,
    LOADED_DYNAMIC_CLEARANCE,
)

print(f"loaded corridor clearance = {LOADED_CORRIDOR_CLEARANCE}")
print(f"loaded dynamic clearance  = {LOADED_DYNAMIC_CLEARANCE}")

planner = SupermarketGridPlanner(
    resolution=0.10,
    robot_radius=0.22,
    corridor_clearance=LOADED_CORRIDOR_CLEARANCE,
    dynamic_clearance=LOADED_DYNAMIC_CLEARANCE,
)

# --- synthesise a realistic lidar cloud -------------------------------------
# Dense shelf-face hits (the band that used to be clamped onto the boundary),
# plus the walls, plus a couple of randomised boxes in the arena centre.
points = []
for shelf_centre_x in (-1.735, -0.850, 0.035, 0.920, 1.805):
    for x in np.arange(shelf_centre_x - 0.43, shelf_centre_x + 0.43, 0.05):
        for y in np.arange(3.173, 3.474, 0.05):
            points.append((float(x), float(y)))
for x in np.arange(-2.45, 2.46, 0.06):          # north + south walls
    points.append((float(x), 3.72))
    points.append((float(x), -3.73))
for y in np.arange(-3.7, 3.7, 0.06):            # west + east walls
    points.append((-2.48, float(y)))
    points.append((2.48, float(y)))
for centre, half in (((-0.11, 0.813), 0.30), ((0.29, 1.32), 0.30)):  # boxes
    points.append(centre)
    for dx in np.arange(-half, half, 0.08):
        points.append((centre[0] + float(dx), centre[1] - half))
        points.append((centre[0] + float(dx), centre[1] + half))
print(f"synthetic lidar points: {len(points)}")

cases = [
    # (label, start, goal) - the real delivery geometry: leave the shelf line
    # in front of shelf A/B/C and drive to the delivery table.
    ("A-front to table", (-1.95, 2.42), (-1.82, -2.84)),
    ("B-front to table", (-1.07, 2.42), (-1.82, -2.84)),
    ("C-front to table", (0.03, 2.42), (-1.82, -2.84)),
    ("D-front to table", (0.92, 2.42), (-1.82, -2.84)),
    ("table back to A", (-1.82, -2.84), (-1.95, 2.42)),
]

failures = 0
for label, start, goal in cases:
    route = planner.plan(start, goal, points)
    ok = bool(route)
    if not ok:
        failures += 1
    print(f"  {label:22s}: {'OK  ' if ok else 'FAIL'} "
          f"waypoints={len(route) if ok else 0}"
          + (f" first={np.round(route[0], 2).tolist()}" if ok else ""))

print()
print("RESULT:", "PASS - delivery A* finds routes" if failures == 0
      else f"FAIL - {failures} of {len(cases)} legs still have no route")
sys.exit(0 if failures == 0 else 1)
