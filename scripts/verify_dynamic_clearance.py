#!/usr/bin/env python3
"""Is the corridor still routable at the physically-required dynamic inflation?

Context (2026-08-23)
--------------------
The client's own comment derives the dynamic envelope for a LaserScan surface
hit on a 0.71 m box:

    required centre clearance 0.58 m (box half-diagonal + chassis half)
    maps to sqrt(R^2 + 0.36^2) >= 0.58  ->  R ~= 0.45

That value was later overridden to 0.26 ("match the static envelope"), and the
unloaded planner uses no override at all, i.e. robot_radius = 0.22.  The
shelf-side navigation then threads routes through the randomised box cluster:
a formal run stalled at (0.93, 1.35) re-planning the identical 3-waypoint route
[[0.85, 2.02], [0.35, 2.12], [-1.93, 2.34]] eight times without traversing it.

This script is the arbiter: with the REAL seed-11 box layout it sweeps the
dynamic clearance and reports whether
  (a) the start pocket can still reach a shelf picking line, and
  (b) the resulting route actually keeps the robot centre clear of every box
      BODY (not just its lidar-visible near face).
"""
from __future__ import annotations

import math
import os
import pathlib
import sys

sys.path.insert(0, "/workspace/baseline")
sys.path.insert(0, "/workspace/baseline/examples/supermarket_sorting")

import numpy as np  # noqa: E402

from navigation.grid_planner import SupermarketGridPlanner  # noqa: E402

# retail_competition.xml: the five obstacle boxes are 0.71 x 0.71 x 0.71 m
# (half-extent 0.355 ~= 0.36) and the server samples them as box CENTRES.
BOX_HALF = 0.36
OBSTACLE_ZONE_ORIGIN = np.array([-0.96, -1.01])
START = (1.92, -3.17)
SHELF_LINE_Y = 2.42
SHELF_XS = (-1.735, -0.850, 0.035, 0.920, 1.805)


def seed11_box_centres() -> list[tuple[float, float]]:
    """Reproduce randomize_obstacle_positions for the official seed."""
    from supermarket_sorting_server import randomize_obstacle_positions

    overrides = randomize_obstacle_positions(11 + 1000003)
    centres = []
    for body, (x, y, _z, _yaw) in overrides.items():
        centres.append((
            float(x + OBSTACLE_ZONE_ORIGIN[0]),
            float(y + OBSTACLE_ZONE_ORIGIN[1]),
        ))
    return centres


def box_surface_hits(centres, samples_per_edge=14):
    """Approximate what the 2-D LaserScan sees: the box outlines."""
    pts = []
    for cx, cy in centres:
        for i in range(samples_per_edge + 1):
            t = -BOX_HALF + 2.0 * BOX_HALF * i / samples_per_edge
            pts.append((cx + t, cy - BOX_HALF))
            pts.append((cx + t, cy + BOX_HALF))
            pts.append((cx - BOX_HALF, cy + t))
            pts.append((cx + BOX_HALF, cy + t))
    return pts


def route_clears_boxes(route, start, centres):
    """True if no segment of the route passes within BOX_HALF of a box centre."""
    previous = np.asarray(start, dtype=float)
    for point in route:
        current = np.asarray(point, dtype=float)
        for cx, cy in centres:
            centre = np.array([cx, cy])
            seg = current - previous
            length_sq = float(seg @ seg)
            if length_sq < 1e-12:
                t = 0.0
            else:
                t = float(np.clip((centre - previous) @ seg / length_sq, 0.0, 1.0))
            closest = previous + t * seg
            if float(np.linalg.norm(closest - centre)) < BOX_HALF:
                return False
        previous = current
    return True


def main() -> int:
    centres = seed11_box_centres()
    print(f"seed-11 obstacle boxes ({len(centres)}):")
    for cx, cy in centres:
        print(f"   centre=({cx:+.2f},{cy:+.2f}) "
              f"body x[{cx - BOX_HALF:+.2f},{cx + BOX_HALF:+.2f}] "
              f"y[{cy - BOX_HALF:+.2f},{cy + BOX_HALF:+.2f}]")

    hits = box_surface_hits(centres)
    print(f"\nsynthetic surface hits: {len(hits)}")

    print("\n clearance | route to shelf | route survives the box BODY")
    print(" ----------+----------------+------------------------------")
    failures = []
    for clearance in (0.22, 0.26, 0.32, 0.38, 0.45, 0.50):
        planner = SupermarketGridPlanner(dynamic_clearance=clearance)
        ok_all = True
        body_ok_all = True
        for shelf_x in SHELF_XS:
            goal = (shelf_x, SHELF_LINE_Y)
            route = planner.plan(START, goal, dynamic_points=hits)
            if not route:
                ok_all = False
                continue
            if not route_clears_boxes(route, START, centres):
                body_ok_all = False
        print(f"   {clearance:5.2f}    | {'yes' if ok_all else 'NO ':>14} | "
              f"{'yes' if body_ok_all else 'NO - threads through a box'}")
        if not ok_all:
            failures.append((clearance, "no route"))

    print()
    if failures:
        print("VERDICT: some clearances cannot route at all: "
              + ", ".join(f"{c}" for c, _ in failures))
    # The physically required value from the client's own derivation.
    print("physically required R from the client's own comment: ~0.45 m")
    print("(sqrt(R^2 + 0.36^2) >= 0.58 to graze a box side face safely)")
    print(f"0.36 + 0.22 = {BOX_HALF + 0.22:.2f} m is the face-normal requirement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
