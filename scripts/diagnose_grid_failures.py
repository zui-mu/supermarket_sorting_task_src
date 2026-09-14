"""Probe why the four failing navigation tests cannot find a route.

Run inside tissv_client:
    python3 scripts/diagnose_grid_failures.py
"""
from __future__ import annotations

import math
import sys

sys.path.insert(0, "/workspace/baseline")
sys.path.insert(0, "/workspace/baseline/examples/supermarket_sorting")

import numpy as np  # noqa: E402

from examples.supermarket_sorting.navigation.grid_planner import (  # noqa: E402
    SupermarketGridPlanner as P,
)


def describe(planner, label):
    print(f"--- {label}")
    print(f"    bounds x[{planner.xmin},{planner.xmax}] y[{planner.ymin},{planner.ymax}]"
          f" res={planner.resolution} robot_r={planner.robot_radius}"
          f" corridor={planner.corridor_clearance}")
    for rect in planner.static_rects:
        print(f"    inflated rect x[{rect.xmin:+.2f},{rect.xmax:+.2f}] "
              f"y[{rect.ymin:+.2f},{rect.ymax:+.2f}]")


def probe(label, planner, start, goal, dynamic=()):
    route = planner.plan(start, goal, dynamic_points=dynamic)
    print(f"  {label}: start={start} goal={goal} dyn={list(dynamic)}")
    print(f"    goal_blocked={planner._blocked(planner._to_cell(goal), planner._dynamic_cells(dynamic))}"
          f" start_blocked={planner._blocked(planner._to_cell(start), planner._dynamic_cells(dynamic))}")
    print(f"    route={'OK ' + str(len(route)) + ' wp' if route else 'EMPTY'}")
    if route:
        print(f"    first={route[0]} last={route[-1]}")
    return route


def free_columns(planner, y):
    """Report the x-bands that are free along the horizontal line y."""
    bands = []
    run_start = None
    x = planner.xmin
    while x <= planner.xmax + 1e-9:
        cell = planner._to_cell(np.array([x, y]))
        blocked = planner._blocked(cell, set())
        if not blocked and run_start is None:
            run_start = x
        elif blocked and run_start is not None:
            bands.append((run_start, x - planner.resolution))
            run_start = None
        x += planner.resolution
    if run_start is not None:
        bands.append((run_start, planner.xmax))
    return bands


def free_rows(planner, x):
    bands = []
    run_start = None
    y = planner.ymin
    while y <= planner.ymax + 1e-9:
        cell = planner._to_cell(np.array([x, y]))
        blocked = planner._blocked(cell, set())
        if not blocked and run_start is None:
            run_start = y
        elif blocked and run_start is not None:
            bands.append((run_start, y - planner.resolution))
            run_start = None
        y += planner.resolution
    if run_start is not None:
        bands.append((run_start, planner.ymax))
    return bands


print("=" * 78)
print("CASE 2/3: is the arena still connected across the divider?")
for clearance in (0.22, 0.45, 0.65, 0.88):
    pl = P(corridor_clearance=clearance)
    east_band = free_columns(pl, 0.0)
    print(f"  corridor_clearance={clearance}: free x-bands at y=0 -> "
          + ", ".join(f"[{a:+.2f},{b:+.2f}]" for a, b in east_band))
    north_gap = free_rows(pl, 1.75)
    print(f"      free y-bands at x=1.75 (through the divider column) -> "
          + ", ".join(f"[{a:+.2f},{b:+.2f}]" for a, b in north_gap))
    south_gap = free_rows(pl, 1.75)
    print(f"      free y-bands at x=0.00 -> "
          + ", ".join(f"[{a:+.2f},{b:+.2f}]" for a, b in free_rows(pl, 0.00)))

print()
print("=" * 78)
print("CASE 2: test_pruned_route_does_not_cross_static_obstacles")
pl = P()
describe(pl, "default planner")
probe("pruned", pl, (1.60, 2.05), (-1.88, -2.74))

print()
print("=" * 78)
print("CASE 3: test_shelf_recovery_crosses_divider_in_high_band")
pl = P(corridor_clearance=0.65)
describe(pl, "recovery planner")
probe("recovery", pl, (2.03, 1.88), (-1.93, 2.34))

print()
print("=" * 78)
print("CASE 4: test_wall_side_passage_stays_open_with_wall_hits")
pl = P(corridor_clearance=0.88, dynamic_clearance=0.50)
dynamic = [
    (-1.51, -1.05), (-1.20, -1.05), (-0.88, -1.05),
    (-2.50, -1.00), (-2.50, 0.00), (-2.50, -2.00),
]
describe(pl, "wall-side planner")
probe("wall-side", pl, (-0.50, 2.30), (-1.88, -2.74), dynamic)
print(f"    free x-bands at y=-1.05 -> "
      + ", ".join(f"[{a:+.2f},{b:+.2f}]" for a, b in free_columns(pl, -1.05)))

print()
print("=" * 78)
print("CASE 5: test_dynamic_obstacle_forces_a_detour")
pl = P()
probe("detour", pl, (1.20, -2.80), (1.20, -1.60), [(1.20, -2.20)])
print(f"    free x-bands at y=-2.20 -> "
      + ", ".join(f"[{a:+.2f},{b:+.2f}]" for a, b in free_columns(pl, -2.20)))
