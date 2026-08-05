"""Small deterministic A* planner for the official supermarket arena.

The simulation map is known and static, while lidar points are added as
short-lived obstacles for each replan.  This gives the baseline a real global
route around the corridor boxes without requiring a full Nav2 installation.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Rect:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def inflated(self, radius: float) -> "Rect":
        return Rect(
            self.xmin - radius,
            self.xmax + radius,
            self.ymin - radius,
            self.ymax + radius,
        )

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


class SupermarketGridPlanner:
    """Eight-connected A* with obstacle inflation and line-of-sight pruning."""

    # World positions are taken from retail_competition.xml.  Rectangles store
    # physical extents; robot clearance is applied separately.
    STATIC_OBSTACLES = (
        Rect(0.50, 0.56, -3.72, 1.70),       # corridor_right_board
        Rect(-0.638, -0.038, -2.069, -1.669),
        Rect(-0.210, 0.190, 0.530, 1.130),
        Rect(-2.310, -1.710, -1.314, -0.914),
        Rect(-2.322, -1.922, 1.115, 1.715),
        Rect(-1.370, -0.770, -0.397, 0.003),
        Rect(-2.420, -1.460, -3.630, -3.190),  # delivery table
    )

    def __init__(
        self,
        resolution: float = 0.10,
        robot_radius: float = 0.22,
        corridor_clearance: float = 0.45,
        bounds: tuple[float, float, float, float] = (-2.35, 2.35, -3.48, 3.02),
    ):
        self.resolution = float(resolution)
        self.robot_radius = float(robot_radius)
        self.corridor_clearance = float(corridor_clearance)
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.width = int(round((self.xmax - self.xmin) / self.resolution)) + 1
        self.height = int(round((self.ymax - self.ymin) / self.resolution)) + 1
        self.static_rects = (
            self.STATIC_OBSTACLES[0].inflated(self.corridor_clearance),
            *(rect.inflated(self.robot_radius) for rect in self.STATIC_OBSTACLES[1:]),
        )

    def plan(
        self,
        start: Iterable[float],
        goal: Iterable[float],
        dynamic_points: Iterable[Iterable[float]] = (),
    ) -> list[list[float]]:
        start_xy = np.asarray(tuple(start), dtype=float)[:2]
        goal_xy = np.asarray(tuple(goal), dtype=float)[:2]
        start_cell = self._to_cell(start_xy)
        goal_cell = self._to_cell(goal_xy)
        dynamic = self._dynamic_cells(dynamic_points)

        frontier: list[tuple[float, float, tuple[int, int]]] = []
        heapq.heappush(frontier, (self._heuristic(start_cell, goal_cell), 0.0, start_cell))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
        cost_so_far = {start_cell: 0.0}
        moves = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        )

        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current == goal_cell:
                break
            if current_cost > cost_so_far.get(current, float("inf")) + 1e-9:
                continue
            for dx, dy, step_cost in moves:
                nxt = (current[0] + dx, current[1] + dy)
                if not self._in_grid(nxt):
                    continue
                if nxt not in (start_cell, goal_cell) and self._blocked(nxt, dynamic):
                    continue
                if dx and dy:
                    # Prevent diagonal corner cutting between two obstacles.
                    if self._blocked((current[0] + dx, current[1]), dynamic):
                        continue
                    if self._blocked((current[0], current[1] + dy), dynamic):
                        continue
                new_cost = current_cost + step_cost
                if new_cost >= cost_so_far.get(nxt, float("inf")):
                    continue
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                priority = new_cost + self._heuristic(nxt, goal_cell)
                heapq.heappush(frontier, (priority, new_cost, nxt))

        if goal_cell not in came_from:
            return []

        cells = []
        current = goal_cell
        while current is not None:
            cells.append(current)
            current = came_from[current]
        cells.reverse()
        cells = self._prune_line_of_sight(cells, dynamic)
        points = [self._to_world(cell).tolist() for cell in cells[1:]]
        if not points or np.linalg.norm(np.asarray(points[-1]) - goal_xy) > self.resolution:
            points.append(goal_xy.tolist())
        else:
            points[-1] = goal_xy.tolist()
        return points

    def path_is_clear(
        self,
        start: Iterable[float],
        goal: Iterable[float],
        dynamic_points: Iterable[Iterable[float]] = (),
    ) -> bool:
        dynamic = self._dynamic_cells(dynamic_points)
        return self._line_clear(self._to_cell(start), self._to_cell(goal), dynamic)

    def _dynamic_cells(self, points: Iterable[Iterable[float]]) -> set[tuple[int, int]]:
        occupied: set[tuple[int, int]] = set()
        radius_cells = max(1, int(math.ceil(self.robot_radius / self.resolution)))
        for point in points:
            p = np.asarray(tuple(point), dtype=float)
            if p.size < 2 or not np.all(np.isfinite(p[:2])):
                continue
            cx, cy = self._to_cell(p[:2])
            for ix in range(cx - radius_cells, cx + radius_cells + 1):
                for iy in range(cy - radius_cells, cy + radius_cells + 1):
                    if (ix - cx) ** 2 + (iy - cy) ** 2 <= radius_cells ** 2:
                        occupied.add((ix, iy))
        return occupied

    def _blocked(self, cell: tuple[int, int], dynamic: set[tuple[int, int]]) -> bool:
        if cell in dynamic:
            return True
        x, y = self._to_world(cell)
        return any(rect.contains(float(x), float(y)) for rect in self.static_rects)

    def _prune_line_of_sight(
        self,
        cells: list[tuple[int, int]],
        dynamic: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if len(cells) <= 2:
            return cells
        result = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            candidate = len(cells) - 1
            while candidate > anchor + 1 and not self._line_clear(cells[anchor], cells[candidate], dynamic):
                candidate -= 1
            result.append(cells[candidate])
            anchor = candidate
        return result

    def _line_clear(
        self,
        a: tuple[int, int],
        b: tuple[int, int],
        dynamic: set[tuple[int, int]],
    ) -> bool:
        dx, dy = b[0] - a[0], b[1] - a[1]
        # Half-cell sampling is deliberately conservative. Sampling only once
        # per cell can round over a corner and prune a safe A* detour into an
        # unsafe diagonal that clips an inflated obstacle.
        steps = max(2 * max(abs(dx), abs(dy)), 1)
        for i in range(steps + 1):
            t = i / steps
            cell = (int(round(a[0] + dx * t)), int(round(a[1] + dy * t)))
            if cell not in (a, b) and self._blocked(cell, dynamic):
                return False
        return True

    def _to_cell(self, point: Iterable[float]) -> tuple[int, int]:
        x, y = tuple(point)[:2]
        ix = int(round((float(x) - self.xmin) / self.resolution))
        iy = int(round((float(y) - self.ymin) / self.resolution))
        return (
            min(max(ix, 0), self.width - 1),
            min(max(iy, 0), self.height - 1),
        )

    def _to_world(self, cell: tuple[int, int]) -> np.ndarray:
        return np.array([
            self.xmin + cell[0] * self.resolution,
            self.ymin + cell[1] * self.resolution,
        ])

    def _in_grid(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    @staticmethod
    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)
