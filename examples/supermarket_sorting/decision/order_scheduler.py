from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import PickTask, TaskStatus


@dataclass(slots=True)
class SchedulerConfig:
    """Deterministic shelf scan order.

    The official layout uses one shelf row: A..E and C1..C3 are already
    left-to-right in world X.  We intentionally avoid product/category
    heuristics here so the robot does not keep changing its mind at the shelf.
    """

    level_order: tuple[str, ...] = ("L2", "L3", "L1")


class OrderScheduler:
    """Choose the next task by a simple shelf scan rule.

    Order:
    1. left to right by world/navigation X
    2. for the same horizontal position: middle, upper, lower
    3. stable text keys as deterministic tie-breakers
    """

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig()
        self._level_rank = {
            level: idx for idx, level in enumerate(self.config.level_order)
        }

    def rank_tasks(
        self,
        tasks: Iterable[PickTask],
        *,
        last_completed_task: PickTask | None = None,
    ) -> list[PickTask]:
        ranked = sorted(tasks, key=self.sort_key)
        for idx, task in enumerate(ranked):
            # Lower idx means earlier execution. Keep priority numeric so
            # existing logs/export_plan remain easy to inspect.
            task.priority = float(len(ranked) - idx)
        return ranked

    def choose_next(
        self,
        tasks: Iterable[PickTask],
        *,
        last_completed_task: PickTask | None = None,
    ) -> PickTask | None:
        pending = [task for task in tasks if task.status == TaskStatus.PENDING]
        if not pending:
            return None
        return self.rank_tasks(pending, last_completed_task=last_completed_task)[0]

    def sort_key(self, task: PickTask) -> tuple:
        x = float(task.navigation_target.x)
        level_rank = self._level_rank.get(str(task.level), len(self._level_rank))
        return (
            round(x, 4),
            level_rank,
            str(task.shelf),
            str(task.column),
            str(task.product_name),
            str(task.task_id),
        )
