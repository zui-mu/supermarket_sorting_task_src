from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .models import PickTask, TaskStatus


@dataclass(slots=True)
class SchedulerConfig:
    same_shelf_bonus: float = 8.0
    same_level_bonus: float = 3.0
    right_side_bonus: float = 2.5
    middle_level_first_bonus: float = 4.0
    bottle_bonus: float = -6.0
    box_bonus: float = 9.0
    fruit_bonus: float = -12.0
    retry_penalty: float = 6.0
    high_level_penalty: float = 14.0
    low_level_bonus: float = -4.0
    middle_level_bonus: float = 8.0
    path_cost_penalty: float = 5.0
    start_xy: tuple[float, float] = (1.92, -3.17)
    delivery_xy: tuple[float, float] = (-1.88, -2.80)


class OrderScheduler:
    """Rank candidate pick tasks for the next execution cycle.

    This first version is rule-based on purpose: it is simple, explainable,
    and easy to replace later with a learned policy.
    """

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig()

    def rank_tasks(
        self,
        tasks: Iterable[PickTask],
        *,
        last_completed_task: PickTask | None = None,
    ) -> list[PickTask]:
        ranked = sorted(tasks, key=lambda task: self.score_task(task, last_completed_task), reverse=True)
        for task in ranked:
            task.priority = self.score_task(task, last_completed_task)
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

    def score_task(self, task: PickTask, last_completed_task: PickTask | None) -> float:
        score = 100.0
        score -= task.retry_count * self.config.retry_penalty
        score -= self._path_cost(task, last_completed_task) * self.config.path_cost_penalty

        if task.level == "L1":
            score += self.config.low_level_bonus
        elif task.level == "L2":
            score += self.config.middle_level_bonus
        elif task.level == "L3":
            score -= self.config.high_level_penalty

        if task.navigation_target.x > 0.4:
            score += self.config.right_side_bonus
        elif task.navigation_target.x < -0.4:
            score -= 0.5

        if task.product_name in {"kele", "maidong"}:
            score += self.config.bottle_bonus
        elif task.product_name in {"sanmingzhi", "heweidao", "shupian", "zhijin", "kouxiangtang"}:
            score += self.config.box_bonus
        elif task.product_name in {"pingguo", "chengzi"}:
            score += self.config.fruit_bonus

        if task.metadata.get("official_direct"):
            score += 1.5
        if task.metadata.get("search_mode"):
            score -= 1.5

        # Prefer the objects whose front-centre grasps are most stable in the
        # V2 scene. Thin bottles are delayed because their first off-centre
        # contact can trigger an immediate topple penalty before S3.
        product_order_bonus = {
            "zhijin": 9.0,
            "sanmingzhi": 8.0,
            "heweidao": 7.0,
            "shupian": 5.0,
            "kouxiangtang": 2.0,
            "maidong": -5.0,
            "kele": -7.0,
            "pingguo": -6.0,
            "chengzi": -7.0,
        }
        score += product_order_bonus.get(task.product_name, 0.0)

        if task.level == "L3" and task.product_name in {"pingguo", "chengzi"}:
            score -= 8.0
        if task.level == "L3" and task.product_name in {"shupian", "zhijin"}:
            score -= 4.0

        if task.level == "L2":
            score += self.config.middle_level_first_bonus

        if last_completed_task is not None:
            if task.shelf == last_completed_task.shelf:
                score += self.config.same_shelf_bonus
            if task.level == last_completed_task.level:
                score += self.config.same_level_bonus

        return score

    def _path_cost(self, task: PickTask, last_completed_task: PickTask | None) -> float:
        if last_completed_task is None:
            origin = self.config.start_xy
        else:
            origin = self.config.delivery_xy
        shelf = (task.navigation_target.x, task.navigation_target.y)
        delivery = self.config.delivery_xy
        return self._dist(origin, shelf) + self._dist(shelf, delivery)

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
