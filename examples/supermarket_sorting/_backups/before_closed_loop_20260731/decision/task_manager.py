from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import DecisionResult, NavigationTarget, PickTask, TaskStatus
from .order_scheduler import OrderScheduler

DEFAULT_LAYOUT_PATH = Path(__file__).resolve().parent.parent / 'retail_competition_layout.json'
DEFAULT_DELIVERY_TARGET = NavigationTarget(frame_id='map', x=-1.88, y=-2.80, yaw=-1.5707963267948966)
APPROACH_BASE_X = 0.852
APPROACH_BASE_Y = 2.475
APPROACH_YAW = 1.3788101090755203
RIGHT_ARM_OBJECT_X_OFFSET = 0.072


class TaskManager:
    """Convert layout items or order requests into executable pick tasks."""

    def __init__(self, layout_path: str | Path = DEFAULT_LAYOUT_PATH, scheduler: OrderScheduler | None = None):
        self.layout_path = Path(layout_path)
        self.scheduler = scheduler or OrderScheduler()
        self.layout_items = self._load_layout(self.layout_path)
        self.tasks: list[PickTask] = []
        self.last_completed_task: PickTask | None = None
        self.cycle_id = 0

    def build_tasks_for_products(self, product_names: Iterable[str]) -> list[PickTask]:
        requested = list(product_names)
        tasks: list[PickTask] = []
        used_bodies: set[str] = set()
        for product_name in requested:
            matches = [item for item in self.layout_items if item['object_kind'] == product_name and item['body'] not in used_bodies]
            for match in matches:
                task = self._layout_item_to_task(match)
                tasks.append(task)
                used_bodies.add(match['body'])
        self.tasks = tasks
        return tasks

    def build_tasks_from_referee_targets(self, target_bodies: Iterable[str]) -> list[PickTask]:
        wanted = set(target_bodies)
        self.tasks = [self._layout_item_to_task(item) for item in self.layout_items if item['body'] in wanted]
        return self.tasks

    def next_decision(self) -> DecisionResult:
        self.cycle_id += 1
        selected = self.scheduler.choose_next(self.tasks, last_completed_task=self.last_completed_task)
        if selected is None:
            return self._decision_result(None, reason='no pending task')
        selected.status = TaskStatus.ACTIVE
        return self._decision_result(selected, reason='selected highest priority pending task')

    def mark_task_succeeded(self, task_id: str) -> PickTask:
        task = self._find_task(task_id)
        task.status = TaskStatus.SUCCEEDED
        self.last_completed_task = task
        return task

    def mark_task_failed(self, task_id: str, *, requeue: bool = True) -> PickTask:
        task = self._find_task(task_id)
        task.retry_count += 1
        if requeue and task.retry_count <= task.max_retries:
            task.status = TaskStatus.PENDING
        else:
            task.status = TaskStatus.FAILED
        return task

    def export_plan(self) -> list[dict]:
        ranked = self.scheduler.rank_tasks(self.tasks, last_completed_task=self.last_completed_task)
        return [task.to_dict() for task in ranked]

    def build_execution_payload(self, task: PickTask) -> dict:
        return {
            'task_id': task.task_id,
            'product_name': task.product_name,
            'slot_id': task.slot_id,
            'aruco_id': task.aruco_id,
            'navigation_target': task.navigation_target.to_dict(),
            'delivery_target': DEFAULT_DELIVERY_TARGET.to_dict(),
            'grasp_strategy': task.grasp_strategy,
            'priority': task.priority,
            'retry_count': task.retry_count,
            'world_position': {
                'x': task.world_position[0],
                'y': task.world_position[1],
                'z': task.world_position[2],
            },
        }

    def _decision_result(self, selected: PickTask | None, *, reason: str) -> DecisionResult:
        return DecisionResult(
            cycle_id=self.cycle_id,
            selected_task=selected,
            pending_count=sum(task.status == TaskStatus.PENDING for task in self.tasks),
            active_count=sum(task.status == TaskStatus.ACTIVE for task in self.tasks),
            finished_count=sum(task.status == TaskStatus.SUCCEEDED for task in self.tasks),
            failed_count=sum(task.status == TaskStatus.FAILED for task in self.tasks),
            reason=reason,
        )

    def _find_task(self, task_id: str) -> PickTask:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f'unknown task_id: {task_id}')

    def _layout_item_to_task(self, item: dict) -> PickTask:
        shelf = item['shelf']
        level = item['level']
        column = item['column']
        slot_id = f"slot_{shelf}_{level}_{column}"
        task_id = f"{slot_id}_{item['object_kind']}"
        nav_target = NavigationTarget(
            frame_id='map',
            x=float(item['world_position'][0]) - RIGHT_ARM_OBJECT_X_OFFSET,
            y=APPROACH_BASE_Y,
            yaw=APPROACH_YAW,
        )
        grasp_strategy = self._default_grasp_strategy(item['object_kind'], level)
        return PickTask(
            task_id=task_id,
            product_name=item['object_kind'],
            slot_id=slot_id,
            aruco_id=int(item['aruco_id']),
            shelf=shelf,
            level=level,
            column=column,
            world_position=tuple(float(v) for v in item['world_position']),
            navigation_target=nav_target,
            grasp_strategy=grasp_strategy,
            metadata={
                'body': item['body'],
                'gs_ply': item.get('gs_ply'),
            },
        )

    def _default_grasp_strategy(self, object_kind: str, level: str) -> str:
        if object_kind in {'kele', 'maidong'}:
            return 'front_bottle_wrap'
        if object_kind in {'pingguo', 'chengzi'}:
            return 'top_fruit_cup'
        if level == 'L1':
            return 'low_front_pinching'
        return 'front_center'

    @staticmethod
    def _load_layout(path: Path) -> list[dict]:
        with path.open('r', encoding='utf-8') as fh:
            return json.load(fh)
