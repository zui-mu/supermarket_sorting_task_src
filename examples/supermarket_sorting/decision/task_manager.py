from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

from .models import DecisionResult, NavigationTarget, PickTask, TaskStatus
from .order_scheduler import OrderScheduler

DEFAULT_LAYOUT_PATH = Path(__file__).resolve().parent.parent / 'retail_competition_layout.json'
DEFAULT_DELIVERY_TARGET = NavigationTarget(frame_id='map', x=-1.88, y=-2.80, yaw=-1.5707963267948966)
APPROACH_BASE_X = 0.852
APPROACH_BASE_Y = 2.475
APPROACH_YAW = 1.3788101090755203
RIGHT_ARM_OBJECT_X_OFFSET = 0.108


class TaskManager:
    """Convert layout items or order requests into executable pick tasks."""

    def __init__(self, layout_path: str | Path = DEFAULT_LAYOUT_PATH, scheduler: OrderScheduler | None = None):
        self.layout_path = Path(layout_path)
        self.scheduler = scheduler or OrderScheduler()
        self.layout_items = self._load_layout(self.layout_path)
        self.tasks: list[PickTask] = []
        self.last_completed_task: PickTask | None = None
        self.cycle_id = 0
        self.requested_counts: Counter[str] = Counter()
        self.completed_counts: Counter[str] = Counter()
        self.preferred_retry_task_id: str | None = None
        self.lock_official_order = os.getenv('SUPERMARKET_LOCK_OFFICIAL_ORDER', '0') == '1'
        # In official search mode one physical slot is expanded into one
        # candidate per anonymous target. A failed search must blacklist that
        # product/slot pair, otherwise the scheduler revisits the same slot.
        self.failed_search_pairs: set[tuple[str, str]] = set()

    def build_search_tasks_for_targets(self, targets: Iterable[dict]) -> list[PickTask]:
        """Build V2.0-compliant search tasks from official target messages.

        The official /supermarket_sorting/task message gives only an anonymous
        target id and a product kind. It does not reveal the shelf slot.  These
        tasks therefore visit legal shelf slots and require perception to
        confirm the requested kind before grasping.
        """
        normalized: list[dict] = []
        for index, raw in enumerate(targets, 1):
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("kind", "")).strip()
            if not kind:
                continue
            target_id = str(raw.get("id") or f"official_target_{index:02d}")
            normalized.append({"id": target_id, "kind": kind})

        self.completed_counts.clear()
        self.preferred_retry_task_id = None
        self.failed_search_pairs.clear()
        self.requested_counts = Counter(target["kind"] for target in normalized)

        tasks: list[PickTask] = []
        unique_slots = self._unique_slots()
        body_items = {str(item.get("body")): item for item in self.layout_items}
        for order_index, target in enumerate(normalized):
            direct_item = body_items.get(target["id"])
            if direct_item is not None:
                task = self._layout_item_to_task(direct_item)
                task.metadata.update({
                    "official_target_id": target["id"],
                    "official_order": order_index,
                    "official_direct": True,
                    "search_mode": False,
                })
                tasks.append(task)
                continue
            for slot in unique_slots:
                task = self._slot_search_task(slot, target["kind"], target["id"])
                task.metadata["official_order"] = order_index
                tasks.append(task)
        self.tasks = tasks
        return tasks

    def build_tasks_for_products(self, product_names: Iterable[str]) -> list[PickTask]:
        requested = list(product_names)
        requested_limits = self._parse_product_requests(requested)
        if not requested_limits or any(name in {'all', '*'} for name in requested_limits):
            requested_limits = {
                item['object_kind']: None
                for item in self.layout_items
                if item.get('object_kind')
            }
        self.completed_counts.clear()
        self.preferred_retry_task_id = None
        tasks: list[PickTask] = []
        used_bodies: set[str] = set()
        self.requested_counts = Counter()
        for product_name, requested_limit in sorted(requested_limits.items()):
            if product_name in {'all', '*'}:
                continue
            matches = [item for item in self.layout_items if item['object_kind'] == product_name and item['body'] not in used_bodies]
            target_count = len(matches) if requested_limit is None else min(requested_limit, len(matches))
            self.requested_counts[product_name] = target_count
            for match in matches:
                task = self._layout_item_to_task(match)
                tasks.append(task)
                used_bodies.add(match['body'])
        self.tasks = tasks
        return tasks

    @staticmethod
    def _parse_product_requests(product_names: Iterable[str]) -> dict[str, int | None]:
        """Parse order specs.

        Plain names mean "all available" for that product.  Use "kele:1" to
        intentionally request only one item, or "kele:3" for a fixed quantity.
        """
        requests: dict[str, int | None] = {}
        for raw_name in product_names:
            spec = str(raw_name).strip()
            if not spec:
                continue
            if ':' in spec:
                product_name, raw_count = [part.strip() for part in spec.split(':', 1)]
                if not product_name:
                    continue
                if raw_count in {'', '*', 'all', 'ALL'}:
                    requests[product_name] = None
                else:
                    try:
                        requests[product_name] = max(0, int(raw_count))
                    except ValueError:
                        requests[product_name] = None
            else:
                requests[spec] = None
        return requests

    def build_tasks_from_referee_targets(self, target_bodies: Iterable[str]) -> list[PickTask]:
        wanted = set(target_bodies)
        self.tasks = [self._layout_item_to_task(item) for item in self.layout_items if item['body'] in wanted]
        return self.tasks

    def next_decision(self) -> DecisionResult:
        self.cycle_id += 1
        pending = [
            task for task in self.tasks
            if task.status == TaskStatus.PENDING
            and self.completed_counts[task.product_name] < self.requested_counts[task.product_name]
            and (
                task.product_name,
                task.slot_id,
            ) not in self.failed_search_pairs
        ]
        selected = None
        if self.preferred_retry_task_id is not None:
            selected = next(
                (task for task in pending if task.task_id == self.preferred_retry_task_id),
                None,
            )
            self.preferred_retry_task_id = None
        if selected is None and self.lock_official_order:
            ordered_pending = [
                task for task in pending
                if "official_order" in task.metadata
            ]
            if ordered_pending:
                # Optional compatibility mode for strictly order-locked runs.
                selected = min(
                    ordered_pending,
                    key=lambda task: (
                        int(task.metadata.get("official_order", 10**9)),
                        task.retry_count,
                        task.task_id,
                    ),
                )
        if selected is None:
            selected = self.scheduler.choose_next(pending, last_completed_task=self.last_completed_task)
        if selected is None:
            return self._decision_result(None, reason='no pending task')
        selected.status = TaskStatus.ACTIVE
        return self._decision_result(selected, reason='selected highest priority pending task')

    def mark_task_succeeded(self, task_id: str) -> PickTask:
        task = self._find_task(task_id)
        task.status = TaskStatus.SUCCEEDED
        self.failed_search_pairs.discard((task.product_name, task.slot_id))
        self.last_completed_task = task
        self.completed_counts[task.product_name] += 1
        # In search mode many candidate tasks can point at the same physical
        # slot. Once one succeeds, do not try to reuse that shelf position for
        # another anonymous target.
        for candidate in self.tasks:
            if (
                candidate.status == TaskStatus.PENDING
                and candidate.slot_id == task.slot_id
            ):
                candidate.status = TaskStatus.FAILED
        # Once the requested quantity is fulfilled, unused alternatives are no
        # longer pending work for this order.
        if self.completed_counts[task.product_name] >= self.requested_counts[task.product_name]:
            for candidate in self.tasks:
                if candidate.product_name == task.product_name and candidate.status == TaskStatus.PENDING:
                    candidate.status = TaskStatus.FAILED
        return task

    def mark_task_failed(self, task_id: str, *, requeue: bool = True) -> PickTask:
        task = self._find_task(task_id)
        task.retry_count += 1
        if task.metadata.get("search_mode"):
            # The visual search result applies to the physical slot, not just
            # this anonymous official target. Skip duplicate candidates for
            # the same product and slot in this run.
            pair = (task.product_name, task.slot_id)
            self.failed_search_pairs.add(pair)
            for candidate in self.tasks:
                if (
                    candidate.status == TaskStatus.PENDING
                    and candidate.metadata.get("search_mode")
                    and candidate.product_name == task.product_name
                    and candidate.slot_id == task.slot_id
                ):
                    candidate.status = TaskStatus.FAILED
            task.status = TaskStatus.FAILED
            self.preferred_retry_task_id = None
            return task
        if requeue and task.retry_count <= task.max_retries:
            task.status = TaskStatus.PENDING
            self.preferred_retry_task_id = task.task_id
        else:
            task.status = TaskStatus.FAILED
            self.preferred_retry_task_id = None
        return task

    def export_plan(self) -> list[dict]:
        ranked = self.scheduler.rank_tasks(self.tasks, last_completed_task=self.last_completed_task)
        return [task.to_dict() for task in ranked]

    def build_execution_payload(self, task: PickTask) -> dict:
        return {
            'task_id': task.task_id,
            'product_name': task.product_name,
            'slot_id': task.slot_id,
            'display_slot': task.metadata.get('display_slot', task.slot_id),
            'referee_body': task.metadata.get('body'),
            'official_target_id': task.metadata.get('official_target_id'),
            'search_mode': bool(task.metadata.get('search_mode')),
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
                # The randomized shelf/level/column labels describe where the
                # item is displayed; body is the referee's physical identity.
                'display_slot': slot_id,
                'gs_ply': item.get('gs_ply'),
            },
        )

    def _unique_slots(self) -> list[dict]:
        slots: dict[str, dict] = {}
        for item in self.layout_items:
            shelf = item['shelf']
            level = item['level']
            column = item['column']
            slot_id = f"slot_{shelf}_{level}_{column}"
            if slot_id not in slots:
                slots[slot_id] = item
        return list(slots.values())

    def _slot_search_task(self, item: dict, product_name: str, target_id: str) -> PickTask:
        shelf = item['shelf']
        level = item['level']
        column = item['column']
        slot_id = f"slot_{shelf}_{level}_{column}"
        task_id = f"{target_id}_{slot_id}_{product_name}"
        nav_target = NavigationTarget(
            frame_id='map',
            x=float(item['world_position'][0]) - RIGHT_ARM_OBJECT_X_OFFSET,
            y=APPROACH_BASE_Y,
            yaw=APPROACH_YAW,
        )
        return PickTask(
            task_id=task_id,
            product_name=product_name,
            slot_id=slot_id,
            aruco_id=int(item['aruco_id']),
            shelf=shelf,
            level=level,
            column=column,
            world_position=tuple(float(v) for v in item['world_position']),
            navigation_target=nav_target,
            grasp_strategy=self._default_grasp_strategy(product_name, level),
            max_retries=0,
            metadata={
                'display_slot': slot_id,
                'official_target_id': target_id,
                'search_mode': True,
            },
        )

    def _default_grasp_strategy(self, object_kind: str, level: str) -> str:
        if object_kind in {'kele', 'maidong'}:
            return 'front_bottle_wrap'
        if object_kind in {'sanmingzhi', 'heweidao', 'shupian', 'zhijin', 'kouxiangtang'}:
            return 'front_box_clamp'
        if object_kind in {'pingguo', 'chengzi'}:
            return 'front_fruit_cradle'
        if level == 'L1':
            return 'low_front_pinching'
        return 'front_center'

    @staticmethod
    def _load_layout(path: Path) -> list[dict]:
        with path.open('r', encoding='utf-8') as fh:
            return json.load(fh)
