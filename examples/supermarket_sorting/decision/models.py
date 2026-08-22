from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class NavigationTarget:
    frame_id: str
    x: float
    y: float
    yaw: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScanViewpoint:
    viewpoint_id: str
    scope: str
    shelf: str
    navigation_target: NavigationTarget
    head_pitches: tuple[float, ...]
    covered_aruco_ids: tuple[int, ...] = ()
    column: str | None = None
    priority: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["navigation_target"] = self.navigation_target.to_dict()
        return payload


@dataclass(slots=True)
class PickTask:
    task_id: str
    product_name: str
    slot_id: str
    aruco_id: int
    shelf: str
    level: str
    column: str
    world_position: tuple[float, float, float]
    navigation_target: NavigationTarget
    grasp_strategy: str
    priority: float = 0.0
    retry_count: int = 0
    max_retries: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)
    # The route target is frozen once navigation starts.  A separate fresh
    # grasp pose is allowed to move after the chassis has parked and the same
    # frame has re-confirmed both the active ArUco id and product class.
    navigation_world_position: tuple[float, float, float] | None = None
    fresh_grasp_world: tuple[float, float, float] | None = None
    fresh_grasp_seen_at: float | None = None
    status: TaskStatus = TaskStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(slots=True)
class DecisionResult:
    cycle_id: int
    selected_task: PickTask | None
    pending_count: int
    active_count: int
    finished_count: int
    failed_count: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "selected_task": None if self.selected_task is None else self.selected_task.to_dict(),
            "pending_count": self.pending_count,
            "active_count": self.active_count,
            "finished_count": self.finished_count,
            "failed_count": self.failed_count,
            "reason": self.reason,
        }
