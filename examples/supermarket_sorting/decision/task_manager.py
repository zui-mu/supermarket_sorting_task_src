from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from .models import DecisionResult, NavigationTarget, PickTask, TaskStatus
from .order_scheduler import OrderScheduler

DEFAULT_LAYOUT_PATH = Path(__file__).resolve().parent.parent / 'retail_competition_layout.json'
DEFAULT_DELIVERY_TARGET = NavigationTarget(
    frame_id='map',
    x=float(os.getenv("SUPERMARKET_DELIVERY_GOAL_X", "-1.82")),
    y=float(os.getenv("SUPERMARKET_DELIVERY_TARGET_Y", "-2.80")),
    yaw=-1.5707963267948966,
)
APPROACH_BASE_X = 0.852
APPROACH_BASE_Y = 2.475
APPROACH_YAW = 1.5707963267948966
RIGHT_ARM_OBJECT_X_OFFSET = 0.108
# Product locations are measured from RGB-D, while the chassis must stop in
# front of the shelf.  This is a calibrated robot/shelf clearance, not a
# randomized item coordinate or server-side layout truth.
SHELF_APPROACH_STANDOFF_Y = float(os.getenv("SUPERMARKET_SHELF_APPROACH_STANDOFF_Y", "0.768"))
# The long tissue box uses a dedicated shelf standoff. It remains separate
# from the upright-product calibration so a collision-checked tissue planner
# can refine it without perturbing all other product profiles.
TOP_BOX_APPROACH_STANDOFF_Y = float(os.getenv("SUPERMARKET_TOP_BOX_APPROACH_STANDOFF_Y", "0.58"))
TOP_BOX_LOWER_APPROACH_STANDOFF_Y = float(os.getenv(
    "SUPERMARKET_TOP_BOX_LOWER_APPROACH_STANDOFF_Y", "0.70"
))
SHELF_APPROACH_Y_MIN = float(os.getenv("SUPERMARKET_SHELF_APPROACH_Y_MIN", "2.20"))
SHELF_APPROACH_Y_MAX = float(os.getenv("SUPERMARKET_SHELF_APPROACH_Y_MAX", "2.70"))
INVENTORY_MIN_CONFIDENCE = float(os.getenv("SUPERMARKET_INVENTORY_MIN_CONFIDENCE", "0.55"))
INVENTORY_MIN_HITS = max(1, int(os.getenv("SUPERMARKET_INVENTORY_MIN_HITS", "2")))
INVENTORY_WORLD_TOLERANCE = float(os.getenv("SUPERMARKET_INVENTORY_WORLD_TOLERANCE", "0.18"))
INVENTORY_SLOT_X_TOL = float(os.getenv("SUPERMARKET_INVENTORY_SLOT_X_TOL", "0.28"))
INVENTORY_SLOT_Y_TOL = float(os.getenv("SUPERMARKET_INVENTORY_SLOT_Y_TOL", "0.30"))
INVENTORY_SLOT_X_BLEND_LIMIT = float(os.getenv("SUPERMARKET_INVENTORY_SLOT_X_BLEND_LIMIT", "0.045"))
INVENTORY_SLOT_Y_BLEND_LIMIT = float(os.getenv("SUPERMARKET_INVENTORY_SLOT_Y_BLEND_LIMIT", "0.080"))
INVENTORY_MAX_AGE_SEC = max(
    0.001, float(os.getenv("SUPERMARKET_INVENTORY_MAX_AGE_SEC", "12.0"))
)
# PR3: split the inventory into two time scales.  IDENTITY (aruco_id -> kind)
# is long-lived within one run_prefix (the tag keeps telling you what slot it
# is even after the robot delivered an item); POSE (exact coordinates) is
# short-lived - it must be re-acquired right before grasping.  Keeping one
# 12 s age for both made an already-scanned slot "expire" after a delivery
# and forced a full rescan of the whole shelf.
INVENTORY_POSE_MAX_AGE_SEC = max(
    0.001, float(os.getenv("SUPERMARKET_INVENTORY_POSE_MAX_AGE_SEC", "12.0"))
)
INVENTORY_IDENTITY_MAX_AGE_SEC = max(
    0.001, float(os.getenv("SUPERMARKET_INVENTORY_IDENTITY_MAX_AGE_SEC", "3600.0"))
)
# 归一化参考往返距离(m): 路径项除以它之后再乘权重, 使路径、置信度、新鲜度、
# 风险各项处于同一量级, 避免"就近优先"完全压过其他信号。
INVENTORY_ROUTE_REFERENCE_M = max(
    0.001, float(os.getenv("SUPERMARKET_INVENTORY_ROUTE_REFERENCE_M", "10.0"))
)
INVENTORY_STATE_OBSERVED = "observed"
INVENTORY_STATE_CONFIRMED = "confirmed"
INVENTORY_STATE_RESERVED = "reserved"
INVENTORY_STATE_CONSUMED = "consumed"
INVENTORY_STATE_DISTURBED = "disturbed"
INVENTORY_CONFIDENCE_WEIGHT = float(os.getenv("SUPERMARKET_INVENTORY_CONFIDENCE_WEIGHT", "40.0"))
INVENTORY_FRESHNESS_WEIGHT = float(os.getenv("SUPERMARKET_INVENTORY_FRESHNESS_WEIGHT", "18.0"))
INVENTORY_ROUTE_WEIGHT = float(os.getenv("SUPERMARKET_INVENTORY_ROUTE_WEIGHT", "12.0"))
INVENTORY_RISK_WEIGHT = float(os.getenv("SUPERMARKET_INVENTORY_RISK_WEIGHT", "24.0"))
INVENTORY_RESERVATION_BONUS = float(os.getenv("SUPERMARKET_INVENTORY_RESERVATION_BONUS", "14.0"))
INVENTORY_SAME_SHELF_BONUS = float(os.getenv("SUPERMARKET_INVENTORY_SAME_SHELF_BONUS", "8.0"))
INVENTORY_SAME_LEVEL_BONUS = float(os.getenv("SUPERMARKET_INVENTORY_SAME_LEVEL_BONUS", "4.0"))
INVENTORY_RETRY_PENALTY = float(os.getenv("SUPERMARKET_INVENTORY_RETRY_PENALTY", "4.0"))
SEARCH_PRODUCT = "__search__"
SEARCH_SLOT_MAX_RETRIES = max(
    0,
    int(os.getenv("SUPERMARKET_SEARCH_SLOT_RETRIES", "1")),
)
SHELF_SURFACE_Z = {
    "L1": 0.499,
    "L2": 0.851,
    "L3": 1.189,
}


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
        self.current_official_order: int | None = None
        self.lock_official_order = os.getenv('SUPERMARKET_LOCK_OFFICIAL_ORDER', '0') == '1'
        # In official search mode, scan a physical slot once. If perception
        # cannot confirm the target there, move on instead of trying the same
        # shelf position again under another anonymous target id.
        self.failed_search_slots: set[str] = set()
        self.inventory_by_aruco: dict[int, dict] = {}
        self.slot_by_aruco = {
            int(item["aruco_id"]): item for item in self.layout_items
        }
        self.product_half_height = self._estimate_product_half_heights()

    def _estimate_product_half_heights(self) -> dict[str, float]:
        heights: dict[str, list[float]] = {}
        for item in self.layout_items:
            kind = str(item.get("object_kind", ""))
            level = str(item.get("level", ""))
            surface = SHELF_SURFACE_Z.get(level)
            if not kind or surface is None:
                continue
            try:
                half_height = float(item["world_position"][2]) - float(surface)
            except (KeyError, TypeError, ValueError):
                continue
            if 0.01 <= half_height <= 0.18:
                heights.setdefault(kind, []).append(half_height)
        return {
            kind: sum(values) / len(values)
            for kind, values in heights.items()
            if values
        }

    def _inventory_world_for_slot(
        self,
        *,
        aruco_id: int,
        kind: str,
        observed_world: tuple[float, float, float],
    ) -> tuple[float, float, float] | None:
        """Clamp an ArUco-bound product pose to the physical shelf slot.

        YOLO depth gives a useful class and a small lateral/depth correction,
        but its 3-D centre can jump to an adjacent layer or visible package
        face. The marker identifies the legal slot; combine that slot with the
        detected kind's known half-height so randomized official layouts stay
        anonymous while producing reachable grasp poses.
        """
        slot = self.slot_by_aruco.get(int(aruco_id))
        if slot is None:
            return observed_world
        try:
            slot_world = tuple(float(value) for value in slot["world_position"])
        except (KeyError, TypeError, ValueError):
            return observed_world
        level = str(slot.get("level", ""))
        if (
            abs(float(observed_world[0]) - slot_world[0]) > INVENTORY_SLOT_X_TOL
            or abs(float(observed_world[1]) - slot_world[1]) > INVENTORY_SLOT_Y_TOL
        ):
            return None
        half_height = self.product_half_height.get(str(kind))
        surface_z = SHELF_SURFACE_Z.get(level)
        if half_height is None or surface_z is None:
            target_z = slot_world[2]
        else:
            target_z = surface_z + half_height
        dx = max(
            -INVENTORY_SLOT_X_BLEND_LIMIT,
            min(INVENTORY_SLOT_X_BLEND_LIMIT, float(observed_world[0]) - slot_world[0]),
        )
        dy = max(
            -INVENTORY_SLOT_Y_BLEND_LIMIT,
            min(INVENTORY_SLOT_Y_BLEND_LIMIT, float(observed_world[1]) - slot_world[1]),
        )
        return (
            float(slot_world[0] + dx),
            float(slot_world[1] + dy),
            float(target_z),
        )

    def register_inventory_observations(self, observations: Iterable[dict]) -> int:
        """Store stable RGB-D observations keyed by their visible ArUco ID.

        A single detector frame is insufficient to turn an anonymous slot into
        a grasp target.  The record becomes usable only after repeated,
        spatially consistent observations of the same class.
        """
        accepted = 0
        now = time.time()
        for observation in observations:
            try:
                aruco_id = int(observation["aruco_id"])
                kind = str(observation["kind"]).strip()
                confidence = float(observation.get("confidence", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            world = self._valid_observation_world(observation.get("world"))
            if (
                not 0 <= aruco_id < 45
                or not kind
                or confidence < INVENTORY_MIN_CONFIDENCE
                or world is None
            ):
                continue
            world = self._inventory_world_for_slot(
                aruco_id=aruco_id,
                kind=kind,
                observed_world=world,
            )
            if world is None:
                continue
            marker_world = self._valid_observation_world(observation.get("marker_world"))
            stamp = observation.get("stamp")
            try:
                stamp = float(stamp)
            except (TypeError, ValueError):
                stamp = None
            # Freshness must age against the local wall clock. A header stamp
            # from another clock domain (sim time, paused /clock) would make
            # every record instantly stale or never stale; in that case age
            # from the wall clock instead. The raw stamp is still preserved on
            # the record for diagnostics.
            fresh_at = stamp if stamp is not None and abs(stamp - now) <= 60.0 else now
            previous = self.inventory_by_aruco.get(aruco_id)
            if previous is None:
                self.inventory_by_aruco[aruco_id] = {
                    "kind": kind,
                    "confidence": confidence,
                    "world": world,
                    "marker_world": marker_world,
                    "hits": 1,
                    "confirmed": INVENTORY_MIN_HITS == 1,
                    "state": INVENTORY_STATE_CONFIRMED if INVENTORY_MIN_HITS == 1 else INVENTORY_STATE_OBSERVED,
                    "first_seen": stamp if stamp is not None else now,
                    "last_seen": stamp if stamp is not None else now,
                    "fresh_at": fresh_at,
                    "identity_seen_at": fresh_at,
                    "pose_seen_at": fresh_at,
                }
                if INVENTORY_MIN_HITS == 1:
                    accepted += 1
                continue

            if previous.get("state") == "consumed":
                continue

            if previous.get("state") == INVENTORY_STATE_RESERVED:
                # An active task has claimed this slot. Keep the reservation
                # intact until that task settles it; otherwise a stream of
                # fresh frames would silently demote `reserved` back to
                # `confirmed` while the robot is already executing on it.
                continue

            if previous.get("kind") != kind:
                # Do not replace a confirmed inventory record on a transient
                # misclassification.  Before confirmation, restart evidence
                # collection for the newly observed class.
                if previous.get("confirmed"):
                    continue
                self.inventory_by_aruco[aruco_id] = {
                    "kind": kind,
                    "confidence": confidence,
                    "world": world,
                    "marker_world": marker_world,
                    "hits": 1,
                    "confirmed": INVENTORY_MIN_HITS == 1,
                    "state": INVENTORY_STATE_CONFIRMED if INVENTORY_MIN_HITS == 1 else INVENTORY_STATE_OBSERVED,
                    "first_seen": stamp if stamp is not None else now,
                    "last_seen": stamp if stamp is not None else now,
                    "fresh_at": fresh_at,
                    "identity_seen_at": fresh_at,
                    "pose_seen_at": fresh_at,
                }
                if INVENTORY_MIN_HITS == 1:
                    accepted += 1
                continue

            previous_world = previous.get("world")
            if previous_world is not None:
                distance = math.dist(previous_world, world)
                if distance > INVENTORY_WORLD_TOLERANCE:
                    # A different depth surface or a moved item must be seen
                    # consistently before it can replace the existing pose.
                    if previous.get("confirmed"):
                        continue
                    previous["world"] = world
                    previous["marker_world"] = marker_world
                    previous["hits"] = 1
                    previous["confidence"] = confidence
                    previous["last_seen"] = stamp if stamp is not None else now
                    previous["fresh_at"] = fresh_at
                    previous["state"] = INVENTORY_STATE_OBSERVED
                    continue

            was_confirmed = bool(previous.get("confirmed"))
            old_hits = int(previous.get("hits", 1))
            new_hits = min(old_hits + 1, 12)
            blend = 1.0 / float(min(old_hits + 1, 6))
            previous["world"] = tuple(
                (1.0 - blend) * float(old) + blend * float(new)
                for old, new in zip(previous.get("world", world), world)
            )
            previous["confidence"] = max(float(previous.get("confidence", 0.0)), confidence)
            if marker_world is not None:
                previous["marker_world"] = marker_world
            previous["hits"] = new_hits
            previous["confirmed"] = new_hits >= INVENTORY_MIN_HITS
            previous["state"] = INVENTORY_STATE_CONFIRMED if previous["confirmed"] else INVENTORY_STATE_OBSERVED
            previous["last_seen"] = stamp if stamp is not None else now
            previous["fresh_at"] = fresh_at
            previous["pose_seen_at"] = fresh_at
            if previous["confirmed"] and not was_confirmed:
                previous["identity_seen_at"] = fresh_at
                accepted += 1
        return accepted

    @staticmethod
    def _inventory_age(record: dict, now: float | None = None) -> float | None:
        # Prefer the wall-clock freshness stamp over the raw sensor timestamp;
        # records created by tests or older versions may only carry last_seen.
        seen = record.get("fresh_at", record.get("last_seen"))
        try:
            seen = float(seen)
        except (TypeError, ValueError):
            return None
        if now is None:
            now = time.time()
        return max(0.0, float(now) - seen)

    def _inventory_is_fresh(self, record: dict, *, now: float | None = None) -> bool:
        # PR3: pose freshness (short-lived, re-acquire before grasp).
        age = self._inventory_age(record, now=now)
        return False if age is None else age <= INVENTORY_POSE_MAX_AGE_SEC

    def _inventory_identity_fresh(self, record: dict, *, now: float | None = None) -> bool:
        # PR3: identity (aruco_id -> kind) is long-lived within the run: once
        # confirmed it stays selectable (for a second identical order, for
        # re-delivery) even after the pose went stale - the robot just has to
        # re-observe the slot before grasping.
        if now is None:
            now = time.time()
        seen = record.get("identity_seen_at")
        try:
            seen = float(seen)
        except (TypeError, ValueError):
            seen = record.get("fresh_at", record.get("last_seen"))
            try:
                seen = float(seen)
            except (TypeError, ValueError):
                return False
        return max(0.0, float(now) - seen) <= INVENTORY_IDENTITY_MAX_AGE_SEC

    def _reserve_inventory_record(self, task: PickTask) -> bool:
        record = self.inventory_by_aruco.get(int(task.aruco_id))
        if record is None or not record.get("confirmed"):
            return False
        if record.get("state") in {INVENTORY_STATE_CONSUMED, INVENTORY_STATE_DISTURBED}:
            return False
        record["state"] = INVENTORY_STATE_RESERVED
        record["reserved_task_id"] = task.task_id
        record["reserved_at"] = time.time()
        return True

    def _release_inventory_reservation(self, task: PickTask) -> None:
        record = self.inventory_by_aruco.get(int(task.aruco_id))
        if record is None:
            return
        if record.get("reserved_task_id") not in {None, task.task_id}:
            return
        if record.get("state") == INVENTORY_STATE_RESERVED:
            record["state"] = INVENTORY_STATE_CONFIRMED if record.get("confirmed") else INVENTORY_STATE_OBSERVED
        record.pop("reserved_task_id", None)
        record.pop("reserved_at", None)

    def _mark_inventory_state(self, task: PickTask, state: str) -> None:
        record = self.inventory_by_aruco.get(int(task.aruco_id))
        if record is None:
            return
        record["state"] = state
        record["last_seen"] = time.time()
        record["fresh_at"] = time.time()

    @staticmethod
    def _valid_observation_world(raw_world) -> tuple[float, float, float] | None:
        try:
            world = tuple(float(value) for value in raw_world)
        except (TypeError, ValueError):
            return None
        if len(world) != 3 or not all(math.isfinite(value) for value in world):
            return None
        # Conservative global bounds reject depth holes/floor returns without
        # encoding any randomized product positions.
        if not (-3.5 <= world[0] <= 3.5 and 1.8 <= world[1] <= 3.8 and 0.25 <= world[2] <= 1.7):
            return None
        return world

    def build_search_tasks_for_targets(
        self,
        targets: Iterable[dict],
        *,
        trust_layout_positions: bool = False,
    ) -> list[PickTask]:
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
        self.current_official_order = None
        self.failed_search_slots.clear()
        self.requested_counts = Counter(target["kind"] for target in normalized)
        # PR3: a new run (new /task payload) starts a fresh inventory - the
        # ArUco tags stay fixed but the products under them were re-randomised,
        # so last run's identity table is void.
        self.inventory_by_aruco.clear()
        # A new order must not inherit same-shelf/same-level bonuses from the
        # previous order's last completed task.
        self.last_completed_task = None

        tasks: list[PickTask] = []
        unique_slots = self._unique_slots()
        body_items = {str(item.get("body")): item for item in self.layout_items}
        unresolved_targets: list[tuple[int, dict]] = []
        for order_index, target in enumerate(normalized):
            direct_item = body_items.get(target["id"])
            if direct_item is not None and trust_layout_positions:
                task = self._layout_item_to_task(direct_item)
                task.metadata.update({
                    "official_target_id": target["id"],
                    "official_order": order_index,
                    "official_direct": True,
                    "search_mode": False,
                })
                tasks.append(task)
                continue
            unresolved_targets.append((order_index, target))

        if unresolved_targets:
            requested_kinds = sorted({target["kind"] for _, target in unresolved_targets})
            # A search action belongs to a physical shelf slot, not to a guessed
            # product. The former product x slot expansion revisited a failed
            # position under a different target id and caused target switching.
            for slot_index, slot in enumerate(unique_slots):
                task = self._slot_search_task(slot, SEARCH_PRODUCT, f"search_{slot_index:02d}")
                task.metadata.update({
                    "requested_kinds": requested_kinds,
                })
                tasks.append(task)
        self.tasks = tasks
        return tasks

    def build_tasks_for_products(self, product_names: Iterable[str]) -> list[PickTask]:
        requested = list(product_names)
        requested_limits = self._parse_product_requests(requested)
        if not requested_limits or any(name in {'all', '*'} for name in requested_limits):
            requested_limits = {}
            for item in self.layout_items:
                kind = str(item.get('object_kind') or "").strip()
                if not kind or kind in requested_limits:
                    continue
                requested_limits[kind] = None
        self.completed_counts.clear()
        self.preferred_retry_task_id = None
        self.current_official_order = None
        # A new order must not inherit same-shelf/same-level bonuses from the
        # previous order's last completed task.
        self.last_completed_task = None
        tasks: list[PickTask] = []
        used_bodies: set[str] = set()
        self.requested_counts = Counter()
        for item in self.layout_items:
            product_name = str(item.get('object_kind') or "").strip()
            if not product_name or product_name not in requested_limits:
                continue
            if item['body'] in used_bodies:
                continue
            requested_limit = requested_limits[product_name]
            if requested_limit is not None and self.requested_counts[product_name] >= requested_limit:
                continue
            task = self._layout_item_to_task(item)
            tasks.append(task)
            used_bodies.add(item['body'])
            self.requested_counts[product_name] += 1
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

    def _inventory_has_all_requested(self) -> bool:
        """PR4: True when every requested kind already has enough CONFIRMED
        inventory (identity) - scanning can stop, remaining picks use the
        observed slots instead of visiting every shelf."""
        if not self.requested_counts:
            return False
        now = time.time()
        for kind, need in self.requested_counts.items():
            if need <= 0:
                continue
            have = sum(
                1 for rec in self.inventory_by_aruco.values()
                if rec.get("confirmed") and rec.get("kind") == kind
                and self._inventory_identity_fresh(rec, now=now)
                and rec.get("state") != INVENTORY_STATE_CONSUMED
            )
            if have < need:
                return False
        return True

    def next_decision(self) -> DecisionResult:
        self.cycle_id += 1
        requested_total = sum(self.requested_counts.values())
        completed_total = sum(
            min(self.completed_counts[kind], count)
            for kind, count in self.requested_counts.items()
        )
        all_requested = self._inventory_has_all_requested()
        pending = [
            task for task in self.tasks
            if task.status == TaskStatus.PENDING
            and self._task_is_selectable(task)
            and (
                (
                    task.metadata.get("search_mode")
                    and completed_total < requested_total
                    and (
                        # PR4: once every requested kind is confirmed in the
                        # inventory, stop visiting un-scanned shelves - only
                        # the already-OBSERVED search tasks stay selectable
                        # (they are the ones we now go grasp).
                        not all_requested
                        or (
                            (observation := self.inventory_by_aruco.get(int(task.aruco_id))) is not None
                            and observation.get("confirmed")
                        )
                    )
                )
                or self.completed_counts[task.product_name] < self.requested_counts[task.product_name]
            )
            and task.slot_id not in self.failed_search_slots
        ]
        selected = None
        if self.preferred_retry_task_id is not None:
            selected = next(
                (task for task in pending if task.task_id == self.preferred_retry_task_id),
                None,
            )
            self.preferred_retry_task_id = None
        if selected is None:
            now = time.time()
            observed_matches = [
                task for task in pending
                if task.metadata.get("search_mode")
                and (observation := self.inventory_by_aruco.get(int(task.aruco_id))) is not None
                and observation.get("confirmed")
                # PR3: identity (this tag IS this kind) is long-lived within
                # the run - a slot already scanned must stay selectable after a
                # delivery instead of expiring and forcing a full rescan.  The
                # short-lived pose is re-acquired right before grasping.
                and self._inventory_identity_fresh(observation, now=now)
                and self.completed_counts[observation["kind"]] < self.requested_counts[observation["kind"]]
            ]
            if observed_matches:
                scored_matches = [
                    (
                        self._inventory_candidate_score(task, now=now),
                        self.scheduler.sort_key(task),
                        task.task_id,
                        task,
                    )
                    for task in observed_matches
                ]
                selected_score, _, _, selected = min(
                    scored_matches,
                    key=lambda item: (
                        -item[0],
                        item[1],
                        item[2],
                    ),
                )
                selected.metadata["selection_score"] = selected_score
        if selected is None:
            ordered_pending = [
                task for task in pending
                if "official_order" in task.metadata
            ]
            if ordered_pending and self.lock_official_order:
                active_order = self.current_official_order
                if active_order is None:
                    active_order = min(
                        int(task.metadata.get("official_order", 10**9))
                        for task in ordered_pending
                    )
                same_order = [
                    task for task in ordered_pending
                    if int(task.metadata.get("official_order", 10**9)) == active_order
                ]
                pool = same_order or ordered_pending
                selected = min(pool, key=self.scheduler.sort_key)
        if selected is None:
            selected = self.scheduler.choose_next(pending, last_completed_task=self.last_completed_task)
        if selected is None:
            return self._decision_result(None, reason='no pending task')
        self.apply_inventory_observation(selected)
        if selected.metadata.get("search_mode"):
            self._reserve_inventory_record(selected)
        if "official_order" in selected.metadata and self.lock_official_order:
            self.current_official_order = int(selected.metadata.get("official_order", 0))
        selected.status = TaskStatus.ACTIVE
        return self._decision_result(selected, reason='selected highest priority pending task')

    def apply_inventory_observation(self, task: PickTask) -> bool:
        """Bind a confirmed anonymous slot to its camera-observed product pose."""
        if not task.metadata.get("search_mode"):
            return False
        if task.metadata.get("inventory_confirmed"):
            # The first confirmed observation establishes the physical slot and
            # its pose stays fixed for this attempt; later re-observations must
            # not rewrite the navigation target underneath a moving robot.
            return False
        observation = self.inventory_by_aruco.get(int(task.aruco_id))
        if not observation or not observation.get("confirmed"):
            return False
        product_name = str(observation.get("kind", ""))
        if not product_name or self.completed_counts[product_name] >= self.requested_counts[product_name]:
            return False
        world = self._valid_observation_world(observation.get("world"))
        if world is None:
            return False

        already_applied = (
            task.metadata.get("inventory_confirmed")
            and task.product_name == product_name
            and tuple(task.world_position) == world
        )
        if already_applied:
            return False

        task.product_name = product_name
        task.world_position = world
        task.grasp_strategy = self._default_grasp_strategy(product_name, task.level)
        if task.grasp_strategy == "front_short_axis_box_clamp":
            standoff_y = (
                TOP_BOX_APPROACH_STANDOFF_Y
                if str(task.level) == "L3"
                else TOP_BOX_LOWER_APPROACH_STANDOFF_Y
            )
        else:
            standoff_y = SHELF_APPROACH_STANDOFF_Y
        task.navigation_target = NavigationTarget(
            frame_id="map",
            x=world[0] - RIGHT_ARM_OBJECT_X_OFFSET,
            y=float(max(
                SHELF_APPROACH_Y_MIN,
                min(world[1] - standoff_y, SHELF_APPROACH_Y_MAX),
            )),
            yaw=APPROACH_YAW,
        )
        task.metadata.update({
            "detected_kind": product_name,
            "inventory_confirmed": True,
            "inventory_hits": int(observation.get("hits", 0)),
            "inventory_world": list(world),
            "inventory_state": observation.get("state", INVENTORY_STATE_CONFIRMED),
        })
        return True

    def mark_task_succeeded(self, task_id: str, *, referee_verified: bool = True) -> PickTask:
        """Record a completed execution and preserve how it was verified."""
        task = self._find_task(task_id)
        if task.product_name == SEARCH_PRODUCT:
            raise RuntimeError("cannot complete an unbound shelf search task")
        task.status = TaskStatus.SUCCEEDED
        task.metadata["completion_evidence"] = (
            "referee_s5" if referee_verified else "local_execution_only"
        )
        self.failed_search_slots.discard(task.slot_id)
        self._mark_inventory_state(task, INVENTORY_STATE_CONSUMED)
        self.last_completed_task = task
        self.completed_counts[task.product_name] += 1
        self._retire_inventory_after_completion(task, state="consumed")
        if "official_order" in task.metadata:
            self.current_official_order = None
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

    def find_task_by_referee_body(self, referee_body: str) -> PickTask | None:
        """Return the task that corresponds to a referee-confirmed body id."""
        body = str(referee_body or "")
        if not body:
            return None
        for task in self.tasks:
            if (
                task.task_id == body
                or str(task.metadata.get("body") or "") == body
                or str(task.metadata.get("official_target_id") or "") == body
            ):
                return task
        return None

    def requested_kind_for_body(self, referee_body: str) -> str | None:
        """Development-only legacy lookup for a named, non-anonymous body.

        Anonymous V2 tasks must derive the product class from perception.  The
        public static slot geometry intentionally cannot be used to infer the
        product behind an arbitrary referee body name.
        """
        if os.getenv("SUPERMARKET_ALLOW_RUNTIME_LAYOUT", "0") != "1":
            return None
        body = str(referee_body or "")
        item = next(
            (candidate for candidate in self.layout_items if str(candidate.get("body")) == body),
            None,
        )
        if item is None:
            return None
        kind = str(item.get("object_kind") or "")
        if not kind or self.completed_counts[kind] >= self.requested_counts[kind]:
            return None
        return kind

    def bind_search_task_product(self, task_id: str, product_name: str) -> PickTask | None:
        """Bind one physical-slot search action to a detected product kind."""
        task = self._find_task(task_id)
        if not task.metadata.get("search_mode"):
            return None
        product_name = str(product_name or "")
        if self.completed_counts[product_name] >= self.requested_counts[product_name]:
            return None
        task.product_name = product_name
        task.grasp_strategy = self._default_grasp_strategy(product_name, task.level)
        task.metadata["detected_kind"] = product_name
        return task

    def rebind_active_task_to_referee_body(
        self,
        referee_body: str,
        current_task_id: str | None,
    ) -> PickTask | None:
        """Adopt the item the referee says is actually in the gripper.

        In randomized V2 scenes, a geometry-only grasp can bind a neighbouring
        shelf body.  The score cares about delivered referee targets, not our
        stale internal guess, so keep carrying any valid unsucceeded target
        instead of declaring failure and retracting the loaded arm.
        """
        grabbed = self.find_task_by_referee_body(referee_body)
        if grabbed is None and current_task_id:
            try:
                current = self._find_task(current_task_id)
            except KeyError:
                current = None
            if current is not None and current.metadata.get("search_mode"):
                # The referee's body string confirms contact, but never tells
                # the official controller which product was in that slot.
                # Keep only the class already established by perception.
                actual_kind = str(current.metadata.get("detected_kind") or "")
                if not actual_kind or self.completed_counts[actual_kind] >= self.requested_counts[actual_kind]:
                    return None
                current.product_name = actual_kind
                current.grasp_strategy = self._default_grasp_strategy(actual_kind, current.level)
                current.metadata["body"] = str(referee_body)
                current.metadata["bound_referee_body"] = str(referee_body)
                current.metadata["detected_kind"] = actual_kind
                return current
        if grabbed is None or grabbed.status == TaskStatus.SUCCEEDED:
            return None

        if current_task_id and grabbed.task_id != current_task_id:
            try:
                previous = self._find_task(current_task_id)
            except KeyError:
                previous = None
            if previous is not None and previous.status == TaskStatus.ACTIVE:
                previous.status = TaskStatus.FAILED
                self._fail_same_physical_target(previous)
                # The retired task may still hold an inventory reservation on
                # its slot; release it so the slot cannot starve later.
                self._release_inventory_reservation(previous)

        grabbed.status = TaskStatus.ACTIVE
        if "official_order" in grabbed.metadata:
            self.current_official_order = int(grabbed.metadata.get("official_order", 0))
        return grabbed

    def mark_task_failed(self, task_id: str, *, requeue: bool = True) -> PickTask:
        task = self._find_task(task_id)
        task.retry_count += 1
        if task.metadata.get("search_mode"):
            # Keep one bounded retry for navigation failures before contact.
            # The decision client passes requeue=False after a touch/drop or
            # an exhausted grasp attempt, so a disturbed slot is still retired
            # and cannot be revisited under another anonymous target id.
            if requeue and task.retry_count <= task.max_retries:
                task.status = TaskStatus.PENDING
                self._release_inventory_reservation(task)
                self.preferred_retry_task_id = task.task_id
                return task
            self.failed_search_slots.add(task.slot_id)
            if not requeue:
                self._mark_inventory_state(task, INVENTORY_STATE_DISTURBED)
                self._retire_inventory_after_completion(task, state="disturbed")
            else:
                # Retry budget exhausted without touching the product: release
                # the reservation so the record does not stay `reserved` under
                # a dead task id and starve the same slot in later orders.
                self._release_inventory_reservation(task)
            for candidate in self.tasks:
                if (
                    candidate.status == TaskStatus.PENDING
                    and candidate.metadata.get("search_mode")
                    and candidate.slot_id == task.slot_id
                ):
                    candidate.status = TaskStatus.FAILED
            task.status = TaskStatus.FAILED
            self.preferred_retry_task_id = None
            if "official_order" in task.metadata:
                order = int(task.metadata.get("official_order", 0))
                has_pending_same_order = any(
                    candidate.status == TaskStatus.PENDING
                    and int(candidate.metadata.get("official_order", -1)) == order
                    for candidate in self.tasks
                )
                self.current_official_order = order if has_pending_same_order else None
            return task
        if requeue and task.retry_count < task.max_retries:
            task.status = TaskStatus.PENDING
            self._release_inventory_reservation(task)
            self.preferred_retry_task_id = task.task_id
        else:
            task.status = TaskStatus.FAILED
            self._fail_same_physical_target(task)
            if not requeue:
                self._mark_inventory_state(task, INVENTORY_STATE_DISTURBED)
                self._retire_inventory_after_completion(task, state="disturbed")
            self._release_inventory_reservation(task)
            self.preferred_retry_task_id = None
            if "official_order" in task.metadata:
                self.current_official_order = None
        return task

    def _retire_inventory_after_completion(self, task: PickTask, *, state: str) -> None:
        record = self.inventory_by_aruco.get(int(task.aruco_id))
        if record is None:
            return
        record["state"] = state
        record["confirmed"] = False
        record["hits"] = 0
        record["last_seen"] = time.time()
        record["fresh_at"] = time.time()
        record.pop("reserved_task_id", None)
        record.pop("reserved_at", None)

    def _task_is_selectable(self, task: PickTask) -> bool:
        record = self.inventory_by_aruco.get(int(task.aruco_id))
        if record is None:
            return True
        state = str(record.get("state") or "").strip()
        reserved_task = record.get("reserved_task_id")
        if state in {INVENTORY_STATE_CONSUMED, INVENTORY_STATE_DISTURBED}:
            return False
        if state == INVENTORY_STATE_RESERVED and reserved_task not in {None, task.task_id}:
            return False
        return True

    def _inventory_candidate_score(self, task: PickTask, *, now: float | None = None) -> float:
        record = self.inventory_by_aruco.get(int(task.aruco_id)) or {}
        confidence = float(record.get("confidence", 0.0))
        age = self._inventory_age(record, now=now)
        if age is None:
            freshness = 0.0
        else:
            # PR3: score the pose freshness on the POSE scale (short-lived),
            # so a slot whose identity is long-lived but whose pose went stale
            # ranks lower (needs re-observe) but stays selectable.
            pose_scale = max(0.001, INVENTORY_POSE_MAX_AGE_SEC)
            freshness = max(0.0, 1.0 - min(age, pose_scale) / pose_scale)
        route_cost = self._estimate_roundtrip_cost(task)
        risk = self._task_risk(task)
        score = 100.0
        score += confidence * INVENTORY_CONFIDENCE_WEIGHT
        score += freshness * INVENTORY_FRESHNESS_WEIGHT
        # Normalize the raw metre cost before weighting: without this, a
        # ~12 m round trip contributes ~143 points and drowns out confidence,
        # freshness and risk, reducing the ranking to "nearest first".
        score -= (route_cost / INVENTORY_ROUTE_REFERENCE_M) * INVENTORY_ROUTE_WEIGHT
        score -= risk * INVENTORY_RISK_WEIGHT
        score -= float(task.retry_count) * INVENTORY_RETRY_PENALTY
        if record.get("state") == INVENTORY_STATE_RESERVED and record.get("reserved_task_id") == task.task_id:
            score += INVENTORY_RESERVATION_BONUS
        if self.last_completed_task is not None:
            if task.shelf == self.last_completed_task.shelf:
                score += INVENTORY_SAME_SHELF_BONUS
            if task.level == self.last_completed_task.level:
                score += INVENTORY_SAME_LEVEL_BONUS
        return score

    @staticmethod
    def _task_risk(task: PickTask) -> float:
        level_risk = {
            "L2": 0.00,
            "L3": 0.10,
            "L1": 0.08,
        }.get(str(task.level), 0.12)
        strategy = str(task.grasp_strategy or "")
        if "bottle" in strategy:
            kind_risk = 0.10
        elif "box" in strategy:
            kind_risk = 0.16
        elif "fruit" in strategy:
            kind_risk = 0.22
        elif "pinching" in strategy:
            kind_risk = 0.14
        else:
            kind_risk = 0.12
        return level_risk + kind_risk

    @staticmethod
    def _estimate_roundtrip_cost(task: PickTask) -> float:
        pick = (float(task.navigation_target.x), float(task.navigation_target.y))
        delivery = (float(DEFAULT_DELIVERY_TARGET.x), float(DEFAULT_DELIVERY_TARGET.y))
        return 2.0 * math.dist(pick, delivery)

    def _fail_same_physical_target(self, task: PickTask) -> None:
        """Retire duplicate candidates for an item that is no longer reliable.

        After the gripper touches or drops an item, the public shelf coordinate
        is stale.  Re-selecting another task that points to the same body/slot
        makes the robot grab the empty original location, so blacklist sibling
        candidates for that physical target.
        """
        body = str(task.metadata.get("body") or "")
        for candidate in self.tasks:
            if candidate.task_id == task.task_id or candidate.status != TaskStatus.PENDING:
                continue
            candidate_body = str(candidate.metadata.get("body") or "")
            same_body = bool(body and candidate_body == body)
            same_slot_product = (
                candidate.slot_id == task.slot_id
                and candidate.product_name == task.product_name
            )
            if same_body or same_slot_product:
                candidate.status = TaskStatus.FAILED

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
        level_rank = {"L2": 0, "L3": 1, "L1": 2}
        return sorted(
            slots.values(),
            key=lambda item: (
                float(item["world_position"][0]),
                level_rank.get(str(item["level"]), 99),
                str(item["shelf"]),
                str(item["column"]),
            ),
        )

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
            max_retries=SEARCH_SLOT_MAX_RETRIES,
            metadata={
                'display_slot': slot_id,
                'official_target_id': target_id,
                'search_mode': True,
            },
        )

    def _default_grasp_strategy(self, object_kind: str, level: str) -> str:
        if object_kind in {'kele', 'maidong'}:
            return 'front_bottle_wrap'
        if object_kind == 'zhijin':
            return 'front_short_axis_box_clamp'
        if object_kind in {'sanmingzhi', 'heweidao', 'shupian', 'kouxiangtang'}:
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
