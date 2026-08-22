#!/usr/bin/env python3
"""Decision-aware wrapper client for the Supermarket Sorting task.

This file keeps the original baseline motion pipeline, but inserts the new
TaskManager / OrderScheduler decision layer before execution so we can:
- turn a requested product list into ranked tasks
- log the selected task payload in a structured format
- mark the task lifecycle when the baseline reaches DONE

Current limitation:
The manipulation pipeline uses one front-grasp state machine with per-product
strategy parameters.  For full multi-product runs, use the GT projection
perception backend or a future multi-class detector.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from decision.mission_metrics import MissionMetrics
from decision.task_manager import TaskManager
from manipulation.arm_capabilities import requires_mirrored_left_arm
from supermarket_sorting_client import (
    DEPLOY,
    DONE,
    NAV_SHELF,
    PHASE_NAME,
    REACH_FWD_MAX,
    REACH_FWD_MIN,
    REACH_Z_MAX,
    REACH_Z_MIN,
    PickPlaceClient,
)

TASK_DIR = Path(__file__).resolve().parent
RUNTIME_LAYOUT_JSON = TASK_DIR / "runtime_layout.json"
DECISION_TRACE_PATH = TASK_DIR / "decision_trace.jsonl"
MISSION_METRICS_PATH = TASK_DIR / "mission_metrics.jsonl"
# 官方"无间歇连续作业"奖励(10分)以 ≤15s 停顿为基准。定期记录底盘位姿心跳,
# 由 analyze_mission_metrics.py 统计超时停顿区间。
NAV_HEARTBEAT_INTERVAL = float(os.getenv("SUPERMARKET_NAV_HEARTBEAT_INTERVAL", "5.0"))
# Soft-hold grace: when the gripper measurement disagrees with the referee
# after a placement, wait this long for the referee to record S5 before
# declaring a hard carry-hold (the referee waits for the bottle to settle).
CARRYING_HOLD_GRACE_S = float(os.getenv("SUPERMARKET_CARRYING_HOLD_GRACE_S", "15.0"))


def resolve_layout_path():
    override = os.getenv("SUPERMARKET_RUNTIME_LAYOUT_PATH", "").strip()
    if override:
        override_path = Path(override)
        if override_path.exists():
            return override_path
    if RUNTIME_LAYOUT_JSON.exists():
        return RUNTIME_LAYOUT_JSON
    return None


class DecisionPickPlaceClient(PickPlaceClient):
    def __init__(self):
        super().__init__()
        self.allow_runtime_layout = os.getenv("SUPERMARKET_ALLOW_RUNTIME_LAYOUT", "0") == "1"
        layout_path = resolve_layout_path() if self.allow_runtime_layout else None
        try:
            if layout_path is not None:
                self.get_logger().info(
                    '[decision] using runtime layout for development only: %s' % layout_path
                )
                self.task_manager = TaskManager(layout_path=layout_path)
            else:
                self.get_logger().info(
                    '[decision] using public slot geometry; runtime layout truth disabled'
                )
                self.task_manager = TaskManager()
        except Exception as exc:
            self.get_logger().warn(
                '[decision] failed to load layout (%s); falling back to static layout' % exc
            )
            self.task_manager = TaskManager()
        self.active_task = None
        self.active_payload = None
        self._orders_finished = False
        self._decision_ready = False
        self._waiting_for_official_task = False
        self._task_wait_started = self.now()
        self._last_task_wait_log = 0.0
        self._official_task_payload = None
        self._volatile_task_fallback = False
        self._last_nav_heartbeat = 0.0
        self._trace_path = Path(os.getenv("SUPERMARKET_DECISION_TRACE", DECISION_TRACE_PATH))
        self.metrics = MissionMetrics(os.getenv("SUPERMARKET_MISSION_METRICS", str(MISSION_METRICS_PATH)))
        if os.getenv("SUPERMARKET_APPEND_DECISION_TRACE", "0") != "1":
            # A read-only deployment must not fail at startup just because the
            # trace/metrics files cannot be truncated.
            try:
                self._trace_path.write_text("", encoding="utf-8")
            except OSError as exc:
                self.get_logger().warn(
                    "[decision] cannot truncate decision trace: %s" % exc)
            try:
                self.metrics.reset_file()
            except OSError as exc:
                self.get_logger().warn(
                    "[decision] cannot reset metrics file: %s" % exc)
        self._setup_decision_layer()
        # NOTE: the inventory subscription lives in the BASE client (single
        # subscription).  Python binds self.inventory_observation_cb to THIS
        # subclass method, so the base class subscription invokes the decision
        # layer here.  Adding another subscription here (PR5: removed) would
        # deliver each message twice and double-count hits, making
        # INVENTORY_MIN_HITS=2 behave like a one-frame confirmation.

    def inventory_observation_cb(self, msg: String) -> None:
        try:
            observations = json.loads(msg.data).get("observations", [])
        except (json.JSONDecodeError, AttributeError):
            return
        try:
            accepted = self.task_manager.register_inventory_observations(observations)
            # PR5: the client ArUco kind gate must only reflect CONFIRMED
            # inventory (the base class version of this callback is shadowed
            # here, so update the map explicitly).
            try:
                for aruco_id, record in self.task_manager.inventory_by_aruco.items():
                    if record.get("confirmed"):
                        self.inventory_aruco_kind[int(aruco_id)] = str(record.get("kind", ""))
            except Exception:  # noqa: BLE001
                pass
            if accepted:
                self.get_logger().info(f"[inventory] accepted {accepted} ArUco-bound observations")
            # A RESERVED record refreshes only the short-lived grasp pose and
            # therefore does not increment ``accepted``.  Still run the
            # active-task refresh path on every schema-v3 frame.
            self._refresh_active_task_from_inventory()
        except Exception as exc:  # noqa: BLE001 - a malformed observation must not kill the node
            self.get_logger().warn(f"[inventory] observation handling failed: {exc}")

    def _refresh_active_task_from_inventory(self) -> None:
        """Use a newly confirmed slot pose before the arm locks a grasp target."""
        if self.active_task is None or self.target_locked:
            return
        if not getattr(self.active_task, "metadata", {}).get("search_mode"):
            return
        if getattr(self.active_task, "metadata", {}).get("inventory_confirmed"):
            # The first confirmed observation establishes the physical slot.
            # Keep that pose fixed for this attempt: continuing to blend
            # RGB-D centres while the base is parking changes the endpoint
            # underneath the arm and creates a late, unstable replan.
            return
        if not self.task_manager.apply_inventory_observation(self.active_task):
            return
        # The first inventory binding can change the required standoff by more
        # than 25 cm (for example, a tissue top-clamp versus a bottle front
        # grasp).  It is still safe to re-park during the beginning of DEPLOY:
        # the arm has not moved and no object frame has been locked.  Keeping
        # the old generic search-slot parking pose here made the chassis creep
        # into the shelf and then time out before the pinch point.
        can_repark_before_arm_motion = (
            self.phase == DEPLOY
            and not self.deploy_set
            and not self.target_locked
        )
        can_replan_route = (
            self.base_xy is not None
            and (self.phase == NAV_SHELF or can_repark_before_arm_motion)
        )
        if can_replan_route:
            self.configure_pick_task(self.active_task)
            if can_repark_before_arm_motion:
                self.set_twist(0.0, 0.0)
                self.prepare_inventory_repark()
                self.phase = NAV_SHELF
                self.state_t0 = self.now()
                self.get_logger().info(
                    "[inventory] category changed shelf standoff before arm motion; "
                    "returning to nav->shelf for a bounded same-slot re-park"
                )
        else:
            self.active_product_name = str(self.active_task.product_name)
            self.active_task_level = str(self.active_task.level)
            self.grasp_profile = self.profile_for_task(self.active_task)
            self.expected_object_world = np.asarray(
                self.active_task.world_position, dtype=float
            )
            self.search_slot_world = self.expected_object_world.copy()
            self.get_logger().info(
                "[inventory] applied late pose without resetting active route/arm phase"
            )
        self.active_payload = self.task_manager.build_execution_payload(self.active_task)
        self.get_logger().info(
            "[inventory] refreshed active task from confirmed ArUco inventory: %s"
            % json.dumps(self.active_payload, ensure_ascii=False)
        )
        self._trace("inventory_pose_applied", self.active_payload)

    def _fresh_grasp_platform_stable(self) -> bool:
        """Require a parked chassis and settled head/slide before pose lock."""
        if self.phase != DEPLOY or self.deploy_set:
            return False
        if abs(float(self.cur_lin)) > 0.01 or abs(float(self.cur_ang)) > 0.02:
            return False
        if abs(float(self.des_lin)) > 0.01 or abs(float(self.des_ang)) > 0.02:
            return False
        velocities = self.jvel or {}
        for joint in ("slide_joint", "head_yaw_joint", "head_pitch_joint"):
            try:
                if abs(float(velocities.get(joint, 0.0))) > 0.03:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def det_cb(self, msg) -> None:
        """Do not let a bare Detection3D lock an anonymous official slot.

        The generic topic has no ArUco id.  It remains useful as a heartbeat
        and, after a strict lock, as a displacement monitor; the initial grasp
        lock comes only from schema-v3 ArUco-bound observations.
        """
        if self.active_search_mode() and not self.target_locked:
            self.detection_stream_seen_at = self.now()
            return
        super().det_cb(msg)

    def _lock_target(self):
        if not self.active_search_mode():
            return super()._lock_target()
        if self.active_task is None or self.active_product_name == "__search__":
            return False
        if not self._fresh_grasp_platform_stable():
            return False
        if not self.task_manager.apply_fresh_grasp_observation(self.active_task):
            if self.now() - self.last_wait_log > 1.0:
                self.get_logger().info(
                    "[fresh_grasp] waiting for 3 current, stable frames bound to "
                    f"ArUco {self.active_task.aruco_id} and {self.active_product_name}"
                )
                self.last_wait_log = self.now()
            return False

        surface_world = np.asarray(self.active_task.fresh_grasp_world, dtype=float)
        # The schema-v3 RGB-D point is the visible package surface.  Reuse the
        # per-product forward calibration used by the generic detection path,
        # but keep Z from TaskManager: it has already fused the ArUco slot
        # level with the product half-height and must not receive the legacy
        # surface-to-centre Z correction a second time.
        object_world = self._vision_to_object_center(surface_world)
        object_world[2] = surface_world[2]
        marker_world = self.active_task.metadata.get("fresh_grasp_marker_world")
        anchored_world = self.task_manager.anchor_grasp_center_depth_to_marker(
            aruco_id=self.active_task.aruco_id,
            center_world=tuple(float(value) for value in object_world),
            marker_world=(
                tuple(float(value) for value in marker_world)
                if marker_world is not None
                else None
            ),
        )
        object_world = np.asarray(anchored_world, dtype=float)
        fp = self.world_to_footprint(object_world)
        reach_lateral_max = float(self.grasp_profile.get("reach_lateral_max", 0.35))
        if (
            fp[0] < REACH_FWD_MIN
            or fp[0] > REACH_FWD_MAX
            or abs(float(fp[1])) > reach_lateral_max
            or object_world[2] < REACH_Z_MIN
            or object_world[2] > REACH_Z_MAX
        ):
            self.get_logger().warn(
                "[fresh_grasp] current ArUco-bound pose is unreachable; "
                f"world={np.round(object_world, 3)} fp={np.round(fp, 3)}"
            )
            return False
        if not self._neighbor_clearance_ok(object_world):
            return False

        # Update only the grasp endpoint.  navigation_target and
        # navigation_world_position remain frozen from the first inventory
        # confirmation, so a late RGB-D correction cannot move the chassis.
        self.expected_object_world = object_world.copy()
        self.search_slot_world = object_world.copy()
        self.active_task.world_position = tuple(float(value) for value in object_world)
        self.active_task.fresh_grasp_world = self.active_task.world_position
        self.active_task.metadata["fresh_grasp_surface_world"] = [
            float(value) for value in surface_world
        ]
        self.active_task.metadata["fresh_grasp_world"] = [
            float(value) for value in object_world
        ]
        self.active_payload = self.task_manager.build_execution_payload(self.active_task)
        self.lock_grasp_geometry(object_world, source="aruco-v3")
        self.get_logger().info(
            "[fresh_grasp] locked active slot from schema v3: "
            f"aruco={self.active_task.aruco_id} kind={self.active_product_name} "
            f"world={np.round(object_world, 3)} "
            f"std={self.active_task.metadata.get('fresh_grasp_pose_std')}"
        )
        self._trace("fresh_grasp_pose_locked", self.active_payload)
        return True

    def _setup_decision_layer(self) -> None:
        order_spec = os.getenv('SUPERMARKET_ORDER', 'official').strip()
        if order_spec.lower() in {'official', 'task', 'task_topic', 'auto'}:
            qos = QoSProfile(depth=1)
            qos.reliability = ReliabilityPolicy.RELIABLE
            qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self.create_subscription(
                String,
                "/supermarket_sorting/task",
                self.official_task_cb,
                qos,
            )
            self._waiting_for_official_task = True
            self._task_wait_started = self.now()
            self.get_logger().info(
                '[decision] waiting for official /supermarket_sorting/task JSON; '
                'set SUPERMARKET_ORDER=all for old local stress tests'
            )
            return

        self._build_legacy_order_plan(order_spec, reason='env SUPERMARKET_ORDER')

    def _build_legacy_order_plan(self, order_spec: str, *, reason: str) -> None:
        products = [item.strip() for item in order_spec.split(',') if item.strip()]
        if not products:
            products = ['all']

        tasks = self.task_manager.build_tasks_for_products(products)

        self.get_logger().info(
            '[decision] requested products: %s' % ', '.join(products)
        )
        self.get_logger().info('[decision] built candidate tasks: %d' % len(tasks))
        self.get_logger().info(
            '[decision] initial ranked plan: %s' %
            json.dumps(self.task_manager.export_plan(), ensure_ascii=False)
        )
        self._trace("plan_built", {
            "source": reason,
            "requested_products": products,
            "candidate_count": len(tasks),
            "ranked_plan": self.task_manager.export_plan(),
        })
        self._decision_ready = True
        self._select_next_task('initial selection')

    def official_task_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn('[decision] invalid official task JSON: %s' % exc)
            return
        if not isinstance(payload, dict):
            self.get_logger().warn(
                '[decision] official task payload is not an object: %r' % (payload,))
            return
        targets = payload.get("targets") or []
        if not isinstance(targets, list) or not targets:
            self.get_logger().warn('[decision] official task has no targets: %s' % msg.data)
            return
        try:
            signature = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            if self._official_task_payload == signature and self._decision_ready:
                return
            self._official_task_payload = signature
            self._waiting_for_official_task = False
            self._orders_finished = False
            tasks = self.task_manager.build_search_tasks_for_targets(
                targets,
                trust_layout_positions=self.allow_runtime_layout,
            )
            # PR5: a new run re-randomises products under the fixed tags, so
            # the client ArUco kind gate (and the manager inventory) must be
            # reset together; last run's identity table is void.
            self.inventory_aruco_kind.clear()
        except Exception as exc:  # noqa: BLE001 - a malformed task must not kill the node
            self.get_logger().error(
                '[decision] failed to apply official task: %s' % exc)
            return
        anonymous_targets = any(str(target.get("id", "")).startswith("item_") for target in targets)
        direct_count = sum(
            bool(task.metadata.get("official_direct"))
            for task in tasks
        )
        self.get_logger().info(
            '[decision] official task received: count=%s targets=%s' %
            (payload.get("count", len(targets)), json.dumps(targets, ensure_ascii=False))
        )
        self.get_logger().info(
            '[decision] built target plan: %d tasks (%d direct-location, %d physical-slot search); '
            'scan order is left-to-right, middle/upper/lower; head=%s' %
            (
                len(tasks),
                direct_count,
                len(tasks) - direct_count,
                json.dumps(self.task_manager.export_plan()[:6], ensure_ascii=False),
            )
        )
        if anonymous_targets and not self.allow_runtime_layout:
            self.get_logger().warn(
                '[decision] anonymous official task received with runtime layout disabled; '
                'a legal run requires a multi-class product detector; '
                'runtime layout and gt are development-only diagnostics'
            )
        self._trace("official_task_received", {
            "payload": payload,
            "candidate_count": len(tasks),
            "ranked_plan_head": self.task_manager.export_plan()[:12],
        })
        self._decision_ready = True
        self._select_next_task('official task received')

    def _trace(self, event: str, payload: dict) -> None:
        record = {
            "time": self.now(),
            "event": event,
            "payload": payload,
        }
        try:
            with self._trace_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.get_logger().warn("[decision] trace write failed: %s" % exc)

    def _metric(self, event: str, payload: dict | None = None) -> None:
        try:
            self.metrics.event(self.now(), event, payload or {})
        except OSError as exc:
            self.get_logger().warn("[metrics] write failed: %s" % exc)

    def _next_executable_decision(self):
        """Retire tasks unsupported by the current right-arm-only executor."""
        decision = self.task_manager.next_decision()
        while decision.selected_task is not None and requires_mirrored_left_arm(
            decision.selected_task
        ):
            task = self.task_manager.mark_task_failed(
                decision.selected_task.task_id,
                requeue=False,
            )
            reason = (
                "top-right L3 tissue slot requires the pending mirrored left-arm strategy; "
                "skipping before navigation"
            )
            self.get_logger().warn(
                "[decision] skipping unsupported task before navigation: %s" % task.task_id
            )
            self._trace("task_skipped_unsupported", {
                "task": task.to_dict(),
                "reason": reason,
            })
            self._metric("task_skipped_unsupported", {
                "task": task.to_dict(),
                "reason": reason,
            })
            decision = self.task_manager.next_decision()
        return decision

    def _select_next_task(self, reason: str) -> None:
        decision = self._next_executable_decision()
        self.active_task = decision.selected_task
        self.get_logger().info(
            '[decision] next decision (%s): %s' %
            (reason, json.dumps(decision.to_dict(), ensure_ascii=False))
        )
        self._trace("decision_selected", {
            "reason": reason,
            "decision": decision.to_dict(),
        })
        self._metric("decision_selected", {
            "reason": reason,
            "decision": decision.to_dict(),
        })

        if self.active_task is None:
            self.active_payload = None
            self._orders_finished = True
            self.set_twist(0.0, 0.0)
            self.phase = DONE
            self.get_logger().info('[decision] all available tasks finished; client will idle')
            return

        self.reset_for_next_pick()
        self.configure_pick_task(self.active_task)
        self.active_payload = self.task_manager.build_execution_payload(self.active_task)
        self._metric("task_started", self.active_payload)
        self.get_logger().info(
            '[decision] execution payload: %s' % json.dumps(self.active_payload, ensure_ascii=False)
        )

        if self.active_task.product_name != 'kele':
            self.get_logger().info(
                '[decision] non-kele task selected; using strategy=%s. '
                'Use perception backend gt or a multi-class detector for best reliability.'
                % self.active_task.grasp_strategy
            )

    def tick(self):
        if not self._decision_ready:
            if (
                self._waiting_for_official_task
                and self.now() - self._task_wait_started
                > float(os.getenv("SUPERMARKET_TASK_WAIT_TIMEOUT", "8.0"))
            ):
                fallback = os.getenv("SUPERMARKET_TASK_FALLBACK_ORDER", "").strip()
                if fallback:
                    self.get_logger().warn(
                        '[decision] official task timeout; falling back to %s' % fallback
                    )
                    self._waiting_for_official_task = False
                    self._build_legacy_order_plan(fallback, reason='task timeout fallback')
                else:
                    hard_timeout = float(os.getenv(
                        "SUPERMARKET_TASK_HARD_TIMEOUT", "90.0"))
                    if (
                        self._waiting_for_official_task
                        and not self._volatile_task_fallback
                        and self.now() - self._task_wait_started > hard_timeout
                    ):
                        # The latched TRANSIENT_LOCAL subscription never
                        # matched (e.g. a server publishing with VOLATILE
                        # durability). Add a volatile fallback subscription so
                        # the task can still be received; signature dedup makes
                        # the double subscription safe.
                        self.get_logger().error(
                            '[decision] /supermarket_sorting/task still missing '
                            'after %.0fs; adding a VOLATILE-QoS fallback '
                            'subscription in case the server publishes without '
                            'TRANSIENT_LOCAL durability' % hard_timeout)
                        self._volatile_task_fallback = True
                        volatile_qos = QoSProfile(depth=1)
                        volatile_qos.reliability = ReliabilityPolicy.RELIABLE
                        volatile_qos.durability = DurabilityPolicy.VOLATILE
                        self.create_subscription(
                            String,
                            "/supermarket_sorting/task",
                            self.official_task_cb,
                            volatile_qos,
                        )
                    else:
                        # Do NOT reset _task_wait_started here: that would push
                        # the 90 s VOLATILE-QoS fallback forever and turn it
                        # into dead code.  Rate-limit the log separately.
                        if self.now() - self._last_task_wait_log > 8.0:
                            self.get_logger().info(
                                '[decision] still waiting for /supermarket_sorting/task; '
                                'old mirror users can set SUPERMARKET_ORDER=all'
                            )
                            self._last_task_wait_log = self.now()
            self.set_twist(0.0, 0.0)
            if self.base_xy is not None and self.jpos is not None:
                self.ramp_twist()
                self.smooth_step()
                self.publish()
            return

        super().tick()
        try:
            if self.base_xy is not None:
                self.metrics.observe_nav_pose(self.base_xy)
            self.metrics.observe_phase(
                self.now(),
                PHASE_NAME.get(self.phase, str(self.phase)),
                {
                    "task_id": None if self.active_task is None else self.active_task.task_id,
                    "product": None if self.active_task is None else self.active_task.product_name,
                    "failure_reason": self.failure_reason,
                },
            )
        except Exception as exc:  # noqa: BLE001 - metrics must never crash the controller
            self.get_logger().warn("[metrics] observation failed: %s" % exc)
        try:
            now = self.now()
            if now - self._last_nav_heartbeat >= NAV_HEARTBEAT_INTERVAL:
                self._last_nav_heartbeat = now
                self.metrics.event(now, "nav_heartbeat", {
                    "x": None if self.base_xy is None else float(self.base_xy[0]),
                    "y": None if self.base_xy is None else float(self.base_xy[1]),
                    "phase": PHASE_NAME.get(self.phase, str(self.phase)),
                })
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn("[metrics] heartbeat failed: %s" % exc)

        if self.phase == DONE and self.active_task is not None and not self._orders_finished:
            if self.execution_failed:
                reason = self.failure_reason or ""
                dropped = (
                    reason.startswith("carried object dropped")
                    or self.has_active_drop_report()
                )
                toppled = (
                    "target toppled or dropped" in reason
                    or "toppled" in reason
                )
                stale_or_skip_target = any(
                    marker in reason
                    for marker in (
                        "skip this item",
                        "skip fallen shelf item",
                        "target was already touched",
                        "not stable enough now",
                        # A shelf post blocks this arm's physical approach.
                        # Requeuing it only makes the scheduler revisit the
                        # identical unreachable slot until its retry budget
                        # is exhausted.
                        "blocked by a shelf post",
                        "requires the pending mirrored left-arm strategy",
                    )
                )
                # "may have moved" is NOT in the skip list: an empty close
                # ("referee did not confirm S3 grasp; target may have moved")
                # usually means the fingers closed on nothing while the bottle
                # is still standing in its slot.  The user observed exactly
                # this (visual11: right finger shoved the bottle a few cm,
                # fingers closed empty, robot froze on a count=1 run).  With
                # the refined close-distance gate the same slot is usually
                # graspable on a retry, so requeue it and try again instead of
                # abandoning the only task.  A genuinely toppled bottle is
                # still caught by `toppled` / the referee drop report.
                stale_or_skip_target = stale_or_skip_target or (
                    "may have moved" in reason
                    and self.active_target_knocked_or_dropped()
                )
                delivery_exhausted = reason.startswith("delivery navigation recovery limit exceeded")
                # Once S3 is confirmed the object may still be physically in the
                # gripper. Only continue to another task when the referee/drop
                # logic says the object is already gone; otherwise hold safely.
                if self.grasp_was_confirmed and not dropped:
                    if self.has_active_drop_report():
                        # The referee confirmed the carried object is already
                        # gone. Release the safety hold and settle this task as
                        # failed/dropped instead of freezing for the whole run.
                        self.get_logger().info(
                            '[decision] referee confirmed the carried object '
                            'dropped; releasing the carry hold')
                        dropped = True
                    elif int(self.referee_state.get("completed", 0)) > self.completed_before_task:
                        # The referee completed this target (S5) even though our
                        # gripper measurement disagrees (e.g. the bottle slipped
                        # out during the release and the close command lagged:
                        # count5_full_v35 froze here for the whole match after a
                        # "placement gripper did not open" while the referee had
                        # already scored S5).  The object is gone - do not hold
                        # the match hostage.
                        self.get_logger().info(
                            '[decision] referee completed this target despite '
                            'gripper disagreement; releasing the carry hold')
                        dropped = True
                    else:
                        # The gripper measurement disagrees with the referee.
                        # The referee may simply be slower to record S5 (it
                        # waits for the bottle to settle / speed<0.02), so an
                        # immediate completed check (added round 85) was still
                        # too early and the match froze (v38: "execution
                        # stopped while carrying" 5 s after the release while
                        # the referee scored S5 shortly after).  Soft-hold:
                        # re-check the referee for a grace period; only a
                        # persistent disagreement becomes a hard hold.
                        now = self.now()
                        if getattr(self, "_carrying_hold_since", None) is None:
                            self._carrying_hold_since = now
                        if int(self.referee_state.get("completed", 0)) > self.completed_before_task:
                            self._carrying_hold_since = None
                            self.get_logger().info(
                                '[decision] referee completed this target during the '
                                'carry-hold grace; releasing the carry hold')
                            dropped = True
                        elif now - self._carrying_hold_since > CARRYING_HOLD_GRACE_S:
                            self._carrying_hold_since = None
                            self.get_logger().error(
                                '[decision] execution stopped while carrying a confirmed target; '
                                'holding position instead of selecting another pick task'
                            )
                            self._trace("carrying_failure_hold", {
                                "task": self.active_task.to_dict(),
                                "reason": self.failure_reason,
                                "referee_state": self.referee_state,
                            })
                            self._orders_finished = True
                            self.set_twist(0.0, 0.0)
                            return
                        else:
                            # keep waiting for the referee to settle S5; do
                            # not select another task yet.
                            self.set_twist(0.0, 0.0)
                            return
                task = self.task_manager.mark_task_failed(
                    self.active_task.task_id,
                    requeue=not (dropped or toppled or stale_or_skip_target or delivery_exhausted),
                )
                self.get_logger().warn(
                    '[decision] task failed: %s reason=%s' %
                    (json.dumps(task.to_dict(), ensure_ascii=False), self.failure_reason)
                )
                self._trace("task_failed", {
                    "task": task.to_dict(),
                    "reason": self.failure_reason,
                })
                self._metric("task_failed", {
                    "task": task.to_dict(),
                    "reason": self.failure_reason,
                    "summary": self.metrics.summary(),
                })
                self._select_next_task('failure recovery')
            else:
                referee_verified = bool(self.test_oracle_enabled)
                try:
                    task = self.task_manager.mark_task_succeeded(
                        self.active_task.task_id,
                        referee_verified=referee_verified,
                    )
                except RuntimeError as exc:
                    # e.g. an unbound `__search__` slot reached DONE. It must
                    # never crash the tick callback; settle it as a failure.
                    self.get_logger().error(
                        '[decision] task completion rejected: %s' % exc)
                    self.failure_reason = 'unbound search task reached DONE'
                    self.execution_failed = True
                    self.set_twist(0.0, 0.0)
                    return
                completion_event = (
                    "task_succeeded" if referee_verified else "task_local_complete"
                )
                verification_note = (
                    "referee confirmed S5"
                    if referee_verified
                    else "local execution complete; official S5 is not observable in this mode"
                )
                self.get_logger().info(
                    '[decision] %s: %s' % (
                        verification_note,
                        json.dumps(task.to_dict(), ensure_ascii=False),
                    )
                )
                self._trace(completion_event, {"task": task.to_dict()})
                self._metric(completion_event, {
                    "task": task.to_dict(),
                    "summary": self.metrics.summary(),
                })
                self._select_next_task('previous task succeeded')


def main():
    rclpy.init()
    node = None
    try:
        node = DecisionPickPlaceClient()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop_robot()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
