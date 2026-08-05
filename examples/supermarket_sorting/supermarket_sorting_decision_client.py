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

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from decision.task_manager import TaskManager
from supermarket_sorting_client import DONE, PickPlaceClient

TASK_DIR = Path(__file__).resolve().parent
RUNTIME_LAYOUT_JSON = TASK_DIR / "runtime_layout.json"
DECISION_TRACE_PATH = TASK_DIR / "decision_trace.jsonl"


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
        self._official_task_payload = None
        self._trace_path = Path(os.getenv("SUPERMARKET_DECISION_TRACE", DECISION_TRACE_PATH))
        if os.getenv("SUPERMARKET_APPEND_DECISION_TRACE", "0") != "1":
            self._trace_path.write_text("", encoding="utf-8")
        self._setup_decision_layer()

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
        targets = payload.get("targets") or []
        if not isinstance(targets, list) or not targets:
            self.get_logger().warn('[decision] official task has no targets: %s' % msg.data)
            return
        signature = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        if self._official_task_payload == signature and self._decision_ready:
            return
        self._official_task_payload = signature
        self._waiting_for_official_task = False
        self._orders_finished = False
        tasks = self.task_manager.build_search_tasks_for_targets(targets)
        direct_count = sum(
            bool(task.metadata.get("official_direct"))
            for task in tasks
        )
        self.get_logger().info(
            '[decision] official task received: count=%s targets=%s' %
            (payload.get("count", len(targets)), json.dumps(targets, ensure_ascii=False))
        )
        self.get_logger().info(
            '[decision] built target plan: %d tasks (%d direct official, %d search fallback); '
            'order is locked until success/retry exhaustion; head=%s' %
            (
                len(tasks),
                direct_count,
                len(tasks) - direct_count,
                json.dumps(self.task_manager.export_plan()[:6], ensure_ascii=False),
            )
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

    def _select_next_task(self, reason: str) -> None:
        decision = self.task_manager.next_decision()
        self.active_task = decision.selected_task
        self.get_logger().info(
            '[decision] next decision (%s): %s' %
            (reason, json.dumps(decision.to_dict(), ensure_ascii=False))
        )
        self._trace("decision_selected", {
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
        self.active_payload = self.task_manager.build_execution_payload(self.active_task)
        self.configure_pick_task(self.active_task)
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
                    self.get_logger().info(
                        '[decision] still waiting for /supermarket_sorting/task; '
                        'old mirror users can set SUPERMARKET_ORDER=all'
                    )
                    self._task_wait_started = self.now()
            self.set_twist(0.0, 0.0)
            if self.base_xy is not None and self.jpos is not None:
                self.ramp_twist()
                self.smooth_step()
                self.publish()
            return

        super().tick()

        if self.phase == DONE and self.active_task is not None and not self._orders_finished:
            if self.execution_failed:
                dropped = (
                    self.failure_reason.startswith("carried object dropped")
                    or self.has_active_drop_report()
                )
                toppled = (
                    "target toppled or dropped" in self.failure_reason
                    or "toppled" in self.failure_reason
                )
                delivery_exhausted = self.failure_reason.startswith("delivery navigation recovery limit exceeded")
                # Once S3 is confirmed the object may still be physically in the
                # gripper. Only continue to another task when the referee/drop
                # logic says the object is already gone; otherwise hold safely.
                if self.grasp_was_confirmed and not dropped:
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
                task = self.task_manager.mark_task_failed(
                    self.active_task.task_id,
                    requeue=not (dropped or toppled or delivery_exhausted),
                )
                self.get_logger().warn(
                    '[decision] task failed: %s reason=%s' %
                    (json.dumps(task.to_dict(), ensure_ascii=False), self.failure_reason)
                )
                self._trace("task_failed", {
                    "task": task.to_dict(),
                    "reason": self.failure_reason,
                })
                self._select_next_task('failure recovery')
            else:
                task = self.task_manager.mark_task_succeeded(self.active_task.task_id)
                self.get_logger().info(
                    '[decision] task finished: %s' %
                    json.dumps(task.to_dict(), ensure_ascii=False)
                )
                self._trace("task_succeeded", {"task": task.to_dict()})
                self._select_next_task('previous task succeeded')


def main():
    rclpy.init()
    node = DecisionPickPlaceClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
