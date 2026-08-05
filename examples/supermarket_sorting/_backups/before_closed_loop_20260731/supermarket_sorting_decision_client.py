#!/usr/bin/env python3
"""Decision-aware wrapper client for the Supermarket Sorting task.

This file keeps the original baseline motion pipeline, but inserts the new
TaskManager / OrderScheduler decision layer before execution so we can:
- turn a requested product list into ranked tasks
- log the selected task payload in a structured format
- mark the task lifecycle when the baseline reaches DONE

Current limitation:
The underlying perception and manipulation pipeline is still the official
single-class `kele` baseline. So this wrapper is a safe first integration,
not the final multi-product autonomous solution yet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import rclpy

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
        layout_path = resolve_layout_path()
        try:
            if layout_path is not None:
                self.get_logger().info('[decision] using runtime layout: %s' % layout_path)
                self.task_manager = TaskManager(layout_path=layout_path)
            else:
                self.get_logger().warn(
                    '[decision] runtime layout not found; falling back to static layout'
                )
                self.task_manager = TaskManager()
        except Exception as exc:
            self.get_logger().warn(
                '[decision] failed to load runtime layout (%s); falling back to static layout' % exc
            )
            self.task_manager = TaskManager()
        self.active_task = None
        self.active_payload = None
        self._orders_finished = False
        self._trace_path = Path(os.getenv("SUPERMARKET_DECISION_TRACE", DECISION_TRACE_PATH))
        if os.getenv("SUPERMARKET_APPEND_DECISION_TRACE", "0") != "1":
            self._trace_path.write_text("", encoding="utf-8")
        self._setup_decision_layer()

    def _setup_decision_layer(self) -> None:
        order_spec = os.getenv('SUPERMARKET_ORDER', 'kele')
        products = [item.strip() for item in order_spec.split(',') if item.strip()]
        if not products:
            products = ['kele']

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
            "requested_products": products,
            "candidate_count": len(tasks),
            "ranked_plan": self.task_manager.export_plan(),
        })
        self._select_next_task('initial selection')

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

        self.active_payload = self.task_manager.build_execution_payload(self.active_task)
        self.configure_pick_task(self.active_task)
        self.reset_for_next_pick()
        self.get_logger().info(
            '[decision] execution payload: %s' % json.dumps(self.active_payload, ensure_ascii=False)
        )

        if self.active_task.product_name != 'kele':
            self.get_logger().warn(
                '[decision] current perception still only supports kele; '
                'non-kele tasks are planned but not yet executable'
            )

    def tick(self):
        super().tick()

        if self.phase == DONE and self.active_task is not None and not self._orders_finished:
            if self.execution_failed:
                task = self.task_manager.mark_task_failed(self.active_task.task_id)
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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
