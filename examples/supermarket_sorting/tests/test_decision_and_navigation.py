"""Small dependency-free regression tests for the competition control logic."""

from __future__ import annotations

import os
import time
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

CURRENT_FILE = Path(__file__).resolve()
PACKAGE_ROOT = CURRENT_FILE.parents[1]
REPO_ROOT = CURRENT_FILE.parents[3]
for _path in (str(PACKAGE_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from manipulation.grasp_pose import finger_closing_axis, grasp_rotation_for_strategy
    from manipulation.arm_capabilities import requires_mirrored_left_arm
    from decision.models import NavigationTarget, PickTask
    from decision.task_manager import SEARCH_PRODUCT, TaskManager
    from navigation.grid_planner import SupermarketGridPlanner
    from perception.backends import stable_class_consensus
    from perception.inventory import associate_detections_to_markers
except ModuleNotFoundError:
    # Allow `python -m unittest ...` from the repository root as well as the
    # container scripts, which add examples/supermarket_sorting to PYTHONPATH.
    from examples.supermarket_sorting.manipulation.grasp_pose import (
        finger_closing_axis,
        grasp_rotation_for_strategy,
    )
    from examples.supermarket_sorting.manipulation.arm_capabilities import (
        requires_mirrored_left_arm,
    )
    from examples.supermarket_sorting.decision.models import NavigationTarget, PickTask
    from examples.supermarket_sorting.decision.task_manager import SEARCH_PRODUCT, TaskManager
    from examples.supermarket_sorting.navigation.grid_planner import SupermarketGridPlanner
    from examples.supermarket_sorting.perception.backends import stable_class_consensus
    from examples.supermarket_sorting.perception.inventory import associate_detections_to_markers


class TaskManagerTests(unittest.TestCase):
    @staticmethod
    def _manual_search_task(
        task_id: str,
        aruco_id: int,
        *,
        nav_x: float,
        nav_y: float,
        level: str = "L2",
        shelf: str = "A",
        column: str = "C1",
        world_x: float = 0.73,
        world_y: float = 3.23,
        world_z: float = 0.94,
    ) -> PickTask:
        slot_id = f"slot_{shelf}_{level}_{column}"
        return PickTask(
            task_id=task_id,
            product_name=SEARCH_PRODUCT,
            slot_id=slot_id,
            aruco_id=aruco_id,
            shelf=shelf,
            level=level,
            column=column,
            world_position=(world_x, world_y, world_z),
            navigation_target=NavigationTarget(frame_id="map", x=nav_x, y=nav_y, yaw=1.57),
            grasp_strategy="front_center",
            metadata={"search_mode": True},
        )

    def test_anonymous_targets_scan_each_physical_slot_once(self):
        manager = TaskManager()
        tasks = manager.build_search_tasks_for_targets([
            {"id": "item_01", "kind": "kele"},
            {"id": "item_02", "kind": "maidong"},
            {"id": "item_03", "kind": "kele"},
        ])

        self.assertEqual(len(tasks), len({task.slot_id for task in tasks}))
        self.assertTrue(all(task.product_name == SEARCH_PRODUCT for task in tasks))
        self.assertEqual(
            [task.slot_id for task in manager.scheduler.rank_tasks(tasks)[:3]],
            ["slot_A_L2_C1", "slot_A_L3_C1", "slot_A_L1_C1"],
        )

    def test_all_product_order_preserves_layout_order(self):
        manager = TaskManager()
        tasks = manager.build_tasks_for_products(["all"])
        expected = [
            str(item["object_kind"])
            for item in manager.layout_items
            if item.get("object_kind")
        ]

        self.assertEqual([task.product_name for task in tasks], expected)
        self.assertEqual(sum(manager.requested_counts.values()), len(tasks))

    def test_zhijin_uses_short_axis_box_clamp_strategy(self):
        manager = TaskManager()
        self.assertEqual(
            manager._default_grasp_strategy("zhijin", "L2"),
            "front_short_axis_box_clamp",
        )

    def test_short_axis_box_clamp_closes_across_tissue_depth(self):
        rot = grasp_rotation_for_strategy(
            "front_short_axis_box_clamp",
            {
                "wrist_pitch_deg": 90.0,
                "wrist_roll_deg": 0.0,
                "wrist_yaw_deg": 90.0,
            },
        )
        expected = Rotation.from_euler("z", 90.0, degrees=True).as_matrix() @ Rotation.from_euler(
            "y", 90.0, degrees=True
        ).as_matrix()
        self.assertTrue(np.allclose(rot, expected))
        # The product box is 172 mm along shelf X and only 85 mm in depth.
        # With the base facing north, footprint -X becomes world -Y.
        self.assertTrue(np.allclose(finger_closing_axis(rot), [-1.0, 0.0, 0.0]))

    def test_inventory_bound_top_grasp_parks_within_reachable_standoff(self):
        manager = TaskManager()
        task = self._manual_search_task(
            "search_zhijin",
            3,
            nav_x=0.6,
            nav_y=2.475,
            world_x=-1.955,
            world_y=3.243,
            world_z=0.895,
        )
        manager.tasks = [task]
        manager.requested_counts["zhijin"] = 1
        observation = {
            "aruco_id": 3,
            "kind": "zhijin",
            "confidence": 0.9,
            "world": (-1.95, 3.21, 0.92),
        }
        manager.register_inventory_observations([observation])
        manager.register_inventory_observations([observation])

        self.assertTrue(manager.apply_inventory_observation(task))
        self.assertEqual(task.grasp_strategy, "front_short_axis_box_clamp")
        self.assertGreater(task.navigation_target.y, 2.50)
        self.assertLessEqual(task.navigation_target.y, 2.70)

    def test_top_box_inventory_binding_changes_the_search_standoff(self):
        """A late class binding must not reuse the generic shelf stop pose."""
        manager = TaskManager()
        task = self._manual_search_task(
            "search_zhijin",
            3,
            nav_x=0.6,
            nav_y=2.475,
            world_x=-1.955,
            world_y=3.243,
            world_z=0.895,
        )
        manager.tasks = [task]
        manager.requested_counts["zhijin"] = 1
        observation = {
            "aruco_id": 3,
            "kind": "zhijin",
            "confidence": 0.9,
            "world": (-1.95, 3.21, 0.92),
        }
        manager.register_inventory_observations([observation])
        manager.register_inventory_observations([observation])

        self.assertTrue(manager.apply_inventory_observation(task))
        self.assertGreater(
            task.navigation_target.y,
            2.475,
            "top clamp must park closer than the generic anonymous-slot stop",
        )

    def test_referee_binding_uses_actual_body_kind(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([
            {"id": "item_01", "kind": "maidong"},
        ])
        active = manager.next_decision().selected_task
        manager.bind_search_task_product(active.task_id, "maidong")
        bound = manager.rebind_active_task_to_referee_body(
            "slot_A_L2_C2_maidong", active.task_id)

        self.assertIs(bound, active)
        self.assertEqual(bound.product_name, "maidong")
        self.assertEqual(bound.metadata["body"], "slot_A_L2_C2_maidong")

    def test_task_manager_does_not_reuse_search_slot_after_success(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([
            {"id": "item_01", "kind": "kele"},
        ])
        active = manager.next_decision().selected_task
        manager.bind_search_task_product(active.task_id, "kele")
        manager.mark_task_succeeded(active.task_id)
        self.assertIsNone(manager.next_decision().selected_task)

    def test_unbound_search_slot_cannot_be_recorded_as_delivery_success(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([
            {"id": "item_01", "kind": "kele"},
        ])
        active = manager.next_decision().selected_task

        with self.assertRaises(RuntimeError):
            manager.mark_task_succeeded(active.task_id)

    def test_local_completion_is_not_labelled_as_referee_confirmation(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([
            {"id": "item_01", "kind": "kele"},
        ])
        active = manager.next_decision().selected_task
        manager.bind_search_task_product(active.task_id, "kele")

        task = manager.mark_task_succeeded(
            active.task_id,
            referee_verified=False,
        )

        self.assertEqual(task.metadata["completion_evidence"], "local_execution_only")

    def test_search_slot_requeues_only_for_bounded_navigation_failure(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([
            {"id": "item_01", "kind": "kele"},
        ])
        active = manager.next_decision().selected_task

        retried = manager.mark_task_failed(active.task_id, requeue=True)
        self.assertEqual(retried.status.value, "pending")
        self.assertIs(manager.next_decision().selected_task, active)

        exhausted = manager.mark_task_failed(active.task_id, requeue=True)
        self.assertEqual(exhausted.status.value, "failed")
        next_task = manager.next_decision().selected_task
        self.assertIsNotNone(next_task)
        self.assertNotEqual(next_task.slot_id, active.slot_id)

    def test_search_slot_is_retired_when_failure_can_have_moved_object(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([
            {"id": "item_01", "kind": "kele"},
        ])
        active = manager.next_decision().selected_task
        manager.mark_task_failed(active.task_id, requeue=False)
        next_task = manager.next_decision().selected_task
        self.assertIsNotNone(next_task)
        self.assertNotEqual(next_task.slot_id, active.slot_id)

    def test_search_slot_is_retired_for_static_unreachable_arm_geometry(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([
            {"id": "item_01", "kind": "zhijin"},
        ])
        active = manager.next_decision().selected_task

        # A shelf post is not a transient navigation failure.  The caller
        # must retire this task rather than retrying the same blocked approach.
        retired = manager.mark_task_failed(active.task_id, requeue=False)

        self.assertEqual(retired.status.value, "failed")
        self.assertEqual(retired.retry_count, 1)
        self.assertNotEqual(manager.next_decision().selected_task, active)

    def test_top_right_tissue_requires_mirrored_left_arm(self):
        task = self._manual_search_task(
            "tissue_top_right",
            41,
            nav_x=0.50,
            nav_y=2.45,
            level="L3",
            column="C3",
        )
        task.product_name = "zhijin"
        self.assertTrue(requires_mirrored_left_arm(task))

        task.column = "C2"
        self.assertFalse(requires_mirrored_left_arm(task))
        task.column = "C3"
        task.product_name = "kele"
        self.assertFalse(requires_mirrored_left_arm(task))

    def test_inventory_observation_prioritises_matching_aruco_slot(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([{"id": "item_01", "kind": "kele"}])
        observation = {
            "aruco_id": 31,
            "kind": "kele",
            "confidence": 0.92,
            "world": [0.73, 3.23, 0.94],
        }
        self.assertEqual(manager.register_inventory_observations([observation]), 0)
        self.assertEqual(manager.register_inventory_observations([observation]), 1)
        selected = manager.next_decision().selected_task
        self.assertEqual(selected.aruco_id, 31)
        self.assertEqual(selected.product_name, "kele")
        self.assertAlmostEqual(selected.world_position[0], 0.875)
        self.assertAlmostEqual(selected.world_position[1], 3.23)
        self.assertAlmostEqual(selected.world_position[2], 0.9235)
        self.assertTrue(selected.metadata["inventory_confirmed"])
        self.assertAlmostEqual(selected.navigation_target.x, 0.875 - 0.108)
        self.assertAlmostEqual(selected.navigation_target.y, 3.23 - 0.768)

    def test_inventory_observation_clamps_depth_z_to_slot_level_and_product_height(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([{"id": "item_01", "kind": "shupian"}])
        observation = {
            "aruco_id": 2,
            "kind": "shupian",
            "confidence": 0.92,
            "world": [-1.52, 3.20, 1.196],
        }
        self.assertEqual(manager.register_inventory_observations([observation]), 0)
        self.assertEqual(manager.register_inventory_observations([observation]), 1)

        selected = manager.next_decision().selected_task
        self.assertEqual(selected.aruco_id, 2)
        self.assertEqual(selected.product_name, "shupian")
        self.assertAlmostEqual(selected.world_position[2], 0.604)

    def test_single_unconfirmed_inventory_frame_does_not_change_search_route(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([{"id": "item_01", "kind": "kele"}])
        manager.register_inventory_observations([{
            "aruco_id": 31,
            "kind": "kele",
            "confidence": 0.99,
            "world": [0.73, 3.23, 0.94],
        }])
        selected = manager.next_decision().selected_task
        self.assertNotEqual(selected.aruco_id, 31)
        self.assertEqual(selected.product_name, SEARCH_PRODUCT)

    def test_inventory_rejects_invalid_depth_world_point(self):
        manager = TaskManager()
        manager.register_inventory_observations([{
            "aruco_id": 3,
            "kind": "kele",
            "confidence": 0.99,
            "world": [0.1, 0.0, 0.0],
        }])
        self.assertNotIn(3, manager.inventory_by_aruco)

    def test_inventory_records_keep_timestamp_and_marker_world(self):
        manager = TaskManager()
        observation = {
            "aruco_id": 31,
            "kind": "kele",
            "confidence": 0.99,
            "world": [0.73, 3.23, 0.94],
            "marker_world": [0.71, 3.21, 0.89],
            "stamp": 123.45,
        }
        manager.register_inventory_observations([observation, observation])
        record = manager.inventory_by_aruco[31]

        self.assertEqual(record["kind"], "kele")
        self.assertEqual(record["state"], "confirmed")
        self.assertAlmostEqual(record["last_seen"], 123.45)
        self.assertEqual(record["marker_world"], (0.71, 3.21, 0.89))

    def test_success_consumes_inventory_record(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([{"id": "item_01", "kind": "kele"}])
        active = manager.next_decision().selected_task
        manager.register_inventory_observations([{
            "aruco_id": active.aruco_id,
            "kind": "kele",
            "confidence": 0.99,
            "world": [active.world_position[0], active.world_position[1] - 0.02, 0.94],
            "stamp": 100.0,
        }, {
            "aruco_id": active.aruco_id,
            "kind": "kele",
            "confidence": 0.99,
            "world": [active.world_position[0], active.world_position[1] - 0.02, 0.94],
            "stamp": 100.1,
        }])
        manager.apply_inventory_observation(active)
        manager.mark_task_succeeded(active.task_id)
        record = manager.inventory_by_aruco[active.aruco_id]

        self.assertEqual(record["state"], "consumed")
        self.assertFalse(record["confirmed"])

    def test_selection_reserves_inventory_and_retry_releases_it(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([{"id": "item_01", "kind": "kele"}])
        manager.register_inventory_observations([{
            "aruco_id": 3,
            "kind": "kele",
            "confidence": 0.99,
            "world": [-1.955, 3.23, 0.94],
            "stamp": 100.0,
        }, {
            "aruco_id": 3,
            "kind": "kele",
            "confidence": 0.99,
            "world": [-1.955, 3.23, 0.94],
            "stamp": 100.1,
        }])
        active = manager.next_decision().selected_task
        manager.apply_inventory_observation(active)
        reserved = manager.inventory_by_aruco[active.aruco_id]

        self.assertEqual(reserved["state"], "reserved")
        self.assertEqual(reserved["reserved_task_id"], active.task_id)

        manager.mark_task_failed(active.task_id, requeue=True)
        released = manager.inventory_by_aruco[active.aruco_id]

        self.assertEqual(released["state"], "confirmed")
        self.assertNotIn("reserved_task_id", released)

    def test_inventory_scoring_prefers_shorter_roundtrip_route(self):
        manager = TaskManager()
        task_left = self._manual_search_task("search_left", 11, nav_x=0.20, nav_y=2.45)
        task_right = self._manual_search_task("search_right", 12, nav_x=1.20, nav_y=2.45, column="C2")
        manager.tasks = [task_left, task_right]
        manager.requested_counts = Counter({"kele": 2})
        manager.register_inventory_observations([{
            "aruco_id": 11,
            "kind": "kele",
            "confidence": 0.90,
            "world": [0.20, 2.45, 0.94],
            "stamp": 100.0,
        }, {
            "aruco_id": 11,
            "kind": "kele",
            "confidence": 0.90,
            "world": [0.20, 2.45, 0.94],
            "stamp": 100.1,
        }, {
            "aruco_id": 12,
            "kind": "kele",
            "confidence": 0.90,
            "world": [1.20, 2.45, 0.94],
            "stamp": 100.0,
        }, {
            "aruco_id": 12,
            "kind": "kele",
            "confidence": 0.90,
            "world": [1.20, 2.45, 0.94],
            "stamp": 100.1,
        }])

        selected = manager.next_decision().selected_task

        self.assertEqual(selected.aruco_id, 11)
        self.assertLess(
            manager._estimate_roundtrip_cost(task_left),
            manager._estimate_roundtrip_cost(task_right),
        )

    def test_inventory_scoring_prefers_fresher_record_when_routes_match(self):
        manager = TaskManager()
        task_fresh = self._manual_search_task("search_fresh", 21, nav_x=0.70, nav_y=2.45)
        task_stale = self._manual_search_task("search_stale", 22, nav_x=0.70, nav_y=2.45, column="C2")
        manager.tasks = [task_fresh, task_stale]
        manager.requested_counts = Counter({"kele": 2})
        now = time.time()
        manager.inventory_by_aruco[21] = {
            "kind": "kele",
            "confidence": 0.88,
            "world": (0.70, 2.45, 0.94),
            "hits": 3,
            "confirmed": True,
            "state": "confirmed",
            "first_seen": now - 1.0,
            "last_seen": now - 1.0,
        }
        manager.inventory_by_aruco[22] = {
            "kind": "kele",
            "confidence": 0.99,
            "world": (0.70, 2.45, 0.94),
            "hits": 5,
            "confirmed": True,
            "state": "confirmed",
            "first_seen": now - 9.0,
            "last_seen": now - 9.0,
        }

        selected = manager.next_decision().selected_task

        self.assertEqual(selected.aruco_id, 21)
        self.assertGreater(
            manager._inventory_candidate_score(task_fresh, now=now),
            manager._inventory_candidate_score(task_stale, now=now),
        )

    def test_inventory_scoring_honours_manual_reservation_bonus(self):
        manager = TaskManager()
        task_reserved = self._manual_search_task("search_reserved", 31, nav_x=0.90, nav_y=2.45)
        task_unreserved = self._manual_search_task("search_unreserved", 32, nav_x=0.90, nav_y=2.45, column="C2")
        manager.tasks = [task_reserved, task_unreserved]
        manager.requested_counts = Counter({"kele": 2})
        manager.inventory_by_aruco[31] = {
            "kind": "kele",
            "confidence": 0.90,
            "world": (0.90, 2.45, 0.94),
            "hits": 3,
            "confirmed": True,
            "state": "reserved",
            "reserved_task_id": "search_reserved",
            "first_seen": time.time() - 1.0,
            "last_seen": time.time() - 1.0,
        }
        manager.inventory_by_aruco[32] = {
            "kind": "kele",
            "confidence": 0.90,
            "world": (0.90, 2.45, 0.94),
            "hits": 3,
            "confirmed": True,
            "state": "confirmed",
            "first_seen": time.time() - 1.0,
            "last_seen": time.time() - 1.0,
        }

        selected = manager.next_decision().selected_task

        self.assertEqual(selected.aruco_id, 31)
        self.assertGreater(
            manager._inventory_candidate_score(task_reserved, now=time.time()),
            manager._inventory_candidate_score(task_unreserved, now=time.time()),
        )


class PerceptionTests(unittest.TestCase):
    def test_inventory_association_uses_marker_below_product(self):
        matches = associate_detections_to_markers(
            [{"x": 200, "y": 150, "w": 70, "h": 100}],
            [
                {"id": 7, "corners": [[192, 220], [208, 220], [208, 236], [192, 236]]},
                {"id": 8, "corners": [[310, 220], [326, 220], [326, 236], [310, 236]]},
            ],
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["detection_index"], 0)
        self.assertEqual(matches[0]["aruco_id"], 7)

    def test_single_frame_class_is_not_a_search_consensus(self):
        self.assertIsNone(stable_class_consensus(["kele"]))

    def test_search_consensus_rejects_flapping_classes(self):
        self.assertIsNone(stable_class_consensus(
            ["kele", "maidong", "kele"], min_samples=3, min_ratio=0.67
        ))

    def test_search_consensus_accepts_stable_class(self):
        self.assertEqual(
            stable_class_consensus(
                ["kele", "kele", "kele", "maidong"],
                min_samples=3,
                min_ratio=0.67,
            ),
            "kele",
        )


class YoloBackendTests(unittest.TestCase):
    def test_missing_checkpoint_raises_immediately(self):
        from perception.backends import YoloBackend

        with self.assertRaises(FileNotFoundError):
            YoloBackend(r"C:\does\not\exist\supermarket_multiclass.pt")

    def test_strict_official_mode_rejects_non_official_classes(self):
        from perception.backends import YoloBackend

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            ckpt_path = tmp.name
        try:
            fake_torch = types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: False),
                device=lambda name: name,
                load=lambda *args, **kwargs: None,
            )

            class FakeYOLO:
                def __init__(self, path):
                    self.path = path
                    self.names = {"0": "foo"}
                    self.model = self

                def to(self, device):
                    self.device = device
                    return self

                def eval(self):
                    return None

            with mock.patch.dict(
                sys.modules,
                {"torch": fake_torch, "ultralytics": types.SimpleNamespace(YOLO=FakeYOLO)},
            ), mock.patch.dict(os.environ, {"SUPERMARKET_YOLO_REQUIRE_OFFICIAL_CLASSES": "1"}):
                with self.assertRaises(RuntimeError):
                    YoloBackend(ckpt_path)
        finally:
            os.unlink(ckpt_path)

    def test_detect_requires_initialized_model(self):
        from perception.backends import YoloBackend

        backend = object.__new__(YoloBackend)
        backend.ckpt_path = "dummy.pt"
        backend.model = None

        with self.assertRaises(RuntimeError):
            backend.detect(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint16), np.eye(3))


class NavigationTests(unittest.TestCase):
    def test_zhijin_empty_grasp_retry_advances_depth_without_x_sweep(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object),
            "std_srvs.srv": types.SimpleNamespace(Trigger=object),
            "vision_msgs.msg": types.SimpleNamespace(Detection3DArray=object),
            "discoverse.utils": types.SimpleNamespace(step_func=lambda *args, **kwargs: None),
            "mmk2_kdl": types.SimpleNamespace(MMK2Kdl=object),
            "perception.backends": types.SimpleNamespace(stable_class_consensus=lambda *args, **kwargs: None),
            "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=SupermarketGridPlanner),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            import importlib

            client_mod = importlib.import_module("supermarket_sorting_client")

        client = object.__new__(client_mod.PickPlaceClient)
        client.active_product_name = "zhijin"
        client.grasp_profile = client_mod.PRODUCT_GRASP_PROFILES["zhijin"]
        client.grasp_yaw = client_mod.YAW_NORTH
        client.local_grasp_retries = 1
        client.drop_recoveries = 0
        client.last_grasp_retry_reason = "closed gripper without grasp evidence"
        client.now = lambda: 10.0
        client.get_logger = lambda: types.SimpleNamespace(info=lambda *args, **kwargs: None)

        object_world = np.array([-1.958, 3.215, 0.895])
        client_mod.PickPlaceClient.lock_grasp_geometry(
            client, object_world, source="vision")

        self.assertAlmostEqual(client.PINCH_WORLD[0], object_world[0] - 0.001)
        self.assertAlmostEqual(client.GRASP_ENDPOINT_WORLD[1], object_world[1] + 0.030)
        self.assertGreater(client.GRASP_ENDPOINT_WORLD[1], object_world[1] + 0.020)

    def test_vision_displacement_waits_for_creep_settle_and_confirmation(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object),
            "std_srvs.srv": types.SimpleNamespace(Trigger=object),
            "vision_msgs.msg": types.SimpleNamespace(Detection3DArray=object),
            "discoverse.utils": types.SimpleNamespace(step_func=lambda *args, **kwargs: None),
            "mmk2_kdl": types.SimpleNamespace(MMK2Kdl=object),
            "perception.backends": types.SimpleNamespace(stable_class_consensus=lambda *args, **kwargs: None),
            "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=SupermarketGridPlanner),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            import importlib
            client_mod = importlib.import_module("supermarket_sorting_client")

        clock = [10.0]
        client = object.__new__(client_mod.PickPlaceClient)
        client.OBJECT_WORLD = np.array([0.0, 0.0, 0.9])
        client.live_object_world = np.array([0.14, 0.0, 0.9])
        client.live_object_seen_at = clock[0]
        client.vision_lock_confirmed = True
        client.grasp_lock_source = "vision"
        client.creep_started_at = clock[0]
        client.grasp_profile = {}
        client.last_live_monitor_log = -100.0
        client.live_target_displacement_hits = 0
        client.now = lambda: clock[0]
        client.get_logger = lambda: types.SimpleNamespace(info=lambda *args, **kwargs: None, warn=lambda *args, **kwargs: None)

        # Camera view shifts while the arm is still settling: no failure.
        self.assertFalse(client.live_target_displaced())
        clock[0] += client_mod.VISION_MONITOR_ARM_SETTLE_TIME + 0.1
        # One or two outliers do not retire a physical shelf slot.
        for _ in range(client_mod.VISION_MONITOR_CONFIRM_SAMPLES - 1):
            self.assertFalse(client.live_target_displaced())
        self.assertTrue(client.live_target_displaced())

    def test_only_invariant_structures_are_static(self):
        # V2 randomises cardboard-box poses, so they must enter through lidar
        # observations rather than stale coordinates in the global map. The
        # static set holds the divider, the delivery table and the four
        # perimeter walls (all invariant MJCF structures).
        self.assertEqual(len(SupermarketGridPlanner.STATIC_OBSTACLES), 6)

    def test_lidar_hits_on_static_structures_are_not_double_inflated(self):
        # A hit on the west wall face belongs to the static wall model; adding
        # it to the dynamic set again would inflate the wall and close real
        # wall-side passages (verified: the delivery A* returned no route).
        planner = SupermarketGridPlanner()
        self.assertEqual(planner._dynamic_cells([(-2.50, -1.0), (-1.0, -1.0)]),
                         planner._dynamic_cells([(-1.0, -1.0)]))

    def test_wall_side_passage_stays_open_with_wall_hits(self):
        # box_04-like obstacle just east of the west wall: the wall-side gap
        # (wall face x=-2.47, box west corner x=-1.51) must remain routable
        # even when lidar reports the wall surface itself.
        planner = SupermarketGridPlanner(
            corridor_clearance=0.88, dynamic_clearance=0.50)
        start = (-0.50, 2.30)
        goal = (-1.88, -2.74)
        dynamic = [
            (-1.51, -1.05), (-1.20, -1.05), (-0.88, -1.05),  # box_04 north face
            (-2.50, -1.00), (-2.50, 0.00), (-2.50, -2.00),   # west wall surface
        ]
        route = planner.plan(start, goal, dynamic_points=dynamic)
        self.assertTrue(route, "wall-side passage was closed by wall/box hits")
        self.assertEqual(route[-1], list(goal))
        previous = start
        for point in route:
            self.assertTrue(planner.path_is_clear(previous, point, dynamic))
            previous = point

    def test_pruned_route_does_not_cross_static_obstacles(self):
        planner = SupermarketGridPlanner()
        start = (1.60, 2.05)
        goal = (-1.88, -2.74)
        route = planner.plan(start, goal)

        self.assertTrue(route)
        self.assertEqual(route[-1], list(goal))
        previous = start
        for point in route:
            self.assertTrue(planner.path_is_clear(previous, point))
            previous = point

    def test_loaded_delivery_crosses_divider_above_arm_clearance_line(self):
        planner = SupermarketGridPlanner(corridor_clearance=0.88)
        # The old y=2.06 and y=2.38 lateral crossings clear a bare base at
        # best, but not the carried elbow/gripper envelope. The delivery gate
        # stays in the high shelf-side band.
        self.assertFalse(planner.path_is_clear((0.82, 2.06), (-0.50, 2.06)))
        self.assertFalse(planner.path_is_clear((0.82, 2.38), (-0.50, 2.38)))
        self.assertTrue(planner.path_is_clear((0.82, 2.62), (-0.50, 2.62)))

    def test_dynamic_obstacle_forces_a_detour(self):
        planner = SupermarketGridPlanner()
        start = (1.20, -2.80)
        goal = (1.20, -1.60)
        dynamic = [(1.20, -2.20)]
        blocked = planner.plan(start, goal, dynamic_points=dynamic)

        self.assertTrue(blocked)
        self.assertFalse(planner.path_is_clear(start, goal, dynamic))
        previous = start
        for point in blocked:
            self.assertTrue(planner.path_is_clear(previous, point, dynamic))
            previous = point

    def test_loaded_dynamic_clearance_protects_carried_arm_envelope(self):
        start = (-1.0, 0.0)
        goal = (0.0, 0.0)
        dynamic = [(-0.50, 0.40)]
        bare = SupermarketGridPlanner(dynamic_clearance=0.22)
        loaded = SupermarketGridPlanner(dynamic_clearance=0.42)

        self.assertTrue(bare.path_is_clear(start, goal, dynamic))
        self.assertFalse(loaded.path_is_clear(start, goal, dynamic))

    def test_shelf_recovery_uses_fixed_corridor_after_stuck_event(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object),
            "std_srvs.srv": types.SimpleNamespace(Trigger=object),
            "vision_msgs.msg": types.SimpleNamespace(Detection3DArray=object),
            "discoverse.utils": types.SimpleNamespace(step_func=lambda *args, **kwargs: None),
            "mmk2_kdl": types.SimpleNamespace(MMK2Kdl=object),
            "perception.backends": types.SimpleNamespace(stable_class_consensus=lambda *args, **kwargs: None),
            "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=SupermarketGridPlanner),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            import importlib

            client_mod = importlib.import_module("supermarket_sorting_client")

        self.assertGreaterEqual(client_mod.SHELF_CROSS_Y, 2.20)
        self.assertLessEqual(client_mod.SHELF_CROSS_Y, 2.35)
        self.assertGreaterEqual(client_mod.DELIVERY_CROSS_Y, 2.60)
        self.assertGreaterEqual(client_mod.LOADED_CORRIDOR_CLEARANCE, 0.85)
        # The carry tuck is disabled by default since round 76: the slew fix
        # made it actually move, but the 6 s timeout then left the arm
        # mid-pose and tilted the gripped bottle ~37 deg (count5_v5).  The
        # proven S5 path carries the frozen grasp pose and pre-raises before
        # the table approach.  The env escape hatch stays for A/B testing.
        self.assertFalse(client_mod.CARRY_TUCK_ENABLED)
        self.assertGreaterEqual(client_mod.START_BASE_X, 1.80)
        self.assertGreater(client_mod.STARTUP_STRAIGHT_SPEED, 0.0)
        self.assertGreater(client_mod.STARTUP_STOW_DWELL, 0.0)
        self.assertLessEqual(
            float(client_mod.PRODUCT_GRASP_PROFILES["zhijin"]["deploy_offset"][2]),
            0.0,
        )
        self.assertEqual(
            float(client_mod.PRODUCT_GRASP_PROFILES["zhijin"]["contact_z_bias"]),
            0.0,
        )

        # L3 tissue boxes need a column-dependent chassis standoff.  A centre
        # stop is not IK-reachable at the real slide travel; do not reintroduce
        # the previous impossible -0.10 m slide workaround.
        profile_client = object.__new__(client_mod.PickPlaceClient)
        for column, expected_bias in (("C1", 0.22), ("C2", 0.22), ("C3", -0.19)):
            task = types.SimpleNamespace(
                product_name="zhijin", level="L3", column=column
            )
            profile = client_mod.PickPlaceClient.profile_for_task(profile_client, task)
            self.assertAlmostEqual(profile["base_x_bias"], expected_bias)
            self.assertAlmostEqual(profile["grasp_slide"], -0.030)
            self.assertAlmostEqual(profile["shelf_nav_y"], 2.68)
            self.assertAlmostEqual(profile["reach_lateral_max"], 0.34)

        client = object.__new__(client_mod.PickPlaceClient)
        client.base_xy = np.array([1.58, 2.00], dtype=float)
        client.route_to_shelf = [[9.0, 9.0]]
        client.front_blocked = False
        client.nav_recovery_count = 1
        client.route_needs_plan = False
        client.nav_mode = "drive"
        client.last_replan_time = 0.0
        client.now = lambda: 100.0
        client.get_logger = lambda: types.SimpleNamespace(
            info=lambda *args, **kwargs: None
        )
        # Round 56: delivery goals are staggered per placed item; a fresh
        # client (0 placed) must target the base DELIVERY_GOAL.
        client.delivery_goal_current = client_mod.DELIVERY_GOAL.copy()
        client.placed_success_count = 0

        goal = np.array([0.82, 2.45], dtype=float)
        route = client_mod.PickPlaceClient.plan_route(client, goal, "shelf")
        expected = client.shelf_corridor_route(goal)

        self.assertEqual(route, expected)
        self.assertNotEqual(route, client.route_to_shelf)
        self.assertTrue(route)
        self.assertEqual(route[-1], goal.tolist())
        self.assertIn([float(goal[0]), client_mod.SHELF_CROSS_Y], route)

        client.base_xy = np.array([0.82, 2.06], dtype=float)
        delivery = client_mod.PickPlaceClient.delivery_corridor_route(client)
        # East-side start must cross the divider's north end (y > 1.70) at the
        # high band BEFORE descending the west lane. The old low-corridor
        # crossing (y=0.92) is geometrically blocked by the divider and was
        # verified in simulation to end in a loaded-arm collision (C1) and a
        # turn stall; the crossing band now sits ~0.6 m above the divider end
        # so heading errors cannot wedge the chassis against its face.
        self.assertEqual(delivery[0], [0.82, 2.30])
        self.assertGreaterEqual(
            float(delivery[0][1]),
            client_mod.DIVIDER_NORTH_END_Y + 0.55,
        )
        self.assertEqual(delivery[1], [client_mod.DELIVERY_VERTICAL_LANE_X, 2.30])
        self.assertEqual(delivery[2], [client_mod.DELIVERY_VERTICAL_LANE_X, -0.78])
        self.assertEqual(delivery[-1], client_mod.DELIVERY_GOAL.tolist())

    def test_delivery_goal_is_staggered_per_placed_item(self):
        """Round 56: consecutive items must not land on the same table spot
        (fixed DELIVERY_GOAL let item 2 knock item 1's bottle off the table)."""
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object),
            "std_srvs.srv": types.SimpleNamespace(Trigger=object),
            "vision_msgs.msg": types.SimpleNamespace(Detection3DArray=object),
            "discoverse.utils": types.SimpleNamespace(step_func=lambda *args, **kwargs: None),
            "mmk2_kdl": types.SimpleNamespace(MMK2Kdl=object),
            "perception.backends": types.SimpleNamespace(stable_class_consensus=lambda *args, **kwargs: None),
            "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=SupermarketGridPlanner),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            import importlib

            client_mod = importlib.import_module("supermarket_sorting_client")

        client = object.__new__(client_mod.PickPlaceClient)
        client.base_xy = np.array([0.82, 2.06], dtype=float)
        client.route_to_shelf = [[9.0, 9.0]]
        client.front_blocked = False
        client.nav_recovery_count = 0
        client.route_needs_plan = False
        client.nav_mode = "drive"
        client.last_replan_time = 0.0
        client.now = lambda: 100.0
        client.get_logger = lambda: types.SimpleNamespace(
            info=lambda *args, **kwargs: None
        )

        seen_xy = []
        for placed in range(5):
            client.placed_success_count = placed
            client.delivery_goal_current = client_mod.DELIVERY_GOAL.copy()
            idx = placed % len(client_mod.PLACE_X_OFFSETS)
            client.delivery_goal_current[0] += client_mod.PLACE_X_OFFSETS[idx]
            client.delivery_goal_current[1] += client_mod.PLACE_Y_OFFSETS[idx]
            route = client_mod.PickPlaceClient.delivery_corridor_route(client)
            goal_x = float(route[-1][0])
            goal_y = float(route[-1][1])
            seen_xy.append((goal_x, goal_y))
            # Every goal must stay inside the S5 box footprint.  NOTE: the
            # route goal is the BASE standoff (y ~ -2.84); the released bottle
            # centre lands further south (~-3.4) inside the S5 box y range.
            self.assertGreaterEqual(goal_x, -2.42)
            self.assertLessEqual(goal_x, -1.46)
            self.assertGreaterEqual(goal_y, -3.05)
            self.assertLessEqual(goal_y, -2.60)
        # Consecutive items must never target the same x spot (the y stagger
        # is zero in the official baseline; v70 proves a fixed spot is safe,
        # so a 3-spot x rotation with distinct neighbours is sufficient).
        # Round 61: five unique spots at 8 cm spacing.
        for i in range(1, len(seen_xy)):
            self.assertGreaterEqual(
                abs(seen_xy[i][0] - seen_xy[i - 1][0]), 0.07)
        self.assertEqual(len(client_mod.PLACE_X_OFFSETS), 5)
        self.assertEqual(len(client_mod.PLACE_Y_OFFSETS), 5)

    def test_shelf_crossing_approach_is_speed_capped_near_rack(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object),
            "std_srvs.srv": types.SimpleNamespace(Trigger=object),
            "vision_msgs.msg": types.SimpleNamespace(Detection3DArray=object),
            "discoverse.utils": types.SimpleNamespace(step_func=lambda *args, **kwargs: None),
            "mmk2_kdl": types.SimpleNamespace(MMK2Kdl=object),
            "perception.backends": types.SimpleNamespace(stable_class_consensus=lambda *args, **kwargs: None),
            "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=SupermarketGridPlanner),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            import importlib

            client_mod = importlib.import_module("supermarket_sorting_client")

        client = object.__new__(client_mod.PickPlaceClient)
        commands = []
        client.phase = client_mod.NAV_SHELF
        client.recovery_state = "idle"
        client.route_needs_plan = False
        client.route_goal = None
        client.route_purpose = "shelf"
        client.front_blocked = False
        client.front_blocked_since = None
        client.carry_departure_settle_until = 0.0
        client.last_replan_time = 0.0
        client.nav_idx = 0
        client.nav_mode = "drive"
        client.base_xy = np.array([1.92, 2.00], dtype=float)
        client.base_yaw = client_mod.YAW_NORTH
        client.turn_tol = 0.05
        client.grasp_profile = {}
        client.last_nav_progress_xy = np.array(client.base_xy, dtype=float)
        client.last_nav_progress_time = 10.0
        client.now = lambda: 10.1
        client.set_twist = lambda linear, angular: commands.append((linear, angular))
        client.maybe_start_stuck_recovery = lambda target: False

        route = [[1.92, client_mod.SHELF_CROSS_Y], [1.48, client_mod.SHELF_CROSS_Y]]
        self.assertFalse(client_mod.PickPlaceClient.follow_route(
            client, route, client_mod.GRASP_YAW))
        self.assertTrue(commands)
        self.assertLessEqual(commands[-1][0], client_mod.SHELF_APPROACH_LINEAR_CAP)
        self.assertLessEqual(abs(commands[-1][1]), client_mod.SHELF_APPROACH_ANGULAR_CAP)

    def test_shelf_lateral_crossing_turns_in_place_before_driving(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object),
            "std_srvs.srv": types.SimpleNamespace(Trigger=object),
            "vision_msgs.msg": types.SimpleNamespace(Detection3DArray=object),
            "discoverse.utils": types.SimpleNamespace(step_func=lambda *args, **kwargs: None),
            "mmk2_kdl": types.SimpleNamespace(MMK2Kdl=object),
            "perception.backends": types.SimpleNamespace(stable_class_consensus=lambda *args, **kwargs: None),
            "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=SupermarketGridPlanner),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            import importlib

            client_mod = importlib.import_module("supermarket_sorting_client")

        client = object.__new__(client_mod.PickPlaceClient)
        commands = []
        route = [
            [1.92, client_mod.SHELF_CROSS_Y],
            [1.48, client_mod.SHELF_CROSS_Y],
            [0.85, client_mod.SHELF_CROSS_Y],
        ]
        client.phase = client_mod.NAV_SHELF
        client.recovery_state = "idle"
        client.route_needs_plan = False
        client.route_goal = None
        client.route_purpose = "shelf"
        client.front_blocked = False
        client.front_blocked_since = None
        client.carry_departure_settle_until = 0.0
        client.last_replan_time = 0.0
        client.nav_idx = 1
        client.nav_mode = "turn"
        client.base_xy = np.array([1.92, client_mod.SHELF_CROSS_Y], dtype=float)
        client.base_yaw = client_mod.YAW_NORTH
        client.turn_tol = 0.05
        client.grasp_profile = {}
        client.route_to_shelf = [list(point) for point in route]
        client.last_nav_progress_xy = np.array(client.base_xy, dtype=float)
        client.last_nav_progress_time = 10.0
        client.shelf_turn_progress_yaw = None
        client.delivery_turn_progress_yaw = None
        client.nav_recovery_count = 0
        client.now = lambda: 10.1
        client.set_twist = lambda linear, angular: commands.append((linear, angular))
        client.maybe_start_stuck_recovery = lambda target: False
        client.maybe_start_delivery_turn_recovery = lambda: False

        self.assertFalse(client_mod.PickPlaceClient.follow_route(
            client, route, client_mod.GRASP_YAW))
        self.assertTrue(commands)
        self.assertEqual(commands[-1][0], 0.0)
        self.assertLessEqual(abs(commands[-1][1]), client_mod.SHELF_CROSS_ANGULAR_CAP)

    def test_place_raise_search_uses_lower_reachable_ik_target(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object),
            "std_srvs.srv": types.SimpleNamespace(Trigger=object),
            "vision_msgs.msg": types.SimpleNamespace(Detection3DArray=object),
            "discoverse.utils": types.SimpleNamespace(step_func=lambda *args, **kwargs: None),
            "mmk2_kdl": types.SimpleNamespace(MMK2Kdl=object),
            "perception.backends": types.SimpleNamespace(stable_class_consensus=lambda *args, **kwargs: None),
            "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=SupermarketGridPlanner),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            import importlib

            client_mod = importlib.import_module("supermarket_sorting_client")

        client = object.__new__(client_mod.PickPlaceClient)
        attempted = []
        reachable_z = client_mod.PLACE_RELEASE_EE_Z - 0.02
        client.ee_world = lambda: np.array([-1.97, -3.25, 0.66], dtype=float)
        client.world_to_footprint = lambda p: np.array(p, dtype=float)
        client.footprint_to_world = lambda p: np.array(p, dtype=float)
        client.grasp_rot = np.eye(3)

        def fake_arm_to(world_pos, rot=None):
            attempted.append(float(world_pos[2]))
            return float(world_pos[2]) <= reachable_z + 1e-9

        client.arm_to = fake_arm_to
        selected = client_mod.PickPlaceClient.arm_to_place_raise(client, 0.66)

        self.assertAlmostEqual(selected, reachable_z)
        self.assertGreater(len(attempted), 1)
        self.assertEqual(attempted[-1], reachable_z)

    def test_place_raise_can_center_lateral_target_when_exact_pose_unreachable(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object),
            "std_srvs.srv": types.SimpleNamespace(Trigger=object),
            "vision_msgs.msg": types.SimpleNamespace(Detection3DArray=object),
            "discoverse.utils": types.SimpleNamespace(step_func=lambda *args, **kwargs: None),
            "mmk2_kdl": types.SimpleNamespace(MMK2Kdl=object),
            "perception.backends": types.SimpleNamespace(stable_class_consensus=lambda *args, **kwargs: None),
            "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=SupermarketGridPlanner),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            import importlib

            client_mod = importlib.import_module("supermarket_sorting_client")

        client = object.__new__(client_mod.PickPlaceClient)
        attempted_fp_y = []
        client.ee_world = lambda: np.array([0.45, -0.096, 0.64], dtype=float)
        client.world_to_footprint = lambda p: np.array(p, dtype=float)
        client.footprint_to_world = lambda p: np.array(p, dtype=float)
        client.grasp_rot = np.eye(3)
        client.get_logger = lambda: types.SimpleNamespace(info=lambda *args, **kwargs: None)

        def fake_arm_to(world_pos, rot=None):
            attempted_fp_y.append(float(world_pos[1]))
            return abs(float(world_pos[1])) < 0.01

        client.arm_to = fake_arm_to
        selected = client_mod.PickPlaceClient.arm_to_place_raise(client, 0.64)

        self.assertIsNotNone(selected)
        self.assertIn(0.0, attempted_fp_y)
        self.assertAlmostEqual(attempted_fp_y[-1], 0.0)

    def test_referee_collision_recovery_ignores_pre_delivery_collision(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object),
            "std_srvs.srv": types.SimpleNamespace(Trigger=object),
            "vision_msgs.msg": types.SimpleNamespace(Detection3DArray=object),
            "discoverse.utils": types.SimpleNamespace(step_func=lambda *args, **kwargs: None),
            "mmk2_kdl": types.SimpleNamespace(MMK2Kdl=object),
            "perception.backends": types.SimpleNamespace(stable_class_consensus=lambda *args, **kwargs: None),
            "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=SupermarketGridPlanner),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            import importlib

            client_mod = importlib.import_module("supermarket_sorting_client")

        client = object.__new__(client_mod.PickPlaceClient)

        client.referee_state = {"collided": True}
        client.delivery_collision_baseline = True
        self.assertFalse(client_mod.PickPlaceClient.delivery_referee_collision_is_new(client))

        client.delivery_collision_baseline = False
        self.assertTrue(client_mod.PickPlaceClient.delivery_referee_collision_is_new(client))

    def test_kele_precontact_guard_catches_logged_lateral_sweep_case(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object),
            "std_srvs.srv": types.SimpleNamespace(Trigger=object),
            "vision_msgs.msg": types.SimpleNamespace(Detection3DArray=object),
            "discoverse.utils": types.SimpleNamespace(step_func=lambda *args, **kwargs: None),
            "mmk2_kdl": types.SimpleNamespace(MMK2Kdl=object),
            "perception.backends": types.SimpleNamespace(stable_class_consensus=lambda *args, **kwargs: None),
            "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=SupermarketGridPlanner),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            import importlib

            client_mod = importlib.import_module("supermarket_sorting_client")

        profile = dict(client_mod.PRODUCT_GRASP_PROFILES["kele"])
        remaining = 0.098
        lateral_error = 0.049

        self.assertLessEqual(remaining, profile["creep_precontact_guard_distance"])
        self.assertGreater(abs(lateral_error), profile["creep_precontact_guard_lateral"])
        self.assertGreater(remaining, profile["touch_close_remaining"])
        self.assertAlmostEqual(float(np.clip(-0.70 * lateral_error, -0.030, 0.030)), -0.030)

    def test_delivery_final_table_approach_uses_fine_speed_cap(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object),
            "std_srvs.srv": types.SimpleNamespace(Trigger=object),
            "vision_msgs.msg": types.SimpleNamespace(Detection3DArray=object),
            "discoverse.utils": types.SimpleNamespace(step_func=lambda *args, **kwargs: None),
            "mmk2_kdl": types.SimpleNamespace(MMK2Kdl=object),
            "perception.backends": types.SimpleNamespace(stable_class_consensus=lambda *args, **kwargs: None),
            "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=SupermarketGridPlanner),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            import importlib

            client_mod = importlib.import_module("supermarket_sorting_client")

        client = object.__new__(client_mod.PickPlaceClient)
        commands = []
        goal = client_mod.DELIVERY_GOAL.copy()
        client.phase = client_mod.NAV_TABLE
        client.recovery_state = "idle"
        client.route_needs_plan = False
        client.route_goal = None
        client.route_purpose = "delivery"
        client.front_blocked = False
        client.front_blocked_since = None
        client.carry_departure_settle_until = 0.0
        client.last_replan_time = 0.0
        client.nav_idx = 1
        client.nav_mode = "drive"
        client.base_xy = goal + np.array([0.0, 0.24], dtype=float)
        client.base_yaw = client_mod.YAW_SOUTH
        client.grasp_profile = {}
        client.last_nav_progress_xy = np.array(client.base_xy, dtype=float)
        client.last_nav_progress_time = 10.0
        client.now = lambda: 10.1
        client.set_twist = lambda linear, angular: commands.append((linear, angular))
        client.maybe_start_stuck_recovery = lambda target: False

        route = [[goal[0], goal[1] + 0.80], goal.tolist()]
        self.assertFalse(client_mod.PickPlaceClient.follow_route(
            client, route, client_mod.YAW_SOUTH))
        self.assertTrue(commands)
        self.assertLessEqual(commands[-1][0], client_mod.DELIVERY_FINAL_FINE_LINEAR_CAP)
        self.assertLessEqual(abs(commands[-1][1]), client_mod.DELIVERY_FINAL_ANGULAR_CAP)

    def test_place_reverse_requires_full_clear_distance(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object),
            "std_srvs.srv": types.SimpleNamespace(Trigger=object),
            "vision_msgs.msg": types.SimpleNamespace(Detection3DArray=object),
            "discoverse.utils": types.SimpleNamespace(step_func=lambda *args, **kwargs: None),
            "mmk2_kdl": types.SimpleNamespace(MMK2Kdl=object),
            "perception.backends": types.SimpleNamespace(stable_class_consensus=lambda *args, **kwargs: None),
            "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=SupermarketGridPlanner),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            import importlib

            client_mod = importlib.import_module("supermarket_sorting_client")

        client = object.__new__(client_mod.PickPlaceClient)
        client.place_reverse_start = np.array([-1.82, -2.73], dtype=float)

        client.base_xy = client.place_reverse_start + np.array([0.0, 0.02], dtype=float)
        travel, done = client_mod.PickPlaceClient.place_reverse_progress(client)
        self.assertLess(travel, client_mod.PLACE_REVERSE_DISTANCE)
        self.assertFalse(done)

        # Round 61: the full egress reverse is 0.25 m (was 0.07 m) so the
        # tucked arm clears the placed bottle before the turn.
        client.base_xy = client.place_reverse_start + np.array(
            [0.0, client_mod.PLACE_REVERSE_DISTANCE + 0.001], dtype=float)
        travel, done = client_mod.PickPlaceClient.place_reverse_progress(client)
        self.assertGreaterEqual(travel, client_mod.PLACE_REVERSE_DISTANCE)
        self.assertTrue(done)

    def test_startup_clearance_stows_before_straight_exit_without_yaw(self):
        """The right-wall spawn may not enter route-turn control with a moving arm."""
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object),
            "std_srvs.srv": types.SimpleNamespace(Trigger=object),
            "vision_msgs.msg": types.SimpleNamespace(Detection3DArray=object),
            "discoverse.utils": types.SimpleNamespace(step_func=lambda *args, **kwargs: None),
            "mmk2_kdl": types.SimpleNamespace(MMK2Kdl=object),
            "perception.backends": types.SimpleNamespace(stable_class_consensus=lambda *args, **kwargs: None),
            "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=SupermarketGridPlanner),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            import importlib

            client_mod = importlib.import_module("supermarket_sorting_client")

        client = object.__new__(client_mod.PickPlaceClient)
        client.tc = np.zeros(19)
        client.tc[12:18] = client_mod.INIT_ARM_R
        client.base_xy = np.array([1.92, -3.10], dtype=float)
        client.base_yaw = client_mod.YAW_NORTH
        client.jpos = {"slide_joint": 0.12}
        client.jpos.update({
            f"right_arm_joint{i + 1}": float(client_mod.INIT_ARM_R[i]) + 0.20
            for i in range(6)
        })
        client.startup_clearance_pending = True
        client.startup_stow_ready_at = None
        client.startup_heading = None
        client.last_startup_clearance_log = -100.0
        client.nav_idx = 4
        client.nav_mode = "drive"
        client.last_nav_progress_xy = None
        client.last_nav_progress_time = 0.0
        client.nav_waypoint_last_dist = None
        client.recovery_until = 0.0
        client.recovery_state = "idle"
        client.recovery_linear = -0.18
        client.nav_recovery_count = 0
        client.front_blocked = False
        client.front_blocked_since = None
        clock = [1.0]
        commands = []
        client.now = lambda: clock[0]
        client.set_twist = lambda linear, angular: commands.append((linear, angular))
        client.get_logger = lambda: types.SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warn=lambda *args, **kwargs: None,
        )

        # Measured arm is not yet stowed: no chassis command, especially no yaw.
        self.assertTrue(client.startup_clearance_step())
        self.assertEqual(commands[-1], (0.0, 0.0))

        # Once measurements match the compact pose, the dwell still holds still.
        client.jpos["slide_joint"] = client_mod.SLIDE_TRAVEL
        client.jpos.update({
            f"right_arm_joint{i + 1}": float(client_mod.INIT_ARM_R[i])
            for i in range(6)
        })
        clock[0] = 2.0
        self.assertTrue(client.startup_clearance_step())
        self.assertEqual(commands[-1], (0.0, 0.0))

        # The only motion permitted inside the pocket is straight ahead.
        clock[0] += client_mod.STARTUP_STOW_DWELL + 0.01
        self.assertTrue(client.startup_clearance_step())
        self.assertAlmostEqual(commands[-1][0], client_mod.STARTUP_STRAIGHT_SPEED)
        self.assertEqual(commands[-1][1], 0.0)

        # Crossing the exit line releases ordinary navigation without a turn.
        client.base_xy[1] = client_mod.START_EXIT_Y + 0.02
        self.assertTrue(client.startup_clearance_step())
        self.assertFalse(client.startup_clearance_pending)
        self.assertEqual(client.nav_idx, 0)
        self.assertEqual(client.nav_mode, "turn")
        self.assertEqual(commands[-1], (0.0, 0.0))
        self.assertFalse(client.startup_clearance_step())


if __name__ == "__main__":
    unittest.main()
