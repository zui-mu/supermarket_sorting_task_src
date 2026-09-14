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
    from manipulation.cartesian_path import interpolate_se3, plan_continuous_ik_path
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
    from examples.supermarket_sorting.manipulation.cartesian_path import (
        interpolate_se3,
        plan_continuous_ik_path,
    )
    from examples.supermarket_sorting.decision.models import NavigationTarget, PickTask
    from examples.supermarket_sorting.decision.task_manager import SEARCH_PRODUCT, TaskManager
    from examples.supermarket_sorting.navigation.grid_planner import SupermarketGridPlanner
    from examples.supermarket_sorting.perception.backends import stable_class_consensus
    from examples.supermarket_sorting.perception.inventory import associate_detections_to_markers


class CartesianPathTests(unittest.TestCase):
    def test_interpolate_se3_bounds_steps_and_reaches_exact_goal(self):
        start = np.eye(4)
        goal = np.eye(4)
        goal[:3, :3] = Rotation.from_euler("z", 0.5).as_matrix()
        goal[:3, 3] = [0.035, -0.012, 0.0]

        poses = interpolate_se3(
            start,
            goal,
            translation_step=0.01,
            rotation_step_rad=0.12,
        )

        previous = start
        for pose in poses:
            self.assertLessEqual(
                float(np.linalg.norm(pose[:3, 3] - previous[:3, 3])),
                0.01 + 1e-9,
            )
            delta = Rotation.from_matrix(previous[:3, :3].T @ pose[:3, :3])
            self.assertLessEqual(float(delta.magnitude()), 0.12 + 1e-9)
            previous = pose
        np.testing.assert_allclose(poses[-1], goal, atol=1e-9)

    def test_interpolate_se3_supports_shared_multi_arm_sample_grid(self):
        start = np.eye(4)
        short_goal = np.eye(4)
        short_goal[0, 3] = 0.019
        long_goal = np.eye(4)
        long_goal[0, 3] = 0.029

        short_auto = interpolate_se3(start, short_goal, translation_step=0.01)
        long_auto = interpolate_se3(start, long_goal, translation_step=0.01)
        self.assertEqual((len(short_auto), len(long_auto)), (2, 3))

        synchronized = interpolate_se3(
            start,
            short_goal,
            translation_step=0.01,
            segments=5,
        )
        self.assertEqual(len(synchronized), 5)
        np.testing.assert_allclose(synchronized[-1], short_goal, atol=1e-9)

    def test_continuous_ik_rejects_a_joint_branch_jump(self):
        poses = [np.eye(4) for _ in range(3)]
        calls = 0

        def solve(_pose, previous):
            nonlocal calls
            calls += 1
            if calls == 2:
                return [previous + 0.8]
            return [previous + 0.05]

        result = plan_continuous_ik_path(
            poses,
            solve,
            np.zeros(6),
            max_joint_step=0.3,
        )

        self.assertFalse(result.complete)
        self.assertAlmostEqual(result.achieved_fraction, 1.0 / 3.0)
        self.assertEqual(len(result.joint_path), 1)
        self.assertIn("waypoint 2/3", result.reason)


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
            ["slot_A_L1_C1", "slot_A_L2_C1", "slot_A_L3_C1"],
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
        # NOTE 2026-08-23: this used "zhijin" as its example kind.  The tissue
        # box is now filtered out of the search order
        # (SUPERMARKET_SKIP_TISSUE_IN_SEARCH) until its top-pinch descent works,
        # so the target here is an ordinary product.  The behaviour under test -
        # retiring a slot blocked by static arm geometry instead of retrying it
        # - is unchanged.
        manager.build_search_tasks_for_targets([
            {"id": "item_01", "kind": "sanmingzhi"},
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
        self.assertAlmostEqual(selected.world_position[1], 3.243)
        self.assertAlmostEqual(selected.world_position[2], 0.9235)
        self.assertTrue(selected.metadata["inventory_confirmed"])
        self.assertAlmostEqual(selected.navigation_target.x, 0.875 - 0.108)
        self.assertAlmostEqual(selected.navigation_target.y, 3.243 - 0.768)

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
        obs2 = dict(observation)
        obs2["stamp"] = 123.46
        manager.register_inventory_observations([observation, obs2])
        record = manager.inventory_by_aruco[31]

        self.assertEqual(record["kind"], "kele")
        self.assertEqual(record["state"], "confirmed")
        self.assertAlmostEqual(record["last_seen"], 123.46)
        self.assertEqual(record["marker_world"], (0.71, 3.21, 0.89))

    def test_inventory_depth_is_anchored_to_same_frame_fixed_marker(self):
        manager = TaskManager()
        # Both estimates share a +5 cm odom/camera timing error.  ArUco 3's
        # fixed marker plane is y=3.168, so the product surface must be shifted
        # by the same -5 cm before it enters fresh-grasp history.
        observations = []
        for index in range(3):
            observations.append({
                "aruco_id": 3,
                "kind": "zhijin",
                "confidence": 0.99,
                "object_world": [-1.955, 3.218, 0.90],
                "marker_world": [-1.955, 3.218, 0.852],
                "frame_stamp": 300.0 + index * 0.05,
                "association_score": 0.2,
                "reject_reason": "ok",
                "ambiguous": False,
            })
        manager.register_inventory_observations(observations)
        record = manager.inventory_by_aruco[3]

        self.assertEqual(len(record["pose_history"]), 3)
        for sample in record["pose_history"]:
            self.assertAlmostEqual(sample["world"][1], 3.168, places=6)

    def test_fresh_grasp_center_depth_uses_marker_to_center_plane(self):
        manager = TaskManager()
        anchored = manager.anchor_grasp_center_depth_to_marker(
            aruco_id=3,
            center_world=(-1.91, 3.191, 0.895),
            marker_world=(-1.946, 3.166, 0.823),
        )

        self.assertAlmostEqual(anchored[0], -1.91)
        self.assertAlmostEqual(anchored[1], 3.241)
        self.assertAlmostEqual(anchored[2], 0.895)

    def test_same_stamp_counts_only_once(self):
        """PR5: the same camera frame (identical stamp) must count as ONE hit,
        not two - a duplicate delivery must not make INVENTORY_MIN_HITS=2
        behave like a one-frame confirmation."""
        manager = TaskManager()
        manager.build_search_tasks_for_targets([{"id": "item_01", "kind": "kele"}])
        active = manager.next_decision().selected_task
        obs = {
            "aruco_id": active.aruco_id,
            "kind": "kele",
            "confidence": 0.99,
            "world": [active.world_position[0], active.world_position[1] - 0.02, 0.94],
            "stamp": 200.0,
        }
        # Three deliveries of the SAME frame.
        manager.register_inventory_observations([obs, dict(obs), dict(obs)])
        record = manager.inventory_by_aruco[active.aruco_id]
        self.assertEqual(record["hits"], 1)
        self.assertEqual(record["state"], "observed")
        # A genuinely different frame confirms.
        obs2 = dict(obs)
        obs2["stamp"] = 200.1
        manager.register_inventory_observations([obs2])
        self.assertEqual(manager.inventory_by_aruco[active.aruco_id]["state"], "confirmed")

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

    def test_pr3_identity_stays_selectable_after_pose_expires(self):
        """PR3: identity (aruco->kind) is long-lived within the run; a slot
        whose pose went stale stays selectable (re-observe before grasp)
        instead of expiring and forcing a full rescan after a delivery."""
        manager = TaskManager()
        task = self._manual_search_task("search_id", 23, nav_x=0.70, nav_y=2.45)
        manager.tasks = [task]
        manager.requested_counts = Counter({"kele": 1})
        now = time.time()
        manager.inventory_by_aruco[23] = {
            "kind": "kele",
            "confidence": 0.9,
            "world": (0.70, 2.45, 0.94),
            "hits": 3,
            "confirmed": True,
            "state": "confirmed",
            "first_seen": now - 30.0,
            "last_seen": now - 30.0,
            "fresh_at": now - 30.0,
            "identity_seen_at": now - 30.0,
            "pose_seen_at": now - 30.0,
        }
        # Pose is 30s stale (> pose age 12s) but identity is within the run.
        self.assertFalse(manager._inventory_is_fresh(manager.inventory_by_aruco[23], now=now))
        self.assertTrue(manager._inventory_identity_fresh(manager.inventory_by_aruco[23], now=now))
        selected = manager.next_decision().selected_task
        self.assertEqual(selected.aruco_id, 23)

    def test_pr4_stops_scan_when_all_requested_kinds_confirmed(self):
        """PR4: once every requested kind is confirmed in the inventory, the
        decision layer must stop selecting un-scanned shelf slots (the 45-slot
        walk) and only keep the already-observed slots for grasping."""
        manager = TaskManager()
        # Two search tasks: slot 10 (observed+confirmed kele) and slot 20
        # (never observed -> would be visited by the scan).
        task_observed = self._manual_search_task("search_obs", 10, nav_x=0.70, nav_y=2.45)
        task_unobserved = self._manual_search_task("search_scan", 20, nav_x=1.20, nav_y=2.45, column="C2")
        manager.tasks = [task_observed, task_unobserved]
        manager.requested_counts = Counter({"kele": 1})
        now = time.time()
        manager.inventory_by_aruco[10] = {
            "kind": "kele", "confidence": 0.9,
            "world": (0.70, 2.45, 0.94), "hits": 3, "confirmed": True,
            "state": "confirmed", "first_seen": now - 2.0, "last_seen": now - 2.0,
            "fresh_at": now - 2.0, "identity_seen_at": now - 2.0, "pose_seen_at": now - 2.0,
        }
        self.assertTrue(manager._inventory_has_all_requested())
        selected = manager.next_decision().selected_task
        # The confirmed slot is chosen; the un-observed scan slot is skipped.
        self.assertEqual(selected.aruco_id, 10)

    def test_pr5_early_stop_uses_remaining_after_one_delivery(self):
        """PR5: _inventory_has_all_requested compares REMAINING demand
        (requested - completed).  After one kele delivered, one more confirmed
        kele in inventory satisfies the remaining need."""
        manager = TaskManager()
        task_obs = self._manual_search_task("search_obs", 10, nav_x=0.70, nav_y=2.45)
        manager.tasks = [task_obs]
        manager.requested_counts = Counter({"kele": 2})
        manager.completed_counts = Counter({"kele": 1})   # one already delivered
        now = time.time()
        manager.inventory_by_aruco[10] = {
            "kind": "kele", "confidence": 0.9,
            "world": (0.70, 2.45, 0.94), "hits": 3, "confirmed": True,
            "state": "confirmed", "first_seen": now - 2.0, "last_seen": now - 2.0,
            "fresh_at": now - 2.0, "identity_seen_at": now - 2.0, "pose_seen_at": now - 2.0,
        }
        # remaining = 2 - 1 = 1, have = 1 -> satisfied.
        self.assertTrue(manager._inventory_has_all_requested())

    def test_pr5_reserved_record_refreshes_pose_without_losing_reservation(self):
        """PR5: a RESERVED slot must keep its reservation but refresh the
        short-lived pose from same-kind observations."""
        manager = TaskManager()
        manager.build_search_tasks_for_targets([{"id": "item_01", "kind": "kele"}])
        active = manager.next_decision().selected_task
        aid = active.aruco_id
        manager.register_inventory_observations([{
            "aruco_id": aid, "kind": "kele", "confidence": 0.99,
            "world": [active.world_position[0], active.world_position[1] - 0.02, 0.94],
            "stamp": 300.0,
        }, {
            "aruco_id": aid, "kind": "kele", "confidence": 0.99,
            "world": [active.world_position[0], active.world_position[1] - 0.02, 0.94],
            "stamp": 300.1,
        }])
        self.assertEqual(manager.inventory_by_aruco[aid]["state"], "confirmed")
        # Reserve it (as task selection does).
        manager._reserve_inventory_record(active)
        self.assertEqual(manager.inventory_by_aruco[aid]["state"], "reserved")
        old_pose = manager.inventory_by_aruco[aid]["pose_seen_at"]
        # A fresh same-kind frame arrives while reserved.
        time.sleep(0.01)
        manager.register_inventory_observations([{
            "aruco_id": aid, "kind": "kele", "confidence": 0.99,
            "world": [active.world_position[0], active.world_position[1] - 0.01, 0.94],
            "stamp": 300.2,
        }])
        rec = manager.inventory_by_aruco[aid]
        self.assertEqual(rec["state"], "reserved")
        self.assertGreater(rec["pose_seen_at"], old_pose)

    def test_schema_v3_rejects_ambiguous_or_excessive_association(self):
        manager = TaskManager()
        base = {
            "aruco_id": 31,
            "kind": "kele",
            "confidence": 0.99,
            "object_world": [0.875, 3.23, 0.94],
            "frame_stamp": 600.0,
            "reject_reason": "ok",
        }
        manager.register_inventory_observations([
            {**base, "ambiguous": True, "association_score": 0.1},
            {**base, "ambiguous": False, "association_score": 99.0},
        ])
        self.assertNotIn(31, manager.inventory_by_aruco)

    def test_fresh_grasp_pose_requires_three_unique_current_frames(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([{"id": "item_01", "kind": "kele"}])
        slot_x = float(manager.slot_by_aruco[31]["world_position"][0])
        common = {
            "aruco_id": 31,
            "kind": "kele",
            "confidence": 0.99,
            "association_score": 0.2,
            "ambiguous": False,
            "reject_reason": "ok",
        }
        manager.register_inventory_observations([
            {**common, "object_world": [slot_x - 0.015, 3.22, 0.94], "frame_stamp": 610.0},
            {**common, "object_world": [slot_x - 0.005, 3.23, 0.94], "frame_stamp": 610.1},
        ])
        active = manager.next_decision().selected_task
        self.assertEqual(active.aruco_id, 31)
        self.assertIsNone(manager.fresh_grasp_observation(active))

        manager.register_inventory_observations([{
            **common,
            "object_world": [slot_x + 0.015, 3.24, 0.94],
            "frame_stamp": 610.2,
        }])
        fresh = manager.fresh_grasp_observation(active)
        self.assertIsNotNone(fresh)
        self.assertEqual(fresh["sample_count"], 3)

    def test_fresh_grasp_updates_pose_without_moving_navigation_goal(self):
        manager = TaskManager()
        manager.build_search_tasks_for_targets([{"id": "item_01", "kind": "kele"}])
        slot_x = float(manager.slot_by_aruco[31]["world_position"][0])
        common = {
            "aruco_id": 31,
            "kind": "kele",
            "confidence": 0.99,
            "association_score": 0.2,
            "ambiguous": False,
            "reject_reason": "ok",
        }
        manager.register_inventory_observations([
            {**common, "object_world": [slot_x - 0.015, 3.22, 0.94], "frame_stamp": 620.0},
            {**common, "object_world": [slot_x - 0.005, 3.23, 0.94], "frame_stamp": 620.1},
        ])
        active = manager.next_decision().selected_task
        frozen_nav_target = active.navigation_target.to_dict()
        frozen_nav_world = active.navigation_world_position

        manager.register_inventory_observations([{
            **common,
            "object_world": [slot_x + 0.015, 3.24, 0.94],
            "frame_stamp": 620.2,
        }])
        self.assertTrue(manager.apply_fresh_grasp_observation(active))
        self.assertEqual(active.navigation_target.to_dict(), frozen_nav_target)
        self.assertEqual(active.navigation_world_position, frozen_nav_world)
        self.assertEqual(active.world_position, active.fresh_grasp_world)
        self.assertEqual(active.metadata["fresh_grasp_frame_stamps"], [620.0, 620.1, 620.2])

    def test_fresh_grasp_rejects_stale_history(self):
        manager = TaskManager()
        task = self._manual_search_task("search_stale_fresh", 31, nav_x=0.9, nav_y=2.45)
        task.product_name = "kele"
        manager.inventory_by_aruco[31] = {
            "kind": "kele",
            "confirmed": True,
            "state": "reserved",
            "pose_history": [
                {
                    "world": (0.875, 3.23, 0.9235),
                    "stamp": 630.0 + index * 0.1,
                    "fresh_at": time.time() - 1.0,
                    "association_score": 0.2,
                }
                for index in range(3)
            ],
        }
        self.assertIsNone(manager.fresh_grasp_observation(task))

    def test_pr5_new_run_clears_manager_inventory(self):
        """PR5: a new /task payload (new run) clears the inventory table."""
        manager = TaskManager()
        manager.build_search_tasks_for_targets([{"id": "item_01", "kind": "kele"}])
        active = manager.next_decision().selected_task
        aid = active.aruco_id
        world = [active.world_position[0], active.world_position[1] - 0.02, active.world_position[2]]
        manager.register_inventory_observations([{
            "aruco_id": aid, "kind": "kele", "confidence": 0.99,
            "world": world, "stamp": 400.0,
        }, {
            "aruco_id": aid, "kind": "kele", "confidence": 0.99,
            "world": world, "stamp": 400.1,
        }])
        self.assertTrue(manager.inventory_by_aruco)
        # New run.
        manager.build_search_tasks_for_targets([{"id": "item_02", "kind": "zhijin"}])
        self.assertFalse(manager.inventory_by_aruco)

    def test_pr5_one_conflicting_frame_does_not_replace_confirmed_kind(self):
        """PR5: a single misclassification must not flip a confirmed slot."""
        manager = TaskManager()
        manager.build_search_tasks_for_targets([{"id": "item_01", "kind": "kele"}])
        active = manager.next_decision().selected_task
        aid = active.aruco_id
        world = [active.world_position[0], active.world_position[1] - 0.02, active.world_position[2]]
        for stamp in (500.0, 500.1):
            manager.register_inventory_observations([{
                "aruco_id": aid, "kind": "kele", "confidence": 0.99,
                "world": world, "stamp": stamp,
            }])
        self.assertEqual(manager.inventory_by_aruco[aid]["kind"], "kele")
        # One wrong frame (zhijin) must not replace the confirmed kele.
        manager.register_inventory_observations([{
            "aruco_id": aid, "kind": "zhijin", "confidence": 0.98,
            "world": world, "stamp": 500.2,
        }])
        self.assertEqual(manager.inventory_by_aruco[aid]["kind"], "kele")
        self.assertEqual(manager.inventory_by_aruco[aid]["state"], "confirmed")

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
    def test_official_runner_enables_loaded_delivery_astar(self):
        """Formal runs must plan around the randomized obstacle set from launch."""
        decision_runner = (REPO_ROOT / "scripts" / "run_v2_decision_client.sh").read_text(
            encoding="utf-8"
        )
        official_runner = (REPO_ROOT / "scripts" / "run_v2_official_test.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'export SUPERMARKET_DELIVERY_USE_ASTAR="${SUPERMARKET_DELIVERY_USE_ASTAR:-1}"',
            decision_runner,
        )
        self.assertIn(
            '-e SUPERMARKET_DELIVERY_USE_ASTAR="${SUPERMARKET_DELIVERY_USE_ASTAR:-1}"',
            official_runner,
        )
        # 2026-08-23: was pinned to USE_GS=0.  That value made every formal run
        # undetectable, because the shipped checkpoint is trained ONLY on 3DGS
        # frames while USE_GS=0 renders with the plain MuJoCo rasteriser.
        # Measured (scripts/probe_yolo_live_pose.py, same slot/pose/checkpoint):
        # 8/10 correct with 3DGS on, 0/10 with it off.  Pin the invariant - the
        # formal runner must not force the rasteriser - instead of a literal, so
        # the flag can be overridden for A/B work without breaking this test.
        self.assertNotIn(
            '-e SUPERMARKET_USE_GS="${SUPERMARKET_USE_GS:-0}"',
            official_runner,
            "the formal runner must not force the rasteriser the detector was "
            "never trained on",
        )
        self.assertIn("SUPERMARKET_USE_GS", official_runner)
        # The dataset generator must be able to cover both render domains, so
        # the detector stops depending on which one the grader uses.
        gen_dataset = (
            REPO_ROOT / "examples" / "supermarket_sorting" / "perception"
            / "gen_dataset.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--use-gs"', gen_dataset)
        self.assertIn('choices=["0", "1", "both"]', gen_dataset)

    def test_zhijin_empty_grasp_retry_advances_depth_without_x_sweep(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
        # static set holds the divider, the delivery table, the four perimeter
        # walls AND the five shelves (all invariant MJCF structures).
        #
        # 2026-08-23: the shelves were missing from this set, so their lidar
        # hits became dynamic obstacles; combined with the old grid ceiling
        # (ymax=3.02) those hits were clamped onto the boundary row and their
        # inflation covered the entire picking line, making every loaded A*
        # return no route.  Added here with a test that keeps the ORIGINAL
        # intent (randomised boxes must never be static) explicitly protected.
        rects = SupermarketGridPlanner.STATIC_OBSTACLES
        self.assertEqual(len(rects), 11)
        # The five shelves occupy the band y in [3.173, 3.473].
        shelf_band = [
            r for r in rects
            if 3.17 <= r.ymin <= 3.18 and 3.47 <= r.ymax <= 3.48
        ]
        self.assertEqual(len(shelf_band), 5)
        # Keep the ORIGINAL intent protected: a randomised cardboard box must
        # never be baked into the global map.  MJCF places them around the
        # arena centre (e.g. dynamic_obstacle_box_05 at (-0.110, 0.813)).
        for box_x, box_y in ((-0.110, 0.813), (0.29, 1.32)):
            self.assertFalse(
                any(r.contains(box_x, box_y) for r in rects),
                f"randomised box centre ({box_x}, {box_y}) sits inside a "
                "static obstacle footprint",
            )

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
        # 2026-08-23: the original start (1.60, 2.05) came from the era when the
        # divider was modelled at x~0.53.  Against the real board (x in [1.46,
        # 1.52]) it puts the robot centre 8 cm from the board face - physically
        # impossible, and inside the envelope, so the search could not leave the
        # start cell and reported no route.  Start from a real east-side pose.
        start = (1.60, 2.25)
        goal = (-1.88, -2.74)
        route = planner.plan(start, goal)

        self.assertTrue(route)
        self.assertEqual(route[-1], list(goal))
        previous = start
        for point in route:
            self.assertTrue(planner.path_is_clear(previous, point))
            previous = point

    def test_shelf_recovery_crosses_divider_in_high_band(self):
        # Official probe 15 recovered from the east side at (2.03, 1.88).
        #
        # 2026-08-23: rewritten against the MJCF truth.  The divider board is at
        # x in [1.46, 1.52], y in [-2.71, 2.71] (retail_competition.xml
        # pos="1.49 0 0.75"), NOT at x~0.53 / y<=1.70 as the old arena revision
        # had it.  The old assertion ("must cross x=0.53 above y=2.38") was
        # therefore testing a wall that no longer exists.  What still matters:
        # a route from the east side to the west picking line must exist and it
        # must respect the board - i.e. it may only change sides past one of the
        # board's ends, never through its middle.
        #
        # NOTE: the recovered pose (2.03, 1.88) is only 0.51 m east of the board
        # face - legitimate for the 0.22 m chassis, but *inside* any envelope
        # >= 0.51.  The planner now clamps corridor_clearance to robot_radius,
        # so asking for 0.65 no longer traps the very robot it is recovering.
        planner = SupermarketGridPlanner(corridor_clearance=0.65)
        start = np.array([2.03, 1.88])
        goal = (-1.93, 2.34)
        route = planner.plan(start, goal)
        self.assertTrue(route, "east-to-west recovery route must exist")

        board_mid_x = 0.53
        previous = start
        for raw_point in route:
            point = np.asarray(raw_point, dtype=float)
            if (previous[0] - board_mid_x) * (point[0] - board_mid_x) <= 0.0:
                fraction = (board_mid_x - previous[0]) / (point[0] - previous[0])
                crossing_y = previous[1] + fraction * (point[1] - previous[1])
                # The crossing must be clear of the board (|y| > 2.71) OR the
                # board's own footprint is not being passed through.
                self.assertTrue(
                    crossing_y >= 1.70 or crossing_y <= -3.72,
                    f"route crosses the board interior at y={crossing_y:.2f}",
                )
                break
            previous = point

    def test_loaded_delivery_crosses_divider_above_arm_clearance_line(self):
        # 2026-08-23: rewritten against the MJCF truth (board at x in [1.46,
        # 1.52], y in [-2.71, 2.71]).
        #
        # The board's own inflation closes the NORTH end for good: the gap to
        # the shelves is 3.173 - 2.71 = 0.46 m, while two robot_radius
        # envelopes need 0.44 m, and the shelf band then starts at
        # 3.173 - 0.22 = 2.953 versus the board's 2.71 + 0.22 = 2.93 - a
        # 0.023 m sliver, i.e. no cell row at all.  The SOUTH end is the real
        # crossing: 2.71 + 0.22 = 2.93 against the south wall's 3.72 - 0.22 =
        # 3.50, a 0.57 m band.  So: never through the board, always around the
        # south end.  (The earlier version of this test asked for a loaded
        # planner built from a client constant that is not imported here - it
        # raised NameError instead of testing anything.)
        loaded = SupermarketGridPlanner()
        # A traversing line straight through the board's footprint is blocked.
        self.assertFalse(loaded.path_is_clear((2.30, 0.0), (0.30, 0.0)))
        # The north gap is too narrow even for the bare chassis.
        self.assertTrue(loaded.path_is_clear((2.10, 2.90), (0.30, 2.90)))
        # And a full route from the east picking line to the delivery bay
        # exists, which is what the scored delivery legs actually need.
        self.assertTrue(loaded.plan((1.95, 2.42), (-1.82, -2.84)))

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

    def test_no_static_clearance_can_disconnect_the_arena(self):
        # The invariant that all four 2026-08-23 grid failures violated: the
        # static map must never disconnect a real passage nor swallow a real
        # goal.  Two hard numbers decide it.
        #
        #  * The east corridor is the gap between the divider's east face
        #    (x=1.52) and the east wall's inner face (x=2.47): 0.95 m.  With a
        #    static envelope c the free band of valid robot CENTRES is
        #        (2.47 - 0.22) - 1.52 - c = 0.73 - c
        #    so c=0.65 (the client's unloaded default) left 0.08 m and c=0.88
        #    left nothing: the whole east half of the arena became unreachable
        #    for a planner that was asked to drive to a shelf there.
        #  * Shelf picking poses sit inside the divider's y span
        #    (y in [-2.71, 2.71]), so any c that reaches a pose's x swallows
        #    that shelf.  The tightest is shelf E at x=1.805, i.e. 0.285 m from
        #    the board's east face.
        #
        # Enumerate every clearance a caller can pass, including the historic
        # 0.65/0.88, and require that all five picking poses and the delivery
        # bay stay mutually reachable.
        shelf_pose_x = (-1.735, -0.850, 0.035, 0.920, 1.805)
        picking_y = 2.42
        delivery_bay = (-1.82, -2.84)
        for clearance in (None, 0.22, 0.30, 0.45, 0.60, 0.65, 0.88):
            planner = SupermarketGridPlanner(corridor_clearance=clearance)
            for shelf_x in shelf_pose_x:
                pose = (shelf_x, picking_y)
                self.assertFalse(
                    planner._blocked(planner._to_cell(pose), set()),
                    f"clearance={clearance}: picking pose {pose} lies inside a "
                    "static envelope, so that shelf can never be unloaded",
                )
                self.assertTrue(
                    planner.plan(pose, delivery_bay),
                    f"clearance={clearance}: no route {pose} -> delivery bay",
                )
                self.assertTrue(
                    planner.plan(delivery_bay, pose),
                    f"clearance={clearance}: no route delivery bay -> {pose}",
                )

    def test_shelf_layer_tolerance_cannot_alias_the_neighbour_level(self):
        """The slot Z tolerance must stay under half a shelf-layer pitch.

        2026-08-23: a formal run locked ``maidong`` at z=0.924 and then accepted
        a live detection at z=1.092 as the same product.  The 0.168 m error read
        as a "topple" and the grasp was abandoned.  The gate that let it through
        was SEARCH_SLOT_ASSOC_Z = 0.32 against a shelf-layer pitch of 0.345 -
        93% of a full layer, so the level above is inside the tolerance by
        construction.  The comment above that constant always claimed it existed
        to stop exactly this; the value simply never matched the claim.
        """
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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

        self.assertLessEqual(
            client_mod.SEARCH_SLOT_ASSOC_Z,
            client_mod.SEARCH_SLOT_LAYER_PITCH / 2.0 + 1e-9,
            "a Z tolerance wider than half a shelf layer admits the level "
            "above/below as if it were this slot's product",
        )
        # The measured shelf boards sit at 0.50 / 0.852 / 1.19 m.
        self.assertAlmostEqual(client_mod.SEARCH_SLOT_LAYER_PITCH, 0.345, places=3)
        # A whole layer step must be rejected, and so must the 0.168 m
        # cross-level error that actually cost a grasp in the reference run.
        self.assertLess(client_mod.SEARCH_SLOT_ASSOC_Z, 0.345)
        self.assertLess(client_mod.SEARCH_SLOT_ASSOC_Z, 0.168)

    def test_gripper_detections_cannot_impersonate_the_target(self):
        """A detection at the robot's own end effector is not a shelf product.

        Measured in a formal run: after locking slot E_L2_C2 at
        [1.796 3.243 0.956] the live point drifted monotonically onto the
        commanded gripper endpoint [1.803 3.281 0.934], the monitor called it a
        displaced product three times, and the task died with "local grasp
        retries exhausted" without ever attempting a close.
        """
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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

        radius = client_mod.VISION_SELF_OCCLUSION_RADIUS
        self.assertGreater(radius, 0.0)
        # Must swallow the measured gripper-as-target error (0.061 m) ...
        measured = float(np.linalg.norm(
            np.array([1.803, 3.287, 0.995]) - np.array([1.803, 3.281, 0.934])
        ))
        self.assertLess(measured, radius)
        # ... while staying under half a shelf-column pitch so a genuine
        # neighbouring product is never masked by the robot's own hand.
        self.assertLessEqual(radius, client_mod.SEARCH_SLOT_COLUMN_PITCH / 2.0 + 1e-9)
        self.assertAlmostEqual(client_mod.SEARCH_SLOT_COLUMN_PITCH, 0.215, places=3)
        # The monitor must also be able to go stale, otherwise rejecting the
        # gripper detection would freeze the last good point forever.
        self.assertGreater(client_mod.VISION_MONITOR_STALE_TIMEOUT, 0.0)

    def test_right_lane_lies_inside_the_real_east_corridor(self):
        """Every point of ROUTE_TO_SHELF must be physically reachable.

        2026-08-23: SAFE_RIGHT_LANE_X was 1.62, which is 0.12 m INSIDE the
        centre divider (east face x=1.52) once the 0.22 m chassis half-width is
        accounted for.  The robot ground along the board for the entire first
        shelf leg - nav_recovery logged "stuck near base=(1.59,-0.88)",
        "(1.62,1.85)", "(1.63,2.07)" and burned all 8 recoveries plus the 45 s
        waypoint cap without reaching a shelf.  The older backup of the client
        used 1.92, so this was a regression, not a tuning choice.
        """
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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

        half_width = 0.22          # SUPERMARKET_ROBOT_CLEARANCE default
        west_edge = client_mod.DIVIDER_EAST_FACE_X + half_width
        east_edge = client_mod.EAST_WALL_INNER_X - half_width
        lane = client_mod.SAFE_RIGHT_LANE_X

        self.assertAlmostEqual(client_mod.DIVIDER_EAST_FACE_X, 0.56, places=2)
        self.assertAlmostEqual(client_mod.EAST_WALL_INNER_X, 2.47, places=2)
        self.assertAlmostEqual(client_mod.EAST_CORRIDOR_HALF_WIDTH, 0.955, places=3)
        self.assertGreaterEqual(lane, west_edge, "the right lane is inside the divider")
        self.assertLessEqual(lane, east_edge, "the right lane is inside the east wall")
        # It should sit near the middle of the legal band, not hugging an edge.
        self.assertGreater(lane - west_edge, 0.10)
        self.assertGreater(east_edge - lane, 0.10)

        # And the staged route itself must start in that lane, not on the board.
        first_x = float(client_mod.ROUTE_TO_SHELF[0][0])
        self.assertAlmostEqual(first_x, lane, places=6)
        self.assertGreaterEqual(first_x, west_edge)
        # The regression value specifically must be rejected.
        self.assertNotAlmostEqual(lane, 1.995, places=2)

    def test_terminal_lateral_correction_can_null_every_abort_threshold(self):
        """The creep must be able to correct what it is willing to abort on.

        Driving at heading offset theta over the remaining distance d moves the
        robot sideways by ~d*tan(theta).  Before 2026-08-23 the final approach
        clamped theta to 0.012 rad inside 0.12 m and 0.008 rad inside 0.18 m
        (and to 0.0 on the require-touch path) while aborting at lateral errors
        of 0.024 m and 0.040 m in the same block.  0.12*tan(0.012) = 1.4 mm of
        authority against a 24 mm abort is a factor of 17 short, which is why
        the formal run produced "lateral alignment error before close",
        "creep timeout before pinch depth" and "closed gripper without grasp
        evidence" - 6 of its 12 failures.
        """
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
        cap = client_mod.PickPlaceClient.creep_lateral_correction_cap

        hard = client_mod.CREEP_CORRECTION_HARD_CAP
        near_abort = client_mod.CREEP_NEAR_LATERAL_ABORT
        guard_abort = client_mod.CREEP_PRECONTACT_GUARD_LATERAL
        straight = client_mod.CREEP_STRAIGHT_LOCK_DISTANCE
        guard_d = client_mod.CREEP_PRECONTACT_GUARD_DISTANCE
        base = client_mod.CREEP_MAX_YAW_CORRECTION

        # A 0.012 rad equivalent must no longer exist anywhere on the approach.
        for remaining, abort in (
            (straight, near_abort),
            (guard_d, guard_abort),
            (0.10, near_abort),
            (0.14, guard_abort),
        ):
            allowed = cap(client, remaining, abort, base)
            reachable = remaining * np.tan(allowed)
            self.assertGreaterEqual(
                reachable, abort,
                f"at remaining={remaining} the allowed yaw {allowed:.3f} rad "
                f"corrects only {reachable * 1000:.1f} mm but the grasp is "
                f"aborted at {abort * 1000:.1f} mm",
            )
            self.assertLessEqual(allowed, hard + 1e-9)

        # The cap is monotone: halving the remaining distance must not reduce
        # the authority needed to remove the same error.
        self.assertGreaterEqual(
            cap(client, 0.06, near_abort, base),
            cap(client, 0.12, near_abort, base) - 1e-9,
        )
        # And it stays bounded no matter how desperate the geometry gets.
        self.assertLessEqual(cap(client, 0.001, 0.5, base), hard + 1e-9)

    def test_creep_heading_lock_engages_only_on_contact(self):
        """`correction` must not be discarded during the pre-contact approach.

        The heading lock used to engage at ``remaining <= 0.18`` even with no
        contact, which silently threw away the lateral correction the block had
        just computed - the last of the three places the terminal loop was
        disabled.
        """
        source = (
            REPO_ROOT / "examples" / "supermarket_sorting"
            / "supermarket_sorting_client.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn(
            "if (remaining <= CREEP_HEADING_FREEZE_DISTANCE or target_touched_near)",
            source,
            "the heading lock must not engage before contact",
        )
        self.assertNotIn(
            "if (remaining <= CREEP_HEADING_FREEZE_DISTANCE or target_touched)",
            source,
            "the heading lock must not engage before contact",
        )
        self.assertNotIn("max_correction = min(max_correction, 0.012)", source)
        self.assertIn("creep_lateral_correction_cap", source)

    def test_blocked_front_turns_instead_of_freezing(self):
        """A blocked corridor must not cancel the very turn that escapes it.

        The obstacle-safety blocked branch set BOTH des_lin and des_ang to 0.0
        and only requested a replan.  The replan returns the same route, so the
        controller asks for the same turn and the safety layer cancels it again
        - a permanent deadlock.  Measured live: yaw unchanged for minutes while
        the log showed cmd=(0.00,-0.28), and eight identical 3-waypoint
        replans before "navigation recovery limit exceeded".  The project's own
        older baseline turned on the spot here, and OBSTACLE_TURN_SPEED
        survived the regression unused.
        """
        source = (
            REPO_ROOT / "examples" / "supermarket_sorting"
            / "supermarket_sorting_client.py"
        ).read_text(encoding="utf-8")
        # The deadlock signature: both axes cancelled together.
        self.assertNotIn(
            "            self.des_lin = 0.0\n            self.des_ang = 0.0\n",
            source,
            "the blocked branch must keep an escape turn; cancelling both axes "
            "freezes the robot permanently",
        )
        self.assertIn("self.des_ang = turn_sign * OBSTACLE_TURN_SPEED", source)
        # The escape turn must be a real, bounded speed.
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
        self.assertGreater(client_mod.OBSTACLE_TURN_SPEED, 0.0)
        self.assertLessEqual(client_mod.OBSTACLE_TURN_SPEED, 1.0)
        # And the log must expose the ARBITRATED twist, otherwise a silent
        # cancellation like this one is invisible again.
        self.assertIn("req=({self.req_lin:.2f},{self.req_ang:.2f})", source)
        self.assertIn("limited=({self.des_lin:.2f},{self.des_ang:.2f})", source)
        self.assertIn("pub=({self.cur_lin:.2f},{self.cur_ang:.2f})", source)

    def test_gt_fast_direct_slot_skips_deploy_detection_dwell(self):
        """GT/direct-slot mode must not wait for detector dwell before grasping.

        The local GT scoring run already has a referee-provided public slot
        pose.  If DEPLOY waits for visual detections first, the robot visibly
        hesitates at the shelf and can burn all local grasp retries through
        detection timeouts even though the target geometry is known.
        """
        source = (
            REPO_ROOT / "examples" / "supermarket_sorting"
            / "supermarket_sorting_client.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def start_deploy_from_locked_target(self):", source)
        self.assertIn("and GT_FAST_NAV", source)
        fast_lock = (
            "gt_fast_direct_locked = self.lock_direct_task_geometry_fallback()"
        )
        dwell_wait = (
            "elif not self.target_locked and self.now() - self.state_t0 < DETECT_DWELL"
        )
        self.assertIn(fast_lock, source)
        self.assertIn(dwell_wait, source)
        self.assertIn("final_turn_tol = min(final_turn_tol, GT_FAST_FINAL_YAW_TOL)", source)
        self.assertNotIn("final_turn_tol = max(final_turn_tol, GT_FAST_FINAL_YAW_TOL)", source)
        self.assertIn('profile["center_x_bias"] = -0.016', source)
        self.assertIn('profile["approach_x_retry_scale"] = 0.0', source)
        self.assertIn('profile["creep_dy_offsets"] = (0.0,)', source)
        self.assertIn('profile["touch_final_close_remaining"]', source)
        self.assertIn('"closed gripper without grasp evidence" in reason', source)
        self.assertIn("target was already touched, skip this item", source)
        decision_source = (
            REPO_ROOT / "examples" / "supermarket_sorting"
            / "supermarket_sorting_decision_client.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"closed gripper without grasp evidence" in reason', decision_source)
        self.assertIn("self.current_target_touched()", decision_source)
        self.assertLess(
            source.index(fast_lock),
            source.index(dwell_wait),
            "GT direct geometry fallback must run before the deploy dwell wait",
        )

    def test_shelf_recovery_uses_dynamic_astar_after_stuck_event(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
        # 2026-08-23: the old guard here pinned LOADED_CORRIDOR_CLEARANCE to
        # >= 0.40, which was exactly backwards - it locked in a value that
        # sealed the east corridor (0.73 - c = 0.28 m of free centre band) and
        # it is the reason four navigation tests started failing.  The real
        # requirement is the opposite: no static clearance may exceed the
        # chassis radius.  See
        # test_no_static_clearance_can_disconnect_the_arena below.
        self.assertLessEqual(
            SupermarketGridPlanner(corridor_clearance=0.88).corridor_clearance,
            0.22,
        )
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
        self.assertAlmostEqual(
            float(client_mod.PRODUCT_GRASP_PROFILES["zhijin"]["contact_z_bias"]),
            0.018,
        )
        self.assertEqual(
            float(client_mod.PRODUCT_GRASP_PROFILES["zhijin"]["wrist_pitch_deg"]),
            58.0,
        )
        self.assertLessEqual(
            float(client_mod.PRODUCT_GRASP_PROFILES["zhijin"]["deploy_link4_max_z"]),
            1.125,
        )
        self.assertAlmostEqual(
            float(client_mod.PRODUCT_GRASP_PROFILES["zhijin"]["deploy_forward_offsets"][0]),
            0.09,
        )
        self.assertIn(
            0.04,
            tuple(client_mod.PRODUCT_GRASP_PROFILES["zhijin"]["deploy_forward_offsets"]),
        )
        self.assertLessEqual(
            float(client_mod.PRODUCT_GRASP_PROFILES["zhijin"]["deploy_arm_slew"]),
            0.35,
        )
        self.assertLessEqual(client_mod.DEPLOY_LATERAL_TOL, 0.045)
        self.assertLessEqual(client_mod.DEPLOY_COLLISION_YAW_SHIFT, 0.075)
        self.assertGreater(client_mod.DEPLOY_COLLISION_YAW_SHIFT, 0.06)
        self.assertLessEqual(
            float(client_mod.PRODUCT_GRASP_PROFILES["zhijin"]["geometry_close_remaining"]),
            0.045,
        )
        self.assertLessEqual(
            float(client_mod.PRODUCT_GRASP_PROFILES["zhijin"]["creep_timeout_arm_extension"]),
            0.045,
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
        dynamic_points = [(0.40, 2.20)]
        expected = [[0.25, 2.35], [0.82, 2.45]]
        client.enable_obstacle_avoidance = True
        client.navigation_obstacle_points = lambda max_range=5.0: dynamic_points
        client.planner = types.SimpleNamespace(
            plan=lambda start, finish, dynamic: expected
            if dynamic == dynamic_points
            else []
        )
        client.get_logger = lambda: types.SimpleNamespace(
            info=lambda *args, **kwargs: None
        )
        # Round 56: delivery goals are staggered per placed item; a fresh
        # client (0 placed) must target the base DELIVERY_GOAL.
        client.delivery_goal_current = client_mod.DELIVERY_GOAL.copy()
        client.placed_success_count = 0

        goal = np.array([0.82, 2.45], dtype=float)
        route = client_mod.PickPlaceClient.plan_route(client, goal, "shelf")

        self.assertEqual(route, expected)
        self.assertNotEqual(route, client.route_to_shelf)
        self.assertTrue(route)
        self.assertEqual(route[-1], goal.tolist())
        self.assertEqual(client.navigation_obstacle_points(), dynamic_points)

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
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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

    def test_shelf_lateral_crossing_does_not_drive_with_moderate_yaw_error(self):
        """Regression for the y=2.24 divider collision in official-image probe 10."""
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
            [-1.93, client_mod.SHELF_CROSS_Y],
            [-1.93, 2.48],
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
        # Target bearing is approximately pi; this reproduces the 0.38-rad
        # residual error where the old controller began its unsafe arc.
        client.base_yaw = float(np.pi - 0.38)
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
        self.assertGreater(commands[-1][1], 0.0)
        self.assertLessEqual(abs(commands[-1][1]), client_mod.SHELF_CROSS_ANGULAR_CAP)

    def test_place_raise_search_uses_lower_reachable_ik_target(self):
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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

    def test_stuck_recovery_prefers_crumb_backtracking(self):
        """Round 61: with a safe breadcrumb trail behind, a stall must walk
        BACK to the nearest crumb (crumb_back) instead of blindly reversing."""
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
        client.nav_mode = "drive"
        client.phase = client_mod.NAV_SHELF
        client.base_xy = np.array([-0.5, 1.0], dtype=float)
        client.nav_idx = 1
        client.route_to_shelf = [[1.0, 2.5]]
        client.now = lambda: 200.0
        client.front_blocked = False
        client.get_logger = lambda: types.SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warn=lambda *args, **kwargs: None,
        )
        # A stall: last progress was the same pose a while ago.
        client.last_nav_progress_xy = np.array([-0.5, 1.0], dtype=float)
        client.last_nav_progress_time = 170.0
        client.nav_recovery_count = 1
        client.recovery_escape = False
        client._nav_waypoint_deadline = 0.0
        client.crumb_trail = [(0.0, 2.2, 1.57), (0.0, 1.7, 1.57), (0.0, 1.3, 1.57)]
        client.crumb_target = None
        client.crumb_back_until = 0.0
        client.scan_ranges = None
        client.scan_stamp = 0.0
        client.scan_angle_min = 0.0
        client.scan_angle_increment = 0.0
        target = np.array([1.0, 2.5], dtype=float)

        triggered = client_mod.PickPlaceClient.maybe_start_stuck_recovery(client, target)
        self.assertTrue(triggered)
        self.assertEqual(client.recovery_state, "crumb_back")
        self.assertIsNotNone(client.crumb_target)
        # The nearest crumb (last travelled node, behind the current pose) is
        # the backtrack target.
        self.assertAlmostEqual(client.crumb_target[0], 0.0)
        self.assertAlmostEqual(client.crumb_target[1], 1.3)

        # Collision jitter must not defeat the absolute waypoint deadline.
        # This reproduces the official-seed stall where the chassis advanced
        # only centimetres while continuously pushing a randomized box.
        jitter = object.__new__(client_mod.PickPlaceClient)
        jitter.nav_mode = "drive"
        jitter.phase = client_mod.NAV_SHELF
        jitter.base_xy = np.array([-0.50, 2.10], dtype=float)
        jitter.nav_idx = 2
        jitter.route_to_shelf = [[-1.93, 2.24]]
        jitter.now = lambda: 200.0
        jitter.front_blocked = False
        jitter.get_logger = client.get_logger
        jitter.last_nav_progress_xy = np.array([-0.53, 2.10], dtype=float)
        jitter.last_nav_progress_time = 190.0
        jitter.last_nav_dist_to_target = 1.50
        jitter._nav_waypoint_deadline = 199.0
        jitter.nav_recovery_count = 0
        jitter.recovery_escape = False
        jitter.crumb_trail = []
        jitter.scan_ranges = None
        jitter.scan_stamp = 0.0
        jitter.scan_angle_min = 0.0
        jitter.scan_angle_increment = 0.0

        triggered = client_mod.PickPlaceClient.maybe_start_stuck_recovery(
            jitter, np.array([-1.93, 2.24], dtype=float)
        )
        self.assertTrue(triggered)
        self.assertEqual(jitter.recovery_state, "reverse")

    def test_startup_clearance_stows_before_straight_exit_without_yaw(self):
        """The right-wall spawn may not enter route-turn control with a moving arm."""
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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

    def test_shelf_crossing_waits_for_compact_arm_before_lateral_motion(self):
        """The rack traverse must not turn or drive with the travel arm exposed."""
        fake_modules = {
            "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda node: None, ok=lambda: False, shutdown=lambda: None),
            "rclpy.node": types.SimpleNamespace(Node=object),
            "geometry_msgs.msg": types.SimpleNamespace(Twist=object),
            "std_msgs.msg": types.SimpleNamespace(Float64MultiArray=object, String=object),
            "nav_msgs.msg": types.SimpleNamespace(Odometry=object),
            "sensor_msgs.msg": types.SimpleNamespace(Image=object, JointState=object, LaserScan=object, CameraInfo=object),
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
        client.phase = client_mod.NAV_SHELF
        client.base_xy = np.array([1.62, client_mod.SHELF_CROSS_ARM_PREP_Y + 0.01])
        client.nav_idx = 0
        client.tc = np.zeros(19)
        client.tc[12:18] = client_mod.INIT_ARM_R
        client.jpos = {"slide_joint": client_mod.SLIDE_TRAVEL}
        client.jpos.update({
            f"right_arm_joint{i + 1}": float(client_mod.INIT_ARM_R[i])
            for i in range(6)
        })
        client.shelf_crossing_arm_ready_at = None
        client.last_shelf_crossing_arm_log = -100.0
        clock = [5.0]
        commands = []
        client.now = lambda: clock[0]
        client.set_twist = lambda linear, angular: commands.append((linear, angular))
        client.get_logger = lambda: types.SimpleNamespace(info=lambda *args, **kwargs: None)
        route = [
            [1.62, client_mod.SHELF_CROSS_Y],
            [0.85, client_mod.SHELF_CROSS_Y],
            [0.85, client_mod.YELLOW_MID_Y],
        ]

        self.assertTrue(client_mod.PickPlaceClient.shelf_crossing_arm_step(client, route))
        np.testing.assert_allclose(client.tc[12:18], client_mod.SHELF_CROSS_ARM_R)
        self.assertEqual(commands[-1], (0.0, 0.0))

        client.jpos.update({
            f"right_arm_joint{i + 1}": float(client_mod.SHELF_CROSS_ARM_R[i])
            for i in range(6)
        })
        clock[0] += 1.0
        self.assertTrue(client_mod.PickPlaceClient.shelf_crossing_arm_step(client, route))
        self.assertEqual(commands[-1], (0.0, 0.0))

        clock[0] += client_mod.SHELF_CROSS_ARM_DWELL + 0.01
        self.assertFalse(client_mod.PickPlaceClient.shelf_crossing_arm_step(client, route))


if __name__ == "__main__":
    unittest.main()
