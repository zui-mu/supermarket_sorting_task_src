"""Dependency-free regression tests for the referee flow fixes.

The referee normally runs against MuJoCo; these tests drive the same logic
through small stand-ins for mj_model / mj_data so the fixed behaviours
(no repeat C2 for delivered items, no stuck S2/S3 flows, flow timeout) stay
covered without a simulation backend.

Run with::

    python -m unittest tests.test_referee_flow_fixes
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

CURRENT_FILE = Path(__file__).resolve()
PACKAGE_ROOT = CURRENT_FILE.parents[1]
for _path in (str(PACKAGE_ROOT),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from referee import Flow, Referee  # noqa: E402


class _Body:
    def __init__(self, name: str, body_id: int, xpos=None):
        self.name = name
        self._id = body_id
        self.xpos = np.asarray(xpos if xpos is not None else [0.0, 2.0, 0.6], dtype=float)
        self.xquat = np.array([1.0, 0.0, 0.0, 0.0])
        self.jntadr = [body_id]

    @property
    def id(self) -> int:
        return self._id


class _Geom:
    def __init__(self, name: str):
        self.name = name


class _Contact:
    def __init__(self, geom1: int, geom2: int):
        self.geom1 = geom1
        self.geom2 = geom2


class _Model:
    def __init__(self, bodies: dict[str, _Body]):
        self._bodies = bodies
        self._geoms = {}
        self.geom_bodyid = []
        for body_id, body in enumerate(self._bodies.values()):
            for _ in range(2):  # two geoms per body: geom id = 2*body_id (+1)
                self._geoms[len(self.geom_bodyid)] = _Geom(f"geom_of_{body.name}")
                self.geom_bodyid.append(body_id)

    @property
    def jnt_dofadr(self):
        # free-joint convention used by referee._speed: dof address == jntadr
        return list(range(200))

    def body(self, name: str) -> _Body:
        if name not in self._bodies:
            raise KeyError(name)
        return self._bodies[name]

    def geom(self, index: int) -> _Geom:
        return self._geoms[index]


class _Data:
    def __init__(self, model: _Model, time: float = 0.0):
        self.model = model
        self.time = time
        self.contact: list[_Contact] = []
        self.ncon = 0
        self.qvel = np.zeros(256)

    def body(self, name: str) -> _Body:
        return self.model.body(name)

    def site(self, name: str):
        return self._site

    def set_contacts(self, pairs: list[tuple[str, str]]):
        self.contact = []
        for left, right in pairs:
            self.contact.append(
                _Contact(self.model.body(left).id * 2, self.model.body(right).id * 2)
            )
        self.ncon = len(self.contact)


def _make_referee(objects, config_overrides=None):
    config = {"time_limit_s": 100000.0, "flow_timeout_s": 240.0, "ungripped_rest_s": 1.2}
    if config_overrides:
        config.update(config_overrides)
    bodies = {}
    robot_names = [
        "agv_link", "slide_link",
        "rgt_arm_link1", "rgt_arm_link2", "rgt_arm_link3",
        "rgt_arm_link4", "rgt_arm_link5", "rgt_arm_link6",
        "rgt_finger_left_link", "rgt_finger_right_link",
        "shelf_A", "delivery_table",
    ]
    for index, name in enumerate(robot_names):
        bodies[name] = _Body(name, index)
    for index, name in enumerate(objects, start=len(robot_names)):
        bodies[name] = _Body(name, index)
    model = _Model(bodies)
    referee = Referee(model, objects, objects, config=config)
    data = _Data(model)
    data._site = type("_Site", (), {"xpos": np.array([0.0, 0.0, 0.0])})()
    referee.reset(data)
    return referee, data


class RefereeFlowTests(unittest.TestCase):
    def _base_in_pick(self, data, xy=(0.0, 2.0)):
        data._site.xpos = np.array([xy[0], xy[1], 0.0])

    def test_delivered_object_does_not_repeat_topple_penalty(self):
        referee, data = _make_referee(["A", "B"])
        # A was delivered earlier: its position is far from its shelf origin.
        data.body("A").xpos[:] = [-1.9, -3.4, 0.9]
        referee.retired.add("A")
        # Only B moved beyond the topple threshold from its snapshot.
        data.body("B").xpos[:2] = data.body("B").xpos[:2] + [0.0, 0.20]
        hit = referee._toppled_other_object(data, None)
        self.assertEqual(hit, "B")

    def test_toppled_object_is_penalized_only_once(self):
        referee, data = _make_referee(["A", "B"])
        data.body("B").xpos[:2] = data.body("B").xpos[:2] + [0.0, 0.20]
        hit = referee._toppled_other_object(data, None)
        self.assertEqual(hit, "B")
        referee.toppled_penalized.add("B")
        self.assertIsNone(referee._toppled_other_object(data, None))

    def test_s3_binds_a_neighbouring_target_instead_of_freezing(self):
        referee, data = _make_referee(["A", "B"])
        self._base_in_pick(data)
        flow = referee.flow = Flow(0.0)
        flow.step = 2
        flow.touched = {"B"}  # the robot brushed B but actually grabbed A
        # A: gripped by both fingers and pulled far out of its slot.
        data.body("A").xpos[:] = data.body("A").xpos[:] + [0.0, 0.25, 0.0]
        data.set_contacts([
            ("rgt_finger_left_link", "A"),
            ("rgt_finger_right_link", "A"),
        ])
        referee.update(data)
        self.assertEqual(flow.step, 3)
        self.assertEqual(flow.target, "A")
        # The legitimately grasped object must not be charged as C2 topple in
        # the same tick S3 binds it.
        self.assertFalse(flow.toppled)

    def test_high_surface_drop_settles_the_flow(self):
        referee, data = _make_referee(["A", "B"])
        self._base_in_pick(data)
        flow = referee.flow = Flow(0.0)
        flow.step = 3
        flow.target = "A"
        # Object rests high on a shelf (z above drop_z), not gripped.
        data.body("A").xpos[:] = [0.0, 2.0, 0.85]
        flow.ungripped_since = 3.0
        data.time = 5.0
        referee.update(data)
        self.assertIsNone(referee.flow)
        self.assertEqual(len(referee.records), 1)
        self.assertTrue(referee.records[0]["dropped"])
        self.assertIn("A", referee.retired)

    def test_flow_timeout_cancels_stuck_flow_without_penalty(self):
        referee, data = _make_referee(["A", "B"], config_overrides={"flow_timeout_s": 100.0})
        self._base_in_pick(data)
        flow = referee.flow = Flow(0.0)
        flow.step = 2
        flow.t_last_advance = -200.0  # 200s without progress
        data.time = 0.0
        referee.update(data)
        self.assertIsNone(referee.flow)
        self.assertEqual(referee.records, [])
        self.assertNotIn("A", referee.retired)
        self.assertTrue(any("\u6d41\u7a0b\u4f5c\u5e9f" in event for event in referee.events))

    def test_touches_keep_accumulating_at_step_two(self):
        referee, data = _make_referee(["A", "B"])
        self._base_in_pick(data)
        flow = referee.flow = Flow(0.0)
        flow.step = 2
        flow.touched = {"A"}
        data.set_contacts([("rgt_arm_link6", "B")])
        referee.update(data)
        self.assertIn("B", flow.touched)


if __name__ == "__main__":
    unittest.main()
