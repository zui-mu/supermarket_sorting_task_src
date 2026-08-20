"""Tests for ArUco slot identity decoding and robust one-to-one association."""
import os
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

# Path to the perception package (inventory.py is a sibling of this test).
_PERCEPTION = Path(__file__).resolve().parent.parent / "perception"
sys.path.insert(0, str(_PERCEPTION))


def _fake_modules():
    return {
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
        "navigation.grid_planner": types.SimpleNamespace(SupermarketGridPlanner=object),
    }


def _marker(aid, u, v, size=8):
    half = size / 2.0
    return {"id": aid, "corners": [[u - half, v - half], [u + half, v - half],
                                     [u + half, v + half], [u - half, v + half]]}


def _det(x, y, w=40, h=40):
    return {"class": "kele", "x": x, "y": y, "w": w, "h": h}


def test_all_45_ids_decode():
    import inventory
    for aid in range(45):
        slot = inventory.aruco_id_to_slot(aid)
        assert slot is not None, aid
        assert slot["shelf_index"] == aid // 9
        assert slot["level"] == ("L1", "L2", "L3")[(aid % 9) // 3]
        assert slot["column"] == ("C1", "C2", "C3")[aid % 3]
    assert inventory.aruco_id_to_slot(45) is None
    assert inventory.aruco_id_to_slot(-1) is None
    print("OK 45 ids decode")


def test_adjacent_slots_do_not_cross():
    """A product above its tag must bind to ITS tag, not the neighbour."""
    import inventory
    # Three tags side by side (same row), products centred above each.
    tags = [_marker(0, 100, 200), _marker(1, 200, 200), _marker(2, 300, 200)]
    dets = [_det(100, 140), _det(200, 140), _det(300, 140)]
    matches, _ = inventory.match_detections_to_markers(dets, tags)
    bound = {m["aruco_id"] for m in matches if m["aruco_id"] is not None}
    assert bound == {0, 1, 2}, bound
    # One-to-one: exactly 3 products, 3 tags, all used once.
    assert len([m for m in matches if m["aruco_id"] is not None]) == 3
    print("OK adjacent slots no cross")


def test_two_products_cannot_claim_one_tag():
    """One tag, two overlapping detections -> only one binds, other rejected."""
    import inventory
    tags = [_marker(7, 200, 200)]
    dets = [_det(200, 140), _det(210, 140)]   # both above tag 7
    matches, _ = inventory.match_detections_to_markers(dets, tags)
    bound = [m for m in matches if m["aruco_id"] is not None]
    assert len(bound) == 1, f"expected one bound, got {len(bound)}"
    assert bound[0]["aruco_id"] == 7
    # The other detection is rejected (no free marker below).
    rejected = [m for m in matches if m["aruco_id"] is None]
    assert any(m["reject_reason"] == "no_marker_below" for m in rejected)
    print("OK two products one tag")


def test_ambiguous_two_markers_rejected():
    """A product centred exactly between two tags is ambiguous -> rejected."""
    import inventory
    tags = [_marker(0, 100, 200), _marker(1, 300, 200)]
    # Wide box spanning both tags (x0=70..x1=330) so BOTH are candidates with
    # nearly equal scores -> the pair is ambiguous and must be rejected.
    det = _det(200, 140, w=260, h=40)
    matches, _ = inventory.match_detections_to_markers([det], tags)
    assert matches[0]["aruco_id"] is None, matches
    assert matches[0]["ambiguous"] is True
    print("OK ambiguous rejected")


def test_no_marker_below_rejected():
    import inventory
    tags = [_marker(5, 100, 400)]       # tag far below
    det = _det(100, 100, w=40, h=40)
    matches, _ = inventory.match_detections_to_markers([det], tags)
    assert matches[0]["aruco_id"] is None
    assert matches[0]["reject_reason"] == "no_marker_below"
    print("OK no marker below")


class ArucoInventoryTest(unittest.TestCase):
    def test_all_45_ids_decode(self):
        test_all_45_ids_decode()

    def test_adjacent_slots_do_not_cross(self):
        test_adjacent_slots_do_not_cross()

    def test_two_products_cannot_claim_one_tag(self):
        test_two_products_cannot_claim_one_tag()

    def test_ambiguous_two_markers_rejected(self):
        test_ambiguous_two_markers_rejected()

    def test_no_marker_below_rejected(self):
        test_no_marker_below_rejected()


if __name__ == "__main__":
    unittest.main(verbosity=2)
