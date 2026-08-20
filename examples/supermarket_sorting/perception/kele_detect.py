#!/usr/bin/env python3
"""
Multi-class perception node for the Supermarket Sorting task.

Mirrors the reference material_detection_client/yolo_detect.py architecture
and outputs poses in the WORLD frame (the client consumes world-frame
targets directly via arm_to()).

Pipeline
--------
  /head_camera/color/image_raw            (RGB,  bgr8 / rgb8)
  /head_camera/aligned_depth_to_color/... (depth, mono16 in mm)
  /head_camera/color/camera_info          (K)
  /joint_states + /odom                   (drive MMK2FK -> camera-in-world)
        |
        v  2-D detector backend (Blob / GT / YOLO)  -> bbox centre (u,v)
        v  pixel2cam: deproject (u,v,depth) with K  -> camera-frame point
        v  T_cam_world @ p_cam (MMK2FK headeye site) -> WORLD point
        |
        v  publish /supermarket_sorting/detections (vision_msgs/Detection3DArray, world frame)
           publish /supermarket_sorting/result_image (debug overlay)

The camera->world transform uses the repo's MMK2FK.get_head_camera_pose(),
fed with the live base pose (odom) + slide/head joints (joint_states).  The
'headeye' site already carries the OpenGL->OpenCV optical-frame flip, so the
deprojected point maps to world with NO extra axis swap (validated to
0.0 mm round-trip error).
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, JointState
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from std_msgs.msg import String

from discoverse.robots.mmk2.mmk2_fk import MMK2FK

from backends import GtProjectionBackend, BlobBackend, YoloBackend
from inventory import (
    associate_detections_to_markers,
    match_detections_to_markers,
    aruco_id_to_slot,
)

LAYOUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "retail_competition_layout.json")
RUNTIME_LAYOUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "runtime_layout.json")
DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", "supermarket_multiclass.pt")
MAX_RGB_DEPTH_SKEW_SEC = float(os.getenv("SUPERMARKET_RGB_DEPTH_SKEW_SEC", "0.12"))
DETECTIONS_TOPIC = os.getenv("SUPERMARKET_DETECTIONS_TOPIC", "/supermarket_sorting/detections").strip() or "/supermarket_sorting/detections"
LEGACY_DETECTIONS_TOPIC = os.getenv("SUPERMARKET_LEGACY_DETECTIONS_TOPIC", "/kele/detections").strip() or "/kele/detections"
RESULT_IMAGE_TOPIC = os.getenv("SUPERMARKET_RESULT_IMAGE_TOPIC", "/supermarket_sorting/result_image").strip() or "/supermarket_sorting/result_image"
LEGACY_RESULT_IMAGE_TOPIC = os.getenv("SUPERMARKET_LEGACY_RESULT_IMAGE_TOPIC", "/kele/result_image").strip() or "/kele/result_image"


def resolve_layout_path():
    if os.getenv("SUPERMARKET_ALLOW_RUNTIME_LAYOUT", "0") != "1":
        return LAYOUT_JSON
    override = os.getenv("SUPERMARKET_RUNTIME_LAYOUT_PATH", "").strip()
    if override:
        candidate = Path(override)
        if candidate.exists():
            return str(candidate)
    if os.path.exists(RUNTIME_LAYOUT_JSON):
        return RUNTIME_LAYOUT_JSON
    return LAYOUT_JSON


class KeleDetectNode(Node):
    def __init__(self, backend="blob", pub_res_img=True):
        super().__init__("kele_detect")
        self.bridge = CvBridge()
        self.pub_res_img = pub_res_img

        # camera intrinsics (from camera_info)
        self.K = None
        self._depth_msg = None
        self._depth_stamp_sec = None

        # live robot state for the camera->world transform
        self.fk = MMK2FK()
        self.base_pos = None        # [x, y, z]
        self.base_quat = None       # [w, x, y, z]
        self.slide = 0.0
        self.head = [0.0, 0.0]

        # detector backend (pluggable)
        self.backend_name = backend
        self.layout_path = resolve_layout_path()
        if backend == "gt":
            self.detector = GtProjectionBackend(self.layout_path)
        elif backend == "yolo":
            weights = (
                os.getenv("SUPERMARKET_BASELINE_WEIGHTS", "").strip()
                or os.getenv("SUPERMARKET_YOLO_WEIGHTS", "").strip()
                or DEFAULT_CKPT
            )
            self.detector = YoloBackend(os.path.expanduser(weights))
        else:
            self.detector = BlobBackend()
        if (
            backend == "yolo"
            and os.getenv("SUPERMARKET_ORDER", "official").strip().lower() == "official"
            and not getattr(self.detector, "is_official_multiclass", False)
            and os.getenv("SUPERMARKET_ALLOW_SINGLE_CLASS", "0") != "1"
        ):
            raise RuntimeError(
                "official anonymous mode requires a nine-class YOLO checkpoint "
                "with the official product names; use SUPERMARKET_ALLOW_SINGLE_CLASS=1 "
                "only for a non-scoring development test"
            )
        self.get_logger().info(f"kele_detect up; backend={backend}; layout={self.layout_path}")
        if backend == "gt" and os.getenv("SUPERMARKET_ALLOW_RUNTIME_LAYOUT", "0") != "1":
            # The GT backend projects STATIC layout positions. If the server
            # randomized the scene, every projection silently lands on the
            # wrong slot. The smoke-test script sets the flag; manual runs
            # must too, so fail loudly instead of producing wrong data.
            self.get_logger().warn(
                "[gt] reading the STATIC layout while the server may have "
                "randomized product positions (SUPERMARKET_RANDOMIZE). Set "
                "SUPERMARKET_ALLOW_RUNTIME_LAYOUT=1 to project the live "
                "runtime layout for development tests."
            )
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # PR2: sub-pixel corner refinement -> more stable marker centre and a
        # better basis for the image-plane association and later PnP.
        try:
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except Exception:
            pass
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # subscriptions
        self.create_subscription(CameraInfo, "/head_camera/color/camera_info",
                                 self.camera_info_cb, 10)
        self.create_subscription(Image, "/head_camera/aligned_depth_to_color/image_raw",
                                 self.depth_cb, 10)
        self.create_subscription(Image, "/head_camera/color/image_raw",
                                 self.rgb_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.js_cb, 10)
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom",
                                 self.odom_cb, 10)

        # publishers
        self.det_pub = self.create_publisher(Detection3DArray, DETECTIONS_TOPIC, 10)
        self.legacy_det_pub = self.create_publisher(Detection3DArray, LEGACY_DETECTIONS_TOPIC, 10)
        self.inventory_pub = self.create_publisher(String, "/supermarket_sorting/inventory_observations", 10)
        self.img_pub = self.create_publisher(Image, RESULT_IMAGE_TOPIC, 5)
        self.legacy_img_pub = self.create_publisher(Image, LEGACY_RESULT_IMAGE_TOPIC, 5)
        self.last_detection_log = 0.0

    # ---- state callbacks ----
    def camera_info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def depth_cb(self, msg: Image):
        self._depth_msg = msg
        self._depth_stamp_sec = self._stamp_sec(msg)

    @staticmethod
    def _stamp_sec(msg):
        stamp = msg.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def js_cb(self, msg: JointState):
        jp = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}
        self.slide = jp.get("slide_joint", self.slide)
        self.head = [jp.get("head_yaw_joint", self.head[0]),
                     jp.get("head_pitch_joint", self.head[1])]

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_pos = [p.x, p.y, p.z]
        self.base_quat = [q.w, q.x, q.y, q.z]

    # ---- camera->world transform from live state ----
    def camera_world_tmat(self):
        """4x4 camera(optical)->world built from odom + slide/head via MMK2FK."""
        if self.base_pos is None or self.base_quat is None:
            return None
        self.fk.set_base_pose(self.base_pos, self.base_quat)
        self.fk.set_slide_joint(float(self.slide))
        self.fk.set_head_joints([float(self.head[0]), float(self.head[1])])
        self.fk.set_left_arm_joints([0.0] * 6)
        self.fk.set_right_arm_joints([0.0] * 6)
        pos, quat = self.fk.get_head_camera_pose()   # quat wxyz, world
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        return T

    def pixel_to_cam(self, u, v, depth_m):
        """Deproject a pixel + metric depth to a camera-optical-frame point."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        return np.array([x, y, depth_m])

    @staticmethod
    def patch_depth_m(depth_img, u, v, r=4):
        """Median depth (m) over a patch, ignoring zero (invalid) pixels."""
        h, w = depth_img.shape[:2]
        y0, y1 = max(0, v - r), min(h, v + r + 1)
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        patch = depth_img[y0:y1, x0:x1].astype(np.float32)
        valid = patch[patch > 0]
        return float(np.median(valid)) * 1e-3 if len(valid) else 0.0

    @staticmethod
    def robust_bbox_depth_m(depth_img, x0, y0, x1, y1, *, frac=0.4, depth_max=2.0):
        """Robust surface depth over the CENTRAL region of a product box.

        PR2: a single centre pixel often lands on packaging, the shelf board
        behind the product, or a transparent edge.  We take the central
        ``frac`` of the box, keep only valid depths under ``depth_max`` (drops
        far background/shelf), then take the LOW percentile of the closest
        surface cluster - i.e. the nearest solid surface, which is the product
        face, not what is behind it.  Returns metres or 0.0 when unusable.
        """
        h, w = depth_img.shape[:2]
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        bw = max(4.0, x1 - x0)
        bh = max(4.0, y1 - y0)
        hw, hh = bw * frac / 2.0, bh * frac / 2.0
        u0 = max(0, int(cx - hw))
        u1 = min(w, int(cx + hw) + 1)
        v0 = max(0, int(cy - hh))
        v1 = min(h, int(cy + hh) + 1)
        if u1 <= u0 or v1 <= v0:
            return 0.0
        region = depth_img[v0:v1, u0:u1].astype(np.float32)
        valid = region[region > 0]
        if len(valid) == 0:
            return 0.0
        # Metres; drop far background (shelf behind product).
        valid_m = valid * 1e-3
        valid_m = valid_m[valid_m < depth_max]
        if len(valid_m) == 0:
            return 0.0
        # Nearest-surface cluster: low percentile of the near side.
        return float(np.percentile(valid_m, 15))

    # ---- main RGB callback ----
    def rgb_cb(self, msg: Image):
        if self.K is None or self._depth_msg is None:
            return
        rgb_stamp_sec = self._stamp_sec(msg)
        if (
            self._depth_stamp_sec is not None
            and rgb_stamp_sec > 0.0
            and self._depth_stamp_sec > 0.0
            and abs(rgb_stamp_sec - self._depth_stamp_sec) > MAX_RGB_DEPTH_SKEW_SEC
        ):
            # A moving head with mismatched RGB/depth frames creates a world
            # point that can be centimetres off. Wait for the matching frame.
            return
        T_cam_world = self.camera_world_tmat()
        if T_cam_world is None:
            return

        rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(self._depth_msg)  # mono16, mm

        dets = self.detector.detect(rgb, depth, self.K, T_cam_world)

        out = []
        vis = rgb.copy() if self.pub_res_img else rgb
        for det_index, d in enumerate(dets):
            u, v = int(d["x"]), int(d["y"])
            # PR2: use the central-box robust depth instead of a single-pixel
            # patch (single pixel can hit packaging/shelf/transparent edge).
            depth_m = self.robust_bbox_depth_m(
                depth, int(d["x"]) - int(d["w"]) // 2, int(d["y"]) - int(d["h"]) // 2,
                int(d["x"]) + int(d["w"]) // 2, int(d["y"]) + int(d["h"]) // 2,
            )
            if depth_m <= 0.0:
                continue
            p_cam = self.pixel_to_cam(u, v, depth_m)
            p_world = (T_cam_world @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]

            rec = {
                "class": d["class"],
                "conf": d.get("conf", 0.0),
                "world": p_world,
                "source_index": det_index,
            }
            # coord-bridge validation logging (GT backend only)
            if "gt_world_pos" in d:
                err = np.linalg.norm(p_world - d["gt_world_pos"]) * 1e3
                rec["gt_err_mm"] = err
                self.get_logger().info(
                    f"[{d.get('body','?')}] world={np.round(p_world,3)} "
                    f"gt={np.round(d['gt_world_pos'],3)} err={err:.2f}mm")
            out.append(rec)

            if self.pub_res_img:
                w, h = int(d["w"]), int(d["h"])
                cv2.rectangle(vis, (u - w // 2, v - h // 2),
                              (u + w // 2, v + h // 2), (0, 255, 0), 2)
                cv2.putText(vis, f"{d['class']} ({p_world[0]:.2f},{p_world[1]:.2f},{p_world[2]:.2f})",
                            (u - 60, v - h // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

        self.publish_detections(out, msg.header.stamp)
        self.publish_inventory_observations(rgb, dets, out, depth, T_cam_world, rgb_stamp_sec, vis)
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_detection_log > 1.0:
            self.get_logger().info(
                f"[perception_debug] raw={len(dets)} published={len(out)} backend={self.backend_name}"
            )
            self.last_detection_log = now
        if self.pub_res_img:
            image_msg = self.bridge.cv2_to_imgmsg(vis, "bgr8")
            self.img_pub.publish(image_msg)
            self.legacy_img_pub.publish(image_msg)

    def publish_detections(self, recs, stamp):
        msg = Detection3DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        for r in recs:
            det = Detection3D()
            det.header = msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(r["class"])
            hyp.hypothesis.score = float(r["conf"])
            hyp.pose.pose.position.x = float(r["world"][0])
            hyp.pose.pose.position.y = float(r["world"][1])
            hyp.pose.pose.position.z = float(r["world"][2])
            det.results.append(hyp)
            msg.detections.append(det)
        self.det_pub.publish(msg)
        self.legacy_det_pub.publish(msg)

    def publish_inventory_observations(self, rgb, detections, records, depth, T_cam_world, stamp_sec, vis=None):
        """Publish only unambiguous product-to-visible-marker observations."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        if ids is None or not records:
            return
        markers = [
            {"id": int(marker_id), "corners": corner.reshape(-1, 2).tolist()}
            for marker_id, corner in zip(ids.flatten(), corners)
            if 0 <= int(marker_id) <= 44
        ]
        if not markers:
            return
        by_source = {record["source_index"]: record for record in records}
        # Only depth-valid detections may claim a marker. A raw detection whose
        # depth was rejected would otherwise consume the marker and starve the
        # correct, depth-valid observation for the same slot.
        indexed_valid = [
            (index, det)
            for index, det in enumerate(detections)
            if index in by_source
        ]
        filtered = [det for _, det in indexed_valid]
        source_of_filtered = [index for index, _ in indexed_valid]
        observations = []
        # Round 62 (PR1): robust one-to-one matcher with rejection reasons.
        matches, assoc_details = match_detections_to_markers(filtered, markers)
        for match_index, match in enumerate(matches):
            source_index = source_of_filtered[match["detection_index"]]
            record = by_source.get(source_index)
            if record is None:
                continue
            if match.get("aruco_id") is None:
                # Diagnostic-only observation (not written to formal inventory):
                # report why this detection was NOT bound to a slot.
                observations.append({
                    "aruco_id": None,
                    "kind": record["class"],
                    "confidence": float(record["conf"]),
                    "reject_reason": match.get("reject_reason", "unbound"),
                    "ambiguous": bool(match.get("ambiguous", False)),
                    "world": [float(value) for value in record["world"]],
                    "stamp": float(stamp_sec),
                })
                continue
            marker = next((m for m in markers if m["id"] == match["aruco_id"]), None)
            if marker is None:
                continue
            marker_corners = np.asarray(marker["corners"], dtype=float)
            marker_u, marker_v = np.mean(marker_corners, axis=0).astype(int)
            # PR1 diagnostics: draw the product->tag link + score on the debug
            # image so every frame answers "bound to which ArUco, how good".
            if vis is not None:
                det = filtered[match["detection_index"]]
                pu, pv = int(det["x"]), int(det["y"])
                colour = (0, 255, 0) if not match.get("ambiguous") else (0, 165, 255)
                cv2.line(vis, (pu, pv), (int(marker_u), int(marker_v)), colour, 1)
                cv2.putText(
                    vis,
                    f"a{match['aruco_id']}:{match['score']:.2f}",
                    ((pu + int(marker_u)) // 2 + 4, (pv + int(marker_v)) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1,
                )
            marker_depth_m = self.patch_depth_m(depth, int(marker_u), int(marker_v))
            marker_world = None
            if marker_depth_m > 0.0:
                marker_cam = self.pixel_to_cam(marker_u, marker_v, marker_depth_m)
                marker_world = (T_cam_world @ np.array([
                    marker_cam[0], marker_cam[1], marker_cam[2], 1.0,
                ]))[:3]
            slot = aruco_id_to_slot(match["aruco_id"]) or {}
            observations.append({
                "aruco_id": match["aruco_id"],
                "slot_id": slot.get("slot_id"),
                "shelf": slot.get("shelf"),
                "level": slot.get("level"),
                "column": slot.get("column"),
                "kind": record["class"],
                "confidence": float(record["conf"]),
                "association_score": float(match["score"]),
                "reject_reason": match.get("reject_reason", "ok"),
                "ambiguous": bool(match.get("ambiguous", False)),
                "world": [float(value) for value in record["world"]],
                "marker_world": (
                    None if marker_world is None
                    else [float(value) for value in marker_world]
                ),
                "stamp": float(stamp_sec),
            })
        if observations:
            payload = String()
            payload.data = json.dumps({
                "schema_version": 2,
                "observations": observations,
            })
            self.inventory_pub.publish(payload)


def main():
    parser = argparse.ArgumentParser(description="supermarket perception node")
    parser.add_argument("--backend", default="blob",
                        choices=["blob", "gt", "yolo"],
                        help="2-D detector backend (default: blob)")
    parser.add_argument("--no-result-image", action="store_true",
        help="disable result image publishing")
    args = parser.parse_args()

    rclpy.init()
    node = KeleDetectNode(backend=args.backend, pub_res_img=not args.no_result_image)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
