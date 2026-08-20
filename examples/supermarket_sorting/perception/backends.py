#!/usr/bin/env python3
"""
Pluggable 2-D detector backends for the kele perception pipeline.

Each backend exposes a single method::

    detect(rgb, depth, K, T_cam_world=None) -> list[dict]

where every dict has::

    class  : str   鈥?always 'kele' for all backends here
    x, y   : int   鈥?pixel coords of the bbox centre
    w, h   : int   鈥?bbox size in pixels
    conf   : float 鈥?confidence in [0, 1]

The downstream node does pixel鈫抍amera-frame deprojection and the
camera鈫抴orld transform.  Backends do NOT need to know the world frame.

Exception: GtProjectionBackend (for coord-bridge validation) also
accepts T_cam_world to project GT world positions to pixels; it sets
an extra 'gt_world_pos' key in each result so the node can log the
round-trip error.

Usage (in the node):
    from perception.backends import GtProjectionBackend, BlobBackend, YoloBackend
    detector = BlobBackend()                       # black-background scene
    detector = GtProjectionBackend(layout_path)    # coord-bridge validation
    detector = YoloBackend(ckpt_path)              # final trained model
"""

import os
import json
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
OFFICIAL_CLASSES = {
    "sanmingzhi", "heweidao", "shupian", "zhijin", "maidong",
    "kele", "kouxiangtang", "pingguo", "chengzi",
}
_CLASS = os.getenv("SUPERMARKET_BLOB_CLASS", "generic_blob")


def stable_class_consensus(
    labels,
    *,
    min_samples: int = 3,
    min_ratio: float = 0.67,
):
    """Return a label only after a short, unambiguous temporal consensus.

    A single detector frame is not a safe inventory observation: occlusion by
    the gripper and depth edge noise can change the class for one frame.  This
    helper is intentionally dependency-free so the decision layer can test the
    rule without importing ROS or Ultralytics.
    """
    values = [str(label).strip() for label in labels if str(label).strip()]
    if len(values) < max(1, int(min_samples)):
        return None
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    label, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    ratio = count / float(len(values))
    if ratio < float(min_ratio):
        return None
    return label


def _safe_depth_m(depth_img: np.ndarray, cx: int, cy: int, r: int = 4) -> float:
    """Median depth (metres) of a square patch centred on (cx,cy), ignoring zeros."""
    h, w = depth_img.shape[:2]
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    patch = depth_img[y0:y1, x0:x1].astype(np.float32)
    valid = patch[patch > 0]
    return float(np.median(valid)) * 1e-3 if len(valid) > 0 else 0.0


# ---------------------------------------------------------------------------
# GtProjectionBackend
# ---------------------------------------------------------------------------
class GtProjectionBackend:
    """Validate the coordinate bridge.

    Projects each requested slot's GT world position to pixels via T_cam_world,
    then the node's pixel鈫抴orld path should reconstruct the same point.
    Attach 'gt_world_pos' to each detection dict for logging.

    This backend requires T_cam_world to be passed in (not None).
    """

    def __init__(self, layout_path: str):
        with open(layout_path, 'r') as f:
            layout = json.load(f)
        products = {
            item.strip()
            for item in os.getenv("SUPERMARKET_DETECT_PRODUCTS", "all").split(",")
            if item.strip()
        }
        if not products or products & {"all", "*"}:
            self.slots = [s for s in layout if s.get('object_kind')]
        else:
            self.slots = [s for s in layout if s.get('object_kind') in products]

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        T_cam_world: np.ndarray | None = None,
    ) -> list[dict]:
        """Project GT object positions to pixels; return in-frame detections."""
        if T_cam_world is None:
            return []  # can't project without the transform

        H, W = rgb.shape[:2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        T_world_cam = np.linalg.inv(T_cam_world)  # world 鈫?camera

        detections = []
        for slot in self.slots:
            pw = np.array(slot['world_position'] + [1.0], dtype=float)
            pc = T_world_cam @ pw   # camera frame
            if pc[2] <= 0.05:       # behind or too close
                continue
            u = int(fx * pc[0] / pc[2] + cx)
            v = int(fy * pc[1] / pc[2] + cy)
            if not (0 <= u < W and 0 <= v < H):
                continue            # out of frame
            # fake bbox (10 脳 10 px around projection)
            bbox_px = 20
            detections.append({
                'class': slot.get('object_kind', _CLASS),
                'x': u,
                'y': v,
                'w': bbox_px,
                'h': bbox_px,
                'conf': 1.0,
                'gt_world_pos': np.array(slot['world_position']),
                'body': slot['body'],
            })
        return detections


# ---------------------------------------------------------------------------
# BlobBackend
# ---------------------------------------------------------------------------
class BlobBackend:
    """Find non-black object blobs in the GS-rendered image.

    With a black background, every Gaussian-splatted object renders as a
    non-black pixel cluster.  Simple thresholding + connected components
    gives reliable bboxes with no model weights needed.

    Parameters
    ----------
    min_area   : minimum blob area in pixels (filters noise)
    max_area   : maximum blob area (filters full-frame glare)
    value_thr  : HSV value threshold (pixels brighter than this are "object")
    depth_min  : ignore detections with depth < this (metres; filters close noise)
    depth_max  : ignore detections with depth > this (metres; shelf is ~0.5 m away)
    """

    def __init__(
        self,
        min_area: int = 200,
        max_area: int = 80_000,
        value_thr: int = 15,
        depth_min: float = 0.15,
        depth_max: float = 2.50,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.value_thr = value_thr
        self.depth_min = depth_min
        self.depth_max = depth_max

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        T_cam_world: np.ndarray | None = None,
    ) -> list[dict]:
        # threshold: keep pixels with V > value_thr in HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 2] > self.value_thr).astype(np.uint8) * 255

        # morphological cleanup: close small holes, open stray pixels
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        detections = []
        for i in range(1, n_labels):  # skip label 0 (background)
            area = stats[i, cv2.CC_STAT_AREA]
            if not (self.min_area <= area <= self.max_area):
                continue
            cx_px = int(centroids[i, 0])
            cy_px = int(centroids[i, 1])
            w_px  = int(stats[i, cv2.CC_STAT_WIDTH])
            h_px  = int(stats[i, cv2.CC_STAT_HEIGHT])
            # depth gate: only accept blobs at reasonable shelf distance
            d = _safe_depth_m(depth, cx_px, cy_px)
            if d < self.depth_min or d > self.depth_max:
                continue
            detections.append({
                'class': _CLASS,
                'x': cx_px,
                'y': cy_px,
                'w': w_px,
                'h': h_px,
                'conf': 0.80,   # nominal confidence for blob detections
            })
        return detections


# ---------------------------------------------------------------------------
# YoloBackend  (drop-in final backend once kele.pt is trained)
# ---------------------------------------------------------------------------
class YoloBackend:
    """YOLOv8 detector (ultralytics).

    This backend fails fast on missing or invalid weights so formal runs do
    not "succeed" while silently returning empty detections.
    """

    def __init__(self, ckpt_path: str, conf_thresh: float = 0.65):
        self.conf_thresh = conf_thresh
        self.model = None
        self.class_names = {}
        self.class_count = 0
        self.is_official_multiclass = False
        self.ckpt_path = ckpt_path
        self.require_official_classes = (
            os.getenv("SUPERMARKET_YOLO_REQUIRE_OFFICIAL_CLASSES", "0").strip() == "1"
        )

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[YoloBackend] checkpoint not found: {ckpt_path}")

        try:
            import torch
            from ultralytics import YOLO

            requested_device = os.getenv("SUPERMARKET_YOLO_DEVICE", "auto").strip().lower()
            if requested_device in {"", "auto"}:
                device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
            elif requested_device == "cpu":
                device_name = "cpu"
            elif requested_device in {"cuda", "cuda:0"} and torch.cuda.is_available():
                device_name = "cuda:0"
            else:
                raise RuntimeError(
                    "SUPERMARKET_YOLO_DEVICE must be auto, cpu, or cuda:0; "
                    f"got {requested_device!r}"
                )
            device = torch.device(device_name)

            # Patch torch.load for pre-2.6 checkpoints.
            _orig = torch.load

            def _compat(*a, **kw):
                kw.setdefault("weights_only", False)
                return _orig(*a, **kw)

            torch.load = _compat
            try:
                self.model = YOLO(ckpt_path).to(device)
                self.model.model.eval()
                # A competition checkpoint may contain all product classes.
                # Never silently relabel every prediction as kele.
                names = getattr(self.model, "names", None)
                if names is None:
                    names = getattr(self.model.model, "names", {})
                self.class_names = names or {}
            finally:
                torch.load = _orig

            if isinstance(self.class_names, dict):
                class_count = len(self.class_names)
            else:
                class_count = len(self.class_names)
            self.class_count = class_count
            if class_count <= 0:
                raise RuntimeError(f"[YoloBackend] checkpoint {ckpt_path} exposes no class names")

            names = set(str(value) for value in (
                self.class_names.values() if isinstance(self.class_names, dict)
                else self.class_names
            ))
            self.is_official_multiclass = (
                class_count == len(OFFICIAL_CLASSES) and names == OFFICIAL_CLASSES
            )
            mode = "multi-class" if class_count > 1 else "single-class"
            print(
                f"[YoloBackend] loaded {ckpt_path}; classes={class_count} "
                f"({mode}); device={device_name}"
            )
            if self.require_official_classes and not self.is_official_multiclass:
                raise RuntimeError(
                    "[YoloBackend] official mode requires a nine-class checkpoint "
                    "with the official product names"
                )
            if class_count > 1 and not self.is_official_multiclass:
                print(
                    "[YoloBackend] warning: checkpoint class names do not match "
                    "the nine official product kinds"
                )
        except Exception as e:
            raise RuntimeError(f"[YoloBackend] failed to load model from {ckpt_path}: {e}") from e

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        T_cam_world: np.ndarray | None = None,
    ) -> list[dict]:
        if self.model is None:
            raise RuntimeError(
                f"[YoloBackend] detector is not initialized; checkpoint={self.ckpt_path}"
            )
        results = self.model(rgb, verbose=False)[0]
        detections = []
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf < self.conf_thresh:
                continue
            cls_id = int(box.cls.item())
            if isinstance(self.class_names, dict):
                # Ultralytics normally uses integer keys, but exported models
                # can expose JSON-style string keys.
                class_name = self.class_names.get(
                    cls_id, self.class_names.get(str(cls_id))
                )
            else:
                class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else None
            if class_name is None:
                continue
            x0, y0, x1, y1 = map(int, box.xyxy[0].cpu().numpy())
            detections.append({
                'class': str(class_name),
                'x': (x0 + x1) // 2,
                'y': (y0 + y1) // 2,
                'w': x1 - x0,
                'h': y1 - y0,
                'conf': conf,
            })
        return detections

