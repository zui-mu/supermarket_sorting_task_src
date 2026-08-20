#!/usr/bin/env python3
"""Generate a multi-class YOLO dataset from the offline simulator.

Runs INSIDE the Docker container (needs gsplat + discoverse + a GPU).  Builds
the retail_competition scene with the 3D Gaussian Splatting renderer on (same
config path as supermarket_sorting_server.build_config), places the MMK2 camera
near the baseline shelf-D approach pose, renders the head camera, and auto-labels
every visible kele bottle.

Auto-labelling (no manual boxes)
--------------------------------
For each visible product slot we project a class-specific 3-D box around its
GT world_position via
the live `headeye` site (camera-optical -> world, the exact transform the
perception node uses).  A rendered-depth gate at the projected centre rejects
occluded/off-screen bottles.  This works with the fitted retail background 3DGS,
where foreground segmentation is not reliable because the shelf/background is
also visible.
The simulator truth is used only for offline labels.  The runtime detector
never imports this module or reads runtime_layout.json.

Domain randomisation
--------------------
Because the real background will not be black, every kept frame is saved as the
raw render PLUS several variants where the black void is replaced with random
content (solid colour / noise / gradient / optional natural texture) and the
foreground gets mild brightness/contrast jitter.  Boxes are identical across a
frame's variants.

Output (YOLOv8 layout)
----------------------
    dataset/images/{train,val}/*.jpg
    dataset/labels/{train,val}/*.txt    # class cx cy w h  (normalised)
    dataset/data.yaml                   # names: the nine official classes

Run (headless, inside container):
    cd examples/supermarket_sorting
    MUJOCO_GL=egl python3 perception/gen_dataset.py --frames 12 --variants 0 --overwrite
"""
import os
import sys
import json
import math
import random
import argparse
import shutil
from pathlib import Path

import numpy as np
import cv2

TASK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_DIR.parents[1]
ASSETS_DIR = TASK_DIR / "models"
os.environ.setdefault("DISCOVERSE_ASSETS_DIR", str(ASSETS_DIR))
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(REPO_ROOT))

from scipy.spatial.transform import Rotation  # noqa: E402

from discoverse.robots_env.mmk2_base import MMK2Base, MMK2Cfg  # noqa: E402
from discoverse.utils import camera2k, get_site_tmat  # noqa: E402

LAYOUT_JSON = TASK_DIR / "retail_competition_layout.json"
SOURCE_XML = TASK_DIR / "mjcf" / "retail_competition.xml"
RUNTIME_XML = Path("/tmp/retail_competition_dataset_runtime.xml")

HEAD_CAM_ID = 0
FOVY_DEG = 45.29
IMG_W, IMG_H = 640, 480

# foreground / occlusion thresholds
VALUE_THR = 12          # pixel brighter than this (any channel) counts as object
DEPTH_TOL = 0.10        # m: |rendered_depth - projected_depth| gate for the target
MIN_AREA = 120          # px: drop specks
MAX_AREA = 60_000       # px: drop runaway blobs
MIN_WH, MAX_WH = 6, 360  # px bbox side limits
PRODUCT_NAMES = (
    "sanmingzhi", "heweidao", "shupian", "zhijin", "maidong",
    "kele", "kouxiangtang", "pingguo", "chengzi",
)
CLASS_IDS = {name: index for index, name in enumerate(PRODUCT_NAMES)}

MAX_FRAMES = 2000
MAX_VARIANTS = 5
MAX_ESTIMATED_OUTPUT_BYTES = 8 * 1024**3


# ---------------------------------------------------------------------------
# scene construction (mirrors supermarket_sorting_server.build_config)
# ---------------------------------------------------------------------------
def _local_robot_gs_model_dict():
    out = {}
    for name, path in MMK2Cfg.gs_model_dict.items():
        if path.startswith("mobile_chassis/mmk2/"):
            out[name] = path.replace("mobile_chassis/mmk2/", "mmk2/")
        elif path.startswith("manipulator/airbot_play/"):
            out[name] = path.replace("manipulator/airbot_play/", "airbot_play/")
        else:
            out[name] = path
    return out


def _write_runtime_xml():
    text = SOURCE_XML.read_text().replace("__REPO_ROOT__", str(TASK_DIR))
    RUNTIME_XML.write_text(text)
    return str(RUNTIME_XML)


def _resolve_background_ply():
    fit = ASSETS_DIR / "3dgs" / "shentoon" / "retail_background_fit.ply"
    if fit.exists():
        return "shentoon/retail_background_fit.ply"
    return "shentoon/dummy_background.ply"


def build_sim():
    """Build an MMK2Base with the GS renderer on (no ROS)."""
    cfg = MMK2Cfg()
    cfg.mjcf_file_path = _write_runtime_xml()
    cfg.use_gaussian_renderer = True
    cfg.enable_render = True
    cfg.headless = True

    layout = json.loads(LAYOUT_JSON.read_text())
    cfg.obj_list = [slot["body"] for slot in layout]
    cfg.gs_model_dict = _local_robot_gs_model_dict()
    cfg.gs_model_dict["background"] = _resolve_background_ply()
    for slot in layout:
        cfg.gs_model_dict[slot["body"]] = slot["gs_ply"]

    cfg.obs_rgb_cam_id = [HEAD_CAM_ID]
    cfg.obs_depth_cam_id = [HEAD_CAM_ID]
    cfg.lidar_s2_sim = False
    cfg.render_set = {"fps": 24, "width": IMG_W, "height": IMG_H}

    sim = MMK2Base(cfg)
    sim.reset()
    return sim


# ---------------------------------------------------------------------------
# camera projection from the live headeye site (matches perception node)
# ---------------------------------------------------------------------------
def head_cam_K():
    return camera2k(FOVY_DEG * math.pi / 180.0, IMG_W, IMG_H)


def T_cam_world(sim):
    """4x4 camera-optical -> world, read straight from the sim's headeye site."""
    return get_site_tmat(sim.mj_data, "headeye")


def project_world_to_px(pw, K, T_cw):
    """World point -> (u, v, depth_m). Returns None if behind camera."""
    T_wc = np.linalg.inv(T_cw)
    pc = T_wc @ np.array([pw[0], pw[1], pw[2], 1.0])
    if pc[2] <= 0.05:
        return None
    u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
    v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
    return float(u), float(v), float(pc[2])


# ---------------------------------------------------------------------------
# robot posing
# ---------------------------------------------------------------------------
def set_robot_pose(sim, base_xy, yaw, slide, head_pitch):
    """Write base/slide/head into qpos (layout matches MMK2FK) and forward."""
    import mujoco
    q = sim.mj_data.qpos
    q[0:3] = [base_xy[0], base_xy[1], 0.0]
    q[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]  # wxyz, z-rot
    q[9] = slide
    q[10] = 0.0           # head yaw
    q[11] = head_pitch
    sim.mj_data.qvel[:] = 0.0
    mujoco.mj_forward(sim.mj_model, sim.mj_data)


def sample_pose(rng, mode):
    """Randomised camera pose.

    baseline: concentrate near the real baseline stop in front of shelf D.
    wide: cover all shelf columns for multi-class detector training.
    """
    if mode == "wide":
        base_x = rng.uniform(-2.3, 2.3)
        base_y = rng.uniform(2.35, 2.95)
        yaw_jitter = 0.10
        slide = rng.uniform(0.0, 0.25)
        head_pitch = rng.uniform(-0.75, -0.35)
    else:
        base_x = float(np.clip(rng.normal(0.91, 0.22), 0.35, 1.35))
        base_y = float(np.clip(rng.normal(2.475, 0.16), 2.20, 2.82))
        yaw_jitter = 0.07
        slide = rng.uniform(0.06, 0.18)
        head_pitch = rng.uniform(-0.72, -0.42)
    yaw = math.pi / 2.0 + rng.uniform(-yaw_jitter, yaw_jitter)
    return [base_x, base_y], yaw, slide, head_pitch


# ---------------------------------------------------------------------------
# auto-labelling
# ---------------------------------------------------------------------------
def foreground_mask(rgb):
    """Non-black pixels -> object foreground (uint8 0/255)."""
    v = rgb.max(axis=2)
    mask = (v > VALUE_THR).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)
    return mask


def label_objects(rgb, depth_m, K, T_cw, slots):
    """Return visible object boxes as ``(class_id, x0, y0, x1, y1)``.

    Each slot is projected to a pixel, gated by rendered depth, and associated
    with the connected component containing that projection.
    """
    fg = foreground_mask(rgb)
    boxes = []
    for slot in slots:
        class_id = CLASS_IDS.get(str(slot.get("object_kind", "")).strip())
        if class_id is None:
            continue
        proj = project_world_to_px(slot["world_position"], K, T_cw)
        if proj is None:
            continue
        u, v, zp = proj
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < IMG_W and 0 <= vi < IMG_H):
            continue
        # depth gate: keep foreground whose rendered depth is near the target's
        dgate = (np.abs(depth_m - zp) < DEPTH_TOL).astype(np.uint8) * 255
        local = cv2.bitwise_and(fg, dgate)
        # the projection pixel must land on a kept (object) pixel
        if local[vi, ui] == 0:
            # nudge: search a small window for the nearest object pixel
            r = 6
            y0, y1 = max(0, vi - r), min(IMG_H, vi + r + 1)
            x0, x1 = max(0, ui - r), min(IMG_W, ui + r + 1)
            patch = local[y0:y1, x0:x1]
            ys, xs = np.where(patch > 0)
            if len(ys) == 0:
                continue
            vi, ui = y0 + int(ys[0]), x0 + int(xs[0])
        n, lab, stats, _ = cv2.connectedComponentsWithStats(local, connectivity=8)
        cid = lab[vi, ui]
        if cid == 0:
            continue
        area = stats[cid, cv2.CC_STAT_AREA]
        bx = stats[cid, cv2.CC_STAT_LEFT]
        by = stats[cid, cv2.CC_STAT_TOP]
        bw = stats[cid, cv2.CC_STAT_WIDTH]
        bh = stats[cid, cv2.CC_STAT_HEIGHT]
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        if not (MIN_WH <= bw <= MAX_WH and MIN_WH <= bh <= MAX_WH):
            continue
        boxes.append((class_id, bx, by, bx + bw, by + bh))
    return boxes


def label_kele(rgb, depth_m, K, T_cw, kele_slots):
    """Backward-compatible single-class wrapper for old validation scripts."""
    return [box[1:] for box in label_objects(rgb, depth_m, K, T_cw, kele_slots)]


def _patch_depth(depth_m, u, v, r=4):
    h, w = depth_m.shape[:2]
    y0, y1 = max(0, v - r), min(h, v + r + 1)
    x0, x1 = max(0, u - r), min(w, u + r + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    return float(np.median(valid)) if len(valid) else 0.0


def _collision_geom_world_points(sim, slot):
    """Return surface points from the item's current MuJoCo collision geom.

    This is intentionally based on the live model rather than hand-maintained
    per-class dimensions.  It remains an offline label generator only.
    """
    import mujoco

    model, data = sim.mj_model, sim.mj_data
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, slot["body"])
    if body_id < 0:
        return []
    geom_start = int(model.body_geomadr[body_id])
    geom_count = int(model.body_geomnum[body_id])
    points = []
    for geom_id in range(geom_start, geom_start + geom_count):
        center = np.asarray(data.geom_xpos[geom_id], dtype=float)
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        size = np.asarray(model.geom_size[geom_id], dtype=float)
        geom_type = int(model.geom_type[geom_id])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            local = np.array([
                [sx * size[0], sy * size[1], sz * size[2]]
                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
            ])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
            local = np.array([
                [size[0] * math.cos(angle), size[0] * math.sin(angle), z * size[1]]
                for z in (-1, 1) for angle in angles
            ])
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[geom_id])
            vertex_start = int(model.mesh_vertadr[mesh_id])
            vertex_count = int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(
                model.mesh_vert[vertex_start:vertex_start + vertex_count],
                dtype=float,
            )
            if len(vertices) == 0:
                continue
            lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            local = np.array([
                [x, y, z]
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])
            ])
        else:
            # Spheres and mesh collision hulls use the official geom bound.
            radius = float(model.geom_rbound[geom_id])
            local = np.array([
                [sx * radius, sy * radius, sz * radius]
                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
            ])
        points.extend(center + rotation @ point for point in local)
    return points


def _projected_geometry_bbox(sim, slot, K, T_cw, margin=1.04):
    points = _collision_geom_world_points(sim, slot)
    pts = [project_world_to_px(point, K, T_cw) for point in points]
    pts = [p for p in pts if p is not None]
    if len(pts) < 4:
        return None
    us = np.array([p[0] for p in pts])
    vs = np.array([p[1] for p in pts])
    x0, x1 = float(us.min()), float(us.max())
    y0, y1 = float(vs.min()), float(vs.max())
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    bw, bh = (x1 - x0) * margin, (y1 - y0) * margin
    x0 = int(max(0, round(cx - bw * 0.5)))
    x1 = int(min(IMG_W - 1, round(cx + bw * 0.5)))
    y0 = int(max(0, round(cy - bh * 0.5)))
    y1 = int(min(IMG_H - 1, round(cy + bh * 0.5)))
    if x1 <= x0 or y1 <= y0:
        return None
    bw, bh = x1 - x0, y1 - y0
    if not (MIN_WH <= bw <= MAX_WH and MIN_WH <= bh <= MAX_WH):
        return None
    return x0, y0, x1, y1


def label_projected(sim, depth_m, K, T_cw, slots):
    """Label visible slots as ``(class_id, x0, y0, x1, y1)`` tuples."""
    boxes = []
    for slot in slots:
        kind = str(slot.get("object_kind", "")).strip()
        class_id = CLASS_IDS.get(kind)
        if class_id is None:
            continue
        pw = slot["world_position"]
        proj = project_world_to_px(pw, K, T_cw)
        if proj is None:
            continue
        u, v, zp = proj
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < IMG_W and 0 <= vi < IMG_H):
            continue
        d = _patch_depth(depth_m, ui, vi)
        if d <= 0.0 or abs(d - zp) > DEPTH_TOL:
            continue
        box = _projected_geometry_bbox(sim, slot, K, T_cw)
        if box is not None:
            boxes.append((class_id, *box))
    return boxes


def label_kele_projected(sim, depth_m, K, T_cw, kele_slots):
    """Backward-compatible single-class wrapper for old validation scripts."""
    return [box[1:] for box in label_projected(sim, depth_m, K, T_cw, kele_slots)]


# ---------------------------------------------------------------------------
# domain randomisation
# ---------------------------------------------------------------------------
def _random_background(rng, h, w):
    """One random background image (BGR uint8)."""
    kind = rng.integers(0, 4)
    if kind == 0:                                   # solid colour
        col = rng.integers(0, 256, size=3)
        bg = np.ones((h, w, 3), np.uint8) * col.astype(np.uint8)
    elif kind == 1:                                 # uniform noise
        bg = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    elif kind == 2:                                 # vertical/horizontal gradient
        c0 = rng.integers(0, 256, size=3).astype(np.float32)
        c1 = rng.integers(0, 256, size=3).astype(np.float32)
        if rng.random() < 0.5:
            t = np.linspace(0, 1, w)[None, :, None]
        else:
            t = np.linspace(0, 1, h)[:, None, None]
        bg = (c0[None, None, :] * (1 - t) + c1[None, None, :] * t).astype(np.uint8)
        bg = np.broadcast_to(bg, (h, w, 3)).copy()
    else:                                           # blurry low-freq noise
        small = rng.integers(0, 256, size=(h // 16, w // 16, 3), dtype=np.uint8)
        bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        bg = cv2.GaussianBlur(bg, (0, 0), 5)
    return bg


def _jitter_foreground(rng, rgb):
    """Mild brightness/contrast jitter on the whole frame."""
    alpha = float(rng.uniform(0.85, 1.15))   # contrast
    beta = float(rng.uniform(-18, 18))       # brightness
    return cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)


def domain_randomise(rng, rgb, n_variants):
    """Return list of DR variants (BGR uint8). Background void -> random."""
    fg = foreground_mask(rgb)
    fg3 = (fg > 0)[:, :, None]
    h, w = rgb.shape[:2]
    out = []
    for _ in range(n_variants):
        bg = _random_background(rng, h, w)
        comp = np.where(fg3, rgb, bg)
        comp = _jitter_foreground(rng, comp)
        out.append(comp)
    return out


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def boxes_to_yolo(boxes, *, legacy_class_id=None):
    lines = []
    for item in boxes:
        if len(item) == 4:
            if legacy_class_id is None:
                raise ValueError("box is missing class id")
            class_id, x0, y0, x1, y1 = legacy_class_id, *item
        elif len(item) == 5:
            class_id, x0, y0, x1, y1 = item
        else:
            raise ValueError(f"invalid box tuple: {item!r}")
        cx = (x0 + x1) / 2.0 / IMG_W
        cy = (y0 + y1) / 2.0 / IMG_H
        bw = (x1 - x0) / IMG_W
        bh = (y1 - y0) / IMG_H
        lines.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def save_sample(out_dir, split, name, img_bgr, boxes):
    img_path = out_dir / "images" / split / f"{name}.jpg"
    lab_path = out_dir / "labels" / split / f"{name}.txt"
    if not cv2.imwrite(str(img_path), img_bgr):
        raise RuntimeError(f"failed to write image: {img_path}")
    lab_path.write_text(boxes_to_yolo(boxes), encoding="ascii")


def main():
    ap = argparse.ArgumentParser(description="generate nine-class YOLO dataset (offline GS)")
    ap.add_argument("--frames", type=int, default=200, help="base poses to render (1-2000)")
    ap.add_argument("--variants", type=int, default=1, help="DR variants per frame (0-5)")
    ap.add_argument("--pose-mode", default="wide", choices=["baseline", "wide"],
                    help="wide covers all shelf columns; baseline is a development subset")
    ap.add_argument("--label-mode", default="geometry", choices=["geometry", "projected", "mask"],
                    help="geometry projects live MuJoCo collision geometry; mask is diagnostic only")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default=str(TASK_DIR / "perception" / "dataset"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", default="all",
                    help="comma-separated official classes, or all")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing output directory")
    ap.add_argument("--debug-overlay", action="store_true",
                    help="also dump boxed previews to <out>/debug")
    args = ap.parse_args()

    if not 1 <= args.frames <= MAX_FRAMES:
        ap.error(f"--frames must be between 1 and {MAX_FRAMES}")
    if not 0 <= args.variants <= MAX_VARIANTS:
        ap.error(f"--variants must be between 0 and {MAX_VARIANTS}")
    if not 0.0 <= args.val_frac < 1.0:
        ap.error("--val-frac must be in [0, 1)")
    requested = [item.strip() for item in str(args.classes).split(",") if item.strip()]
    selected_names = list(PRODUCT_NAMES) if not requested or requested == ["all"] else requested
    unknown = sorted(set(selected_names) - set(PRODUCT_NAMES))
    if unknown:
        ap.error(f"unknown official classes: {', '.join(unknown)}")
    estimated_bytes = args.frames * (1 + args.variants) * IMG_W * IMG_H * 3 // 3
    if estimated_bytes > MAX_ESTIMATED_OUTPUT_BYTES:
        ap.error(
            f"estimated output is {estimated_bytes / 1024**3:.2f} GiB; "
            f"reduce frames/variants below {MAX_ESTIMATED_OUTPUT_BYTES / 1024**3:.0f} GiB"
        )

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    if out_dir.exists():
        if not args.overwrite:
            ap.error(f"output exists: {out_dir}; pass --overwrite explicitly")
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    if args.debug_overlay:
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)

    sim = build_sim()
    K = head_cam_K()
    layout = json.loads(LAYOUT_JSON.read_text())
    slots = [s for s in layout if s.get("object_kind") in selected_names]
    background_ply = _resolve_background_ply()
    print(f"[gen] {len(slots)} slots; classes={selected_names}; rendering {args.frames} frames "
          f"x (1 + {args.variants}) variants, pose={args.pose_mode}, "
          f"labels={args.label_mode}, background={background_ply}")

    n_frames = 0
    n_boxes = 0
    n_imgs = 0
    attempts = 0
    max_attempts = args.frames * 4
    class_counts = {name: 0 for name in selected_names}
    while n_frames < args.frames and attempts < max_attempts:
        attempts += 1
        base_xy, yaw, slide, pitch = sample_pose(rng, args.pose_mode)
        set_robot_pose(sim, base_xy, yaw, slide, pitch)
        sim.render()
        rgb = sim.img_rgb_obs_s[HEAD_CAM_ID]              # uint8 HxWx3, RGB
        depth_m = sim.img_depth_obs_s[HEAD_CAM_ID]        # float, metres
        T_cw = T_cam_world(sim)
        if args.label_mode == "mask":
            boxes = label_objects(rgb, depth_m, K, T_cw, slots)
        else:
            boxes = label_projected(sim, depth_m, K, T_cw, slots)
        if not boxes:
            continue

        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        split = "val" if rng.random() < args.val_frac else "train"
        base_name = f"f{n_frames:04d}"

        variants = [rgb_bgr] + domain_randomise(rng, rgb_bgr, args.variants)
        for vi, img in enumerate(variants):
            save_sample(out_dir, split, f"{base_name}_v{vi}", img, boxes)
            n_imgs += 1
        if args.debug_overlay:
            dbg = rgb_bgr.copy()
            for class_id, x0, y0, x1, y1 in boxes:
                cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(dbg, PRODUCT_NAMES[class_id], (x0, max(12, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imwrite(str(out_dir / "debug" / f"{base_name}.jpg"), dbg)

        n_frames += 1
        n_boxes += len(boxes)
        for class_id, *_ in boxes:
            class_counts[PRODUCT_NAMES[class_id]] += 1
        if n_frames % 25 == 0:
            print(f"[gen] {n_frames}/{args.frames} frames, {n_imgs} imgs, "
                  f"{n_boxes} boxes")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(PRODUCT_NAMES)}\n"
        f"names: [{', '.join(PRODUCT_NAMES)}]\n"
    )
    print(f"[gen] done: {n_frames} frames, {n_imgs} images, {n_boxes} boxes")
    print(f"[gen] class boxes: {json.dumps(class_counts, ensure_ascii=True, sort_keys=True)}")
    print(f"[gen] data.yaml -> {data_yaml}")
    if n_frames < args.frames:
        print(f"[gen] WARNING: only {n_frames}/{args.frames} frames had a "
              f"visible selected product (hit attempt cap {max_attempts}).")


if __name__ == "__main__":
    main()
