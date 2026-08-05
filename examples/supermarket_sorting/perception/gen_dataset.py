#!/usr/bin/env python3
"""Generate a YOLO detection dataset for the kele bottle using GS rendering.

Runs INSIDE the Docker container (needs gsplat + discoverse + a GPU).  Builds
the retail_competition scene with the 3D Gaussian Splatting renderer on (same
config path as supermarket_sorting_server.build_config), places the MMK2 camera
near the baseline shelf-D approach pose, renders the head camera, and auto-labels
every visible kele bottle.

Auto-labelling (no manual boxes)
--------------------------------
For each kele slot we project a small 3-D box around its GT world_position via
the live `headeye` site (camera-optical -> world, the exact transform the
perception node uses).  A rendered-depth gate at the projected centre rejects
occluded/off-screen bottles.  This works with the fitted retail background 3DGS,
where foreground segmentation is not reliable because the shelf/background is
also visible.
Only kele bodies are ever labelled (single class 0 = 'kele').

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
    dataset/data.yaml                   # names: [kele]

Run (headless, inside container):
    cd examples/supermarket_sorting
    MUJOCO_GL=egl python3 perception/gen_dataset.py --frames 300 --variants 1
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
KELE_BOX_HALF = np.array([0.032, 0.032, 0.075], dtype=float)


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
    wide: cover a larger shelf row for more generic detector training.
    """
    if mode == "wide":
        base_x = rng.uniform(-2.3, 1.4)
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


def label_kele(rgb, depth_m, K, T_cw, kele_slots):
    """Return (boxes, dropped). boxes = list of (x0,y0,x1,y1) for visible kele.

    For each kele slot: project its world pos to a pixel, gate on the rendered
    depth near the projected depth, then take the connected component holding the
    projection pixel as the tight bbox.  This guarantees only kele get boxed.
    """
    fg = foreground_mask(rgb)
    boxes = []
    for slot in kele_slots:
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
        boxes.append((bx, by, bx + bw, by + bh))
    return boxes


def _patch_depth(depth_m, u, v, r=4):
    h, w = depth_m.shape[:2]
    y0, y1 = max(0, v - r), min(h, v + r + 1)
    x0, x1 = max(0, u - r), min(w, u + r + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    return float(np.median(valid)) if len(valid) else 0.0


def _projected_bbox(pw, K, T_cw, half=KELE_BOX_HALF, margin=1.08):
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(np.array(pw) + np.array([sx, sy, sz]) * half)
    pts = [project_world_to_px(c, K, T_cw) for c in corners]
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


def label_kele_projected(depth_m, K, T_cw, kele_slots):
    boxes = []
    for slot in kele_slots:
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
        box = _projected_bbox(pw, K, T_cw)
        if box is not None:
            boxes.append(box)
    return boxes


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
def boxes_to_yolo(boxes):
    lines = []
    for x0, y0, x1, y1 in boxes:
        cx = (x0 + x1) / 2.0 / IMG_W
        cy = (y0 + y1) / 2.0 / IMG_H
        bw = (x1 - x0) / IMG_W
        bh = (y1 - y0) / IMG_H
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def save_sample(out_dir, split, name, img_bgr, boxes):
    img_path = out_dir / "images" / split / f"{name}.jpg"
    lab_path = out_dir / "labels" / split / f"{name}.txt"
    cv2.imwrite(str(img_path), img_bgr)
    lab_path.write_text(boxes_to_yolo(boxes))


def main():
    ap = argparse.ArgumentParser(description="generate kele YOLO dataset (GS)")
    ap.add_argument("--frames", type=int, default=200, help="base poses to render")
    ap.add_argument("--variants", type=int, default=3, help="DR variants per frame")
    ap.add_argument("--pose-mode", default="baseline", choices=["baseline", "wide"],
                    help="baseline focuses on the shelf-D stop used by the client")
    ap.add_argument("--label-mode", default="projected", choices=["projected", "mask"],
                    help="projected works with the fitted retail background; mask is for black background")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default=str(TASK_DIR / "perception" / "dataset"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug-overlay", action="store_true",
                    help="also dump boxed previews to <out>/debug")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    if args.debug_overlay:
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)

    sim = build_sim()
    K = head_cam_K()
    layout = json.loads(LAYOUT_JSON.read_text())
    kele_slots = [s for s in layout if s.get("object_kind") == "kele"]
    background_ply = _resolve_background_ply()
    print(f"[gen] {len(kele_slots)} kele slots; rendering {args.frames} frames "
          f"x (1 + {args.variants}) variants, pose={args.pose_mode}, "
          f"labels={args.label_mode}, background={background_ply}")

    n_frames = 0
    n_boxes = 0
    n_imgs = 0
    attempts = 0
    max_attempts = args.frames * 4
    while n_frames < args.frames and attempts < max_attempts:
        attempts += 1
        base_xy, yaw, slide, pitch = sample_pose(rng, args.pose_mode)
        set_robot_pose(sim, base_xy, yaw, slide, pitch)
        sim.render()
        rgb = sim.img_rgb_obs_s[HEAD_CAM_ID]              # uint8 HxWx3, RGB
        depth_m = sim.img_depth_obs_s[HEAD_CAM_ID]        # float, metres
        T_cw = T_cam_world(sim)
        if args.label_mode == "mask":
            boxes = label_kele(rgb, depth_m, K, T_cw, kele_slots)
        else:
            boxes = label_kele_projected(depth_m, K, T_cw, kele_slots)
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
            for x0, y0, x1, y1 in boxes:
                cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.imwrite(str(out_dir / "debug" / f"{base_name}.jpg"), dbg)

        n_frames += 1
        n_boxes += len(boxes)
        if n_frames % 25 == 0:
            print(f"[gen] {n_frames}/{args.frames} frames, {n_imgs} imgs, "
                  f"{n_boxes} boxes")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: [kele]\n"
    )
    print(f"[gen] done: {n_frames} frames, {n_imgs} images, {n_boxes} boxes")
    print(f"[gen] data.yaml -> {data_yaml}")
    if n_frames < args.frames:
        print(f"[gen] WARNING: only {n_frames}/{args.frames} frames had a "
              f"visible kele (hit attempt cap {max_attempts}).")


if __name__ == "__main__":
    main()
