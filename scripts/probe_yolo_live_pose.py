#!/usr/bin/env python3
"""A/B probe: is the YOLO domain gap the RENDERER, the POSE, or both?

Two independent mismatches are suspected between the training images and the
live stream.  This script renders the same slot under each combination, saves
the frames as PNG, and runs the shipped checkpoint on them.

Mismatch 1 - renderer (perception/gen_dataset.py vs scripts/run_v2_official_test.sh)
    dataset : cfg.use_gaussian_renderer = True   (hard-coded) + retail 3DGS
              background PLY
    live    : SUPERMARKET_USE_GS defaults to 0 in the official runner, i.e. the
              plain MuJoCo rasteriser and no background model at all

Mismatch 2 - camera pose (dataset sample_pose vs the client's search policy)
    dataset : head_pitch in [-0.75, -0.35], slide in [0.00, 0.25],
              base_y in [2.35, 2.95]
    live    : head_pitch = {L1: -0.38, L2: -0.18, L3: -0.06},
              slide = 0.03, base_y = 2.34

Run inside the SERVER image (tissv_client has no working EGL):

    bash scripts/run_probe_yolo_pose.sh
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = pathlib.Path("/workspace/baseline")
TASK = REPO / "examples" / "supermarket_sorting"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TASK))

import numpy as np  # noqa: E402

from perception import gen_dataset as G  # noqa: E402

CKPT = TASK / "perception" / "checkpoints" / "supermarket_multiclass.pt"
DATASET = TASK / "perception" / "dataset"
RUNTIME_LAYOUT = TASK / "runtime_layout.json"
# Under the mounted repo so the frames survive the --rm container and can be
# inspected by eye.  A black or otherwise degenerate render is the first thing
# to rule out when a detector answers the same class for every input.
OUT_DIR = REPO / "logs_probe_frames"
HEAD_CAM = 0
CONF_SHOW = 0.12

SEARCH_OBSERVE_PITCH_BY_LEVEL = {"L1": -0.38, "L2": -0.18, "L3": -0.06}
SEARCH_LOCK_SLIDE = 0.03
SHELF_CROSS_Y = 2.34
BASE_X_OFFSET = 0.045
YAW_FACING_SHELF = math.pi / 2.0

TRAIN_PITCH = -0.55
TRAIN_SLIDE = 0.12
TRAIN_BASE_Y = 2.65

REFERENCE_SLOTS = ["A_L1_C1", "A_L1_C2", "A_L2_C1", "A_L2_C2", "A_L3_C1"]


def load_slots():
    raw = json.loads(RUNTIME_LAYOUT.read_text())
    items = raw if isinstance(raw, list) else raw.get("items", [])
    return {f"{it['shelf']}_{it['level']}_{it['column']}": it for it in items}


def runtime_pos_overrides():
    """Rebuild the server's pos_overrides from an exported runtime layout.

    CRITICAL: the first version of this probe rendered the NOMINAL scene
    (gen_dataset.build_sim loads retail_competition_layout.json) and then scored
    it against the RANDOMISED runtime layout.  The two disagree on which kind
    sits in which slot, so every one of the 20 combinations was scored against
    the wrong answer and the whole run was worthless.  Rendering the runtime
    scene makes the truth and the pixels agree.
    """
    raw = json.loads(RUNTIME_LAYOUT.read_text())
    items = raw if isinstance(raw, list) else raw.get("items", [])
    return {it["body"]: tuple(float(v) for v in it["world_position"]) for it in items}


def build_sim(use_gs: bool, runtime_scene: bool = True):
    """Same construction as gen_dataset.build_sim, with renderer and scene selectable."""
    from discoverse.robots_env.mmk2_base import MMK2Base, MMK2Cfg

    cfg = MMK2Cfg()
    if runtime_scene:
        from supermarket_sorting_server import write_runtime_xml
        cfg.mjcf_file_path = write_runtime_xml(runtime_pos_overrides(), None)
    else:
        cfg.mjcf_file_path = G._write_runtime_xml()
    cfg.use_gaussian_renderer = bool(use_gs)
    cfg.enable_render = True
    cfg.headless = True

    layout = json.loads(G.LAYOUT_JSON.read_text())
    cfg.obj_list = [slot["body"] for slot in layout]
    cfg.gs_model_dict = G._local_robot_gs_model_dict()
    if use_gs:
        cfg.gs_model_dict["background"] = G._resolve_background_ply()
    for slot in layout:
        cfg.gs_model_dict[slot["body"]] = slot["gs_ply"]

    cfg.obs_rgb_cam_id = [HEAD_CAM]
    cfg.obs_depth_cam_id = [HEAD_CAM]
    cfg.lidar_s2_sim = False
    cfg.render_set = {"fps": 24, "width": G.IMG_W, "height": G.IMG_H}

    sim = MMK2Base(cfg)
    sim.reset()
    return sim


def main() -> int:
    import cv2

    from perception.backends import YoloBackend

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Use the PRODUCTION loader, not a re-implementation: it applies the
    # torch>=2.6 weights_only compatibility patch and the same device policy,
    # so a difference we measure here is a difference the run would see.
    backend = YoloBackend(str(CKPT))
    names = backend.class_names
    K = G.head_cam_K()
    slots = load_slots()
    print(f"[probe] model={CKPT.name} classes={len(names)} "
          f"layout_slots={len(slots)} out={OUT_DIR}")

    def predict(rgb, depth):
        # PRODUCTION PASSES BGR: perception/kele_detect.py does
        #     rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        #     dets = self.detector.detect(rgb, depth, self.K, T_cam_world)
        # so the array handed to ultralytics is BGR.  The first version of this
        # probe fed the sim's native RGB straight in, i.e. with the red and blue
        # channels swapped, and then blamed the model for the result.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dets = backend.detect(bgr, depth, K, None)
        out = [(str(d["class"]), float(d["conf"])) for d in dets]
        out.sort(key=lambda t: -t[1])
        return out, dets

    # ---- CONTROL: the same model on its own validation images -------------
    # Scored with IoU-matched boxes, NOT "is any predicted class also present
    # anywhere in the frame".  The dataset labels 5-8 of the 9 classes per
    # frame, so the loose test is satisfied by chance and would have declared a
    # broken model healthy.
    def iou(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return inter / ua if ua > 0 else 0.0

    names_by_id = {int(k): str(v) for k, v in
                   (names.items() if isinstance(names, dict)
                    else enumerate(names))}
    val_imgs = sorted((DATASET / "images" / "val").glob("*.jpg"))
    limit = int(os.getenv("PROBE_VAL_IMAGES", "60"))
    val_imgs = val_imgs[:limit]
    print(f"\n[control] model on its OWN validation split ({len(val_imgs)} images), "
          f"IoU>=0.5 matched")
    tp = fp = fn = 0
    for img_path in val_imgs:
        lbl = DATASET / "labels" / "val" / f"{img_path.stem}.txt"
        bgr = cv2.imread(str(img_path))            # BGR, exactly like training
        h, w = bgr.shape[:2]
        gts = []
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            gts.append((names_by_id.get(cid, "?"),
                        ((cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h)))
        dets = backend.detect(bgr, np.zeros((h, w), dtype=float), K, None)
        preds = [((d["x"] - d["w"] / 2, d["y"] - d["h"] / 2,
                   d["x"] + d["w"] / 2, d["y"] + d["h"] / 2),
                  str(d["class"])) for d in dets]
        used = set()
        for gname, gbox in gts:
            best_i, best_iou = -1, 0.0
            for i, (pbox, _) in enumerate(preds):
                if i in used:
                    continue
                v = iou(gbox, pbox)
                if v > best_iou:
                    best_i, best_iou = i, v
            if best_i >= 0 and best_iou >= 0.5:
                used.add(best_i)
                if preds[best_i][1] == gname:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            else:
                fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    print(f"[control] val boxes: tp={tp} fp={fp} fn={fn}  "
          f"precision={prec:.3f} recall={rec:.3f}")
    print("[control] a healthy checkpoint should be well above 0.5 here; "
          "near 0 means the checkpoint cannot do the task at all")

    poses = {
        "live": (SEARCH_LOCK_SLIDE, SHELF_CROSS_Y, "level"),
        "train": (TRAIN_SLIDE, TRAIN_BASE_Y, "fixed"),
    }

    summary = []
    for use_gs in (True, False):
        tag = "gs1" if use_gs else "gs0"
        print("\n" + "#" * 74)
        print(f"# renderer: {'3DGS' if use_gs else 'MuJoCo rasteriser'}"
              f"  (SUPERMARKET_USE_GS={'1' if use_gs else '0'})"
              f"  scene=RUNTIME (matches the failed run)")
        print("#" * 74)
        sim = build_sim(use_gs, runtime_scene=True)
        for key in REFERENCE_SLOTS:
            slot = slots.get(key)
            if slot is None:
                continue
            level = slot["level"]
            true_kind = slot["object_kind"]
            sx = float(slot["world_position"][0])
            base_x = sx + BASE_X_OFFSET
            for pose_name, (slide, base_y, pitch_mode) in poses.items():
                if pitch_mode == "level":
                    pitch = SEARCH_OBSERVE_PITCH_BY_LEVEL.get(level, -0.18)
                else:
                    pitch = TRAIN_PITCH
                G.set_robot_pose(sim, (base_x, base_y), YAW_FACING_SHELF, slide, pitch)
                sim.render()
                rgb = sim.img_rgb_obs_s[HEAD_CAM]
                depth = sim.img_depth_obs_s[HEAD_CAM]
                png = OUT_DIR / f"{tag}_{pose_name}_{key}.png"
                cv2.imwrite(str(png), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                preds, _ = predict(rgb, depth)
                hit = any(nm == true_kind for nm, _ in preds)
                summary.append((tag, pose_name, key, true_kind, hit, preds))
                # A degenerate render (all black / all one colour) produces a
                # constant prediction; print enough to spot that immediately.
                print(f"  {tag} {pose_name:<5} {key:<8} true={true_kind:<12} "
                      f"dets={len(preds):<3} {'FOUND' if hit else 'MISSED'}  "
                      f"rgb_mean={rgb.mean():6.1f} max={int(rgb.max()):3d}  "
                      f"top={[f'{n}:{c:.2f}' for n, c in preds[:4]]}")

    print("\n" + "=" * 74)
    print("SUMMARY  (renderer, pose, slot, true kind, found?)")
    for tag, pose_name, key, true_kind, hit, _ in summary:
        print(f"  {tag}  {pose_name:<5} {key:<8} true={true_kind:<12} "
              f"{'FOUND' if hit else 'MISSED'}")
    ok = sum(1 for *_, hit, _ in summary if hit)
    print(f"\n  {ok}/{len(summary)} combinations detected the true kind "
          f"at conf >= {CONF_SHOW}")
    print(f"  frames saved under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
