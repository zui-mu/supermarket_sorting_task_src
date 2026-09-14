#!/usr/bin/env python3
"""Time scene rendering under the current display backend.

Decides whether a live visual run can afford 3DGS (the renderer the detector is
trained on) or must fall back to the plain MuJoCo rasteriser.

    python3 scripts/probe_render_speed.py [--use-gs 0|1] [--frames 5]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

REPO = pathlib.Path("/workspace/baseline")
TASK = REPO / "examples" / "supermarket_sorting"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TASK))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-gs", type=int, default=1, choices=(0, 1))
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    args = ap.parse_args()

    from discoverse.robots_env.mmk2_base import MMK2Base, MMK2Cfg

    from perception import gen_dataset as G

    cfg = MMK2Cfg()
    cfg.mjcf_file_path = G._write_runtime_xml()
    cfg.use_gaussian_renderer = bool(args.use_gs)
    cfg.enable_render = True
    # headless=False so the window path (glfw) is exercised when DISPLAY is set
    cfg.headless = os.getenv("SUPERMARKET_HEADLESS", "0") == "1"

    layout = json.loads(G.LAYOUT_JSON.read_text())
    cfg.obj_list = [slot["body"] for slot in layout]
    cfg.gs_model_dict = G._local_robot_gs_model_dict()
    if args.use_gs:
        cfg.gs_model_dict["background"] = G._resolve_background_ply()
    for slot in layout:
        cfg.gs_model_dict[slot["body"]] = slot["gs_ply"]

    cfg.obs_rgb_cam_id = [0]
    cfg.obs_depth_cam_id = [0]
    cfg.lidar_s2_sim = False
    cfg.render_set = {"fps": 60, "width": args.w, "height": args.h}

    print(f"[speed] MUJOCO_GL={os.getenv('MUJOCO_GL')} DISPLAY={os.getenv('DISPLAY')!r} "
          f"use_gs={args.use_gs} headless={cfg.headless} {args.w}x{args.h}")
    t0 = time.time()
    sim = MMK2Base(cfg)
    sim.reset()
    print(f"[speed] scene build + reset: {time.time() - t0:.1f} s")

    # a representative shelf-view pose
    G.set_robot_pose(sim, (-1.90, 2.45), math.pi / 2.0, 0.03, -0.18)

    times = []
    for i in range(args.frames):
        t = time.time()
        sim.render()
        _ = sim.img_rgb_obs_s[0]
        dt = time.time() - t
        times.append(dt)
        print(f"[speed] frame {i + 1}: {dt:.3f} s")

    mean = sum(times) / len(times)
    print(f"[speed] MEAN {mean:.3f} s/frame -> {1.0 / mean:.2f} fps")
    print(f"[speed] a 600 s (sim) official run needs ~{600 * mean / 60:.0f} min "
          f"of wall clock at this rate (1 sim step == 1 frame here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
