#!/usr/bin/env python3
"""Capture the live head-camera render (with YOLO overlay when available) to
video + PNG frames, so the human can inspect the simulation visually.

Usage (inside the client container, ROS_DOMAIN_ID=107):
    python3 scripts/capture_sim_video.py --seconds 40 --fps 2 --out /workspace/baseline/capture/sim

Subscribes:
  /head_camera/color/image_raw          raw render (bgr8)
  /supermarket_sorting/result_image     YOLO debug overlay (bgr8, when published)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

RAW_TOPIC = "/head_camera/color/image_raw"
OVERLAY_TOPIC = "/supermarket_sorting/result_image"


def decode_image(msg: Image) -> np.ndarray | None:
    if msg is None or msg.data is None:
        return None
    encoding = str(msg.encoding)
    try:
        if encoding in ("bgr8", "rgb8", "8UC3"):
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            arr = arr.reshape((msg.height, msg.width, 3))
            if encoding == "rgb8":
                arr = arr[:, :, ::-1]  # -> BGR for OpenCV
            return arr.copy()
        if encoding in ("mono8", "8UC1"):
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            return arr.reshape((msg.height, msg.width)).copy()
    except Exception:
        return None
    return None


class CaptureNode(Node):
    def __init__(self, out_prefix: str, seconds: float, fps: float):
        super().__init__("sim_capture_node")
        self.raw_image = None
        self.overlay_image = None
        self.out_prefix = Path(out_prefix)
        self.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        self.seconds = seconds
        self.fps = fps
        self.raw_sub = self.create_subscription(
            Image, RAW_TOPIC, self.raw_cb, 5
        )
        self.overlay_sub = self.create_subscription(
            Image, OVERLAY_TOPIC, self.overlay_cb, 5
        )
        self.frames: list[np.ndarray] = []
        self.timestamps: list[float] = []

    def raw_cb(self, msg: Image) -> None:
        dec = decode_image(msg)
        if dec is not None:
            self.raw_image = dec

    def overlay_cb(self, msg: Image) -> None:
        dec = decode_image(msg)
        if dec is not None:
            self.overlay_image = dec

    def capture_loop(self) -> int:
        self.get_logger().info(
            f"capturing {self.seconds}s at {self.fps} fps -> {self.out_prefix}"
        )
        start = time.monotonic()
        next_sample = start
        while time.monotonic() - start < self.seconds:
            now = time.monotonic()
            if now >= next_sample:
                next_sample = now + 1.0 / self.fps
                self.sample(now - start)
            self._spin_once()
            time.sleep(0.02)
        # flush remaining callbacks
        for _ in range(20):
            self._spin_once()
            time.sleep(0.02)
        return self.write_outputs()

    def _spin_once(self) -> None:
        rclpy.spin_once(self, timeout_sec=0.02)

    def sample(self, elapsed: float) -> None:
        raw = self.raw_image
        if raw is None:
            self.get_logger().warn(f"no raw frame yet at t={elapsed:.1f}s")
            return
        frame = raw.copy()
        ov = self.overlay_image
        if ov is not None and ov.shape == raw.shape:
            frame = np.hstack((raw, ov))
        self.frames.append(frame)
        self.timestamps.append(elapsed)

    def write_outputs(self) -> int:
        n = len(self.frames)
        if n == 0:
            self.get_logger().error("no frames captured")
            return 1
        h, w = self.frames[0].shape[:2]
        self.get_logger().info(f"saved {n} frames, each {w}x{h}")
        # PNG keyframes (first, middle, last)
        for idx, label in ((0, "first"), (n // 2, "mid"), (n - 1, "last")):
            png_path = f"{self.out_prefix}_{label}.png"
            import cv2
            cv2.imwrite(png_path, self.frames[idx])
            self.get_logger().info(f"wrote {png_path}")
        # Video via OpenCV (mp4v or MJPG fallback)
        import cv2
        for codec in ("mp4v", "MJPG"):
            try:
                ext = "mp4" if codec == "mp4v" else "avi"
                vpath = f"{self.out_prefix}.{ext}"
                writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*codec), float(self.fps), (w, h))
                if not writer.isOpened():
                    writer.release()
                    continue
                for frame in self.frames:
                    writer.write(frame)
                writer.release()
                self.get_logger().info(f"wrote video {vpath} ({codec})")
                return 0
            except Exception as exc:
                self.get_logger().warn(f"video write with {codec} failed: {exc}")
        self.get_logger().error("no video codec available; only PNGs written")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--out", default="/workspace/baseline/capture/sim")
    args = parser.parse_args()
    rclpy.init()
    node = CaptureNode(args.out, args.seconds, args.fps)
    try:
        return node.capture_loop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
