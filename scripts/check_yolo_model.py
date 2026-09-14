#!/usr/bin/env python3
"""Inspect the YOLO checkpoint: class list + a real-scene detection histogram."""
import os

CKPT = os.getenv(
    "SUPERMARKET_YOLO_WEIGHTS",
    "/workspace/baseline/examples/supermarket_sorting/perception/checkpoints/supermarket_multiclass.pt",
)
print("ckpt:", CKPT, "exists:", os.path.exists(CKPT))
try:
    import hashlib
    with open(CKPT, "rb") as fh:
        print("sha256:", hashlib.sha256(fh.read()).hexdigest())
    print("size:", os.path.getsize(CKPT))
except Exception as exc:
    print("hash failed:", exc)

try:
    from ultralytics import YOLO
    model = YOLO(CKPT)
    names = model.names
    print("nc:", len(names))
    print("names:", names)
except Exception as exc:
    print("model load failed:", repr(exc))

# What does the running perception node subscribe/publish?
print("--- env ---")
for key in (
    "SUPERMARKET_YOLO_WEIGHTS",
    "SUPERMARKET_YOLO_REQUIRE_OFFICIAL_CLASSES",
    "SUPERMARKET_YOLO_CONF",
    "SUPERMARKET_DETECT_PRODUCTS",
    "SUPERMARKET_DETECT_BACKEND",
):
    print(f"{key}={os.getenv(key)}")
