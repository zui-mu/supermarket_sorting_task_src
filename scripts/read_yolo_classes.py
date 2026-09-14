#!/usr/bin/env python3
"""Read the YOLO checkpoint's class names directly (torch.load, trusted file)."""
import os

import torch

CKPT = os.getenv(
    "SUPERMARKET_YOLO_WEIGHTS",
    "/workspace/baseline/examples/supermarket_sorting/perception/checkpoints/supermarket_multiclass.pt",
)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = ckpt.get("model") if isinstance(ckpt, dict) else None
names = getattr(model, "names", None)
print("type:", type(ckpt).__name__)
print("keys:", list(ckpt.keys())[:12] if isinstance(ckpt, dict) else "n/a")
print("names:", names)
if isinstance(names, dict):
    print("nc:", len(names))
    print("class list:", list(names.values()))
elif isinstance(names, (list, tuple)):
    print("nc:", len(names))
    print("class list:", list(names))
# training metadata if present
for key in ("train_args", "date", "version", "epoch"):
    if isinstance(ckpt, dict) and key in ckpt:
        print(f"{key}:", str(ckpt[key])[:300])
