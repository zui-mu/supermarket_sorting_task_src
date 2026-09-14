#!/usr/bin/env python3
"""Run the trained checkpoint over its own validation images and report which
classes it actually finds.  Distinguishes "broken model" (fails even on its own
data) from "domain gap" (works on dataset images, fails on live camera frames).
"""
import os
import glob
from collections import Counter

os.environ.setdefault("YOLO_VERBOSE", "False")
import torch  # noqa: E402

CKPT = "/workspace/baseline/examples/supermarket_sorting/perception/checkpoints/supermarket_multiclass.pt"
VAL_DIR = "/workspace/baseline/examples/supermarket_sorting/perception/dataset/images/val"

from ultralytics import YOLO  # noqa: E402

# Same compatibility patch the perception node uses for pre-2.6 checkpoints.
_orig_load = torch.load


def _compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)


torch.load = _compat
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
names = ckpt["model"].names
print("ckpt classes:", list(names.values()))

model = YOLO(CKPT)
torch.load = _orig_load
images = sorted(glob.glob(os.path.join(VAL_DIR, "*")))[:40]
print(f"validation images sampled: {len(images)} (of {len(glob.glob(os.path.join(VAL_DIR, '*')))} total)")

for conf in (0.30, 0.10):
    found = Counter()
    with torch.no_grad():
        results = model.predict(images, conf=conf, verbose=False, imgsz=640)
    for res in results:
        for box in res.boxes:
            cls_id = int(box.cls.item())
            found[names.get(cls_id, str(cls_id))] += 1
    print(f"--- conf>={conf}: {sum(found.values())} boxes ---")
    for name, count in found.most_common():
        print(f"   {name}: {count}")
    missing = [n for n in names.values() if n not in found]
    print("   NOT detected:", missing)
