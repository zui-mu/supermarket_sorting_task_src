#!/usr/bin/env python3
"""Compare training-image geometry with the live camera stream geometry."""
import glob
import os

from PIL import Image

VAL = "/workspace/baseline/examples/supermarket_sorting/perception/dataset/images/val"
TRAIN = "/workspace/baseline/examples/supermarket_sorting/perception/dataset/images/train"

for name, d in (("train", TRAIN), ("val", VAL)):
    files = sorted(glob.glob(os.path.join(d, "*")))
    print(f"{name}: {len(files)} images")
    sizes = {}
    for f in files[:60]:
        try:
            sizes[Image.open(f).size] = sizes.get(Image.open(f).size, 0) + 1
        except Exception as exc:
            print("  open failed", f, exc)
    print("  sizes:", sizes)

# bbox pixel sizes in a val label file (yolo format: cls cx cy w h normalized)
labels = sorted(glob.glob(os.path.join(os.path.dirname(VAL), "..", "labels", "val", "*.txt")))
print("label files:", len(labels))
import statistics  # noqa: E402

heights = []
for f in labels:
    with open(f) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 5:
                heights.append(float(parts[4]))
if heights:
    print("normalized bbox heights: mean=%.4f min=%.4f max=%.4f" % (
        statistics.mean(heights), min(heights), max(heights)))
    for img_h in (960, 720, 480, 360):
        print(f"  if image height={img_h}: mean box = {statistics.mean(heights)*img_h:.1f} px, "
              f"min = {min(heights)*img_h:.1f} px")
