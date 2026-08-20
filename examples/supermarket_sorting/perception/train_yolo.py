#!/usr/bin/env python3
"""Train the official nine-class detector on offline simulator data.

Runs INSIDE the Docker container (needs ultralytics + a GPU).  Expects the
dataset produced by gen_dataset.py at perception/dataset/.  After training it
copies the best weights to perception/checkpoints/supermarket_multiclass.pt
and prints validation metrics.  This script is intentionally conservative:
it validates the dataset, uses no dataloader workers, disables dataset caching,
and refuses oversized datasets or a nearly-full GPU.

Run (inside container):
    cd examples/supermarket_sorting
    python3 perception/train_yolo.py --epochs 40 --batch 2
"""
import os
import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import yaml

PERCEPTION_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PERCEPTION_DIR / "dataset" / "data.yaml"
CKPT_DIR = PERCEPTION_DIR / "checkpoints"
FINAL_CKPT = CKPT_DIR / "supermarket_multiclass.pt"
DEFAULT_INIT_WEIGHTS = CKPT_DIR / "kele.pt"
OFFICIAL_CLASSES = (
    "sanmingzhi", "heweidao", "shupian", "zhijin", "maidong",
    "kele", "kouxiangtang", "pingguo", "chengzi",
)
MAX_DATASET_BYTES = 8 * 1024**3
MIN_FREE_GPU_BYTES = 2 * 1024**3
MIN_TRAIN_BOXES_PER_CLASS = 20
MIN_VAL_BOXES_PER_CLASS = 5


def validate_dataset(data_path: Path) -> dict:
    if not data_path.is_file():
        raise FileNotFoundError(f"dataset yaml not found: {data_path}")
    config = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda value: int(value))]
    if list(names or []) != list(OFFICIAL_CLASSES) or int(config.get("nc", 0)) != len(OFFICIAL_CLASSES):
        raise ValueError(
            "dataset must contain the nine official classes in this order: "
            + ", ".join(OFFICIAL_CLASSES)
        )
    root = Path(config.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    total_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if total_bytes > MAX_DATASET_BYTES:
        raise ValueError(f"dataset is {total_bytes / 1024**3:.2f} GiB; limit is 8 GiB")
    train_images = root / str(config["train"])
    val_images = root / str(config["val"])
    train_files = list(train_images.glob("*.jpg")) if train_images.is_dir() else []
    val_files = list(val_images.glob("*.jpg")) if val_images.is_dir() else []
    train_count, val_count = len(train_files), len(val_files)
    if train_count < 9 or val_count < 9:
        raise ValueError(
            f"dataset is too small for nine-class training: train={train_count}, val={val_count}"
        )
    def label_distribution(image_files):
        counts = Counter()
        for image_path in image_files:
            label_path = root / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            for line in label_path.read_text(encoding="ascii").splitlines():
                fields = line.split()
                if fields:
                    counts[int(fields[0])] += 1
        return counts

    train_labels = label_distribution(train_files)
    val_labels = label_distribution(val_files)
    required_ids = set(range(len(OFFICIAL_CLASSES)))
    weak_train = [class_id for class_id in sorted(required_ids)
                  if train_labels[class_id] < MIN_TRAIN_BOXES_PER_CLASS]
    weak_val = [class_id for class_id in sorted(required_ids)
                if val_labels[class_id] < MIN_VAL_BOXES_PER_CLASS]
    if weak_train or weak_val:
        raise ValueError(
            "every official class needs enough examples in both splits; "
            f"weak train={weak_train}, val={weak_val}; "
            "generate more wide-view frames"
        )
    print(
        f"[train] dataset validated: {total_bytes / 1024**2:.1f} MiB, "
        f"train={train_count}, val={val_count}, classes={len(names)}, "
        f"boxes(train/val)={sum(train_labels.values())}/{sum(val_labels.values())}"
    )
    return config


def read_final_metrics(results_csv: Path) -> dict[str, float]:
    """Read the final validation metrics already produced during training."""
    if not results_csv.is_file():
        return {}
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    if not rows:
        return {}
    row = {key.strip(): value for key, value in rows[-1].items() if key and value}
    metrics = {}
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if key in row:
            metrics[key] = float(row[key])
    return metrics


def main():
    ap = argparse.ArgumentParser(description="train the official nine-class detector")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument(
        "--weights",
        default=str(DEFAULT_INIT_WEIGHTS) if DEFAULT_INIT_WEIGHTS.is_file() else "yolov8n.pt",
        help="pretrained weights to fine-tune from (local kele.pt when available)",
    )
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=2,
                    help="fixed conservative batch size")
    ap.add_argument("--patience", type=int, default=25,
                    help="early-stop patience (epochs)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--amp", default="false",
                    choices=["true", "false", "1", "0", "yes", "no", "on", "off"],
                    help="enable AMP training checks; false avoids downloading yolov8n.pt")
    ap.add_argument("--project", default=str(PERCEPTION_DIR / "runs"))
    ap.add_argument("--name", default="supermarket_multiclass_yolov8n")
    args = ap.parse_args()

    if not 1 <= args.epochs <= 200:
        ap.error("--epochs must be between 1 and 200")
    if not 1 <= args.batch <= 8:
        ap.error("--batch must be between 1 and 8 on the 8 GiB GPU profile")
    config = validate_dataset(Path(args.data).resolve())

    import torch
    from ultralytics import YOLO

    if str(args.device).lower() not in {"cpu", "-1"} and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"[train] GPU memory free={free_bytes / 1024**3:.2f} GiB / "
              f"total={total_bytes / 1024**3:.2f} GiB")
        if free_bytes < MIN_FREE_GPU_BYTES:
            raise RuntimeError("less than 2 GiB GPU memory is free; close other GPU jobs first")
    elif str(args.device).lower() not in {"cpu", "-1"}:
        raise RuntimeError("CUDA requested but no CUDA device is available")

    # ultralytics 8.0.196 + torch>=2.6 needs weights_only=False for old ckpts
    _orig = torch.load
    def _compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)
    torch.load = _compat

    try:
        model = YOLO(args.weights)
        amp = str(args.amp).lower() in {"true", "1", "yes", "on"}
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            device=args.device,
            amp=amp,
            workers=0,
            cache=False,
            plots=False,
            max_det=100,
            project=args.project,
            name=args.name,
            exist_ok=True,
            verbose=True,
        )
    finally:
        torch.load = _orig

    # locate best.pt from the run
    best = Path(args.project) / args.name / "weights" / "best.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if best.is_file():
        shutil.copy2(best, FINAL_CKPT)
        print(f"[train] copied {best} -> {FINAL_CKPT}")
    else:
        print(f"[train] WARNING: best.pt not found at {best}")

    metrics = read_final_metrics(Path(args.project) / args.name / "results.csv")
    if metrics:
        print(f"[train] val mAP50-95 = {metrics.get('metrics/mAP50-95(B)', 0.0):.4f}")
        print(f"[train] val mAP50    = {metrics.get('metrics/mAP50(B)', 0.0):.4f}")
    else:
        print("[train] WARNING: results.csv was not available after training")


if __name__ == "__main__":
    main()
