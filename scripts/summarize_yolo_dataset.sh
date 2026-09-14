#!/usr/bin/env bash
# Summarise the YOLO training dataset: sizes, splits, image geometry, labels.
set -u
cd /workspace/baseline/examples/supermarket_sorting/perception

echo "=== data.yaml ==="
cat dataset/data.yaml 2>/dev/null

echo
echo "=== dataset tree (top 2 levels) ==="
find dataset -maxdepth 2 -type d | sort

echo
echo "=== file counts ==="
for d in $(find dataset -maxdepth 2 -type d | sort); do
  n=$(find "$d" -maxdepth 1 -type f | wc -l)
  [ "$n" -gt 0 ] && echo "$d: $n files"
done

echo
echo "=== image geometry (sample of up to 6) ==="
find dataset -path '*images*' -name '*.jpg' -o -path '*images*' -name '*.png' 2>/dev/null \
  | head -6 | while read -r f; do
      python3 - "$f" <<'PY'
import sys
try:
    from PIL import Image
    im = Image.open(sys.argv[1])
    print(f"  {sys.argv[1]}  {im.size[0]}x{im.size[1]}  mode={im.mode}")
except Exception as exc:      # pragma: no cover - diagnostic only
    print(f"  {sys.argv[1]}  <{exc}>")
PY
    done

echo
echo "=== label class histogram (all splits) ==="
python3 - <<'PY'
import collections
import pathlib

root = pathlib.Path("dataset")
names = ['sanmingzhi', 'heweidao', 'shupian', 'zhijin', 'maidong',
         'kele', 'kouxiangtang', 'pingguo', 'chengzi']
hist = collections.Counter()
boxes = collections.Counter()
files = 0
for lbl in root.rglob("*.txt"):
    if lbl.name in ("data.yaml",):
        continue
    files += 1
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cid = int(float(parts[0]))
        hist[cid] += 1
        boxes[cid] += 1
print(f"  label files: {files}")
for cid in sorted(hist):
    nm = names[cid] if cid < len(names) else f"class{cid}"
    print(f"  {cid} {nm:<14} {hist[cid]}")
print(f"  TOTAL boxes: {sum(hist.values())}")
PY
