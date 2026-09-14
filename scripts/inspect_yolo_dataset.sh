#!/bin/bash
# Inspect the YOLO training dataset: per-class label counts.
cd /workspace/baseline/examples/supermarket_sorting/perception || exit 1

echo "=== perception dir ==="
ls -la | head -20

echo
echo "=== dataset layout ==="
find dataset -maxdepth 2 -type d 2>/dev/null | head -20

echo
echo "=== label file counts ==="
for d in dataset/train/labels dataset/val/labels dataset/labels dataset/train/images dataset/val/images; do
  if [ -d "$d" ]; then
    echo "$d: $(ls "$d" 2>/dev/null | wc -l) files"
  fi
done

echo
echo "=== per-class instance counts (all label txt under dataset/) ==="
find dataset -name '*.txt' -path '*label*' -exec cat {} + 2>/dev/null \
  | awk '{print $1}' | sort -n | uniq -c | sort -rn

echo
echo "=== data.yaml ==="
cat dataset/data.yaml 2>/dev/null

echo
echo "=== training runs ==="
ls -la runs 2>/dev/null | head
ls -la checkpoints 2>/dev/null

echo
echo "=== any other datasets on disk ==="
find /workspace/baseline -maxdepth 4 -name 'data.yaml' 2>/dev/null | head
