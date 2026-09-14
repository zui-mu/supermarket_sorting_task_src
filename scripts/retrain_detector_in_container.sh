#!/usr/bin/env bash
# Regenerate the detector dataset over BOTH renderer domains, then retrain.
#
# Why (2026-08-23 audit)
# ----------------------
# The shipped checkpoint was trained only on 3DGS frames
# (gen_dataset.build_sim hard-coded use_gaussian_renderer = True), while the
# formal runner rendered with the plain MuJoCo rasteriser
# (SUPERMARKET_USE_GS=0).  scripts/probe_yolo_live_pose.py measured the cost:
# 8/10 slots correct with 3DGS on, 0/10 with it off.
#
# Flipping the runner flag fixes today's run, but it leaves the detector
# dependent on a flag.  This script removes that dependency by rendering both
# domains into one dataset with `--use-gs both`.
#
# It writes to a NEW output directory on purpose: `--overwrite` on the existing
# perception/dataset would destroy the only data that reproduces the current
# checkpoint, and a failed retrain would then leave nothing to fall back to.
#
# Run inside the SERVER image (needs gsplat + EGL + a GPU):
#     bash /workspace/baseline/scripts/retrain_detector_in_container.sh
set -uo pipefail

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash

cd /workspace/baseline
export PYTHONPATH="/workspace/baseline:/workspace/baseline/examples/supermarket_sorting:/workspace/baseline/examples/ros2:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages"

FRAMES="${RETRAIN_FRAMES:-1400}"     # poses; split across the two renderers
VARIANTS="${RETRAIN_VARIANTS:-1}"    # extra domain-randomised copies per pose
EPOCHS="${RETRAIN_EPOCHS:-60}"
BATCH="${RETRAIN_BATCH:-4}"
DATASET_DIR="${RETRAIN_DATASET:-/workspace/baseline/examples/supermarket_sorting/perception/dataset_v2}"

echo "=============================================================="
echo " step 1/2  generate dataset (both renderers)"
echo "   frames=${FRAMES} variants=${VARIANTS} out=${DATASET_DIR}"
echo "=============================================================="
rm -rf "${DATASET_DIR}"
python3 examples/supermarket_sorting/perception/gen_dataset.py \
  --frames "${FRAMES}" \
  --variants "${VARIANTS}" \
  --pose-mode wide \
  --use-gs both \
  --out "${DATASET_DIR}" \
  --overwrite
gen_status=$?
if [ "${gen_status}" -ne 0 ]; then
  echo "[retrain] dataset generation failed (exit ${gen_status})" >&2
  exit "${gen_status}"
fi

echo
echo "=== generated split ==="
for d in "${DATASET_DIR}"/images/*; do
  [ -d "$d" ] && echo "  $d: $(ls "$d" | wc -l) images"
done
echo "  renderer tags:"
ls "${DATASET_DIR}"/images/train | sed 's/.*_\(gs[01]\)_.*/\1/' | sort | uniq -c

echo
echo "=============================================================="
echo " step 2/2  train (epochs=${EPOCHS} batch=${BATCH})"
echo "=============================================================="
python3 examples/supermarket_sorting/perception/train_yolo.py \
  --data "${DATASET_DIR}/data.yaml" \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --name supermarket_multiclass_gsboth
train_status=$?

echo
echo "[retrain] gen=${gen_status} train=${train_status}"
echo "[retrain] checkpoint: examples/supermarket_sorting/perception/checkpoints/supermarket_multiclass.pt"
exit "${train_status}"
