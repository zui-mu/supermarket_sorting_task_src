#!/usr/bin/env bash
# Compare the background 3DGS asset the DATASET was rendered with against the one
# the LIVE server renders with.  A mismatch here is a first-order domain gap.
set -u
cd /workspace/baseline/examples/supermarket_sorting

echo "=== 3dgs asset directory ==="
ls -la models/3dgs/shentoon/ 2>/dev/null | head -20

echo
echo "=== what gen_dataset.py would pick ==="
python3 - <<'PY'
import pathlib
assets = pathlib.Path("models")
fit = assets / "3dgs" / "shentoon" / "retail_background_fit.ply"
dummy = assets / "3dgs" / "shentoon" / "dummy_background.ply"
print(f"  retail_background_fit.ply exists: {fit.exists()}"
      + (f"  ({fit.stat().st_size} bytes)" if fit.exists() else ""))
print(f"  dummy_background.ply      exists: {dummy.exists()}"
      + (f"  ({dummy.stat().st_size} bytes)" if dummy.exists() else ""))
print("  gen_dataset.py would use:"
      + ("  retail_background_fit.ply" if fit.exists() else "  dummy_background.ply"))
PY

echo
echo "=== what the live server uses (resolve_background_ply) ==="
grep -n -A 14 "def resolve_background_ply" supermarket_sorting_server.py

echo
echo "=== env overrides ==="
env | grep -i -E 'SUPERMARKET_(USE_GS|BACKGROUND)' || echo "  (none set)"
