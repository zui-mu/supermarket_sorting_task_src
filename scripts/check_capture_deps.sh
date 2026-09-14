#!/bin/bash
python3 - <<'PY'
import importlib
for mod in ("cv2", "cv_bridge", "rclpy"):
    try:
        m = importlib.import_module(mod)
        print(mod, "OK", getattr(m, "__version__", ""))
    except Exception as e:
        print(mod, "MISSING", repr(e)[:120])
PY
which ffmpeg && ffmpeg -version 2>/dev/null | head -1 || echo "ffmpeg NOT found"
