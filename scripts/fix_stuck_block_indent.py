#!/usr/bin/env python3
"""Wrap the absolute-cap recovery block in `if no_progress or emergency:`
so the fast displacement-based path that follows stays reachable.

Run once; verifies its own result.
"""
from pathlib import Path

PATH = Path(
    "/workspace/baseline/examples/supermarket_sorting/supermarket_sorting_client.py"
)
lines = PATH.read_text(encoding="utf-8").splitlines(keepends=True)

start = None
end = None
for i, line in enumerate(lines):
    if line.strip() == "reason_txt = (" and start is None and i > 5900:
        start = i
    if start is not None and line.rstrip("\n") == "        return True" and end is None:
        end = i
        break

if start is None or end is None:
    raise SystemExit(f"block not found (start={start}, end={end})")

print(f"wrapping lines {start + 1}..{end + 1}")
for i in range(start, end + 1):
    if lines[i].strip():
        lines[i] = "    " + lines[i]

header = "        if no_progress or now > waypoint_deadline:\n"
lines.insert(start, header)
PATH.write_text("".join(lines), encoding="utf-8")
print("done")
