#!/usr/bin/env python3
"""List every non-shelf, non-box static geometry in the arena MJCF.

Settles the dispute: the code/tests claim the divider board is at x=0.53 with
its north end at y=1.70, while the MJCF body says pos="1.49 0 0.75".
"""
import re
from pathlib import Path

XML = Path(
    "/workspace/baseline/examples/supermarket_sorting/mjcf/retail_competition.xml"
)
text = XML.read_text(encoding="utf-8", errors="replace")

pattern = re.compile(
    r'<geom\b[^>]*?name="([^"]+)"[^>]*?size="([^"]+)"[^>]*?pos="([^"]+)"',
    re.DOTALL,
)
rows = []
for match in pattern.finditer(text):
    name, size, pos = match.group(1), match.group(2), match.group(3)
    if name.startswith("shelf_") or name.startswith("aruco"):
        continue
    if "dynamic_obstacle" in name:
        continue
    rows.append((name, size, pos))

print(f"{'name':38s} {'half-size':22s} {'pos':20s} world extents")
for name, size, pos in rows:
    vals = [float(v) for v in size.split()]
    p = [float(v) for v in pos.split()]
    if len(vals) == 3 and len(p) == 3:
        ext = (
            f"x[{p[0]-vals[0]:+.2f},{p[0]+vals[0]:+.2f}] "
            f"y[{p[1]-vals[1]:+.2f},{p[1]+vals[1]:+.2f}] "
            f"z[{p[2]-vals[2]:+.2f},{p[2]+vals[2]:+.2f}]"
        )
    else:
        ext = "n/a"
    print(f"{name:38s} {size:22s} {pos:20s} {ext}")

print()
print("Boxes the client models statically (from grid_planner.STATIC_OBSTACLES):")
import sys  # noqa: E402

sys.path.insert(0, "/workspace/baseline/examples/supermarket_sorting")
from navigation.grid_planner import SupermarketGridPlanner  # noqa: E402

for rect in SupermarketGridPlanner.STATIC_OBSTACLES:
    print(f"   Rect({rect.xmin:+.3f}, {rect.xmax:+.3f}, "
          f"{rect.ymin:+.3f}, {rect.ymax:+.3f})")
