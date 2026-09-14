#!/usr/bin/env python3
"""Print the NOMINAL layout kinds, i.e. what the probe actually rendered.

The renderer probe uses gen_dataset.build_sim(), which loads
retail_competition_layout.json (the nominal, non-randomised layout).  Scoring it
against runtime_layout.json (the randomised layout of a previous run) is
meaningless - the two disagree on which kind sits where.
"""
from __future__ import annotations

import json
import pathlib

TASK = pathlib.Path("/workspace/baseline/examples/supermarket_sorting")

for name in ("retail_competition_layout.json", "runtime_layout.json"):
    path = TASK / name
    raw = json.loads(path.read_text())
    items = raw if isinstance(raw, list) else raw.get("items", [])
    print(f"=== {name}: {len(items)} items ===")
    for it in items:
        key = f"{it['shelf']}_{it['level']}_{it['column']}"
        if key in {"A_L1_C1", "A_L1_C2", "A_L2_C1", "A_L2_C2", "A_L3_C1"}:
            print(f"  {key:<10} {it['object_kind']:<14} "
                  f"world={[round(float(v), 3) for v in it['world_position']]}")
    print()
