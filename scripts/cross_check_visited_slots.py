#!/usr/bin/env python3
"""Cross-check the slots the client actually visited against the true layout.

The official run failed 77 times with "anonymous search candidate was not
visually locked".  That message can mean two very different things:

  (a) the model never labels a requested kind correctly (a perception bug), or
  (b) the client kept re-visiting slots that genuinely hold non-requested kinds
      (a search-strategy bug).

This script reads the decision log's failed task list and the runtime layout
that the server exported for the same run, and reports which of the two it was.
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

REPO = pathlib.Path("/workspace/baseline")
LOG = REPO / (sys.argv[1] if len(sys.argv) > 1 else "logs_official/decision_client.log")
LAYOUT = REPO / (sys.argv[2] if len(sys.argv) > 2 else
                 "examples/supermarket_sorting/runtime_layout.json")

REQUESTED = {"sanmingzhi", "heweidao", "shupian", "maidong"}   # zhijin is skipped

# ---- true layout -----------------------------------------------------------
raw = json.loads(LAYOUT.read_text())
items = raw if isinstance(raw, list) else raw.get("items", [])
by_slot: dict[str, dict] = {}
for it in items:
    slot = f"slot_{it['shelf']}_{it['level']}_{it['column']}"
    by_slot[slot] = it

print(f"runtime layout: {len(items)} items, {len(by_slot)} distinct slots")
kinds = collections.Counter(it.get("object_kind") for it in items)
print("true kinds in scene:", dict(sorted(kinds.items())))
wanted_slots = {s: it["object_kind"] for s, it in by_slot.items()
                if it.get("object_kind") in REQUESTED}
print(f"slots truly holding a REQUESTED kind: {len(wanted_slots)}")
for slot, kind in sorted(wanted_slots.items()):
    print(f"   {slot:<18} {kind}")

# ---- what the client did ---------------------------------------------------
text = LOG.read_text()
failed = re.findall(r'"slot_id": "(slot_[A-Z0-9_]+)"', text)
counts = collections.Counter(failed)
print(f"\ndecision log: {len(failed)} task records, {len(counts)} distinct slots")
print("visit counts (slot -> attempts, true kind):")
for slot, n in counts.most_common():
    kind = by_slot.get(slot, {}).get("object_kind", "??")
    mark = "  <== REQUESTED" if kind in REQUESTED else ""
    print(f"   {slot:<18} x{n:<4} true={kind}{mark}")

visited_requested = [s for s in counts if by_slot.get(s, {}).get("object_kind") in REQUESTED]
never_visited = sorted(set(wanted_slots) - set(counts))
print(f"\nrequested slots VISITED   : {len(visited_requested)} "
      f"{sorted(visited_requested)}")
print(f"requested slots NEVER seen: {len(never_visited)} {never_visited}")
print()
if not visited_requested:
    print("VERDICT: the client never once visited a slot that truly holds a "
          "requested kind -> SEARCH-STRATEGY bug, not a perception bug.")
elif len(never_visited):
    print("VERDICT: the client visited some requested slots but skipped others, "
          "and still never bound one -> PERCEPTION bug (model mislabels the "
          "requested kinds) or the bind path is broken.")
else:
    print("VERDICT: every requested slot was visited and none bound -> "
          "PERCEPTION bug.")
