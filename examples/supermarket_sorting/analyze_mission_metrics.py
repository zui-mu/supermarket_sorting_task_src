#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from decision.mission_metrics import load_jsonl

# 官方"无间歇连续作业"奖励以 ≤15s 停顿为基准
PAUSE_THRESHOLD_SEC = 15.0
PAUSE_MIN_SHIFT_M = 0.05


def _pauses_from_heartbeats(records):
    beats = [record for record in records if record.get("event") == "nav_heartbeat"]
    pauses = []
    for prev, cur in zip(beats, beats[1:]):
        try:
            t0, t1 = float(prev["time"]), float(cur["time"])
        except (TypeError, ValueError):
            continue
        p0 = prev.get("payload") or {}
        p1 = cur.get("payload") or {}
        if p0.get("x") is None or p1.get("x") is None:
            continue
        shift = math.hypot(float(p1["x"]) - float(p0["x"]),
                           float(p1["y"]) - float(p0["y"]))
        if shift < PAUSE_MIN_SHIFT_M and (t1 - t0) > PAUSE_THRESHOLD_SEC:
            pauses.append((t0, t1, p1.get("phase")))
    return pauses


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize supermarket mission JSONL metrics.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "mission_metrics.jsonl"),
        help="metrics JSONL path",
    )
    args = parser.parse_args()
    records = load_jsonl(args.path)
    if not records:
        print(f"No metrics found: {args.path}")
        return

    events = Counter(str(record.get("event")) for record in records)
    phase_counts = Counter()
    failures: list[str] = []
    successes = 0
    latest_summary = {}
    for record in records:
        event = record.get("event")
        payload = record.get("payload") or {}
        if event == "phase_changed":
            phase = payload.get("product"), payload.get("task_id")
            phase_counts[str(payload.get("previous_phase")) + " -> " + str(phase)] += 1
        elif event == "task_failed":
            failures.append(str(payload.get("reason", "")))
            latest_summary = payload.get("summary") or latest_summary
        elif event == "task_succeeded":
            successes += 1
            latest_summary = payload.get("summary") or latest_summary

    print("Mission metrics summary")
    print(f"records: {len(records)}")
    print(f"successes: {successes}")
    print(f"failures: {len(failures)}")
    if latest_summary:
        print(f"path_length_m: {latest_summary.get('path_length_m', 'n/a')}")
    print("\nevents:")
    for name, count in events.most_common():
        print(f"  {name}: {count}")
    if failures:
        print("\nfailure reasons:")
        for reason, count in Counter(failures).most_common():
            print(f"  {count}x {reason}")

    pauses = _pauses_from_heartbeats(records)
    print(f"\ncontinuous-operation pauses (>{PAUSE_THRESHOLD_SEC:.0f}s stationary): {len(pauses)}")
    for t0, t1, phase in pauses:
        print(f"  {t0:7.1f}s -> {t1:7.1f}s ({t1 - t0:5.1f}s) phase={phase}")


if __name__ == "__main__":
    main()
