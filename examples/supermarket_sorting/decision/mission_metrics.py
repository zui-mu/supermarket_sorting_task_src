from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


class MissionMetrics:
    """Append-only JSONL event log plus lightweight mission counters."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.counters: Counter[str] = Counter()
        self._last_phase: str | None = None
        self._last_nav_pose: tuple[float, float] | None = None
        self._path_length = 0.0

    def reset_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def event(self, time_sec: float, event: str, payload: dict[str, Any] | None = None) -> None:
        self.counters[event] += 1
        record = {
            "time": float(time_sec),
            "event": event,
            "payload": payload or {},
            "counters": dict(self.counters),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def observe_phase(self, time_sec: float, phase: str, payload: dict[str, Any] | None = None) -> None:
        if phase == self._last_phase:
            return
        previous = self._last_phase
        self._last_phase = phase
        data = dict(payload or {})
        data["previous_phase"] = previous
        self.event(time_sec, "phase_changed", data)

    def observe_nav_pose(self, xy: Iterable[float] | None) -> None:
        if xy is None:
            return
        try:
            points = tuple(float(v) for v in xy)
        except (TypeError, ValueError):
            return
        if len(points) < 2:
            return
        x, y = points[0], points[1]
        if self._last_nav_pose is not None:
            px, py = self._last_nav_pose
            self._path_length += ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        self._last_nav_pose = (x, y)

    def summary(self) -> dict[str, Any]:
        return {
            "path_length_m": round(self._path_length, 3),
            "counters": dict(self.counters),
        }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    candidate = Path(path)
    if not candidate.exists():
        return records
    with candidate.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"event": "malformed_jsonl", "raw": line})
    return records
