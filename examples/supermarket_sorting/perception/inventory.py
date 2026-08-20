"""Pure helpers for binding product detections to visible shelf ArUco tags."""

from __future__ import annotations

import math


def associate_detections_to_markers(detections, markers, *, max_gap_px=150):
    """Associate a product box with the tag immediately below it.

    ``detections`` use the backend ``x/y/w/h`` centre representation and
    ``markers`` contain ``id`` plus a four-corner pixel array.  Ambiguous tags
    are intentionally omitted rather than guessed.
    """
    matches = []
    used_marker_ids = set()
    for index, detection in enumerate(detections):
        x0 = float(detection["x"]) - float(detection["w"]) / 2.0
        x1 = float(detection["x"]) + float(detection["w"]) / 2.0
        y1 = float(detection["y"]) + float(detection["h"]) / 2.0
        width = max(1.0, float(detection["w"]))
        candidates = []
        for marker in markers:
            marker_id = int(marker["id"])
            if marker_id in used_marker_ids:
                continue
            corners = marker["corners"]
            mx = sum(float(point[0]) for point in corners) / len(corners)
            my = sum(float(point[1]) for point in corners) / len(corners)
            if my < y1 - 8.0 or my - y1 > max_gap_px:
                continue
            if mx < x0 - 0.35 * width or mx > x1 + 0.35 * width:
                continue
            score = abs(mx - float(detection["x"])) / width + (my - y1) / max_gap_px
            candidates.append((score, marker_id))
        if not candidates:
            continue
        candidates.sort()
        score, marker_id = candidates[0]
        if len(candidates) > 1 and candidates[1][0] - score < 0.15:
            continue
        used_marker_ids.add(marker_id)
        matches.append({"detection_index": index, "aruco_id": marker_id, "score": score})
    return matches
