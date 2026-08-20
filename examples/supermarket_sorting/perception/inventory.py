"""Pure helpers for binding product detections to visible shelf ArUco tags.

Round 62 (PR1): arUco -> slot identity decoding and robust one-to-one
association with rejection reasons.  The association stays in the IMAGE
plane (a tag below the product box, same column, sufficient score margin) so
an unstable depth never decides "which slot this product belongs to".
"""

from __future__ import annotations

import math

# Official fixed mapping: ArUco id -> shelf/level/column (ids 0..44).
#   shelf  = id // 9          (0=A, 1=B, 2=C, 3=D, 4=E)
#   level  = (id % 9) // 3    (0=L1, 1=L2, 2=L3)
#   column = id % 3           (0=C1, 1=C2, 2=C3)
SHELF_LETTERS = ("A", "B", "C", "D", "E")
LEVEL_LETTERS = ("L1", "L2", "L3")
COLUMN_LETTERS = ("C1", "C2", "C3")

# Minimum margin between the best and the second-best association for the
# best one to be accepted (a close second candidate is ambiguity, not a hit).
ASSOC_MIN_MARGIN = float(__import__("os").getenv("SUPERMARKET_ASSOC_MIN_MARGIN", "0.12"))
# Absolute score ceiling: even the best candidate above this is rejected
# (tag too far / too weakly under the box).  Keeps a far neighbour's tag from
# being grabbed when nothing better exists (PR5).
ASSOC_MAX_SCORE = float(__import__("os").getenv("SUPERMARKET_ASSOC_MAX_SCORE", "1.5"))


def aruco_id_to_slot(aruco_id: int) -> dict | None:
    """Decode an ArUco id into its fixed shelf slot identity.

    Returns ``{"shelf","level","column","slot_id","shelf_index"}`` or None
    for ids outside 0..44.  This is the *identity* anchor: the tag tells you
    WHICH slot, never what product is in it.
    """
    try:
        aid = int(aruco_id)
    except (TypeError, ValueError):
        return None
    if not 0 <= aid < 45:
        return None
    shelf_index = aid // 9
    level_index = (aid % 9) // 3
    column_index = aid % 3
    return {
        "aruco_id": aid,
        "shelf_index": shelf_index,
        "shelf": SHELF_LETTERS[shelf_index],
        "level": LEVEL_LETTERS[level_index],
        "column": COLUMN_LETTERS[column_index],
        "slot_id": f"slot_{SHELF_LETTERS[shelf_index]}_{LEVEL_LETTERS[level_index]}_{COLUMN_LETTERS[column_index]}",
    }


def _corner_center(corners) -> tuple[float, float]:
    """Mean of a four-corner pixel array."""
    return (
        sum(float(point[0]) for point in corners) / len(corners),
        sum(float(point[1]) for point in corners) / len(corners),
    )


def _bbox_geom(detection: dict) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) from a backend x/y/w/h centre representation."""
    w = max(1.0, float(detection["w"]))
    h = max(1.0, float(detection["h"]))
    x = float(detection["x"])
    y = float(detection["y"])
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def associate_detections_to_markers(detections, markers, *, max_gap_px=150):
    """Associate a product box with the tag immediately below it (one-to-one).

    Round 62 (PR1): keep the original greedy API for compatibility but use the
    scored one-to-one matcher so two products can never claim the same tag.
    """
    matches, _ = match_detections_to_markers(
        detections, markers, max_gap_px=max_gap_px
    )
    return matches


def match_detections_to_markers(detections, markers, *, max_gap_px=150):
    """Robust one-to-one image-plane association.

    Returns ``(matches, details)``.  ``matches`` are dicts with
    ``detection_index``, ``aruco_id``, ``score`` (lower better) and
    ``reject_reason``/``ambiguous`` markers.  ``details`` is a per-detection
    list of candidate scores for diagnostics.

    Rules (per official V2.0: tag = slot identity, product = random):
      * tag must lie below the product box bottom (within ``max_gap_px``);
      * tag horizontal centre must be inside the box x-range (with 0.35 box
        width tolerance);
      * one tag per product, one product per tag (one-to-one, greedy by score);
      * if the best and second-best are within ``ASSOC_MIN_MARGIN`` the pair
        is ambiguous -> rejected (never guess).
    """
    matches: list[dict] = []
    details: list[list[tuple[float, int]]] = []
    used_marker_ids: set[int] = set()

    # Build per-detection candidate scores first.
    candidates_by_det: list[list[tuple[float, int, str]]] = []
    for detection in detections:
        x0, y0, x1, y1 = _bbox_geom(detection)
        width = max(1.0, x1 - x0)
        cands: list[tuple[float, int, str]] = []
        for marker in markers:
            marker_id = int(marker["id"])
            mx, my = _corner_center(marker["corners"])
            if my < y1 - 8.0 or my - y1 > max_gap_px:
                continue  # not below the box
            if mx < x0 - 0.35 * width or mx > x1 + 0.35 * width:
                continue  # not same column
            score = abs(mx - (x0 + x1) / 2.0) / width + (my - y1) / max_gap_px
            cands.append((score, marker_id, "ok"))
        cands.sort(key=lambda c: c[0])
        candidates_by_det.append(cands)

    # PR5: decide ambiguity BEFORE any greedy allocation.  A detection with
    # two near-equal candidates is ambiguous regardless of what other
    # detections do - the old greedy could leave the "real" candidate taken by
    # a neighbour, then suddenly the ambiguous second became "the" answer.
    ambiguous_dets: set[int] = set()
    for det_index, cands in enumerate(candidates_by_det):
        if len(cands) > 1 and cands[1][0] - cands[0][0] < ASSOC_MIN_MARGIN:
            ambiguous_dets.add(det_index)

    # Greedy one-to-one: iterate detections by best-score; a tag used once is
    # removed, so two products can never claim the same ArUco.  Ambiguous
    # detections never take part in the allocation.
    ranked = sorted(
        enumerate(candidates_by_det), key=lambda pair: pair[1][0][0] if pair[1] else 1e9
    )
    for det_index, cands in ranked:
        details.append([(score, mid) for score, mid, _ in cands])
        if det_index in ambiguous_dets:
            matches.append({
                "detection_index": det_index,
                "aruco_id": None,
                "score": cands[0][0] if cands else float("inf"),
                "reject_reason": "ambiguous_two_markers",
                "ambiguous": True,
            })
            continue
        viable = [c for c in cands if c[1] not in used_marker_ids]
        if not viable:
            matches.append({
                "detection_index": det_index,
                "aruco_id": None,
                "score": float("inf"),
                "reject_reason": "no_marker_below",
                "ambiguous": False,
            })
            continue
        best_score, best_id, _ = viable[0]
        if best_score > ASSOC_MAX_SCORE:
            matches.append({
                "detection_index": det_index,
                "aruco_id": None,
                "score": best_score,
                "reject_reason": "score_above_max",
                "ambiguous": False,
            })
            continue
        used_marker_ids.add(best_id)
        matches.append({
            "detection_index": det_index,
            "aruco_id": best_id,
            "score": best_score,
            "reject_reason": "ok",
            "ambiguous": False,
        })
    return matches, details
