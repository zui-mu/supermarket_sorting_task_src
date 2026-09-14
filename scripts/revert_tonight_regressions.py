#!/usr/bin/env python3
"""Revert the regressions introduced by tonight's ad-hoc patches.

The three independent audits agreed that these specific edits made behaviour
worse than before:

  * VISION_MONITOR_MAX_SHIFT_* relaxed past the referee's own C2 threshold
    (the file's own comment forbids it)
  * RELOCK_MAX_SHIFT_Z raised so the "target toppled" branch could never win
  * current_target_touched() alone gates the vision correction, and that
    function is always False in official mode -> retries lost live vision
  * DEPLOY_LATERAL_TOL tightened while CREEP_PRECONTACT_GUARD_LATERAL was
    loosened: opposite directions, different coordinate frames
  * per-product vision_monitor / guard values relaxed the same way
  * _unique_slots ordering change was a no-op (execution order comes from
    order_scheduler.level_order), so it only added churn

Every replacement asserts it matched exactly once.
"""
from pathlib import Path

ROOT = Path("/workspace/baseline/examples/supermarket_sorting")
CLIENT = ROOT / "supermarket_sorting_client.py"

EDITS = [
    # 1. relock z back below the (restored) monitor threshold
    (
        'RELOCK_MAX_SHIFT_Z = float(os.getenv("SUPERMARKET_RELOCK_MAX_SHIFT_Z", "0.300"))',
        'RELOCK_MAX_SHIFT_Z = float(os.getenv("SUPERMARKET_RELOCK_MAX_SHIFT_Z", "0.120"))',
    ),
    # 2/3. kele + maidong vision monitor thresholds back to the audited values
    (
        '''        # 2026-08-23: 0.040 -> 0.100 xy and 0.080 -> 0.250 z.  The y/z halves of
        # a depth-deprojected live point carry RGB-D noise (z up to 0.44 m on
        # white packaging), so the old 4/8 cm gate discarded healthy items as
        # "displaced".  xy stays at 10 cm so a real 14 cm shove is still caught.
        "vision_monitor_max_shift_xy": 0.100,
        "vision_monitor_max_shift_z": 0.250,''',
        '''        # Must stay below the referee's C2 displacement line (5 cm); RGB-D
        # noise is handled by multi-frame agreement, not by moving this bar.
        "vision_monitor_max_shift_xy": 0.040,
        "vision_monitor_max_shift_z": 0.080,''',
    ),
    (
        '''        # 2026-08-23: relaxed together with kele for the real YOLO pipeline
        # (depth-deprojected live z carries RGB-D noise; the old 6/9 cm gate
        # discarded healthy items as "displaced").
        "vision_monitor_max_shift_xy": 0.100,
        "vision_monitor_max_shift_z": 0.250,''',
        '''        "vision_monitor_max_shift_xy": 0.060,
        "vision_monitor_max_shift_z": 0.090,''',
    ),
    # 4. restore live-vision correction on pre-contact retries
    (
        '''            # 2026-08-23: only adopt vision X/Y when the referee actually
            # reports contact (the product may then have been nudged).  The old
            # `or self.local_grasp_retries > 0` clause injected up to +-12 cm of
            # RGB-D noise into an otherwise exact slot truth on EVERY pre-contact
            # retry: official mode has no touch oracle, so current_target_touched()
            # is always False and every retry re-aimed at a noisy point - the
            # exact shape of the 3-in-a-row "lateral alignment error" failures.
            if self.current_target_touched():''',
        '''            # Restored after audit: current_target_touched() is always False in
            # official mode, so gating on it alone silently disabled the ONLY
            # live measurement available to a pre-contact retry.
            if self.local_grasp_retries > 0 or self.current_target_touched():''',
    ),
    # 5. deploy gate back to the (wider) documented value
    (
        'DEPLOY_LATERAL_TOL = float(os.getenv("SUPERMARKET_DEPLOY_LATERAL_TOL", "0.030"))',
        'DEPLOY_LATERAL_TOL = float(os.getenv("SUPERMARKET_DEPLOY_LATERAL_TOL", "0.045"))',
    ),
    # 6. creep guard back to 0.040
    (
        'CREEP_PRECONTACT_GUARD_LATERAL = float(os.getenv("SUPERMARKET_CREEP_PRECONTACT_GUARD_LATERAL", "0.045"))',
        'CREEP_PRECONTACT_GUARD_LATERAL = float(os.getenv("SUPERMARKET_CREEP_PRECONTACT_GUARD_LATERAL", "0.040"))',
    ),
    # 7/8. kele per-product creep gates back to the pinned regression values
    (
        '''        "creep_near_lateral_abort": 0.040,
        "creep_precontact_guard_distance": 0.24,
        # 2026-08-23: 0.026 -> 0.045 (see CREEP_PRECONTACT_GUARD_LATERAL).  The
        # 2.6 cm kele gate fired the retreat on nearly every real approach, yet
        # 4.5 cm still catches the pinned 4.9 cm sideways-sweep case.
        "creep_precontact_guard_lateral": 0.045,''',
        '''        "creep_near_lateral_abort": 0.030,
        "creep_precontact_guard_distance": 0.24,
        "creep_precontact_guard_lateral": 0.026,''',
    ),
]

text = CLIENT.read_text(encoding="utf-8")
for index, (old, new) in enumerate(EDITS, start=1):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"edit {index}: matched {count} times (expected 1)")
    text = text.replace(old, new)
    print(f"edit {index}: ok")

CLIENT.write_text(text, encoding="utf-8")
print("reverted")
