"""Static arm reachability constraints verified in the competition scene."""

from __future__ import annotations


# With the base facing shelf A--E, a right-arm overhead clamp on the top
# rightmost tissue slot sweeps rgt_arm_link6 through the shelf's right-front
# post. A mirrored left-arm executor is required for this geometry.
RIGHT_ARM_TOP_BOX_POST_BLOCKED_SLOTS = {("L3", "C3")}


def requires_mirrored_left_arm(task) -> bool:
    """Return whether the current right-arm-only executor cannot pick ``task``."""
    return (
        str(getattr(task, "product_name", "")) == "zhijin"
        and (
            str(getattr(task, "level", "")),
            str(getattr(task, "column", "")),
        ) in RIGHT_ARM_TOP_BOX_POST_BLOCKED_SLOTS
    )
