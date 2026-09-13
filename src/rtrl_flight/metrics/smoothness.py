"""Control smoothness (jerk) metric.

NOTE: agent_docs/evaluation.md describes jerk as "mean absolute second
derivative of each action channel over time." This message's explicit spec
instead defines it as a first difference (mean absolute step-to-step
change, summed across the three surfaces): "mean absolute step-to-step
change in each control surface ... mean of |Δaileron| + |Δelevator| +
|Δrudder|". Implemented per that explicit, self-contained formula rather
than evaluation.md's phrasing -- flagging the discrepancy rather than
silently picking one. A true second-derivative (jerk) variant could be
added alongside this one later if evaluation.md's definition is the one
that should actually drive the acceptance bar.
"""

from __future__ import annotations

import pandas as pd

ACTION_COLS = ("action_aileron", "action_elevator", "action_rudder")


def control_jerk(df: pd.DataFrame) -> float:
    """mean_t(|Δaileron_t| + |Δelevator_t| + |Δrudder_t|). The first row's
    Δ is undefined (no previous step) and excluded from the mean -- pandas'
    .diff() leaves it NaN and .mean() skips NaN by default.
    """
    total_abs_delta = sum(df[col].diff().abs() for col in ACTION_COLS)
    return float(total_abs_delta.mean())
