"""Time-to-recover metric (fault scenarios only). See agent_docs/
evaluation.md's "Time-to-recover" definition.

NOTE: evaluation.md defines recovery relative to a controller's own
pre-fault RMSE ("returns to within a factor ... e.g. 1.2x ... of its
pre-fault value"), which needs a separate pre-fault baseline computed over
a window before onset. This message's explicit spec instead uses a single
fixed absolute threshold_rad with no pre-fault baseline -- simpler and
self-contained from one trace, but a real simplification of evaluation.md's
definition, not a silent substitution. Implemented per the explicit spec
here; a pre-fault-relative variant could sit alongside this one if the
fixed threshold proves too strict/loose across severities/controllers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rtrl_flight.metrics.tracking import PITCH_ERROR_COL, ROLL_ERROR_COL, STEP_COL

FAULT_SEVERITY_COLS = (
    "fault_aileron_severity",
    "fault_elevator_severity",
    "fault_rudder_severity",
)
CONSECUTIVE_STEPS_REQUIRED = 10


def time_to_recover(df: pd.DataFrame, threshold_rad: float = 0.1) -> float | None:
    """Finds the fault onset step (first step where any fault_*_severity
    < 1.0), then the first step at/after onset where the pointwise combined
    attitude error (sqrt(pitch_error^2 + roll_error^2)) drops below
    threshold_rad and stays there for CONSECUTIVE_STEPS_REQUIRED consecutive
    steps. Returns steps-from-onset (using the `step` column, not raw row
    index, in case a trace is ever non-contiguous), or None if:
      - no fault is present in this trace at all, or
      - no such recovery window occurs before the episode ends.
    """
    fault_active = (df[list(FAULT_SEVERITY_COLS)] < 1.0).any(axis=1)
    onset_positions = np.nonzero(fault_active.to_numpy())[0]
    if len(onset_positions) == 0:
        return None
    onset_pos = int(onset_positions[0])

    pointwise_attitude_error = np.sqrt(
        np.square(df[PITCH_ERROR_COL].to_numpy()) + np.square(df[ROLL_ERROR_COL].to_numpy())
    )
    below_threshold = pointwise_attitude_error < threshold_rad

    n = len(below_threshold)
    for pos in range(onset_pos, n):
        window_end = pos + CONSECUTIVE_STEPS_REQUIRED
        if window_end > n:
            break
        if below_threshold[pos:window_end].all():
            return float(df[STEP_COL].iloc[pos] - df[STEP_COL].iloc[onset_pos])

    return None
