"""Standalone attitude-tracking reward function -- called by AttitudeHoldTask
(rtrl_flight.env.attitude_task) but kept free of any jsbgym/Simulation
dependency so it's directly unit-testable and reusable (e.g. for imitation
warm-start loss shaping, per agent_docs/experiments.md).

Formula (negative weighted L2, per agent_docs/environment.md's "Reward"
section): tracking error on pitch/roll dominates, with a rate-penalty term
for smoothness (ties to the jerk metric in agent_docs/evaluation.md) and a
small action-magnitude penalty discouraging unnecessarily aggressive control.
Unbounded (more negative for larger errors) -- clipping to a fixed range is
NormalizeWrapper's job (rtrl_flight.env.wrappers.normalize), not this
function's, so the raw signal stays inspectable for debugging.

Default weights are a starting point, not tuned -- changing them is a
"what to ask first" item per CLAUDE.md since it invalidates prior runs, so
route real changes through configs/env/*.yaml once that config exists rather
than editing the defaults here.
"""

from __future__ import annotations

from collections.abc import Sequence

# Tracking error dominates (pitch/roll each weighted 1.0). Rate penalty
# discourages oscillation. Action penalty discourages large/aggressive
# commands. All weights >= 0.
DEFAULT_W_PITCH = 1.0
DEFAULT_W_ROLL = 1.0
DEFAULT_W_RATE = 0.1
DEFAULT_W_ACTION = 0.01


def attitude_tracking_reward(
    pitch_error_rad: float,
    roll_error_rad: float,
    p_rad_s: float,
    q_rad_s: float,
    r_rad_s: float,
    action: Sequence[float],
    *,
    w_pitch: float = DEFAULT_W_PITCH,
    w_roll: float = DEFAULT_W_ROLL,
    w_rate: float = DEFAULT_W_RATE,
    w_action: float = DEFAULT_W_ACTION,
) -> float:
    """reward = -(w_pitch*pitch_error^2 + w_roll*roll_error^2
                  + w_rate*(p^2+q^2+r^2) + w_action*sum(action_i^2))

    All error/rate arguments in radians / radians-per-second. `action` is
    the raw (aileron, elevator, rudder) command in [-1, 1] -- pre-fault
    commanded action, not the fault-degraded effective one; the fault's
    effect on tracking shows up through pitch_error/roll_error/rates, not by
    penalizing the controller twice for a fault it didn't command.
    """
    tracking_cost = w_pitch * pitch_error_rad**2 + w_roll * roll_error_rad**2
    rate_cost = w_rate * (p_rad_s**2 + q_rad_s**2 + r_rad_s**2)
    action_cost = w_action * sum(a**2 for a in action)
    return -(tracking_cost + rate_cost + action_cost)
