"""Tracking metrics computed from a loaded parquet trace DataFrame. See
agent_docs/evaluation.md's "Metrics" section for definitions. Per
agent_docs/architecture.md design invariant #4, these operate ONLY on the
trace (a pandas DataFrame), never on live env/controller state.

UNITS WARNING: these assume the trace's obs_error_pitch_error_rad /
obs_error_roll_error_rad columns are in real radians, not NormalizeWrapper's
[-1, 1]-squashed range. Every controller in this repo is trained/calibrated
on raw-scale obs (see rtrl_flight.env.make.force_raw_obs's docstring for
why), so the experiment runner (scripts/run_experiment.py) always builds its
evaluation env with normalize forced off -- these functions do not, and
cannot, detect or correct a normalized trace themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PITCH_ERROR_COL = "obs_error_pitch_error_rad"
ROLL_ERROR_COL = "obs_error_roll_error_rad"
REWARD_COL = "reward"
STEP_COL = "step"


def pitch_rmse(df: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean(np.square(df[PITCH_ERROR_COL]))))


def roll_rmse(df: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean(np.square(df[ROLL_ERROR_COL]))))


def attitude_rmse(df: pd.DataFrame) -> float:
    """Combined RMSE: sqrt(mean(pitch_error^2 + roll_error^2)), per
    agent_docs/evaluation.md ("RMSE of (pitch - target_pitch, roll -
    target_roll)")."""
    combined_sq = np.square(df[PITCH_ERROR_COL]) + np.square(df[ROLL_ERROR_COL])
    return float(np.sqrt(np.mean(combined_sq)))


def settling_time(df: pd.DataFrame, threshold_rad: float = 0.05) -> float | None:
    """Steps until |pitch_error| AND |roll_error| both stay < threshold_rad
    for the remainder of the episode. Returns the step value (from the
    `step` column) of the first such index, or None if the trace never
    settles (i.e. some later step always violates the threshold again).
    """
    within = (df[PITCH_ERROR_COL].abs() < threshold_rad) & (
        df[ROLL_ERROR_COL].abs() < threshold_rad
    )
    within_arr = within.to_numpy()

    # suffix_all[i] = True iff within_arr[i:] are all True -- computed via a
    # reverse scan since numpy has no built-in "all from here to the end".
    suffix_all = np.zeros(len(within_arr), dtype=bool)
    running = True
    for i in range(len(within_arr) - 1, -1, -1):
        running = running and bool(within_arr[i])
        suffix_all[i] = running

    settled = np.nonzero(suffix_all)[0]
    if len(settled) == 0:
        return None
    return float(df[STEP_COL].iloc[int(settled[0])])


def episode_return(df: pd.DataFrame) -> float:
    return float(df[REWARD_COL].sum())
