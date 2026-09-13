"""Tests for rtrl_flight.metrics.{tracking,smoothness,recovery} -- all
operate on synthetic pandas DataFrames matching the trace schema
TraceWrapper actually writes (see tests/test_env_wrappers.py's
test_trace_wrapper_writes_parquet_with_expected_schema for the real column
names).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rtrl_flight.metrics.recovery import time_to_recover
from rtrl_flight.metrics.smoothness import control_jerk
from rtrl_flight.metrics.tracking import (
    attitude_rmse,
    episode_return,
    pitch_rmse,
    roll_rmse,
    settling_time,
)


def _make_df(
    pitch_error: list[float],
    roll_error: list[float],
    reward: list[float] | None = None,
    action_aileron: list[float] | None = None,
    action_elevator: list[float] | None = None,
    action_rudder: list[float] | None = None,
    fault_aileron_severity: list[float] | None = None,
    fault_elevator_severity: list[float] | None = None,
    fault_rudder_severity: list[float] | None = None,
) -> pd.DataFrame:
    n = len(pitch_error)
    return pd.DataFrame(
        {
            "step": list(range(n)),
            "episode": [0] * n,
            "obs_error_pitch_error_rad": pitch_error,
            "obs_error_roll_error_rad": roll_error,
            "action_aileron": action_aileron or [0.0] * n,
            "action_elevator": action_elevator or [0.0] * n,
            "action_rudder": action_rudder or [0.0] * n,
            "reward": reward or [0.0] * n,
            "fault_aileron_severity": fault_aileron_severity or [1.0] * n,
            "fault_elevator_severity": fault_elevator_severity or [1.0] * n,
            "fault_rudder_severity": fault_rudder_severity or [1.0] * n,
            "wind_level": ["off"] * n,
        }
    )


# --- tracking.py -----------------------------------------------------------


def test_pitch_and_roll_rmse():
    df = _make_df(pitch_error=[0.1, -0.1, 0.2], roll_error=[0.0, 0.0, 0.0])
    assert pitch_rmse(df) == pytest.approx(np.sqrt((0.01 + 0.01 + 0.04) / 3))
    assert roll_rmse(df) == pytest.approx(0.0)


def test_attitude_rmse_combines_pitch_and_roll():
    df = _make_df(pitch_error=[0.1, 0.1], roll_error=[0.1, 0.1])
    expected = np.sqrt(np.mean([0.01 + 0.01, 0.01 + 0.01]))
    assert attitude_rmse(df) == pytest.approx(expected)


def test_settling_time_finds_first_permanently_settled_step():
    # Dips below threshold at step 1 but pops back above at step 2 -- must
    # not count step 1 as settled. Truly settles from step 3 onward.
    pitch = [0.2, 0.01, 0.2, 0.01, 0.01, 0.01]
    roll = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    df = _make_df(pitch_error=pitch, roll_error=roll)
    assert settling_time(df, threshold_rad=0.05) == 3.0


def test_settling_time_none_if_never_settles():
    df = _make_df(pitch_error=[0.2, 0.2, 0.2], roll_error=[0.0, 0.0, 0.0])
    assert settling_time(df, threshold_rad=0.05) is None


def test_episode_return_sums_reward_column():
    df = _make_df(pitch_error=[0, 0, 0], roll_error=[0, 0, 0], reward=[1.0, -2.0, 0.5])
    assert episode_return(df) == pytest.approx(-0.5)


# --- smoothness.py -----------------------------------------------------------


def test_control_jerk_is_mean_summed_abs_delta():
    df = _make_df(
        pitch_error=[0, 0, 0],
        roll_error=[0, 0, 0],
        action_aileron=[0.0, 1.0, 1.0],
        action_elevator=[0.0, 0.0, 2.0],
        action_rudder=[0.0, 0.0, 0.0],
    )
    # deltas: aileron [nan,1,0], elevator [nan,0,2], rudder [nan,0,0]
    # summed per-step (excluding first row): [1, 2] -> mean = 1.5
    assert control_jerk(df) == pytest.approx(1.5)


# --- recovery.py -------------------------------------------------------------


def test_time_to_recover_finds_sustained_recovery_after_onset():
    n = 20
    onset_step = 5
    pitch = [0.0] * onset_step + [0.3] * 3 + [0.01] * (n - onset_step - 3)
    roll = [0.0] * n
    fault_aileron = [1.0] * onset_step + [0.5] * (n - onset_step)
    df = _make_df(
        pitch_error=pitch,
        roll_error=roll,
        fault_aileron_severity=fault_aileron,
    )
    # error drops below 0.1 starting at index onset_step+3=8, stays there
    # for the remaining 12 steps (>= 10 required) -> recovers 3 steps after
    # onset.
    result = time_to_recover(df, threshold_rad=0.1)
    assert result == pytest.approx(3.0)


def test_time_to_recover_none_if_no_fault_present():
    df = _make_df(pitch_error=[0.01] * 15, roll_error=[0.01] * 15)
    assert time_to_recover(df) is None


def test_time_to_recover_none_if_never_recovers():
    n = 15
    onset_step = 3
    fault_aileron = [1.0] * onset_step + [0.5] * (n - onset_step)
    df = _make_df(
        pitch_error=[0.0] * onset_step + [0.3] * (n - onset_step),
        roll_error=[0.0] * n,
        fault_aileron_severity=fault_aileron,
    )
    assert time_to_recover(df, threshold_rad=0.1) is None
