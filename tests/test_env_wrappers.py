"""Env layer tests -- AttitudeHoldTask, the wrapper stack, and make_env. See
agent_docs/environment.md for the confirmed jsbgym API these build on.

These instantiate the real jsbgym/JSBSim env (no mocking the sim -- there's
nothing meaningful to assert about our wrappers with a fake env, since the
whole point is that they compose correctly against the real obs/action
shapes and info dict jsbgym produces). Slower than the pure-math tests, but
still a handful of seconds.
"""

from __future__ import annotations

import jsbgym.properties as prp
import numpy as np
import pytest
from jsbgym.aircraft import c172
from jsbgym.environment import NoFGJsbSimEnv
from jsbgym.tasks import Shaping

from rtrl_flight.env.attitude_task import AttitudeHoldTask
from rtrl_flight.env.make import make_env
from rtrl_flight.env.wrappers.fault import FaultWrapper
from rtrl_flight.env.wrappers.trace import TraceWrapper
from rtrl_flight.env.wrappers.wind import WindWrapper


def _make_base_env():
    return NoFGJsbSimEnv(
        aircraft=c172,
        task_type=AttitudeHoldTask,
        agent_interaction_freq=5,
        shaping=Shaping.STANDARD,
    )


# --- AttitudeHoldTask --------------------------------------------------


def test_attitude_hold_task_obs_shape_and_error_slots():
    env = _make_base_env()
    try:
        obs, _info = env.reset(seed=0)
        assert obs.shape == (11,)

        task = env.unwrapped.task
        # slots 9-10 must be our own pitch/roll error, not HeadingControlTask's
        # altitude/track error (agent_docs/environment.md, Mismatch 1).
        assert task.state_variables[9] is task.pitch_error_rad
        assert task.state_variables[10] is task.roll_error_rad
        assert task.pitch_error_rad.name == "error/pitch-error-rad"
        assert task.roll_error_rad.name == "error/roll-error-rad"

        # error = target - current, per the reset sample and current attitude.
        sim = env.unwrapped.sim
        expected_pitch_error = sim[task.target_pitch_rad] - sim[prp.pitch_rad]
        assert obs[9] == pytest.approx(expected_pitch_error, abs=1e-6)
    finally:
        env.close()


def test_attitude_hold_task_step_returns_finite_reward():
    env = _make_base_env()
    try:
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.zeros(3))
        assert obs.shape == (11,)
        assert np.isfinite(reward)
        assert terminated is False
        assert truncated is False
    finally:
        env.close()


# --- FaultWrapper --------------------------------------------------------


def test_fault_wrapper_default_full_authority():
    env = FaultWrapper(_make_base_env())
    try:
        assert env.fault_state == {"aileron": 1.0, "elevator": 1.0, "rudder": 1.0}
    finally:
        env.close()


def test_fault_wrapper_trigger_and_effective_action():
    env = FaultWrapper(_make_base_env())
    try:
        env.reset(seed=0)
        env.trigger_fault("aileron", 0.3)
        assert env.fault_state["aileron"] == 0.3
        assert env.fault_state["elevator"] == 1.0

        _obs, _reward, _term, _trunc, info = env.step(np.array([1.0, 1.0, 1.0]))
        assert info["fault"]["effective_action"] == pytest.approx([0.3, 1.0, 1.0])
        assert info["fault"]["commanded_action"] == pytest.approx([1.0, 1.0, 1.0])
    finally:
        env.close()


def test_fault_wrapper_reset_restores_full_authority():
    env = FaultWrapper(_make_base_env())
    try:
        env.reset(seed=0)
        env.trigger_fault("rudder", 0.0)
        assert env.fault_state["rudder"] == 0.0

        env.reset(seed=1)
        assert env.fault_state == {"aileron": 1.0, "elevator": 1.0, "rudder": 1.0}
    finally:
        env.close()


def test_fault_wrapper_rejects_unknown_surface_and_bad_severity():
    env = FaultWrapper(_make_base_env())
    try:
        with pytest.raises(ValueError):
            env.trigger_fault("flaps", 0.5)
        with pytest.raises(ValueError):
            env.trigger_fault("aileron", 1.5)
    finally:
        env.close()


# --- WindWrapper -----------------------------------------------------------


def test_wind_wrapper_set_severity_cycles():
    env = WindWrapper(_make_base_env(), level="off")
    try:
        env.reset(seed=0)
        assert env.wind_level == "off"

        for level in ("light", "moderate", "severe", "off"):
            env.set_severity(level)
            assert env.wind_level == level

        _obs, _reward, _term, _trunc, info = env.step(np.zeros(3))
        assert info["wind_level"] == "off"
    finally:
        env.close()


def test_wind_wrapper_rejects_unknown_level():
    env = WindWrapper(_make_base_env())
    try:
        with pytest.raises(ValueError):
            env.set_severity("hurricane")
    finally:
        env.close()


# --- TraceWrapper ------------------------------------------------------


def test_trace_wrapper_writes_parquet_with_expected_schema(tmp_path):
    env = TraceWrapper(FaultWrapper(WindWrapper(_make_base_env())), output_dir=tmp_path)
    try:
        env.reset(seed=0)
        for _ in range(4):
            env.step(np.zeros(3))
        # episode ends only when the task terminates (fixed-length episode),
        # so force a flush the same way reset() does, to check schema without
        # running a full ~300-step episode.
        env._flush()

        parquet_files = sorted(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 1

        import pandas as pd

        df = pd.read_parquet(parquet_files[0])
        assert len(df) == 4

        expected_columns = {
            "step",
            "episode",
            "action_aileron",
            "action_elevator",
            "action_rudder",
            "reward",
            "fault_aileron_severity",
            "fault_elevator_severity",
            "fault_rudder_severity",
            "wind_level",
        }
        assert expected_columns.issubset(set(df.columns))
        assert any(col.startswith("obs_") for col in df.columns)
        assert (df["fault_aileron_severity"] == 1.0).all()
        assert (df["wind_level"] == "off").all()
    finally:
        env.close()


def test_trace_wrapper_new_episode_gets_new_file(tmp_path):
    env = TraceWrapper(_make_base_env(), output_dir=tmp_path)
    try:
        env.reset(seed=0)
        env.step(np.zeros(3))
        env._flush()

        env.reset(seed=1)
        env.step(np.zeros(3))
        env._flush()

        parquet_files = sorted(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 2
    finally:
        env.close()


# --- make_env ------------------------------------------------------------


def test_make_env_nominal_config_instantiates():
    env = make_env({"agent_interaction_freq": 5, "normalize": True})
    try:
        obs, _info = env.reset(seed=0)
        assert obs.shape == (11,)
        assert obs.min() >= -1.0 and obs.max() <= 1.0

        action = np.zeros(3)
        obs, reward, terminated, truncated, _info = env.step(action)
        assert obs.shape == (11,)
        assert obs.min() >= -1.0 and obs.max() <= 1.0
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_make_env_full_stack_config(tmp_path):
    cfg = {
        "agent_interaction_freq": 5,
        "normalize": True,
        "fault": {"enabled": True},
        "wind": {"enabled": True, "level": "light"},
        "trace": {"enabled": True, "output_dir": str(tmp_path)},
    }
    env = make_env(cfg)
    try:
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (11,)
        assert "fault" in info
        assert info["wind_level"] == "light"
        # episode not finished yet -- nothing flushed until reset()/close().
        assert list(tmp_path.glob("*.parquet")) == []
    finally:
        env.close()
    # close() flushes any buffered rows so no data is lost on shutdown.
    assert len(list(tmp_path.glob("*.parquet"))) == 1
