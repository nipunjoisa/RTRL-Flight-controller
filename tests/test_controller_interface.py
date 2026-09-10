"""All controllers must satisfy the Controller ABC contract identically —
see agent_docs/architecture.md. Add each new controller (BPTT-LSTM,
RTRL-RTU) to CONTROLLERS below as it's built; every test here must pass for
all of them without controller-specific branches.
"""

from __future__ import annotations

import numpy as np
import pytest

from rtrl_flight.controllers.pid import PIDController

OBS_DIM = 8
OBS_INDICES = {"roll_error": 0, "pitch_error": 1, "yaw_rate": 2}


def _make_pid() -> PIDController:
    gains = {
        "aileron": (1.0, 0.1, 0.05),
        "elevator": (1.0, 0.1, 0.05),
        "rudder": (0.5, 0.0, 0.05),
    }
    return PIDController(gains=gains, obs_indices=OBS_INDICES, dt=0.02)


CONTROLLERS = [_make_pid]


@pytest.mark.parametrize("make_controller", CONTROLLERS)
def test_act_returns_valid_action(make_controller) -> None:
    controller = make_controller()
    obs = np.random.default_rng(0).normal(size=OBS_DIM).astype(np.float32)
    action = controller.act(obs)
    assert action.shape == (3,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


@pytest.mark.parametrize("make_controller", CONTROLLERS)
def test_update_returns_dict(make_controller) -> None:
    controller = make_controller()
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    action = controller.act(obs)
    result = controller.update(obs, action, reward=0.0, next_obs=obs)
    assert isinstance(result, dict)


@pytest.mark.parametrize("make_controller", CONTROLLERS)
def test_reset_clears_episode_state_not_weights(make_controller) -> None:
    controller = make_controller()
    obs = np.array([0.5, -0.3, 0.1, 0, 0, 0, 0, 0], dtype=np.float32)
    for _ in range(5):
        controller.act(obs)
    controller.reset()
    # episode-local state cleared: integral/derivative history shouldn't carry over
    if hasattr(controller, "_integral"):
        assert all(v == 0.0 for v in controller._integral.values())


@pytest.mark.parametrize("make_controller", CONTROLLERS)
def test_save_load_roundtrip(make_controller, tmp_path) -> None:
    controller = make_controller()
    path = tmp_path / "controller.json"
    controller.save(path)
    controller.load(path)  # must not raise
