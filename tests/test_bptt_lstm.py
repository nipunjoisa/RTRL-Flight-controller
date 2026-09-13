"""BPTT-LSTM controller tests -- same pattern as
tests/test_controller_interface.py (parametric over controller factories),
plus BPTT-LSTM-specific hidden-state and save/load checks.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rtrl_flight.controllers.bptt_lstm import ACTION_DIM, INPUT_DIM, BPTTLSTMController

OBS_DIM = INPUT_DIM


def _make_bptt_lstm() -> BPTTLSTMController:
    return BPTTLSTMController(hidden_size=8, num_layers=2, lr=1e-3)


CONTROLLERS = [_make_bptt_lstm]


# --- Controller ABC contract (same pattern as test_controller_interface.py) --


@pytest.mark.parametrize("make_controller", CONTROLLERS)
def test_act_returns_valid_action(make_controller) -> None:
    controller = make_controller()
    obs = np.random.default_rng(0).normal(size=OBS_DIM).astype(np.float32)
    action = controller.act(obs)
    assert action.shape == (ACTION_DIM,)
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
    weights_before = controller.output_layer.weight.detach().clone()

    obs = np.random.default_rng(1).normal(size=OBS_DIM).astype(np.float32)
    for _ in range(5):
        controller.act(obs)
    controller.reset()

    assert torch.equal(controller.output_layer.weight.detach(), weights_before)


@pytest.mark.parametrize("make_controller", CONTROLLERS)
def test_save_load_roundtrip(make_controller, tmp_path) -> None:
    controller = make_controller()
    path = tmp_path / "controller.pt"
    controller.save(path)
    controller.load(path)  # must not raise


# --- BPTT-LSTM-specific behavior --------------------------------------------


def test_act_output_shape_and_range() -> None:
    controller = _make_bptt_lstm()
    obs = np.random.default_rng(2).normal(size=OBS_DIM).astype(np.float32)
    action = controller.act(obs)
    assert action.shape == (3,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_reset_zeros_hidden_state() -> None:
    controller = _make_bptt_lstm()
    obs = np.random.default_rng(3).normal(size=OBS_DIM).astype(np.float32)
    controller.act(obs)  # populate non-zero hidden state
    controller.reset()

    assert controller.h.norm().item() == 0.0
    assert controller.c.norm().item() == 0.0


def test_hidden_state_persists_across_act_calls() -> None:
    controller = _make_bptt_lstm()
    assert controller.h.norm().item() == 0.0  # zero at construction (reset() in __init__)

    obs = np.random.default_rng(4).normal(size=OBS_DIM).astype(np.float32)
    controller.act(obs)

    assert controller.h.norm().item() > 0.0
    assert controller.c.norm().item() > 0.0


def test_save_load_produces_same_output_for_same_obs(tmp_path) -> None:
    controller = _make_bptt_lstm()
    obs = np.random.default_rng(5).normal(size=OBS_DIM).astype(np.float32)

    path = tmp_path / "bptt_lstm.pt"
    controller.save(path)

    fresh = BPTTLSTMController(hidden_size=controller.hidden_size, num_layers=controller.num_layers)
    fresh.load(path)

    action_original = controller.act(obs)
    action_loaded = fresh.act(obs)

    np.testing.assert_allclose(action_original, action_loaded, atol=1e-6)


def test_update_is_a_noop() -> None:
    controller = _make_bptt_lstm()
    obs = np.random.default_rng(6).normal(size=OBS_DIM).astype(np.float32)
    action = controller.act(obs)

    h_before = controller.h.detach().clone()
    c_before = controller.c.detach().clone()
    weights_before = {name: p.detach().clone() for name, p in controller.lstm.named_parameters()}
    output_weights_before = controller.output_layer.weight.detach().clone()

    result = controller.update(obs, action, reward=1.0, next_obs=obs)

    assert result == {}
    assert torch.equal(controller.h, h_before)
    assert torch.equal(controller.c, c_before)
    for name, p in controller.lstm.named_parameters():
        assert torch.equal(p.detach(), weights_before[name])
    assert torch.equal(controller.output_layer.weight.detach(), output_weights_before)
