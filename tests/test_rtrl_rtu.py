"""RTRL-RTU controller tests -- same ABC-contract pattern as
tests/test_controller_interface.py and tests/test_bptt_lstm.py, plus
RTRL-RTU-specific hidden-state/sensitivity and online-update checks.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rtrl_flight.controllers.bptt_lstm import ACTION_DIM, INPUT_DIM
from rtrl_flight.controllers.rtrl_rtu import RTRLRTUController

OBS_DIM = INPUT_DIM


def _make_rtrl_rtu(online_updates: bool = True) -> RTRLRTUController:
    return RTRLRTUController(hidden_size=8, lr=1e-2, online_updates=online_updates)


CONTROLLERS = [_make_rtrl_rtu]


# --- Controller ABC contract (same pattern as the other controllers' tests) -


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
def test_save_load_roundtrip(make_controller, tmp_path) -> None:
    controller = make_controller()
    path = tmp_path / "controller.pt"
    controller.save(path)
    controller.load(path)  # must not raise


# --- RTRL-RTU-specific behavior ----------------------------------------


def test_reset_zeros_h_and_sensitivity() -> None:
    controller = _make_rtrl_rtu()
    obs = np.random.default_rng(1).normal(size=OBS_DIM).astype(np.float32)
    controller.act(obs)  # populate non-zero h and sensitivity
    controller.reset()

    assert controller.h.norm().item() == 0.0
    assert controller.sensitivity_frobenius_norm() == 0.0


def test_h_and_sensitivity_update_after_act() -> None:
    controller = _make_rtrl_rtu()
    assert controller.h.norm().item() == 0.0  # zero at construction (reset() in __init__)
    assert controller.sensitivity_frobenius_norm() == 0.0

    obs = np.random.default_rng(2).normal(size=OBS_DIM).astype(np.float32)
    controller.act(obs)

    assert controller.h.norm().item() > 0.0
    assert controller.sensitivity_frobenius_norm() > 0.0


def test_save_load_preserves_h_and_sensitivity(tmp_path) -> None:
    controller = _make_rtrl_rtu()
    obs = np.random.default_rng(3).normal(size=OBS_DIM).astype(np.float32)
    controller.act(obs)
    controller.act(obs)  # a couple of steps so h/S are non-trivial

    h_before = controller.h.clone()
    s_norm_before = controller.sensitivity_frobenius_norm()

    path = tmp_path / "rtrl_rtu.pt"
    controller.save(path)

    fresh = RTRLRTUController(hidden_size=controller.hidden_size, lr=controller.lr)
    fresh.load(path)

    assert torch.equal(fresh.h, h_before)
    assert fresh.sensitivity_frobenius_norm() == pytest.approx(s_norm_before)


def test_online_updates_false_is_a_full_noop() -> None:
    controller = _make_rtrl_rtu(online_updates=False)
    obs = np.random.default_rng(4).normal(size=OBS_DIM).astype(np.float32)
    action = controller.act(obs)

    h_before = controller.h.clone()
    s_norm_before = controller.sensitivity_frobenius_norm()
    params_before = [p.detach().clone() for p in controller.parameters()]

    result = controller.update(obs, action, reward=1.0, next_obs=obs)

    assert result == {}
    assert torch.equal(controller.h, h_before)
    assert controller.sensitivity_frobenius_norm() == s_norm_before
    for p_before, p_after in zip(params_before, controller.parameters(), strict=True):
        assert torch.equal(p_before, p_after.detach())


def test_online_updates_true_changes_output_layer_via_per_channel_credit() -> None:
    """Verifies PART A's per-channel credit assignment (rtrl_rtu.py's
    update()): with a non-trivial obs carrying real pitch/roll errors in
    slots 9/10, output_layer.weight/bias must actually change -- their
    gradient is now dL_dy = (2/action_dim)*(y_t - targets)*loss_scale,
    a genuine function of those errors, not (as the old, now-removed
    uniform-broadcast loss + OUTPUT_WEIGHT_DECAY combination used to
    produce) a side effect of weight decay pulling params toward zero
    regardless of whether the actual RTRL gradient was zero.
    """
    controller = _make_rtrl_rtu(online_updates=True)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[9] = 0.2  # error/pitch-error-rad
    obs[10] = -0.3  # error/roll-error-rad
    action = controller.act(obs)

    output_weight_before = controller.output_layer.weight.detach().clone()
    output_bias_before = controller.output_layer.bias.detach().clone()
    params_before = [p.detach().clone() for p in controller.parameters()]

    result = controller.update(obs, action, reward=1.0, next_obs=obs)

    assert "online_loss" in result
    assert not torch.equal(controller.output_layer.weight.detach(), output_weight_before), (
        "output_layer.weight did not change -- per-channel credit assignment should give it "
        "a real, non-zero gradient from the pitch/roll errors in next_obs[9]/[10]"
    )
    assert not torch.equal(controller.output_layer.bias.detach(), output_bias_before)

    changed = any(
        not torch.equal(p_before, p_after.detach())
        for p_before, p_after in zip(params_before, controller.parameters(), strict=True)
    )
    assert changed, "expected at least one parameter to change after an online update() step"
