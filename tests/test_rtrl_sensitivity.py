"""The most important test in the repo (see CLAUDE.md, "Critical correctness
bar"). Verifies the online, causal, no-unroll forward-sensitivity recursion
in rtrl_flight.rtrl.sensitivity produces gradients identical to full-unroll
autograd on a toy problem, to ~1e-5. Also cross-checks against finite
differences directly on the loss. If this test doesn't pass, we are not
doing RTRL — we are doing something that looks like it.
"""

from __future__ import annotations

import torch

from rtrl_flight.rtrl.rtu_cell import RTUCell
from rtrl_flight.rtrl.sensitivity import (
    SensitivityState,
    parameter_gradients,
    readout_credit,
    step_sensitivity,
)

torch.manual_seed(0)

INPUT_SIZE = 3
HIDDEN_SIZE = 5
OUTPUT_SIZE = 2
T = 6
TOL = 1e-5


def _make_cell_and_readout() -> tuple[RTUCell, torch.nn.Linear]:
    cell = RTUCell(INPUT_SIZE, HIDDEN_SIZE)
    readout = torch.nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE, bias=False)
    return cell, readout


def _toy_inputs_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.randn(T, INPUT_SIZE)
    targets = torch.randn(T, OUTPUT_SIZE)
    return xs, targets


def _rtrl_online_gradients(
    cell: RTUCell, readout: torch.nn.Linear, xs: torch.Tensor, targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Run the online, step-by-step RTRL recursion for T steps and return the
    accumulated (summed over t) gradient for each recurrent parameter — no
    unroll, no autograd through time, exactly what Controller.update() does
    per agent_docs/rtrl.md.
    """
    state = SensitivityState.zeros(HIDDEN_SIZE, INPUT_SIZE)
    h = cell.init_hidden()

    total_grads = {
        "W_in": torch.zeros_like(cell.W_in),
        "b_in": torch.zeros_like(cell.b_in),
        "a_raw": torch.zeros_like(cell.a_raw),
    }

    for t in range(T):
        x_t = xs[t]
        state = step_sensitivity(cell, state, h_prev=h, x_t=x_t)
        h = cell.step(h, x_t)

        with torch.no_grad():
            y_t = readout(torch.tanh(h))
            dL_dy = y_t - targets[t]  # d/dy of 0.5*||y - target||^2
            dL_dh = readout_credit(dL_dy, readout.weight, h)
            grads_t = parameter_gradients(dL_dh, state)

        for k in total_grads:
            total_grads[k] += grads_t[k]

    return total_grads


def _bptt_reference_gradients(
    cell: RTUCell, readout: torch.nn.Linear, xs: torch.Tensor, targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Full unroll + autograd over the same T steps — the ground truth these
    T-step traces should match exactly (T-step RTRL == T-step truncated BPTT
    when the truncation length equals the episode length).
    """
    for p in cell.parameters():
        p.grad = None
    h = cell.init_hidden()
    loss = torch.zeros(())
    for t in range(T):
        h = cell.step(h, xs[t])
        y_t = readout(torch.tanh(h))
        loss = loss + 0.5 * ((y_t - targets[t]) ** 2).sum()
    loss.backward()
    return {
        "W_in": cell.W_in.grad.clone(),
        "b_in": cell.b_in.grad.clone(),
        "a_raw": cell.a_raw.grad.clone(),
    }


def test_online_rtrl_matches_full_unroll_autograd() -> None:
    cell, readout = _make_cell_and_readout()
    xs, targets = _toy_inputs_and_targets()

    online_grads = _rtrl_online_gradients(cell, readout, xs, targets)
    reference_grads = _bptt_reference_gradients(cell, readout, xs, targets)

    for name in ("W_in", "b_in", "a_raw"):
        diff = (online_grads[name] - reference_grads[name]).abs().max().item()
        assert diff < TOL, f"{name} gradient mismatch: max abs diff {diff} >= {TOL}"


def test_online_rtrl_matches_finite_differences() -> None:
    """Autograd-independent cross-check: perturb one scalar parameter at a
    time, rerun the T-step rollout, compare central finite differences
    against the analytic online-RTRL gradient.
    """
    cell, readout = _make_cell_and_readout()
    xs, targets = _toy_inputs_and_targets()
    online_grads = _rtrl_online_gradients(cell, readout, xs, targets)

    eps = 1e-4
    fd_tol = 5e-3  # finite differences are noisier than autograd comparison

    def loss_for(cell_: RTUCell) -> float:
        h = cell_.init_hidden()
        loss = torch.zeros(())
        for t in range(T):
            h = cell_.step(h, xs[t])
            y_t = readout(torch.tanh(h))
            loss = loss + 0.5 * ((y_t - targets[t]) ** 2).sum()
        return loss.item()

    import copy

    # Spot-check a handful of scalar entries per parameter tensor.
    checks = [
        ("a_raw", (0,)),
        ("a_raw", (2,)),
        ("b_in", (1,)),
        ("W_in", (0, 0)),
        ("W_in", (3, 2)),
    ]
    for name, idx in checks:
        cell_plus = copy.deepcopy(cell)
        cell_minus = copy.deepcopy(cell)
        with torch.no_grad():
            getattr(cell_plus, name)[idx] += eps
            getattr(cell_minus, name)[idx] -= eps
        fd_grad = (loss_for(cell_plus) - loss_for(cell_minus)) / (2 * eps)
        analytic_grad = online_grads[name][idx].item()
        assert abs(fd_grad - analytic_grad) < fd_tol, (
            f"{name}{idx} finite-diff mismatch: fd={fd_grad}, analytic={analytic_grad}"
        )
