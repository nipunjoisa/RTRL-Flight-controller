"""Forward sensitivity tracking for RTUCell and the readout credit-assignment
step, implementing the recursion in agent_docs/rtrl.md ("Forward sensitivity
recursion" and "Turning sensitivities into a gradient").

Because RTUCell's recurrence is diagonal, each recurrent parameter's
sensitivity is localized to the single hidden unit it feeds — no N x N
tensor, just one scalar per (unit, parameter) pair. This is what keeps exact
RTRL O(N) per parameter instead of O(N^2).

Do not extend this to a dense/full-Jacobian sensitivity without discussion —
that reintroduces the O(N^4) cost RTUs exist to avoid (see CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rtrl_flight.rtrl.rtu_cell import RTUCell


@dataclass
class SensitivityState:
    """S_t^theta for theta in {W_in, b_in, a_raw}. Shapes match the params:
    S_W_in: (N, D), S_b_in: (N,), S_a_raw: (N,).
    """

    S_W_in: torch.Tensor
    S_b_in: torch.Tensor
    S_a_raw: torch.Tensor

    @staticmethod
    def zeros(hidden_size: int, input_size: int) -> SensitivityState:
        return SensitivityState(
            S_W_in=torch.zeros(hidden_size, input_size),
            S_b_in=torch.zeros(hidden_size),
            S_a_raw=torch.zeros(hidden_size),
        )


def step_sensitivity(
    cell: RTUCell, state: SensitivityState, h_prev: torch.Tensor, x_t: torch.Tensor
) -> SensitivityState:
    """Advance S_{t-1} -> S_t given the previous hidden state and current input.

    h_prev must be the hidden state the cell's step() is about to consume
    (i.e. call this before or after cell.step() with the same h_prev/x_t —
    order relative to the h update doesn't matter since both consume h_prev).
    """
    a = cell.gain().detach()  # (N,) — sensitivity bookkeeping is not autograd-tracked
    da_daraw = 1.0 - a**2  # d/d(a_raw) of tanh(a_raw)

    S_W_in = a.unsqueeze(-1) * state.S_W_in + x_t.detach().unsqueeze(0)
    S_b_in = a * state.S_b_in + 1.0
    S_a_raw = a * state.S_a_raw + da_daraw * h_prev.detach()

    return SensitivityState(S_W_in=S_W_in, S_b_in=S_b_in, S_a_raw=S_a_raw)


def readout_credit(dL_dy: torch.Tensor, W_out: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
    """dL_t/dh_t[i] = sum_o dL_dy[o] * W_out[o,i] * (1 - tanh(h_t[i])^2).

    Instantaneous (non-recurrent) credit assignment through the readout at
    time t only — see agent_docs/rtrl.md.
    """
    dtanh = 1.0 - torch.tanh(h_t) ** 2
    return (dL_dy @ W_out) * dtanh


def parameter_gradients(dL_dh: torch.Tensor, state: SensitivityState) -> dict[str, torch.Tensor]:
    """dL_t/dtheta = dL_dh[i] * S_t^theta[i] for each recurrent parameter."""
    return {
        "W_in": dL_dh.unsqueeze(-1) * state.S_W_in,
        "b_in": dL_dh * state.S_b_in,
        "a_raw": dL_dh * state.S_a_raw,
    }
