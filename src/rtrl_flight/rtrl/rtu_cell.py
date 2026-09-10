"""Recurrent Trace Unit (RTU) cell: diagonal linear recurrence + nonlinear
readout, chosen specifically because the diagonal recurrence keeps the
forward-sensitivity Jacobian diagonal and RTRL tractable. See
agent_docs/rtrl.md for the derivation this module implements verbatim.

Recurrence is deliberately linear (no nonlinearity on `h` itself) — pushing
the nonlinearity into the readout only is what keeps d(h_t)/d(h_{t-1})
diagonal. Do not add an activation inside `step()`.
"""

from __future__ import annotations

import torch
from torch import nn


class RTUCell(nn.Module):
    """h_t[i] = a[i] * h_{t-1}[i] + (W_in @ x_t + b_in)[i], a = tanh(a_raw).

    Readout (y_t = W_out @ tanh(h_t) + b_out) is a plain nn.Linear applied by
    the caller to `tanh(h_t)` — kept out of this module since it carries no
    recurrent dependency and needs no sensitivity tracking (see
    agent_docs/rtrl.md, "The RTU cell").
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_in = nn.Parameter(torch.empty(hidden_size, input_size))
        self.b_in = nn.Parameter(torch.zeros(hidden_size))
        self.a_raw = nn.Parameter(torch.zeros(hidden_size))
        nn.init.xavier_uniform_(self.W_in)

    def gain(self) -> torch.Tensor:
        """a = tanh(a_raw), elementwise, shape (hidden_size,). |a| < 1 always."""
        return torch.tanh(self.a_raw)

    def step(self, h_prev: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """One recurrence step. h_prev: (..., N), x_t: (..., D) -> h_t: (..., N)."""
        z_t = x_t @ self.W_in.T + self.b_in
        return self.gain() * h_prev + z_t

    def init_hidden(self, batch_shape: tuple[int, ...] = ()) -> torch.Tensor:
        return torch.zeros(*batch_shape, self.hidden_size)
