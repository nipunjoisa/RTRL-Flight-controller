"""RTRL-RTU controller: online, exact-RTRL-trained RTU, warm-started from a
BPTT-LSTM checkpoint (rtrl_flight.rtrl.warmstart). See agent_docs/rtrl.md for
the math this implements and src/rtrl_flight/controllers/base.py for the
Controller ABC contract.

READOUT FORMULA NOTE: agent_docs/rtrl.md specifies
`y_t = W_out @ tanh(h_t) + b_out` (tanh applied to h *before* the linear
readout), and rtrl_flight.rtrl.sensitivity.readout_credit() is the tested,
verified gradient for exactly that formula (dtanh = 1 - tanh(h_t)^2 inside
readout_credit is the derivative of tanh(h), not of a post-linear output).
This deliberately differs from a literal "tanh(W_out @ h)" reading (tanh
*after* the linear layer, applied to raw h) -- that alternative formula
would need a different, not-yet-verified gradient (derivative of tanh
applied to the *linear output*, not to h), and would leave
readout_credit()/parameter_gradients() unused despite being exactly what
agent_docs/rtrl.md's math and tests/test_rtrl_sensitivity.py's correctness
bar are built around. To keep the action bounded to [-1, 1] for the env's
action space (rtrl.md's y_t itself is an unconstrained affine map of a
bounded tanh(h) vector, not bounded to [-1, 1] on its own), an *additional*
outer tanh is applied to y_t to produce the actual action:
`action = tanh(y_t) = tanh(W_out @ tanh(h_t) + b_out)`. This keeps
rtrl.md's formula and readout_credit()'s gradient exactly intact -- the
outer squash is one more chain-rule factor applied in update(), not a
change to the readout math itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rtrl_flight.controllers.base import Controller
from rtrl_flight.controllers.bptt_lstm import ACTION_DIM, INPUT_DIM
from rtrl_flight.rtrl.rtu_cell import RTUCell
from rtrl_flight.rtrl.sensitivity import (
    SensitivityState,
    parameter_gradients,
    readout_credit,
    step_sensitivity,
)


class RTRLRTUController(Controller):
    def __init__(
        self,
        hidden_size: int = 64,
        lr: float = 1e-3,
        online_updates: bool = True,
        input_size: int = INPUT_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.action_dim = action_dim
        self.lr = lr
        self.online_updates = online_updates

        self.cell = RTUCell(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.h: torch.Tensor
        self.sensitivity: SensitivityState
        self.reset()

    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self.cell.parameters()) + list(self.output_layer.parameters())

    def act(self, obs: np.ndarray) -> np.ndarray:
        x_t = torch.as_tensor(obs, dtype=torch.float32)

        # Advance S_{t-1} -> S_t using h_prev = self.h (per agent_docs/rtrl.md,
        # order relative to the h update doesn't matter since both consume
        # h_prev -- see step_sensitivity's docstring).
        self.sensitivity = step_sensitivity(self.cell, self.sensitivity, h_prev=self.h, x_t=x_t)

        with torch.no_grad():
            self.h = self.cell.step(self.h, x_t)
            tanh_h = torch.tanh(self.h)
            y_t = self.output_layer(tanh_h)
            action = torch.tanh(y_t)

        return action.numpy().astype(np.float32)

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool = False,
    ) -> dict[str, float]:
        del (
            obs,
            action,
            next_obs,
            done,
        )  # unused: the gradient is a function of self.h/self.sensitivity + reward only
        if not self.online_updates:
            return {}

        loss = -float(reward)

        # No natural per-output-channel decomposition of a scalar env reward
        # exists (there's no differentiable path from the environment's
        # reward back through the action), so dL/dy_t is a uniform broadcast
        # of the scalar loss across all action channels -- each channel is
        # "blamed"/"credited" equally for the single scalar outcome. This is
        # the online objective's approximation (agent_docs/rtrl.md: "online
        # RTRL: whatever the deployed objective is"); the RTRL *sensitivity*
        # math feeding off of it (readout_credit, parameter_gradients) is
        # exact, per agent_docs/rtrl.md.
        dL_dy = torch.full((self.action_dim,), loss, dtype=torch.float32)

        tanh_h = torch.tanh(self.h)
        dL_dh = readout_credit(dL_dy, self.output_layer.weight, self.h)
        recurrent_grads = parameter_gradients(dL_dh, self.sensitivity)

        # W_out/b_out gradients are the ordinary instantaneous backprop
        # gradient (no S term -- not part of the recurrence, per
        # agent_docs/rtrl.md): d(loss)/d(W_out) = dL_dy (outer) tanh_h,
        # d(loss)/d(b_out) = dL_dy.
        dL_dWout = dL_dy.unsqueeze(-1) * tanh_h.unsqueeze(0)
        dL_dbout = dL_dy.clone()

        self.optimizer.zero_grad()
        self.cell.W_in.grad = recurrent_grads["W_in"].clone()
        self.cell.b_in.grad = recurrent_grads["b_in"].clone()
        self.cell.a_raw.grad = recurrent_grads["a_raw"].clone()
        self.output_layer.weight.grad = dL_dWout.clone()
        self.output_layer.bias.grad = dL_dbout.clone()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {"online_loss": loss}

    def reset(self) -> None:
        self.h = self.cell.init_hidden()
        self.sensitivity = SensitivityState.zeros(self.hidden_size, self.input_size)

    def sensitivity_frobenius_norm(self) -> float:
        """sqrt(sum of squares) across all three sensitivity tensors --
        a single scalar for trace logging / smoke-test monitoring."""
        return float(
            torch.cat(
                [
                    self.sensitivity.S_W_in.flatten(),
                    self.sensitivity.S_b_in.flatten(),
                    self.sensitivity.S_a_raw.flatten(),
                ]
            )
            .norm()
            .item()
        )

    def save(self, path: Path) -> None:
        torch.save(
            {
                "cell_state_dict": self.cell.state_dict(),
                "output_layer_state_dict": self.output_layer.state_dict(),
                "h": self.h,
                "S_W_in": self.sensitivity.S_W_in,
                "S_b_in": self.sensitivity.S_b_in,
                "S_a_raw": self.sensitivity.S_a_raw,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "hidden_size": self.hidden_size,
                "input_size": self.input_size,
                "action_dim": self.action_dim,
                "online_updates": self.online_updates,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint: dict[str, Any] = torch.load(path, weights_only=False)
        self.cell.load_state_dict(checkpoint["cell_state_dict"])
        self.output_layer.load_state_dict(checkpoint["output_layer_state_dict"])
        self.h = checkpoint["h"]
        self.sensitivity = SensitivityState(
            S_W_in=checkpoint["S_W_in"],
            S_b_in=checkpoint["S_b_in"],
            S_a_raw=checkpoint["S_a_raw"],
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.online_updates = checkpoint["online_updates"]
