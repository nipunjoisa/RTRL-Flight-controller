"""BPTT-LSTM baseline controller. Frozen-weights-at-deployment baseline for
agent_docs/experiments.md Scenarios 2/3: warm-started offline via imitation
(rtrl_flight.training.imitation), never updated online -- update() is a
no-op, per the Controller ABC (src/rtrl_flight/controllers/base.py) and
agent_docs/architecture.md's "PID and BPTT-LSTM implement update as a no-op".

NOTE: agent_docs/controllers.md does not exist in this repo (checked before
writing this file) -- built directly against architecture.md's Controller
ABC section and base.py instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from rtrl_flight.controllers.base import Controller

INPUT_DIM = 11  # AttitudeHoldTask's obs vector, agent_docs/environment.md
ACTION_DIM = 3  # aileron, elevator, rudder


class BPTTLSTMController(Controller):
    """2-layer LSTM -> Linear(hidden, 3) -> tanh, in [-1, 1].

    Hidden state (h, c) is instance state (`self.h`, `self.c`), persisted
    across act() calls within an episode and zeroed by reset(). act() is the
    single-step, persistent-hidden-state *online interaction* interface
    (numpy in, numpy out, per the Controller ABC); full-sequence BPTT
    training uses `forward_sequence()` instead (see
    rtrl_flight.training.imitation.train_bptt), which runs the whole
    (batch, seq_len, INPUT_DIM) episode through nn.LSTM in one call so
    gradients flow through the entire unrolled sequence.

    act() deliberately never wraps its computation in `torch.no_grad()` --
    only the returned action is detached to cross the numpy boundary the ABC
    requires. This means calling act() many times without ever calling
    `.backward()` (e.g. a long evaluation rollout) keeps extending the same
    autograd graph and uses more memory than necessary; callers doing
    gradient-free rollouts should wrap their own call site in
    `torch.no_grad()` (that's a caller decision, not something act() should
    force on every use, since the training loop needs the graph intact).
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, lr: float = 1e-3) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.lstm = nn.LSTM(INPUT_DIM, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, ACTION_DIM)
        self.h: torch.Tensor
        self.c: torch.Tensor
        self.reset()

    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self.lstm.parameters()) + list(self.output_layer.parameters())

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        h0: torch.Tensor | None = None,
        c0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """obs_seq: (batch, seq_len, INPUT_DIM) -> (batch, seq_len, ACTION_DIM),
        tanh-squashed. Full-sequence forward pass for offline BPTT training --
        not used by act(). Defaults to a zero initial hidden state (each
        sequence is one full episode, so it always starts from zero).
        """
        batch = obs_seq.shape[0]
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, batch, self.hidden_size)
        out, _ = self.lstm(obs_seq, (h0, c0))
        return torch.tanh(self.output_layer(out))

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).reshape(1, 1, INPUT_DIM)
        out, (self.h, self.c) = self.lstm(obs_tensor, (self.h, self.c))
        action = torch.tanh(self.output_layer(out.reshape(-1)))
        return action.detach().numpy().astype(np.float32)

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
    ) -> dict[str, float]:
        # No-op: BPTT-LSTM is frozen at deployment. This is the baseline
        # condition for Scenario 2/3 — the frozen controller cannot adapt
        # to faults or wind mid-episode. See agent_docs/experiments.md.
        return {}

    def reset(self) -> None:
        self.h = torch.zeros(self.num_layers, 1, self.hidden_size)
        self.c = torch.zeros(self.num_layers, 1, self.hidden_size)

    def save(self, path: Path) -> None:
        torch.save(
            {
                "lstm_state_dict": self.lstm.state_dict(),
                "output_layer_state_dict": self.output_layer.state_dict(),
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, weights_only=True)
        self.lstm.load_state_dict(checkpoint["lstm_state_dict"])
        self.output_layer.load_state_dict(checkpoint["output_layer_state_dict"])
