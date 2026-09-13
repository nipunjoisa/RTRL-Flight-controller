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

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jsbgym.tasks import FlightTask
from torch import nn

from rtrl_flight.controllers.base import Controller
from rtrl_flight.controllers.bptt_lstm import ACTION_DIM, INPUT_DIM
from rtrl_flight.env.attitude_task import AttitudeHoldTask
from rtrl_flight.rtrl.rtu_cell import RTUCell
from rtrl_flight.rtrl.sensitivity import (
    SensitivityState,
    parameter_gradients,
    readout_credit,
    step_sensitivity,
)

# Fixed, known obs bounds (same source NormalizeWrapper uses -- see
# agent_docs/environment.md's obs table) for RTRLRTUController's OWN
# internal input normalization. Found while retraining for PART C: the
# warm-started (pre-any-online-update) controller already saturated at a
# constant [1,1,1] action -- h's norm reached ~28000 after 30 steps on RAW
# obs (needed by PID/BPTT-LSTM, forced via rtrl_flight.env.make.
# force_raw_obs), because RTUCell's recurrence z_t = W_in@x_t + b_in is
# DELIBERATELY linear and unbounded (agent_docs/rtrl.md -- nonlinearity is
# kept out of the recurrence to preserve exact RTRL). altitude~5000ft times
# even a modest W_in column norm (~0.56, confirmed empirically) alone
# produces ||z_t|| ~2800, and with the recurrence gain a=0.9 that geometric-
# series' steady state is ~10x that -- unlike an LSTM's sigmoid/tanh gates,
# nothing here bounds it regardless of input scale. This is independent of
# (and predates) PART B's online-update-objective fixes: it happens with
# zero online update steps taken. Normalizing this controller's OWN input
# to [-1, 1] (see act() below) fixes the root cause without touching
# PID/BPTT-LSTM/NormalizeWrapper/metrics, all of which correctly need raw
# obs for other reasons.
_OBS_PROPERTIES = FlightTask.base_state_variables + (
    AttitudeHoldTask.pitch_error_rad,
    AttitudeHoldTask.roll_error_rad,
)
OBS_LOW = np.array([p.min for p in _OBS_PROPERTIES], dtype=np.float32)
OBS_HIGH = np.array([p.max for p in _OBS_PROPERTIES], dtype=np.float32)
OBS_RANGE = OBS_HIGH - OBS_LOW

# PART B fixes (see the "RTRL online update objective" retuning session):
# raw -reward as loss was too noisy/unscaled (rewards ~-300/episode) and
# drove the online update to saturate the policy at constant [1,1,1] action
# over 20 episodes. Normalizing by a running window, a lower online lr, and
# tighter grad clipping are all defenses against that same failure mode, not
# independent tweaks.
REWARD_WINDOW = 100
ONLINE_GRAD_CLIP_NORM = 0.5
SENSITIVITY_NORM_LIMIT = 1e4

# Found while retraining for PART C, beyond PART B's explicit 4 fixes: even
# with all of them applied, 50 episodes of online updates still drifted `a`
# (recurrence gain) from a uniform 0.9 to ~0.94 and grew W_in/b_in enough
# that a FRESH episode's h (starting at 0, per reset()) reached norm ~90
# within 60 steps -- fully saturating tanh(h) and producing a constant
# action regardless of input. Nothing in RTUCell bounds h directly; only
# `a < 1`'s geometric decay does, and that stopped being enough once `a`
# and W_in/b_in drifted during training. This clamp is the direct analog of
# the sensitivity-norm sanity check PART B asked for (guard against runaway
# growth) applied to h instead, since h is the more immediate cause of
# saturation. tanh saturates by |x| ~ 4, so 5.0 leaves headroom for genuine
# (non-runaway) variation while preventing the observed unbounded growth.
HIDDEN_STATE_CLIP = 5.0

# REMOVED (see the per-channel credit-assignment fix in update() below):
# OUTPUT_WEIGHT_DECAY and MAX_OUTPUT_NORM were band-aids over the real bug
# -- a single scalar loss broadcast identically to all 3 output channels
# every step, which is systematically correlated in direction regardless of
# its (correctly bounded, per PART B) magnitude. Weight decay/norm-capping
# could only ever bound the symptom (W_out's magnitude), never stop the
# drift itself, and a hard-enough cap just traded "saturated from growth"
# for "saturated at the cap." Per-channel credit (each output gets its own
# target and its own gradient sign/magnitude, not a shared broadcast value)
# removes the systematic correlation at the source, so neither band-aid is
# needed once it's in place.


class RTRLRTUController(Controller):
    def __init__(
        self,
        hidden_size: int = 64,
        lr: float = 1e-3,
        online_lr: float = 1e-4,
        online_updates: bool = True,
        input_size: int = INPUT_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.action_dim = action_dim
        self.lr = lr  # kept for reference/back-compat; not used to build the optimizer below
        self.online_lr = online_lr
        self.online_updates = online_updates

        self.cell = RTUCell(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, action_dim)
        # online_lr (1e-4 default), not lr (1e-3) -- the online step-by-step
        # update needs a much smaller step size than an offline warm-start
        # optimizer would, per PART B. No weight_decay on either param group
        # (see the "REMOVED" note above) -- per-channel credit assignment
        # removes the systematic drift that decay used to be a band-aid for.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=online_lr, weight_decay=0.0)

        self.h: torch.Tensor
        self.sensitivity: SensitivityState
        self._reward_history: deque[float]
        self._last_logits: torch.Tensor
        self.reset()

    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self.cell.parameters()) + list(self.output_layer.parameters())

    def act(self, obs: np.ndarray) -> np.ndarray:
        # Normalize to [-1, 1] using the same fixed bounds NormalizeWrapper
        # would (see OBS_LOW/OBS_HIGH's module-level docstring) -- this
        # controller's own defense against RTUCell's unbounded linear
        # recurrence blowing up on raw-scale obs (e.g. altitude ~5000ft),
        # independent of whatever the caller's env normalization is set to.
        normalized_obs = np.clip(2.0 * (obs - OBS_LOW) / OBS_RANGE - 1.0, -1.0, 1.0)
        x_t = torch.as_tensor(normalized_obs, dtype=torch.float32)

        # Advance S_{t-1} -> S_t using h_prev = self.h (per agent_docs/rtrl.md,
        # order relative to the h update doesn't matter since both consume
        # h_prev -- see step_sensitivity's docstring).
        self.sensitivity = step_sensitivity(self.cell, self.sensitivity, h_prev=self.h, x_t=x_t)

        with torch.no_grad():
            self.h = torch.clamp(self.cell.step(self.h, x_t), -HIDDEN_STATE_CLIP, HIDDEN_STATE_CLIP)
            tanh_h = torch.tanh(self.h)
            y_t = self.output_layer(tanh_h)
            action = torch.tanh(y_t)

        # Pre-tanh logits, stored (not recomputed) for update()'s per-channel
        # loss -- see PART A of the per-channel-credit retuning session.
        self._last_logits = y_t

        return action.numpy().astype(np.float32)

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool = False,
    ) -> dict[str, float]:
        del obs, action, done  # unused: next_obs IS used now (PART A per-channel targets)
        if not self.online_updates:
            return {}

        # Guard against a non-finite reward or an already-NaN h (found while
        # retraining for PART C: AttitudeHoldTask has no safety-bound
        # termination yet -- a known, pre-existing gap, see
        # agent_docs/environment.md -- so a mid-episode flight-dynamics
        # divergence can produce reward=nan for the rest of that episode).
        # Skipping the update this step is not a fix for the underlying env
        # gap; it only prevents one crashed episode from permanently
        # poisoning self.parameters() with NaN via the Adam step (h/
        # sensitivity are wiped by the next reset() regardless, but
        # parameters persist across the whole training run).
        if not np.isfinite(reward) or torch.isnan(self.h).any():
            return {}

        # Sanity check (PART B #4): the sensitivity recursion's steady-state
        # magnitude grows as the recurrence gain `a` -> 1 (observed
        # empirically: 166 -> 577 over 20 episodes in the pre-fix online
        # run). Left unchecked this eventually explodes; reset to zeros
        # rather than let a single pathological episode poison every
        # subsequent gradient with an astronomically large S.
        pre_update_sensitivity_norm = self.sensitivity_frobenius_norm()
        if pre_update_sensitivity_norm > SENSITIVITY_NORM_LIMIT:
            print(
                f"WARNING: RTRLRTUController sensitivity norm "
                f"{pre_update_sensitivity_norm:.2f} exceeded "
                f"{SENSITIVITY_NORM_LIMIT} -- resetting sensitivity to zeros"
            )
            self.sensitivity = SensitivityState.zeros(self.hidden_size, self.input_size)

        # Reward normalization (PART B #1, kept): still used, but now only as
        # a *multiplicative weight* on the per-channel loss below (upweight
        # when reward is bad, downweight when good) -- not as the loss
        # itself. Raw -reward as the loss (unscaled, ~-300/episode) was too
        # noisy; that problem doesn't go away just because the loss is now
        # per-channel.
        self._reward_history.append(float(reward))
        running_mean = float(np.mean(self._reward_history))
        running_std = float(np.std(self._reward_history))
        normalized_reward = (float(reward) - running_mean) / (running_std + 1e-8)
        loss_scale = 1.0 + max(0.0, -normalized_reward)

        # Per-channel tracking loss (PART A fix -- the actual root-cause
        # fix): each output channel gets its OWN target and therefore its
        # OWN gradient sign/magnitude, instead of one scalar loss broadcast
        # identically to all 3 channels every step. That broadcast was what
        # made the credit signal systematically correlated in direction
        # across ~15000 steps regardless of its magnitude -- no amount of
        # clipping/decay/capping on the *symptom* (parameter magnitude)
        # could fix a *directional* bias at the source. Targets: aileron
        # drives roll, elevator drives pitch (see agent_docs/environment.md
        # for the control mapping), rudder has no direct tracking target
        # here (its role is yaw-rate damping, not error-tracking).
        pitch_error = float(next_obs[9])  # error/pitch-error-rad
        roll_error = float(next_obs[10])  # error/roll-error-rad
        targets = torch.tensor([-roll_error, -pitch_error, 0.0], dtype=torch.float32)

        y_t = self._last_logits  # pre-tanh logits from the most recent act()
        channel_losses = (y_t - targets) ** 2
        loss = channel_losses.mean().item() * loss_scale
        # d(mean_i (y_i - t_i)^2)/dy_i = (2/action_dim)*(y_i - t_i); loss_scale
        # is a plain multiplicative constant w.r.t. y_t, so it carries
        # through unchanged by the product rule.
        dL_dy = (2.0 / self.action_dim) * (y_t - targets) * loss_scale

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

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=ONLINE_GRAD_CLIP_NORM)
        self.optimizer.step()

        return {"online_loss": loss}

    def reset(self) -> None:
        self.h = self.cell.init_hidden()
        self.sensitivity = SensitivityState.zeros(self.hidden_size, self.input_size)
        self._reward_history = deque(maxlen=REWARD_WINDOW)
        self._last_logits = torch.zeros(self.action_dim)

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
                "reward_history": list(self._reward_history),
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
        self._reward_history = deque(checkpoint.get("reward_history", []), maxlen=REWARD_WINDOW)
