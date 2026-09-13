"""Imitation warm-start for BPTT-LSTM and (later) RTRL-RTU: collect PID
rollouts, train a controller to regress the PID's action via BPTT. See
agent_docs/experiments.md's "Imitation warm-start" section -- cold-start
online RTRL diverges (agent_docs/rtrl.md), and BPTT-LSTM needs *some* warm
start too rather than starting from a random policy in the fault/wind
scenarios it's compared in.

PID obs_indices below (pitch_error=9, roll_error=10, yaw_rate=8) match
AttitudeHoldTask's fixed state_variables order documented in
agent_docs/environment.md -- not configurable here, since that ordering is
the task's contract, not a training-time choice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from rtrl_flight.controllers.bptt_lstm import BPTTLSTMController
from rtrl_flight.controllers.pid import PIDController
from rtrl_flight.env.make import cfg_get, force_raw_obs, make_env

# agent_docs/environment.md's obs table: slot 8 = velocities/r-rad_sec
# (yaw rate), slot 9 = error/pitch-error-rad, slot 10 = error/roll-error-rad.
PID_OBS_INDICES = {"pitch_error": 9, "roll_error": 10, "yaw_rate": 8}
# Matches configs/controller/pid.yaml's gains -- retuned via
# scripts/tune_pid.py after finding the original gains caused full
# divergence (a sign bug in PIDController.act(), fixed there; see
# configs/controller/pid.yaml's comment for the full story).
PID_GAINS = {
    "aileron": (0.4, 0.02, 0.15),
    "elevator": (0.4, 0.02, 0.15),
    "rudder": (0.1, 0.0, 0.05),
}
PID_RUDDER_ROLL_COORDINATION_GAIN = 0.1


# Raw-obs crash-detection bounds (see collect_pid_rollouts): AttitudeHoldTask
# has no safety-bound termination yet (agent_docs/environment.md flags this
# as a known gap), and PID_GAINS's placeholder values (never validated
# against the real C172 dynamics) can lose control of the aircraft entirely
# -- observed empirically as a slow dive from the 5000ft start to ground
# impact, with pitch/roll blowing far past the ±0.3/±0.4 rad target range,
# eventually producing float32-overflowing state as JSBSim keeps integrating
# an already-crashed aircraft for the rest of the fixed-length episode.
# These bounds assume RAW (non-normalized) obs -- see rtrl_flight.env.make.force_raw_obs.
CRASH_MIN_ALTITUDE_FT = 100.0
CRASH_MAX_ABS_PITCH_RAD = 1.4  # task only ever targets ±0.3 rad
CRASH_MAX_ABS_ROLL_RAD = 2.5  # task only ever targets ±0.4 rad


def _departed_controlled_flight(obs: np.ndarray) -> bool:
    altitude_ft, pitch_rad, roll_rad = obs[0], obs[1], obs[2]
    return (
        altitude_ft < CRASH_MIN_ALTITUDE_FT
        or abs(pitch_rad) > CRASH_MAX_ABS_PITCH_RAD
        or abs(roll_rad) > CRASH_MAX_ABS_ROLL_RAD
    )


def collect_pid_rollouts(
    env, pid_controller: PIDController, n_episodes: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Roll out pid_controller in env for n_episodes. Returns a list of
    (obs (T, INPUT_DIM), action (T, ACTION_DIM)) float32 arrays, one pair
    per episode -- the imitation targets for train_bptt.

    Stops recording a rollout early (keeping its already-collected, valid
    prefix) if the aircraft departs controlled flight per
    _departed_controlled_flight -- expert demonstrations should show
    attitude-hold behavior, not a post-crash tail the PID never recovers
    from (see the crash-detection bounds above).

    Wrapped in torch.no_grad() per the spec; PID itself has no torch state,
    so this has no practical effect here, but keeps the collection loop
    correct-by-construction if a torch-based expert ever replaces PID.
    """
    rollouts = []
    with torch.no_grad():
        for _ in range(n_episodes):
            pid_controller.reset()
            obs, _info = env.reset()
            obs_list: list[np.ndarray] = []
            action_list: list[np.ndarray] = []
            terminated = truncated = False
            while not (terminated or truncated) and not _departed_controlled_flight(obs):
                action = pid_controller.act(obs)
                obs_list.append(obs)
                action_list.append(action)
                obs, _reward, terminated, truncated, _info = env.step(action)
            if obs_list:
                rollouts.append(
                    (
                        np.asarray(obs_list, dtype=np.float32),
                        np.asarray(action_list, dtype=np.float32),
                    )
                )
    return rollouts


class _RolloutDataset(Dataset):
    def __init__(self, rollouts: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.rollouts = rollouts

    def __len__(self) -> int:
        return len(self.rollouts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        obs, action = self.rollouts[idx]
        return torch.as_tensor(obs), torch.as_tensor(action)


def _collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads variable-length episodes to the batch's max length and returns a
    (batch, seq_len) mask so padded steps don't contribute to the loss.
    """
    obs_seqs, action_seqs = zip(*batch, strict=True)
    lengths = torch.tensor([len(o) for o in obs_seqs])
    obs_padded = pad_sequence(obs_seqs, batch_first=True)
    action_padded = pad_sequence(action_seqs, batch_first=True)
    mask = (torch.arange(obs_padded.shape[1])[None, :] < lengths[:, None]).float()
    return obs_padded, action_padded, mask


def train_bptt(
    lstm_controller: BPTTLSTMController,
    rollouts: list[tuple[np.ndarray, np.ndarray]],
    n_epochs: int,
    lr: float,
    batch_size: int = 8,
) -> list[float]:
    """Trains lstm_controller to regress the PID actions in `rollouts` via
    full backprop-through-time (one forward_sequence() call per batch, whole
    episode at once). Returns the per-epoch mean MSE loss (len == n_epochs).
    """
    dataset = _RolloutDataset(rollouts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.Adam(lstm_controller.parameters(), lr=lr)

    epoch_losses = []
    for _epoch in range(n_epochs):
        batch_losses = []
        for obs_batch, action_batch, mask in loader:
            optimizer.zero_grad()
            pred = lstm_controller.forward_sequence(obs_batch)
            per_step_mse = ((pred - action_batch) ** 2).mean(dim=-1)  # (batch, seq_len)
            loss = (per_step_mse * mask).sum() / mask.sum()
            loss.backward()
            # BPTT over a ~300-step untrained LSTM occasionally produces an
            # exploding gradient (observed empirically: NaN loss in roughly
            # 1 of several dozen random inits, not deterministic) -- standard
            # RNN-training defense, not a sign of a deeper bug in the model.
            torch.nn.utils.clip_grad_norm_(lstm_controller.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        mean_loss = float(np.mean(batch_losses))
        epoch_losses.append(mean_loss)
        print(f"epoch {_epoch + 1}/{n_epochs}  mean_mse_loss={mean_loss:.6f}")
    return epoch_losses


def pretrain(
    env_cfg: Any,
    n_episodes: int,
    n_epochs: int,
    checkpoint_path: str | Path,
    lr: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
    batch_size: int = 8,
) -> float:
    """collect_pid_rollouts + train_bptt, then saves a checkpoint. Returns
    the final epoch's mean MSE loss.
    """
    env = make_env(force_raw_obs(env_cfg))
    try:
        agent_interaction_freq = cfg_get(env_cfg, "agent_interaction_freq", 5)
        pid = PIDController(
            gains=PID_GAINS,
            obs_indices=PID_OBS_INDICES,
            dt=1.0 / agent_interaction_freq,
            rudder_roll_coordination_gain=PID_RUDDER_ROLL_COORDINATION_GAIN,
        )
        rollouts = collect_pid_rollouts(env, pid, n_episodes)
    finally:
        env.close()

    lstm_controller = BPTTLSTMController(hidden_size=hidden_size, num_layers=num_layers, lr=lr)
    losses = train_bptt(lstm_controller, rollouts, n_epochs, lr, batch_size=batch_size)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    lstm_controller.save(checkpoint_path)

    return losses[-1]
