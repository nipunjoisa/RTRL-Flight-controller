"""PID baseline controller. Not learned — update() is a no-op, per
agent_docs/architecture.md ("update is called every step for every
controller ... PID and BPTT-LSTM implement it as a no-op")."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rtrl_flight.controllers.base import Controller


class PIDController(Controller):
    """Independent PID loop per surface, driven by (target - measured) error
    read out of fixed observation indices. Indices are wired up once
    jsbsim_env.py pins the observation layout (agent_docs/environment.md).
    """

    def __init__(
        self,
        gains: dict[str, tuple[float, float, float]],
        obs_indices: dict[str, int],
        dt: float,
    ) -> None:
        """gains: {"aileron": (kp, ki, kd), "elevator": (...), "rudder": (...)}.
        obs_indices: maps each error signal ("roll_error", "pitch_error",
        "yaw_rate") to its index in the flattened observation vector.
        """
        self.gains = gains
        self.obs_indices = obs_indices
        self.dt = dt
        self._integral = dict.fromkeys(gains, 0.0)
        self._prev_error = dict.fromkeys(gains, 0.0)

    def _pid(self, surface: str, error: float) -> float:
        kp, ki, kd = self.gains[surface]
        self._integral[surface] += error * self.dt
        derivative = (error - self._prev_error[surface]) / self.dt
        self._prev_error[surface] = error
        return kp * error + ki * self._integral[surface] + kd * derivative

    def act(self, obs: np.ndarray) -> np.ndarray:
        roll_error = obs[self.obs_indices["roll_error"]]
        pitch_error = obs[self.obs_indices["pitch_error"]]
        yaw_rate = obs[self.obs_indices["yaw_rate"]]

        aileron = self._pid("aileron", roll_error)
        elevator = self._pid("elevator", pitch_error)
        rudder = self._pid("rudder", -yaw_rate)  # yaw-rate damping, no yaw setpoint

        return np.clip([aileron, elevator, rudder], -1.0, 1.0).astype(np.float32)

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
    ) -> dict[str, float]:
        return {}

    def reset(self) -> None:
        self._integral = dict.fromkeys(self.gains, 0.0)
        self._prev_error = dict.fromkeys(self.gains, 0.0)

    def save(self, path: Path) -> None:
        import json

        path.write_text(json.dumps({"gains": self.gains}))

    def load(self, path: Path) -> None:
        import json

        self.gains = json.loads(path.read_text())["gains"]
