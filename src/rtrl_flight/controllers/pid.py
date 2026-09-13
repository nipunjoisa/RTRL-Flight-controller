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
        rudder_roll_coordination_gain: float = 0.0,
    ) -> None:
        """gains: {"aileron": (kp, ki, kd), "elevator": (...), "rudder": (...)}.
        obs_indices: maps each error signal ("roll_error", "pitch_error",
        "yaw_rate") to its index in the flattened observation vector.
        rudder_roll_coordination_gain: small proportional term added to the
        rudder's yaw-rate-damping output, proportional to roll_error --
        adverse yaw couples roll and yaw, so a bank commanded via aileron
        benefits from a coordinating rudder nudge in the same direction
        (scripts/tune_pid.py PART A: "key facts about the real C172
        dynamics"). Defaults to 0.0 (no coordination term) for backward
        compatibility with any caller not passing it explicitly.
        """
        self.gains = gains
        self.obs_indices = obs_indices
        self.dt = dt
        self.rudder_roll_coordination_gain = rudder_roll_coordination_gain
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
        # NOTE: positive elevator-cmd-norm pitches the nose DOWN in this
        # JSBSim C172 config (confirmed empirically: a constant +elevator
        # command measurably decreased pitch -- see scripts/tune_pid.py's
        # open-loop probe in the PART A session). pitch_error is
        # target_pitch - pitch, so a positive error (need MORE pitch, nose
        # up) must produce a NEGATIVE elevator command. The un-negated
        # version was a positive-feedback sign bug, not just a magnitude
        # problem -- no gain retuning could have fixed it (higher gain made
        # divergence worse, exactly as observed before this fix).
        elevator = self._pid("elevator", -pitch_error)
        # yaw-rate damping (no yaw setpoint) + a small roll-coordination
        # term: adverse yaw couples roll and yaw, so banking via aileron
        # benefits from a rudder nudge in the same direction as roll_error.
        rudder = self._pid("rudder", -yaw_rate) + self.rudder_roll_coordination_gain * roll_error

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
