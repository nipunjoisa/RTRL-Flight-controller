"""Actuator degradation wrapper. See agent_docs/architecture.md: faults are
a Gymnasium wrapper, not controller logic, and composable with WindWrapper
in either order (`WindWrapper(FaultWrapper(env))` and the reverse both need
to work) -- this wrapper owns only its own episode-local fault state and
resets it independently of anything WindWrapper does.

Scaling happens on the *commanded* action before it reaches the base env --
`effective_action[i] = action[i] * severity[i]` -- so the fault's effect
shows up in the aircraft's actual response (and therefore in tracking
error/reward), while `info["fault"]` still carries the fault-free vs.
degraded action pair for trace logging (see
agent_docs/architecture.md's `true_action_*` trace column).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

SURFACES = ("aileron", "elevator", "rudder")


class FaultWrapper(gym.Wrapper):
    """severity=1.0 (full authority) per surface until trigger_fault() is
    called; reset() restores full authority for the next episode.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._severity: dict[str, float] = dict.fromkeys(SURFACES, 1.0)

    @property
    def fault_state(self) -> dict[str, float]:
        return dict(self._severity)

    def trigger_fault(self, surface: str, severity: float) -> None:
        """Degrade `surface`'s effective authority to `severity` (1.0 = full
        authority, 0.0 = no authority) starting from the next step. Callable
        mid-episode -- this is how fault onset within an episode window
        (agent_docs/experiments.md, Scenario 2) is triggered.
        """
        if surface not in SURFACES:
            raise ValueError(f"unknown surface {surface!r}, expected one of {SURFACES}")
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {severity}")
        self._severity[surface] = severity

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self._severity = dict.fromkeys(SURFACES, 1.0)
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        severity_vec = np.array([self._severity[s] for s in SURFACES])
        effective_action = np.asarray(action) * severity_vec

        obs, reward, terminated, truncated, info = self.env.step(effective_action)
        info["fault"] = {
            "severity": self.fault_state,
            "commanded_action": np.asarray(action).tolist(),
            "effective_action": effective_action.tolist(),
        }
        return obs, reward, terminated, truncated, info
