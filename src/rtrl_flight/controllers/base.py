"""Controller ABC. See agent_docs/architecture.md for the contract and why
the runner must never `isinstance`-check on controller type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Controller(ABC):
    """All controllers (PID, BPTT-LSTM, RTRL-RTU) implement this exactly."""

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs -> action (aileron, elevator, rudder), each in [-1, 1]."""

    @abstractmethod
    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
    ) -> dict[str, float]:
        """Online weight update. No-op for PID/BPTT-LSTM (return {}).

        Returns scalars (e.g. online loss, sensitivity norm) for trace logging.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear episode-local state (hidden state, sensitivity tensors).

        Must NOT reset learned weights.
        """

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...
