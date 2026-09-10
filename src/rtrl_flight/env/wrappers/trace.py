"""TraceLogger wrapper -- one parquet file per episode, buffered in memory
and flushed on episode end. See agent_docs/architecture.md's trace schema
and design invariant #4 ("analysis code only ever reads parquet, never live
state") -- this wrapper is the only place per-step data crosses from live
env/wrapper state into the durable, replayable trace.

Kept as the outermost wrapper in the composition order
(agent_docs/architecture.md: AttitudeHoldTask -> NoFGJsbSimEnv ->
NormalizeWrapper -> FaultWrapper -> WindWrapper -> TraceWrapper) so it can
read FaultWrapper's/WindWrapper's per-step info dict entries regardless of
their relative order to each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

ACTION_NAMES = ("action_aileron", "action_elevator", "action_rudder")
SURFACES = ("aileron", "elevator", "rudder")


class TraceWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        output_dir: str | Path = "data/traces",
        episode_id_start: int = 0,
    ) -> None:
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self._episode = episode_id_start - 1
        self._step = 0
        self._buffer: list[dict[str, Any]] = []
        self._obs_names = self._resolve_obs_names()

    def _resolve_obs_names(self) -> list[str]:
        task = getattr(self.env.unwrapped, "task", None)
        if task is not None and hasattr(task, "state_variables"):
            return [f"obs_{prop.get_legal_name()}" for prop in task.state_variables]
        # Fallback for a non-jsbgym base env: numbered columns.
        dim = self.env.observation_space.shape[0]
        return [f"obs_{i}" for i in range(dim)]

    @property
    def last_output_path(self) -> Path | None:
        return getattr(self, "_last_output_path", None)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self._flush()
        self._episode += 1
        self._step = 0
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(self._make_row(obs, action, reward, info))
        self._step += 1
        if terminated or truncated:
            self._flush()
        return obs, reward, terminated, truncated, info

    def _make_row(
        self, obs: np.ndarray, action: np.ndarray, reward: float, info: dict[str, Any]
    ) -> dict[str, Any]:
        row: dict[str, Any] = {"step": self._step, "episode": self._episode}
        for name, value in zip(self._obs_names, obs, strict=True):
            row[name] = float(value)
        for name, value in zip(ACTION_NAMES, action, strict=True):
            row[name] = float(value)
        row["reward"] = float(reward)

        fault_severity = info.get("fault", {}).get("severity", dict.fromkeys(SURFACES, 1.0))
        for surface in SURFACES:
            row[f"fault_{surface}_severity"] = float(fault_severity.get(surface, 1.0))

        row["wind_level"] = str(info.get("wind_level", "off"))
        return row

    def _flush(self) -> None:
        if not self._buffer:
            return
        df = pd.DataFrame(self._buffer)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"episode_{self._episode:06d}.parquet"
        df.to_parquet(path, index=False)
        self._last_output_path = path
        self._buffer = []

    def close(self) -> None:
        self._flush()
        self.env.close()
