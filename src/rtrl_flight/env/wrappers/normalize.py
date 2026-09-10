"""Per-channel observation normalization + reward clipping. See
agent_docs/architecture.md's wrapper stack: NormalizeWrapper sits directly
on the base env so everything above it sees a fixed [-1, 1] range,
regardless of a channel's native units (ft, rad, ft/s, ...).

Deliberately reads bounds from the wrapped env's own `observation_space`
rather than re-hardcoding them here -- the bounds already live on
AttitudeHoldTask's state_variables (agent_docs/environment.md), duplicating
them here would just create a second place for them to drift out of sync.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class NormalizeWrapper(gym.Wrapper):
    """obs -> 2*(obs - low)/(high - low) - 1, clipped to [-1, 1].
    reward -> clipped to [-1, 1].

    Fixed normalization from the wrapped env's declared observation_space
    bounds -- not running statistics. Running normalization would leak
    episode-to-episode statistics into what's supposed to be a fixed
    mapping; if that's ever wanted it should be a distinct, explicitly
    documented wrapper (see agent_docs/architecture.md's note that running
    stats must be a deliberate, documented choice), not a default baked in
    here.
    """

    def __init__(self, env: gym.Env, reward_clip: float = 1.0) -> None:
        super().__init__(env)
        self._low = env.observation_space.low.astype(np.float64)
        self._high = env.observation_space.high.astype(np.float64)
        self._range = self._high - self._low
        self._reward_clip = reward_clip

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env.observation_space.shape, dtype=np.float64
        )

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        scaled = 2.0 * (obs - self._low) / self._range - 1.0
        return np.clip(scaled, -1.0, 1.0)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._normalize_obs(obs), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        norm_obs = self._normalize_obs(obs)
        norm_reward = float(np.clip(reward, -self._reward_clip, self._reward_clip))
        return norm_obs, norm_reward, terminated, truncated, info
