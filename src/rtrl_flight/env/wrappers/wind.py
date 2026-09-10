"""Dryden turbulence wrapper, via JSBSim's built-in Tustin-model turbulence
implementation (MIL-F-8785C-derived). See agent_docs/architecture.md:
composable with FaultWrapper in either order -- this wrapper only ever
touches JSBSim's `atmosphere/*` properties, never the action vector, so it
has no interaction with FaultWrapper's action-scaling regardless of nesting
order.

Property names confirmed to exist against the installed jsbsim==1.3.1
property catalog (`sim.jsbsim.get_property_catalog()`); jsbgym itself
defines no turbulence properties (not in jsbgym/properties.py), so these are
raw JSBSim property names, not jsbgym BoundedProperty catalog entries. Numeric
severity/windspeed values below are a best-effort mapping to
light/moderate/severe -- not verified against a MIL-F-8785C probability
table -- revisit if wind-scenario results look off (agent_docs/experiments.md
Scenario 3).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from jsbgym.properties import Property

TURB_TYPE = Property("atmosphere/turb-type", "turbulence model (0=none, 3=Tustin)")
TURB_SEVERITY = Property("atmosphere/turbulence/milspec/severity", "MIL-8785C severity index")
TURB_WINDSPEED = Property(
    "atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps",
    "reference windspeed at 20ft AGL [ft/s]",
)

TUSTIN_TURB_TYPE = 3
NO_TURB_TYPE = 0

# level -> (turb-type, milspec severity index, windspeed_at_20ft_AGL-fps)
LEVELS: dict[str, tuple[int, int, float]] = {
    "off": (NO_TURB_TYPE, 0, 0.0),
    "light": (TUSTIN_TURB_TYPE, 3, 15.0),
    "moderate": (TUSTIN_TURB_TYPE, 5, 30.0),
    "severe": (TUSTIN_TURB_TYPE, 7, 45.0),
}


class WindWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, level: str = "off") -> None:
        super().__init__(env)
        self._level = level
        self._validate(level)
        self._apply_to_sim()

    @property
    def wind_level(self) -> str:
        return self._level

    def _validate(self, level: str) -> None:
        if level not in LEVELS:
            raise ValueError(f"unknown wind level {level!r}, expected one of {tuple(LEVELS)}")

    def set_severity(self, level: str) -> None:
        self._validate(level)
        self._level = level
        self._apply_to_sim()

    def _apply_to_sim(self) -> None:
        # env.unwrapped.sim is None until the first reset() (see jsbgym's
        # JsbSimEnv.__init__) -- if set_severity() is called before that,
        # reset() re-applies the current level once the sim exists.
        sim = self.env.unwrapped.sim
        if sim is None:
            return
        turb_type, severity, windspeed = LEVELS[self._level]
        sim[TURB_TYPE] = turb_type
        sim[TURB_SEVERITY] = severity
        sim[TURB_WINDSPEED] = windspeed

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        result = self.env.reset(**kwargs)
        # Wind severity is an ambient run-level setting, not per-episode
        # state -- unlike FaultWrapper's reset-to-full-authority, we
        # deliberately re-apply (not clear) the current level here, since
        # JSBSim's atmosphere properties reset to their own defaults on
        # sim.reinitialise().
        self._apply_to_sim()
        return result

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["wind_level"] = self._level
        return obs, reward, terminated, truncated, info
