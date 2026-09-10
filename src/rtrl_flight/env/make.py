"""Factory composing the env wrapper stack from config. See
agent_docs/architecture.md's system diagram: AttitudeHoldTask ->
NoFGJsbSimEnv -> NormalizeWrapper -> FaultWrapper -> WindWrapper ->
TraceWrapper. Fault/Wind/Trace are only added if cfg says to -- a nominal
run doesn't need fault or wind wrappers at all.

`cfg` is duck-typed to accept either a Hydra/OmegaConf DictConfig or a plain
dict/None (configs/env/*.yaml doesn't exist yet -- see agent_docs/
architecture.md's directory map -- so this can't assume OmegaConf's
attribute-style access is always available).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from jsbgym.aircraft import c172
from jsbgym.environment import NoFGJsbSimEnv
from jsbgym.tasks import Shaping

from rtrl_flight.env.attitude_task import AttitudeHoldTask
from rtrl_flight.env.wrappers.fault import FaultWrapper
from rtrl_flight.env.wrappers.normalize import NormalizeWrapper
from rtrl_flight.env.wrappers.trace import TraceWrapper
from rtrl_flight.env.wrappers.wind import WindWrapper


def _get(cfg: Any, key: str, default: Any) -> Any:
    """cfg.get(key, default) that works for a DictConfig, a plain dict, or None."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def make_env(cfg: Any = None) -> gym.Env:
    agent_interaction_freq = _get(cfg, "agent_interaction_freq", 5)

    env: gym.Env = NoFGJsbSimEnv(
        aircraft=c172,
        task_type=AttitudeHoldTask,
        agent_interaction_freq=agent_interaction_freq,
        shaping=Shaping.STANDARD,
    )
    # NOTE: episode_time_s is not yet plumbed through -- NoFGJsbSimEnv calls
    # `task_type(shaping, agent_interaction_freq, aircraft)` positionally
    # (jsbgym/environment.py), with no hook for extra constructor kwargs, so
    # AttitudeHoldTask always uses its DEFAULT_EPISODE_TIME_S for now.

    if _get(cfg, "normalize", True):
        env = NormalizeWrapper(env)

    fault_cfg = _get(cfg, "fault", None)
    if _get(fault_cfg, "enabled", False):
        env = FaultWrapper(env)

    wind_cfg = _get(cfg, "wind", None)
    if _get(wind_cfg, "enabled", False):
        env = WindWrapper(env, level=_get(wind_cfg, "level", "off"))

    trace_cfg = _get(cfg, "trace", None)
    if _get(trace_cfg, "enabled", False):
        env = TraceWrapper(env, output_dir=_get(trace_cfg, "output_dir", "data/traces"))

    return env
