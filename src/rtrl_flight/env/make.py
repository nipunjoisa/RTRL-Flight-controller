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


def cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """cfg.get(key, default) that works for a DictConfig, a plain dict, or None."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def force_raw_obs(env_cfg: Any) -> Any:
    """Returns a copy of env_cfg with normalize forced to False.

    Every controller in this repo is trained/calibrated on raw radian-scale
    obs, not NormalizeWrapper's [-1, 1]-squashed range: PID_GAINS assumes raw
    error magnitudes (see rtrl_flight.controllers.pid), and BPTT-LSTM/
    RTRL-RTU are warm-started from PID rollouts collected with normalize
    forced off (rtrl_flight.training.imitation.pretrain). Metrics computed
    in physical units (rad) are equally meaningless on normalized obs. Yet
    configs/env/cessna172_*.yaml default to `normalize: true` -- so both
    training (imitation.py) and evaluation (the experiment runner) need this
    override; centralized here instead of duplicated in both places.
    """
    if env_cfg is None:
        return {"normalize": False}
    if hasattr(env_cfg, "get"):
        merged = dict(env_cfg)
        merged["normalize"] = False
        return merged
    return env_cfg


def make_env(cfg: Any = None) -> gym.Env:
    agent_interaction_freq = cfg_get(cfg, "agent_interaction_freq", 5)

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

    if cfg_get(cfg, "normalize", True):
        env = NormalizeWrapper(env)

    fault_cfg = cfg_get(cfg, "fault", None)
    if cfg_get(fault_cfg, "enabled", False):
        env = FaultWrapper(env)

    wind_cfg = cfg_get(cfg, "wind", None)
    if cfg_get(wind_cfg, "enabled", False):
        env = WindWrapper(env, level=cfg_get(wind_cfg, "level", "off"))

    trace_cfg = cfg_get(cfg, "trace", None)
    if cfg_get(trace_cfg, "enabled", False):
        env = TraceWrapper(env, output_dir=cfg_get(trace_cfg, "output_dir", "data/traces"))

    return env
