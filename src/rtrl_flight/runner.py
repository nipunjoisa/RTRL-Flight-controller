"""Experiment runner: build env + controller from Hydra config, run one
episode, load the resulting parquet trace, compute metrics, print a summary
table. See agent_docs/experiments.md (scenario definitions) and
agent_docs/evaluation.md (metric definitions).

Kept in src/ (not scripts/) per CLAUDE.md: "scripts/ # thin CLI
entrypoints; no logic" -- scripts/run_experiment.py and
scripts/run_ablation.py are both thin wrappers around run_experiment()
below.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import hydra
import pandas as pd

from rtrl_flight.controllers.base import Controller
from rtrl_flight.env.make import cfg_get, force_raw_obs, make_env
from rtrl_flight.env.wrappers.fault import FaultWrapper
from rtrl_flight.env.wrappers.trace import TraceWrapper
from rtrl_flight.metrics.recovery import time_to_recover
from rtrl_flight.metrics.smoothness import control_jerk
from rtrl_flight.metrics.tracking import (
    attitude_rmse,
    episode_return,
    pitch_rmse,
    roll_rmse,
    settling_time,
)
from rtrl_flight.training.imitation import PID_OBS_INDICES

PID_TARGET = "rtrl_flight.controllers.pid.PIDController"
CHECKPOINT_DIR = Path("data/checkpoints")
CHECKPOINT_SUFFIXES = ("_pretrain.pt", "_online.pt")  # checked in this order


def find_wrapper(env: gym.Env, wrapper_type: type) -> Any | None:
    """Walks the wrapper stack for an instance of wrapper_type. Not a
    controller-type isinstance check (agent_docs/architecture.md's design
    invariant #1 is specifically about the runner not branching on
    *controller* type) -- this is ordinary Gym wrapper-stack introspection:
    only FaultWrapper exposes trigger_fault() and only TraceWrapper exposes
    last_output_path, and neither is guaranteed to be present in the stack.
    """
    node = env
    while isinstance(node, gym.Wrapper):
        if isinstance(node, wrapper_type):
            return node
        node = node.env
    return None


def build_controller(controller_cfg: Any) -> Controller:
    """hydra.utils.instantiate(controller_cfg), with one factory-level
    special case: PIDController.__init__ requires obs_indices, which isn't
    (and shouldn't be) a Hydra config field -- it's tied to
    AttitudeHoldTask's fixed observation slot order (agent_docs/
    environment.md), not a per-run choice. Checking `_target_` here is the
    kind of controller-type awareness CLAUDE.md's "no isinstance checks
    outside factory functions" explicitly carves out room for -- this
    function *is* the factory.
    """
    target = cfg_get(controller_cfg, "_target_", "")
    if target == PID_TARGET:
        return hydra.utils.instantiate(controller_cfg, obs_indices=PID_OBS_INDICES)
    return hydra.utils.instantiate(controller_cfg)


def find_checkpoint(controller_name: str) -> Path | None:
    """data/checkpoints/<name>_pretrain.pt checked first, then
    data/checkpoints/<name>_online.pt. Returns None if neither exists (e.g.
    PID has no learned checkpoint)."""
    for suffix in CHECKPOINT_SUFFIXES:
        candidate = CHECKPOINT_DIR / f"{controller_name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_checkpoint_if_present(
    controller: Controller, controller_name: str, force_online_updates: bool | None
) -> None:
    path = find_checkpoint(controller_name)
    if path is None:
        print(f"  (no checkpoint found for '{controller_name}' -- using fresh weights)")
        return
    controller.load(path)
    print(f"  loaded checkpoint: {path}")
    # RTRLRTUController.load() restores its own saved online_updates value --
    # this run's requested value (plain config, or the ablation override)
    # must win over whatever was saved, not be silently clobbered by it.
    if force_online_updates is not None and hasattr(controller, "online_updates"):
        controller.online_updates = force_online_updates


def run_one_episode(
    env: gym.Env, controller: Controller, fault_cfg: Any, seed: int | None = None
) -> None:
    fault_wrapper = find_wrapper(env, FaultWrapper)
    onset_step = cfg_get(fault_cfg, "onset_step", None)
    surface = cfg_get(fault_cfg, "surface", None)
    severity = cfg_get(fault_cfg, "severity", None)

    controller.reset()
    if seed is not None:
        # env.reset(seed=...) alone is NOT sufficient for reproducibility:
        # AttitudeHoldTask._new_episode_init samples target_pitch_rad/
        # target_roll_rad via Python's global `random.uniform`, not
        # gymnasium's seeded self.np_random (a pre-existing gap in
        # attitude_task.py, found while wiring up a real multi-seed sweep --
        # a "seed=" CLI arg that never reached env.reset() at all, and even
        # once it does, the actual episode-defining randomness lives
        # outside Gymnasium's seeding convention). Seeding both is what
        # actually makes a given seed reproduce a specific episode.
        random.seed(seed)
    obs, _info = env.reset(seed=seed)
    step_count = 0
    terminated = truncated = False

    while not (terminated or truncated):
        if fault_wrapper is not None and onset_step is not None and step_count == onset_step:
            fault_wrapper.trigger_fault(surface, severity)

        action = controller.act(obs)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        # Exactly the 4 positional args the Controller ABC guarantees (see
        # src/rtrl_flight/controllers/base.py) -- PID/BPTT-LSTM's update()
        # don't accept a `done` kwarg, so passing one here would break the
        # "no branching across controllers" invariant this loop depends on.
        controller.update(obs, action, reward, next_obs)
        obs = next_obs
        step_count += 1


def compute_metrics(trace_path: Path) -> dict[str, float | None]:
    df = pd.read_parquet(trace_path)
    return {
        "pitch_rmse": pitch_rmse(df),
        "roll_rmse": roll_rmse(df),
        "attitude_rmse": attitude_rmse(df),
        "settling_time": settling_time(df),
        "episode_return": episode_return(df),
        "control_jerk": control_jerk(df),
        "time_to_recover": time_to_recover(df),
    }


def format_summary(
    controller_name: str,
    env_name: str,
    seed: int,
    metrics: dict[str, float | None],
    ablation: bool = False,
) -> str:
    def fmt(value: float | None, unit: str = "") -> str:
        return "N/A" if value is None else f"{value:.4f}{unit}"

    def fmt_steps(value: float | None) -> str:
        return "N/A" if value is None else f"{int(value)} steps"

    lines = []
    if ablation:
        lines.append("*** ABLATION: online updates DISABLED ***")
    lines.append(f"Controller : {controller_name}")
    lines.append(f"Env        : {env_name}")
    lines.append(f"Seed       : {seed}")
    lines.append("-" * 29)  # plain ASCII: Windows console cp1252 can't encode "─"
    lines.append(f"Pitch RMSE        : {fmt(metrics['pitch_rmse'], ' rad')}")
    lines.append(f"Roll RMSE         : {fmt(metrics['roll_rmse'], ' rad')}")
    lines.append(f"Attitude RMSE     : {fmt(metrics['attitude_rmse'], ' rad')}")
    lines.append(f"Settling time     : {fmt_steps(metrics['settling_time'])}")
    lines.append(f"Episode return    : {metrics['episode_return']:.1f}")
    lines.append(f"Control jerk      : {metrics['control_jerk']:.4f}")
    lines.append(f"Time to recover   : {fmt_steps(metrics['time_to_recover'])}")
    return "\n".join(lines)


def run_experiment(
    cfg: Any,
    controller_name: str,
    env_name: str,
    ablation: bool = False,
) -> dict[str, float | None]:
    """The full pipeline described in this message: build env -> instantiate
    controller -> load checkpoint -> run one episode -> compute metrics from
    the trace -> print + return the summary.
    """
    # See rtrl_flight.env.make.force_raw_obs: every controller here is
    # trained/calibrated on raw radian-scale obs, and metrics reported in
    # physical units (rad) are meaningless on NormalizeWrapper's [-1, 1]
    # output -- cessna172_*.yaml's `normalize: true` default is overridden
    # for evaluation for exactly the same reason imitation.py overrides it
    # for rollout collection.
    env = make_env(force_raw_obs(cfg.env))

    controller = build_controller(cfg.controller)
    force_online_updates = False if ablation else cfg_get(cfg.controller, "online_updates", None)
    load_checkpoint_if_present(controller, controller_name, force_online_updates)

    try:
        run_one_episode(env, controller, cfg_get(cfg.env, "fault", None), seed=cfg.seed)
    finally:
        trace_wrapper = find_wrapper(env, TraceWrapper)
        env.close()

    if trace_wrapper is None or trace_wrapper.last_output_path is None:
        raise RuntimeError(
            "No trace was written -- cfg.env.trace.enabled must be true for "
            "run_experiment to compute metrics (agent_docs/architecture.md "
            "design invariant #4: analysis reads parquet, never live state)."
        )

    metrics = compute_metrics(trace_wrapper.last_output_path)
    print(format_summary(controller_name, env_name, cfg.seed, metrics, ablation=ablation))
    return metrics
