"""Full experiment sweep: 3 seeds x 3 controllers x 4 scenarios + the
online-off ablation. Plain Python (no Hydra @hydra.main) -- builds env/
controller config as plain dicts and reuses rtrl_flight.runner's
build_controller/load_checkpoint_if_present/run_one_episode/compute_metrics
directly, since all of them are already duck-typed to accept a plain dict
or a Hydra DictConfig (rtrl_flight.env.make.cfg_get).

Trace output dirs are deterministic
(data/traces/sweep_<scenario>_<controller>_seed<seed>/), not the
timestamp-based scheme configs/env/*.yaml uses -- this sweep runs many
back-to-back episodes and needs to reliably find "the fault-scenario trace
for controller X" afterward (scripts/make_plots.py), which a bare
timestamp can't disambiguate on its own.

Usage:
  python scripts/run_sweep.py              # run the full sweep
  python scripts/run_sweep.py --summary    # just print the summary table
                                            # from an existing parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from rtrl_flight.analysis.aggregate import load_sweep, print_summary
from rtrl_flight.env.make import force_raw_obs, make_env
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
from rtrl_flight.runner import (
    build_controller,
    find_wrapper,
    load_checkpoint_if_present,
    run_one_episode,
)

RESULTS_PATH = Path("data/results/sweep_results.parquet")

SEEDS = [0, 1, 2]

CONTROLLER_CFGS: dict[str, dict[str, Any]] = {
    "pid": {
        "_target_": "rtrl_flight.controllers.pid.PIDController",
        "dt": 0.2,
        "gains": {
            "aileron": [0.4, 0.02, 0.15],
            "elevator": [0.4, 0.02, 0.15],
            "rudder": [0.1, 0.0, 0.05],
        },
        "rudder_roll_coordination_gain": 0.1,
    },
    "bptt_lstm": {
        "_target_": "rtrl_flight.controllers.bptt_lstm.BPTTLSTMController",
        "hidden_size": 64,
        "num_layers": 2,
        "lr": 1e-3,
    },
    "rtrl_rtu": {
        "_target_": "rtrl_flight.controllers.rtrl_rtu.RTRLRTUController",
        "hidden_size": 64,
        "lr": 1e-3,
        "online_lr": 1e-4,
        "online_updates": True,
    },
}
ALL_CONTROLLERS = ("pid", "bptt_lstm", "rtrl_rtu")

FAULT_CFG = {"enabled": True, "surface": "aileron", "severity": 0.5, "onset_step": 150}
WIND_CFG = {"enabled": True, "level": "moderate"}


def _trace_dir(scenario: str, controller_name: str, seed: int) -> str:
    return f"data/traces/sweep_{scenario}_{controller_name}_seed{seed}"


def _env_cfg(scenario: str, controller_name: str, seed: int) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "agent_interaction_freq": 5,
        "normalize": True,  # overridden to False by force_raw_obs regardless
        "fault": {"enabled": False},
        "wind": {"enabled": False},
        "trace": {"enabled": True, "output_dir": _trace_dir(scenario, controller_name, seed)},
    }
    if scenario in ("fault", "combined", "ablation"):
        cfg["fault"] = dict(FAULT_CFG)
    if scenario in ("wind", "combined"):
        cfg["wind"] = dict(WIND_CFG)
    return cfg


# Each entry: (scenario_name, [controller_names], ablation)
SCENARIO_PLAN = [
    ("nominal", ALL_CONTROLLERS, False),
    ("fault", ALL_CONTROLLERS, False),
    ("wind", ALL_CONTROLLERS, False),
    ("combined", ALL_CONTROLLERS, False),
    ("ablation", ("rtrl_rtu",), True),
]


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


def run_single(scenario: str, controller_name: str, seed: int, ablation: bool) -> dict[str, Any]:
    env_cfg = _env_cfg(scenario, controller_name, seed)
    env = make_env(force_raw_obs(env_cfg))

    controller_cfg = dict(CONTROLLER_CFGS[controller_name])
    controller = build_controller(controller_cfg)

    force_online_updates = False if ablation else controller_cfg.get("online_updates")
    load_checkpoint_if_present(controller, controller_name, force_online_updates)

    try:
        run_one_episode(env, controller, env_cfg.get("fault"), seed=seed)
    finally:
        trace_wrapper = find_wrapper(env, TraceWrapper)
        env.close()

    if trace_wrapper is None or trace_wrapper.last_output_path is None:
        raise RuntimeError(f"no trace written for {scenario}/{controller_name}/seed={seed}")

    metrics = compute_metrics(trace_wrapper.last_output_path)
    metrics["scenario"] = scenario
    metrics["controller"] = controller_name
    metrics["seed"] = seed
    return metrics


def run_sweep() -> pd.DataFrame:
    runs = [
        (scenario, controller_name, seed, ablation)
        for scenario, controllers, ablation in SCENARIO_PLAN
        for controller_name in controllers
        for seed in SEEDS
    ]
    total = len(runs)
    rows = []

    for i, (scenario, controller_name, seed, ablation) in enumerate(runs, start=1):
        metrics = run_single(scenario, controller_name, seed, ablation)
        rows.append(metrics)
        rmse = metrics["attitude_rmse"]
        jerk = metrics["control_jerk"]
        print(
            f"[{i}/{total}] {controller_name} | {scenario} | seed={seed} "
            f"-> RMSE={rmse:.4f}, jerk={jerk:.4f}"
        )

    df = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS_PATH, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the summary table from the existing parquet without re-running the sweep.",
    )
    args = parser.parse_args()

    if args.summary:
        df = load_sweep(RESULTS_PATH)
        print_summary(df)
        return

    df = run_sweep()
    print(f"\nsaved {len(df)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
