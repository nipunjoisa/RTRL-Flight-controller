"""Aggregation over a sweep's results DataFrame (scripts/run_sweep.py's
sweep_results.parquet). Per agent_docs/architecture.md design invariant #4,
this reads only the saved parquet -- never live env/controller state.

agent_docs/evaluation.md calls for median/IQR ("flight-dynamics metrics are
heavy-tailed"), but this module reports mean +/- std as this task's PART B
explicitly specifies -- flagging the deviation rather than silently
resolving it, same as prior sessions' documented departures from
evaluation.md's phrasing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METRIC_COLS = (
    "pitch_rmse",
    "roll_rmse",
    "attitude_rmse",
    "episode_return",
    "control_jerk",
    "time_to_recover",
)
GROUP_COLS = ("scenario", "controller")


def load_sweep(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Groups by scenario x controller, computing mean +/- std for each
    metric in METRIC_COLS. None/NaN rows are excluded from each metric's
    own mean/std independently (e.g. a controller that never recovers in
    any seed has time_to_recover excluded entirely for that cell, not
    counted as 0 or dropped from pitch_rmse's mean too) -- pandas' groupby
    .mean()/.std() already skip NaN per-column by default, which is exactly
    this behavior.
    """
    rows = []
    for (scenario, controller), group in df.groupby(list(GROUP_COLS), sort=False):
        row: dict[str, object] = {"scenario": scenario, "controller": controller}
        for metric in METRIC_COLS:
            values = group[metric].astype(float)
            valid = values.dropna()
            row[f"{metric}_mean"] = float(valid.mean()) if len(valid) else np.nan
            row[f"{metric}_std"] = float(valid.std()) if len(valid) > 1 else 0.0
            row[f"{metric}_n"] = int(len(valid))
        rows.append(row)

    result = pd.DataFrame(rows)
    # Stable, readable ordering: group by scenario in the canonical order
    # the sweep itself uses, not alphabetically.
    scenario_order = {name: i for i, name in enumerate(df["scenario"].unique())}
    result["_scenario_order"] = result["scenario"].map(scenario_order)
    result = result.sort_values(["_scenario_order", "controller"]).drop(columns="_scenario_order")
    return result.reset_index(drop=True)


def print_summary(df: pd.DataFrame) -> None:
    table = summary_table(df)
    for scenario, group in table.groupby("scenario", sort=False):
        print(f"\n=== {scenario} ===")
        for _, row in group.iterrows():
            print(f"  {row['controller']:<12s}", end="")
            for metric in METRIC_COLS:
                mean = row[f"{metric}_mean"]
                std = row[f"{metric}_std"]
                n = row[f"{metric}_n"]
                if np.isnan(mean):
                    formatted = "N/A"
                elif metric == "time_to_recover":
                    formatted = f"{mean:.1f}+/-{std:.1f} steps (n={n})"
                else:
                    formatted = f"{mean:.4f}+/-{std:.4f} (n={n})"
                print(f"  {metric}={formatted}", end="")
            print()


def write_markdown_table(df: pd.DataFrame, path: str | Path) -> None:
    """report/tables/main_results.md (PART D): rows scenario x controller,
    columns Attitude RMSE / Episode Return (mean +/- std), Control Jerk
    (mean), Time to Recover (mean steps or N/A). Generated from the sweep
    parquet via summary_table(), not written by hand.
    """
    table = summary_table(df)
    lines = [
        "# Main Results",
        "",
        "Generated from `data/results/sweep_results.parquet` via "
        "`rtrl_flight.analysis.aggregate.write_markdown_table` -- do not hand-edit.",
        "",
        "| Scenario | Controller | Attitude RMSE (rad) | Episode Return "
        "| Control Jerk | Time to Recover (steps) |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in table.iterrows():
        attitude_rmse = f"{row['attitude_rmse_mean']:.4f} +/- {row['attitude_rmse_std']:.4f}"
        episode_return = f"{row['episode_return_mean']:.1f} +/- {row['episode_return_std']:.1f}"
        control_jerk = f"{row['control_jerk_mean']:.4f}"
        if np.isnan(row["time_to_recover_mean"]):
            time_to_recover = "N/A"
        else:
            time_to_recover = f"{row['time_to_recover_mean']:.1f}"
        lines.append(
            f"| {row['scenario']} | {row['controller']} | {attitude_rmse} | "
            f"{episode_return} | {control_jerk} | {time_to_recover} |"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
