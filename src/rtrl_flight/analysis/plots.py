"""Report figures generated from the sweep results parquet + raw per-step
traces. See scripts/make_plots.py for the CLI entrypoint. Matplotlib only
(already a project dependency, per scripts/live_demo.py) -- no new
dependency, and headless (Agg backend) since these are saved report
figures, not an interactive window.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rtrl_flight.analysis.aggregate import summary_table  # noqa: E402

CONTROLLER_ORDER = ("pid", "bptt_lstm", "rtrl_rtu")
CONTROLLER_LABELS = {"pid": "PID", "bptt_lstm": "BPTT-LSTM", "rtrl_rtu": "RTRL-RTU"}
CONTROLLER_COLORS = {"pid": "#4C72B0", "bptt_lstm": "#DD8452", "rtrl_rtu": "#55A868"}


def _bar_chart_attitude_rmse(df: pd.DataFrame, scenario: str, title: str, out_path: Path) -> None:
    table = summary_table(df)
    scenario_rows = table[table["scenario"] == scenario].set_index("controller")

    controllers = [c for c in CONTROLLER_ORDER if c in scenario_rows.index]
    means = [scenario_rows.loc[c, "attitude_rmse_mean"] for c in controllers]
    stds = [scenario_rows.loc[c, "attitude_rmse_std"] for c in controllers]
    labels = [CONTROLLER_LABELS[c] for c in controllers]
    colors = [CONTROLLER_COLORS[c] for c in controllers]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=stds, capsize=5, color=colors)
    ax.set_ylabel("Attitude RMSE (rad)")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def nominal_tracking(df: pd.DataFrame, out_path: Path) -> None:
    _bar_chart_attitude_rmse(df, "nominal", "Nominal Scenario: Attitude Tracking", out_path)


def wind_robustness(df: pd.DataFrame, out_path: Path) -> None:
    _bar_chart_attitude_rmse(df, "wind", "Wind Scenario: Attitude Tracking", out_path)


def combined_stress(df: pd.DataFrame, out_path: Path) -> None:
    _bar_chart_attitude_rmse(df, "combined", "Combined Scenario: Attitude Tracking", out_path)


def ablation(df: pd.DataFrame, out_path: Path) -> None:
    """Side-by-side bars: RTRL-RTU online-ON (scenario='fault') vs
    online-OFF (scenario='ablation') attitude RMSE, per
    agent_docs/experiments.md Scenario 5 -- the paper's central causal
    claim, so this figure gets its own dedicated comparison rather than
    reusing the generic per-scenario bar chart.
    """
    table = summary_table(df)
    on_row = table[(table["scenario"] == "fault") & (table["controller"] == "rtrl_rtu")].iloc[0]
    off_row = table[(table["scenario"] == "ablation") & (table["controller"] == "rtrl_rtu")].iloc[0]

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["online ON", "online OFF"]
    means = [on_row["attitude_rmse_mean"], off_row["attitude_rmse_mean"]]
    stds = [on_row["attitude_rmse_std"], off_row["attitude_rmse_std"]]
    ax.bar(labels, means, yerr=stds, capsize=5, color=["#55A868", "#C44E52"])
    ax.set_ylabel("Attitude RMSE (rad)")
    ax.set_title("Ablation: RTRL-RTU Online Updates (Fault Scenario)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def fault_recovery(trace_paths: dict[str, Path], out_path: Path, onset_step: int = 150) -> None:
    """Per-step pitch_error/roll_error over one fault-scenario episode, all
    controllers overlaid, with a vertical dashed line at fault onset.

    trace_paths: {controller_name: path to that controller's seed-0
    fault-scenario trace parquet} -- see scripts/make_plots.py for how
    these are located (the sweep's deterministic trace dirs, not a
    timestamp scan).
    """
    fig, (ax_pitch, ax_roll) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for controller_name in CONTROLLER_ORDER:
        if controller_name not in trace_paths:
            continue
        trace_df = pd.read_parquet(trace_paths[controller_name])
        steps = trace_df["step"]
        color = CONTROLLER_COLORS[controller_name]
        label = CONTROLLER_LABELS[controller_name]
        ax_pitch.plot(steps, trace_df["obs_error_pitch_error_rad"], color=color, label=label)
        ax_roll.plot(steps, trace_df["obs_error_roll_error_rad"], color=color, label=label)

    ax_pitch.axvline(onset_step, color="black", linestyle="--", label="fault onset")
    ax_roll.axvline(onset_step, color="black", linestyle="--")
    ax_pitch.set_ylabel("Pitch error (rad)")
    ax_roll.set_ylabel("Roll error (rad)")
    ax_roll.set_xlabel("step")
    ax_pitch.legend(fontsize=8)
    fig.suptitle("Fault Recovery: Tracking Error Over Time (seed=0)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
