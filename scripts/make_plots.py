"""Generates all report figures (report/figures/*.pdf) and the markdown
results table (report/tables/main_results.md) from
data/results/sweep_results.parquet plus the sweep's raw per-step traces.
Run after scripts/run_sweep.py. No logic here beyond wiring paths -- see
rtrl_flight.analysis.plots and rtrl_flight.analysis.aggregate.
"""

from __future__ import annotations

from pathlib import Path

from rtrl_flight.analysis import plots
from rtrl_flight.analysis.aggregate import load_sweep, write_markdown_table

RESULTS_PATH = Path("data/results/sweep_results.parquet")
FIGURES_DIR = Path("report/figures")
TABLES_DIR = Path("report/tables")


def _fault_trace_paths() -> dict[str, Path]:
    """seed-0 fault-scenario trace for each controller. scripts/run_sweep.py
    uses deterministic trace dirs (data/traces/sweep_<scenario>_<controller>
    _seed<seed>/), so this is a direct lookup rather than scanning
    data/traces/ for "the most recent" timestamped subdir.
    """
    paths: dict[str, Path] = {}
    for controller_name in ("pid", "bptt_lstm", "rtrl_rtu"):
        trace_dir = Path(f"data/traces/sweep_fault_{controller_name}_seed0")
        episode_files = sorted(trace_dir.glob("episode_*.parquet"))
        if episode_files:
            paths[controller_name] = episode_files[-1]
        else:
            print(f"  WARNING: no fault trace found for {controller_name} at {trace_dir}")
    return paths


def main() -> None:
    df = load_sweep(RESULTS_PATH)

    plots.nominal_tracking(df, FIGURES_DIR / "nominal_tracking.pdf")
    plots.fault_recovery(_fault_trace_paths(), FIGURES_DIR / "fault_recovery.pdf")
    plots.wind_robustness(df, FIGURES_DIR / "wind_robustness.pdf")
    plots.combined_stress(df, FIGURES_DIR / "combined_stress.pdf")
    plots.ablation(df, FIGURES_DIR / "ablation.pdf")

    write_markdown_table(df, TABLES_DIR / "main_results.md")

    print("Generated figures:")
    for path in sorted(FIGURES_DIR.glob("*.pdf")):
        print(f"  {path}")
    print(f"Generated table: {TABLES_DIR / 'main_results.md'}")


if __name__ == "__main__":
    main()
