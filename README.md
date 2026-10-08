# RTRL Flight Surface Controller

A flight controller for a Cessna 172 in JSBSim that predicts aileron, elevator and rudder deflections to hold a target attitude (pitch and roll).
It uses Real-Time Recurrent Learning (RTRL) with Recurrent Trace Units to adapt its weights in flight, aiming to recover from actuator faults.
It is benchmarked against PID and BPTT-LSTM baselines under nominal, fault, wind and combined scenarios.

## Results (3 seeds per cell, attitude RMSE in rad, lower is better)

| Scenario | PID | BPTT-LSTM | RTRL-RTU |
|---|---|---|---|
| Nominal | 0.175 | 0.319 | 1.582 |
| Fault | 0.222 | 0.545 | 1.534 |
| Wind | 0.205 | 0.332 | 1.598 |
| Combined | 0.253 | 0.470 | 1.693 |

PID was best everywhere and the frozen BPTT-LSTM second. The RTRL controller did not hold attitude even in calm air, so the fault-recovery hypothesis is untested rather than disproved. The ablation (online updates off) was too noisy to show that online learning helps.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Usage

```bash
python scripts/train_bptt.py             # train the BPTT-LSTM baseline
python scripts/train_rtrl_warmstart.py   # imitation warm-start for RTRL
python scripts/run_experiment.py         # run one experiment (Hydra config)
python scripts/run_sweep.py              # full scenario x controller x seed sweep
python scripts/run_ablation.py           # RTRL with online updates disabled
python scripts/make_plots.py             # parquet traces -> figures
python scripts/live_demo.py              # live demo
pytest                                   # tests, incl. tests/test_rtrl_sensitivity.py
```

## Layout

```
src/rtrl_flight/   env/ controllers/ rtrl/ training/ metrics/ analysis/
scripts/           thin CLI entrypoints
configs/           Hydra configs (controller/, env/, experiment/)
tests/             pytest suite
agent_docs/        design docs (architecture, RTRL math, experiments)
report/            written report and figures
```

## Design

- All controllers implement one `Controller` interface (`act`, `update`, `reset`, `save`, `load`).
- Faults and wind are composable Gymnasium wrappers.
- Episodes are logged to parquet traces; analysis reads only those traces.
- `tests/test_rtrl_sensitivity.py` checks the RTRL sensitivity against finite differences.

See `agent_docs/` for details.
