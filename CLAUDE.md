# CLAUDE.md

Project context for Claude Code. Read this before making changes.

## Project

**RTRL Flight Surface Controller**

Build a deep-learning flight controller for a fixed-wing aircraft (Cessna 172 in JSBSim) that predicts aileron/elevator/rudder deflections from aircraft state. The core novelty is using **Real-Time Recurrent Learning (RTRL)** for genuine in-flight weight adaptation, benchmarked against BPTT-LSTM and PID baselines under three scenarios: nominal, actuator faults, and wind turbulence.

**Success = a working RTRL controller that measurably recovers from mid-flight actuator degradation where a frozen BPTT-LSTM does not.** Everything else is secondary.

Full context lives in `agent_docs/`. Start there for any substantive question:
- `project_overview.md` — scope, timeline, success criteria
- `architecture.md` — system diagram, component contracts
- `rtrl.md` — the math (forward sensitivity, update rule)
- `experiments.md` — the 4 scenarios + the ablation
- `research_context.md` — what's novel vs prior art

## Tech stack

- **Python 3.11**, **PyTorch 2.4+** (uses `torch.func` for per-parameter Jacobians)
- **jsbgym** (maintained gym-jsbsim fork) + **Gymnasium 1.3.0** <!-- updated after jsbgym probe — original spec said Gymnasium 0.29+ -->
- **Hydra** for configs, **W&B** for logging, **uv** for env management
- **pytest** for tests, **ruff** for lint/format
- No JAX, no Ray/RLlib, no Docker — kept out deliberately

## Repository layout

```
src/rtrl_flight/    # library code
  env/              # jsbgym + fault/wind/normalize/trace wrappers
  controllers/      # PID, BPTT-LSTM, RTRL-RTU (all satisfy Controller ABC)
  rtrl/             # RTU cell, forward sensitivity, online update, warm-start
  training/         # imitation, BPTT, online RTRL loops
  metrics/          # tracking, smoothness, recovery
  analysis/         # parquet → DataFrames → plots
scripts/            # thin CLI entrypoints; no logic
configs/            # Hydra configs (controller/, env/, experiment/)
tests/              # pytest — see "Testing" section below
agent_docs/         # design docs (source of truth for intent)
notebooks/          # exploratory only, never imported by src/
```

## Design invariants — do not violate

1. **All controllers implement the `Controller` ABC** (`act`, `update`, `reset`, `save`, `load`). The experiment runner does not know controller types. No `isinstance` checks anywhere outside factory functions.
2. **Faults and wind are Gymnasium wrappers**, not controller logic. Composable: `WindWrapper(FaultWrapper(env))`.
3. **RTRL's online weight update is `controller.update()`** — called by the runner after each step. PID and BPTT-LSTM implement `update` as a no-op.
4. **Episode data flows through `TraceLogger` → parquet.** Analysis reads parquet, never live state. This decouples runs from plots.
5. **One Hydra config per experiment.** No hardcoded paths or hyperparameters in scripts.
6. **`src/` never imports from `scripts/` or `notebooks/`.** One-way dependency.

## Critical correctness bar

**`tests/test_rtrl_sensitivity.py` is the most important file in the repo.** It verifies the forward sensitivity computation matches a finite-difference gradient on a toy problem to ~1e-5 tolerance. If this test doesn't pass, we are not doing RTRL — we are doing something that looks like it. Never merge changes to `src/rtrl_flight/rtrl/` without this test green.

## Working conventions

**Before writing code:** re-read the relevant `agent_docs/` file. Docs are the source of truth for intent; code implements them. If code and docs disagree, ask which is right — don't silently pick one.

**Before adding a dependency:** check `pyproject.toml`. Justify additions in the commit message. Reject anything that pulls in JAX, TensorFlow, Ray, or a new deep-learning framework.

**When touching `src/rtrl_flight/rtrl/`:** the sensitivity test must stay green. Run `pytest tests/test_rtrl_sensitivity.py -v` before and after.

**When touching controllers:** all three must still pass `tests/test_controller_interface.py`. The ABC contract is load-bearing.

**When touching env wrappers:** verify composition still works (`WindWrapper(FaultWrapper(base))` and the reverse). Wrappers must not leak state between episodes — reset semantics matter.

**Commit style:** short imperative subject, body explains *why*. Reference the `agent_docs/` file the change implements when relevant.

**Reproducibility:** every experiment run gets a Hydra output dir with config, seed, git SHA, and W&B run ID. Never delete these. If a result can't be reproduced from its output dir, it doesn't count.

## What to do without asking

- Add unit tests for new functions
- Fix lint (`ruff check --fix`) and format (`ruff format`) before committing
- Update the relevant `agent_docs/` file when changing behavior it describes
- Add docstrings to public functions in `src/`
- Prefer small, focused PRs over sweeping changes

## What to ask first

- Adding a new controller, wrapper, or metric (touches the ABC or the trace schema)
- Changing the observation or action space
- Changing the reward function (invalidates prior runs)
- Adding a new dependency
- Anything that would require re-running completed experiments
- Anything that touches `src/rtrl_flight/rtrl/sensitivity.py` or `update.py`

## What NOT to do

- Do not add F-16 or other airframes. Cessna 172 only.
- Do not add trajectory or heading tracking. Attitude hold only (target pitch + target roll).
- Do not add PPO or SAC baselines. PID and BPTT-LSTM only.
- Do not add physics-informed loss terms or PID+residual architectures — those were explicitly deferred.
- Do not "improve" the RTRL update with tricks from papers we haven't discussed (Kronecker approximations, sparsity heuristics, etc.). We use RTUs specifically because they make exact RTRL tractable — don't approximate what's already tractable.
- Do not hardcode paths. Use Hydra config or `pathlib` relative to the repo root.
- Do not commit `data/`, `outputs/`, W&B artifacts, or notebook outputs.
- Do not skip the imitation warm-start on RTRL training. Cold-start online RTRL on flight dynamics diverges — this is known, not a bug to fix.

## Scenarios and what "done" looks like

Four scenarios × three controllers × N seeds, each producing a parquet trace:
1. **Nominal** — attitude hold in calm air. All controllers should track within ~15% RMSE of PID.
2. **Fault** — aileron/elevator/rudder degraded mid-episode (independent runs per surface). RTRL should recover; frozen baselines should not.
3. **Wind** — Dryden turbulence at 3 severity levels, unseen in training.
4. **Combined** — fault + wind simultaneously. The headline demo.
5. **Ablation** — RTRL with online updates disabled at deploy time. Delta vs online-on is the paper's central claim.

Report metrics: tracking RMSE, settling time, control smoothness (jerk), time-to-recover post-fault. Statistical protocol in `agent_docs/evaluation.md`.

## Scope discipline

This is a **14-day project**, not a research program. If a task would take more than half a day and isn't on the critical path to the five scenarios above, flag it before starting. The fault experiment is the strongest result and must be protected — cut wind before cutting faults, cut the ablation before cutting either.

## Where to look when stuck

- Env issues → `agent_docs/environment.md`, then `src/rtrl_flight/env/`
- RTRL math → `agent_docs/rtrl.md`, then Irie et al. ICLR 2024 and Elelimy et al. NeurIPS 2024 (cited in `research_context.md`)
- Training instability → check imitation warm-start ran, check learning rate, check sensitivity test still passes
- Weird results → load the parquet trace, don't trust W&B aggregates alone