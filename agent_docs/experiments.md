# Experiments

Each scenario below is a Hydra config under `configs/experiment/`. All scenarios run all three controllers (PID, BPTT-LSTM, RTRL-RTU) unless noted, across N seeds (N=10 default — enough for the statistical protocol in `evaluation.md` without burning the 14-day budget; drop to 5 only if time-constrained, note it in the run's output dir).

## Common setup

- Airframe: Cessna 172, JSBSim, via `jsbgym`.
- Task: attitude hold — track a target pitch and target roll, sampled per episode (see `environment.md` for ranges). No trajectory/heading tracking.
- Episode length: fixed (see `configs/env/base.yaml`); truncation, not early termination on tracking error, so post-fault recovery has time to show up in the trace.
- All controllers warm-started via imitation (see below) before any scenario runs. PID needs no warm-start (it's not learned) but runs through the same harness for a fair baseline.

## Imitation warm-start (prerequisite for BPTT-LSTM and RTRL-RTU)

Expert: the tuned PID controller (`configs/controller/pid.yaml`) run over a broad distribution of target-pitch/target-roll setpoints and initial conditions, logged to parquet. Both learned controllers are pretrained with supervised regression against the PID's action, offline, before any online loop runs. RTRL-RTU's warm-start uses full BPTT (offline, so the O(N^4) cost doesn't matter for a short warm-start horizon) to get a good starting point for `W_in`, `a`, `W_out` — online RTRL only takes over after deployment. This is why cold-start online RTRL is out of scope (see `rtrl.md`).

## Scenario 1 — Nominal

Calm air, no faults, held setpoints across the full training-distribution range (and a held-out range for generalization). **Acceptance bar:** all three controllers track within ~15% RMSE of PID. This scenario exists to prove RTRL/BPTT-LSTM don't regress the easy case, not to differentiate them — if RTRL wins big here, be suspicious of the setup before celebrating.

## Scenario 2 — Fault (the headline result)

One control surface (aileron, elevator, or rudder — three independent sub-runs, not combined) has its effectiveness scaled down by `FaultWrapper` at a random step within a configured mid-episode window, and stays degraded for the rest of the episode. Severity levels: configured as a sweep (e.g. 40%, 60%, 80% effectiveness loss) — see `configs/env/fault.yaml`.

**What "done" looks like:** RTRL-RTU's post-fault tracking RMSE recovers toward pre-fault levels within the time-to-recover metric (`evaluation.md`); frozen BPTT-LSTM's does not (stays degraded for the rest of the episode, since its weights can't change). PID may or may not recover depending on whether its gains were tuned assuming full effectiveness — report it either way, don't tune PID specifically to make this comparison look better or worse.

## Scenario 3 — Wind

Dryden turbulence injected via `WindWrapper` at three severity levels, **unseen during warm-start** (warm-start data is calm-air only, deliberately, so this tests generalization, not memorization). No faults active. Tests whether online adaptation (or just RTU architecture) helps with a disturbance type the network was never trained against.

## Scenario 4 — Combined (fault + wind)

`WindWrapper(FaultWrapper(env))` — mid-episode fault on top of ambient turbulence, at moderate severity for both (not the worst-case of each, since that's likely unrecoverable by any controller and uninformative). This is the demo scenario — the one that best resembles "real degraded flight." Only run once faults (scenario 2) show a clean recovery signal; don't debug the combined case before the isolated fault case works.

## Scenario 5 — Ablation (online-update-disabled)

Identical to Scenario 2 (fault), but RTRL-RTU is deployed with `online: false` in its config — same warm-started weights, `controller.update()` is simply never called by the runner (no branching needed, see `architecture.md`). Everything else (seeds, fault severities, episode setup) is held identical to Scenario 2 so the two runs are directly comparable.

**This is the paper's central causal claim.** The delta between Scenario 2 (online-on) and Scenario 5 (online-off, same architecture) isolates *online adaptation* as the mechanism, ruling out "RTUs just happen to be a more robust architecture even frozen" as an alternative explanation for any recovery seen in Scenario 2.

## What each run's output dir must contain

Per CLAUDE.md's reproducibility requirement: Hydra config snapshot, seed, git SHA, W&B run ID, and the parquet trace(s). If any of these is missing, the run doesn't count as evidence for the scenario's acceptance bar — rerun it.

## Statistical protocol, metrics definitions, "recovered" threshold

Lives in `evaluation.md` — kept separate so the scenario *design* here doesn't get tangled with the scenario *analysis*.
