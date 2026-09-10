# Project Overview

## What we're building

A flight-surface controller for a fixed-wing aircraft (Cessna 172, simulated in JSBSim via `jsbgym`) that maps aircraft state to aileron/elevator/rudder deflection commands, holding a target pitch and target roll (attitude hold — no trajectory or heading tracking).

The controller is a small recurrent neural net trained with **Real-Time Recurrent Learning (RTRL)**, using **Recurrent Trace Units (RTUs)** to make exact (non-truncated, non-approximated) RTRL computationally tractable online, in real time, during flight.

## Why this exists

Standard practice for learned flight controllers is: train offline (imitation learning, BPTT, or RL), freeze the weights, deploy. When the airframe degrades mid-flight — a stuck or reduced-effectiveness control surface, e.g. from ice, battle damage, or actuator wear — a frozen network cannot adapt. It only knows what BPTT saw on the training distribution.

RTRL computes exact gradients online, one timestep at a time, with no replay buffer and no unrolled backward pass through history. In principle this lets weights keep adapting *during* the episode that damages the airframe, not just across episodes. Historically RTRL was abandoned for anything but tiny networks because its exact cost is O(N^4) in the number of hidden units. RTUs sidestep this: a diagonal (elementwise) linear recurrence gives a diagonal Jacobian, so the sensitivity update collapses to O(N) per unit, making exact RTRL O(N^2) total — the same order as a dense forward pass. See `rtrl.md` for the derivation and `research_context.md` for how this differs from prior tractable-RTRL work.

## Success criteria (in priority order)

1. **The fault scenario is the paper.** RTRL must measurably recover tracking performance after a mid-episode actuator fault (reduced control effectiveness on aileron, elevator, or rudder, tested independently) where a frozen BPTT-LSTM does not. "Recovery" is defined quantitatively in `evaluation.md` (time-to-recover, post-fault RMSE delta).
2. **The sensitivity test is the floor, not a nice-to-have.** `tests/test_rtrl_sensitivity.py` must pass to ~1e-5 against finite differences before any training result is trusted. A controller that "looks adaptive" without a passing sensitivity test is not evidence of anything.
3. **The ablation isolates the online-update mechanism.** RTRL with `controller.update()` disabled at deploy time (weights frozen after warm-start) vs. online-on, same architecture, same warm-start. The delta between these two is the causal claim: online adaptation, not just RTU architecture, drives recovery.
4. **Nominal and wind scenarios are supporting evidence**, not the headline: nominal shows RTRL doesn't regress baseline tracking when nothing goes wrong; wind (Dryden turbulence, unseen severities) shows the adaptation generalizes beyond the one fault mode we trained toward.

If time runs out, protect in this order: fault > ablation > nominal > wind > combined. See "Scope discipline" in `CLAUDE.md`.

## Non-goals

Explicitly out of scope (see CLAUDE.md "What NOT to do" for the enforced list): other airframes, trajectory/heading tracking, PPO/SAC baselines, physics-informed losses, RTRL approximations (Kronecker factoring, sparsity tricks) — RTUs already make *exact* RTRL tractable, so approximating it defeats the point.

## Timeline shape (14 days)

Rough phase breakdown — not a rigid schedule, but the dependency order is real:

1. Env + wrappers + PID baseline working end-to-end (trace logging included) — everything downstream depends on this.
2. RTU cell + forward sensitivity + sensitivity test green — the correctness bar in CLAUDE.md.
3. Imitation warm-start pipeline (both BPTT-LSTM and RTRL-RTU need this; cold-start online RTRL diverges — known, not a bug).
4. BPTT-LSTM baseline trained and evaluated on nominal.
5. Online RTRL loop (`controller.update()` wired into the runner) trained and evaluated on nominal.
6. Fault scenario — the headline result. Iterate here until the recovery signal is real and reproducible.
7. Ablation (online-off) — cheap once RTRL/fault work, but don't skip it; it's the causal claim.
8. Wind, then combined, time permitting.
9. Analysis pass: parquet → plots → statistical protocol per `evaluation.md`.

## Where the rest of the context lives

- `architecture.md` — system diagram, the `Controller` ABC, wrapper composition, trace schema
- `rtrl.md` — RTU cell definition, forward sensitivity recursion, the online update rule
- `experiments.md` — the 4 scenarios + ablation in full parametric detail
- `research_context.md` — what's novel here vs. Irie et al. (ICLR 2024), Elelimy et al. (NeurIPS 2024), and prior tractable-RTRL work
- `evaluation.md` — metrics, statistical protocol, what "recovered" means numerically
- `environment.md` — jsbgym/JSBSim setup, observation/action space, known env quirks
