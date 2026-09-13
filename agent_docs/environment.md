# Environment

## jsbgym / JSBSim

`jsbgym` wraps JSBSim (the flight dynamics model) as a Gymnasium env. **Installed and probed** (`scripts/probe_jsbgym.py`, throwaway, not part of `src/`) — `jsbgym==0.4.3` / `jsbsim==1.3.1`. It ships a Cessna 172 aircraft model and many others, registered as `gym.make()` IDs enumerated in `jsbgym.Envs`. Everything below reflects the *real* probed API, corrected from the original pre-install target spec.

**Confirmed by the probe: `jsbgym` registers only `HeadingControlTask` and `TurnHeadingControlTask` — no attitude-hold task exists.** *(# updated after jsbgym probe — original spec said the base env would directly expose an attitude-hold observation/action layout, implicitly assuming a ready-made task)* `AttitudeHoldTask` subclasses `jsbgym.tasks.FlightTask` (a `Task` subclass; see `architecture.md` for why `FlightTask` specifically) and is **built and in use**, at `src/rtrl_flight/env/attitude_task.py` (# updated at final health-check — this doc previously said "not yet implemented" at a `jsbsim_env.py` path that was never actually used). It:

- Injects two target properties, `target/pitch-rad` and `target/roll-rad`, sampled at episode reset (same pattern `HeadingControlTask` uses for its own target-heading property) — these condition the episode's setpoint and are used internally to compute the error observations below. They are not separate observation-vector slots (see obs layout — the vector shape is kept identical to `HeadingControlTask`'s, so no dimensionality change).
- Replaces `HeadingControlTask`'s two error slots (indices 9-10: `error/altitude-error-ft`, `error/track-error-deg`) with our own computed `error/pitch-error-rad` and `error/roll-error-rad`.
  **Sign, corrected (# updated at final health-check): `= target/pitch-rad - attitude/pitch-rad` and `= target/roll-rad - attitude/roll-rad` (target MINUS current, not current minus target as this doc previously said)** — confirmed against `attitude_task.py`'s `_update_errors()`. This sign is exactly what the PID-retuning session's sign bug was about: a positive error means "need more of that axis," and `PIDController` initially had elevator's sign backwards relative to this convention (see `configs/controller/pid.yaml`'s comment for the full story) — gain magnitude alone couldn't fix a sign mismatch against this convention.

## Observation space (confirmed via probe, 11-dim, `Box(float64)`)

*(# updated after jsbgym probe — original spec was a bulleted description of intended categories, not confirmed property names/indices)*

| idx | property | meaning | bounds |
|---|---|---|---|
| 0 | `position/h-sl-ft` | altitude above mean sea level (context only, not a tracked target) | [-1400, 85000] |
| 1 | `attitude/pitch-rad` | pitch | [-1.5708, 1.5708] |
| 2 | `attitude/roll-rad` | roll | [-3.1416, 3.1416] |
| 3 | `velocities/u-fps` | body-frame x-axis velocity | [-2200, 2200] |
| 4 | `velocities/v-fps` | body-frame y-axis velocity | [-2200, 2200] |
| 5 | `velocities/w-fps` | body-frame z-axis velocity | [-2200, 2200] |
| 6 | `velocities/p-rad_sec` | roll rate | [-6.2832, 6.2832] |
| 7 | `velocities/q-rad_sec` | pitch rate | [-6.2832, 6.2832] |
| 8 | `velocities/r-rad_sec` | yaw rate | [-6.2832, 6.2832] |
| 9 | `error/pitch-error-rad` (custom, `AttitudeHoldTask`) | `target_pitch - pitch` (# updated at final health-check: sign was previously documented backwards), replaces `HeadingControlTask`'s `error/altitude-error-ft` | derived, ~[-3.1416, 3.1416] |
| 10 | `error/roll-error-rad` (custom, `AttitudeHoldTask`) | `target_roll - roll` (# updated at final health-check: sign was previously documented backwards), replaces `HeadingControlTask`'s `error/track-error-deg` | derived, ~[-6.2832, 6.2832] |

Indices 0-8 are taken verbatim from `HeadingControlTask`'s `state_variables` (confirmed identical for our custom task, since we subclass rather than rebuild). Indices 9-10 are repurposed as described above — same 11-dim shape as the probed `HeadingControlTask` observation, different semantics for the last two slots.

*(# updated after jsbgym probe — original spec described slots 3-5 as "Airspeed, angle of attack, sideslip")* **Correction:** slots 3-5 are the raw body-frame velocity triple (`u`, `v`, `w` in ft/s), not a scalar airspeed/AoA/sideslip decomposition. True airspeed and angle of attack are derivable from `(u, v, w)` (e.g. `airspeed = sqrt(u^2+v^2+w^2)`, `alpha = atan2(w, u)`) but are **not** separate properties in this observation — we keep the raw 3-axis form as-is and do not add derived properties to the base obs. If a derived scalar turns out to be needed later (e.g. for reward shaping), that's a "what to ask first" change per CLAUDE.md, not a silent addition here.

Previous action (aileron, elevator, rudder): **decided against** (# updated at final health-check — original spec floated this as a possible addition). The obs vector stayed at 11 dims throughout every controller built (`INPUT_DIM = 11` in `rtrl_flight.controllers.bptt_lstm`, reused by `rtrl_rtu.py`) — control-rate awareness wasn't needed in practice, and adding it now would be an observation-space change, a "what to ask first" item per CLAUDE.md.

Normalization: handled by `NormalizeWrapper`, not baked into the base env — keeps the base env's raw output inspectable for debugging (unchanged from original spec).

## Action space (confirmed via probe, matches original target spec exactly)

`Box(-1.0, 1.0, shape=(3,), float64)`:

| idx | property | meaning |
|---|---|---|
| 0 | `fcs/aileron-cmd-norm` | aileron commanded position, normalized |
| 1 | `fcs/elevator-cmd-norm` | elevator commanded position, normalized |
| 2 | `fcs/rudder-cmd-norm` | rudder commanded position, normalized |

Throttle is **not** part of the action space, confirmed — attitude hold, not full flight control; hold throttle fixed per episode (a config value, not learned).

## Episode structure

- Reset: sample initial trim condition (airspeed, altitude within a safe band) and a target pitch/target roll setpoint (`target/pitch-rad`, `target/roll-rad` on `AttitudeHoldTask`). Fault/wind wrappers also reset their episode-local state here (see `architecture.md`). Probed `env.reset(seed=0)` on `HeadingControlTask` behaved deterministically and returned `(obs, info)` — no surprises to carry over to the custom task.
- Step: JSBSim advances at its native integration rate; probed `env.step()` returns the standard 5-tuple (`obs, reward, terminated, truncated, info`) without needing an adapter (see Gymnasium version note below). Frame-skip / agent-rate vs. sim-rate split still needs pinning down for `AttitudeHoldTask` specifically once it's built.
- Termination: attitude-hold task uses truncation (fixed episode length) rather than early termination on large tracking error, per `experiments.md` — early termination would cut off exactly the post-fault recovery window the fault scenario needs to observe. Probed `HeadingControlTask` did not terminate early over 5 random-action steps; `AttitudeHoldTask`'s own safety-bound termination conditions (stall, spin, altitude floor) still need to be set wide enough not to truncate a struggling-but-recovering controller — tune deliberately, don't inherit `HeadingControlTask`'s defaults uncritically.

## Reward

**Built, at `src/rtrl_flight/env/reward.py`'s `attitude_tracking_reward()`** (# updated at final health-check — original spec said "not yet built"): negative weighted sum of squared pitch/roll error, squared body rates (p, q, r), and squared action magnitude --
`-(w_pitch*pitch_error^2 + w_roll*roll_error^2 + w_rate*(p^2+q^2+r^2) + w_action*sum(action_i^2))`,
defaults `w_pitch=w_roll=1.0`, `w_rate=0.1`, `w_action=0.01` (module-level constants, not yet promoted to `configs/env/*.yaml` — still hardcoded in `reward.py`, which is itself a "what to ask first" item per CLAUDE.md if changed, since it invalidates prior runs). Not the altitude/heading-shaped reward `HeadingControlTask` ships with; this is a from-scratch attitude-hold-specific function.

## Known quirks / things to verify early

- ~~JSBSim aircraft config file location and whether `jsbgym` ships a ready Cessna 172 model or requires pointing at JSBSim's own aircraft XML.~~ **Resolved by probe:** `jsbgym` ships a ready C172 model, registered directly via `gym.make(jsbgym.Envs.C172_HeadingControlTask_Shaping_STANDARD_NoFG_v0.value)`; no manual XML pointing needed.
- ~~Whether jsbgym's Gymnasium API is 0.29-compatible out of the box (`terminated`/`truncated` split) or needs an adapter.~~ **Resolved by probe:** installed Gymnasium is **1.3.0**, not 0.29+. *(# updated after jsbgym probe — original spec/CLAUDE.md said "Gymnasium 0.29+")* The 1.x `step()`/`reset()` API (5-tuple step, `terminated`/`truncated` split, `seed=` kwarg on `reset()`) is confirmed working with no adapter needed.
- **Sim determinism given a fixed seed — confirmed broken, then worked around (# updated at final health-check).** `env.reset(seed=...)` alone does NOT reproduce a specific episode: `AttitudeHoldTask._new_episode_init` samples `target_pitch_rad`/`target_roll_rad` via Python's global `random.uniform`, not gymnasium's seeded `self.np_random` -- found while wiring up `scripts/run_sweep.py`'s multi-seed sweep (seeds were silently producing whatever randomness happened to occur, not a specific reproducible episode). Worked around at the call site, not fixed in the task itself: `rtrl_flight.runner.run_one_episode(..., seed=...)` calls `random.seed(seed)` immediately before `env.reset(seed=seed)`. `AttitudeHoldTask` itself still doesn't use `self.np_random` for its own sampling -- a real fix (routing `_new_episode_init`'s `random.uniform` calls through the task's own seeded RNG) is still open if this env is ever driven from somewhere that doesn't go through `run_one_episode`.
- ~~`AttitudeHoldTask` itself (subclassing `jsbgym.tasks.Task`, injecting target properties, computing pitch/roll error) is not yet implemented.~~ **Built**, at `src/rtrl_flight/env/attitude_task.py`, subclassing `jsbgym.tasks.FlightTask` (see `architecture.md`'s note on why `FlightTask` rather than raw `Task`). Its own safety-bound termination (stall/spin/altitude-floor) is **still not implemented** -- this is a real, observed gap, not just a theoretical one: online RTRL training runs (`scripts/train_rtrl_warmstart.py`) hit `reward=nan` mid-episode from an uncaught flight-dynamics divergence more than once. `RTRLRTUController.update()` guards against the *consequence* (skips the gradient step on a non-finite reward, agent_docs/rtrl.md's "online update" section), but the *env* still has no termination condition to end the episode cleanly when this happens.

This section is expected to grow during the first implementation phase — that's fine, update it rather than letting undocumented tribal knowledge accumulate in code comments only.
