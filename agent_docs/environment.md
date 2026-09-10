# Environment

## jsbgym / JSBSim

`jsbgym` wraps JSBSim (the flight dynamics model) as a Gymnasium env. **Installed and probed** (`scripts/probe_jsbgym.py`, throwaway, not part of `src/`) — `jsbgym==0.4.3` / `jsbsim==1.3.1`. It ships a Cessna 172 aircraft model and many others, registered as `gym.make()` IDs enumerated in `jsbgym.Envs`. Everything below reflects the *real* probed API, corrected from the original pre-install target spec.

**Confirmed by the probe: `jsbgym` registers only `HeadingControlTask` and `TurnHeadingControlTask` — no attitude-hold task exists.** *(# updated after jsbgym probe — original spec said the base env would directly expose an attitude-hold observation/action layout, implicitly assuming a ready-made task)* We will subclass `jsbgym.tasks.Task` to create `AttitudeHoldTask`, built in `src/rtrl_flight/env/jsbsim_env.py`. It will:

- Inject two target properties, `target/pitch-rad` and `target/roll-rad`, sampled at episode reset (same pattern `HeadingControlTask` uses for its own target-heading property) — these condition the episode's setpoint and are used internally to compute the error observations below. They are not separate observation-vector slots (see obs layout — the vector shape is kept identical to `HeadingControlTask`'s, so no dimensionality change).
- Replace `HeadingControlTask`'s two error slots (indices 9-10: `error/altitude-error-ft`, `error/track-error-deg`) with our own computed `error/pitch-error-rad` and `error/roll-error-rad` (`= attitude/pitch-rad - target/pitch-rad`, `= attitude/roll-rad - target/roll-rad`).

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
| 9 | `error/pitch-error-rad` (custom, `AttitudeHoldTask`) | `pitch - target_pitch`, replaces `HeadingControlTask`'s `error/altitude-error-ft` | derived, ~[-3.1416, 3.1416] |
| 10 | `error/roll-error-rad` (custom, `AttitudeHoldTask`) | `roll - target_roll`, replaces `HeadingControlTask`'s `error/track-error-deg` | derived, ~[-6.2832, 6.2832] |

Indices 0-8 are taken verbatim from `HeadingControlTask`'s `state_variables` (confirmed identical for our custom task, since we subclass rather than rebuild). Indices 9-10 are repurposed as described above — same 11-dim shape as the probed `HeadingControlTask` observation, different semantics for the last two slots.

*(# updated after jsbgym probe — original spec described slots 3-5 as "Airspeed, angle of attack, sideslip")* **Correction:** slots 3-5 are the raw body-frame velocity triple (`u`, `v`, `w` in ft/s), not a scalar airspeed/AoA/sideslip decomposition. True airspeed and angle of attack are derivable from `(u, v, w)` (e.g. `airspeed = sqrt(u^2+v^2+w^2)`, `alpha = atan2(w, u)`) but are **not** separate properties in this observation — we keep the raw 3-axis form as-is and do not add derived properties to the base obs. If a derived scalar turns out to be needed later (e.g. for reward shaping), that's a "what to ask first" change per CLAUDE.md, not a silent addition here.

Previous action (aileron, elevator, rudder) — still planned, per the original spec, as an addition on top of this 11-dim base vector if the controller needs control-rate awareness; not yet probed since it's assembled by our wrapper, not by `jsbgym` itself.

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

Tracking-error based (e.g. negative weighted L2 of `(pitch - target_pitch, roll - target_roll)`, possibly with a smoothness/action-rate penalty term) — unchanged from original spec, not yet built for `AttitudeHoldTask`. Probed `HeadingControlTask`'s built-in reward returns a plain float from `step()` (e.g. `0.983`) plus a richer `jsbgym.rewards.Reward` object in `info["reward"]`; confirms the shape we should target for `AttitudeHoldTask`'s reward, though the altitude/heading-shaped reward itself is not reusable as-is. Exact functional form belongs in `configs/env/*.yaml`, not hardcoded, and changing it is a "what to ask first" item per CLAUDE.md since it invalidates prior runs.

## Known quirks / things to verify early

- ~~JSBSim aircraft config file location and whether `jsbgym` ships a ready Cessna 172 model or requires pointing at JSBSim's own aircraft XML.~~ **Resolved by probe:** `jsbgym` ships a ready C172 model, registered directly via `gym.make(jsbgym.Envs.C172_HeadingControlTask_Shaping_STANDARD_NoFG_v0.value)`; no manual XML pointing needed.
- ~~Whether jsbgym's Gymnasium API is 0.29-compatible out of the box (`terminated`/`truncated` split) or needs an adapter.~~ **Resolved by probe:** installed Gymnasium is **1.3.0**, not 0.29+. *(# updated after jsbgym probe — original spec/CLAUDE.md said "Gymnasium 0.29+")* The 1.x `step()`/`reset()` API (5-tuple step, `terminated`/`truncated` split, `seed=` kwarg on `reset()`) is confirmed working with no adapter needed.
- Sim determinism given a fixed seed — needed for the reproducibility requirement in CLAUDE.md (same seed + config must reproduce the same trace). Probed `reset(seed=0)` produced a consistent initial state; full determinism across a longer rollout not yet stress-tested.
- `AttitudeHoldTask` itself (subclassing `jsbgym.tasks.Task`, injecting target properties, computing pitch/roll error) is not yet implemented — this doc specifies the target design, `src/rtrl_flight/env/jsbsim_env.py` is where it gets built and where any further drift from this spec gets corrected.

This section is expected to grow during the first implementation phase — that's fine, update it rather than letting undocumented tribal knowledge accumulate in code comments only.
