# Environment

## jsbgym / JSBSim

`jsbgym` wraps JSBSim (the flight dynamics model) as a Gymnasium env. **Not yet installed in this environment** — first task when building `src/rtrl_flight/env/jsbsim_env.py` is `uv add jsbgym`, confirm it exposes a Cessna 172 aircraft model, and confirm the exact observation/action keys it provides. Everything below is the *target* spec for our wrapper; verify each field against the installed package and correct this doc if it drifts (docs are source of truth for intent, but intent has to match what the library actually exposes — if they disagree, fix the doc, don't silently code around it, per CLAUDE.md's "ask which is right" rule).

## Observation space (target)

Flattened vector, attitude-hold-relevant subset of JSBSim state (exact indices to be pinned down in `jsbsim_env.py` and documented in its docstring once implemented):

- Attitude: pitch, roll, (yaw available but not tracked — attitude hold is pitch+roll only)
- Attitude rates: pitch rate (q), roll rate (p), yaw rate (r)
- Airspeed, angle of attack, sideslip
- Altitude (context, not a tracked target)
- Target pitch, target roll (the episode's setpoint — part of the observation so the controller is conditioned on it, not just the raw state)
- Previous action (aileron, elevator, rudder) — included so the controller can learn control-rate-aware behavior

Normalization: handled by `NormalizeWrapper`, not baked into the base env — keeps the base env's raw output inspectable for debugging.

## Action space (target)

`[aileron, elevator, rudder]`, each in `[-1, 1]`, mapped to JSBSim's control surface deflection range internally. Throttle is **not** part of the action space — attitude hold, not full flight control; hold throttle fixed per episode (a config value, not learned).

## Episode structure

- Reset: sample initial trim condition (airspeed, altitude within a safe band) and a target pitch/target roll setpoint. Fault/wind wrappers also reset their episode-local state here (see `architecture.md`).
- Step: JSBSim advances at its native integration rate; the Gym `step()` likely runs several JSBSim ticks per env step (frame-skip / agent-rate vs. sim-rate split) — confirm jsbgym's default and whether it's configurable, since this affects both wall-clock training time and how "mid-episode" fault-onset windows are specified in steps vs. seconds.
- Termination: attitude-hold task uses truncation (fixed episode length) rather than early termination on large tracking error, per `experiments.md` — early termination would cut off exactly the post-fault recovery window the fault scenario needs to observe. If JSBSim's default env terminates early on extreme states (e.g. stall, spin, altitude floor), that termination condition must stay wide enough not to truncate a struggling-but-recovering controller — tune the safety bounds, don't rely on JSBSim's defaults uncritically.

## Reward

Tracking-error based (e.g. negative weighted L2 of `(pitch - target_pitch, roll - target_roll)`, possibly with a smoothness/action-rate penalty term). Used for the RL-style framing where relevant (e.g. if online RTRL's loss is framed as a reward-derived error signal rather than pure supervised imitation error) — exact functional form belongs in `configs/env/*.yaml`, not hardcoded, and changing it is a "what to ask first" item per CLAUDE.md since it invalidates prior runs.

## Known quirks / things to verify early (fill in as discovered)

- JSBSim aircraft config file location and whether `jsbgym` ships a ready Cessna 172 model or requires pointing at JSBSim's own aircraft XML.
- Whether jsbgym's Gymnasium API is 0.29-compatible out of the box (`terminated`/`truncated` split) or needs an adapter.
- Sim determinism given a fixed seed — needed for the reproducibility requirement in CLAUDE.md (same seed + config must reproduce the same trace).

This section is expected to grow during the first implementation phase — that's fine, update it rather than letting undocumented tribal knowledge accumulate in code comments only.
