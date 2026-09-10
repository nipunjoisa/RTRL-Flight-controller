# Architecture

## System diagram

```
                         ┌─────────────────────────────────────────┐
                         │              Hydra config                │
                         │  configs/{controller,env,experiment}/*   │
                         └───────────────────┬───────────────────────┘
                                              │
                                              v
scripts/run_experiment.py  ──builds──>  ExperimentRunner
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              v                                v                               v
      env: gymnasium.Env             controller: Controller ABC        TraceLogger
      WindWrapper(                   PID | BPTT-LSTM | RTRL-RTU        (per-step dict -> parquet)
        FaultWrapper(
          NormalizeWrapper(
            JSBSimEnv (jsbgym)
          )))

Runner loop, per step:
  obs            = env.step(prev_action) -> (obs, reward, terminated, truncated, info)
  action         = controller.act(obs)
  controller.update(obs, action, reward, next_obs)   # no-op for PID/BPTT-LSTM
  trace_logger.log(obs, action, info, controller-internal metrics)

  on episode end: controller.reset(); env.reset()
```

## The `Controller` ABC

Location: `src/rtrl_flight/controllers/base.py`. All three controllers (PID, BPTT-LSTM, RTRL-RTU) implement this exactly. **The experiment runner never does `isinstance` checks on controller type** — any controller-specific behavior (e.g. "does this controller adapt online") is a property/flag on the ABC, not a type check in the runner.

```python
class Controller(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs -> action (aileron, elevator, rudder), each in [-1, 1]."""

    @abstractmethod
    def update(self, obs, action, reward, next_obs) -> dict:
        """Online weight update. No-op for PID/BPTT-LSTM (return {}).
        Returns a dict of scalars (e.g. online loss, sensitivity norm) for trace logging."""

    @abstractmethod
    def reset(self) -> None:
        """Clear episode-local state (hidden state, sensitivity tensors). Does NOT reset weights."""

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...
```

`update` is called every step for every controller — PID and BPTT-LSTM implement it as a no-op that returns `{}`. This keeps the runner loop identical across controllers (design invariant #3 in CLAUDE.md) and means the ablation (RTRL with online updates disabled) is just "call `act`, skip `update`" — no branching needed in the runner, only in how the RTRL-RTU controller is constructed (an `online: bool` flag in its Hydra config).

## Env wrapper stack

Base env (`src/rtrl_flight/env/jsbsim_env.py`) wraps `jsbgym` and exposes a fixed observation/action space for attitude hold (see `environment.md` for the exact vector layout). Everything else is a `gymnasium.Wrapper`:

- `FaultWrapper` — at a configured step (or step range), scales one surface's effective deflection by a `severity` factor. Independent per surface; combined-fault is a separate experiment config, not a wrapper change.
- `WindWrapper` — injects Dryden turbulence into the effective airflow the base env sees, at a configured severity level.
- `NormalizeWrapper` — running or fixed obs/action normalization. Must not leak statistics across episodes unless explicitly configured to (running stats are a deliberate choice, documented in the wrapper's docstring, not an accident).
- `TraceWrapper` — the thin layer that hands per-step data to `TraceLogger`; kept separate from the runner so traces can be captured even from ad-hoc scripts/notebooks exercising the env directly.

**Composability is load-bearing** (design invariant #2): `WindWrapper(FaultWrapper(env))` and `FaultWrapper(WindWrapper(env))` must both work, and the combined-fault-and-wind scenario depends on this. Each wrapper's `reset()` must clear its own episode-local state (active fault, current turbulence seed) — no wrapper may depend on another wrapper's reset order.

## Trace schema

`TraceLogger` (`src/rtrl_flight/metrics/tracing.py`) writes one row per env step to a buffered writer, flushed to parquet at episode end. Columns (minimum set — controllers may add more via their `update()` return dict, which gets merged in with a `controller/` prefix):

| column | meaning |
|---|---|
| `episode_id`, `step` | run identity |
| `obs_*` | flattened observation vector |
| `action_*` | commanded deflections |
| `true_action_*` | actual post-fault effective deflection (differs from `action_*` only when a fault is active) |
| `target_pitch`, `target_roll` | attitude-hold setpoints for this episode |
| `reward` | task reward (tracking error based) |
| `fault_active`, `fault_surface`, `fault_severity` | from `FaultWrapper`, `NaN`/`None` when inactive |
| `wind_severity` | from `WindWrapper` |
| `controller/*` | controller-specific (e.g. `controller/online_loss`, `controller/sensitivity_norm` for RTRL-RTU) |

**Analysis code only ever reads parquet** (design invariant #4) — never live env/controller state, never W&B aggregates alone (see "Where to look when stuck" in CLAUDE.md). This is what makes a run reproducible after the fact from its Hydra output dir.

## Directory-to-concept map

| concept | lives in |
|---|---|
| RTU cell, sensitivity, online update | `src/rtrl_flight/rtrl/` |
| Controller implementations (wrap RTU/LSTM/PID into the ABC) | `src/rtrl_flight/controllers/` |
| Env + wrappers | `src/rtrl_flight/env/` |
| Imitation / BPTT / online-RTRL training loops | `src/rtrl_flight/training/` |
| Trace logging, smoothness/recovery metrics | `src/rtrl_flight/metrics/` |
| parquet → DataFrame → plots | `src/rtrl_flight/analysis/` |
| CLI entrypoints (thin, no logic) | `scripts/` |
| Hydra configs | `configs/{controller,env,experiment}/` |
