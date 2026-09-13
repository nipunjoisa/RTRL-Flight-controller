"""Live matplotlib dashboard -- a demo tool, NOT part of the training loop
(no CLAUDE.md/agent_docs section describes this dashboard as of writing;
built entirely from this file's own spec). Runs one episode interactively,
lets you trigger faults / cycle wind / reset from the keyboard, and plots
attitude tracking, control surfaces, and RTRL diagnostics live.

Plain argparse, no Hydra -- this is meant to be launched ad hoc from a
terminal, not composed into an experiment sweep.

Usage:
  python scripts/live_demo.py --controller rtrl_rtu --env cessna172_fault

Keys (while the plot window has focus):
  f       trigger_fault('aileron', 0.5)   (no-op + console note if this env
  e       trigger_fault('elevator', 0.5)   has no FaultWrapper, e.g. nominal)
  r       trigger_fault('rudder', 0.5)
  w       cycle wind off -> light -> moderate -> severe -> off (same caveat
          for envs with no WindWrapper)
  space   reset the episode (controller weights are untouched --
          reset() only clears episode-local state, per the Controller ABC)
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from rtrl_flight.env.make import force_raw_obs, make_env
from rtrl_flight.env.wrappers.fault import FaultWrapper
from rtrl_flight.env.wrappers.wind import WindWrapper
from rtrl_flight.runner import build_controller, load_checkpoint_if_present

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
HISTORY_LEN = 200
SMOOTHING_WINDOW = 10
WIND_CYCLE = ("off", "light", "moderate", "severe")


def _available_names(subdir: str) -> list[str]:
    return sorted(p.stem for p in (CONFIGS_DIR / subdir).glob("*.yaml"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True, choices=_available_names("controller"))
    parser.add_argument("--env", required=True, choices=_available_names("env"))
    return parser.parse_args()


def load_yaml_cfg(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def find_wrapper(env: Any, wrapper_type: type) -> Any | None:
    """Same wrapper-stack walk as rtrl_flight.runner._find_wrapper --
    duplicated locally rather than imported since it's a 5-line, dependency-
    free utility and this script deliberately avoids pulling in more of the
    training-side machinery than build_controller/load_checkpoint_if_present.
    """
    node = env
    while hasattr(node, "env"):
        if isinstance(node, wrapper_type):
            return node
        node = node.env
    return None


def get_targets(env: Any) -> tuple[float, float]:
    """AttitudeHoldTask's target_pitch_rad/target_roll_rad are internal sim
    properties, not separate obs slots (agent_docs/environment.md) -- read
    them directly off the sim, same as the task itself does.
    """
    task = env.unwrapped.task
    sim = env.unwrapped.sim
    return sim[task.target_pitch_rad], sim[task.target_roll_rad]


def rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (x_offsets, smoothed) for a simple trailing moving average,
    window shrinking near the start of the buffer so a plot is produced
    from the very first step rather than only after `window` samples exist.
    """
    n = len(values)
    if n == 0:
        return np.array([]), np.array([])
    smoothed = np.array([values[max(0, i - window + 1) : i + 1].mean() for i in range(n)])
    return np.arange(n), smoothed


class DemoState:
    """Rolling history buffers (last HISTORY_LEN steps) for the dashboard."""

    def __init__(self) -> None:
        self.reset_requested = False
        self._new_episode()

    def _new_episode(self) -> None:
        self.step_count = 0
        self.episode_return = 0.0
        self.steps: deque[int] = deque(maxlen=HISTORY_LEN)
        self.pitch: deque[float] = deque(maxlen=HISTORY_LEN)
        self.roll: deque[float] = deque(maxlen=HISTORY_LEN)
        self.target_pitch: deque[float] = deque(maxlen=HISTORY_LEN)
        self.target_roll: deque[float] = deque(maxlen=HISTORY_LEN)
        self.aileron: deque[float] = deque(maxlen=HISTORY_LEN)
        self.elevator: deque[float] = deque(maxlen=HISTORY_LEN)
        self.rudder: deque[float] = deque(maxlen=HISTORY_LEN)
        self.online_loss: deque[float] = deque(maxlen=HISTORY_LEN)
        self.sensitivity_norm: deque[float] = deque(maxlen=HISTORY_LEN)

    def record(
        self,
        pitch: float,
        roll: float,
        target_pitch: float,
        target_roll: float,
        action: np.ndarray,
        online_loss: float,
        sensitivity_norm: float,
    ) -> None:
        self.steps.append(self.step_count)
        self.pitch.append(pitch)
        self.roll.append(roll)
        self.target_pitch.append(target_pitch)
        self.target_roll.append(target_roll)
        self.aileron.append(float(action[0]))
        self.elevator.append(float(action[1]))
        self.rudder.append(float(action[2]))
        self.online_loss.append(online_loss)
        self.sensitivity_norm.append(sensitivity_norm)
        self.step_count += 1


def get_online_loss(update_result: dict[str, float]) -> float:
    return update_result.get("online_loss", 0.0)


def get_sensitivity_norm(controller: Any) -> float:
    """Duck-typed, not an isinstance check: only RTRLRTUController defines
    sensitivity_frobenius_norm(); PID/BPTT-LSTM naturally plot flat 0 lines
    with no controller-type branching anywhere in this script.
    """
    fn = getattr(controller, "sensitivity_frobenius_norm", None)
    return float(fn()) if fn is not None else 0.0


def format_fault_line(configured_fault: dict[str, Any] | None, fault_wrapper: Any | None) -> str:
    if fault_wrapper is None:
        return "Fault: (no FaultWrapper for this env)"
    active = {s: sev for s, sev in fault_wrapper.fault_state.items() if sev < 1.0}
    if active:
        parts = ", ".join(f"{s} @ {int(sev * 100)}%" for s, sev in active.items())
        return f"Fault: {parts}  [ACTIVE]"
    surface = (configured_fault or {}).get("surface", "none")
    severity = (configured_fault or {}).get("severity", 1.0)
    return f"Fault: {surface} @ {int(severity * 100)}%  [inactive]"


def format_wind_line(wind_wrapper: Any | None) -> str:
    if wind_wrapper is None:
        return "Wind: (no WindWrapper for this env)"
    return f"Wind: {wind_wrapper.wind_level}"


def redraw(
    fig: plt.Figure,
    axes: tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes],
    state: DemoState,
    controller_name: str,
    configured_fault: dict[str, Any] | None,
    fault_wrapper: Any | None,
    wind_wrapper: Any | None,
    max_steps: int,
) -> None:
    ax_attitude, ax_controls, ax_diag, ax_text = axes

    steps = np.array(state.steps)

    ax_attitude.clear()
    if len(steps) > 0:
        ax_attitude.plot(steps, state.target_pitch, "b--", label="target pitch")
        ax_attitude.plot(steps, state.pitch, "b-", label="pitch")
        ax_attitude.plot(steps, state.target_roll, "r--", label="target roll")
        ax_attitude.plot(steps, state.roll, "r-", label="roll")
        ax_attitude.legend(loc="upper left", fontsize=8)
    ax_attitude.set_ylim(-0.6, 0.6)
    ax_attitude.set_title("Attitude tracking (rad)")
    ax_attitude.set_xlabel("step")

    ax_controls.clear()
    if len(steps) > 0:
        ax_controls.plot(steps, state.aileron, label="aileron")
        ax_controls.plot(steps, state.elevator, label="elevator")
        ax_controls.plot(steps, state.rudder, label="rudder")
        ax_controls.legend(loc="upper left", fontsize=8)
    ax_controls.set_ylim(-1.1, 1.1)
    ax_controls.set_title("Control surfaces")
    ax_controls.set_xlabel("step")

    ax_diag.clear()
    if hasattr(ax_diag, "_twin"):
        ax_diag._twin.clear()
    else:
        ax_diag._twin = ax_diag.twinx()
    if len(steps) > 0:
        loss_x, loss_smoothed = rolling_mean(np.array(state.online_loss), SMOOTHING_WINDOW)
        (line1,) = ax_diag.plot(
            steps[loss_x], loss_smoothed, color="tab:blue", label="online loss (smoothed)"
        )
        (line2,) = ax_diag._twin.plot(
            steps, state.sensitivity_norm, color="tab:orange", label="sensitivity ||S||_F"
        )
        ax_diag.legend(handles=[line1, line2], loc="upper left", fontsize=8)
    ax_diag.set_title("RTRL diagnostics")
    ax_diag.set_xlabel("step")
    ax_diag.set_ylabel("online loss", color="tab:blue")
    ax_diag._twin.set_ylabel("||S||_F", color="tab:orange")

    ax_text.clear()
    ax_text.axis("off")
    lines = [
        f"Controller: {controller_name}",
        format_fault_line(configured_fault, fault_wrapper),
        format_wind_line(wind_wrapper),
        f"Step: {state.step_count} / {max_steps}",
        f"Episode return: {state.episode_return:.1f}",
    ]
    ax_text.text(0.05, 0.9, "\n".join(lines), va="top", family="monospace", fontsize=12)

    fig.tight_layout()


def main() -> None:
    args = parse_args()

    controller_cfg = load_yaml_cfg(CONFIGS_DIR / "controller" / f"{args.controller}.yaml")
    env_cfg = load_yaml_cfg(CONFIGS_DIR / "env" / f"{args.env}.yaml")
    # Same raw-obs override as rtrl_flight.runner (see force_raw_obs's
    # docstring): every controller here is calibrated on raw radian-scale
    # obs. Trace logging is also disabled -- this is a demo tool, not a run
    # that should leave parquet artifacts behind (per this file's own
    # docstring: "not part of the training loop").
    env_cfg = force_raw_obs(env_cfg)
    env_cfg = dict(env_cfg)
    env_cfg["trace"] = {"enabled": False}

    env = make_env(env_cfg)
    controller = build_controller(controller_cfg)
    load_checkpoint_if_present(controller, args.controller, force_online_updates=None)

    fault_wrapper = find_wrapper(env, FaultWrapper)
    wind_wrapper = find_wrapper(env, WindWrapper)
    configured_fault = load_yaml_cfg(CONFIGS_DIR / "env" / f"{args.env}.yaml").get("fault")

    state = DemoState()
    wind_index = WIND_CYCLE.index(wind_wrapper.wind_level) if wind_wrapper is not None else 0

    def on_key(event: Any) -> None:
        nonlocal wind_index
        if event.key == "f":
            if fault_wrapper is not None:
                fault_wrapper.trigger_fault("aileron", 0.5)
            else:
                print("no FaultWrapper in this env -- 'f' is a no-op")
        elif event.key == "e":
            if fault_wrapper is not None:
                fault_wrapper.trigger_fault("elevator", 0.5)
            else:
                print("no FaultWrapper in this env -- 'e' is a no-op")
        elif event.key == "r":
            if fault_wrapper is not None:
                fault_wrapper.trigger_fault("rudder", 0.5)
            else:
                print("no FaultWrapper in this env -- 'r' is a no-op")
        elif event.key == "w":
            if wind_wrapper is not None:
                wind_index = (wind_index + 1) % len(WIND_CYCLE)
                wind_wrapper.set_severity(WIND_CYCLE[wind_index])
            else:
                print("no WindWrapper in this env -- 'w' is a no-op")
        elif event.key == " ":
            state.reset_requested = True

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.canvas.mpl_connect("key_press_event", on_key)

    def do_reset() -> np.ndarray:
        controller.reset()
        obs, _info = env.reset()
        state._new_episode()
        state.reset_requested = False
        return obs

    obs = do_reset()
    max_steps = int(env.unwrapped.task.steps_left.max)

    plt.ion()
    plt.show()

    while plt.fignum_exists(fig.number):
        if state.reset_requested:
            obs = do_reset()

        action = controller.act(obs)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        # Exactly the Controller ABC's 4 args (see rtrl_flight.runner's same
        # note) -- PID/BPTT-LSTM's update() don't accept extra kwargs.
        update_result = controller.update(obs, action, reward, next_obs)
        state.episode_return += reward

        target_pitch, target_roll = get_targets(env)
        state.record(
            pitch=float(obs[1]),
            roll=float(obs[2]),
            target_pitch=target_pitch,
            target_roll=target_roll,
            action=action,
            online_loss=get_online_loss(update_result),
            sensitivity_norm=get_sensitivity_norm(controller),
        )
        obs = next_obs

        if terminated or truncated:
            obs = do_reset()

        redraw(
            fig,
            (ax1, ax2, ax3, ax4),
            state,
            args.controller,
            configured_fault,
            fault_wrapper,
            wind_wrapper,
            max_steps,
        )
        plt.pause(0.01)

    env.close()


if __name__ == "__main__":
    main()
