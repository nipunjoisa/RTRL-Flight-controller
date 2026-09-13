"""Throwaway PID gain-tuning diagnostic (PART A of the retuning task).
NOT wired into any pipeline, saves nothing -- edit GAINS below and rerun
until mean |pitch_error| < 0.2 rad and mean |roll_error| < 0.3 rad over
N_EPISODES nominal episodes with normalize=False (raw obs, real units).
"""

from __future__ import annotations

import numpy as np

from rtrl_flight.controllers.pid import PIDController
from rtrl_flight.env.make import make_env

N_EPISODES = 5
OBS_INDICES = {"pitch_error": 9, "roll_error": 10, "yaw_rate": 8}

GAINS = {
    "aileron": (0.4, 0.02, 0.15),
    "elevator": (0.4, 0.02, 0.15),
    "rudder": (0.1, 0.0, 0.05),
}
RUDDER_ROLL_COORDINATION_GAIN = 0.1


def main() -> None:
    env = make_env({"agent_interaction_freq": 5, "normalize": False})
    pid = PIDController(
        gains=GAINS,
        obs_indices=OBS_INDICES,
        dt=1 / 5,
        rudder_roll_coordination_gain=RUDDER_ROLL_COORDINATION_GAIN,
    )

    pitch_errors: list[float] = []
    roll_errors: list[float] = []
    aileron_cmds: list[float] = []
    elevator_cmds: list[float] = []
    rudder_cmds: list[float] = []

    for _ in range(N_EPISODES):
        pid.reset()
        obs, _info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = pid.act(obs)
            pitch_errors.append(float(obs[OBS_INDICES["pitch_error"]]))
            roll_errors.append(float(obs[OBS_INDICES["roll_error"]]))
            aileron_cmds.append(float(action[0]))
            elevator_cmds.append(float(action[1]))
            rudder_cmds.append(float(action[2]))
            obs, _reward, terminated, truncated, _info = env.step(action)

    env.close()

    pitch_arr = np.abs(np.array(pitch_errors))
    roll_arr = np.abs(np.array(roll_errors))

    print(f"gains: {GAINS}")
    print(f"episodes: {N_EPISODES}, steps collected: {len(pitch_errors)}")
    print(f"mean |pitch_error|: {pitch_arr.mean():.4f} rad")
    print(f"max  |pitch_error|: {pitch_arr.max():.4f} rad")
    print(f"mean |roll_error|:  {roll_arr.mean():.4f} rad")
    print(f"max  |roll_error|:  {roll_arr.max():.4f} rad")
    print(f"mean |aileron|:  {np.mean(np.abs(aileron_cmds)):.4f}")
    print(f"mean |elevator|: {np.mean(np.abs(elevator_cmds)):.4f}")
    print(f"mean |rudder|:   {np.mean(np.abs(rudder_cmds)):.4f}")


if __name__ == "__main__":
    main()
