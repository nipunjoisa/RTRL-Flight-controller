"""AttitudeHoldTask: a custom jsbgym Task for pitch+roll attitude hold.

jsbgym ships no attitude-hold task (only HeadingControlTask and
TurnHeadingControlTask -- see agent_docs/environment.md, "Mismatch 1").
This subclasses jsbgym.tasks.FlightTask (itself a Task subclass -- see
FlightTask's docstring in .venv/Lib/site-packages/jsbgym/tasks.py) to reuse
its proven sim-stepping/reset machinery, but overrides task_step() to bypass
jsbgym's Assessor/RewardComponent framework entirely in favour of the plain
reward function in rtrl_flight.env.reward -- there's no attitude-hold reward
components to reuse from HeadingControlTask (altitude/track-error shaped),
and the assessor abstraction buys nothing here.

Constructor signature intentionally matches HeadingControlTask's
(shaping_type, step_frequency_hz, aircraft, ...) because jsbgym's
JsbSimEnv.__init__ instantiates `task_type(shaping, agent_interaction_freq,
aircraft)` positionally (see jsbgym/environment.py) -- this is what lets
AttitudeHoldTask be dropped into NoFGJsbSimEnv as a drop-in task_type.
`shaping_type` is accepted and stored but unused: attitude-hold has no
shaping-reward variants (yet).
"""

from __future__ import annotations

import math
import random

import jsbgym.properties as prp
from jsbgym.aircraft import Aircraft
from jsbgym.properties import BoundedProperty, Property
from jsbgym.rewards import RewardStub
from jsbgym.simulation import Simulation
from jsbgym.tasks import FlightTask, Shaping

from rtrl_flight.env.reward import attitude_tracking_reward

# Sampled uniformly at episode reset. See agent_docs/environment.md.
TARGET_PITCH_RANGE_RAD = (-0.3, 0.3)
TARGET_ROLL_RANGE_RAD = (-0.4, 0.4)


class AttitudeHoldTask(FlightTask):
    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8
    INITIAL_HEADING_DEG = 270  # attitude hold doesn't track heading; fixed value only
    DEFAULT_EPISODE_TIME_S = 60.0

    # Target properties, sampled at reset -- not separate obs slots (see
    # agent_docs/environment.md); used internally to compute the two error
    # slots below, same pattern jsbgym's HeadingControlTask uses for
    # target/track-deg.
    target_pitch_rad = BoundedProperty(
        "target/pitch-rad", "target pitch [rad]", *TARGET_PITCH_RANGE_RAD
    )
    target_roll_rad = BoundedProperty(
        "target/roll-rad", "target roll [rad]", *TARGET_ROLL_RANGE_RAD
    )

    # Replace HeadingControlTask's error/altitude-error-ft and
    # error/track-error-deg (slots 9-10) with our own. Bounds per
    # agent_docs/environment.md.
    pitch_error_rad = BoundedProperty(
        "error/pitch-error-rad", "target_pitch - pitch [rad]", -math.pi, math.pi
    )
    roll_error_rad = BoundedProperty(
        "error/roll-error-rad", "target_roll - roll [rad]", -2 * math.pi, 2 * math.pi
    )

    action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd)

    def __init__(
        self,
        shaping_type: Shaping,
        step_frequency_hz: float,
        aircraft: Aircraft,
        episode_time_s: float = DEFAULT_EPISODE_TIME_S,
        positive_rewards: bool = True,
    ) -> None:
        self.shaping_type = shaping_type  # unused; kept for env constructor compatibility
        self.aircraft = aircraft
        self.max_time_s = episode_time_s
        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.steps_left = BoundedProperty(
            "info/steps_left", "steps remaining in episode", 0, episode_steps
        )
        self.extra_state_variables = (self.pitch_error_rad, self.roll_error_rad)
        self.state_variables = FlightTask.base_state_variables + self.extra_state_variables

        self.last_state = None
        self._make_state_class()
        self.debug = False

    def get_initial_conditions(self) -> dict[Property, float]:
        extra_conditions = {
            prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
            prp.initial_v_fps: 0,
            prp.initial_w_fps: 0,
            prp.initial_p_radps: 0,
            prp.initial_q_radps: 0,
            prp.initial_r_radps: 0,
            prp.initial_roc_fpm: 0,
            prp.initial_heading_deg: self.INITIAL_HEADING_DEG,
        }
        return {**FlightTask.base_initial_conditions, **extra_conditions}

    def _new_episode_init(self, sim: Simulation) -> None:
        sim.start_engines()
        sim.raise_landing_gear()
        sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)
        sim[self.steps_left] = self.steps_left.max
        sim[self.target_pitch_rad] = random.uniform(*TARGET_PITCH_RANGE_RAD)
        sim[self.target_roll_rad] = random.uniform(*TARGET_ROLL_RANGE_RAD)
        self._store_reward(RewardStub(0.0, 0.0), sim)

    def _update_custom_properties(self, sim: Simulation) -> None:
        self._update_errors(sim)
        sim[self.steps_left] -= 1

    def _update_errors(self, sim: Simulation) -> None:
        sim[self.pitch_error_rad] = sim[self.target_pitch_rad] - sim[prp.pitch_rad]
        sim[self.roll_error_rad] = sim[self.target_roll_rad] - sim[prp.roll_rad]

    def _is_terminal(self, sim: Simulation) -> bool:
        # Truncation-shaped episode end (fixed length), per agent_docs/experiments.md
        # -- early termination on tracking error would cut off exactly the
        # post-fault recovery window the fault scenario needs to observe.
        # NOTE: jsbgym's JsbSimEnv.step() hardcodes `truncated=False` in its
        # return tuple (see environment.py) -- there is no separate
        # truncation channel to use here, so "terminal" is how the episode
        # actually ends. Safety-bound (stall/spin/altitude-floor) termination
        # is not yet added -- agent_docs/environment.md flags this as still
        # to be tuned once AttitudeHoldTask sees real training data.
        return sim[self.steps_left] <= 0

    def _reward_terminal_override(self, reward, sim: Simulation):
        return reward

    def task_step(self, sim: Simulation, action, sim_steps: int):
        for prop, command in zip(self.action_variables, action, strict=True):
            sim[prop] = command

        for _ in range(sim_steps):
            sim.run()

        self._update_custom_properties(sim)
        state = self.State(*(sim[prop] for prop in self.state_variables))
        terminated = self._is_terminal(sim)

        reward_value = self.get_reward(
            pitch_error_rad=sim[self.pitch_error_rad],
            roll_error_rad=sim[self.roll_error_rad],
            p_rad_s=sim[prp.p_radps],
            q_rad_s=sim[prp.q_radps],
            r_rad_s=sim[prp.r_radps],
            action=action,
        )
        reward = RewardStub(float(reward_value), float(reward_value))
        if terminated:
            reward = self._reward_terminal_override(reward, sim)

        if self.debug:
            self._validate_state(state, terminated, False, action, reward)
        self._store_reward(reward, sim)
        self.last_state = state
        info = {"reward": reward}

        return state, reward.agent_reward(), terminated, False, info

    def get_reward(
        self,
        pitch_error_rad: float,
        roll_error_rad: float,
        p_rad_s: float,
        q_rad_s: float,
        r_rad_s: float,
        action,
    ) -> float:
        """Thin wrapper delegating to the standalone reward function in
        rtrl_flight.env.reward -- see that module for the weighted formula.
        """
        return attitude_tracking_reward(
            pitch_error_rad=pitch_error_rad,
            roll_error_rad=roll_error_rad,
            p_rad_s=p_rad_s,
            q_rad_s=q_rad_s,
            r_rad_s=r_rad_s,
            action=action,
        )

    def get_props_to_output(self) -> tuple:
        return (
            prp.pitch_rad,
            prp.roll_rad,
            self.target_pitch_rad,
            self.target_roll_rad,
            self.pitch_error_rad,
            self.roll_error_rad,
            self.last_agent_reward,
            self.steps_left,
        )
