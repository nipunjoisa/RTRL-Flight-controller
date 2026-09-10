"""Throwaway probe script — NOT part of src/. Verifies the real jsbgym/JSBSim
obs/action API against agent_docs/environment.md's target spec before any
env wrapper code gets written.

Finding this script exists to confirm/refute: jsbgym ships
HeadingControlTask and TurnHeadingControlTask only (see jsbgym/__init__.py's
registered Envs enum) — there is no ready-made "attitude-hold" task that
tracks a target pitch + target roll. This script therefore instantiates the
closest available C172 task (HeadingControlTask, NoFlightGear, STANDARD
shaping) so we can inspect the real observation/action properties and decide
how to build attitude-hold on top of jsbgym (custom Task subclass, most
likely) rather than assuming one exists.
"""

from __future__ import annotations

from importlib.metadata import version

import gymnasium as gym
import jsbgym
import numpy as np

ENV_ID = jsbgym.Envs.C172_HeadingControlTask_Shaping_STANDARD_NoFG_v0.value


def describe_space(space: gym.spaces.Box, names: list[str] | None = None) -> None:
    print(f"  shape: {space.shape}, dtype: {space.dtype}")
    if names is None:
        names = [f"dim_{i}" for i in range(space.shape[0])]
    for i, name in enumerate(names):
        lo = space.low[i]
        hi = space.high[i]
        print(f"    [{i}] {name:35s} low={lo:>14.6f}  high={hi:>14.6f}")


def main() -> None:
    print(f"gymnasium version: {gym.__version__}")
    print(f"jsbgym version: {version('jsbgym')}")
    print(f"jsbsim version: {version('jsbsim')}")
    print(f"Instantiating: {ENV_ID}")
    print(
        "NOTE: no attitude-hold (target-pitch/target-roll) task is registered by "
        "jsbgym -- using the closest available task, HeadingControlTask, as a stand-in "
        "to probe the real property/space API. See docstring above."
    )
    print()

    env = gym.make(ENV_ID)
    task = env.unwrapped.task

    state_names = [p.name for p in task.state_variables]
    action_names = [p.name for p in task.action_variables]

    print("=" * 70)
    print("OBSERVATION SPACE")
    print("=" * 70)
    describe_space(env.observation_space, state_names)

    print()
    print("=" * 70)
    print("ACTION SPACE")
    print("=" * 70)
    describe_space(env.action_space, action_names)

    print()
    print("=" * 70)
    print("STEPPING: 5 random actions")
    print("=" * 70)

    obs, info = env.reset(seed=0)
    print("reset() obs dict:")
    for name, value in zip(state_names, obs, strict=True):
        print(f"    {name:35s} = {value:.6f}")
    print(f"  info: {info}")

    rng = np.random.default_rng(0)
    for step in range(5):
        action = rng.uniform(env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n--- step {step} ---")
        print(f"  action: {dict(zip(action_names, action, strict=True))}")
        print("  obs dict:")
        for name, value in zip(state_names, obs, strict=True):
            print(f"    {name:35s} = {value:.6f}")
        print(f"  reward: {reward}")
        print(f"  terminated: {terminated}, truncated: {truncated}")
        print(f"  info: {info}")

        if terminated or truncated:
            print("  episode ended early, resetting")
            obs, info = env.reset()

    env.close()
    print("\nenv.close() completed cleanly.")


if __name__ == "__main__":
    main()
