"""Online RTRL training loop: warm-start an RTRL-RTU controller from a
BPTT-LSTM checkpoint, then run it in the env with controller.update() called
every step (agent_docs/architecture.md's runner loop: act -> step ->
update). This is the "online RTRL loop" phase of agent_docs/
project_overview.md's timeline -- no offline unroll, no replay buffer, one
step of work per env step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rtrl_flight.controllers.rtrl_rtu import RTRLRTUController
from rtrl_flight.env.make import make_env
from rtrl_flight.rtrl.warmstart import apply_warmstart, load_bptt_checkpoint


def warmstart_and_train(
    env_cfg: Any,
    checkpoint_path: str | Path,
    n_episodes: int,
    save_every: int,
    output_checkpoint: str | Path,
    hidden_size: int = 64,
    lr: float = 1e-3,
) -> list[float]:
    """Loads a warm-started RTRLRTUController and runs it online for
    n_episodes, calling controller.update() every step (online_updates must
    be True on the controller for this to do anything -- see
    agent_docs/experiments.md Scenario 5 for the online-off ablation, which
    uses this same loop with online_updates=False instead of a different
    code path, per agent_docs/architecture.md's "no isinstance/branching"
    invariant).

    Prints per-episode return, sensitivity Frobenius norm, and step count.
    Saves a checkpoint every `save_every` episodes (and always at the end).
    Returns the list of per-episode returns.
    """
    bptt_state_dict = load_bptt_checkpoint(checkpoint_path)
    controller = RTRLRTUController(hidden_size=hidden_size, lr=lr, online_updates=True)
    apply_warmstart(controller, bptt_state_dict)

    env = make_env(env_cfg)
    returns: list[float] = []
    output_checkpoint = Path(output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    try:
        for episode in range(n_episodes):
            controller.reset()
            obs, _info = env.reset()
            episode_return = 0.0
            step_count = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action = controller.act(obs)
                next_obs, reward, terminated, truncated, _info = env.step(action)
                controller.update(obs, action, reward, next_obs, done=terminated or truncated)
                episode_return += reward
                step_count += 1
                obs = next_obs

            returns.append(episode_return)
            sensitivity_norm = controller.sensitivity_frobenius_norm()
            print(
                f"episode {episode + 1}/{n_episodes}  "
                f"return={episode_return:.4f}  "
                f"sensitivity_norm={sensitivity_norm:.4f}  "
                f"steps={step_count}"
            )

            if (episode + 1) % save_every == 0:
                controller.save(output_checkpoint)
    finally:
        env.close()

    controller.save(output_checkpoint)
    return returns
