"""Thin Hydra entrypoint for warm-started online RTRL training. No logic
here -- see rtrl_flight.training.online_train.warmstart_and_train.

Invoke with e.g.:
  python scripts/train_rtrl_warmstart.py +controller=rtrl_rtu +env=cessna172_nominal \
      training.n_episodes=20 training.save_every=5
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from rtrl_flight.training.online_train import warmstart_and_train


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Resolve against the original invocation dir, not Hydra's per-run
    # output dir -- same reasoning as scripts/train_bptt.py.
    original_cwd = Path(hydra.utils.get_original_cwd())
    bptt_checkpoint_path = original_cwd / cfg.training.checkpoint_path
    output_checkpoint = original_cwd / cfg.training.output_checkpoint

    returns = warmstart_and_train(
        env_cfg=cfg.env,
        checkpoint_path=bptt_checkpoint_path,
        n_episodes=cfg.training.n_episodes,
        save_every=cfg.training.save_every,
        output_checkpoint=output_checkpoint,
        hidden_size=cfg.controller.get("hidden_size", 64),
        lr=cfg.controller.get("lr", cfg.training.lr),
    )
    print(f"mean return over {len(returns)} episodes: {sum(returns) / len(returns):.4f}")


if __name__ == "__main__":
    main()
