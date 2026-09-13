"""Thin Hydra entrypoint for BPTT-LSTM imitation warm-start pretraining. No
logic here -- see rtrl_flight.training.imitation for collect/train/pretrain.

Invoke with e.g.:
  python scripts/train_bptt.py +controller=bptt_lstm +env=cessna172_nominal \
      training.episodes=50 training.epochs=20
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from rtrl_flight.training.imitation import pretrain


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Hydra may chdir into a per-run output dir; resolve checkpoint_path
    # against the original invocation directory so it always lands at the
    # literal repo-relative path from config, not nested under outputs/.
    checkpoint_path = Path(hydra.utils.get_original_cwd()) / cfg.training.checkpoint_path

    final_loss = pretrain(
        env_cfg=cfg.env,
        n_episodes=cfg.training.episodes,
        n_epochs=cfg.training.epochs,
        checkpoint_path=checkpoint_path,
        lr=cfg.controller.get("lr", cfg.training.lr),
        hidden_size=cfg.controller.get("hidden_size", 64),
        num_layers=cfg.controller.get("num_layers", 2),
        batch_size=cfg.training.batch_size,
    )
    print(f"final loss: {final_loss}")


if __name__ == "__main__":
    main()
