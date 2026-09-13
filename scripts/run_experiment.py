"""Thin Hydra entrypoint for running one experiment episode and printing its
metrics summary. No logic here -- see rtrl_flight.runner.run_experiment.

Invoke with e.g.:
  python scripts/run_experiment.py +controller=pid +env=cessna172_nominal seed=0
  python scripts/run_experiment.py +controller=rtrl_rtu +env=cessna172_fault \
      seed=0 controller.online_updates=false
"""

from __future__ import annotations

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from rtrl_flight.runner import run_experiment


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    choices = HydraConfig.get().runtime.choices
    controller_name = choices.get("controller", "unknown_controller")
    env_name = choices.get("env", "unknown_env")

    run_experiment(cfg, controller_name=controller_name, env_name=env_name)


if __name__ == "__main__":
    main()
