"""Thin Hydra entrypoint for the online-off ablation (agent_docs/
experiments.md Scenario 5). Identical to scripts/run_experiment.py except
it forces controller.online_updates=False for any controller that has that
attribute -- no separate config file needed. No logic here -- see
rtrl_flight.runner.run_experiment(..., ablation=True).

Invoke with e.g.:
  python scripts/run_ablation.py +controller=rtrl_rtu +env=cessna172_fault seed=0
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

    run_experiment(cfg, controller_name=controller_name, env_name=env_name, ablation=True)


if __name__ == "__main__":
    main()
