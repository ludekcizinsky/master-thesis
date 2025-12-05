
import wandb
from omegaconf import OmegaConf
import os

from pathlib import Path


def init_logging(cfg):     
    wandb_path = Path("/scratch/izar/cizinsky/thesis/")
    os.makedirs(wandb_path, exist_ok=True)
    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.logger.tags,
        dir=wandb_path,
        group=cfg.group_name,
        mode="online" if not cfg.debug else "disabled",
    )

    internal_run_id = f"{wandb.run.name}_{wandb.run.id}"
    wandb.run.summary["internal_run_id"] = internal_run_id
    if cfg.debug:
        print("--- FYI: Running in debug mode, wandb logging is disabled.")
    else:
        print(f"--- FYI: Logging to wandb project {cfg.logger.project}, entity {cfg.logger.entity}, group {cfg.group_name}.")

