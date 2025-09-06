import wandb
from omegaconf import OmegaConf
import os

from pathlib import Path


def init_logging(cfg):     
    wandb_path = Path("/scratch/izar/cizinsky/thesis/") / "wandb"
    os.makedirs(wandb_path, exist_ok=True)
    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.logger.tags,
        dir=wandb_path,
        group=cfg.scene_name,
        mode="online" if not cfg.debug else "disabled",
    )
    if cfg.debug:
        print("--- FYI: Running in debug mode, wandb logging is disabled.")
    else:
        print(f"--- FYI: Logging to wandb project {cfg.logger.project}, entity {cfg.logger.entity}, group {cfg.scene_name}.")