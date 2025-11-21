"""
Usage example:
    playground/prepare_gas_data.py group_name=v7_default_corrected scene_name=taichi resume=true
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from omegaconf import DictConfig

import numpy as np
import torch
import hydra
from PIL import Image

# Make training and evaluation modules available
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_HOME", "/scratch/izar/cizinsky/.cache")
os.environ.setdefault("HF_HOME", "/scratch/izar/cizinsky/.cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

from training.helpers.checkpointing import ModelCheckpointManager
from training.helpers.dataset import build_training_dataset
from training.helpers.model_init import create_splats_with_optimizers
from training.helpers.render import render_splats
from training.helpers.smpl_utils import update_skinning_weights



@hydra.main(version_base=None, config_path=str(REPO_ROOT / "configs"), config_name="train")
def main(cfg: DictConfig) -> None:
    render_output_dir = Path("/scratch/izar/cizinsky/thesis/playground_output")

    device = "cuda"
    ckpt_manager = ModelCheckpointManager(
        scene_output_dir=Path(cfg.output_dir),
        group_name=cfg.group_name,
        tids=cfg.tids,
    )

    # Use progressive SAM masks from the checkpoint folder (if present) to mirror training setup
    mask_path = ckpt_manager.root / "progressive_sam"
    dataset = build_training_dataset(cfg, mask_path=mask_path)

    all_gs, _, _ = create_splats_with_optimizers(
        device=device, cfg=cfg, ds=dataset, checkpoint_manager=ckpt_manager
    )

    lbs_weights: Optional[List[torch.Tensor]]
    if len(cfg.tids) > 0:
        lbs_weights = update_skinning_weights(
            all_gs, k=int(cfg.lbs_knn), eps=1e-6, device=device
        )
    else:
        lbs_weights = None

    smpl_params_map, _ = ckpt_manager.load_smpl(device=device)

    frame_id = 30
    sample = dataset[frame_id]
    w2c = sample["M_ext"].to(device).unsqueeze(0)
    K = sample["K"].to(device).unsqueeze(0)
    H, W = int(sample["H"]), int(sample["W"])
    smpl_param = smpl_params_map[frame_id].to(device)

    with torch.no_grad():
        colors, _, _ = render_splats(
            all_gs,
            smpl_param,
            lbs_weights,
            w2c,
            K,
            H,
            W,
            sh_degree=int(cfg.sh_degree),
            render_mode="RGB",
        )

    image = (colors.squeeze(0).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    image_path = render_output_dir / f"render_frame_{frame_id:04d}.png"
    Image.fromarray(image).save(image_path)
    print(f"Saved render for frame {frame_id} to {image_path}")


if __name__ == "__main__":
    main()
