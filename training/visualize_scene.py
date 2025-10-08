import os
import sys
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings(
    "ignore",
    message=".*weights_only=False.*",
    category=FutureWarning,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from training.helpers.checkpointing import GaussianCheckpointManager
from training.helpers.dataset import FullSceneDataset
from training.helpers.model_init import SceneSplats, init_smpl_server
from training.helpers.render import render_splats
from training.helpers.smpl_utils import update_skinning_weights


def _select_available_tids(
    checkpoint_manager: GaussianCheckpointManager,
    requested_tids: List[int],
    device: torch.device,
) -> tuple[List[int], List[torch.nn.ParameterDict]]:
    """Load human checkpoints, skipping any missing ones."""
    available_tids: List[int] = []
    dynamic_splats: List[torch.nn.ParameterDict] = []

    for tid in requested_tids:
        params, iteration = checkpoint_manager.load_human(tid, device=device)
        if params is None:
            print(f"--- FYI: No checkpoint found for tid {tid}; skipping.")
            continue

        available_tids.append(tid)
        dynamic_splats.append(params)
        if iteration is not None:
            print(f"--- FYI: Loaded {params['means'].shape[0]} splats for human {tid} from iteration {iteration}.")
        else:
            print(f"--- FYI: Loaded {params['means'].shape[0]} splats for human {tid}.")

    return available_tids, dynamic_splats


@hydra.main(config_path="../configs", config_name="visualize_scene.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_grad_enabled(False)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"--- FYI: rendering on device {device}")

    requested_tids = list(cfg.render_tids) if cfg.render_tids is not None else []

    checkpoint_manager = GaussianCheckpointManager(
        Path(cfg.output_dir),
        cfg.group_name,
        requested_tids,
    )

    static_splats = None
    if bool(cfg.render_bg):
        static_splats, static_iteration = checkpoint_manager.load_static(device=device)
        if static_splats is None:
            print("--- FYI: No static checkpoint found; background rendering disabled.")
        else:
            if static_iteration is not None:
                print(f"--- FYI: Loaded static splats from iteration {static_iteration}.")
            else:
                print("--- FYI: Loaded static splats.")

    available_tids, dynamic_splats = _select_available_tids(
        checkpoint_manager,
        requested_tids,
        device,
    )

    has_static = static_splats is not None and bool(cfg.render_bg)
    has_dynamic = len(dynamic_splats) > 0

    if not has_static and not has_dynamic:
        raise RuntimeError("No scene components available to render (static and dynamic checkpoints missing).")

    smpl_c_info = None
    if has_dynamic:
        smpl_c_info = init_smpl_server(device)

    scene_splats = SceneSplats(
        static=static_splats if has_static else None,
        dynamic=dynamic_splats,
        smpl_c_info=smpl_c_info,
    )

    lbs_weights = None
    if has_dynamic:
        lbs_weights = [
            w.detach()
            for w in update_skinning_weights(
                scene_splats,
                k=int(cfg.lbs_knn),
                eps=1e-6,
                device=str(device),
            )
        ]
        print(f"--- FYI: Prepared LBS weights for {len(lbs_weights)} dynamic components.")

    dataset_tids = available_tids
    dataset = FullSceneDataset(
        Path(cfg.preprocess_dir),
        dataset_tids,
        cloud_downsample=int(cfg.cloud_downsample),
        train_bg=bool(has_static),
    )
    print(f"--- FYI: Dataset ready with {len(dataset)} frames.")

    group_dir = Path(cfg.output_dir) / "visualisations" / cfg.group_name 
    image_dir = group_dir / "images"
    render_dir = group_dir / "full_render"
    image_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- FYI: Saving originals to {image_dir}")
    print(f"--- FYI: Saving renders to {render_dir}")

    for idx in tqdm(range(len(dataset)), desc="Rendering frames"):
        sample = dataset[idx]
        fid = int(sample["fid"])
        image = sample["image"]
        K = sample["K"].unsqueeze(0).to(device)
        w2c = sample["M_ext"].unsqueeze(0).to(device)
        smpl_param = sample["smpl_param"].unsqueeze(0).to(device)
        H, W = int(sample["H"]), int(sample["W"])

        with torch.no_grad():
            colors, _, _ = render_splats(
                scene_splats,
                smpl_param,
                lbs_weights,
                w2c,
                K,
                H,
                W,
                sh_degree=int(cfg.sh_degree),
            )

        render_np = (colors[0].cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
        Image.fromarray(render_np).save(render_dir / f"{fid:04d}.png")

        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(image_dir / f"{fid:04d}.png")

    print("✅ Rendering complete.")


if __name__ == "__main__":
    main()
