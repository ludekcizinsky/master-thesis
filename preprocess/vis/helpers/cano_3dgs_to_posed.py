import sys
import os
from pathlib import Path
from dataclasses import dataclass

import torch
import tyro

from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from training.simple_multi_human_trainer import build_renderer
from training.helpers.dataset import root_dir_to_smplx_dir
from training.helpers.eval_metrics import posed_gs_list_to_serializable_dict

@dataclass
class Poser3DGSConfig:
    scene_dir: Path = Path("/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair15_fight")
    cam_id: int = 4

def _load_smplx(path: Path, device: torch.device):

    npz = np.load(path)

    def add_key(key, default=None):
        if key in npz:
            arrs = torch.from_numpy(npz[key]).float()
        elif default is not None:
            arrs = torch.from_numpy(default).float()
        else:
            raise KeyError(f"{key} is not a file in the archive")
        return arrs.to(device)  # [P, ...]

    betas = add_key("betas")
    n_persons = betas.shape[0]

    smplx = {
        "betas": betas,
        "root_pose": add_key("root_pose"),   # [P,3] world axis-angle
        "body_pose": add_key("body_pose"),
        "jaw_pose": add_key("jaw_pose"),
        "leye_pose": add_key("leye_pose"),
        "reye_pose": add_key("reye_pose"),
        "lhand_pose": add_key("lhand_pose"),
        "rhand_pose": add_key("rhand_pose"),
        "trans": add_key("trans"),           # [P,3] world translation
        "expr": add_key("expression", default=np.zeros((n_persons, 100), dtype=np.float32)),
    }

    return smplx

def _get_posed_3dgs_single_view_and_person(cano_gs_model_list, query_points, smplx_single_view, animate_func):

    # Pose 3dgs
    n_persons = len(cano_gs_model_list)

    all_posed_gs_list = []
    for person_idx in range(n_persons):
        person_canon_3dgs = cano_gs_model_list[person_idx]
        person_query_pt = query_points[person_idx]
        person_smplx_data = {k: v[person_idx : person_idx + 1] for k, v in smplx_single_view.items()}
        posed_gs, neutral_posed_gs = animate_func(
            person_canon_3dgs,
            person_query_pt,
            person_smplx_data,
        )
        all_posed_gs_list.append(posed_gs)

    # Merge all persons
    merged_posed_3dgs = posed_gs_list_to_serializable_dict(all_posed_gs_list)

    return merged_posed_3dgs

def get_posed_3dgs(scene_dir: Path, frames: list[str]):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load smplx params
    smplx_dir = root_dir_to_smplx_dir(scene_dir)
    ds = list()
    for fname in frames:
        smplx_path = smplx_dir / f"{fname}.npz"
        smplx_params = _load_smplx(smplx_path, device)
        ds.append(smplx_params)

    # build renderer to get query points
    renderer = build_renderer().to(device)

    
    smplx_params = ds[0]
    query_points, transform_mat_neutral_pose = renderer.get_query_points(smplx_params, device)

    # Load Canonical 3DGS
    root_gs_model_dir = scene_dir / "canon_3dgs_lhm"
    gs_model_list = torch.load(root_gs_model_dir / "union" / "gs.pt", map_location=device, weights_only=False)

    # Get single view SMPLX data
    total_n_frames = len(ds)
    all_view_posed_3dgs = []
    for i in tqdm(range(total_n_frames), desc="Posing 3DGS for all views", total=total_n_frames):
        # - Get SMPLX params for this view
        smplx_single_view = ds[i]
        smplx_single_view["transform_mat_neutral_pose"] = transform_mat_neutral_pose

        # - Get posed 3dgs for this view
        posed_3dgs_per_view = _get_posed_3dgs_single_view_and_person(
            gs_model_list,
            query_points,
            smplx_single_view,
            renderer.animate_single_view_and_person,
        )
        all_view_posed_3dgs.append(posed_3dgs_per_view)

    return all_view_posed_3dgs


if __name__ == "__main__":
    args = tyro.cli(Poser3DGSConfig)
    get_posed_3dgs(args.scene_dir, args.cam_id)
