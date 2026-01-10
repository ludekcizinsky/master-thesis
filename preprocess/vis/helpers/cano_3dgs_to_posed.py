import sys
import os
from pathlib import Path
from dataclasses import dataclass

import torch
import tyro

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from training.simple_multi_human_trainer import build_renderer, load_skip_frames
from training.helpers.dataset import SceneDataset
from training.helpers.eval_metrics import posed_gs_list_to_serializable_dict

@dataclass
class Poser3DGSConfig:
    scene_dir: Path = Path("/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair15_fight")
    cam_id: int = 4

def get_posed_3dgs_single_view_and_person(cano_gs_model_list, query_points, smplx_single_view, animate_func):

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

def get_posed_3dgs(scene_dir: Path, cam_id: int):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    skip_frames = load_skip_frames(scene_dir)
    ds = SceneDataset(
        scene_root_dir=scene_dir,
        src_cam_id=cam_id,
        skip_frames=skip_frames,
    )

    # build renderer to get query points
    renderer = build_renderer().to(device)

    
    smplx_params = ds[0]["smplx_params"]
    query_points, transform_mat_neutral_pose = renderer.get_query_points(smplx_params, device)

    # Load Canonical 3DGS
    root_gs_model_dir = scene_dir / "canon_3dgs_lhm"
    gs_model_list = torch.load(root_gs_model_dir / "union" / "hi4d_gs.pt", map_location=device, weights_only=False)

    # Get single view SMPLX data
    total_n_frames = len(ds)
    all_view_posed_3dgs = []
    for i in range(total_n_frames):
        # - Get SMPLX params for this view
        data = ds[i]
        smplx_single_view = data["smplx_params"]
        smplx_single_view["transform_mat_neutral_pose"] = transform_mat_neutral_pose

        # - Get posed 3dgs for this view
        posed_3dgs_per_view = get_posed_3dgs_single_view_and_person(
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