import os
import joblib
from pathlib import Path

import numpy as np

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose

import torch
from pytorch3d.transforms import matrix_to_axis_angle

from pycocotools import mask as mask_utils


def run_human4d(cfg):
    # Import registers the submodule's ConfigStore entries
    import submodules.humans4d.track as track  # exposes HMR2_4dhuman or similar

    # We are already inside a Hydra app => must clear before re-initializing
    GlobalHydra.instance().clear()

    # Re-initialize Hydra for the submodule's config (uses its ConfigStore)
    with initialize(version_base="1.2", config_path=None, job_name="human4d"):
        composed = compose(
            config_name="config",   # the name registered in the submodule
            overrides=[
                f"video.source={cfg.output_dir}/preprocess/images",
                f"video.output_dir={cfg.output_dir}/phalp_v2",
                "render.enable=True",
                "render.type=HUMAN_MESH",
            ],
        )

    # Run tracker programmatically
    tracker = track.HMR2_4dhuman(composed)
    tracker.track()


def visualise_human4d(cfg):

    # 0. Load dataset
    human4d_resfile = os.path.join(cfg.output_dir, "phalp_v2", "results", "demo_images_jpg.pkl")
    assert human4d_resfile.exists(), f"Visualing of humans4d failed. Did not find the resulting file {str(human4d_resfile)}"
    h4d_results = load_human4d_results(human4d_resfile)
    print("--- FYI: Loaded Human4D results")


def phalp_smpl2op(smpl_joints):
    """
    Mapping from extracted SMPL joints to OpenPose format.
    In particular, the reordering is done for the joints between left and right feet.


    Args:
        smpl_joints: SMPL joints in the format (N, 2)

    Returns:
        op_joints: OpenPose joints in the format (N, 2)
    """

    j_inds = np.arange(25)
    j_inds[19] = 22
    j_inds[20] = 23
    j_inds[21] = 24
    j_inds[22] = 19
    j_inds[23] = 20
    j_inds[24] = 21

    op_joints = []
    for j_ind in j_inds:
        if j_ind >= len(smpl_joints):
            raise ValueError(f"smpl_joints has only {len(smpl_joints)} joints, but trying to access {j_ind}")
        else:
            op_joints.append(smpl_joints[j_ind])

    return op_joints


def get_op_joints(v, person_track_id):
    """
    Given tracking result for given frame, and target person tracking id,
    extract estimated 2D joints and convert them from SMPL to OpenPose format.

    Args:
        v: dict containing tracking information for a specific frame
        person_track_id: int representing the target person's tracking ID

    Returns:
        j2d: OpenPose 2D joints in the format (N, 2)
    """

    # Parse tracking info for the given frame and person
    i = v['tid'].index(person_track_id)
    img_shape = v['size'][i]
    h, w = img_shape
    img_size = max(h, w)
    j2d = v['2d_joints'][i].reshape(-1, 2)

    # Scale and center the 2D joints into pixel space (originally between 0 and 1)
    j2d = j2d * img_size
    j2d = j2d + np.array([[w - img_size, h - img_size]]) / 2

    # Select only first 25 joints (for reference, google OP Body 25)
    j2d = j2d[:25]
    j2d = phalp_smpl2op(j2d)

    return j2d

def rotmat_to_axisangle(R: torch.Tensor) -> np.ndarray:
    """
    Convert rotation matrices to axis-angle (Rodrigues) vectors.
    Args:
        R: (..., 3, 3) tensor of rotation matrices (e.g., (N,3,3)).
    Returns:
        rotvec: (..., 3) tensor of axis-angle vectors.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3) rotation matrices, got {R.shape}")
    return matrix_to_axis_angle(R).numpy()

def get_smpl_params(smpl_estimation):

    global_orient = rotmat_to_axisangle(smpl_estimation['global_orient']).reshape(-1)
    body_pose = rotmat_to_axisangle(smpl_estimation['body_pose']).reshape(-1)
    beta = smpl_estimation['betas']
    transl = smpl_estimation['transl']

    smpl_param = np.concatenate([
        np.ones(1, dtype=np.float32),
        transl,
        global_orient,
        body_pose,
        beta
    ])

    return smpl_param

def load_human4d_results(path_to_results):
    """
    Load the human4d tracking results from a pickle file.

    Args:
        path_to_results: str, path to the pickle file containing tracking results

    Returns:
        dict: a dictionary containing the tracking results
    """

    # Load the tracking results from phalp (tracking ids, poses etc.)
    phalp_res = joblib.load(path_to_results)

    # Get all tracked_ids
    tracked_ids = []
    for v in phalp_res.values():
        tracked_ids.extend(v['tracked_ids'])
    tracked_ids = list(set(tracked_ids))

    # Init the result dict
    track_res = dict()
    for pid in tracked_ids:
        track_res[pid] = dict()

    extract_frame_id = lambda name: int(name.split("_")[1].split(".")[0])

    for k in sorted(list(phalp_res.keys())):
        v = phalp_res[k]
        frame_id = extract_frame_id(Path(k).name)
        for _, pid in enumerate(v['tracked_ids']):
            # Extract Open Pose joints (Body-25)
            j2d = get_op_joints(v, pid)

            # Extract mask for the given human
            i = v['tid'].index(pid)
            mask = mask_utils.decode(v['mask'][i][0])

            # Extract SMPL estimates
            smpl_estimation = v['smpl'][i]
            smpl_param = get_smpl_params(smpl_estimation)

            # Save the results
            track_res[pid][frame_id] = dict(
                bbox=v['bbox'][i],
                smpl_param=smpl_param,
                phalp_mask=mask,
                phalp_j3ds=v['3d_joints'][i],
                phalp_j2ds=j2d
            )

    return track_res