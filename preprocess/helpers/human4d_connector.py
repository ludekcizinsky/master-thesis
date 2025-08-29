import os
import cv2
import joblib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose

import torch
from pytorch3d.transforms import matrix_to_axis_angle
import supervision as sv

from pycocotools import mask as mask_utils

from preprocess.helpers.pose import OP19_SKELETON_1B, xywh_to_xyxy, op25_to_op19
from preprocess.helpers.video_utils import extract_frame_id, load_images, frames_to_video


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

    human4d_resfile = os.path.join(cfg.output_dir, "phalp_v2", "results", "demo_images.pkl")
    os.path.exists(human4d_resfile), f"Visualing of humans4d failed. Did not find the resulting file {str(human4d_resfile)}"
    h4d_results = load_human4d_results(human4d_resfile)
    print("--- FYI: Loaded Human4D results")

    img_dir = os.path.join(cfg.output_dir, "preprocess", "images")
    img_dict = load_images(img_dir)
    print("--- FYI: Loaded images")

    edge_annotator   = sv.EdgeAnnotator(thickness=2, edges=OP19_SKELETON_1B, color=sv.Color.BLUE)
    vertex_annotator = sv.VertexAnnotator(radius=5, color=sv.Color.GREEN)
    box_annotator    = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
    mask_annotator = sv.MaskAnnotator()

    for fid, frame_results in tqdm(h4d_results.items()):
        bboxes = []
        pids = []
        joints = []
        masks = []
        for pid, frame_res in frame_results.items():
            bboxes.append(frame_res['bbox'])
            pids.append(pid)
            joints.append(frame_res['phalp_j2ds'])
            masks.append(frame_res['phalp_mask'])
        img = img_dict[fid]
        bboxes = np.stack(bboxes)
        joints_arr = np.stack(joints)  # (P, 25, 2)
        op19_joints = op25_to_op19(joints_arr)
        xyxy_bboxes = xywh_to_xyxy(bboxes)
        masks_arr = np.stack(masks).astype(bool)  # (P, H, W)
        pids = np.array(pids)

        if len(bboxes) > 0:

            det = sv.Detections(xyxy=xyxy_bboxes, class_id=pids, mask=masks_arr)

            img = box_annotator.annotate(
                scene=img,
                detections=det,
            )
            img = label_annotator.annotate(
                scene=img,
                detections=det,
            )
            img = mask_annotator.annotate(
                scene=img,
                detections=det,
            )

        if len(joints_arr) > 0:
            kps = sv.KeyPoints(xy=op19_joints, class_id=pids)

            img = edge_annotator.annotate(
                scene=img,
                key_points=kps,
            )
            img = vertex_annotator.annotate(
                scene=img,
                key_points=kps,
            )

        img_dict[fid] = img

    # Save annotated images
    frames_dir = os.path.join(cfg.output_dir, "visualizations", "box_and_pose", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for fid, img in img_dict.items():
        cv2.imwrite(os.path.join(frames_dir, f"frame_{fid:05d}.png"), img)

    output_file = os.path.join(cfg.output_dir, "visualizations", "box_and_pose", "video.mp4")
    frames_to_video(frames_dir, output_file, framerate=12)

def phalp_smpl2op(smpl_joints):
    """
    Remap PHALP/SMPL feet to OP25 foot convention:
    OP25 wants: [BigToe, SmallToe, Heel]
    PHALP gives: [Heel, BigToe, SmallToe]
    """
    j_inds = np.arange(25)

    # left foot
    j_inds[19] = 20  # LBigToe  <- PHALP BigToe
    j_inds[20] = 21  # LSmallToe<- PHALP SmallToe
    j_inds[21] = 19  # LHeel    <- PHALP Heel

    # right foot
    j_inds[22] = 23  # RBigToe  <- PHALP BigToe
    j_inds[23] = 24  # RSmallToe<- PHALP SmallToe
    j_inds[24] = 22  # RHeel    <- PHALP Heel

    return smpl_joints[j_inds]


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

def rotmat_to_axisangle(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrices to axis-angle (Rodrigues) vectors.
    Args:
        R: (..., 3, 3) array of rotation matrices (e.g., (N,3,3)).
    Returns:
        rotvec: (..., 3) array of axis-angle vectors.
    """

    R = torch.tensor(R, dtype=torch.float32)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3) rotation matrices, got {R.shape}")
    return matrix_to_axis_angle(R).numpy()

def get_smpl_params(v, i):


    smpl_estimation = v['smpl'][i]
    global_orient = rotmat_to_axisangle(smpl_estimation['global_orient']).reshape(-1)
    body_pose = rotmat_to_axisangle(smpl_estimation['body_pose']).reshape(-1)
    beta = smpl_estimation['betas']
    transl = v['camera'][i] 

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
    
    Notes:
        See in detail the phalp result file structure here:
        https://github.com/brjathu/PHALP?tab=readme-ov-file#output-pkl-structure
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

    for k in sorted(list(phalp_res.keys())):
        v = phalp_res[k]
        frame_id = extract_frame_id(Path(k).name)
        track_res[frame_id] = dict()
        for _, pid in enumerate(v['tracked_ids']):
            # Extract Open Pose joints (Body-25)
            j2d = get_op_joints(v, pid)

            # Extract mask for the given human
            i = v['tid'].index(pid)
            mask = mask_utils.decode(v['mask'][i][0])

            # Extract SMPL estimates
            smpl_param = get_smpl_params(v, i)

            # Save the results
            track_res[frame_id][pid] = dict(
                bbox=v['bbox'][i],
                smpl_param=smpl_param,
                phalp_mask=mask,
                phalp_j3ds=v['3d_joints'][i],
                phalp_j2ds=j2d
            )

    return track_res