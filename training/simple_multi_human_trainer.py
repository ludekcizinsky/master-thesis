import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque


import pandas as pd
import hydra
import numpy as np
import wandb
from tqdm import tqdm
from omegaconf import DictConfig
from PIL import Image
import trimesh


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "submodules", "lhm")
    ),
)
from training.helpers.gs_renderer import GS3DRenderer
from training.helpers.dataset import (
    SceneDataset, 
    fetch_data_if_available, 
    root_dir_to_image_dir, 
    root_dir_to_mask_dir, 
    root_dir_to_skip_frames_path, 
    root_dir_to_depth_dir,
    root_dir_to_all_cameras_dir,
    fetch_masks_if_exist,
    intr_to_4x4,
    extr_to_w2c_4x4,
    root_dir_to_smplx_dir,
    root_dir_to_smpl_dir
)
from training.helpers.debug import (
    save_depth_comparison, 
    create_and_save_depth_debug_vis,
    save_gt_image_and_mask_comparison,
)
from training.helpers.gs_to_mesh import get_meshes_from_3dgs
from training.helpers.eval_metrics import (
    ssim, psnr, lpips, _ensure_nchw, segmentation_mask_metrics,
    compute_pose_mpjpe_per_frame,
    compute_pose_mve_per_frame,
    compute_pose_contact_distance_per_frame,
    compute_pose_pcdr_per_frame,
    gt_mesh_from_sample,
    merge_mesh_dict,
    posed_gs_list_to_serializable_dict,
    align_pred_meshes_icp,
    apply_se3_to_points,
    save_aligned_meshes,
    compute_chamfer_distance,
    compute_p2s_distance,
    compute_normal_consistency,
    compute_volumetric_iou
)
from fused_ssim import fused_ssim

from training.helpers.difix import difix_refine
from submodules.difix3d.src.pipeline_difix import DifixPipeline
from submodules.smplx import smplx

from training.helpers.masking import (
    get_masks_based_bbox,
    estimate_masks_from_smplx_batch, 
    estimate_masks_from_rgb_and_smplx_batch, 
    estimate_masks_from_smplx_batch, 
    estimate_masks_from_src_reprojection_batch, 
    overlay_mask_on_image, 
    get_masked_images, 
    get_gt_est_masks_overlay, 
    save_segmentation_debug_figures
)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def root_dir_to_difix_debug_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "est_images_debug" / f"{cam_id}"

def root_dir_to_est_masks_debug_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "est_masks_debug" / f"{cam_id}"


def root_dir_to_depths_debug_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "est_depths_debug" / f"{cam_id}"

def save_image(tensor: torch.Tensor, filename: str):
    """
    Accepts HWC, CHW, or BCHW; if batch > 1, saves the first item.
    Assumes values in [0,1]. If channels are a multiple of 3, tiles them along width.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3 and tensor.shape[0] % 3 == 0:
        c = tensor.shape[0]
        if c not in (1, 3):
            n = c // 3
            tensor = tensor.view(n, 3, tensor.shape[1], tensor.shape[2])
            tensor = torch.cat([tensor[i] for i in range(n)], dim=2)
        image = tensor.permute(1, 2, 0).detach().cpu().numpy()
    elif tensor.dim() == 3 and tensor.shape[-1] % 3 == 0:
        c = tensor.shape[-1]
        if c not in (1, 3):
            n = c // 3
            tensor = tensor.view(tensor.shape[0], tensor.shape[1], n, 3)
            tensor = torch.cat([tensor[:, :, i] for i in range(n)], axis=1)
        image = tensor.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported tensor shape for save_image: {tensor.shape}")
    image = (image * 255).clip(0, 255).astype("uint8")
    Image.fromarray(image).save(filename)

def save_binary_mask(tensor: torch.Tensor, filename: str):
    """
    Accepts HWC, CHW, or BCHW where C=1; if batch > 1, saves the first item.
    Assumes values in [0,1]. Saves as a binary (0/255) PNG image.
    """

    # First item only
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # HWC or CHW with C=1 convert to HxWx1
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.permute(1, 2, 0)
    elif tensor.dim() == 3 and tensor.shape[-1] == 1:
        pass
    else:
        raise ValueError(f"Unsupported tensor shape for save_binary_mask: {tensor.shape}")
    
    # Binarize and save
    mask = (tensor.detach().cpu().numpy() > 0.5).astype("uint8") * 255
    Image.fromarray(mask.squeeze(-1)).save(filename)

def build_renderer():
    renderer = GS3DRenderer(
        human_model_path="/scratch/izar/cizinsky/pretrained/pretrained_models/human_model_files",
        subdivide_num=1,
        smpl_type="smplx_2",
        feat_dim=1024,
        query_dim=1024,
        use_rgb=True,
        sh_degree=3,
        mlp_network_config={'activation': 'silu', 'n_hidden_layers': 2, 'n_neurons': 512},
        xyz_offset_max_step=1.0,
        clip_scaling=[100, 0.01, 0.05, 3000],
        shape_param_dim=10,
        expr_param_dim=100,
        cano_pose_type=1,
        fix_opacity=False,
        fix_rotation=False,
        decoder_mlp=False,
        skip_decoder=True,
        decode_with_extra_info=None,
        gradient_checkpointing=True,
        apply_pose_blendshape=False,
        dense_sample_pts=40000,
    )

    return renderer



def bbox_crop(bboxes: List[Tuple[int, int, int, int]], to_crop: torch.Tensor) -> torch.Tensor:
    """
    Crop a batch of tensors according to the provided bounding boxes.

    Args:
        bboxes: List of bounding boxes [(y_min, y_max, x_min, x_max)] for each tensor in the batch.
        to_crop: Tensor of shape [B, H, W, C] to be cropped where C can be any number of channels.

    Returns:
        Cropped tensor of shape [B, H_crop, W_crop, C].
    """

    # Crop tensors
    cropped = []
    for b in range(to_crop.shape[0]):
        y_min, y_max, x_min, x_max = bboxes[b]
        cropped.append(to_crop[b, y_min : y_max + 1, x_min : x_max + 1, :])

    return torch.stack(cropped, dim=0)


def load_skip_frames(scene_dir: Path) -> List[int]:
    """
    Load skip frames from skip_frames.csv in the scene directory.

    Note: the frame indicies are actual frame indexes and the frames dir may
    not always start with frame 0. Therefore, the returnd indices correspond to
    the actual frame indices to skip. e.g. if we have frames 10, 11, 12, 13 and skip_frames.csv contains "11,13",
    we will skip frames 11 and 13.

    Args:
        scene_dir: Path to the scene directory.
    Returns:
        List of frame indices to skip.
    """

    skip_frames_file = root_dir_to_skip_frames_path(scene_dir)
    if not skip_frames_file.exists():
        return []
    with open(skip_frames_file, "r") as f:
        line = f.readline().strip()
        skip_frames = [int(idx_str) for idx_str in line.split(",") if idx_str.isdigit()]
    return skip_frames


def get_all_scene_dir_cams(scene_dir: Path, device: torch.device) -> Dict[int, Dict[str, torch.Tensor]]:

    all_cams_dir = root_dir_to_all_cameras_dir(scene_dir)
    all_camera_ids = [path.stem for path in all_cams_dir.iterdir()]

    cam_id_to_cam_params = dict()
    for camera_id in all_camera_ids:

        # Fetch all camera files for this camera
        all_camera_files = sorted(list((all_cams_dir / camera_id).glob("*.npz")))
        cam_params_for_each_frame = dict()
        for path in all_camera_files:

            frame_cam_params = dict()
            with np.load(path) as cams:
                missing = [k for k in ("intrinsics", "extrinsics") if k not in cams.files]
                if missing:
                    raise KeyError(f"Missing keys {missing} in camera file {path}")

                intrinsics = cams["intrinsics"][0] # (3, 3)
                extrinsics = cams["extrinsics"][0] # (3, 4)


            intr = torch.from_numpy(intrinsics).float().to(device)
            extr = torch.from_numpy(extrinsics).float().to(device)

            w2c = extr_to_w2c_4x4(extr, device)
            frame_cam_params["c2w"] = torch.inverse(w2c)
            frame_cam_params["K"] = intr_to_4x4(intr, device)
            cam_params_for_each_frame[path.stem] = frame_cam_params

        # Store cam params for this camera
        cam_id_to_cam_params[int(camera_id)] = cam_params_for_each_frame

    return cam_id_to_cam_params

def load_pose_dir(
    pose_dir: Path,
    pose_type: str,
    device: torch.device = torch.device("cuda")
) -> Dict[str, Dict[str, torch.Tensor]]:
    # list all files in the pose dir
    pose_files = sorted(list(pose_dir.glob("*.npz")))

    # load all pose params
    frame_to_pose_params = dict()
    for path in pose_files:
        frame_name = path.stem
        if pose_type == "smpl":
            data = SceneDataset._load_smpl(path, device=device)
        elif pose_type == "smplx":
            data = SceneDataset._load_smplx(path, device=device)
        else:
            raise ValueError(f"Unknown pose type: {pose_type}")
        frame_to_pose_params[frame_name] = data

    return frame_to_pose_params


def pose_params_dict_to_batch(pose_params: Dict[str, Dict[str, torch.Tensor]], batch_size=10):
    """
    We are given a dict of frame_name -> pose_params_dict
    where pose params dict is key -> tensor of shape [P, ...]

    We want to convert this to a batched version where we will retuern a list of batches
    where batch is a dict of key -> tensor of shape [B, P, ...] 
    """
    # Union of keys across all frames (some frames may not have optional keys like "contact")
    keys = sorted({k for params in pose_params.values() for k in params.keys()})
    # Find exemplar tensors for missing-key padding
    exemplar = {}
    for k in keys:
        for params in pose_params.values():
            if k in params:
                exemplar[k] = params[k]
                break
        if k not in exemplar:
            raise ValueError(f"Missing exemplar for pose key '{k}'")
    batched_pose_params = []
    batched_fnames = []
    current_batch = {k: [] for k in keys}
    current_fnames = list()
    for frame_name, params in pose_params.items():
        # add another frame to the current batch
        for k in keys:
            if k in params:
                current_batch[k].append(params[k])
            else:
                current_batch[k].append(torch.zeros_like(exemplar[k]))
        current_fnames.append(frame_name)

        # have we filled the batch? 
        if len(current_batch[keys[0]]) == batch_size:
            # stack and add to batched list
            batch = {k: torch.stack(current_batch[k], dim=0) for k in keys}
            batched_pose_params.append(batch)
            batched_fnames.append(current_fnames)

            # reset
            current_batch = {k: [] for k in keys}
            current_fnames = list()


    # add remaining
    if len(current_batch[keys[0]]) > 0:
        batch = {k: torch.stack(current_batch[k], dim=0) for k in keys}
        batched_pose_params.append(batch)
        batched_fnames.append(current_fnames)

    return batched_pose_params, batched_fnames

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class MultiHumanTrainer:
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tuner_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess_dir = Path(cfg.preprocessing_dir)
        self.output_dir = Path(cfg.output_dir)
        self.trn_data_dir = Path(cfg.input_data_dir) / "train" / f"{cfg.scene_name}" / f"{self.cfg.exp_name}"
        self.trn_data_dir.mkdir(parents=True, exist_ok=True)
        self.test_data_dir = Path(cfg.input_data_dir) / "test" / f"{cfg.scene_name}" / f"{self.cfg.exp_name}"
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_params = tuple(cfg.train_params)
        self.sample_every = cfg.sample_every
        self.renderer : GS3DRenderer = build_renderer().to(self.tuner_device)
        self.wandb_run = None

        # Initialize wandb 
        self._init_wandb()

        # Prepare datasets for training and testing
        self._init_train_dataset()
        self._init_test_scene_dir()

        # Preare trainable parameters
        # - (optional) pose tuning
        self._init_pose_tuning()
        # - 3dgs
        self._load_model()

        # Initialize difix
        self._init_nv_generation()
        self.nv_gen_done = False

    
    # ---------------- Datasets ----------------------------
    def _init_train_dataset(self):

        src_cam_id: int = self.cfg.nvs_eval.source_camera_id

        # Fetch training dataset from the specified directories 
        # to the training data dir if not already present
        fetch_data_if_available(
            self.trn_data_dir,
            src_cam_id,
            Path(self.cfg.frames_scene_dir),
            Path(self.cfg.masks_scene_dir),
            Path(self.cfg.cameras_scene_dir),
            Path(self.cfg.smplx_params_scene_dir),
            Path(self.cfg.depths_scene_dir) if self.cfg.use_depth else None,
            Path(self.cfg.smpl_params_scene_dir) if self.cfg.smpl_params_scene_dir is not None else None,
        )

        # Load skip frames if needed
        skip_frames = load_skip_frames(self.trn_data_dir)

        # Create training dataset
        # Note: sample every = 1 because until we augment the data, we assume we use only the input monocular video, and therefore we want to use all the available frames
        self.trn_datasets = list() 
        trn_ds = SceneDataset(
            self.trn_data_dir, 
            src_cam_id, 
            use_depth=self.cfg.use_depth, 
            device=self.tuner_device, 
            sample_every=1,
            skip_frames=skip_frames,
        )
        self.curr_trn_frame_paths = trn_ds.frame_paths
        self.trn_datasets.append(trn_ds)
        self.trn_dataset = trn_ds
        self.trn_render_hw = self.trn_dataset.trn_render_hw
        print(f"Source Camera Training dataset initialised at {self.trn_data_dir} with {len(self.trn_dataset)} images.")

    def _init_test_scene_dir(self):

        # Parse settings
        src_cam_id: int = self.cfg.nvs_eval.source_camera_id
        tgt_cam_ids: List[int] = self.cfg.nvs_eval.target_camera_ids
        all_cam_ids = [src_cam_id] + tgt_cam_ids
        if self.cfg.test_frames_scene_dir is None:
            print("No test frames scene directory specified. Skipping test dataset initialization.")
            return
        frames_dir = Path(self.cfg.test_frames_scene_dir)
        masks_dir = Path(self.cfg.test_masks_scene_dir) if self.cfg.test_masks_scene_dir is not None else None
        cameras_dir = Path(self.cfg.test_cameras_scene_dir) if self.cfg.test_cameras_scene_dir is not None else None
        smplx_params_dir = Path(self.cfg.test_smplx_params_scene_dir) if self.cfg.test_smplx_params_scene_dir is not None else None
        depths_dir = Path(self.cfg.test_depths_scene_dir) if self.cfg.test_depths_scene_dir is not None else None
        smpl_params_dir = Path(self.cfg.test_smpl_params_scene_dir) if self.cfg.test_smpl_params_scene_dir is not None else None
        meshes_dir = Path(self.cfg.test_meshes_scene_dir) if self.cfg.test_meshes_scene_dir is not None else None


        # Fetch testing dataset from the specified directories 
        for cam_id in all_cam_ids:
            fetch_data_if_available(
                self.test_data_dir,
                cam_id,
                frames_dir,
                masks_dir,
                cameras_dir,
                smplx_params_dir,
                depths_dir,
                smpl_params_dir,
                meshes_dir,
            )

    # ---------------- Pose tuning utilities ----------------
    def _init_pose_tuning(self) -> None:
        pose_cfg = self.cfg.pose_tuning
        if not pose_cfg.enable:
            self.pose_tuning_enabled = False
            self.pose_deltas = None
            self.pose_frame_index = {}
            return

        # Map frame names to indices for stable lookup across datasets.
        frame_names = [Path(p).stem for p in self.trn_dataset.frame_paths]
        self.pose_frame_index = {name: idx for idx, name in enumerate(frame_names)}

        # Initialize per-frame pose deltas for selected SMPL-X params.
        sample = self.trn_dataset[0]["smplx_params"]
        params_to_tune = pose_cfg.params
        self.pose_deltas = torch.nn.ParameterDict()
        for key in params_to_tune:
            if key not in sample:
                print(f"Pose tuning: key '{key}' not found in SMPL-X params; skipping.")
                continue
            shape = sample[key].shape  # [P, ...]
            dtype = sample[key].dtype
            self.pose_deltas[key] = torch.nn.Parameter(
                torch.zeros((len(frame_names),) + shape, device=self.tuner_device, dtype=dtype)
            )

        if len(self.pose_deltas) == 0:
            print("Pose tuning enabled but no valid parameters were found; disabling.")
            self.pose_tuning_enabled = False
            self.pose_deltas = None
        else:
            self.pose_tuning_enabled = True
            print("Pose tuning enabled for parameters:", list(self.pose_deltas.keys()))

    def _gather_pose_deltas(
        self, delta_table: torch.Tensor, target: torch.Tensor, frame_names: List[str]
    ) -> torch.Tensor:
        deltas = []
        for name in frame_names:
            idx = self.pose_frame_index.get(str(name), -1)
            if idx < 0:
                raise ValueError(f"Frame name '{name}' not found in pose tuning frame index.")
            else:
                deltas.append(delta_table[idx])
        return torch.stack(deltas, dim=0)

    def _apply_pose_tuning(
        self, smplx_params: Dict[str, torch.Tensor], frame_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        # If disabled, return original params
        if not self.pose_tuning_enabled:
            return smplx_params

        # If enabled, apply deltas to selected params
        tuned = dict(smplx_params)
        for key, delta_table in self.pose_deltas.items():
            # - If not optimising the given key, keep original
            if key not in smplx_params:
                continue
            # - Gather deltas for the current batch frames
            deltas = self._gather_pose_deltas(delta_table, smplx_params[key], frame_names)
            # - Apply deltas
            tuned[key] = smplx_params[key] + deltas

        return tuned

    def _pose_tuning_regularization(self) -> torch.Tensor:
        # If disabled, return zero
        reg_w = self.cfg.pose_tuning.reg_w
        if not self.pose_tuning_enabled or reg_w <= 0.0:
            return torch.zeros((), device=self.tuner_device)

        # If enabled, compute L2 regularization on deltas
        reg = torch.zeros((), device=self.tuner_device)
        for p in self.pose_deltas.values():
            reg = reg + p.pow(2).mean()

        # Finally scale by weight and return
        return reg * reg_w

    @torch.no_grad()
    def _save_pose_tuned_smplx_params(self, dataset: SceneDataset, out_dir: Path) -> None:
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        for batch in loader:
            fnames = batch["frame_name"]
            smplx_params = self._apply_pose_tuning(batch["smplx_params"], fnames)

            betas = smplx_params["betas"]
            expr = smplx_params.get("expr", smplx_params.get("expression"))
            if expr is None:
                expr = torch.zeros(
                    (betas.shape[0], betas.shape[1], 10),
                    device=betas.device,
                    dtype=betas.dtype,
                )
            else:
                # Ensure expression is [B, P, D]
                if expr.dim() == 4:
                    expr = expr[..., 0]
                elif expr.dim() == 2:
                    expr = expr.unsqueeze(0)
                if expr.dim() != 3:
                    raise ValueError(f"Unexpected expression shape {expr.shape}")

            for i, fname in enumerate(fnames):
                save_path = out_dir / f"{fname}.npz"
                np.savez(
                    save_path,
                    betas=betas[i].detach().cpu().numpy(),
                    root_pose=smplx_params["root_pose"][i].detach().cpu().numpy(),
                    body_pose=smplx_params["body_pose"][i].detach().cpu().numpy(),
                    jaw_pose=smplx_params["jaw_pose"][i].detach().cpu().numpy(),
                    leye_pose=smplx_params["leye_pose"][i].detach().cpu().numpy(),
                    reye_pose=smplx_params["reye_pose"][i].detach().cpu().numpy(),
                    lhand_pose=smplx_params["lhand_pose"][i].detach().cpu().numpy(),
                    rhand_pose=smplx_params["rhand_pose"][i].detach().cpu().numpy(),
                    trans=smplx_params["trans"][i].detach().cpu().numpy(),
                    expression=expr[i].detach().cpu().numpy(),
                )


    # ---------------- Model  ------------------------------
    def _load_model(self):

        # Get query points + tranform from neural to zero pose
        # - in theory, to infer query points, we can use betas, face and and joint offset
        # - in practice, I currently use betas only
        smplx_params = self.trn_dataset[0]["smplx_params"]
        self.query_points, self.tranform_mat_neutral_pose = self.renderer.get_query_points(smplx_params, self.tuner_device)

        # Load Canonical 3DGS
        root_gs_model_dir = self.preprocess_dir / "canon_3dgs_lhm"
        self.gs_model_list = torch.load(root_gs_model_dir / "union" / "human3r_gs.pt", map_location=self.tuner_device, weights_only=False)

        if getattr(self.cfg, "use_random_init", False):
            self._randomize_gs_attributes()

    def _randomize_gs_attributes(self) -> None:
        offset_std = 0.002
        scale_min, scale_max = 0.005, 0.02
        opacity_min, opacity_max = 0.05, 0.1
        rgb_min, rgb_max = 0.45, 0.55

        with torch.no_grad():
            for gauss in self.gs_model_list:
                gauss.offset_xyz = torch.randn_like(gauss.offset_xyz) * offset_std

                rotation = torch.zeros_like(gauss.rotation)
                rotation[:, 0] = 1.0
                gauss.rotation = rotation

                gauss.scaling = torch.empty_like(gauss.scaling).uniform_(scale_min, scale_max)
                gauss.opacity = torch.empty_like(gauss.opacity).uniform_(opacity_min, opacity_max)

                if gauss.use_rgb:
                    gauss.shs = torch.empty_like(gauss.shs).uniform_(rgb_min, rgb_max)
                else:
                    gauss.shs = torch.zeros_like(gauss.shs)

    # ---------------- Training utilities ----------------
    def _trainable_tensors(self) -> List[torch.Tensor]:
        params = []
        for pidx, gauss in enumerate(self.gs_model_list):
            print(f"For the {pidx}-th gauss model, adding trainable tensors:")
            for name in self.train_params:
                t = getattr(gauss, name, None)
                if torch.is_tensor(t):
                    t.requires_grad_(True)
                    print(f"    - {name}")
                    params.append(t)

        return params


    def forward(self, batch, background_rgb: Optional[torch.Tensor] = None, apply_tuned_smplx_delta: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Parse batch data
        Ks = batch["K"] # [B, 4, 4],
        c2ws = batch["c2w"] # [B, 4, 4]
        smplx_params = batch["smplx_params"] # each key has value of shape [B, P, ...] where P is num persons
        frame_names = batch["frame_name"]
        if apply_tuned_smplx_delta:
            smplx_params = self._apply_pose_tuning(smplx_params, frame_names)

        # Render target views
        num_views = Ks.shape[0]
        render_h, render_w = self.trn_render_hw
        render_res_list = []
        for view_idx in range(num_views):
            smplx_single_view = {k: v[view_idx] for k, v in smplx_params.items()}
            render_res = self.renderer.animate_and_render(
                self.gs_model_list,
                self.query_points,
                smplx_single_view,
                c2ws[view_idx],
                Ks[view_idx],
                render_h,
                render_w,
                background_rgb=background_rgb,
            )
            render_res_list.append(render_res)

        # Parse outputs
        pred_rgb = torch.cat([res["comp_rgb"] for res in render_res_list], dim=0)  # [B, H, W, 3], 0-1
        pred_mask = torch.cat([res["comp_mask"][..., :1] for res in render_res_list], dim=0)  # [B, H, W, 1]
        pred_depth = torch.cat([res["comp_depth"][..., :1] for res in render_res_list], dim=0)  # [B, H, W, 1]

        return pred_rgb, pred_mask, pred_depth

    def _canonical_regularization(self):
        """Return combined canonical regularization and its components."""
        asap_terms = []
        acap_terms = []
        for gauss in self.gs_model_list:
            # Gaussian Shape Regularization (ASAP): encourage isotropic scales.
            # If scaling is stored in log-space, exp() keeps positivity; otherwise it’s a smooth surrogate.
            scales = gauss.scaling
            scales_pos = torch.exp(scales)
            asap = ((scales_pos - scales_pos.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1).mean()
            asap_terms.append(asap)

            # Positional anchoring (ACAP): hinge on offset magnitude beyond margin.
            offsets = gauss.offset_xyz
            acap = torch.clamp(offsets.norm(dim=-1) - self.cfg.regularization['acap_margin'], min=0.0).mean()
            acap_terms.append(acap)

            # Compute the percentage of Gaussians that are outside the margin
            with torch.no_grad():
                num_outside = (offsets.norm(dim=-1) > self.cfg.regularization['acap_margin']).sum().item()
                total = offsets.shape[0]
                percent_outside = (num_outside / total) * 100.0
                to_log = {"debug/acap_percent_outside": percent_outside}
                if self.wandb_run is not None:
                    wandb.log(to_log)

        asap_loss = torch.stack(asap_terms).mean() * self.cfg.regularization["asap_w"]
        acap_loss = torch.stack(acap_terms).mean() * self.cfg.regularization["acap_w"]

        return asap_loss, acap_loss

    @staticmethod
    def _infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    def _build_loader(self, dataset: SceneDataset) -> DataLoader:
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0, drop_last=False
        )
        return loader

    # ---------------- Logging utilities ----------------
    def _init_wandb(self):
        if not self.cfg.wandb.enable or wandb is None:
            return
        if self.wandb_run is None:
            self.wandb_run = wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config={
                    "epochs": self.cfg.epochs,
                    "batch_size": self.cfg.batch_size,
                    "lr": self.cfg.lr,
                    "weight_decay": self.cfg.weight_decay,
                    "grad_clip": self.cfg.grad_clip,
                    "train_params": self.train_params,
                    "exp_name": self.cfg.exp_name,
                    "scene_name": self.cfg.scene_name,
                    "output_dir": str(self.output_dir),
                    "loss_weights": self.cfg.loss_weights,
                    "sample_every": self.cfg.sample_every,
                },
                name=self.cfg.exp_name,
                tags=list(self.cfg.wandb.tags) if "tags" in self.cfg.wandb else None,
            )

    # ---------------- Training loop ----------------
    def train_loop(self):

        # Prepare dataset and dataloader
        trn_loader = self._build_loader(self.trn_dataset)
        trn_iter = self._infinite_loader(trn_loader)

        # Initialize optimizer
        # - 3dgs setup
        params = self._trainable_tensors()
        opt_groups = [{"params": params, "lr": self.cfg.lr}]
        # - pose params setup
        if self.pose_tuning_enabled:
            pose_lr = float(self.cfg.pose_tuning.lr)
            opt_groups.append({"params": self.pose_deltas.parameters(), "lr": pose_lr})
        # - create optimizer
        optimizer = torch.optim.AdamW(opt_groups, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # Training loop
        for epoch in range(self.cfg.epochs):

            # If we augmented the training data this epoch, rebuild the dataloader
            # using the augmented dataset (with more cameras)
            if self._maybe_augment_training_data(epoch):
                trn_loader = self._build_loader(self.trn_dataset)
                trn_iter = self._infinite_loader(trn_loader)

            # Pre-optimization visualization (epoch 0).
            if self.cfg.eval_pretrain and epoch == 0:
                self.eval_loop(epoch)

            trn_ds_size = len(self.trn_dataset)
            total_ds_size = trn_ds_size 
            n_batches_per_current_epoch = total_ds_size // self.cfg.batch_size
            running_loss = 0.0
            pbar = tqdm(
                total=n_batches_per_current_epoch,
                desc=f"Epoch {epoch + 1}/{self.cfg.epochs}",
                leave=False,
            )

            batch_idx = 0

            while batch_idx < n_batches_per_current_epoch:
                
                # Sample batch
                batch = next(trn_iter)

                # Parse batch data
                fnames = batch["frame_name"] # list of length B
                cam_ids = batch["cam_id"] # [B]
                frames = batch["image"] # [B, H, W, 3],
                masks = batch["mask"] # [B, H, W, 1],
                depths = batch.get("depth", None) # [B, H, W, 1],
                bsize = int(frames.shape[0])
                batched_neutral_pose_transform = self.tranform_mat_neutral_pose.unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
                batch["smplx_params"]["transform_mat_neutral_pose"] = batched_neutral_pose_transform # [B, P, 55, 4, 4]

                # create a mask for samples in the batch that come from src camera
                src_cam_id = self.cfg.nvs_eval.source_camera_id
                cam_id_tensor = batch["cam_id"]  # [B]
                is_src_cam = (cam_id_tensor == src_cam_id)  # [B]

                # Reset gradients
                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                pred_rgb, pred_mask, pred_depth = self.forward(batch)

                # Compute masked ground truth
                mask3 = masks
                if mask3.shape[-1] == 1:
                    mask3 = mask3.repeat(1, 1, 1, 3)
                gt_masked = frames * mask3
                if depths is not None:
                    gt_depth_masked = depths * masks

                # Apply mask also to render if it novel view and this mechanism is enabled
                if self.cfg.apply_mask_to_render_for_nvs:
                    # For src view, use all-ones mask to not affect the render
                    effective_masks = masks.clone().to(pred_rgb.dtype)
                    effective_masks[is_src_cam] = 1.0
                    effective_mask3 = effective_masks
                    if effective_mask3.shape[-1] == 1:
                        effective_mask3 = effective_mask3.repeat(1, 1, 1, 3)
                    pred_rgb = pred_rgb * effective_mask3
                    pred_mask = pred_mask * effective_masks
                    if pred_depth is not None:
                        pred_depth = pred_depth * effective_masks

                # BBOX crop around the union person mask
                if self.cfg.use_bbox_crop:
                    bboxes = get_masks_based_bbox(masks, pad=self.cfg.bbox_pad)
                    loss_pred_rgb = bbox_crop(bboxes, pred_rgb)
                    loss_pred_mask = bbox_crop(bboxes, pred_mask)
                    loss_gt_masked = bbox_crop(bboxes, gt_masked)
                    if depths is not None:
                        loss_pred_depth = bbox_crop(bboxes, pred_depth)
                        loss_gt_depth_masked = bbox_crop(bboxes, gt_depth_masked)
                    loss_masks = bbox_crop(bboxes, masks)
                else:
                    loss_pred_rgb = pred_rgb
                    loss_pred_mask = pred_mask
                    loss_gt_masked = gt_masked
                    if depths is not None:
                        loss_pred_depth = pred_depth
                        loss_gt_depth_masked = gt_depth_masked
                    loss_masks = masks

                # Compute loss
                rgb_loss = self.cfg.loss_weights["rgb"] * F.mse_loss(loss_pred_rgb, loss_gt_masked)
                # - only compute silhouette loss on the source camera.
                #   (Novel-view masks may be bootstrapped and noisy; using them for silhouette supervision
                #   can inject strong, conflicting gradients.)
                if is_src_cam.any():
                    loss_pred_mask_src = loss_pred_mask[is_src_cam]
                    loss_masks_src = loss_masks[is_src_cam]
                    sil_loss = self.cfg.loss_weights["sil"] * F.mse_loss(loss_pred_mask_src, loss_masks_src)
                else:
                    sil_loss = torch.zeros((), device=loss_pred_mask.device, dtype=loss_pred_mask.dtype)
                if depths is not None:
                    depth_loss = self.cfg.loss_weights["depth"] * F.mse_loss(loss_pred_depth, loss_gt_depth_masked)
                else:
                    depth_loss = torch.zeros((), device=loss_pred_rgb.device, dtype=loss_pred_rgb.dtype)
                ssim_val = fused_ssim(_ensure_nchw(loss_pred_rgb), _ensure_nchw(loss_gt_masked), padding="valid")
                ssim_loss = self.cfg.loss_weights["ssim"] * (1.0 - ssim_val)
                asap_loss, acap_loss = self._canonical_regularization()
                reg_loss = asap_loss + acap_loss
                pose_reg_loss = self._pose_tuning_regularization()

                loss = rgb_loss + sil_loss + ssim_loss + reg_loss + depth_loss + pose_reg_loss
                
                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
                optimizer.step()

                # Log the results
                running_loss += loss.item()
                pbar.set_postfix(
                   loss=f"{loss.item():.4f}",
                    asap=f"{asap_loss.item():.4f}",
                    acap=f"{acap_loss.item():.4f}",
                    depth=f"{depth_loss.item():.4f}",
                )

                if self.wandb_run is not None:
                    wandb.log(
                        {
                            "loss/combined": loss.item(),
                            "loss/rgb": rgb_loss.item(),
                            "loss/sil": sil_loss.item(),
                            "loss/ssim": ssim_loss.item(),
                            "loss/depth": depth_loss.item(),
                            "loss/reg": reg_loss.item(),
                            "loss/asap": asap_loss.item(),
                            "loss/acap": acap_loss.item(),
                            "loss/pose_reg": pose_reg_loss.item(),
                            "epoch": epoch + 1,
                        }
                    )

                # space for debug stuff is here
                if batch_idx in [0] and (epoch + 1) % 2 == 0:
                    # - create a joined image from pred_masked and gt_masked for debugging
                    debug_save_dir = self.output_dir / "debug" / self.cfg.exp_name / f"epoch_{epoch+1:04d}" / "rgb"
                    debug_save_dir.mkdir(parents=True, exist_ok=True)
                    overlay = pred_rgb*0.5 + gt_masked*0.5
                    joined_image = torch.cat([pred_rgb, gt_masked, overlay], dim=3)  # Concatenate along width
                    for i in range(joined_image.shape[0]):
                        image = joined_image[i:i+1]
                        frame_name = fnames[i]
                        cam_id = cam_ids[i].item()
                        debug_image_path = debug_save_dir / f"rgb_loss_input_cam{cam_id}_{frame_name}.png"
                        save_image(image.permute(0, 3, 1, 2), str(debug_image_path))

                    # - save depth comparison images
                    debug_save_dir = self.output_dir / "debug" / self.cfg.exp_name / f"epoch_{epoch+1:04d}" / "depth"
                    debug_save_dir.mkdir(parents=True, exist_ok=True)
                    for i in range(pred_depth.shape[0]):
                        frame_name = fnames[i] 
                        cam_id = cam_ids[i].item()
                        save_path = debug_save_dir / f"depth_comparison_cam{cam_id}_frame_{frame_name}.png"
                        save_depth_comparison(pred_depth[i].squeeze(-1), gt_depth_masked[i].squeeze(-1), str(save_path))

                    # - save render and mask comparison images
                    debug_save_dir = self.output_dir / "debug" / self.cfg.exp_name / f"epoch_{epoch+1:04d}" / "gt_input"
                    debug_save_dir.mkdir(parents=True, exist_ok=True)
                    for i in range(masks.shape[0]):
                        frame_name = fnames[i] 
                        cam_id = cam_ids[i].item()
                        save_path = debug_save_dir / f"gt_image_and_mask_comparison_cam{cam_id}_frame_{frame_name}.png"
                        save_gt_image_and_mask_comparison(frames[i], masks[i].squeeze(-1), str(save_path))

                batch_idx += 1
                pbar.update(1)

            # End of epoch
            # - Progress bar stuff
            pbar.close()
            # - Log epoch level metrics
            avg_loss = running_loss / max(1, n_batches_per_current_epoch)
            if self.wandb_run is not None:
                to_log = {
                    "loss/combined_epoch": avg_loss, 
                    "epoch": epoch + 1, 
                    "loss/n_batches_per_epoch": n_batches_per_current_epoch,
                    "loss/total_ds_size": total_ds_size,
                    "loss/trn_ds_size": trn_ds_size,
                }
                wandb.log(to_log)

            # - Run eval loop if neccesary
            if getattr(self.cfg, "eval_every_epoch", 0) > 0 and (epoch + 1) % self.cfg.eval_every_epoch == 0:
                self.eval_loop(epoch + 1)

        # End of training    
        if self.wandb_run is not None:
            self.wandb_run.finish()

    # ---------------- Novel view generation -------------------
    def _maybe_augment_training_data(self, epoch: int) -> bool:
        nv_gen_epoch = getattr(self.cfg, "nv_gen_epoch", 0)
        if self.nv_gen_done or nv_gen_epoch < 0 or epoch != nv_gen_epoch:
            return False

        self._augment_training_data()
        return True

    def _augment_training_data(self) -> None:

        src_cam_id: int = self.cfg.nvs_eval.source_camera_id
        skip_frames = load_skip_frames(self.trn_data_dir)

        # Re-build the trn dataset -> this time using sample_every from config 
        self.trn_datasets = list() 
        trn_ds = SceneDataset(
            self.trn_data_dir, 
            src_cam_id, 
            use_depth=self.cfg.use_depth, 
            device=self.tuner_device, 
            sample_every=self.sample_every,
            skip_frames=skip_frames,
        )
        self.curr_trn_frame_paths = trn_ds.frame_paths
        self.trn_datasets.append(trn_ds)
        self.trn_dataset = trn_ds

        # Generate novel views
        is_nv_gen_done = False
        n_its_nv_gen = 0
        while not is_nv_gen_done:
            is_nv_gen_done = self.novel_view_generation_step()
            n_its_nv_gen += 1
            print(f"Just completed novel view generation step {n_its_nv_gen}. Are we done: {is_nv_gen_done}")


        # Prepare training dataset from all the new cameras
        for cam_id in self.difix_cam_ids_all:
            scene_ds = SceneDataset(
                self.trn_data_dir,
                int(cam_id),
                use_depth=self.cfg.use_depth,
                device=self.tuner_device,
                sample_every=1,
            )
            self.trn_datasets.append(scene_ds)
        self.trn_dataset = ConcatDataset(self.trn_datasets)
        self.nv_gen_done = True

        print(f"Training dataset augmented with novel views from cameras: {self.difix_cam_ids_all}. Currently has {len(self.trn_dataset)} images from {len(self.trn_datasets)} cameras.")

    def _init_nv_generation(self):
        src_cam_id = self.cfg.nvs_eval.source_camera_id
        self.left_cam_id, self.right_cam_id = src_cam_id, src_cam_id
        self.camera_ids = self.cfg.trn_nv_gen.camera_ids
        src_cam_idx = self.camera_ids.index(src_cam_id)
        half_n = (len(self.camera_ids) - 1) // 2
        other_half_n = len(self.camera_ids) - 1 - half_n

        self.from_src_left_traversed_cams = deque((self.camera_ids[src_cam_idx-1::-1] + self.camera_ids[:src_cam_idx:-1])[:half_n])
        self.from_src_right_traversed_cams = deque((self.camera_ids[src_cam_idx+1:] + self.camera_ids[:src_cam_idx])[:other_half_n])
        self.difix_cam_ids_all = list(self.from_src_left_traversed_cams) + list(self.from_src_right_traversed_cams)
        print("NV camera traversal initialized:")
        print(f"    Left traversed cams: {self.from_src_left_traversed_cams}")
        print(f"    Right traversed cams: {self.from_src_right_traversed_cams}")

    @torch.no_grad()
    def prepare_nv_rgb_frames(self, cam_id: int, prev_cam_id: int, difix_pipe: Optional[DifixPipeline] = None):
        """
        Prepare new dataset for the given camera by rendering novel views and refining them using Difix.

        Args:
            difix_pipe: DifixPipeline instance for refinement.
            cam_id: Target camera ID for which to create the new dataset.
            prev_cam_id: Previous camera ID to use as reference for Difix.
        """

        # Load dataset for the previous camera (will use only images + masks as reference for Difix)
        is_prev_cam_always_source = self.cfg.difix.is_prev_cam_always_source
        # - if previous cam is always source cam, use that
        if is_prev_cam_always_source:
            sample_every = self.sample_every
            prev_cam_id = self.cfg.nvs_eval.source_camera_id
            skip_frames = load_skip_frames(self.trn_data_dir)
        # - we just started, so previous cam dataset is the original GT dataset for both left/right
        elif self.left_cam_id == self.right_cam_id:
            sample_every = self.sample_every
            skip_frames = load_skip_frames(self.trn_data_dir)
        # - otherwise, previous cam dataset is the refined dataset from last Difix step
        else:
            sample_every = 1  # use all frames from the refined dataset (they have been already subsampled)
            skip_frames = []  # no skip frames for refined datasetk
        # - finally, load the previous cam dataset
        prev_cam_dataset = SceneDataset(self.trn_data_dir, prev_cam_id, device=self.tuner_device, 
                                        sample_every=sample_every, skip_frames=skip_frames)


        # Load new camera dataset (with dummy images and masks which I will replace with refined later)
        # - generate initial contents of the new cam dataset
        # (if dummy dirs are used, images/masks will be overwritten later)
        refined_frames_save_dir = root_dir_to_image_dir(self.trn_data_dir, cam_id)
        if os.path.exists(refined_frames_save_dir):
            return # already done
        fetch_data_if_available(
            self.trn_data_dir,
            cam_id,
            Path("/dummy/frames/dir"), 
            Path("/dummy/masks/dir"),
            cam_scene_dir=Path(self.cfg.cameras_scene_dir),   
            frame_paths=self.curr_trn_frame_paths,
        )

        # - finally, load the new cam dataset
        new_cam_sample_every = 1 # because curr trn frame paths are already subsampled
        new_cam_dataset = SceneDataset(self.trn_data_dir, cam_id, device=self.tuner_device, sample_every=new_cam_sample_every)
        new_cam_loader = DataLoader(
            new_cam_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
        )

        # Render in batches novel views from target camera
        for batch in tqdm(new_cam_loader, desc=f"Preparing NV RGB frames for cam {cam_id}"):

            # - Update smplx params with neutral pose transform
            frame_indices = batch["frame_idx"] # [B]
            frame_names = batch["frame_name"] # [B]
            bsize = int(batch["image"].shape[0])
            batched_neutral_pose_transform = self.tranform_mat_neutral_pose.unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
            batch["smplx_params"]["transform_mat_neutral_pose"] = batched_neutral_pose_transform # [B, P, 55, 4, 4]

            # - Forward pass to render novel views using GT smplx params and cameras
            pred_rgb, _, _ = self.forward(batch)

            # - Refine rendered images using Difix
            # -- Load reference images from previous camera view
            ref_images = []
            for i in range(bsize):
                sample_dict = prev_cam_dataset[frame_indices[i].item()]
                ref_img, ref_mask = sample_dict["image"], sample_dict["mask"]  # [H, W, 3], [H, W, 1], 0-1
                ref_frame = ref_img * ref_mask # [H, W, 3], 0-1
                ref_images.append(ref_frame.to(self.tuner_device))
            ref_images = torch.stack(ref_images, dim=0)  # [B, H, W, 3]
            # -- Run difix refinement
            refined_rgb = difix_refine(self.cfg.difix, pred_rgb, ref_images, difix_pipe) # [B, H, W, 3]
            # -- Save refined frames
            for i in range(refined_rgb.shape[0]):
                save_path = refined_frames_save_dir / f"{frame_names[i]}.jpg"
                save_image(refined_rgb[i].permute(2, 0, 1), str(save_path))

            # - Debug: save side-by-side comparison of reference, rendered, refined
            difix_debug_dir = root_dir_to_difix_debug_dir(self.trn_data_dir, cam_id)
            difix_debug_dir.mkdir(parents=True, exist_ok=True)
            joined = torch.cat([ref_images, pred_rgb, refined_rgb], dim=2)  # side-by-side along width
            for i in range(joined.shape[0]):
                save_path = difix_debug_dir / f"{frame_names[i]}.png"
                save_image(joined[i].permute(2, 0, 1), str(save_path))

    @torch.no_grad()
    def prepare_nv_depth_maps(self, cam_id: int, allow_overwrite: bool = False):
        
        # Check whether depth maps for this camera already exist
        depths_save_dir = root_dir_to_depth_dir(self.trn_data_dir, cam_id)
        if depths_save_dir.exists() and not allow_overwrite:
            return # already done

        # Commented out for now because DA3 is not good at predicting depth for masked images
        # Use pretrained model from DA3 to infer depth maps
        #print(f"Preparing NV depth maps for cam {cam_id}")
        #script_path = Path(__file__).resolve().parents[1] / "submodules" / "da3" / "nv_inference.py"
        #cmd = [
            #"conda", "run", "-n", "da3",
            #"python", str(script_path),
            #"--scene-dir", str(self.trn_data_dir),
            #"--camera-id", str(cam_id),
        #]
        #subprocess.run(cmd, check=True)

        # Get the dataset for the new camera
        new_cam_sample_every = 1 # because the frames are already subsampled / filtered
        new_cam_dataset = SceneDataset(self.trn_data_dir, cam_id, device=self.tuner_device, sample_every=new_cam_sample_every)
        new_cam_loader = DataLoader(
            new_cam_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
        )

        # Prepare directory to save depth maps
        depths_dir = root_dir_to_depth_dir(self.trn_data_dir, cam_id)
        depths_dir.mkdir(parents=True, exist_ok=True)
        depths_debug_dir = root_dir_to_depths_debug_dir(self.trn_data_dir, cam_id)
        depths_debug_dir.mkdir(parents=True, exist_ok=True)

        # Render in batches novel views from target camera
        for batch in tqdm(new_cam_loader, desc=f"Preparing Depth Maps for cam {cam_id}"):

            # - Update smplx params with neutral pose transform
            frame_indices = batch["frame_idx"] # [B]
            frame_names = batch["frame_name"] # [B]
            bsize = int(batch["image"].shape[0])
            batched_neutral_pose_transform = self.tranform_mat_neutral_pose.unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
            batch["smplx_params"]["transform_mat_neutral_pose"] = batched_neutral_pose_transform # [B, P, 55, 4, 4]

            # - Forward pass to render novel views using smplx params and cameras
            _, _, pred_depth = self.forward(batch) # [B, H, W, 1]


            # Save the raw depth map as a per frame numpy file 
            for i in range(pred_depth.shape[0]):
                dp = pred_depth[i].squeeze(-1).cpu().numpy()  # (H, W)
                frame_name = frame_names[i]
                np.save(depths_dir / (frame_name + ".npy"), dp.astype(np.float32))

                # Debug visualization
                # - Upsample depth to original image resolution
                orig_hw = self.trn_render_hw  # (H,W)
                dp_resized = np.array(Image.fromarray(dp).resize(orig_hw[::-1], resample=Image.BILINEAR))

                create_and_save_depth_debug_vis(
                    pred_depth_np=dp_resized,
                    save_path=str(depths_debug_dir / (frame_name + ".png")),
                )


    @torch.no_grad()
    def prepare_nv_masks(self, cam_id: int):

        # Load dataset for the cam ID to compute masks for
        new_cam_sample_every = 1 # because nv frames are already subsampled / filtered
        mask_estimation_method = self.cfg.mask_estimator_kind
        use_depth = mask_estimation_method == "src_reprojection"
        new_cam_dataset = SceneDataset(
            self.trn_data_dir,
            cam_id,
            use_depth=use_depth,
            device=self.tuner_device,
            sample_every=new_cam_sample_every,
        )
        new_cam_loader = DataLoader(
            new_cam_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
        )

        # If using source reprojection, load source masks + cameras
        src_masks_by_name: Optional[Dict[str, torch.Tensor]] = None
        src_K = None
        src_c2w = None
        if mask_estimation_method == "src_reprojection":
            src_cam_id = self.cfg.nvs_eval.source_camera_id
            skip_frames = load_skip_frames(self.trn_data_dir)
            src_dataset = SceneDataset(
                self.trn_data_dir,
                src_cam_id,
                device=self.tuner_device,
                sample_every=self.sample_every,
                skip_frames=skip_frames,
            )
            src_masks_by_name = {}
            src_K = {}
            src_c2w = {}
            for idx in range(len(src_dataset)):
                sample = src_dataset[idx]
                frame_name = sample["frame_name"]
                src_masks_by_name[frame_name] = sample["mask"]
                src_K[frame_name] = sample["K"]
                src_c2w[frame_name] = sample["c2w"]


        # Compute masks for the nv dataset
        for batch in tqdm(new_cam_loader, desc=f"Estimating masks for cam {cam_id}"):
            # - Parse batch data
            frame_names = batch["frame_name"] # [B]
            rgb_frames = batch["image"] # [B, H, W, 3], 0-1
            if "smplx_params" in batch:
                batch["smplx_params"] = self._apply_pose_tuning(batch["smplx_params"], frame_names)

            # - Estimate binary masks
            mask_estimation_method = self.cfg.mask_estimator_kind
            if mask_estimation_method == "rgb_render_based":
                eps = 10.0 / 255.0
                binary_masks = (rgb_frames > eps).any(dim=-1, keepdim=True).float() # [B, H, W, 1]
            elif mask_estimation_method == "smplx_mesh_based":
                binary_masks = estimate_masks_from_smplx_batch(batch, self.renderer.smplx_model)  # [B, H, W, 1]
            elif mask_estimation_method == "rgb_smplx_band":
                binary_masks = estimate_masks_from_rgb_and_smplx_batch(
                    batch,
                    self.renderer.smplx_model
                )
            elif mask_estimation_method == "src_reprojection":
                seed_masks = estimate_masks_from_smplx_batch(batch, self.renderer.smplx_model)
                assert src_masks_by_name is not None and src_K is not None and src_c2w is not None
                binary_masks = estimate_masks_from_src_reprojection_batch(
                    batch,
                    src_masks_by_name=src_masks_by_name,
                    src_K=src_K,
                    src_c2w=src_c2w,
                    seed_masks=seed_masks,
                )

            # - Save binary masks
            masks_save_dir = root_dir_to_mask_dir(self.trn_data_dir, cam_id)
            masks_save_dir.mkdir(parents=True, exist_ok=True)
            for i in range(binary_masks.shape[0]):
                save_path = masks_save_dir / f"{frame_names[i]}.png"
                save_binary_mask(binary_masks[i].permute(2, 0, 1), str(save_path))

            # - Overlay the estimated masks on refined images and save for debugging
            debug_masks_overlay_dir = root_dir_to_est_masks_debug_dir(self.trn_data_dir, cam_id)
            debug_masks_overlay_dir.mkdir(parents=True, exist_ok=True)
            for i in range(binary_masks.shape[0]):
                overlay = overlay_mask_on_image(
                    rgb_frames[i],
                    binary_masks[i],
                    color_rgb=(1.0, 0.0, 0.0),
                    alpha=0.5,
                )
                save_path = debug_masks_overlay_dir / f"{frame_names[i]}.png"
                save_image(overlay.permute(2, 0, 1), str(save_path))

    @torch.no_grad()
    def novel_view_generation_step(self) -> bool:

        # Select next left/right cameras to process (if any left)
        previous_cams = []
        new_cams = []
        # - Left (only if any left)
        left_changed = False
        if len(self.from_src_left_traversed_cams) > 0: 
            prev_left_cam_id = self.left_cam_id
            new_left_cam_id = self.from_src_left_traversed_cams.popleft() 
            new_cams.append(new_left_cam_id)
            previous_cams.append(prev_left_cam_id)
            left_changed = True
        else:
            new_left_cam_id = self.left_cam_id

        # - Right (only if any left)
        right_changed = False
        if len(self.from_src_right_traversed_cams) > 0:
            prev_right_cam_id = self.right_cam_id
            new_right_cam_id = self.from_src_right_traversed_cams.popleft()
            new_cams.append(new_right_cam_id)
            previous_cams.append(prev_right_cam_id)
            right_changed = True
        else:
            new_right_cam_id = self.right_cam_id

        # If no new cameras to process, return
        if not left_changed and not right_changed:
            return True # returns true to signal that difix is done

        # Generation pipeline
        # - New rgb frames
        # -- Initialize Difix pipeline
        is_enabled = self.cfg.difix.trn_enable
        if not is_enabled:
            difix_pipe = None
        else:
            difix_pipe = DifixPipeline.from_pretrained(
                self.cfg.difix.model_id, 
                trust_remote_code=True, 
                requires_safety_checker=False,
                num_views=2,
            )
            difix_pipe.to(self.tuner_device)
            difix_pipe.set_progress_bar_config(disable=True)

        # -- Generate new novel rgb frames for the new cameras
        for cam_id, prev_cam_id in zip(new_cams, previous_cams):
            self.prepare_nv_rgb_frames(
                cam_id,
                prev_cam_id,
                difix_pipe,
            )
        # -- Delete difix pipe to free memory
        if difix_pipe is not None:
            del difix_pipe
            torch.cuda.empty_cache()

        # - New depth maps for the new cameras
        for cam_id in new_cams:
            self.prepare_nv_depth_maps(cam_id)

        # - New mask maps for the new cameras (if needed)
        if self.cfg.use_estimated_masks:
            for cam_id in new_cams:
                self.prepare_nv_masks(cam_id)
        else:
            for cam_id in new_cams:
                print(f"Preparing NV masks for cam {cam_id} by fetching from scene masks dir")
                were_masks_fetched = fetch_masks_if_exist(Path(self.cfg.masks_scene_dir), self.trn_data_dir, cam_id)
                assert were_masks_fetched, f"Masks for cam {cam_id} not found in {self.cfg.masks_scene_dir}"

        # Update the left and right previous cam ids
        self.left_cam_id = new_left_cam_id
        if left_changed:
            print(f"Updated NV left cam: {prev_left_cam_id} -> {self.left_cam_id}")
        self.right_cam_id = new_right_cam_id
        if right_changed:
            print(f"Updated NV right cam: {prev_right_cam_id} -> {self.right_cam_id}")

        return False


    # ---------------- Evaluation -------------------
    @torch.no_grad()
    def eval_loop_nvs(self, epoch):

        # Parse the evaluation setup
        target_camera_ids: List[int] = self.cfg.nvs_eval.target_camera_ids
        root_save_dir: Path = self.output_dir / "evaluation" / self.cfg.exp_name / f"epoch_{epoch:04d}"
        root_save_dir.mkdir(parents=True, exist_ok=True)

        # Load skip frames
        skip_frames = load_skip_frames(self.test_data_dir)

        # Init source camera dataset for Difix reference images
        src_cam_id: int = self.cfg.nvs_eval.source_camera_id
        src_cam_dataset = SceneDataset(self.test_data_dir, src_cam_id, 
                                       device=self.tuner_device, skip_frames=skip_frames)

        # Init Difix if enabled for the evaluation
        if self.cfg.difix.eval_enable:
            difix_pipe = DifixPipeline.from_pretrained(
                "nvidia/difix",
                trust_remote_code=True,
                requires_safety_checker=False,
                num_views=1,
            )
            difix_pipe.to(self.tuner_device)
            difix_pipe.set_progress_bar_config(disable=True)
        else:
            difix_pipe = None

        # Collector for metrics
        metrics_all_cams_per_frame = list()

        for tgt_cam_id in target_camera_ids:
            # Prepare dataset and dataloader for target camera
            val_dataset = SceneDataset(self.test_data_dir, tgt_cam_id, 
                                       device=self.tuner_device, skip_frames=skip_frames)
            loader = DataLoader(
                val_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
            )

            # Prepare paths where to save results
            save_dir = root_save_dir / f"{tgt_cam_id}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Render in batches novel views from target camera
            for batch in tqdm(loader, desc=f"Evaluation NVS cam {tgt_cam_id}", leave=False):

                # Parse batch data
                frames = batch["image"]             # [B, H, W, 3],
                masks = batch["mask"]               # [B, H, W, 1],
                bsize = int(frames.shape[0])
                batched_neutral_pose_transform = self.tranform_mat_neutral_pose.unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
                batch["smplx_params"]["transform_mat_neutral_pose"] = batched_neutral_pose_transform # [B, P, 55, 4, 4]
                gt_h, gt_w = frames.shape[1], frames.shape[2]
                frame_indices = batch["frame_idx"]      # [B]
                frame_names = batch["frame_name"]      # [B]
                frame_paths = batch["frame_path"]      # List[B]

                # Forward pass to render novel views
                # Note: we use false for apply_tuned_smplx_delta since we are using the gt smplx params for nvs evaluation
                renders, _, _ = self.forward(batch, apply_tuned_smplx_delta=False) 

                # Apply masks to the gt 
                masks3 = masks.repeat(1, 1, 1, 3)
                masked_gt = frames * masks3

                # debug - for difix goes here just in case
                # [Optional] Apply Difix to refine the renders
                if difix_pipe is not None:
                    # - Refine rendered images using Difix
                    # -- Load reference images from previous camera view
                    ref_images = []
                    for i in range(bsize):
                        sample_dict = src_cam_dataset[frame_indices[i].item()]
                        ref_img, ref_mask = sample_dict["image"], sample_dict["mask"]  # [H, W, 3], [H, W, 1], 0-1
                        ref_frame = ref_img * ref_mask # [H, W, 3], 0-1
                        ref_images.append(ref_frame.to(self.tuner_device))
                    ref_images = torch.stack(ref_images, dim=0)  # [B, H, W, 3]
                    # -- Run difix refinement
                    refined_rgb = difix_refine(self.cfg.difix, renders, ref_images, difix_pipe, is_eval=True) # [B, H, W, 3]

                    # - Debug: save side-by-side comparison of reference, rendered, refined
                    difix_debug_dir = save_dir / "difix_debug_comparisons" / f"{tgt_cam_id}"
                    difix_debug_dir.mkdir(parents=True, exist_ok=True)
                    joined = torch.cat([ref_images, renders, refined_rgb, masked_gt], dim=2)  # side-by-side along width
                    for i in range(joined.shape[0]):
                        save_path = difix_debug_dir / f"{frame_names[i]}.png"
                        save_image(joined[i].permute(2, 0, 1), str(save_path))

                    # - Use refined renders for evaluation
                    renders = refined_rgb

                # Combine masked render and masked GT with overlay
                render_vs_gt_dir = save_dir / "render_vs_gt"
                render_vs_gt_dir.mkdir(parents=True, exist_ok=True)
                columns = [renders, masked_gt]
                joined = torch.cat(columns, dim=2)  # side-by-side along width
                for i in range(joined.shape[0]):
                    save_path = render_vs_gt_dir / Path(frame_paths[i]).name
                    save_image(joined[i].permute(2, 0, 1), str(save_path))

                # Save renders on white background for visualization
                render_white_bg_dir = save_dir / "render_white_bg"
                render_white_bg_dir.mkdir(parents=True, exist_ok=True)
                renders_white_bg, _, _ = self.forward(batch, background_rgb=(1.0, 1.0, 1.0), apply_tuned_smplx_delta=False)
                for i in range(renders_white_bg.shape[0]):
                    save_path = render_white_bg_dir / Path(frame_paths[i]).name
                    save_image(renders_white_bg[i].permute(2, 0, 1), str(save_path))

                # Before evaluation, apply downsampling if needed
                if self.cfg.nvs_eval.downscale_factor > 1:
                    ds_factor = self.cfg.nvs_eval.downscale_factor
                    renders = F.interpolate(
                        renders.permute(0, 3, 1, 2),
                        size=(gt_h // ds_factor, gt_w // ds_factor),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                    frames = F.interpolate(
                        frames.permute(0, 3, 1, 2),
                        size=(gt_h // ds_factor, gt_w // ds_factor),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                    masks = F.interpolate(
                        masks.permute(0, 3, 1, 2),
                        size=(gt_h // ds_factor, gt_w // ds_factor),
                        mode="nearest",
                    ).permute(0, 2, 3, 1)
                    masks3 = masks.repeat(1, 1, 1, 3)
                    frames = frames * masks3
                    
                    masks = torch.ones_like(masks)
                    masks3 = masks.repeat(1, 1, 1, 3)
                
                # Save what goes into metrics computation for debugging
                metrics_debug_dir = save_dir / "metrics_debug_inputs"
                metrics_debug_dir.mkdir(parents=True, exist_ok=True)
                masked_renders = renders * masks3
                masked_gt = frames * masks3
                overlayed = 0.5 * masked_renders + 0.5 * masked_gt
                columns = [masked_renders, masked_gt, overlayed]
                joined = torch.cat(columns, dim=2)  # side-by-side along width
                for i in range(joined.shape[0]):
                    save_path = metrics_debug_dir / Path(frame_paths[i]).name
                    save_image(joined[i].permute(2, 0, 1), str(save_path))
                    
                # Compute metrics
                psnr_vals = psnr(
                    images=frames,
                    masks=masks.squeeze(-1),
                    renders=renders,
                )
                ssim_vals = ssim(
                    images=frames,
                    masks=masks.squeeze(-1),
                    renders=renders,
                )
                lpips_vals = lpips(
                    images=frames,
                    masks=masks.squeeze(-1),
                    renders=renders,
                )
                for fname, psnr_val, ssim_val, lpips_val in zip(frame_names, psnr_vals, ssim_vals, lpips_vals):
                    fid = int(fname)
                    metrics_all_cams_per_frame.append(
                        (tgt_cam_id, fid, psnr_val.item(), ssim_val.item(), lpips_val.item())
                    )
            
            # Create a video from render vs gt images using call to ffmpeg
            render_vs_gt_dir = root_save_dir / f"{tgt_cam_id}" / "render_vs_gt"
            video_path = root_save_dir / f"{tgt_cam_id}" / f"cam_{tgt_cam_id}_nvs_epoch_{epoch:04d}.mp4"
            jpg_frames = sorted(p for p in render_vs_gt_dir.glob("*.jpg") if p.stem.isdigit())
            if not jpg_frames:
                raise FileNotFoundError(f"No render_vs_gt frames found in {render_vs_gt_dir}")
            start_number = int(jpg_frames[0].stem)
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-start_number", str(start_number),
                "-framerate", "20",
                "-i", str(render_vs_gt_dir / "%06d.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(video_path),
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
 
        # Save metrics to CSV
        # - save per-frame metrics
        df = pd.DataFrame(metrics_all_cams_per_frame, columns=["camera_id", "frame_id", "psnr", "ssim", "lpips"])
        csv_path = root_save_dir / "metrics_all_cams_per_frame.csv"
        df.to_csv(csv_path, index=False)

        # - save average metrics per camera
        df_avg_cam = df.groupby("camera_id").agg({"psnr": "mean", "ssim": "mean", "lpips": "mean"}).reset_index()
        csv_path_avg_cam = root_save_dir / "metrics_avg_per_camera.csv"
        df_avg_cam.to_csv(csv_path_avg_cam, index=False)

        # - log to wandb the per cam metrics
        if self.cfg.wandb.enable:
            for _, row in df_avg_cam.iterrows():
                to_log = {f"eval_nv/cam_{int(row['camera_id'])}/{metric_name}": row[metric_name] for metric_name in ["psnr", "ssim", "lpips"]}
                to_log["epoch"] = epoch
                wandb.log(to_log)

        # - save overall average metrics excluding self.cfg.nvs_eval.source_camera_id
        df_excluding_source = df[df["camera_id"] != self.cfg.nvs_eval.source_camera_id]
        overall_avg = {
            "psnr": df_excluding_source["psnr"].mean(),
            "ssim": df_excluding_source["ssim"].mean(),
            "lpips": df_excluding_source["lpips"].mean(),
        }
        overall_avg_path = root_save_dir / "novel_view_results.txt"
        with open(overall_avg_path, "w") as f:
            for k, v in overall_avg.items():
                f.write(f"{k}: {v:.4f}\n")

        # - log the overall average metrics to wandb
        if self.cfg.wandb.enable:
            to_log = {f"eval_nv/all_cam/{metric_name}": v for metric_name, v in overall_avg.items()}
            to_log["epoch"] = epoch
            wandb.log(to_log)

        # - log the source training view metrics to wandb
        if self.cfg.wandb.enable:
            df_source = df[df["camera_id"] == self.cfg.nvs_eval.source_camera_id]
            if not df_source.empty:
                source_avg = {
                    "psnr": df_source["psnr"].mean(),
                    "ssim": df_source["ssim"].mean(),
                    "lpips": df_source["lpips"].mean(),
                }
                to_log = {f"eval_nv/trn_cam/{metric_name}": v for metric_name, v in source_avg.items()}
                to_log["epoch"] = epoch
                wandb.log(to_log)

        # Delete difix pipe to free memory
        if difix_pipe is not None:
            del difix_pipe
            torch.cuda.empty_cache()

    @torch.no_grad()
    def eval_loop_segmentation(self, epoch): 

        # Init save directory        
        save_dir : Path = self.output_dir / "evaluation" / self.cfg.exp_name / f"epoch_{epoch:04d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        tgt_cam_ids = self.cfg.nvs_eval.target_camera_ids
        src_cam_id = self.cfg.nvs_eval.source_camera_id
        metrics_per_frame = []
        for tgt_cam_id in tgt_cam_ids:

            # Only eval source camera masks
            if tgt_cam_id != src_cam_id:
                continue

            # Init save directory for this cam
            cam_save_dir : Path = save_dir / f"{tgt_cam_id}"
            cam_save_dir.mkdir(parents=True, exist_ok=True)

            # From cam save dir, init segm_metrics_debug dir
            segm_metrics_debug_dir = cam_save_dir / "segm_metrics_debug_inputs"
            segm_metrics_debug_dir.mkdir(parents=True, exist_ok=True)

            # Prepare gt datasets for given camera
            skip_frames = load_skip_frames(self.trn_data_dir)
            # - For the estimated dataset, use same sample_every and skip frames if evaluating on source camera
            if tgt_cam_id == src_cam_id:
                est_sample_every = self.sample_every
                est_skip_frames = skip_frames
            else:
                est_sample_every = 1
                est_skip_frames = []  # no skip frames for test dataset

            est_dataset = SceneDataset(self.trn_data_dir, tgt_cam_id, 
                                       device=self.tuner_device,sample_every=est_sample_every, skip_frames=est_skip_frames)
            # - For the gt dataset, always use the test dataset with original sample_every and skip frames
            gt_dataset = SceneDataset(self.test_data_dir, tgt_cam_id, 
                                       device=self.tuner_device, sample_every=self.sample_every, skip_frames=skip_frames)

            # Prepare dataloader
            loader = DataLoader(
                est_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
            )


            # Evaluate in batches
            for batch in tqdm(loader, desc=f"Segmentation Eval cam {tgt_cam_id}", leave=False):

                # Parse estimated masks
                est_masks = batch["mask"]               # [B, H, W, 1]
                est_frame_indices = batch["frame_idx"]      # [B]
                fnames = batch["frame_name"]      # [B]
                frames = batch["image"]             # [B, H, W, 3]

                # Parse gt masks
                gt_masks = []
                gt_frame_names = []
                for i in range(est_masks.shape[0]):
                    sample_dict = gt_dataset[est_frame_indices[i].item()]
                    gt_mask = sample_dict["mask"]  # [H, W, 1], 0-1
                    gt_frame_names.append(sample_dict["frame_name"])
                    gt_masks.append(gt_mask.to(self.tuner_device))
                gt_masks = torch.stack(gt_masks, dim=0)  # [B, H, W, 1]
                assert list(gt_frame_names) == list(fnames), "Estimated and GT frame names do not match!"

                # Compute metrics
                segm_metrics = segmentation_mask_metrics(
                    gt_masks=gt_masks.squeeze(-1),
                    pred_masks=est_masks.squeeze(-1),
                )
                segm_iou, segm_recall, segm_f1 = segm_metrics["segm_iou"], segm_metrics["segm_recall"], segm_metrics["segm_f1"]
                for fname, iou, recall, f1 in zip(fnames, segm_iou, segm_recall, segm_f1):
                    fid = int(fname)
                    metrics_per_frame.append(
                        (tgt_cam_id, fid, iou.item(), recall.item(), f1.item())
                    )

                # Save debug inputs for this batch
                gt_masked_frames = get_masked_images(frames, gt_masks)
                est_masked_frames = get_masked_images(frames, est_masks)
                gt_est_masks_overlay = get_gt_est_masks_overlay(gt_masks, est_masks) 
                save_segmentation_debug_figures(
                    gt_masked_frames,
                    est_masked_frames,
                    gt_est_masks_overlay,
                    frame_names=fnames,
                    iou=segm_iou,
                    recall=segm_recall,
                    f1=segm_f1,
                    save_dir=segm_metrics_debug_dir,
                )
                

        # Save metrics to CSV
        # - save per-frame metrics
        df = pd.DataFrame(metrics_per_frame, columns=["camera_id", "frame_id", "iou", "recall", "f1"])
        csv_path = save_dir / "segmentation_metrics_per_frame.csv"
        df.to_csv(csv_path, index=False)

        # - save average metrics per camera
        df_avg_cam = df.groupby("camera_id").agg({"iou": "mean", "recall": "mean", "f1": "mean"}).reset_index()
        csv_path_avg_cam = save_dir / "segmentation_metrics_avg_per_camera.csv"
        df_avg_cam.to_csv(csv_path_avg_cam, index=False)

        # - log to wandb the per cam metrics
        if self.cfg.wandb.enable:
            for _, row in df_avg_cam.iterrows():
                to_log = {f"eval_segm/cam_{int(row['camera_id'])}/{metric_name}": row[metric_name] for metric_name in ["iou", "recall", "f1"]}
                to_log["epoch"] = epoch
                wandb.log(to_log)

        # - save average across all cameras except from source camera
        df_excluding_source = df[df["camera_id"] != self.cfg.nvs_eval.source_camera_id]
        overall_avg = {
            "iou": df_excluding_source["iou"].mean(),
            "recall": df_excluding_source["recall"].mean(),
            "f1": df_excluding_source["f1"].mean(),
        }
        overall_avg_path = save_dir / "segmentation_overall_results.txt"
        with open(overall_avg_path, "w") as f:
            for k, v in overall_avg.items():
                f.write(f"{k}: {v:.4f}\n")
        
        # - log the overall average metrics to wandb
        if self.cfg.wandb.enable:
            to_log = {f"eval_segm/all_cam/{metric_name}": v for metric_name, v in overall_avg.items()}
            to_log["epoch"] = epoch
            wandb.log(to_log)

        # - log the source training view metrics to wandb
        if self.cfg.wandb.enable:
            df_source = df[df["camera_id"] == self.cfg.nvs_eval.source_camera_id]
            if not df_source.empty:
                source_avg = {
                    "iou": df_source["iou"].mean(),
                    "recall": df_source["recall"].mean(),
                    "f1": df_source["f1"].mean(),
                }
                to_log = {f"eval_segm/trn_cam/{metric_name}": v for metric_name, v in source_avg.items()}
                to_log["epoch"] = epoch
                wandb.log(to_log)

    @torch.no_grad()
    def eval_loop_pose_estimation(self, epoch, pose_type="smplx"):

        # Parse source camera ID
        src_cam_id = self.cfg.nvs_eval.source_camera_id

        # Prepare directories to save results
        # - root save directory        
        save_dir : Path = self.output_dir / "evaluation" / self.cfg.exp_name / f"epoch_{epoch:04d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the updated pred SMPL-X params with pose tuning applied
        pose_tuned_smplx_params_dir = save_dir / "smplx"
        pose_tuned_smplx_params_dir.mkdir(parents=True, exist_ok=True)
        skip_frames = load_skip_frames(self.trn_data_dir)
        pred_dataset = SceneDataset(
            self.trn_data_dir,
            src_cam_id,
            skip_frames=skip_frames,
        )
        self._save_pose_tuned_smplx_params(pred_dataset, pose_tuned_smplx_params_dir)

        # Convert the saved pose-tuned SMPL-X params to SMPL format
        # (will create save_dir / "smpl" directory with the converted SMPL params)
        if pose_type == "smpl":
            subprocess.run([
                "bash",
                "submodules/smplx/tools/run_conversion.sh",
                str(save_dir),
                "smplx",
                "smpl",
            ], check=True)
        pose_tuned_smpl_params_dir = save_dir / "smpl"

        # Init datasets
        batch_size = self.cfg.batch_size
        # - pose
        # --- GT
        gt_pose_dir = root_dir_to_smplx_dir(self.test_data_dir) if pose_type == "smplx" else root_dir_to_smpl_dir(self.test_data_dir)
        gt_fname_to_pose_params = load_pose_dir(
            gt_pose_dir, pose_type=pose_type, device=self.tuner_device
        )
        # --- Pred
        pred_pose_dir = pose_tuned_smplx_params_dir if pose_type == "smplx" else pose_tuned_smpl_params_dir
        pred_fname_to_pose_params = load_pose_dir(
            pred_pose_dir, pose_type=pose_type, device=self.tuner_device
        )
        batched_pred_pose_params, batched_pred_fnames = pose_params_dict_to_batch(pred_fname_to_pose_params, batch_size)

        # - cameras
        all_cameras = get_all_scene_dir_cams(self.trn_data_dir, self.tuner_device)
        src_camera_params_over_time = all_cameras[src_cam_id]

        # Get pose layer
        if pose_type == "smplx":
            smplx_layer = getattr(self.renderer.smplx_model, "smplx_layer", None)
            if smplx_layer is None:
                raise RuntimeError("SMPL-X layer not available for pose evaluation.")
            pose_layer = smplx_layer.to(self.tuner_device).eval()
        else:  # smpl
            sample_params = next(iter(pred_fname_to_pose_params.values()), None)
            if sample_params is None:
                raise RuntimeError("No predicted SMPL params available for pose evaluation.")
            num_betas = int(sample_params["betas"].shape[-1])
            model_root = Path(self.renderer.smplx_model.smpl_x.human_model_path)
            pose_layer = smplx.create(
                str(model_root),
                "smpl",
                gender="neutral",
                num_betas=num_betas,
                use_pca=False,
            ).to(self.tuner_device).eval()

        # Evaluate in batches
        metrics_per_frame = []
        has_contact = False
        for pred_pose_params, fnames in tqdm(zip(batched_pred_pose_params, batched_pred_fnames), desc="Evaluating Pose Estimation", leave=False):

            # - Get corresponding gt SMPL-X params
            bsize = len(fnames)
            # -- Loop over batch to get gt SMPL-X params
            gt_pose_params_dict = {fname: gt_fname_to_pose_params[fname] for fname in fnames}
            # -- Stack the returned GT pose params (union keys, fill missing with zeros)
            gt_batched_pose_params, _ = pose_params_dict_to_batch(gt_pose_params_dict, batch_size=bsize)
            gt_pose_params = gt_batched_pose_params[0]
            # -- Track which frames actually have contact annotations
            contact_mask = torch.tensor(
                ["contact" in gt_fname_to_pose_params[fname] for fname in fnames],
                device=self.tuner_device,
            )

            # Get corresponding c2ws
            c2ws = [src_camera_params_over_time[fname]["c2w"] for fname in fnames]
            c2ws = torch.stack(c2ws, dim=0)  # [B, 4, 4]

            # - Compute pose metrics
            # -- MPJPE per frame
            mpjpe_per_frame = compute_pose_mpjpe_per_frame(
                pred_pose_params,
                gt_pose_params,
                pose_layer,
                unit="mm",
                pose_type=pose_type,
            )
            # -- MVE per frame
            mve_per_frame = compute_pose_mve_per_frame(
                pred_pose_params,
                gt_pose_params,
                pose_layer,
                unit="mm",
                pose_type=pose_type,
            )
            # -- Contact distance per frame (if available)
            batch_has_contact = bool(contact_mask.any().item())
            has_contact = has_contact or batch_has_contact
            if batch_has_contact:
                cd_per_frame = compute_pose_contact_distance_per_frame(
                    pred_pose_params,
                    gt_pose_params["contact"],
                    pose_layer,
                    unit="mm",
                    pose_type=pose_type,
                )
            # -- Percentage of Correct Depth Relations
            pcdr = compute_pose_pcdr_per_frame(
                pred_pose_params,
                gt_pose_params,
                c2ws,
                tau=0.15,
                gamma=0.3,
                pose_type=pose_type,
            )

            # - Collect metrics per frame (use NaN for frames without contact)
            for idx, (fname, mpjpe, mve) in enumerate(
                zip(fnames, mpjpe_per_frame, mve_per_frame)
            ):
                fid = int(fname)
                if batch_has_contact and bool(contact_mask[idx].item()):
                    cd_val = cd_per_frame[idx].item()
                else:
                    cd_val = float("nan")
                metrics_per_frame.append(
                    (
                        fid,
                        mpjpe.item(),
                        mve.item(),
                        cd_val,
                        pcdr["pcdr"][idx].item(),
                    )
                )

        # Save metrics to CSV
        # - save per-frame metrics (always include cd_mm; drop if unused)
        df = pd.DataFrame(
            metrics_per_frame,
            columns=[
                "frame_id",
                "mpjpe_mm",
                "mve_mm",
                "cd_mm",
                "pcdr",
            ],
        )
        if not has_contact:
            df = df.drop(columns=["cd_mm"])
        csv_path = save_dir / f"{pose_type}_pose_estimation_metrics_per_frame.csv"
        df.to_csv(csv_path, index=False)

        # - save average metrics
        overall_avg = {
            "mpjpe_mm": df["mpjpe_mm"].mean(),
            "mve_mm": df["mve_mm"].mean(),
            "pcdr": df["pcdr"].mean(),
        }
        if has_contact:
            cd_vals = df["cd_mm"].replace(0.0, np.nan).dropna()
            cd_mean = cd_vals.mean() if not cd_vals.empty else np.nan
            overall_avg["cd_mm"] = 0.0 if pd.isna(cd_mean) else cd_mean
        overall_avg_path = save_dir / f"{pose_type}_pose_estimation_overall_results.txt"
        with open(overall_avg_path, "w") as f:
            for k, v in overall_avg.items():
                f.write(f"{k}: {v:.4f}\n")
        
        # - log the overall average metrics to wandb
        if self.cfg.wandb.enable:
            to_log = {
                f"{pose_type}_eval_pose/mpjpe_mm": overall_avg["mpjpe_mm"],
                f"{pose_type}_eval_pose/mve_mm": overall_avg["mve_mm"],
                f"{pose_type}_eval_pose/pcdr": overall_avg["pcdr"],
            }
            if has_contact and "cd_mm" in overall_avg:
                to_log[f"{pose_type}_eval_pose/cd_mm"] = overall_avg["cd_mm"]
            to_log["epoch"] = epoch
            wandb.log(to_log)

        # - debug
        debug_cd = overall_avg.get("cd_mm", None)
        if debug_cd is not None:
            print(
                "Overall Pose Estimation "
                f"MPJPE (mm): {overall_avg['mpjpe_mm']:.4f}, "
                f"MVE (mm): {overall_avg['mve_mm']:.4f}, "
                f"CD (mm): {overall_avg['cd_mm']:.4f}, "
                f"PCDR: {overall_avg['pcdr']:.2f}"
            )
        else:
            print(
                    "Overall Pose Estimation "
                    f"MPJPE (mm): {overall_avg['mpjpe_mm']:.4f}, "
                    f"MVE (mm): {overall_avg['mve_mm']:.4f}, "
                    f"PCDR: {overall_avg['pcdr']:.2f}"
                )


    @torch.no_grad()
    def eval_loop_reconstruction(self, epoch):

        # Prepare directories to save results
        # - root save directory        
        save_dir_root : Path = self.output_dir / "evaluation" / self.cfg.exp_name / f"epoch_{epoch:04d}"
        save_dir_root.mkdir(parents=True, exist_ok=True)

        # - posed 3DGS save directory (one file per frame)
        save_dir_posed_3dgs: Path = save_dir_root / "posed_3dgs_per_frame"
        save_dir_posed_3dgs.mkdir(parents=True, exist_ok=True)

        # - posed mesh save directory (one merged mesh per frame)
        save_dir_posed_meshes: Path = save_dir_root / "posed_meshes_per_frame"
        save_dir_posed_meshes.mkdir(parents=True, exist_ok=True)

        # - aligned posed mesh save directory (one merged mesh per frame)
        save_dir_aligned_meshes: Path = save_dir_root / "aligned_posed_meshes_per_frame"
        save_dir_aligned_meshes.mkdir(parents=True, exist_ok=True)

        # Fetch configurations
        # - 3dgs to mesh config
        tsdf_camera_files = get_all_scene_dir_cams(self.trn_data_dir, self.tuner_device)
        tsdf_cam_ids = [int(cam_id) for cam_id in tsdf_camera_files.keys()]
        # - reconstruction evaluation config
        icp_cfg = self.cfg["reconstruction_eval"]["icp"]
        metrics_cfg = self.cfg.get("reconstruction_eval", {}).get("metrics", {})
        metrics_n_samples = metrics_cfg.get("n_samples", 50000)
        metrics_units = metrics_cfg.get("units", "cm")
        viou_cfg = self.cfg.get("reconstruction_eval", {}).get("viou", {})
        viou_voxel_size = viou_cfg.get("voxel_size", 0.02)
        viou_padding = viou_cfg.get("padding", 0.05)

        # Init datasets
        skip_frames = load_skip_frames(self.trn_data_dir)
        # - Ground truth dataset
        src_cam_id: int = self.cfg.nvs_eval.source_camera_id # this should not have any effect since we will eval human reconstruction in world coords
        gt_dataset = SceneDataset(
            self.test_data_dir,
            src_cam_id,
            device=self.tuner_device,
            skip_frames=skip_frames,
            use_meshes=True,
            use_masks=False,
            use_cameras=True,
            use_smplx=False,
        )

        # - Prediction dataset 
        pred_dataset = SceneDataset(self.trn_data_dir, src_cam_id, device=self.tuner_device, skip_frames=skip_frames)
        pred_loader = DataLoader(
            pred_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
        )

        # Compute metrics per frame
        metrics_per_frame = []
        for pred_batch in tqdm(pred_loader, desc="Evaluating Reconstruction"):

            # - Parse batch data
            num_views = pred_batch["K"].shape[0]
            fnames = pred_batch["frame_name"]      # [B]
            batched_neutral_pose_transform = self.tranform_mat_neutral_pose.unsqueeze(0).repeat(num_views, 1, 1, 1, 1)
            pred_batch["smplx_params"]["transform_mat_neutral_pose"] = batched_neutral_pose_transform # [B, P, 55, 4, 4]
            smplx_params = pred_batch["smplx_params"] # each key has value of shape [B, P, ...] where P is num persons
            smplx_params = self._apply_pose_tuning(smplx_params, fnames)
            pred_c2ws = [c2w.detach().cpu().numpy() for c2w in pred_batch["c2w"]]  # [B, 4, 4]

            # - Fetch GT meshes directly from the GT dataset for matching frame indices.
            frame_indices = pred_batch["frame_idx"]
            gt_meshes = []
            gt_c2ws = []
            for i in range(frame_indices.shape[0]):
                sample_dict = gt_dataset[frame_indices[i].item()]
                gt_meshes.append(gt_mesh_from_sample(sample_dict["meshes"]))
                gt_c2ws.append(sample_dict["c2w"].detach().cpu().numpy())

            # - For each view, pose 3dgs and convert to meshes (plus save to disk both 3dgs and meshes)
            n_persons = len(self.gs_model_list)
            pred_meshes = []
            pred_meshes_by_person = []
            for view_idx in tqdm(range(num_views), desc="3DGS -> Posed Meshes", total=num_views, leave=False):

                # -- Pose 3dgs
                smplx_single_view = {k: v[view_idx] for k, v in smplx_params.items()}
                all_posed_gs_list = []
                for person_idx in range(n_persons):
                    person_canon_3dgs = self.gs_model_list[person_idx]
                    person_query_pt = self.query_points[person_idx]
                    person_smplx_data = {k: v[person_idx : person_idx + 1] for k, v in smplx_single_view.items()}
                    posed_gs, neutral_posed_gs = self.renderer.animate_single_view_and_person(
                        person_canon_3dgs,
                        person_query_pt,
                        person_smplx_data,
                    )
                    all_posed_gs_list.append(posed_gs)

                fname = fnames[view_idx]
                fname_str = str(fname)
                out_path = save_dir_posed_3dgs / f"{fname_str}.pt"
                posed_gs_state = posed_gs_list_to_serializable_dict(all_posed_gs_list)
                torch.save(posed_gs_state, out_path)

                # -- Posed 3dgs -> posed meshes and save to disk
                render_func = self.renderer.forward_single_view_gsplat
                frame_name = str(fnames[view_idx])
                gs_to_mesh_method = self.cfg.gs_to_mesh_method
                meshes_for_frame = get_meshes_from_3dgs(gs_to_mesh_method, all_posed_gs_list, tsdf_camera_files, 
                                                        tsdf_cam_ids, frame_name, self.trn_render_hw, render_func)
                pred_meshes_by_person.append(meshes_for_frame)
                merged_mesh = merge_mesh_dict(meshes_for_frame)
                pred_meshes.append(merged_mesh)
                merged_path = save_dir_posed_meshes / f"{fname_str}.obj"
                if merged_mesh[0].size and merged_mesh[1].size:
                    trimesh.Trimesh(vertices=merged_mesh[0], faces=merged_mesh[1], process=False).export(merged_path)

            # - Before metric computation, perform ICP alignment (similarity).
            pred_meshes_aligned, pred_meshes_align_T = align_pred_meshes_icp(
                pred_meshes,
                gt_meshes,
                cameras=(pred_c2ws, gt_c2ws),
                n_samples=icp_cfg.get("n_samples", 50000),
                max_iterations=icp_cfg.get("max_iterations", 20),
                threshold=icp_cfg.get("threshold", 1e-5),
            )
            save_aligned_meshes(pred_meshes_aligned, save_dir_aligned_meshes, fnames)

            # - Save aligned per-person meshes (per frame).
            for frame_idx, (meshes_for_frame, T_align) in enumerate(
                zip(pred_meshes_by_person, pred_meshes_align_T)
            ):
                frame_name = str(fnames[frame_idx])
                if frame_name.isdigit():
                    frame_name = f"{int(frame_name):06d}"
                for person_idx in range(n_persons):
                    person_dir = save_dir_aligned_meshes / f"{person_idx:02d}"
                    person_dir.mkdir(parents=True, exist_ok=True)
                    out_path = person_dir / f"{frame_name}.obj"

                    mesh = meshes_for_frame.get(person_idx)
                    if mesh is None:
                        out_path.write_text("")
                        continue
                    vertices, faces = mesh
                    if vertices.size == 0 or faces.size == 0:
                        out_path.write_text("")
                        continue
                    aligned_vertices = apply_se3_to_points(T_align, vertices)
                    trimesh.Trimesh(vertices=aligned_vertices, faces=faces, process=False).export(out_path)

            # - Compute metrics
            v_iou = compute_volumetric_iou(
                pred_meshes_aligned,
                gt_meshes,
                voxel_size=viou_voxel_size,
                padding=viou_padding,
            )
            chamfer = compute_chamfer_distance(
                pred_meshes_aligned,
                gt_meshes,
                n_samples=metrics_n_samples,
                units=metrics_units,
            )
            p2s = compute_p2s_distance(
                pred_meshes_aligned,
                gt_meshes,
                n_samples=metrics_n_samples,
                units=metrics_units,
            )
            normal_consistency = compute_normal_consistency(
                pred_meshes_aligned, gt_meshes, n_samples=metrics_n_samples
            )


            # - Store per-frame metrics
            for idx, fname in enumerate(fnames):
                fid = int(fname)
                row_to_save = (
                    fid,
                    v_iou[idx].item(),
                    chamfer[idx].item(),
                    p2s[idx].item(),
                    normal_consistency[idx].item(),
                )
                metrics_per_frame.append(row_to_save)

                # - Debug: print per-frame metrics
                print(
                    f"Frame {fname}: "
                    f"V-IoU: {v_iou[idx].item():.4f}, "
                    f"Chamfer ({metrics_units}): {chamfer[idx].item():.4f}, "
                    f"P2S ({metrics_units}): {p2s[idx].item():.4f}, "
                    f"Normal Consistency: {normal_consistency[idx].item():.4f} "
                )

        # Save the results
        # - save per-frame metrics
        df = pd.DataFrame(
            metrics_per_frame,
            columns=[
                "frame_id",
                "v_iou",
                f"chamfer_{metrics_units}",
                f"p2s_{metrics_units}",
                "normal_consistency",
            ],
        )
        csv_path = save_dir_posed_meshes.parent / "reconstruction_metrics_per_frame.csv"
        df.to_csv(csv_path, index=False)

        # - save average across frame metrics
        overall_avg = {
            "v_iou": df["v_iou"].mean(),
            f"chamfer_{metrics_units}": df[f"chamfer_{metrics_units}"].mean(),
            f"p2s_{metrics_units}": df[f"p2s_{metrics_units}"].mean(),
            "normal_consistency": df["normal_consistency"].mean(),
        }
        overall_avg_path = save_dir_posed_meshes.parent / "reconstruction_overall_results.txt"
        with open(overall_avg_path, "w") as f:
            for k, v in overall_avg.items():
                f.write(f"{k}: {v:.4f}\n")

        # - log the overall average metrics to wandb
        if self.cfg.wandb.enable:
            to_log = {f"eval_recon/{metric_name}": v for metric_name, v in overall_avg.items()}
            to_log["epoch"] = epoch
            wandb.log(to_log)

#         # - debug (show the overall results)
        # print(
            # "Overall Reconstruction "
            # f"V-IoU: {overall_avg['v_iou']:.4f}, "
            # f"Chamfer ({metrics_units}): {overall_avg[f'chamfer_{metrics_units}']:.4f}, "
            # f"P2S ({metrics_units}): {overall_avg[f'p2s_{metrics_units}']:.4f}, "
            # f"Normal Consistency: {overall_avg['normal_consistency']:.4f} "
        # )


    @torch.no_grad()
    def eval_loop_qualitative(self, epoch):

        # Prepare directories to save results
        # - root save directory        
        save_dir_root : Path = self.output_dir / "evaluation" / self.cfg.exp_name / f"epoch_{epoch:04d}"
        save_dir_root.mkdir(parents=True, exist_ok=True)

        # - posed 3DGS save directory (one file per frame)
        save_dir_posed_3dgs: Path = save_dir_root / "posed_3dgs_per_frame"
        save_dir_posed_3dgs.mkdir(parents=True, exist_ok=True)


        # Prediction dataset 
        src_cam_id: int = self.cfg.nvs_eval.source_camera_id
        skip_frames = load_skip_frames(self.trn_data_dir)
        pred_dataset = SceneDataset(self.trn_data_dir, src_cam_id, device=self.tuner_device, skip_frames=skip_frames, use_depth=True)
        pred_loader = DataLoader(
            pred_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
        )

        # Fetch configs
        # - mesh
        save_dir_posed_meshes: Path = save_dir_root / "posed_meshes_per_frame"
        save_dir_posed_meshes.mkdir(parents=True, exist_ok=True)
        tsdf_camera_files = get_all_scene_dir_cams(self.trn_data_dir, self.tuner_device)
        tsdf_cam_ids = [int(cam_id) for cam_id in tsdf_camera_files.keys()]
        # - smplx
        save_dir_posed_smplx_meshes: Path = save_dir_root / "posed_smplx_meshes_per_frame"
        save_dir_posed_smplx_meshes.mkdir(parents=True, exist_ok=True)
        smplx_layer = getattr(self.renderer.smplx_model, "smplx_layer", None)
        if smplx_layer is None:
            raise RuntimeError("SMPL-X layer not available; cannot export posed SMPL-X meshes.")
        smplx_layer = smplx_layer.to(self.tuner_device).eval()
        smplx_faces = np.asarray(getattr(smplx_layer, "faces"), dtype=np.int32)
        expr_dim = int(getattr(getattr(self.renderer.smplx_model, "smpl_x", None), "expr_param_dim", 0))
        # - cameras
        save_dir_cameras: Path = save_dir_root / "all_cameras" / f"{src_cam_id}"
        save_dir_cameras.mkdir(parents=True, exist_ok=True)
        # - masked depth maps
        save_dir_masked_depth: Path = save_dir_root / "masked_depth_maps"
        save_dir_masked_depth.mkdir(parents=True, exist_ok=True)
        # - images
        save_dir_images: Path = save_dir_root / "images" / f"{src_cam_id}"
        save_dir_images.mkdir(parents=True, exist_ok=True)

        # For each frame in the prediction dataset, save posed 3dgs, posed meshes, cameras, masked depth maps (if available), and images
        for pred_batch in tqdm(pred_loader, desc="Evaluating In-The-Wild"):

            # - Parse batch data
            num_views = pred_batch["K"].shape[0]
            fnames = pred_batch["frame_name"]      # [B]
            batched_neutral_pose_transform = self.tranform_mat_neutral_pose.unsqueeze(0).repeat(num_views, 1, 1, 1, 1)
            pred_batch["smplx_params"]["transform_mat_neutral_pose"] = batched_neutral_pose_transform # [B, P, 55, 4, 4]
            smplx_params = pred_batch["smplx_params"] # each key has value of shape [B, P, ...] where P is num persons
            smplx_params = self._apply_pose_tuning(smplx_params, fnames)


            # - For each view, pose 3dgs and convert to meshes (plus save to disk both 3dgs and meshes)
            n_persons = len(self.gs_model_list)
            pred_meshes_by_person = []
            pred_smplx_meshes_by_person = []

            for view_idx in tqdm(range(num_views), desc="3DGS -> Posed Meshes", total=num_views, leave=False):

                # -- Pose 3dgs
                smplx_single_view = {k: v[view_idx] for k, v in smplx_params.items()}
                all_posed_gs_list = []
                for person_idx in range(n_persons):
                    person_canon_3dgs = self.gs_model_list[person_idx]
                    person_query_pt = self.query_points[person_idx]
                    person_smplx_data = {k: v[person_idx : person_idx + 1] for k, v in smplx_single_view.items()}
                    posed_gs, neutral_posed_gs = self.renderer.animate_single_view_and_person(
                        person_canon_3dgs,
                        person_query_pt,
                        person_smplx_data,
                    )
                    all_posed_gs_list.append(posed_gs)

                fname = fnames[view_idx]
                fname_str = str(fname)
                out_path = save_dir_posed_3dgs / f"{fname_str}.pt"
                posed_gs_state = posed_gs_list_to_serializable_dict(all_posed_gs_list)
                torch.save(posed_gs_state, out_path)

                # -- Save source camera parameters (intrinsics + world-to-camera extrinsics)
                frame_name = fname_str
                if frame_name.isdigit():
                    frame_name = f"{int(frame_name):06d}"
                cam_path = save_dir_cameras / f"{frame_name}.npz"
                intr = pred_batch["K"][view_idx][:3, :3]
                c2w = pred_batch["c2w"][view_idx]
                w2c = torch.inverse(c2w)
                extr = w2c[:3, :4]
                np.savez(
                    cam_path,
                    intrinsics=intr.detach().cpu().numpy()[None, ...],
                    extrinsics=extr.detach().cpu().numpy()[None, ...],
                )

                # -- Save source RGB frame
                frame_img = pred_batch["image"][view_idx]
                frame_np = (frame_img.detach().cpu().numpy() * 255.0).astype(np.uint8)
                Image.fromarray(frame_np).save(save_dir_images / f"{frame_name}.jpg")

                # -- Save masked depth map if available
                depth_batch = pred_batch.get("depth")
                if depth_batch is not None:
                    depth_view = depth_batch[view_idx]
                    mask_batch = pred_batch.get("mask")
                    if mask_batch is not None:
                        depth_view = depth_view * (1.0 - mask_batch[view_idx])
                    depth_np = depth_view.detach().cpu().numpy().astype(np.float32)
                    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
                        depth_np = depth_np[..., 0]
                    np.save(save_dir_masked_depth / f"{frame_name}.npy", depth_np)

                # -- Save per-person posed 3DGS
                for person_idx, person_gs in enumerate(all_posed_gs_list):
                    person_dir = save_dir_posed_3dgs / f"{person_idx:02d}"
                    person_dir.mkdir(parents=True, exist_ok=True)
                    person_path = person_dir / f"{frame_name}.pt"
                    person_state = posed_gs_list_to_serializable_dict([person_gs])
                    torch.save(person_state, person_path)

                # -- Posed 3dgs -> posed meshes and save to disk
                render_func = self.renderer.forward_single_view_gsplat
                gs_to_mesh_method = self.cfg.gs_to_mesh_method
                meshes_for_frame = get_meshes_from_3dgs(gs_to_mesh_method, all_posed_gs_list, tsdf_camera_files, 
                                                        tsdf_cam_ids, frame_name, self.trn_render_hw, render_func)

                pred_meshes_by_person.append(meshes_for_frame)

                # -- SMPL-X parameters -> posed meshes
                smplx_meshes_for_frame: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
                for person_idx in range(n_persons):
                    betas = smplx_single_view["betas"][person_idx : person_idx + 1]
                    root_pose = smplx_single_view["root_pose"][person_idx : person_idx + 1]
                    body_pose = smplx_single_view["body_pose"][person_idx : person_idx + 1]
                    jaw_pose = smplx_single_view["jaw_pose"][person_idx : person_idx + 1]
                    leye_pose = smplx_single_view["leye_pose"][person_idx : person_idx + 1]
                    reye_pose = smplx_single_view["reye_pose"][person_idx : person_idx + 1]
                    lhand_pose = smplx_single_view["lhand_pose"][person_idx : person_idx + 1]
                    rhand_pose = smplx_single_view["rhand_pose"][person_idx : person_idx + 1]
                    trans = smplx_single_view["trans"][person_idx : person_idx + 1]

                    expression = None
                    if expr_dim > 0:
                        expr = smplx_single_view.get("expr")
                        if expr is not None:
                            expr_person = expr[person_idx : person_idx + 1].reshape(1, -1)
                            if expr_person.shape[1] >= expr_dim:
                                expression = expr_person[:, :expr_dim]
                            else:
                                expression = torch.zeros(
                                    (1, expr_dim), device=self.tuner_device, dtype=betas.dtype
                                )
                        else:
                            expression = torch.zeros(
                                (1, expr_dim), device=self.tuner_device, dtype=betas.dtype
                            )

                    smplx_out = smplx_layer(
                        global_orient=root_pose,
                        body_pose=body_pose,
                        jaw_pose=jaw_pose,
                        leye_pose=leye_pose,
                        reye_pose=reye_pose,
                        left_hand_pose=lhand_pose,
                        right_hand_pose=rhand_pose,
                        betas=betas,
                        transl=trans,
                        expression=expression,
                    )
                    vertices = smplx_out.vertices[0].detach().cpu().numpy()
                    smplx_meshes_for_frame[person_idx] = (vertices, smplx_faces)
                pred_smplx_meshes_by_person.append(smplx_meshes_for_frame)

            # - Save per-person meshes and merged mesh (per frame).
            for frame_idx, meshes_for_frame in enumerate(pred_meshes_by_person):
                # -- Parse frame name (we follow the 6digit convention adapted across code base)
                frame_name = str(fnames[frame_idx])
                if frame_name.isdigit():
                    frame_name = f"{int(frame_name):06d}"

                # -- Save per person meshes
                for person_idx in range(n_persons):
                    person_dir = save_dir_posed_meshes / f"{person_idx:02d}"
                    person_dir.mkdir(parents=True, exist_ok=True)
                    out_path = person_dir / f"{frame_name}.obj"

                    mesh = meshes_for_frame.get(person_idx)
                    if mesh is None:
                        out_path.write_text("")
                        continue
                    vertices, faces = mesh
                    if vertices.size == 0 or faces.size == 0:
                        out_path.write_text("")
                        continue
                    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(out_path)

                # -- Save merged mesh
                merged_mesh = merge_mesh_dict(meshes_for_frame)
                merged_path = save_dir_posed_meshes / f"{frame_name}.obj"
                if merged_mesh[0].size and merged_mesh[1].size:
                    trimesh.Trimesh(vertices=merged_mesh[0], faces=merged_mesh[1], process=False).export(merged_path)

                # -- Save SMPL-X meshes
                smplx_meshes_for_frame = pred_smplx_meshes_by_person[frame_idx]
                for person_idx in range(n_persons):
                    person_dir = save_dir_posed_smplx_meshes / f"{person_idx:02d}"
                    person_dir.mkdir(parents=True, exist_ok=True)
                    out_path = person_dir / f"{frame_name}.obj"

                    mesh = smplx_meshes_for_frame.get(person_idx)
                    if mesh is None:
                        out_path.write_text("")
                        continue
                    vertices, faces = mesh
                    if vertices.size == 0 or faces.size == 0:
                        out_path.write_text("")
                        continue
                    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(out_path)

                merged_smplx_mesh = merge_mesh_dict(smplx_meshes_for_frame)
                merged_smplx_path = save_dir_posed_smplx_meshes / f"{frame_name}.obj"
                if merged_smplx_mesh[0].size and merged_smplx_mesh[1].size:
                    trimesh.Trimesh(
                        vertices=merged_smplx_mesh[0],
                        faces=merged_smplx_mesh[1],
                        process=False,
                    ).export(merged_smplx_path)
            

    @torch.no_grad()
    def eval_loop(self, epoch):


        # Quantitative evaluation against provided ground truth
        # - Novel view synthesis evaluation
        if len(self.cfg.nvs_eval.target_camera_ids) == 0:
            print("No target cameras specified for novel view synthesis evaluation. Skipping nvs evaluation.")
        else:
            self.eval_loop_nvs(epoch)

        # - Segmentation evaluation
        if self.cfg.test_masks_scene_dir is None:
            print("No test masks scene directory specified for segmentation evaluation. Skipping segmentation evaluation.")
        else:
            self.eval_loop_segmentation(epoch)
        
        # - Pose estimation evaluation
        if self.cfg.test_smpl_params_scene_dir is None:
            print("No test smpl params scene directory specified for pose estimation evaluation. Skipping pose estimation evaluation.")
        else:
            self.eval_loop_pose_estimation(epoch, pose_type="smplx")
            self.eval_loop_pose_estimation(epoch, pose_type="smpl")

        # - Reconstruction evaluation
        if self.cfg.test_meshes_scene_dir is None:
            print("No test meshes scene directory specified for reconstruction evaluation. Skipping reconstruction evaluation.")
        else:
            self.eval_loop_reconstruction(epoch)

        # Qualitative evaluation (saving posed 3dgs, meshes, cameras)
        # Note: this loop might compute some things for the 2nd time, but for simplicity of things we run it again. 
        self.eval_loop_qualitative(epoch)


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    os.environ["TORCH_HOME"] = str(cfg.torch_home)
    os.environ["HF_HOME"] = str(cfg.hf_home)
    tuner = MultiHumanTrainer(cfg)
    tuner.train_loop()


if __name__ == "__main__":
    main()
