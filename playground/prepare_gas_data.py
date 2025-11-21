"""
Usage example:
    playground/prepare_gas_data.py group_name=v7_default_corrected scene_name=taichi resume=true
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union
from omegaconf import DictConfig

import numpy as np
import torch
import hydra
from PIL import Image
import pyrender
import trimesh

# Make training and evaluation modules available
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_HOME", "/scratch/izar/cizinsky/.cache")
os.environ.setdefault("HF_HOME", "/scratch/izar/cizinsky/.cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
# Force EGL for headless pyrender
os.environ["PYOPENGL_PLATFORM"] = "egl"

from training.helpers.checkpointing import ModelCheckpointManager
from training.helpers.dataset import build_training_dataset
from training.helpers.model_init import create_splats_with_optimizers
from training.helpers.render import render_splats
from training.helpers.smpl_utils import update_skinning_weights
from training.smpl_deformer.smpl_server import SMPLServer


@torch.no_grad()
def render_smpl_normal_map(
    smpl_server: SMPLServer,
    smpl_param: torch.Tensor,
    w2c_cv: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
) -> np.ndarray:
    """Render a normal map of all persons' posed SMPL meshes."""

    # Get vertices and faces for all persons
    faces_raw = smpl_server.faces
    if torch.is_tensor(faces_raw):
        faces_np = faces_raw.detach().cpu().numpy()
    else:
        faces_np = np.asarray(faces_raw)
    if faces_np.ndim == 3:
        faces_np = faces_np[0]
    faces_np = faces_np.astype(np.int64)

    verts_all: List[np.ndarray] = []
    faces_all: List[np.ndarray] = []
    vert_offset = 0
    for idx in range(smpl_param.shape[0]):
        out = smpl_server(smpl_param[idx : idx + 1], absolute=True)
        verts = out["smpl_verts"].squeeze(0).detach().cpu().numpy()
        verts_all.append(verts)
        faces_all.append(faces_np + vert_offset)
        vert_offset += verts.shape[0]

    verts_cat = np.concatenate(verts_all, axis=0)
    faces_cat = np.concatenate(faces_all, axis=0)

    # Create trimesh mesh from all persons' meshes
    mesh = trimesh.Trimesh(vertices=verts_cat, faces=faces_cat, process=False)

    # Define IntrinsicsCamera in pyrender
    K_np = K.squeeze(0).detach().cpu().numpy()
    fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
    cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

    # Convert OpenCV camera pose (world->camera, z forward) to OpenGL (camera looks -z, y up)
    w2c_np = w2c_cv.squeeze(0).detach().cpu().numpy()
    c2w_cv = np.linalg.inv(w2c_np)
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    c2w_gl = c2w_cv @ cv_to_gl

    # Compute vertex normals in world space
    normals = mesh.vertex_normals  # (V, 3)

    # Convert normals from [-1,1] to [0,255]
    n = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    n_vis = ((n * 0.5) + 0.5) * 255
    n_vis = n_vis.astype(np.uint8)

    # Attach normals as per-vertex colors
    mesh.visual.vertex_colors = np.hstack([n_vis, 255 * np.ones((n_vis.shape[0],1), np.uint8)])

    # IMPORTANT: build pyrender mesh WITHOUT any material
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

    # Setup scene
    scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[1,1,1,1])
    scene.add(pr_mesh)
    scene.add(cam, pose=c2w_gl)

    # Render and extract normal map
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
    normal_map = color[...,:3]   # RGB normal map
    renderer.delete()

    return normal_map


@torch.no_grad()
def compute_scene_center(
    smpl_server: SMPLServer, smpl_param: torch.Tensor
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Compute per-person 3D bounding boxes and the merged scene center.

    Returns:
        scene_center: (3,) center of the merged bounding box across all persons, np.float64.
        person_bboxes: list of (bbox_min, bbox_max) for each person, each np.ndarray of shape (3,).
    """
    bbox_mins: List[np.ndarray] = []
    bbox_maxs: List[np.ndarray] = []

    if smpl_param.shape[0] == 0:
        raise ValueError("No SMPL params provided to compute_scene_center.")

    for idx in range(smpl_param.shape[0]):
        out = smpl_server(smpl_param[idx : idx + 1], absolute=True)
        verts = out["smpl_verts"].squeeze(0).detach().cpu().numpy()
        bbox_mins.append(verts.min(axis=0))
        bbox_maxs.append(verts.max(axis=0))

    merged_min = np.min(np.stack(bbox_mins, axis=0), axis=0)
    merged_max = np.max(np.stack(bbox_maxs, axis=0), axis=0)
    scene_center = (merged_min + merged_max) / 2.0

    person_bboxes = list(zip(bbox_mins, bbox_maxs))
    return scene_center, person_bboxes


def create_new_w2c_orbit(
    rot_degree: float,
    w2c: torch.Tensor,                               # (1,4,4) or (4,4) world->camera (OpenCV)
    scene_center: Union[np.ndarray, torch.Tensor],   # (3,)
    device: str = "cuda",
) -> torch.Tensor:
    """
    Rotate the camera around `scene_center` by `rot_degree` (in degrees)
    around the world Y axis, keeping the camera looking at `scene_center`.

    Returns:
        new_w2c: (1,4,4) tensor, world->camera (OpenCV convention).
    """
    # --- prep inputs ---
    w2c = w2c.to(device=device, dtype=torch.float32)
    if w2c.ndim == 3:
        # assume shape (1,4,4)
        w2c = w2c[0]

    if isinstance(scene_center, np.ndarray):
        scene_center = torch.from_numpy(scene_center)
    scene_center = scene_center.to(device=device, dtype=torch.float32)  # (3,)

    # --- extract camera pose in world coordinates ---
    # world->cam: x_c = R_wc x_w + t_wc
    R_wc = w2c[:3, :3]            # (3,3)
    t_wc = w2c[:3, 3]             # (3,)
    # camera center in world coords: C = -R_wc^T * t_wc
    cam_pos = -R_wc.transpose(0, 1) @ t_wc   # (3,)

    # --- build world-space rotation around Y ---
    angle = np.deg2rad(rot_degree)
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    R_y = torch.tensor(
        [
            [ cos_a, 0.0,  sin_a],
            [ 0.0,   1.0,  0.0  ],
            [-sin_a, 0.0,  cos_a],
        ],
        dtype=torch.float32,
        device=device,
    )

    # --- rotate camera position around scene_center ---
    v = cam_pos - scene_center          # vector from center to camera
    v_rot = R_y @ v                     # rotated vector
    new_cam_pos = scene_center + v_rot  # new world-space camera position

    # --- build a "look-at" camera that looks at scene_center ---
    up_world = torch.tensor([0.0, 1.0, 0.0], device=device)

    # camera forward direction (OpenCV: z forward)
    z_axis = (scene_center - new_cam_pos)
    z_axis = z_axis / (z_axis.norm() + 1e-8)

    # right and up (use torch.linalg.cross to avoid deprecation warning)
    x_axis = torch.linalg.cross(z_axis, up_world)
    x_axis = x_axis / (x_axis.norm() + 1e-8)
    y_axis = torch.linalg.cross(z_axis, x_axis)
    y_axis = y_axis / (y_axis.norm() + 1e-8)

    # columns of rotation matrix are basis vectors in world coords (camera-to-world)
    R_c2w = torch.stack([x_axis, y_axis, z_axis], dim=1)  # (3,3)

    # --- convert camera-to-world -> world-to-camera analytically ---
    # world->cam rotation is transpose, translation is -R^T * C
    R_new_wc = R_c2w.transpose(0, 1)           # (3,3)
    t_new_wc = -R_new_wc @ new_cam_pos         # (3,)

    new_w2c = torch.eye(4, dtype=torch.float32, device=device)
    new_w2c[:3, :3] = R_new_wc
    new_w2c[:3, 3] = t_new_wc
    new_w2c[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)

    # add offset to move camera slightly back along its viewing direction
    offset_dist = 0.3
    offset_vec = -z_axis * offset_dist  # (3,)
    new_w2c[:3, 3] += offset_vec

    return new_w2c.unsqueeze(0)  # (1,4,4)


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

    all_w2cs = []
    frame_id = 30
    sample = dataset[frame_id]
    w2c = sample["M_ext"].to(device).unsqueeze(0)
    all_w2cs.append(w2c)
    K = sample["K"].to(device).unsqueeze(0)
    H, W = int(sample["H"]), int(sample["W"])
    smpl_param = smpl_params_map[frame_id].to(device)


    # compute scene center and per-person bounding boxes
    smpl_server = SMPLServer(gender=cfg.smpl_gender)
    scene_center, person_bboxes = compute_scene_center(smpl_server, smpl_param)

    # compute 45 degree rotated camera pose for better visualization
    w2c_rotated = create_new_w2c_orbit(45, w2c[0], scene_center, device=device)
    all_w2cs.append(w2c_rotated)


    for idx, w2c in enumerate(all_w2cs):

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
        image_path = render_output_dir / f"render_frame_{frame_id:04d}_cam_{idx}.png"
        Image.fromarray(image).save(image_path)



        normal_rgb = render_smpl_normal_map(
            smpl_server=smpl_server,
            smpl_param=smpl_param,
            w2c_cv=w2c,
            K=K,
            H=H,
            W=W,
        )
        normal_image = Image.fromarray(normal_rgb.astype(np.uint8))
        normal_path = render_output_dir / f"render_frame_{frame_id:04d}_cam_{idx}_normals.png"
        normal_image.save(normal_path)


if __name__ == "__main__":
    main()
