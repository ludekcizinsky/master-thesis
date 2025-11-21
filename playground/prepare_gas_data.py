"""
Usage example:
    python playground/prepare_gas_data.py group_name=v7 scene_name=hi4d_pair01_hug01_cam76 resume=true tids=[0,1] offset_along_z=-0.2
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

from tqdm import tqdm

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

import cv2

def compute_pelvis_y_after_align(
    smpl_server: SMPLServer,
    smpl_param: torch.Tensor,
    R_align: np.ndarray,   # (3,3), world-space rotation you already use
) -> float:
    pelvis_ys = []
    for idx in range(smpl_param.shape[0]):
        out = smpl_server(smpl_param[idx:idx+1], absolute=True)
        joints = out["smpl_jnts"].squeeze(0).detach().cpu().numpy()  # (J,3)
        pelvis = joints[0]  # SMPL root / pelvis

        pelvis_rot = R_align @ pelvis  # (3,)
        pelvis_ys.append(pelvis_rot[1])  # y-component

    # choose how you want to define "pelvis at 0":
    #   - np.mean(pelvis_ys): average pelvis on ground
    #   - pelvis_ys[0]: first person's pelvis exactly at 0
    #   - min(pelvis_ys): lowest pelvis touches ground, others maybe above
    return float(np.mean(pelvis_ys))

def compute_global_alignment(
    smpl_server: SMPLServer,
    smpl_param: torch.Tensor,
    device: str = "cuda",
):
    """
    Returns:
      R_align: (3,3) rotation in world space
      t_align: (3,) translation in world space
    Such that aligned_verts = (R_align @ (verts - t_align))
    is used for *all* persons.
    """

    pelvis_idx = 0  # SMPL joint 0 is pelvis in standard SMPL

    pelvis_list = []
    # If you want to define orientation from one reference person, use idx_ref = 0
    idx_ref = 0
    R_align = None

    for idx in range(smpl_param.shape[0]):
        out = smpl_server(smpl_param[idx:idx+1], absolute=True)
        verts = out["smpl_verts"].squeeze(0)             # (V,3) torch
        joints = out["smpl_jnts"].squeeze(0)           # (J,3) torch, if you have it

        pelvis_world = joints[pelvis_idx].detach().cpu().numpy()
        pelvis_list.append(pelvis_world)

        if idx == idx_ref:
            # Example: make this person’s pelvis-up vector align with +Y
            head_idx = 15  # any upper body joint (e.g. neck/head) – adjust as you like
            up_vec = (joints[head_idx] - joints[pelvis_idx]).detach()
            up_vec = up_vec / (up_vec.norm() + 1e-8)

            world_up = torch.tensor([0.0, 1.0, 0.0], device=up_vec.device)
            v = up_vec
            u = world_up

            # rotation that maps v -> u
            cross = torch.cross(v, u)
            sin_theta = cross.norm()
            cos_theta = torch.dot(v, u)
            if sin_theta < 1e-6:
                R_align = torch.eye(3, device=device)
            else:
                k = cross / (sin_theta + 1e-8)
                K = torch.tensor(
                    [[0, -k[2], k[1]],
                     [k[2], 0, -k[0]],
                     [-k[1], k[0], 0]], device=device
                )
                R_align = (
                    torch.eye(3, device=device) * cos_theta
                    + K * sin_theta
                    + (1 - cos_theta) * torch.ger(k, k)
                )

    # translation: average pelvis over all people
    pelvis_center = np.mean(np.stack(pelvis_list, axis=0), axis=0)  # (3,)
    t_align = torch.from_numpy(pelvis_center).to(device=device, dtype=torch.float32)

    if R_align is None:
        R_align = torch.eye(3, device=device)

    return R_align, t_align



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

    # compute pelvis->canonical transform for person 0
    R_align, t_align = compute_global_alignment(smpl_server, smpl_param, device="cuda")
    pelvis_y = compute_pelvis_y_after_align(smpl_server, smpl_param, R_align.cpu().numpy())


    verts_all: List[np.ndarray] = []
    faces_all: List[np.ndarray] = []
    vert_offset = 0
    for idx in range(smpl_param.shape[0]):
        out = smpl_server(smpl_param[idx : idx + 1], absolute=True)
        verts = out["smpl_verts"].squeeze(0).detach().cpu().numpy()

        # apply pelvis->canonical transform
        verts = (R_align.cpu().numpy() @ verts.T).T + t_align.cpu().numpy() # (V,3)
        verts[:,1] -= pelvis_y   # align pelvis y to 0
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
    add_coordinate_frame(scene, center=np.array([0.0,0.0,0.0]), scale=0.2)

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

def cam_center_from_w2c(w2c: torch.Tensor):
    R = w2c[0,:3,:3]
    t = w2c[0,:3,3]
    C = -R.T @ t     # camera center in world coords
    return C

def add_coordinate_frame(scene, center, scale=0.3):
    """
    Adds an RGB coordinate frame at `center`.
    X = red, Y = green, Z = blue.
    Works with older trimesh + pyrender (face colors allowed).
    """
    # Create axis mesh using trimesh (face-colored)
    axis_tm = trimesh.creation.axis(
        origin_size=scale * 0.1,
        axis_length=scale
    )

    # IMPORTANT: force pyrender to accept face colors
    axis_pr = pyrender.Mesh.from_trimesh(axis_tm, smooth=False)

    # Add to scene
    node = pyrender.Node(
        mesh=axis_pr,
        translation=np.asarray(center, dtype=np.float32)
    )
    scene.add_node(node)

def create_new_w2c_orbit(
    rot_degree: float,
    w2c: torch.Tensor,                               # (1,4,4) or (4,4) world->camera (OpenCV)
    scene_center: Union[np.ndarray, torch.Tensor],   # (3,)
    offset_along_z: float = 0.3,
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
    offset_vec = -z_axis * offset_along_z  # (3,)
    new_w2c[:3, 3] += offset_vec

    return new_w2c.unsqueeze(0)  # (1,4,4)


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "configs"), config_name="train")
def main(cfg: DictConfig) -> None:
    render_output_dir = Path(f"/scratch/izar/cizinsky/thesis/playground_output/{cfg.scene_name}")

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

    # compute rotated camera poses 
    degrees = [45, 90, 135, 180, 225, 270, 315]
    for deg in degrees:
        w2c_rotated = create_new_w2c_orbit(deg, w2c[0], scene_center, device=device, offset_along_z=cfg.offset_along_z)
        all_w2cs.append(w2c_rotated)


    for idx, w2c in tqdm(enumerate(all_w2cs), total=len(all_w2cs), desc="Rendering views"):

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
        image_dir = render_output_dir / "rgb"/ f"cam_{idx}"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{frame_id:04d}.png"
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
        normal_dir = render_output_dir / "normals"/ f"cam_{idx}"
        normal_dir.mkdir(parents=True, exist_ok=True)
        normal_path = normal_dir / f"{frame_id:04d}.png"
        normal_image.save(normal_path)

    print(f"Rendered images saved to: {render_output_dir}")

if __name__ == "__main__":
    main()
