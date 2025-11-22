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

def center_crop_512(img: Image.Image) -> Image.Image:
    """
    Center crop a PIL image to 512x512.
    Raises if the input is smaller than the crop size.
    """
    target = 512
    width, height = img.size
    if width < target or height < target:
        raise ValueError(f"Image is too small for a 512x512 crop: got {width}x{height}")

    left = (width - target) // 2
    top = (height - target) // 2
    right = left + target
    bottom = top + target
    return img.crop((left, top, right, bottom))

def compute_scene_up_from_joints(
    joints_list: List[np.ndarray],
    pelvis_index: int = 0,
    head_index: int = 15,
) -> np.ndarray:
    """
    Computes a stable 'scene up' vector from a list of SMPL joint arrays.

    Args:
        joints_list: List of (J,3) joints arrays for each person.
        pelvis_index: index of SMPL pelvis joint (usually 0).
        head_index: index of SMPL head joint (usually 15).

    Returns:
        up_vec: (3,) normalized numpy array representing scene up direction.
    """

    up_vectors = []

    for joints in joints_list:
        pelvis = joints[pelvis_index]
        head = joints[head_index]
        up = head - pelvis
        norm = np.linalg.norm(up)
        if norm > 1e-5:
            up_vectors.append(up / norm)

    if len(up_vectors) == 0:
        # fallback: default world Y 
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # average across people
    up_avg = np.mean(np.stack(up_vectors, axis=0), axis=0)
    up_avg = up_avg / (np.linalg.norm(up_avg) + 1e-8)

    return up_avg.astype(np.float64)

def add_scene_up_axis(
    scene: pyrender.Scene,
    center: np.ndarray,     # (3,)
    up_vec: np.ndarray,     # (3,) unit
    axis_length: float = 0.5
):
    """
    Add an XYZ axis marker at `center` where the Y-axis is aligned with `up_vec`.
    This lets you visually inspect the estimated 'scene up' in pyrender.
    """
    center = np.asarray(center, dtype=np.float64)
    up_vec = np.asarray(up_vec, dtype=np.float64)
    up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-8)

    # Build an orthonormal basis: y = up_vec
    y_axis = up_vec

    # Choose an arbitrary vector not parallel to y_axis to build x_axis
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(tmp, y_axis)) > 0.9:
        tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    x_axis = np.cross(y_axis, tmp)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    # Construct rotation matrix whose columns are basis vectors
    R = np.stack([x_axis, y_axis, z_axis], axis=1)  # (3,3)

    # 4x4 transform moving axis marker to scene_center with our orientation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = center

    # Create an axis mesh and apply transform
    axis_tm = trimesh.creation.axis(
        origin_size=0.02,
        axis_length=axis_length,
        transform=T,
    )

    # Important: use smooth=False to avoid the face-color/smooth conflict error
    axis_pr = pyrender.Mesh.from_trimesh(axis_tm, smooth=False)
    scene.add(axis_pr)


@torch.no_grad()
def render_smpl_normal_map(
    smpl_server: SMPLServer,
    smpl_param: torch.Tensor,
    w2c_cv: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    people_center: Optional[np.ndarray] = None,
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
    # add_coordinate_frame(scene, center=np.array([0.0,0.0,0.0]), scale=0.2)
    if people_center is not None:
        add_coordinate_frame(scene, center=people_center, scale=0.2)

    # compute joints for each person to get scene up vector
#     joints_list: List[np.ndarray] = []
    # for idx in range(smpl_param.shape[0]):
        # out = smpl_server(smpl_param[idx : idx + 1], absolute=True)
        # joints = out["smpl_jnts"].squeeze(0).detach().cpu().numpy()
        # joints_list.append(joints)

    # scene_up = compute_scene_up_from_joints(joints_list)
    # add_scene_up_axis(scene, center=np.array([0.0,0.0,0.0]), up_vec=scene_up, axis_length=0.3)


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
    w2c: torch.Tensor,                                 # (1,4,4) or (4,4)
    scene_center: Union[np.ndarray, torch.Tensor],     # (3,)
    scene_up: Union[np.ndarray, torch.Tensor],         # (3,) unit vector
    offset_along_z: float = 0.3,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Orbit the camera around `scene_center` by `rot_degree` degrees using
    `scene_up` as the rotation axis. Builds world->camera (OpenCV) matrix.

    Args:
        scene_up: (3,) unit vector defining 'up' in the scene coordinates.
    """
    # --- prep inputs ---
    w2c = w2c.to(device=device, dtype=torch.float32)
    if w2c.ndim == 3:
        w2c = w2c[0]  # unwrap (1,4,4)

    if isinstance(scene_center, np.ndarray):
        scene_center = torch.from_numpy(scene_center)
    scene_center = scene_center.to(device=device, dtype=torch.float32)

    if isinstance(scene_up, np.ndarray):
        scene_up = torch.from_numpy(scene_up)
    scene_up = scene_up.to(device=device, dtype=torch.float32)
    scene_up = scene_up / (scene_up.norm() + 1e-8)  # ensure unit

    # --- extract camera position C (world coordinates) ---
    R_wc = w2c[:3, :3]     # world->cam
    t_wc = w2c[:3, 3]
    cam_pos = -R_wc.transpose(0, 1) @ t_wc   # C = -R^T t

    # =====================================================
    # 1) ROTATE CAMERA POSITION AROUND scene_up (axis-angle)
    # =====================================================
    angle = np.deg2rad(rot_degree)
    ux, uy, uz = scene_up

    cos_a = float(torch.cos(torch.tensor(angle)))
    sin_a = float(torch.sin(torch.tensor(angle)))

    # Rodrigues rotation matrix for axis=scene_up
    R_axis = torch.tensor([
        [cos_a + ux*ux*(1-cos_a),     ux*uy*(1-cos_a) - uz*sin_a, ux*uz*(1-cos_a) + uy*sin_a],
        [uy*ux*(1-cos_a) + uz*sin_a,  cos_a + uy*uy*(1-cos_a),     uy*uz*(1-cos_a) - ux*sin_a],
        [uz*ux*(1-cos_a) - uy*sin_a,  uz*uy*(1-cos_a) + ux*sin_a, cos_a + uz*uz*(1-cos_a)]
    ], dtype=torch.float32, device=device)

    # apply rotation around scene center
    v = cam_pos - scene_center
    v_rot = R_axis @ v
    new_cam_pos = scene_center + v_rot

    # =====================================================
    # 2) BUILD CAMERA ORIENTATION (c2w) USING SCENE UP
    # =====================================================
    up_world = scene_up  # now YOUR estimated up vector

    # OpenCV: forward camera dir = +Z
    z_axis = (scene_center - new_cam_pos)
    z_axis = z_axis / (z_axis.norm() + 1e-8)

    # right axis
    x_axis = torch.linalg.cross(z_axis, up_world)
    x_axis = x_axis / (x_axis.norm() + 1e-8)

    # corrected up
    y_axis = torch.linalg.cross(z_axis, x_axis)
    y_axis = y_axis / (y_axis.norm() + 1e-8)

    # camera-to-world orientation
    R_c2w = torch.stack([x_axis, y_axis, z_axis], dim=1)

    # =====================================================
    # 3) CONVERT c2w → w2c  (OpenCV)
    # =====================================================
    R_new_wc = R_c2w.transpose(0, 1)            # inverse rotation
    t_new_wc = -R_new_wc @ new_cam_pos          # inverse translation

    new_w2c = torch.eye(4, dtype=torch.float32, device=device)
    new_w2c[:3, :3] = R_new_wc
    new_w2c[:3, 3] = t_new_wc

    # =====================================================
    # 4) SMALL OFFSET BACKWARD ALONG VIEW DIRECTION
    # ====================================================
    offset_vec = -z_axis * offset_along_z
    new_w2c[:3, 3] += offset_vec

    return new_w2c.unsqueeze(0)



@hydra.main(version_base=None, config_path=str(REPO_ROOT / "configs"), config_name="train")
def main(cfg: DictConfig) -> None:
    render_output_dir = Path(f"/scratch/izar/cizinsky/thesis/playground_output/{cfg.scene_name}")

    device = "cuda"
    ckpt_manager = ModelCheckpointManager(
        scene_output_dir=Path(cfg.output_dir),
        group_name=cfg.group_name,
        tids=cfg.tids,
        ckpt_epoch=50,
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
    frame_id = 0
    sample = dataset[frame_id]
    w2c = sample["M_ext"].to(device).unsqueeze(0)
    K = sample["K"].to(device).unsqueeze(0)
    H, W = int(sample["H"]), int(sample["W"])
    smpl_param = smpl_params_map[frame_id].to(device)

    # compute scene center and per-person bounding boxes
    smpl_server = SMPLServer(gender=cfg.smpl_gender)
    scene_center, person_bboxes = compute_scene_center(smpl_server, smpl_param)

    joints_list: List[np.ndarray] = []
    for idx in range(smpl_param.shape[0]):
        out = smpl_server(smpl_param[idx : idx + 1], absolute=True)
        joints = out["smpl_jnts"].squeeze(0).detach().cpu().numpy()
        joints_list.append(joints)

    scene_up = compute_scene_up_from_joints(joints_list)

    # compute rotated camera poses 
    degrees = [45, 90, 135, 180, 225, 270, 315, 360]
    degrees = sorted(degrees, reverse=True)
    for deg in degrees:
        w2c_rotated = create_new_w2c_orbit(deg, w2c[0], scene_center, device=device, offset_along_z=cfg.offset_along_z, scene_up=scene_up)
        all_w2cs.append(w2c_rotated)


    for idx, w2c in tqdm(enumerate(all_w2cs), total=len(all_w2cs), desc="Rendering views"):

        # rgb
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
        image = center_crop_512(Image.fromarray(image))
        image_dir = render_output_dir / "rgb"/ f"cam_{idx}"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{frame_id:04d}.png"
        image.save(image_path)

        # normal map
        normal_rgb = render_smpl_normal_map(
            smpl_server=smpl_server,
            smpl_param=smpl_param,
            w2c_cv=w2c,
            K=K,
            H=H,
            W=W,
        )
        normal_image = Image.fromarray(normal_rgb.astype(np.uint8))
        normal_image = center_crop_512(normal_image)
        normal_dir = render_output_dir / "normals"/ f"cam_{idx}"
        normal_dir.mkdir(parents=True, exist_ok=True)
        normal_path = normal_dir / f"{frame_id:04d}.png"
        normal_image.save(normal_path)

        # overlay RGB and normal map for visualization
        image_arr = np.array(image)
        normal_arr = np.array(normal_image)
        overlay = cv2.addWeighted(image_arr, 0.5, normal_arr, 0.5, 0)
        overlay_image = Image.fromarray(overlay.astype(np.uint8))
        overlay_image = center_crop_512(overlay_image)
        overlay_dir = render_output_dir / "overlay"/ f"cam_{idx}"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"{frame_id:04d}.png"
        overlay_image.save(overlay_path)

    print(f"Rendered images saved to: {render_output_dir}")

if __name__ == "__main__":
    main()
