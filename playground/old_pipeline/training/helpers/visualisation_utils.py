from pathlib import Path
from typing import Optional, Sequence, List, Dict, Any, Tuple, Union
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from omegaconf import DictConfig
import torch
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import cv2
from tqdm import tqdm

from training.helpers.model_init import SceneSplats
from training.helpers.render import render_splats
from training.helpers.smpl_utils import canon_to_posed
from training.smpl_deformer.smpl_server import SMPLServer
import pyrender
import trimesh

_SMPL_VIS_SERVER: Optional[SMPLServer] = None


def _get_smpl_vis_server() -> SMPLServer:
    global _SMPL_VIS_SERVER
    if _SMPL_VIS_SERVER is None:
        _SMPL_VIS_SERVER = SMPLServer().eval()
    return _SMPL_VIS_SERVER


def save_alpha_heatmap(
    alpha_map: torch.Tensor,
    out_path: Path,
    human_idx: int = 0,
    cmap: str = "viridis",
) -> Path:
    """
    Save a heatmap visualisation of an alpha map for a single human.

    Args:
        alpha_map: Alpha tensor of shape [H, W] (or [H, W, 1]) in [0, 1].
        out_path: Destination path for the PNG file.
        human_idx: Optional identifier to include in the figure title.
        cmap: Matplotlib colormap to use for the heatmap.
    """
    alpha_np = alpha_map.squeeze().detach().cpu().float().numpy()
    alpha_np = np.clip(alpha_np, 0.0, 1.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alpha_np, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(f"Alpha heatmap (human {human_idx})")
    ax.set_axis_off()

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Alpha value", rotation=270, labelpad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path



def _pose_body_vertices(
    scene_splats: SceneSplats,
    smpl_params: torch.Tensor,
    device: torch.device,
) -> List[torch.Tensor]:
    smpl_info = getattr(scene_splats, "smpl_c_info", None)
    if smpl_info is None:
        return []

    smpl_server = smpl_info["smpl_server"].to(device)
    verts_c = smpl_info["verts_c"].to(device)
    weights_c = smpl_info["weights_c"].to(device)

    params = smpl_params.to(device)
    if params.ndim == 3:
        params = params[0]
    if params.ndim != 2:
        return []

    posed: list[torch.Tensor] = []
    for human_params in params:
        verts = canon_to_posed(
            smpl_server,
            human_params.unsqueeze(0),
            verts_c,
            weights_c,
            device=device,
        )
        posed.append(verts.squeeze(0))
    return posed


def _gather_gaussian_points(scene_splats: SceneSplats, device: torch.device) -> torch.Tensor:
    points: List[torch.Tensor] = []
    if scene_splats.static is not None and scene_splats.static.get("means") is not None:
        static_means = scene_splats.static["means"]
        if static_means.numel() > 0:
            points.append(static_means.to(device))
    for dyn in scene_splats.dynamic:
        means = dyn.get("means")
        if means is not None and means.numel() > 0:
            points.append(means.to(device))
    if not points:
        return torch.zeros((0, 3), device=device)
    return torch.cat(points, dim=0)


def _compute_gaussian_center(scene_splats: SceneSplats, device: torch.device) -> tuple[torch.Tensor, float]:
    candidates = []
    if scene_splats.static is not None and scene_splats.static.get("means") is not None:
        static_means = scene_splats.static["means"]
        if static_means.numel() > 0:
            candidates.append(static_means.to(device))
    for dyn in scene_splats.dynamic:
        means = dyn.get("means")
        if means is not None and means.numel() > 0:
            candidates.append(means.to(device))

    if not candidates:
        return torch.zeros(3, device=device), 1.0

    stacked = torch.cat(candidates, dim=0)
    center = stacked.mean(dim=0)
    distances = torch.norm(stacked - center.unsqueeze(0), dim=1)
    fallback_radius = float(distances.mean().item()) if distances.numel() > 0 else 1.0
    fallback_radius = max(fallback_radius, 1e-3)
    return center, fallback_radius


def _compute_orbit_focus(
    scene_splats: SceneSplats,
    smpl_params: Optional[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    if smpl_params is not None:
        posed_vertices = _pose_body_vertices(scene_splats, smpl_params, device)
        if posed_vertices:
            all_vertices = torch.cat(posed_vertices, dim=0)
            min_corner = all_vertices.min(dim=0).values
            max_corner = all_vertices.max(dim=0).values
            center = 0.5 * (min_corner + max_corner)
            extent = 0.5 * (max_corner - min_corner)
            fallback_radius = float(extent.norm().item())
            fallback_radius = max(fallback_radius, 1e-3)
            return center, fallback_radius, all_vertices

    center, fallback_radius = _compute_gaussian_center(scene_splats, device)
    gaussian_points = _gather_gaussian_points(scene_splats, device)
    return center, fallback_radius, gaussian_points


def _transform_points_camera(w2c: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return points
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    cam = (R @ points.t()) + t.view(3, 1)
    return cam.t()


def _w2c_to_camera_center(w2c: torch.Tensor) -> torch.Tensor:
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    return (-R.T @ t).clone()


def _project_world_to_pixels(
    points_world: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
) -> np.ndarray:
    if points_world.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)

    device = points_world.device
    cam = _transform_points_camera(w2c.to(device), points_world)
    if cam.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)

    positive_mask = cam[:, 2] > 1e-6
    if not positive_mask.any():
        return np.zeros((0, 2), dtype=np.float32)

    cam = cam[positive_mask]
    K_3 = K[:3, :3].to(device)
    uvw = (K_3 @ cam.t()).t()
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv.detach().cpu().numpy()


def _create_orbit_w2c_matrices(
    center: torch.Tensor,
    num_frames: int,
    device: torch.device,
    horizontal_radius: float,
    height_offset: float,
    start_angle: float,
) -> torch.Tensor:
    center_np = center.detach().cpu().numpy()
    radius = max(horizontal_radius, 1e-4)
    height = float(height_offset)
    angles = np.linspace(start_angle, start_angle + 2.0 * np.pi, num_frames, endpoint=False, dtype=np.float32)

    up_fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    matrices = []
    for theta in angles:
        pos = np.array(
            [
                center_np[0] + radius * np.cos(theta),
                center_np[1] + height,
                center_np[2] + radius * np.sin(theta),
            ],
            dtype=np.float32,
        )
        forward = center_np - pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            forward_norm = 1.0
        forward /= forward_norm

        right = np.cross(forward, up_world)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.cross(forward, up_fallback)
            right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            right_norm = 1.0
        right /= right_norm

        up_vec = np.cross(forward, right)
        up_norm = np.linalg.norm(up_vec)
        if up_norm < 1e-6:
            up_vec = up_world
            up_norm = np.linalg.norm(up_vec)
        up_vec /= up_norm

        R = np.stack([right, up_vec, forward], axis=0)
        t = -R @ pos

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        matrices.append(torch.from_numpy(w2c))

    return torch.stack(matrices, dim=0).to(device)


@torch.no_grad()
def save_orbit_visualization(
    scene_splats: SceneSplats,
    smpl_params: torch.Tensor,
    lbs_weights: Optional[Sequence[torch.Tensor]],
    base_w2c: torch.Tensor,
    K: torch.Tensor,
    image_size: tuple[int, int],
    *,
    device: torch.device,
    sh_degree: int,
    out_path: Path,
    num_frames: int = 120,
    fps: int = 24,
    near_plane: float = 0.2,
    far_plane: float = 200.0,
    packed: bool = False,
    absgrad: bool = False,
    sparse_grad: bool = False,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_device = device
    base_w2c = base_w2c.to(target_device)
    K = K.to(target_device)
    smpl_params = smpl_params.to(target_device)

    lbs_prepared: Optional[Sequence[torch.Tensor]] = None
    if lbs_weights is not None:
        lbs_prepared = [w.to(target_device) for w in lbs_weights]

    center, fallback_radius, diagnostic_points = _compute_orbit_focus(scene_splats, smpl_params, target_device)

    cam_center = _w2c_to_camera_center(base_w2c)
    offset = cam_center - center
    horizontal = offset.clone()
    horizontal[1] = 0.0
    horiz_radius = float(torch.norm(horizontal).item())
    if horiz_radius < 1e-4:
        horiz_radius = fallback_radius

    horiz_radius *= 1.3

    start_angle = 0.0
    if horiz_radius >= 1e-4:
        horizontal_np = horizontal.detach().cpu().numpy()
        start_angle = float(np.arctan2(horizontal_np[2], horizontal_np[0]))

    orbit_w2c = _create_orbit_w2c_matrices(
        center=center,
        num_frames=num_frames,
        device=target_device,
        horizontal_radius=horiz_radius,
        height_offset=float(offset[1].item()),
        start_angle=start_angle,
    )

    H, W = image_size
    frames = []
    for w2c_single in orbit_w2c:
        colors, _, _, _, _, _, _ = render_splats(
            scene_splats,
            smpl_params,
            lbs_prepared,
            w2c_single.unsqueeze(0),
            K.unsqueeze(0),
            H,
            W,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode="RGB+D",
            packed=packed,
            absgrad=absgrad,
            sparse_grad=sparse_grad,
        )
        frame = torch.clamp(colors[0], 0.0, 1.0).detach().cpu().numpy()
        frame_uint8 = (frame * 255.0).astype(np.uint8)
        frames.append(np.ascontiguousarray(frame_uint8))

    imageio.mimwrite(
        out_path,
        frames,
        fps=fps,
        macro_block_size=None,
        format="FFMPEG",
    )

    return out_path


@torch.no_grad()
def save_epoch_smpl_overlays(
    dataset,
    smpl_params_per_frame: Dict[int, torch.Tensor],
    experiment_dir: Path,
    epoch: int,
    *,
    device: torch.device,
    gender: str = "neutral",
    alpha: float = 0.6,
) -> None:
    """
    Render SMPL meshes for every frame and composite them over the input RGB frames.

    Args:
        dataset: FullSceneDataset (or compatible object) providing images, cameras, and SMPL params.
        smpl_params_per_frame: Mapping fid -> [num_people, 86] SMPL parameter tensor.
        experiment_dir: Root experiment directory.
        epoch: Current epoch index (used in the output filename).
        device: Torch device for rendering.
        gender: SMPL gender model to use (default: "neutral").
        alpha: Blending weight applied to the rendered SMPL colours.
    """
    if len(dataset) == 0 or not smpl_params_per_frame:
        return

    smpl_server = SMPLServer(gender=gender).to(device)
    smpl_server.eval()

    image_height = getattr(dataset, "H", None)
    image_width = getattr(dataset, "W", None)
    if image_height is None or image_width is None:
        sample = dataset[0]
        image_height = int(sample["H"])
        image_width = int(sample["W"])

    colour_palette = torch.tensor(
        [
            [0.84, 0.37, 0.37, 0.9],
            [0.37, 0.66, 0.84, 0.9],
            [0.45, 0.80, 0.46, 0.9],
            [0.83, 0.66, 0.37, 0.9],
            [0.59, 0.44, 0.84, 0.9],
            [0.84, 0.44, 0.75, 0.9],
        ],
        dtype=torch.float32,
        device=device,
    )

    out_root = Path(experiment_dir) / "visualizations" / "smpl"
    out_root.mkdir(parents=True, exist_ok=True)

    renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)

    for fid in tqdm(range(len(dataset)), desc=f"Saving SMPL overlays for epoch {epoch:04d}"):
        smpl_params = smpl_params_per_frame.get(fid)
        if smpl_params is None:
            continue

        frame_data = dataset[fid]
        image = frame_data["image"].detach().cpu().numpy()
        K = frame_data["K"].cpu().numpy()
        w2c_cv = frame_data["M_ext"].cpu().numpy()

        smpl_tensor = smpl_params.detach().to(device)
        if smpl_tensor.ndim == 1:
            smpl_tensor = smpl_tensor.unsqueeze(0)

        smpl_output = smpl_server(smpl_tensor, absolute=True)

        verts_tensor = smpl_output["smpl_verts"]
        faces_data = smpl_output["smpl_faces"]

        verts = verts_tensor.detach().cpu().numpy()
        if isinstance(faces_data, torch.Tensor):
            faces = faces_data.detach().cpu().numpy().astype(np.int32)
        else:
            faces = np.asarray(faces_data, dtype=np.int32)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.2, 0.2, 0.2])
        mesh_objects: List[pyrender.Mesh] = []
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        w2c_gl = cv_to_gl @ w2c_cv
        for person_idx in range(verts.shape[0]):
            verts_np = verts[person_idx]
            colour = colour_palette[person_idx % colour_palette.shape[0]].cpu().numpy()
            vertex_colors = np.tile(colour, (verts_np.shape[0], 1))
            tri_mesh = trimesh.Trimesh(vertices=verts_np, faces=faces, vertex_colors=vertex_colors, process=False)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
            mesh_objects.append(mesh)
            scene.add(mesh)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=np.eye(4))

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
        c2w_gl = np.linalg.inv(w2c_gl)
        scene.add(cam, pose=c2w_gl)

        color_rgb, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
        render_rgb = color_rgb.astype(np.float32) / 255.0
        mask = (depth > 0).astype(np.float32)[..., None]
        if mask.max() <= 0.0:
            mask = (np.linalg.norm(render_rgb, axis=-1, keepdims=True) > 1e-6).astype(np.float32)
        alpha_mask = alpha * mask

        pose_rgba = np.concatenate([render_rgb, mask], axis=-1)

        image_rgb = np.clip(image, 0.0, 1.0)
        composite = image_rgb * (1.0 - alpha_mask) + render_rgb * alpha_mask
        composite_np = (np.clip(composite, 0.0, 1.0) * 255.0).astype(np.uint8)

        poses_np = np.clip(pose_rgba, 0.0, 1.0)
        poses_np[..., :3] *= 255.0
        poses_np[..., 3] *= 255.0
        poses_np = poses_np.astype(np.uint8)

        frame_dir = out_root / f"frame_{fid:04d}"
        overlay_dir = frame_dir / "overlay"
        poses_only_dir = frame_dir / "poses_only"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        poses_only_dir.mkdir(parents=True, exist_ok=True)

        overlay_path = overlay_dir / f"epoch_{epoch:04d}.png"
        poses_only_path = poses_only_dir / f"epoch_{epoch:04d}.png"

        Image.fromarray(composite_np).save(overlay_path)
        Image.fromarray(poses_np, mode="RGBA").save(poses_only_path)

        person_current_paths: List[Path] = []
        for person_idx, mesh in enumerate(mesh_objects):
            scene_person = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.2, 0.2, 0.2])
            scene_person.add(mesh)
            scene_person.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=np.eye(4))
            scene_person.add(pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0), pose=c2w_gl)

            person_rgb, person_depth = renderer.render(scene_person, flags=pyrender.constants.RenderFlags.FLAT)
            person_render = person_rgb.astype(np.float32) / 255.0
            person_mask = (person_depth > 0).astype(np.float32)[..., None]
            if person_mask.max() <= 0.0:
                person_mask = (np.linalg.norm(person_render, axis=-1, keepdims=True) > 1e-6).astype(np.float32)
            person_rgba = np.concatenate([person_render, person_mask], axis=-1)
            person_np = np.clip(person_rgba, 0.0, 1.0)
            person_np[..., :3] *= 255.0
            person_np[..., 3] *= 255.0
            person_np = person_np.astype(np.uint8)

            person_dir = poses_only_dir / f"person_{person_idx:02d}"
            person_dir.mkdir(parents=True, exist_ok=True)
            person_path = person_dir / f"epoch_{epoch:04d}.png"
            Image.fromarray(person_np, mode="RGBA").save(person_path)
            person_current_paths.append(person_path)

        comparison_rows: List[np.ndarray] = []
        for person_idx, person_path in enumerate(person_current_paths):
            person_dir = person_path.parent
            epoch_files = sorted(person_dir.glob("epoch_*.png"))
            if len(epoch_files) < 2:
                continue
            baseline_path = epoch_files[0]
            if baseline_path == person_path:
                continue
            baseline_img = np.array(Image.open(baseline_path).convert("RGBA")).astype(np.float32) / 255.0
            current_img = np.array(Image.open(person_path).convert("RGBA")).astype(np.float32) / 255.0

            baseline_alpha = baseline_img[..., 3:4]
            current_alpha = current_img[..., 3:4]

            current_rgba = np.zeros_like(current_img)
            current_rgba[..., 2] = current_alpha[..., 0]
            current_rgba[..., 3] = current_alpha[..., 0]

            baseline_rgba = np.zeros_like(baseline_img)
            baseline_rgba[..., 0] = baseline_alpha[..., 0]
            baseline_rgba[..., 3] = baseline_alpha[..., 0]

            overlay_rgb = baseline_rgba[..., :3] * baseline_alpha + current_rgba[..., :3] * (1.0 - baseline_alpha)
            overlay_alpha = baseline_alpha + current_rgba[..., 3:4] * (1.0 - baseline_alpha)

            overlay_rgba = np.concatenate([overlay_rgb, overlay_alpha], axis=-1)
            overlay_rgba = np.clip(overlay_rgba * 255.0, 0.0, 255.0).astype(np.uint8)

            row_height, row_width = overlay_rgba.shape[:2]
            label_height = 24
            row_with_label = np.zeros((row_height + label_height, row_width, 4), dtype=np.uint8)
            row_with_label[label_height:, :, :] = overlay_rgba

            label_canvas = row_with_label[:label_height, :, :]
            label_canvas[:, :, 3] = 255
            text_img = Image.fromarray(label_canvas, mode="RGBA")
            draw = ImageDraw.Draw(text_img)
            label_text = f"Person {person_idx:02d}: baseline=red, current=blue"
            draw.text((5, 5), label_text, fill=(255, 255, 255, 255), anchor="la")
            row_with_label[:label_height, :, :] = np.array(text_img)

            comparison_rows.append(row_with_label)

        if comparison_rows:
            comparison_img = np.concatenate(comparison_rows, axis=0)
            comparison_dir = frame_dir / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            comparison_path = comparison_dir / f"epoch_{epoch:04d}.png"
            Image.fromarray(comparison_img, mode="RGBA").save(comparison_path)

    renderer.delete()


@torch.no_grad()
def save_smpl_overlay_image(
    image: Union[np.ndarray, torch.Tensor],
    pred_smpl: torch.Tensor,
    K: torch.Tensor,
    w2c_cv: torch.Tensor,
    out_path: Union[str, Path],
    *,
    gt_smpl: Optional[torch.Tensor] = None,
    device: torch.device,
    pred_color: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    gt_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> None:
    """
    Render predicted (and optionally GT) SMPL meshes on top of the input image.
    """

    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
    else:
        img_np = np.asarray(image)
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    img_np = img_np.astype(np.float32)
    H, W = img_np.shape[:2]

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.2, 0.2, 0.2])

    smpl_server = _get_smpl_vis_server().to(device)
    faces = smpl_server.smpl.faces.astype(np.int32)

    def _add_mesh(params: Optional[torch.Tensor], color: Tuple[float, float, float]) -> None:
        if params is None:
            return
        tensor = params.to(device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        smpl_output = smpl_server(tensor, absolute=True)
        verts = smpl_output["smpl_verts"][0].detach().cpu().numpy()
        color_rgba = np.array((*color, 0.85), dtype=np.float32)
        vertex_colors = np.tile(color_rgba, (verts.shape[0], 1))
        tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors, process=False)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
        scene.add(mesh)

    _add_mesh(gt_smpl, gt_color)
    _add_mesh(pred_smpl, pred_color)

    K_np = K.detach().cpu().numpy()
    w2c_np = w2c_cv.detach().cpu().numpy()
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    w2c_gl = cv_to_gl @ w2c_np

    camera = pyrender.IntrinsicsCamera(
        fx=K_np[0, 0],
        fy=K_np[1, 1],
        cx=K_np[0, 2],
        cy=K_np[1, 2],
        znear=0.01,
        zfar=100.0,
    )
    scene.add(camera, pose=np.linalg.inv(w2c_gl))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=np.eye(4))

    color_rgba, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    render_rgb = color_rgba.astype(np.float32) / 255.0
    mask = (depth > 0).astype(np.float32)[..., None]
    if mask.max() <= 0.0:
        mask = (np.linalg.norm(render_rgb, axis=-1, keepdims=True) > 1e-6).astype(np.float32)

    composite = render_rgb * mask + img_np * (1.0 - mask)
    composite_uint8 = np.clip(composite * 255.0, 0.0, 255.0).astype(np.uint8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite_uint8).save(out_path)
