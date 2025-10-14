from pathlib import Path
from typing import Optional, Sequence, List

import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

from training.helpers.model_init import SceneSplats
from training.helpers.render import render_splats
from training.helpers.smpl_utils import canon_to_posed


def save_mask_refinement_figure(
    image: np.ndarray,
    initial_mask: np.ndarray,
    refined_mask: np.ndarray,
    positive_pts: Optional[np.ndarray],
    negative_pts: Optional[np.ndarray],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image)
    axes[0].imshow(initial_mask.astype(float), cmap="Greens", alpha=0.35)
    legend_items = []
    if positive_pts is not None and positive_pts.size > 0:
        axes[0].scatter(
            positive_pts[:, 0],
            positive_pts[:, 1],
            s=45,
            c="lime",
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            label="positive",
        )
        legend_items.append("positive")
    if negative_pts is not None and negative_pts.size > 0:
        axes[0].scatter(
            negative_pts[:, 0],
            negative_pts[:, 1],
            s=45,
            c="red",
            marker="x",
            linewidths=1.2,
            label="negative",
        )
        legend_items.append("negative")
    if legend_items:
        axes[0].legend(loc="upper right")
    axes[0].set_title("SAM2 Input Prompts")
    axes[0].set_axis_off()

    axes[1].imshow(image)
    axes[1].imshow(initial_mask.astype(float), cmap="Reds", alpha=0.4)
    axes[1].imshow(refined_mask.astype(float), cmap="Blues", alpha=0.6)
    axes[1].set_title("Refined Mask Overlay")
    axes[1].set_axis_off()

    # add legend for red and blue overlays
    red_patch = matplotlib.patches.Patch(color="red", alpha=0.4, label="Initial Mask")
    blue_patch = matplotlib.patches.Patch(color="blue", alpha=0.6, label="Refined Mask")
    axes[1].legend(handles=[red_patch, blue_patch], loc="upper right")

    fig.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def save_loss_visualization(
    image_input: torch.Tensor,       # [B, H,W, 3], GT image in [0,1]
    gt: torch.Tensor,        # [B, H,W, 3], 0–1
    prediction: torch.Tensor,    # [B, H,W, 3], predicted image in [0,1]
    out_path: str,
):
    """
    Saves a side-by-side visualization of:
    - original image
    - masked image (image * mask)
    - predicted image
    """

    # it may happen that gt and prediction have different sizes due to cropping
    # in which case, resize them to the input image size, use black background
    if image_input.shape[1:3] != gt.shape[1:3]:
        B, H, W = image_input.shape[0:3]
        gt_resized = torch.zeros((B, H, W, 3), device=gt.device)
        # pr_resized = torch.zeros((B, H, W, 3), device=prediction.device)

        h0 = (H - gt.shape[1]) // 2
        w0 = (W - gt.shape[2]) // 2
        h1 = h0 + gt.shape[1]
        w1 = w0 + gt.shape[2]

        gt_resized[:, h0:h1, w0:w1, :] = gt
        # pr_resized[:, h0:h1, w0:w1, :] = prediction

        gt = gt_resized
        # prediction = pr_resized

    comparison = torch.cat([image_input, gt, prediction], dim=2)  # [B,H,3W,3]
    comparison = comparison[0]  # Take first in batch

    # Convert to uint8 for saving
    img = (comparison.cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(img).save(out_path)

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
        colors, _, _ = render_splats(
            scene_splats,
            smpl_params,
            lbs_prepared,
            w2c_single.unsqueeze(0),
            K.unsqueeze(0),
            H,
            W,
            sh_degree=sh_degree,
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
def save_smpl_pose_overlay(
    *,
    image_np: np.ndarray,
    scene_splats: SceneSplats,
    initial_params: torch.Tensor,
    current_params: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
    device: torch.device,
    out_path: Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)

    init_vertices = _pose_body_vertices(scene_splats, initial_params, device)
    curr_vertices = _pose_body_vertices(scene_splats, current_params, device)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.imshow(img_uint8)

    init_label_added = False
    curr_label_added = False

    for verts in init_vertices:
        if verts.numel() == 0:
            continue
        uv = _project_world_to_pixels(verts, w2c, K)
        if uv.size == 0:
            continue
        ax.scatter(
            uv[:, 0],
            uv[:, 1],
            s=0.75,
            c="#ff4d4d",
            alpha=0.25,
            linewidths=0,
            label="Initial pose" if not init_label_added else None,
        )
        init_label_added = True

    for verts in curr_vertices:
        if verts.numel() == 0:
            continue
        uv = _project_world_to_pixels(verts, w2c, K)
        if uv.size == 0:
            continue
        ax.scatter(
            uv[:, 0],
            uv[:, 1],
            s=0.5,
            c="#32cd32",
            alpha=0.25,
            linewidths=0,
            label="Current pose" if not curr_label_added else None,
        )
        curr_label_added = True

    if init_label_added or curr_label_added:
        legend = ax.legend(loc="upper right")
        for handle in legend.legend_handles:
            handle.set_sizes([50])

    ax.set_axis_off()
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


@torch.no_grad()
def save_pose_progress_overlay(
    *,
    dataset,
    scene_splats: SceneSplats,
    frame_id: int,
    initial_params_list: Sequence[torch.Tensor],
    current_params: torch.Tensor,
    device: torch.device,
    epoch: int,
    out_dir: Path,
) -> Path:
    sample = dataset[frame_id]

    image_np = sample["image"].cpu().numpy()
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (image_np * 255.0).astype(np.uint8)
    image_np = np.ascontiguousarray(image_np)

    K = sample["K"].to(device)
    w2c = sample["M_ext"].to(device)

    initial_params = torch.stack(initial_params_list, dim=0).to(device, dtype=torch.float32)
    current_params = current_params.detach().to(device, dtype=torch.float32)

    out_path = Path(out_dir) / f"pose_overlay_epoch{epoch:03d}_fid{frame_id:04d}.png"
    return save_smpl_pose_overlay(
        image_np=image_np,
        scene_splats=scene_splats,
        initial_params=initial_params,
        current_params=current_params,
        w2c=w2c,
        K=K,
        device=device,
        out_path=out_path,
    )
