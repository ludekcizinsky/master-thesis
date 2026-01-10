import numpy as np
import torch
import torch.nn.functional as F
import pyrender
import trimesh
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt


def estimate_masks_from_smplx_batch(batch: Dict[str, Any], smplx_model) -> torch.Tensor:
    """
    Estimate binary foreground masks by rasterizing posed SMPL-X meshes into each camera view.

    Expected `batch` format (as produced by `training.helpers.dataset.SceneDataset` + PyTorch collation):
      - `batch["image"]`: `torch.Tensor` of shape `[B, H, W, 3]` in `[0, 1]` (used only for H/W and device).
      - `batch["K"]`: `torch.Tensor` of shape `[B, 4, 4]` (camera intrinsics; fx/fy/cx/cy read from it).
      - `batch["c2w"]`: `torch.Tensor` of shape `[B, 4, 4]` (camera-to-world transform).
      - `batch["smplx_params"]`: `dict[str, torch.Tensor]` where each tensor has shape `[B, P, ...]` and
        contains the keys:
          - `"betas"`: `[B, P, D]`
          - `"root_pose"`: `[B, P, 3]`
          - `"body_pose"`: `[B, P, 21, 3]`
          - `"jaw_pose"`, `"leye_pose"`, `"reye_pose"`: `[B, P, 3]`
          - `"lhand_pose"`, `"rhand_pose"`: `[B, P, 15, 3]`
          - `"trans"`: `[B, P, 3]`

    Returns:
      - `masks`: `torch.Tensor` of shape `[B, H, W, 1]` on the same device as `batch["image"]`,
        with values in `{0.0, 1.0}`.
    """

    # Parse batch data
    images: torch.Tensor = batch["image"]
    Ks: torch.Tensor = batch["K"]
    c2ws: torch.Tensor = batch["c2w"]
    smplx_params: Dict[str, torch.Tensor] = batch["smplx_params"]

    device = images.device
    bsize, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])

    # SMPL-X mesh topology.
    smplx_layer = getattr(smplx_model, "smplx_layer", None).to(device)
    faces = np.asarray(getattr(smplx_layer, "faces"), dtype=np.int64)

    # Convert CV camera convention to OpenGL for pyrender.
    cv_to_gl = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=c2ws.dtype,
    )

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    out_masks: List[torch.Tensor] = []

    # Expression is not needed for silhouette quality; keep it zero.
    expr_dim = int(getattr(getattr(smplx_model, "smpl_x", None), "expr_param_dim", 0))

    for bi in range(bsize):
        K = Ks[bi].detach().cpu()
        fx, fy, cx, cy = (
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
        )
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e4)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.5, 0.5, 0.5))

        # Add all persons as meshes to the scene (pyrender will handle occlusions via depth).
        num_people = int(smplx_params["betas"][bi].shape[0])
        for pid in range(num_people):
            betas = smplx_params["betas"][bi, pid : pid + 1]
            root_pose = smplx_params["root_pose"][bi, pid : pid + 1]
            body_pose = smplx_params["body_pose"][bi, pid : pid + 1]
            jaw_pose = smplx_params["jaw_pose"][bi, pid : pid + 1]
            leye_pose = smplx_params["leye_pose"][bi, pid : pid + 1]
            reye_pose = smplx_params["reye_pose"][bi, pid : pid + 1]
            lhand_pose = smplx_params["lhand_pose"][bi, pid : pid + 1]
            rhand_pose = smplx_params["rhand_pose"][bi, pid : pid + 1]
            trans = smplx_params["trans"][bi, pid : pid + 1]

            if expr_dim > 0:
                expression = torch.zeros((1, expr_dim), device=device, dtype=betas.dtype)
            else:
                expression = None

            out = smplx_layer(
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
            verts = out.vertices[0].detach().cpu().numpy()
            mesh_tm = trimesh.Trimesh(verts, faces, process=False)
            scene.add(pyrender.Mesh.from_trimesh(mesh_tm, smooth=False))

        # Camera pose for pyrender.
        c2w = c2ws[bi]
        w2c_cv = torch.inverse(c2w)
        w2c_gl = torch.matmul(cv_to_gl, w2c_cv)
        c2w_gl = torch.inverse(w2c_gl)
        pose = c2w_gl.clone()
        pose[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=pose.dtype)
        scene.add(camera, pose=pose.detach().cpu().numpy())

        # Finally, render. Depending on pyrender version, DEPTH_ONLY may return either
        # the depth array directly or a (color, depth) tuple.
        depth_out = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        depth = depth_out[1] if isinstance(depth_out, tuple) else depth_out
        mask_np = (depth > 0).astype(np.float32)  # [H, W]
        out_masks.append(torch.from_numpy(mask_np).to(device=device, dtype=images.dtype).unsqueeze(-1))

    renderer.delete()
    return torch.stack(out_masks, dim=0)

def _binary_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation on `[B, H, W, 1]` masks using max-pooling."""
    if radius <= 0:
        return mask
    mask_nchw = mask.permute(0, 3, 1, 2)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(mask_nchw, kernel_size=kernel, stride=1, padding=radius)
    return dilated.permute(0, 2, 3, 1)

def estimate_masks_from_rgb_and_smplx_batch(
    batch: Dict[str, Any],
    smplx_model,
    rgb_eps: float = 10.0 / 255.0,
    dilate_px: int = 10,
) -> torch.Tensor:
    """
    Estimate masks by combining high-recall RGB masks with a high-precision SMPL-X seed.

    Expected `batch` keys:
      - `batch["image"]`: `[B, H, W, 3]` float in `[0, 1]` (rendered RGB).
      - `batch["K"]`, `batch["c2w"]`, `batch["smplx_params"]`: for SMPL-X projection.

    Returns:
      - `masks`: `[B, H, W, 1]` float tensor in `{0, 1}`.
    """
    rgb_frames = batch["image"]
    rgb_masks = (rgb_frames > rgb_eps).any(dim=-1, keepdim=True).float()
    seed_masks = estimate_masks_from_smplx_batch(batch, smplx_model)
    band = _binary_dilate(seed_masks, dilate_px)
    combined = (seed_masks > 0.5) | ((rgb_masks > 0.5) & (band > 0.5))
    return combined.float()

def estimate_masks_from_src_reprojection_batch(
    batch: Dict[str, Any],
    src_masks_by_name: Dict[str, torch.Tensor],
    src_K: Union[torch.Tensor, Dict[str, torch.Tensor]],
    src_c2w: Union[torch.Tensor, Dict[str, torch.Tensor]],
    seed_masks: Optional[torch.Tensor] = None,
    depth_min: float = 1e-6,
) -> torch.Tensor:
    """
    Estimate novel-view masks by backprojecting target pixels to 3D using target depth,
    reprojecting into the source camera, and querying the source mask for the same frame.

    Expected `batch` keys (as produced by `SceneDataset` + PyTorch collation):
      - `batch["depth"]`: `[B, H, W, 1]` float depth (meters), target camera.
      - `batch["K"]`: `[B, 4, 4]` intrinsics, target camera.
      - `batch["c2w"]`: `[B, 4, 4]` camera-to-world, target camera.
      - `batch["frame_name"]`: list of length `B` with frame name stems.

    Args:
        src_masks_by_name: dict mapping `frame_name` -> source mask `[H, W, 1]`.
        src_K: `[4, 4]` intrinsics for the source camera or per-frame dict.
        src_c2w: `[4, 4]` camera-to-world for the source camera or per-frame dict.
        seed_masks: optional `[B, H, W, 1]` high-confidence masks (e.g., SMPL-X).
        depth_min: minimum valid depth threshold.

    Returns:
        `[B, H, W, 1]` float tensor in `{0,1}`.
    """

    if "depth" not in batch:
        raise KeyError("batch['depth'] is required for src-reprojection mask estimation.")

    depth = batch["depth"]  # [B, H, W, 1]
    Ks = batch["K"]
    c2ws = batch["c2w"]
    frame_names = batch["frame_name"]

    device = depth.device
    bsize, H, W = int(depth.shape[0]), int(depth.shape[1]), int(depth.shape[2])

    def _get_src_camera(frame_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(src_K, dict) or isinstance(src_c2w, dict):
            if not isinstance(src_K, dict) or not isinstance(src_c2w, dict):
                raise TypeError("src_K and src_c2w must both be dicts when using per-frame cameras.")
            if frame_name not in src_K or frame_name not in src_c2w:
                raise KeyError(f"Missing source camera for frame '{frame_name}'.")
            K_src = src_K[frame_name]
            c2w_src = src_c2w[frame_name]
        else:
            K_src = src_K
            c2w_src = src_c2w

        K_src = K_src.to(device=device, dtype=depth.dtype)
        c2w_src = c2w_src.to(device=device, dtype=depth.dtype)
        w2c_src = torch.inverse(c2w_src)
        return K_src, w2c_src

    # Build a target-view pixel grid (H,W) once; reuse per batch item.
    xs = torch.arange(W, device=device, dtype=depth.dtype)
    ys = torch.arange(H, device=device, dtype=depth.dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")  # [H, W]

    out_masks: List[torch.Tensor] = []
    for bi in range(bsize):
        frame_name = frame_names[bi]
        depth_hw = depth[bi, :, :, 0]
        valid_depth = depth_hw > depth_min

        # Target camera intrinsics for this sample.
        K = Ks[bi]
        fx, fy, cx, cy = (
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
        )

        # Backproject target pixels to target camera space using depth.
        x_cam = (grid_x - cx) / fx * depth_hw
        y_cam = (grid_y - cy) / fy * depth_hw
        z_cam = depth_hw

        # Convert to world coordinates and then into source camera coordinates.
        ones = torch.ones_like(z_cam)
        pts_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1).reshape(-1, 4)  # [N, 4]
        world = (c2ws[bi] @ pts_cam.t()).t()
        K_src, w2c_src = _get_src_camera(frame_name)
        fx_src, fy_src, cx_src, cy_src = (
            float(K_src[0, 0]),
            float(K_src[1, 1]),
            float(K_src[0, 2]),
            float(K_src[1, 2]),
        )
        cam_src = (w2c_src @ world.t()).t() # [N, 4]

        # Project to source pixel coordinates.
        z_src = cam_src[:, 2]
        u_src = fx_src * cam_src[:, 0] / z_src + cx_src
        v_src = fy_src * cam_src[:, 1] / z_src + cy_src

        # Validity mask: in front of both cameras and within source image bounds.
        valid = (
            valid_depth.view(-1)
            & (z_src > depth_min)
            & (u_src >= 0)
            & (u_src < W)
            & (v_src >= 0)
            & (v_src < H)
        )

        # Stabilize invalid projections before rounding.
        u_src = torch.nan_to_num(u_src, nan=-1.0, posinf=-1.0, neginf=-1.0)
        v_src = torch.nan_to_num(v_src, nan=-1.0, posinf=-1.0, neginf=-1.0)
        u_int = u_src.round().long().clamp(0, W - 1)
        v_int = v_src.round().long().clamp(0, H - 1)

        frame_name = frame_names[bi]
        src_mask = src_masks_by_name.get(frame_name)
        if src_mask is None:
            raise KeyError(f"Frame name {frame_name} not found in source mask cache.")
        src_mask_bool = src_mask.to(device=device)[..., 0] > 0.5
        src_vals = src_mask_bool[v_int, u_int]

        # Keep target pixels that reproject into source foreground and were valid.
        mask_flat = valid & src_vals
        mask_hw = mask_flat.view(H, W).unsqueeze(-1).to(depth.dtype)

        if seed_masks is not None:
            # Union with a high-confidence seed (e.g., SMPL-X silhouette).
            seed_hw = seed_masks[bi, :, :, 0] > 0.5
            mask_hw = torch.logical_or(seed_hw, mask_hw[:, :, 0] > 0.5).unsqueeze(-1).to(depth.dtype)

        out_masks.append(mask_hw)

    return torch.stack(out_masks, dim=0)

def overlay_mask_on_image(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    color_rgb: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Alpha-blend a (binary) mask over an RGB image for visualization.

    Args:
        image: `[H, W, 3]` float tensor in `[0, 1]`.
        mask: `[H, W, 1]` float tensor in `{0, 1}` (or `[0, 1]` for soft masks).
        color_rgb: Foreground tint color.
        alpha: Tint opacity on foreground pixels.

    Returns:
        `[H, W, 3]` float tensor in `[0, 1]`.
    """
    mask01 = mask.clamp(0.0, 1.0)
    color = torch.tensor(color_rgb, device=image.device, dtype=image.dtype).view(1, 1, 3)
    alpha_t = torch.as_tensor(alpha, device=image.device, dtype=image.dtype)
    return image * (1.0 - alpha_t * mask01) + color * (alpha_t * mask01)


def get_masked_images(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Apply binary masks to images.

    Args:
        images: `[B, H, W, 3]` float tensor in `[0, 1]`.
        masks: `[B, H, W, 1]` float tensor in `{0, 1}`.

    Returns:
        `[B, H, W, 3]` float tensor in `[0, 1]`.
    """
    return images * masks


def get_gt_est_masks_overlay(gt_masks: torch.Tensor, est_masks: torch.Tensor) -> torch.Tensor:
    """
    Create an overlay image visualizing ground-truth and estimated masks.

    Args:
        gt_masks: `[B, H, W, 1]` float tensor in `{0, 1}`.
        est_masks: `[B, H, W, 1]` float tensor in `{0, 1}`.
    Returns:
        `[B, H, W, 3]` float tensor in `[0, 1]` where:
          - GT-only pixels are green,
          - Est-only pixels are red,
          - Overlapping pixels are yellow,
          - Background pixels are black.
    """

    bsize, H, W = gt_masks.shape[0], gt_masks.shape[1], gt_masks.shape[2]
    overlays: List[torch.Tensor] = []
    for bi in range(bsize):
        gt_mask = gt_masks[bi, :, :, 0] > 0.5
        est_mask = est_masks[bi, :, :, 0] > 0.5
        gt_only = gt_mask & (~est_mask)
        est_only = (~gt_mask) & est_mask
        overlap = gt_mask & est_mask
        overlay = torch.zeros((H, W, 3), device=gt_masks.device, dtype=gt_masks.dtype)
        # GT-only: green
        overlay[:, :, 1][gt_only] = 1.0
        # Est-only: red
        overlay[:, :, 0][est_only] = 1.0
        # Overlap: yellow
        overlay[:, :, 0][overlap] = 1.0
        overlay[:, :, 1][overlap] = 1.0
        overlays.append(overlay)
    return torch.stack(overlays, dim=0)


def save_segmentation_debug_figures(
    gt_masked_frames: torch.Tensor,
    est_masked_frames: torch.Tensor,
    mask_overlays: torch.Tensor,
    frame_names: List[str],
    iou: torch.Tensor,
    recall: torch.Tensor,
    f1: torch.Tensor,
    save_dir: Path,
):
    """
    Save per-sample debug figures showing masked images and mask overlay.

    Args:
        gt_masked_frames: `[B, H, W, 3]` masked with GT masks.
        est_masked_frames: `[B, H, W, 3]` masked with estimated masks.
        mask_overlays: `[B, H, W, 3]` overlay of estimated (red) on GT (green).
        frame_names: list of length `B`, used for filenames.
        iou/recall/f1: `[B]` tensors with per-frame metrics.
        save_dir: directory to save PNGs into.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    titles = [
        "Masked Frames using GT Mask",
        "Masked Frames using Est. Mask",
        "Estimated Mask (red) overlayed\non Ground Truth Mask (green)",
    ]

    for idx in range(gt_masked_frames.shape[0]):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        imgs = [
            gt_masked_frames[idx].detach().cpu().numpy(),
            est_masked_frames[idx].detach().cpu().numpy(),
            mask_overlays[idx].detach().cpu().numpy(),
        ]
        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(f"IoU: {float(iou[idx]):.3f} | Recall: {float(recall[idx]):.3f} | F1: {float(f1[idx]):.3f}")
        out_path = save_dir / f"{frame_names[idx]}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def get_masks_based_bbox(masks: torch.Tensor, pad: int = 5) -> List[Tuple[int, int, int, int]]:
    """
    Get bounding boxes around masks with optional padding.

    Args:
        masks: Tensor of shape [B, H, W, 1] with mask values in [0, 1].
        pad: Number of pixels to pad the bounding box on each side.

    Returns:
        List of bounding boxes [(y_min, y_max, x_min, x_max)] for each mask in the batch.
    """

    per_item_bboxes = []
    global_bbox = None

    for b in range(masks.shape[0]):
        ys, xs = torch.where(masks[b, :, :, 0] > 0.5)
        # No mask, use full image
        if ys.numel() == 0 or xs.numel() == 0:
            y_min, y_max = 0, masks.shape[1] - 1
            x_min, x_max = 0, masks.shape[2] - 1
        else:
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            # Add padding
            y_center = (y_min + y_max) / 2
            x_center = (x_min + x_max) / 2
            h_half = (y_max - y_min) / 2 + pad
            w_half = (x_max - x_min) / 2 + pad
            y_min = max(int(y_center - h_half), 0)
            y_max = min(int(y_center + h_half), masks.shape[1] - 1)
            x_min = max(int(x_center - w_half), 0)
            x_max = min(int(x_center + w_half), masks.shape[2] - 1)

        bbox = (y_min, y_max, x_min, x_max)
        per_item_bboxes.append(bbox)

        if global_bbox is None:
            global_bbox = bbox
        else:
            global_bbox = (
                min(global_bbox[0], y_min),
                max(global_bbox[1], y_max),
                min(global_bbox[2], x_min),
                max(global_bbox[3], x_max),
            )

    if global_bbox is None:
        return []

    return [global_bbox for _ in per_item_bboxes]
