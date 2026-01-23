from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes

from training.helpers.gs_renderer import Camera


# ---------------------------------------------------------
# Configurations
# ---------------------------------------------------------

@dataclass
class TsdfConfig:
    tsdf_voxel_size: float = 0.005
    tsdf_sdf_trunc: float = 0.02
    tsdf_depth_trunc: float = 12.0
    tsdf_alpha_thresh: float = 0.3
    tsdf_min_cluster_triangles: int = 100

@dataclass
class MarchingCubesConfig:
    grid_size: int = 160
    sigma_scale: float = 0.6
    iso_level: float = 0.07
    truncation: float = 2.5
    min_opacity: float = 0.02
    min_sigma: float = 1e-4
    max_sigma: Optional[float] = None
    max_gaussians: Optional[int] = None
    seed: int = 0
    padding: float = 0.05


# ---------------------------------------------------------
# Marching cubes extraction
# ---------------------------------------------------------
def _prepare_gaussians(
    posed_3dgs,
    sigma_scale: float,
    min_sigma: float,
    max_sigma: Optional[float],
    min_opacity: float,
    max_gaussians: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Extract per-person arrays and normalize opacity/scales.
    xyz = posed_3dgs.xyz.detach().cpu().numpy().astype(np.float32)
    scaling = posed_3dgs.scaling.detach().cpu().numpy().astype(np.float32)
    opacity = posed_3dgs.opacity.detach().cpu().numpy().astype(np.float32)

    if opacity.ndim == 2 and opacity.shape[-1] == 1:
        opacity = opacity[:, 0]
    opacity = np.clip(opacity, 0.0, 1.0)

    # Uniform downsample to cap density computation cost.
    if max_gaussians is not None and xyz.shape[0] > max_gaussians:
        rng = np.random.default_rng(seed)
        idx = rng.choice(xyz.shape[0], size=max_gaussians, replace=False)
        xyz = xyz[idx]
        scaling = scaling[idx]
        opacity = opacity[idx]

    # Isotropic sigma from per-axis scales (rotation ignored).
    scales = scaling
    if scales.ndim == 2 and scales.shape[1] == 3:
        sigma = scales.mean(axis=1)
    else:
        raise ValueError(f"Expected scaling shape [N,3], got {scaling.shape}")
    sigma = sigma * float(sigma_scale)
    sigma = np.clip(sigma, min_sigma, max_sigma if max_sigma is not None else np.inf)

    keep = opacity >= min_opacity
    return xyz[keep], sigma[keep], opacity[keep]


def _density_grid_from_gaussians(
    centers: np.ndarray,
    sigmas: np.ndarray,
    weights: np.ndarray,
    grid_size: Tuple[int, int, int],
    padding: float,
    truncation: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    if centers.shape[0] == 0:
        raise ValueError("No gaussians remaining after filtering.")

    # Build a uniform grid over the gaussian bbox with padding.
    min_corner = centers.min(axis=0) - padding
    max_corner = centers.max(axis=0) + padding

    nx, ny, nz = grid_size
    xs = np.linspace(min_corner[0], max_corner[0], nx, dtype=np.float32)
    ys = np.linspace(min_corner[1], max_corner[1], ny, dtype=np.float32)
    zs = np.linspace(min_corner[2], max_corner[2], nz, dtype=np.float32)
    dx = float(xs[1] - xs[0]) if nx > 1 else 1.0
    dy = float(ys[1] - ys[0]) if ny > 1 else 1.0
    dz = float(zs[1] - zs[0]) if nz > 1 else 1.0

    density = np.zeros((nx, ny, nz), dtype=np.float32)

    # Accumulate truncated Gaussian blobs into the grid.
    for mu, sigma, w in zip(centers, sigmas, weights):
        radius = truncation * sigma
        if radius <= 0.0:
            continue
        ix0 = max(0, int(np.floor((mu[0] - radius - min_corner[0]) / dx)))
        ix1 = min(nx - 1, int(np.ceil((mu[0] + radius - min_corner[0]) / dx)))
        iy0 = max(0, int(np.floor((mu[1] - radius - min_corner[1]) / dy)))
        iy1 = min(ny - 1, int(np.ceil((mu[1] + radius - min_corner[1]) / dy)))
        iz0 = max(0, int(np.floor((mu[2] - radius - min_corner[2]) / dz)))
        iz1 = min(nz - 1, int(np.ceil((mu[2] + radius - min_corner[2]) / dz)))

        if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
            continue

        sigma2 = float(sigma * sigma)
        x = xs[ix0 : ix1 + 1] - mu[0]
        y = ys[iy0 : iy1 + 1] - mu[1]
        z = zs[iz0 : iz1 + 1] - mu[2]
        gx = np.exp(-(x * x) / (2.0 * sigma2)).astype(np.float32)
        gy = np.exp(-(y * y) / (2.0 * sigma2)).astype(np.float32)
        gz = np.exp(-(z * z) / (2.0 * sigma2)).astype(np.float32)

        density[ix0 : ix1 + 1, iy0 : iy1 + 1, iz0 : iz1 + 1] += (
            w * gx[:, None, None] * gy[None, :, None] * gz[None, None, :]
        )

    return density, min_corner.astype(np.float32), (dx, dy, dz)


def _extract_mesh(
    density: np.ndarray,
    origin: np.ndarray,
    spacing: Tuple[float, float, float],
    iso_level: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Marching Cubes expects density on a grid; spacing converts to world units.
    verts, faces, _, _ = marching_cubes(density, level=iso_level, spacing=spacing)
    verts = verts + origin[None, :]
    return verts.astype(np.float32), faces.astype(np.int32)


def get_meshes_using_mc(
    all_posed_gs_list: List,
    cfg: MarchingCubesConfig,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert posed 3DGS to per-person meshes
    """
    grid_size = (int(cfg.grid_size), int(cfg.grid_size), int(cfg.grid_size))
    results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    n_persons = len(all_posed_gs_list)

    # Convert each person independently to keep per-person mesh outputs.
    for pid in range(n_persons):

        centers, sigmas, weights = _prepare_gaussians(
            all_posed_gs_list[pid],
            sigma_scale=cfg.sigma_scale,
            min_sigma=cfg.min_sigma,
            max_sigma=cfg.max_sigma,
            min_opacity=cfg.min_opacity,
            max_gaussians=cfg.max_gaussians,
            seed=cfg.seed,
        )


        density, origin, spacing = _density_grid_from_gaussians(
            centers,
            sigmas,
            weights,
            grid_size=grid_size,
            padding=cfg.padding,
            truncation=cfg.truncation,
        )

        vertices, faces = _extract_mesh(density, origin, spacing, cfg.iso_level)

        results[int(pid)] = (vertices, faces)

    return results


# ---------------------------------------------------------
# TSDF based extraction
# ---------------------------------------------------------

def tsdf_fuse_depths(
    depth_entries: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    voxel_size: float,
    sdf_trunc: float,
    depth_trunc: float,
    min_cluster_triangles: int,
) -> Tuple[np.ndarray, np.ndarray]:

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )

    for depth_np, K_np, c2w_np in depth_entries:
        height, width = depth_np.shape[:2]
        fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
        cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))
        color_o3d = o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )
        w2c_np = np.linalg.inv(c2w_np.astype(np.float64))
        volume.integrate(rgbd, intrinsic, w2c_np)

    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    if min_cluster_triangles > 0:
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        remove_mask = cluster_n_triangles[triangle_clusters] < int(min_cluster_triangles)
        if np.any(remove_mask):
            mesh.remove_triangles_by_mask(remove_mask)
            mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    return vertices, faces


def get_meshes_from_3dgs(gs_to_mesh_method: str, 
                         all_posed_gs_list: List, 
                         tsdf_camera_files: Dict[int, Dict[str, Dict]],
                         tsdf_cam_ids: List[int], 
                         frame_name: str, 
                         render_hw: Tuple[int, int],
                         render_func
                        ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:

    if gs_to_mesh_method == "tsdf":
        tsdf_cfg = TsdfConfig()
        alpha_thresh = tsdf_cfg.tsdf_alpha_thresh
        n_persons = len(all_posed_gs_list)

        # Step 1: Get depth maps from all cameras for each person
        depth_entries_by_person: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            person_idx: [] for person_idx in range(n_persons)
        }
        for cam_id in tsdf_cam_ids:

            # - Parse camera params
            cam_sample = tsdf_camera_files[cam_id][frame_name]
            K = cam_sample["K"]
            c2w = cam_sample["c2w"]
            height, width = render_hw

            for person_idx, posed_gs in enumerate(all_posed_gs_list):

                # - Render depth
                render_res = render_func(
                    posed_gs,
                    Camera.from_c2w(c2w, K, height, width),
                )
                depth = (
                    render_res["comp_depth"][0, ..., 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                # - Apply alpha thresholding and clean depth
                if alpha_thresh > 0.0:
                    alpha = (
                        render_res["comp_mask"][0, ..., 0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    depth = np.where(alpha >= alpha_thresh, depth, 0.0)
                depth = np.where(np.isfinite(depth), depth, 0.0)

                # - Store entry
                depth_entries_by_person[person_idx].append(
                    (
                        depth,
                        K.detach().cpu().numpy().astype(np.float32),
                        c2w.detach().cpu().numpy().astype(np.float32),
                    )
                )

        # Step 2: Fuse depths into meshes per person
        meshes_for_frame: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for person_idx in range(n_persons):
            # - Get depth entries for this person
            entries = depth_entries_by_person[person_idx]

            # - Fuse the depth maps into a mesh
            vertices, faces = tsdf_fuse_depths(
                entries,
                voxel_size=tsdf_cfg.tsdf_voxel_size,
                sdf_trunc=tsdf_cfg.tsdf_sdf_trunc,
                depth_trunc=tsdf_cfg.tsdf_depth_trunc,
                min_cluster_triangles=tsdf_cfg.tsdf_min_cluster_triangles,
            )

            # - Store mesh for this person
            meshes_for_frame[person_idx] = (vertices, faces)

        return meshes_for_frame

    elif gs_to_mesh_method == "mc":
        mc_cfg = MarchingCubesConfig()
        return get_meshes_using_mc(
            all_posed_gs_list,
            mc_cfg,
        )
    else:
        raise ValueError(f"Unknown gs_to_mesh_method: {gs_to_mesh_method}")