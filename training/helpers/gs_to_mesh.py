from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import open3d as o3d
import torch
import trimesh
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
    tsdf_virtual_cam_enable: bool = True
    tsdf_virtual_cam_count: int = 12
    tsdf_virtual_cam_height_offset: float = 0.5
    tsdf_virtual_cam_radius_scale: float = 1.0
    tsdf_virtual_cam_min_radius: float = 0.2
    tsdf_virtual_cam_include_bottom: bool = True
    tsdf_virtual_cam_ring_count: int = 8
    tsdf_virtual_cam_ring_radius: float = 2.0
    tsdf_postprocess_enable: bool = True
    tsdf_postprocess_fill_holes: bool = True
    tsdf_postprocess_smooth: bool = True
    tsdf_postprocess_smooth_method: str = "taubin"
    tsdf_postprocess_smooth_iters: int = 10
    tsdf_postprocess_taubin_lambda: float = 0.5
    tsdf_postprocess_taubin_mu: float = -0.53

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

def _normalize_vec(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _estimate_world_up(
    tsdf_camera_files: Dict[int, Dict[str, Dict]],
    tsdf_cam_ids: List[int],
    frame_name: str,
) -> np.ndarray:
    up_vectors = []
    for cam_id in tsdf_cam_ids:
        cam_sample = tsdf_camera_files[cam_id][frame_name]
        c2w = cam_sample["c2w"]
        up = c2w[:3, 1].detach().cpu().numpy().astype(np.float32)
        up_vectors.append(_normalize_vec(up))
    if not up_vectors:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    mean_up = np.mean(np.stack(up_vectors, axis=0), axis=0)
    return _normalize_vec(mean_up)


def _look_at_c2w(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    forward_is_positive_z: bool,
) -> np.ndarray:
    forward = target - eye
    forward = _normalize_vec(forward)
    if np.linalg.norm(forward) < 1e-8:
        return np.eye(4, dtype=np.float32)
    up = _normalize_vec(up)
    z_axis = forward if forward_is_positive_z else -forward
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(fallback, z_axis))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x_axis = np.cross(fallback, z_axis)
    x_axis = _normalize_vec(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = _normalize_vec(y_axis)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = eye.astype(np.float32)
    return c2w


def _person_center_and_head(
    posed_gs,
    world_up: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    xyz = posed_gs.xyz.detach().cpu().numpy().astype(np.float32)
    center = np.median(xyz, axis=0)
    proj = xyz @ world_up
    max_proj = float(np.max(proj))
    center_proj = float(np.dot(center, world_up))
    head_top = center + (max_proj - center_proj) * world_up
    return center, head_top


def _estimate_camera_radius(
    tsdf_camera_files: Dict[int, Dict[str, Dict]],
    tsdf_cam_ids: List[int],
    frame_name: str,
    center: np.ndarray,
    world_up: np.ndarray,
) -> float:
    dists = []
    for cam_id in tsdf_cam_ids:
        cam_sample = tsdf_camera_files[cam_id][frame_name]
        c2w = cam_sample["c2w"]
        cam_pos = c2w[:3, 3].detach().cpu().numpy().astype(np.float32)
        vec = cam_pos - center
        vec = vec - np.dot(vec, world_up) * world_up
        dist = float(np.linalg.norm(vec))
        if dist > 1e-6:
            dists.append(dist)
    if not dists:
        return 1.0
    return float(np.median(dists))


def _virtual_top_cameras_for_person(
    posed_gs,
    tsdf_camera_files: Dict[int, Dict[str, Dict]],
    tsdf_cam_ids: List[int],
    frame_name: str,
    world_up: np.ndarray,
    cfg: TsdfConfig,
    forward_is_positive_z: bool,
) -> List[np.ndarray]:
    if cfg.tsdf_virtual_cam_count <= 0:
        return []
    center, head_top = _person_center_and_head(posed_gs, world_up)
    radius = _estimate_camera_radius(tsdf_camera_files, tsdf_cam_ids, frame_name, center, world_up)
    radius = max(cfg.tsdf_virtual_cam_min_radius, radius * cfg.tsdf_virtual_cam_radius_scale)
    cam_center = head_top + world_up * float(cfg.tsdf_virtual_cam_height_offset)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, world_up))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    basis_u = _normalize_vec(np.cross(world_up, ref))
    basis_v = _normalize_vec(np.cross(world_up, basis_u))
    cams = []
    for idx in range(int(cfg.tsdf_virtual_cam_count)):
        theta = 2.0 * np.pi * float(idx) / float(cfg.tsdf_virtual_cam_count)
        pos = cam_center + radius * (np.cos(theta) * basis_u + np.sin(theta) * basis_v)
        c2w = _look_at_c2w(pos, center, world_up, forward_is_positive_z)
        cams.append(c2w)
    return cams


def _virtual_ring_cameras_for_person(
    posed_gs,
    world_up: np.ndarray,
    cfg: TsdfConfig,
    forward_is_positive_z: bool,
) -> List[np.ndarray]:
    if cfg.tsdf_virtual_cam_ring_count <= 0:
        return []
    center, _ = _person_center_and_head(posed_gs, world_up)
    radius = float(cfg.tsdf_virtual_cam_ring_radius)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, world_up))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    basis_u = _normalize_vec(np.cross(world_up, ref))
    basis_v = _normalize_vec(np.cross(world_up, basis_u))
    cams = []
    for idx in range(int(cfg.tsdf_virtual_cam_ring_count)):
        theta = 2.0 * np.pi * float(idx) / float(cfg.tsdf_virtual_cam_ring_count)
        pos = center + radius * (np.cos(theta) * basis_u + np.sin(theta) * basis_v)
        c2w = _look_at_c2w(pos, center, world_up, forward_is_positive_z)
        cams.append(c2w)
    return cams


def _postprocess_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    cfg: TsdfConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    if not cfg.tsdf_postprocess_enable:
        return vertices, faces
    if vertices.size == 0 or faces.size == 0:
        return vertices, faces
    mesh_tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if cfg.tsdf_postprocess_fill_holes:
        try:
            mesh_tm.fill_holes()
        except Exception:
            try:
                trimesh.repair.fill_holes(mesh_tm)
            except Exception:
                pass
    if not cfg.tsdf_postprocess_smooth:
        return (
            np.asarray(mesh_tm.vertices, dtype=np.float32),
            np.asarray(mesh_tm.faces, dtype=np.int32),
        )
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_tm.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_tm.faces)
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_non_manifold_edges()
    if cfg.tsdf_postprocess_smooth_method == "taubin":
        mesh_o3d = mesh_o3d.filter_smooth_taubin(
            number_of_iterations=int(cfg.tsdf_postprocess_smooth_iters),
            lambda_filter=float(cfg.tsdf_postprocess_taubin_lambda),
            mu=float(cfg.tsdf_postprocess_taubin_mu),
        )
    else:
        mesh_o3d = mesh_o3d.filter_smooth_simple(
            number_of_iterations=int(cfg.tsdf_postprocess_smooth_iters)
        )
    mesh_o3d.compute_vertex_normals()
    return (
        np.asarray(mesh_o3d.vertices, dtype=np.float32),
        np.asarray(mesh_o3d.triangles, dtype=np.int32),
    )


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
        height, width = render_hw
        use_virtual_cams = bool(tsdf_cfg.tsdf_virtual_cam_enable) and int(tsdf_cfg.tsdf_virtual_cam_count) > 0
        if use_virtual_cams and len(tsdf_cam_ids) > 0:
            ref_cam_sample = tsdf_camera_files[tsdf_cam_ids[0]][frame_name]
            K_ref = ref_cam_sample["K"]
            K_ref_np = K_ref.detach().cpu().numpy().astype(np.float32)
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if n_persons > 0:
                ref_center, _ = _person_center_and_head(all_posed_gs_list[0], world_up)
                cam_pos = ref_cam_sample["c2w"][:3, 3].detach().cpu().numpy().astype(np.float32)
                cam_forward = ref_cam_sample["c2w"][:3, 2].detach().cpu().numpy().astype(np.float32)
                forward_is_positive_z = float(np.dot(cam_forward, (ref_center - cam_pos))) >= 0.0
            else:
                forward_is_positive_z = False
        else:
            use_virtual_cams = False

        # Step 1: Get depth maps from all cameras for each person
        depth_entries_by_person: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            person_idx: [] for person_idx in range(n_persons)
        }
        for cam_id in tsdf_cam_ids:

            # - Parse camera params
            cam_sample = tsdf_camera_files[cam_id][frame_name]
            K = cam_sample["K"]
            c2w = cam_sample["c2w"]

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

        if use_virtual_cams:
            up_directions = [world_up]
            if tsdf_cfg.tsdf_virtual_cam_include_bottom:
                up_directions.append(-world_up)
            for person_idx, posed_gs in enumerate(all_posed_gs_list):
                ring_c2ws = _virtual_ring_cameras_for_person(
                    posed_gs,
                    world_up,
                    tsdf_cfg,
                    forward_is_positive_z,
                )
                virtual_c2ws: List[np.ndarray] = []
                virtual_c2ws.extend(ring_c2ws)
                for up_dir in up_directions:
                    virtual_c2ws.extend(
                        _virtual_top_cameras_for_person(
                            posed_gs,
                            tsdf_camera_files,
                            tsdf_cam_ids,
                            frame_name,
                            up_dir,
                            tsdf_cfg,
                            forward_is_positive_z,
                        )
                    )
                for c2w_np in virtual_c2ws:
                    c2w = torch.from_numpy(c2w_np).float().to(K_ref.device)
                    render_res = render_func(
                        posed_gs,
                        Camera.from_c2w(c2w, K_ref, height, width),
                    )
                    depth = (
                        render_res["comp_depth"][0, ..., 0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
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
                    depth_entries_by_person[person_idx].append(
                        (
                            depth,
                            K_ref_np,
                            c2w_np.astype(np.float32),
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

            vertices, faces = _postprocess_mesh(vertices, faces, tsdf_cfg)

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
