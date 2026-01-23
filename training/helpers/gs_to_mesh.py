from __future__ import annotations

from dataclasses import dataclass, fields
from typing import List, Tuple, Dict

import numpy as np
import open3d as o3d

from training.helpers.gs_renderer import Camera

@dataclass
class MeshConfig:
    tsdf_voxel_size: float = 0.005
    tsdf_sdf_trunc: float = 0.02
    tsdf_depth_trunc: float = 12.0
    tsdf_alpha_thresh: float = 0.3


def mesh_config_from_cfg(cfg) -> MeshConfig:
    if cfg is None:
        return MeshConfig()

    if isinstance(cfg, dict):
        cfg_dict = cfg
    else:
        try:
            cfg_dict = {k: cfg.get(k) for k in cfg.keys()}
        except Exception:
            cfg_dict = {}

    kwargs = {}
    for field in fields(MeshConfig):
        if field.name in cfg_dict and cfg_dict[field.name] is not None:
            kwargs[field.name] = cfg_dict[field.name]
    return MeshConfig(**kwargs)


def tsdf_fuse_depths(
    depth_entries: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    voxel_size: float,
    sdf_trunc: float,
    depth_trunc: float,
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
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    return vertices, faces


def get_meshes_from_3dgs(mesh_cfg: MeshConfig, 
                         all_posed_gs_list: List, 
                         tsdf_camera_files: Dict[int, Dict[str, Dict]],
                         tsdf_cam_ids: List[int], 
                         frame_name: str, 
                         render_hw: Tuple[int, int],
                         render_func
                        ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:

    alpha_thresh = mesh_cfg.tsdf_alpha_thresh
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
            voxel_size=mesh_cfg.tsdf_voxel_size,
            sdf_trunc=mesh_cfg.tsdf_sdf_trunc,
            depth_trunc=mesh_cfg.tsdf_depth_trunc,
        )

        # - Store mesh for this person
        meshes_for_frame[person_idx] = (vertices, faces)

    return meshes_for_frame
