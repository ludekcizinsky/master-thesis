from __future__ import annotations

from dataclasses import dataclass, fields
from typing import List, Tuple, Dict

import numpy as np
import open3d as o3d


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
