from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from training.helpers.dataset import root_dir_to_all_cameras_dir, root_dir_to_smplx_dir


def _load_camera_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        if "intrinsics" not in data or "extrinsics" not in data:
            raise KeyError(f"Missing intrinsics/extrinsics in {path}")
        intr = np.asarray(data["intrinsics"])
        extr = np.asarray(data["extrinsics"])

    if intr.shape == (3, 3):
        intr = intr[None, ...]
    if extr.shape == (3, 4):
        extr = extr[None, ...]
    if intr.shape != (1, 3, 3):
        raise ValueError(f"Unexpected intrinsics shape {intr.shape} in {path}")
    if extr.shape != (1, 3, 4):
        raise ValueError(f"Unexpected extrinsics shape {extr.shape} in {path}")
    return intr[0].astype(np.float32), extr[0].astype(np.float32)


def _save_camera_file(path: Path, intrinsics: np.ndarray, extrinsics: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        intrinsics=intrinsics[None, ...].astype(np.float32),
        extrinsics=extrinsics[None, ...].astype(np.float32),
    )


def _camera_center_from_w2c(R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    return -R_w2c.T @ t_w2c


def _estimate_min_fov_from_intrinsics(intrinsics: np.ndarray) -> float:
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    width = max(1.0, 2.0 * cx)
    height = max(1.0, 2.0 * cy)
    fov_x = 2.0 * np.arctan2(width, 2.0 * fx)
    fov_y = 2.0 * np.arctan2(height, 2.0 * fy)
    return float(min(fov_x, fov_y))


def _look_at_w2c(camera_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    forward = target - camera_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("Camera position and target are too close for look-at.")
    forward = forward / forward_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        raise ValueError("Up vector is parallel to view direction.")
    right = right / right_norm

    down = np.cross(forward, right)
    R_w2c = np.stack([right, down, forward], axis=0).astype(np.float32)
    t_w2c = (-R_w2c @ camera_pos).astype(np.float32)
    return R_w2c, t_w2c


@dataclass
class RuntimeCameraConfig:
    strategy: str = "static_orbit"
    body_radius_m: float = 1.3
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)


class CameraTrajectoryStrategy(ABC):
    @abstractmethod
    def generate(
        self,
        target_cam_ids: Sequence[int],
        all_target_cam_ids: Sequence[int],
        frame_names: Sequence[str],
        source_intrinsics_by_frame: Dict[str, np.ndarray],
        source_extrinsics_by_frame: Dict[str, np.ndarray],
        centers_by_frame: Dict[str, np.ndarray],
    ) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Return:
            camera_id -> frame_name -> (intrinsics(3x3), extrinsics(3x4))
        """


class StaticOrbitStrategy(CameraTrajectoryStrategy):
    def __init__(self, body_radius_m: float = 1.3, up: Sequence[float] = (0.0, 1.0, 0.0)):
        self.body_radius_m = float(body_radius_m)
        self.up = np.asarray(up, dtype=np.float32)

    def _compute_orbit_radius(
        self,
        centers_world: np.ndarray,
        source_intrinsics: np.ndarray,
        source_extrinsics_by_frame: Dict[str, np.ndarray],
        centers_by_frame: Dict[str, np.ndarray],
    ) -> float:
        center = np.median(centers_world, axis=0)
        offsets = np.linalg.norm(centers_world - center[None, :], axis=1)
        max_offset = float(np.max(offsets)) if offsets.size > 0 else 0.0

        min_fov = _estimate_min_fov_from_intrinsics(source_intrinsics)
        if min_fov <= 1e-6:
            r_min = self.body_radius_m + max_offset
        else:
            r_min = (self.body_radius_m + max_offset) / max(np.tan(min_fov / 2.0), 1e-6)

        src_dists = []
        for frame_name, center_t in centers_by_frame.items():
            extr = source_extrinsics_by_frame.get(frame_name)
            if extr is None:
                continue
            cam_center = _camera_center_from_w2c(extr[:3, :3], extr[:3, 3])
            src_dists.append(float(np.linalg.norm(center_t - cam_center)))
        r_src = float(np.median(src_dists)) if src_dists else r_min
        return max(r_min, r_src)

    def generate(
        self,
        target_cam_ids: Sequence[int],
        all_target_cam_ids: Sequence[int],
        frame_names: Sequence[str],
        source_intrinsics_by_frame: Dict[str, np.ndarray],
        source_extrinsics_by_frame: Dict[str, np.ndarray],
        centers_by_frame: Dict[str, np.ndarray],
    ) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        if not centers_by_frame:
            raise RuntimeError("No valid SMPL-X centers available for runtime virtual camera generation.")
        if not frame_names:
            raise RuntimeError("No frames available for runtime virtual camera generation.")
        if not all_target_cam_ids:
            return {}

        centers_world = np.stack(list(centers_by_frame.values()), axis=0)
        center = np.median(centers_world, axis=0)
        source_intrinsics = source_intrinsics_by_frame[frame_names[0]]
        radius = self._compute_orbit_radius(
            centers_world=centers_world,
            source_intrinsics=source_intrinsics,
            source_extrinsics_by_frame=source_extrinsics_by_frame,
            centers_by_frame=centers_by_frame,
        )

        generated: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        total_targets = len(all_target_cam_ids)
        for cam_id in target_cam_ids:
            if cam_id not in all_target_cam_ids:
                raise ValueError(f"Camera id {cam_id} missing from planned target camera ids.")
            cam_index = all_target_cam_ids.index(cam_id)
            theta = 2.0 * np.pi * cam_index / total_targets
            cam_pos = center + np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=np.float32) * radius
            R_w2c, t_w2c = _look_at_w2c(cam_pos, center, self.up)
            extrinsics = np.concatenate([R_w2c, t_w2c[:, None]], axis=1).astype(np.float32)

            generated[cam_id] = {}
            for frame_name in frame_names:
                intrinsics = source_intrinsics_by_frame[frame_name]
                generated[cam_id][frame_name] = (intrinsics, extrinsics)
        return generated


def build_camera_strategy(cfg: RuntimeCameraConfig) -> CameraTrajectoryStrategy:
    strategy_name = str(cfg.strategy).lower()
    if strategy_name == "static_orbit":
        return StaticOrbitStrategy(body_radius_m=cfg.body_radius_m, up=cfg.up)
    raise ValueError(f"Unknown runtime camera strategy: {cfg.strategy}")


class VirtualCameraPlanner:
    def __init__(
        self,
        scene_root_dir: Path,
        src_cam_id: int,
        all_target_cam_ids: Sequence[int],
        strategy: CameraTrajectoryStrategy,
    ) -> None:
        self.scene_root_dir = Path(scene_root_dir)
        self.src_cam_id = int(src_cam_id)
        self.all_target_cam_ids = list(all_target_cam_ids)
        self.strategy = strategy

    def _normalize_frame_names(self, frame_paths: Sequence[Path | str]) -> List[str]:
        frame_names: List[str] = []
        for path in frame_paths:
            if isinstance(path, Path):
                frame_names.append(path.stem)
            else:
                frame_names.append(Path(path).stem)
        frame_names = sorted(set(frame_names))
        return frame_names

    def _load_source_camera_data(
        self, frame_names: Sequence[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        source_dir = root_dir_to_all_cameras_dir(self.scene_root_dir) / f"{self.src_cam_id}"
        if not source_dir.exists():
            raise FileNotFoundError(f"Source camera dir not found: {source_dir}")

        intr_by_frame: Dict[str, np.ndarray] = {}
        extr_by_frame: Dict[str, np.ndarray] = {}
        for frame_name in frame_names:
            cam_file = source_dir / f"{frame_name}.npz"
            if not cam_file.exists():
                raise FileNotFoundError(f"Missing source camera file: {cam_file}")
            intr, extr = _load_camera_file(cam_file)
            intr_by_frame[frame_name] = intr
            extr_by_frame[frame_name] = extr
        return intr_by_frame, extr_by_frame

    def _load_centers_by_frame(self, frame_names: Sequence[str]) -> Dict[str, np.ndarray]:
        smplx_dir = root_dir_to_smplx_dir(self.scene_root_dir)
        centers_by_frame: Dict[str, np.ndarray] = {}
        for frame_name in frame_names:
            smplx_path = smplx_dir / f"{frame_name}.npz"
            if not smplx_path.exists():
                continue
            with np.load(smplx_path, allow_pickle=True) as data:
                if "trans" not in data:
                    continue
                trans = np.asarray(data["trans"])
            if trans.size == 0:
                continue
            centers_by_frame[frame_name] = np.mean(trans, axis=0).astype(np.float32)
        return centers_by_frame

    def _camera_files_complete(self, cam_id: int, frame_names: Sequence[str]) -> bool:
        cam_dir = root_dir_to_all_cameras_dir(self.scene_root_dir) / f"{cam_id}"
        if not cam_dir.exists():
            return False
        for frame_name in frame_names:
            if not (cam_dir / f"{frame_name}.npz").exists():
                return False
        return True

    def ensure_camera_files(
        self,
        target_cam_ids: Sequence[int],
        frame_paths: Sequence[Path | str],
    ) -> None:
        frame_names = self._normalize_frame_names(frame_paths)
        if not frame_names:
            raise RuntimeError("No frame names provided to virtual camera planner.")

        missing_cam_ids = [
            int(cam_id)
            for cam_id in target_cam_ids
            if not self._camera_files_complete(int(cam_id), frame_names)
        ]
        if not missing_cam_ids:
            return

        source_intr_by_frame, source_extr_by_frame = self._load_source_camera_data(frame_names)
        centers_by_frame = self._load_centers_by_frame(frame_names)
        generated = self.strategy.generate(
            target_cam_ids=missing_cam_ids,
            all_target_cam_ids=self.all_target_cam_ids,
            frame_names=frame_names,
            source_intrinsics_by_frame=source_intr_by_frame,
            source_extrinsics_by_frame=source_extr_by_frame,
            centers_by_frame=centers_by_frame,
        )

        for cam_id, frame_to_camera in generated.items():
            cam_dir = root_dir_to_all_cameras_dir(self.scene_root_dir) / f"{cam_id}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            for frame_name, (intrinsics, extrinsics) in frame_to_camera.items():
                _save_camera_file(cam_dir / f"{frame_name}.npz", intrinsics, extrinsics)
