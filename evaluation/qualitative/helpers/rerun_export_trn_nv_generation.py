from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import sys

import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.qualitative.helpers.common import resolve_epochs, sorted_numeric_stems


@dataclass
class Args:
    exp_dir: Path
    output_dir: Optional[Path] = None
    epoch: str = "all"  # all, latest, epoch_0006, 6
    max_frames: int = 0  # 0 means all
    spawn_viewer: bool = False
    app_id: str = "thesis_trn_nv_generation"
    output_prefix: str = "trn_nv_generation"


def _read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _numeric_files(root: Path, suffixes: Tuple[str, ...]) -> List[Path]:
    candidates: List[Path] = []
    for suffix in suffixes:
        candidates.extend(root.glob(f"*{suffix}"))
    return sorted_numeric_stems(candidates)


def _infer_source_cam_id(exp_dir: Path) -> Optional[int]:
    scene_name = exp_dir.parent.name
    if not scene_name:
        return None
    scene_json_path = REPO_ROOT / "preprocess" / "scenes" / f"{scene_name}.json"
    if not scene_json_path.is_file():
        return None
    try:
        with open(scene_json_path, "r") as f:
            scene_meta = json.load(f)
    except Exception:
        return None
    cam_id = scene_meta.get("cam_id")
    if cam_id is None:
        return None
    try:
        return int(cam_id)
    except (TypeError, ValueError):
        return None


def _load_mesh(mesh_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if mesh is None:
        return None
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.size == 0 or faces.size == 0:
        return None
    normals = None
    try:
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if vertex_normals.shape == vertices.shape:
            normals = vertex_normals
    except Exception:
        normals = None
    return vertices, faces, normals


def _person_shade_rgb(base_rgb: Tuple[int, int, int], person_idx: int) -> Tuple[int, int, int]:
    brightness = (0.78, 1.00, 1.22, 0.62, 1.38)
    factor = brightness[person_idx % len(brightness)]
    return tuple(
        int(min(255, max(0, round(channel * factor))))
        for channel in base_rgb
    )


def _load_camera_npz(camera_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        with np.load(camera_path) as data:
            if "intrinsics" not in data or "extrinsics" not in data:
                return None
            intrinsics = data["intrinsics"]
            extrinsics = data["extrinsics"]
    except Exception:
        return None

    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if intrinsics.ndim == 3:
        intrinsics = intrinsics[0]
    if extrinsics.ndim == 3:
        extrinsics = extrinsics[0]
    if intrinsics.shape != (3, 3) or extrinsics.shape != (3, 4):
        return None
    return intrinsics, extrinsics


def _infer_image_resolution(images_dir: Path) -> Optional[Tuple[int, int]]:
    if not images_dir.is_dir():
        return None
    image_files = _numeric_files(images_dir, (".jpg", ".png"))
    if not image_files:
        return None
    sample = _read_rgb(image_files[0])
    return int(sample.shape[1]), int(sample.shape[0])  # width, height


def _list_virtual_cam_ids(debug_root: Path) -> List[int]:
    cam_ids: List[int] = []
    if not debug_root.is_dir():
        return cam_ids
    for path in sorted(debug_root.iterdir()):
        if path.is_dir() and path.name.isdigit():
            cam_ids.append(int(path.name))
    return cam_ids


def _send_blueprint(rr, epoch_name: str, virtual_cam_ids: List[int]) -> None:
    import rerun.blueprint as rrb

    right_views: List = []
    for cam_id in virtual_cam_ids:
        right_views.append(
            rrb.Spatial2DView(
                origin=f"/trn_nv_generation/{epoch_name}/debug/cam_{cam_id:03d}/triplet",
                name=f"cam_{cam_id}",
            )
        )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin=f"/trn_nv_generation/{epoch_name}/world",
                    name="World (Pose + Cameras)",
                ),
                rrb.Grid(*right_views) if right_views else rrb.TextDocumentView(
                    origin=f"/trn_nv_generation/{epoch_name}/debug",
                    name="Debug",
                ),
            ),
            rrb.BlueprintPanel(state="expanded"),
            rrb.TimePanel(
                timeline="frame",
                fps=20.0,
                loop_mode="all",
            ),
            collapse_panels=False,
        )
    )


def _log_pose_state_meshes(
    rr,
    epoch_name: str,
    mesh_dir: Path,
    max_frames: int,
    stats: Dict[str, int],
) -> None:
    if not mesh_dir.is_dir():
        return

    person_dirs = sorted(
        [path for path in mesh_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    frame_files = _numeric_files(mesh_dir, (".obj",))
    if max_frames > 0:
        frame_files = frame_files[:max_frames]

    for frame_mesh_path in frame_files:
        if not frame_mesh_path.stem.isdigit():
            continue
        frame_idx = int(frame_mesh_path.stem)
        rr.set_time("frame", sequence=frame_idx)
        for person_dir in person_dirs:
            person_mesh_path = person_dir / frame_mesh_path.name
            if not person_mesh_path.is_file():
                continue
            loaded = _load_mesh(person_mesh_path)
            if loaded is None:
                continue
            vertices, faces, normals = loaded
            person_idx = int(person_dir.name)
            person_rgb = _person_shade_rgb((255, 165, 0), person_idx)
            mesh_kwargs = dict(
                vertex_positions=vertices,
                triangle_indices=faces,
                albedo_factor=(person_rgb[0], person_rgb[1], person_rgb[2], 255),
            )
            if normals is not None:
                mesh_kwargs["vertex_normals"] = normals
            rr.log(
                f"trn_nv_generation/{epoch_name}/world/meshes/person_{person_idx:02d}",
                rr.Mesh3D(**mesh_kwargs),
            )
            stats["mesh_frames"] = stats.get("mesh_frames", 0) + 1


def _log_cameras(
    rr,
    epoch_name: str,
    cameras_root: Path,
    train_images_root: Path,
    source_cam_id: Optional[int],
    max_frames: int,
    stats: Dict[str, int],
) -> None:
    if not cameras_root.is_dir():
        return

    camera_dirs = [
        path for path in sorted(cameras_root.iterdir())
        if path.is_dir() and path.name.isdigit()
    ]
    for camera_dir in camera_dirs:
        cam_id = int(camera_dir.name)
        resolution = _infer_image_resolution(train_images_root / camera_dir.name)
        camera_files = _numeric_files(camera_dir, (".npz",))
        if max_frames > 0:
            camera_files = camera_files[:max_frames]

        cam_entity = f"trn_nv_generation/{epoch_name}/world/cameras/cam_{cam_id:03d}"
        for camera_path in camera_files:
            if not camera_path.stem.isdigit():
                continue
            frame_idx = int(camera_path.stem)
            loaded = _load_camera_npz(camera_path)
            if loaded is None:
                continue
            intrinsics, extrinsics = loaded
            rr.set_time("frame", sequence=frame_idx)
            rr.log(
                cam_entity,
                rr.Transform3D(
                    translation=extrinsics[:, 3],
                    mat3x3=extrinsics[:, :3],
                    relation=rr.TransformRelation.ChildFromParent,
                ),
            )
            if resolution is not None:
                fx = float(intrinsics[0, 0])
                fy = float(intrinsics[1, 1])
                cx = float(intrinsics[0, 2])
                cy = float(intrinsics[1, 2])
                rr.log(
                    f"{cam_entity}/image",
                    rr.Pinhole(
                        focal_length=[fx, fy],
                        principal_point=[cx, cy],
                        resolution=[int(resolution[0]), int(resolution[1])],
                    ),
                )
            stats["camera_frames"] = stats.get("camera_frames", 0) + 1

        if source_cam_id is not None and cam_id == source_cam_id:
            rr.log(
                f"{cam_entity}/label",
                rr.TextLog("source_camera"),
                static=True,
            )


def _log_nvs_debug_images(
    rr,
    epoch_name: str,
    debug_root: Path,
    max_frames: int,
    stats: Dict[str, int],
) -> None:
    if not debug_root.is_dir():
        return

    cam_dirs = [
        path for path in sorted(debug_root.iterdir())
        if path.is_dir() and path.name.isdigit()
    ]
    for cam_dir in cam_dirs:
        cam_id = int(cam_dir.name)
        frame_files = _numeric_files(cam_dir, (".png", ".jpg"))
        if max_frames > 0:
            frame_files = frame_files[:max_frames]
        for frame_path in frame_files:
            if not frame_path.stem.isdigit():
                continue
            frame_idx = int(frame_path.stem)
            joined = _read_rgb(frame_path)
            rr.set_time("frame", sequence=frame_idx)
            rr.log(
                f"trn_nv_generation/{epoch_name}/debug/cam_{cam_id:03d}/triplet",
                rr.Image(joined),
            )
            width = joined.shape[1]
            if width % 3 == 0:
                chunk_w = width // 3
                ref = joined[:, :chunk_w, :]
                rend = joined[:, chunk_w : 2 * chunk_w, :]
                refined = joined[:, 2 * chunk_w :, :]
                rr.log(
                    f"trn_nv_generation/{epoch_name}/debug/cam_{cam_id:03d}/reference",
                    rr.Image(ref),
                )
                rr.log(
                    f"trn_nv_generation/{epoch_name}/debug/cam_{cam_id:03d}/rendered",
                    rr.Image(rend),
                )
                rr.log(
                    f"trn_nv_generation/{epoch_name}/debug/cam_{cam_id:03d}/refined",
                    rr.Image(refined),
                )
            stats["debug_triplets"] = stats.get("debug_triplets", 0) + 1


def _export_epoch(
    args: Args,
    epoch_name: str,
    output_rrd: Path,
    source_cam_id: Optional[int],
) -> Dict[str, int]:
    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError("rerun-sdk is required. Install with: pip install rerun-sdk") from exc

    rr.init(f"{args.app_id}_{epoch_name}", spawn=args.spawn_viewer)
    rr.save(str(output_rrd))

    train_root = args.exp_dir / "input_data" / "train"
    misc_root = train_root / "misc" / "nv_generation"
    pose_state_epoch_dir = misc_root / "pose_state" / epoch_name
    mesh_dir = pose_state_epoch_dir / "posed_smplx_meshes_per_frame"
    debug_root = misc_root / "est_images_debug"
    cameras_root = train_root / "all_cameras"
    train_images_root = train_root / "images"

    virtual_cam_ids = _list_virtual_cam_ids(debug_root)
    _send_blueprint(rr, epoch_name, virtual_cam_ids)

    rr.log(
        f"/trn_nv_generation/{epoch_name}/world",
        rr.ViewCoordinates.RIGHT_HAND_Y_UP,
        static=True,
    )

    stats: Dict[str, int] = {}
    _log_pose_state_meshes(
        rr,
        epoch_name,
        mesh_dir=mesh_dir,
        max_frames=args.max_frames,
        stats=stats,
    )
    _log_cameras(
        rr,
        epoch_name,
        cameras_root=cameras_root,
        train_images_root=train_images_root,
        source_cam_id=source_cam_id,
        max_frames=args.max_frames,
        stats=stats,
    )
    _log_nvs_debug_images(
        rr,
        epoch_name,
        debug_root=debug_root,
        max_frames=args.max_frames,
        stats=stats,
    )
    return stats


def _run(args: Args) -> None:
    console = Console()
    pose_state_root = args.exp_dir / "input_data" / "train" / "misc" / "nv_generation" / "pose_state"
    epoch_names = resolve_epochs(pose_state_root, args.epoch)
    source_cam_id = _infer_source_cam_id(args.exp_dir)

    if args.output_dir is None:
        output_dir = args.exp_dir / "rerun"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    table = Table(title="Rerun Export: Training NV Generation")
    table.add_column("Pose State Epoch")
    table.add_column("Output")
    table.add_column("Mesh frames", justify="right")
    table.add_column("Camera frames", justify="right")
    table.add_column("Debug triplets", justify="right")

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    with Progress(*progress_columns, console=console, transient=True) as progress:
        epoch_task = progress.add_task(
            "Exporting trn_nv_generation epochs",
            total=max(len(epoch_names), 1),
        )
        for epoch_name in epoch_names:
            progress.update(epoch_task, description=f"Exporting {epoch_name}")
            output_rrd = output_dir / f"{args.output_prefix}_{epoch_name}.rrd"
            stats = _export_epoch(
                args=args,
                epoch_name=epoch_name,
                output_rrd=output_rrd,
                source_cam_id=source_cam_id,
            )
            table.add_row(
                epoch_name,
                str(output_rrd),
                str(stats.get("mesh_frames", 0)),
                str(stats.get("camera_frames", 0)),
                str(stats.get("debug_triplets", 0)),
            )
            progress.advance(epoch_task)

    if source_cam_id is not None:
        console.print(f"[cyan]Source camera id: {source_cam_id}[/cyan]")
    else:
        console.print("[yellow]Source camera id could not be inferred from preprocess/scenes.[/yellow]")
    console.print(table)


def main() -> None:
    args = tyro.cli(Args)
    _run(args)


if __name__ == "__main__":
    main()
