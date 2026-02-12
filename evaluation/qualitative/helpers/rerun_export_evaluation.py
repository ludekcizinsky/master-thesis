from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import json

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
    epoch: str = "all"  # all, latest, epoch_0010, 10
    max_frames_per_camera: int = 0  # 0 means all
    max_mesh_frames: int = 0  # 0 means all
    include_nvs: bool = True
    include_pose_smplx: bool = False
    include_pose_smpl: bool = False
    include_meshes: bool = False
    spawn_viewer: bool = False
    app_id: str = "thesis_evaluation"
    output_prefix: str = "evaluation"


def _read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _numeric_files(root: Path, suffixes: Tuple[str, ...]) -> List[Path]:
    candidates: List[Path] = []
    for suffix in suffixes:
        candidates.extend(root.glob(f"*{suffix}"))
    return sorted_numeric_stems(candidates)


def _resize_if_needed(image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image
    return np.array(Image.fromarray(image).resize((target_w, target_h), resample=Image.BILINEAR))


def _stack_pred_gt(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> np.ndarray:
    if gt_rgb.shape[:2] != pred_rgb.shape[:2]:
        gt_rgb = _resize_if_needed(gt_rgb, (pred_rgb.shape[0], pred_rgb.shape[1]))
    return np.concatenate([pred_rgb, gt_rgb], axis=1)


def _log_nvs(
    rr,
    epoch_name: str,
    nvs_dir: Path,
    max_frames_per_camera: int,
    stats: Dict[str, int],
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None,
) -> None:
    pred_root = nvs_dir / "pred"
    gt_root = nvs_dir / "gt_inputs"
    if not pred_root.is_dir():
        return

    cam_dirs = sorted([path for path in pred_root.iterdir() if path.is_dir() and path.name.startswith("cam_")])
    if not cam_dirs:
        return

    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        pred_metrics_input = cam_dir / "metrics_input"
        gt_metrics_input = gt_root / cam_name / "metrics_input"
        if not pred_metrics_input.is_dir():
            continue

        frame_files = _numeric_files(pred_metrics_input, (".jpg", ".png"))
        if max_frames_per_camera > 0:
            frame_files = frame_files[:max_frames_per_camera]

        for pred_file in frame_files:
            if not pred_file.stem.isdigit():
                continue
            frame_idx = int(pred_file.stem)

            gt_file = gt_metrics_input / pred_file.name
            if not gt_file.is_file():
                alt_suffix = ".png" if pred_file.suffix.lower() == ".jpg" else ".jpg"
                gt_file = gt_metrics_input / f"{pred_file.stem}{alt_suffix}"
            if not gt_file.is_file():
                continue

            pred_rgb = _read_rgb(pred_file)
            gt_rgb = _read_rgb(gt_file)
            stacked = _stack_pred_gt(pred_rgb, gt_rgb)
            rr.set_time("frame", sequence=frame_idx)
            rr.log(f"evaluation/{epoch_name}/nvs/{cam_name}/pred_gt", rr.Image(stacked))
            stats["nvs_pred_gt_stacked"] += 1
            if progress is not None and task_id is not None:
                progress.advance(task_id)


def _list_nvs_camera_names(epoch_dir: Path) -> List[str]:
    nvs_pred_dir = epoch_dir / "nvs" / "pred"
    if not nvs_pred_dir.is_dir():
        return []
    cam_names = [
        path.name
        for path in sorted(nvs_pred_dir.iterdir())
        if path.is_dir() and path.name.startswith("cam_")
    ]
    return sorted(cam_names, key=lambda name: int(name.split("_", 1)[1]) if name.split("_", 1)[1].isdigit() else 10**9)


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


def _find_mesh_dir(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("posed_") and child.name.endswith("_meshes_per_frame"):
            return child
    return None


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


def _find_person_mesh_dirs(mesh_dir: Path) -> List[Path]:
    person_dirs = [
        child
        for child in mesh_dir.iterdir()
        if child.is_dir() and child.name.isdigit()
    ]
    return sorted(person_dirs, key=lambda path: int(path.name))


def _person_shade_rgb(base_rgb: Tuple[int, int, int], person_idx: int) -> Tuple[int, int, int]:
    """Return a stable shade variation of a base color for per-person visibility."""
    # Keep the same color family (orange for pred, green for gt) and vary brightness.
    brightness = (0.78, 1.00, 1.22, 0.62, 1.38)
    factor = brightness[person_idx % len(brightness)]
    return tuple(
        int(min(255, max(0, round(channel * factor))))
        for channel in base_rgb
    )


def _log_meshes(
    rr,
    epoch_name: str,
    task_name: str,
    mesh_dir: Optional[Path],
    split_name: str,
    max_mesh_frames: int,
    stats_key: str,
    stats: Dict[str, int],
    color_rgb: Tuple[int, int, int],
) -> None:
    if mesh_dir is None:
        return
    frame_files = _numeric_files(mesh_dir, (".obj",))
    if max_mesh_frames > 0:
        frame_files = frame_files[:max_mesh_frames]
    person_dirs = _find_person_mesh_dirs(mesh_dir)

    base_entity = f"evaluation/{epoch_name}/{task_name}/{split_name}/meshes"

    for mesh_path in frame_files:
        if mesh_path.stem.isdigit():
            rr.set_time("frame", sequence=int(mesh_path.stem))

        for person_dir in person_dirs:
            person_mesh_path = person_dir / mesh_path.name
            if not person_mesh_path.is_file():
                continue
            loaded_person = _load_mesh(person_mesh_path)
            if loaded_person is None:
                continue
            person_vertices, person_faces, person_normals = loaded_person
            person_idx = int(person_dir.name)
            person_entity = f"{base_entity}/person_{person_idx:02d}"
            person_rgb = _person_shade_rgb(color_rgb, person_idx)
            person_albedo_rgba = (
                int(person_rgb[0]),
                int(person_rgb[1]),
                int(person_rgb[2]),
                255,
            )
            person_kwargs = dict(
                vertex_positions=person_vertices,
                triangle_indices=person_faces,
                albedo_factor=person_albedo_rgba,
            )
            if person_normals is not None:
                person_kwargs["vertex_normals"] = person_normals
            rr.log(person_entity, rr.Mesh3D(**person_kwargs))
            stats[stats_key] = stats.get(stats_key, 0) + 1


def _log_pose_task(
    rr,
    epoch_name: str,
    task_name: str,
    task_dir: Path,
    include_meshes: bool,
    max_mesh_frames: int,
    stats: Dict[str, int],
) -> None:
    if not include_meshes:
        return

    spaces = ("root_aligned", "world_aligned")
    splits = ("pred", "gt_inputs")
    split_colors: Dict[str, Tuple[int, int, int]] = {
        "pred": (255, 165, 0),      # orange
        "gt_inputs": (0, 200, 0),   # green
    }

    for space_name in spaces:
        for split_name in splits:
            mesh_dir = _find_mesh_dir(task_dir / split_name / space_name)
            if mesh_dir is None:
                continue
            stats_key = f"pose_{task_name}_{space_name}_{split_name}_meshes"
            _log_meshes(
                rr,
                epoch_name,
                task_name,
                mesh_dir,
                f"{space_name}/{split_name}",
                max_mesh_frames,
                stats_key,
                stats,
                color_rgb=split_colors[split_name],
            )


def _log_pose_source_camera_and_images(
    rr,
    epoch_name: str,
    task_name: str,
    task_dir: Path,
    source_cam_id: Optional[int],
    max_frames: int,
    stats: Dict[str, int],
) -> None:
    if source_cam_id is None:
        return

    cam_dir = task_dir / "viz" / "all_cameras" / str(source_cam_id)
    img_dir = task_dir / "viz" / "images" / str(source_cam_id)
    if not cam_dir.is_dir() and not img_dir.is_dir():
        return

    cam_entity = f"evaluation/{epoch_name}/{task_name}/world_aligned/source_camera"
    cam_image_entity = f"{cam_entity}/image"
    source_view_entity = f"evaluation/{epoch_name}/{task_name}/source_view/rgb"

    resolution = _infer_image_resolution(img_dir)
    camera_files = _numeric_files(cam_dir, (".npz",)) if cam_dir.is_dir() else []
    image_files = _numeric_files(img_dir, (".jpg", ".png")) if img_dir.is_dir() else []
    if max_frames > 0:
        camera_files = camera_files[:max_frames]
        image_files = image_files[:max_frames]

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
                cam_image_entity,
                rr.Pinhole(
                    focal_length=[fx, fy],
                    principal_point=[cx, cy],
                    resolution=[int(resolution[0]), int(resolution[1])],
                ),
            )
        stats["pose_source_camera_frames"] = stats.get("pose_source_camera_frames", 0) + 1

    for image_path in image_files:
        if not image_path.stem.isdigit():
            continue
        frame_idx = int(image_path.stem)
        rr.set_time("frame", sequence=frame_idx)
        rr.log(source_view_entity, rr.Image(_read_rgb(image_path)))
        stats["pose_source_view_frames"] = stats.get("pose_source_view_frames", 0) + 1


def _send_nvs_blueprint(rr, epoch_name: str, cam_names: List[str], source_cam_id: Optional[int]) -> None:
    import rerun.blueprint as rrb

    source_cam_name = f"cam_{source_cam_id}" if source_cam_id is not None else None
    default_visible_name = (
        source_cam_name if source_cam_name in cam_names else cam_names[0]
    )
    cam_views = [
        rrb.Spatial2DView(
            origin=f"/evaluation/{epoch_name}/nvs/{cam_name}/pred_gt",
            name=cam_name,
            visible=(cam_name == default_visible_name),
        )
        for cam_name in cam_names
    ]
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                *cam_views,
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


def _send_pose_blueprint(rr, epoch_name: str, task_name: str) -> None:
    import rerun.blueprint as rrb

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Grid(
                rrb.Spatial3DView(
                    origin=f"/evaluation/{epoch_name}/{task_name}/root_aligned",
                    name="Root Aligned (Pred + GT)",
                    visible=True,
                ),
                rrb.Spatial3DView(
                    origin=f"/evaluation/{epoch_name}/{task_name}/world_aligned",
                    name="World Aligned (Pred + GT)",
                    visible=False,
                ),
                rrb.Spatial2DView(
                    origin=f"/evaluation/{epoch_name}/{task_name}/source_view/rgb",
                    name="Source View",
                    visible=True,
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


def _export_epoch(
    args: Args,
    epoch_name: str,
    output_rrd: Path,
    source_cam_id: Optional[int],
    progress: Optional[Progress] = None,
    nvs_task_id: Optional[int] = None,
) -> Dict[str, int]:
    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError("rerun-sdk is required. Install with: pip install rerun-sdk") from exc

    rr.init(f"{args.app_id}_{epoch_name}", spawn=args.spawn_viewer)
    rr.save(str(output_rrd))
    stats: Dict[str, int] = {
        "nvs_pred_gt_stacked": 0,
    }

    epoch_dir = args.exp_dir / "evaluation" / epoch_name

    if args.include_nvs:
        cam_names = _list_nvs_camera_names(epoch_dir)
        if cam_names:
            _send_nvs_blueprint(rr, epoch_name, cam_names, source_cam_id)

    pose_task_names: List[str] = []
    if args.include_pose_smplx:
        pose_task_names.append("pose_smplx")
    if args.include_pose_smpl:
        pose_task_names.append("pose_smpl")

    if not args.include_nvs and pose_task_names:
        # Pose-only mode: configure a pose-centric blueprint.
        _send_pose_blueprint(rr, epoch_name, pose_task_names[0])

    if args.include_nvs:
        nvs_dir = epoch_dir / "nvs"
        if nvs_dir.is_dir():
            pred_root = nvs_dir / "pred"
            total_frames = 0
            if pred_root.is_dir():
                cam_dirs = sorted(
                    [path for path in pred_root.iterdir() if path.is_dir() and path.name.startswith("cam_")]
                )
                for cam_dir in cam_dirs:
                    pred_metrics_input = cam_dir / "metrics_input"
                    if not pred_metrics_input.is_dir():
                        continue
                    frame_files = _numeric_files(pred_metrics_input, (".jpg", ".png"))
                    if args.max_frames_per_camera > 0:
                        frame_files = frame_files[: args.max_frames_per_camera]
                    total_frames += len(frame_files)

            if progress is not None and nvs_task_id is not None:
                progress.update(
                    nvs_task_id,
                    description=f"Exporting {epoch_name} NVS frames",
                    total=max(total_frames, 1),
                    completed=0,
                )

            if total_frames == 0 and progress is not None and nvs_task_id is not None:
                progress.update(
                    nvs_task_id,
                    description=f"{epoch_name} NVS: no frames found",
                    completed=1,
                )

            _log_nvs(
                rr,
                epoch_name,
                nvs_dir,
                args.max_frames_per_camera,
                stats,
                progress=progress,
                task_id=nvs_task_id,
            )

    for pose_task_name in pose_task_names:
        pose_task_dir = epoch_dir / pose_task_name
        if pose_task_dir.is_dir():
            rr.log(
                f"/evaluation/{epoch_name}/{pose_task_name}/root_aligned",
                rr.ViewCoordinates.RIGHT_HAND_Y_UP,
                static=True,
            )
            rr.log(
                f"/evaluation/{epoch_name}/{pose_task_name}/world_aligned",
                rr.ViewCoordinates.RIGHT_HAND_Y_UP,
                static=True,
            )
            _log_pose_task(
                rr,
                epoch_name,
                pose_task_name,
                pose_task_dir,
                include_meshes=args.include_meshes,
                max_mesh_frames=args.max_mesh_frames,
                stats=stats,
            )
            _log_pose_source_camera_and_images(
                rr,
                epoch_name,
                pose_task_name,
                pose_task_dir,
                source_cam_id=source_cam_id,
                max_frames=args.max_mesh_frames,
                stats=stats,
            )

    return stats


def _run(args: Args) -> None:
    console = Console()
    eval_root = args.exp_dir / "evaluation"
    epoch_names = resolve_epochs(eval_root, args.epoch)
    source_cam_id = _infer_source_cam_id(args.exp_dir)

    if args.output_dir is None:
        output_dir = args.exp_dir / "rerun"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    table = Table(title="Rerun Export: Evaluation")
    table.add_column("Epoch")
    table.add_column("Output")
    table.add_column("NVS pred|gt frames", justify="right")

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    with Progress(*progress_columns, console=console, transient=True) as progress:
        epoch_task = progress.add_task("Exporting evaluation epochs", total=max(len(epoch_names), 1))
        nvs_task = progress.add_task("Exporting NVS frames", total=1)
        if not args.include_nvs:
            progress.update(nvs_task, description="NVS export disabled", completed=1)
        if not epoch_names:
            progress.update(epoch_task, completed=1)
            progress.update(nvs_task, completed=1)
        for epoch_name in epoch_names:
            progress.update(epoch_task, description=f"Exporting evaluation epoch {epoch_name}")
            output_rrd = output_dir / f"{args.output_prefix}_{epoch_name}.rrd"
            stats = _export_epoch(
                args,
                epoch_name,
                output_rrd,
                source_cam_id,
                progress=progress,
                nvs_task_id=nvs_task,
            )
            table.add_row(
                epoch_name,
                str(output_rrd),
                str(stats["nvs_pred_gt_stacked"]),
            )
            progress.advance(epoch_task)
        if args.include_nvs:
            progress.update(nvs_task, description="NVS frame export complete")

    if source_cam_id is None:
        console.print("[yellow]Source camera could not be inferred from preprocess/scenes; defaulting to first camera visible.[/yellow]")
    else:
        console.print(f"[cyan]Default visible camera in each grid: cam_{source_cam_id}[/cyan]")
    console.print(table)


def main() -> None:
    args = tyro.cli(Args)
    _run(args)


if __name__ == "__main__":
    main()
