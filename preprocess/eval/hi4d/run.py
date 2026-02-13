from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import tyro


REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class Scene:
    seq_name: str
    cam_id: int
    raw_gt_dir_path: str


@dataclass
class SlurmConfig:
    job_name: str = "preprocess_eval_hi4d"
    slurm_script: Path = Path("preprocess/eval/hi4d/submit.slurm")
    array_parallelism: Optional[int] = None


@dataclass
class Config:
    repo_dir: Path = REPO_ROOT
    scenes_dir: Optional[Path] = Path("preprocess/scenes")
    output_root_dir: Path = Path("/scratch/izar/cizinsky/thesis/gt_scene_data")
    smpl2smplx_script: Path = Path("submodules/smplx/tools/run_conversion.sh")
    ensure_raw_data_script: Path = Path("preprocess/eval/hi4d/helpers/ensure_raw_hi4d_data.py")
    hf_repo_id: str = "ludekcizinsky/hi4d"
    ensure_raw_data: bool = True
    scenes: List[Scene] = field(default_factory=list)
    seq_name_includes: Optional[str] = None
    seq_name_prefix: str = "hi4d_"
    frame_name_num_digits: int = 6
    include_meta: bool = True
    include_depths_if_available: bool = False
    overwrite_output_scene: bool = True
    update_scene_registry: bool = False
    run_all: bool = False
    submit: bool = False
    dry_run: bool = False
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


def _resolve_repo_path(repo_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_dir / path


def _parse_scene_data(data: dict, path: Path) -> Optional[Scene]:
    required = ["seq_name", "cam_id", "raw_gt_dir_path"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing '{key}' in {path}.")

    raw_gt_dir_path = data["raw_gt_dir_path"]
    if raw_gt_dir_path is None or str(raw_gt_dir_path).strip() == "":
        return None

    return Scene(
        seq_name=str(data["seq_name"]),
        cam_id=int(data["cam_id"]),
        raw_gt_dir_path=str(raw_gt_dir_path),
    )


def _load_scenes_from_dir(path: Path) -> List[Scene]:
    if not path.exists():
        return []

    json_files = sorted(path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No scene JSON files found in {path}.")

    scenes: List[Scene] = []
    for json_path in json_files:
        with json_path.open() as f:
            data = json.load(f)
        scene = _parse_scene_data(data, json_path)
        if scene is not None:
            scenes.append(scene)
    return scenes


def _load_scenes(cfg: Config) -> List[Scene]:
    scenes_dir = cfg.scenes_dir
    if scenes_dir is not None:
        scenes_dir = _resolve_repo_path(cfg.repo_dir, scenes_dir)
        scenes = _load_scenes_from_dir(scenes_dir)
        if scenes:
            return scenes
    if not cfg.scenes:
        raise ValueError("No eval scenes provided.")
    return cfg.scenes


def _filter_scenes(cfg: Config, scenes: Sequence[Scene]) -> List[Scene]:
    filtered = list(scenes)
    if cfg.seq_name_prefix:
        filtered = [scene for scene in filtered if scene.seq_name.startswith(cfg.seq_name_prefix)]
    if cfg.seq_name_includes:
        needle = cfg.seq_name_includes
        filtered = [scene for scene in filtered if needle in scene.seq_name]
    return filtered


def _forwarded_args(argv: Sequence[str]) -> List[str]:
    forwarded: List[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--submit", "--no-submit", "--run-all", "--no-run-all", "--dry-run", "--no-dry-run"}:
            continue
        if arg.startswith("--submit=") or arg.startswith("--run-all=") or arg.startswith("--dry-run="):
            continue
        forwarded.append(arg)
    return forwarded


def _read_slurm_directives(path: Path) -> List[str]:
    directives: List[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("#SBATCH"):
                directives.append(line)
    return directives


def _print_submission_summary(
    cfg: Config,
    scenes: Sequence[Scene],
    slurm_script: Path,
    array_spec: str,
) -> None:
    print("About to submit HI4D GT materialization array with:")
    print(f"  Slurm script: {slurm_script}")
    print(f"  Job name: {cfg.slurm.job_name}")
    print(f"  Array: {array_spec} ({len(scenes)} scenes)")
    print(f"  Output root: {cfg.output_root_dir}")
    if cfg.seq_name_prefix:
        print(f"  Prefix filter: '{cfg.seq_name_prefix}'")
    if cfg.seq_name_includes:
        print(f"  Includes filter: '{cfg.seq_name_includes}'")
    if cfg.slurm.array_parallelism:
        print(f"  Max parallelism: {cfg.slurm.array_parallelism}")
    directives = _read_slurm_directives(slurm_script)
    if directives:
        print("  Slurm directives:")
        for directive in directives:
            print(f"    {directive}")
    print("  Scenes:")
    for scene in scenes:
        print(
            f"    {scene.seq_name} | cam_id={scene.cam_id} | raw_gt_dir_path={scene.raw_gt_dir_path}"
        )


def _confirm_submit() -> bool:
    try:
        response = input("Press Enter to submit, or type anything to cancel: ")
    except EOFError:
        return False
    return response.strip() == ""


def _submit_array(cfg: Config, scenes: Sequence[Scene]) -> None:
    if not scenes:
        raise ValueError("No scenes to submit.")

    array_spec = f"0-{len(scenes) - 1}"
    if cfg.slurm.array_parallelism:
        array_spec = f"{array_spec}%{cfg.slurm.array_parallelism}"

    slurm_script = _resolve_repo_path(cfg.repo_dir, cfg.slurm.slurm_script)
    if not slurm_script.exists():
        raise FileNotFoundError(f"Slurm script not found: {slurm_script}")

    _print_submission_summary(cfg, scenes, slurm_script, array_spec)
    if not _confirm_submit():
        print("Submission cancelled.")
        return

    cmd: List[str] = [
        "sbatch",
        "--job-name",
        cfg.slurm.job_name,
        "--array",
        array_spec,
        "--export",
        "ALL",
        str(slurm_script),
    ]
    cmd.extend(_forwarded_args(sys.argv[1:]))

    if cfg.dry_run:
        print(" ".join(cmd))
        return

    subprocess.run(cmd, check=True)


def _resolve_scene_index() -> Optional[int]:
    env_idx = os.getenv("SLURM_ARRAY_TASK_ID")
    if env_idx is None or env_idx == "":
        return None
    try:
        return int(env_idx)
    except ValueError as exc:
        raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID='{env_idx}'") from exc


def _collect_camera_ids(scene_root_dir: Path) -> List[int]:
    images_root = scene_root_dir / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"Missing images root: {images_root}")
    cam_ids = sorted(
        int(p.name)
        for p in images_root.iterdir()
        if p.is_dir() and p.name.isdigit()
    )
    if not cam_ids:
        raise RuntimeError(f"No camera folders found under {images_root}")
    return cam_ids


def _numeric_stem_to_file(path_dir: Path, suffixes: Set[str]) -> Dict[int, Path]:
    if not path_dir.exists():
        return {}
    out: Dict[int, Path] = {}
    for p in sorted(path_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in suffixes:
            continue
        if not p.stem.isdigit():
            continue
        frame_id = int(p.stem)
        out[frame_id] = p
    return out


def _load_rgb_camera_calibration(scene_root_dir: Path) -> Dict[int, Dict[str, np.ndarray]]:
    cameras_file = scene_root_dir / "cameras" / "rgb_cameras.npz"
    if not cameras_file.exists():
        return {}

    data = np.load(cameras_file, allow_pickle=True)
    required = {"ids", "intrinsics", "extrinsics"}
    missing = required - set(data.files)
    if missing:
        raise RuntimeError(f"Missing camera keys {sorted(missing)} in {cameras_file}")

    cam_ids = data["ids"]
    intrinsics_all = data["intrinsics"]
    extrinsics_all = data["extrinsics"]
    if len(cam_ids) != intrinsics_all.shape[0] or len(cam_ids) != extrinsics_all.shape[0]:
        raise RuntimeError(f"Camera ids/intrinsics/extrinsics size mismatch in {cameras_file}")

    calib: Dict[int, Dict[str, np.ndarray]] = {}
    for idx, cam_id in enumerate(cam_ids):
        intr = intrinsics_all[idx]
        extr = extrinsics_all[idx]
        if intr.shape == (3, 3):
            intr = intr[None, ...]
        if extr.shape == (3, 4):
            extr = extr[None, ...]
        if intr.shape != (1, 3, 3) or extr.shape != (1, 3, 4):
            raise RuntimeError(
                f"Unexpected intr/extr shapes for cam {int(cam_id)} in {cameras_file}: "
                f"{intr.shape}, {extr.shape}"
            )
        calib[int(cam_id)] = {"intrinsics": intr, "extrinsics": extr}
    return calib


def _get_camera_frame_ids_for_cam(
    scene_root_dir: Path,
    cam_id: int,
    image_frame_ids: Set[int],
    rgb_calib: Dict[int, Dict[str, np.ndarray]],
) -> Set[int]:
    # Preferred: per-frame camera files already materialized in all_cameras.
    camera_map = _numeric_stem_to_file(scene_root_dir / "all_cameras" / str(cam_id), {".npz"})
    if camera_map:
        return set(camera_map.keys())

    # Fallback: raw HI4D layout with static camera calibration in cameras/rgb_cameras.npz.
    if cam_id in rgb_calib:
        # Static intr/extr are valid for all frames where images exist.
        return set(image_frame_ids)

    raise RuntimeError(
        f"No camera data found for cam {cam_id} under {scene_root_dir}. "
        "Expected either all_cameras/<cam>/*.npz or cameras/rgb_cameras.npz with this camera id."
    )


def _collect_common_frame_ids_per_camera(scene_root_dir: Path, cam_id: int) -> Set[int]:
    images_map = _numeric_stem_to_file(scene_root_dir / "images" / str(cam_id), IMAGE_SUFFIXES)
    masks_map = _numeric_stem_to_file(
        scene_root_dir / "seg" / "img_seg_mask" / str(cam_id) / "all",
        {".png", ".jpg", ".jpeg"},
    )
    rgb_calib = _load_rgb_camera_calibration(scene_root_dir)
    camera_frame_ids = _get_camera_frame_ids_for_cam(
        scene_root_dir,
        cam_id,
        set(images_map.keys()),
        rgb_calib,
    )
    common = set(images_map.keys()) & set(masks_map.keys()) & set(camera_frame_ids)
    if not common:
        raise RuntimeError(
            f"No common frame ids for cam {cam_id} in images/masks/cameras under {scene_root_dir}."
        )
    return common


def _copy_frames_with_map(
    src_map: Dict[int, Path],
    frame_old_to_new: Dict[int, int],
    dst_dir: Path,
    num_digits: int,
) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for old_id, new_id in frame_old_to_new.items():
        src = src_map.get(old_id)
        if src is None:
            continue
        dst = dst_dir / f"{new_id:0{num_digits}d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        copied += 1
    return copied


def _materialize_scene(cfg: Config, scene: Scene) -> Path:
    src_scene_dir = Path(scene.raw_gt_dir_path)
    if not src_scene_dir.exists():
        raise FileNotFoundError(f"Raw GT scene root does not exist: {src_scene_dir}")

    dst_scene_dir = cfg.output_root_dir / scene.seq_name
    if dst_scene_dir.exists() and cfg.overwrite_output_scene:
        shutil.rmtree(dst_scene_dir)
    dst_scene_dir.mkdir(parents=True, exist_ok=True)

    cam_ids = _collect_camera_ids(src_scene_dir)
    if scene.cam_id not in cam_ids:
        raise RuntimeError(
            f"Source camera id {scene.cam_id} not present in raw scene cameras {cam_ids}."
        )
    rgb_calib = _load_rgb_camera_calibration(src_scene_dir)

    common_frame_ids: Optional[Set[int]] = None
    for cam_id in cam_ids:
        cam_common = _collect_common_frame_ids_per_camera(src_scene_dir, cam_id)
        common_frame_ids = cam_common if common_frame_ids is None else (common_frame_ids & cam_common)
    if common_frame_ids is None or not common_frame_ids:
        raise RuntimeError("No shared frame ids across cameras for images/masks/all_cameras.")

    smpl_map = _numeric_stem_to_file(src_scene_dir / "smpl", {".npz"})
    if len(smpl_map) == 0:
        raise RuntimeError(
            f"No SMPL parameters found in {src_scene_dir}/smpl. "
            "This materializer expects raw GT to provide SMPL, then converts to SMPL-X."
        )
    common_frame_ids = common_frame_ids & set(smpl_map.keys())
    if not common_frame_ids:
        raise RuntimeError(
            f"No common frame ids after intersecting with SMPL files in {src_scene_dir}."
        )

    sorted_old_ids = sorted(common_frame_ids)
    frame_old_to_new = {old_id: new_idx for new_idx, old_id in enumerate(sorted_old_ids, start=1)}
    frame_new_to_old = {new_id: old_id for old_id, new_id in frame_old_to_new.items()}

    # Copy per-camera images/masks/cameras (+ optional depths).
    for cam_id in cam_ids:
        src_images_map = _numeric_stem_to_file(src_scene_dir / "images" / str(cam_id), IMAGE_SUFFIXES)
        copied_images = _copy_frames_with_map(
            src_images_map,
            frame_old_to_new,
            dst_scene_dir / "images" / str(cam_id),
            cfg.frame_name_num_digits,
        )
        if copied_images != len(frame_old_to_new):
            raise RuntimeError(
                f"Copied {copied_images}/{len(frame_old_to_new)} images for cam {cam_id}. "
                "Raw data is missing required frames."
            )

        src_mask_root = src_scene_dir / "seg" / "img_seg_mask" / str(cam_id)
        track_dirs = sorted([p for p in src_mask_root.iterdir() if p.is_dir()])
        if not track_dirs:
            raise RuntimeError(f"No segmentation track dirs in {src_mask_root}")
        for track_dir in track_dirs:
            src_track_map = _numeric_stem_to_file(track_dir, {".png", ".jpg", ".jpeg"})
            copied_masks = _copy_frames_with_map(
                src_track_map,
                frame_old_to_new,
                dst_scene_dir / "seg" / "img_seg_mask" / str(cam_id) / track_dir.name,
                cfg.frame_name_num_digits,
            )
            if copied_masks != len(frame_old_to_new):
                raise RuntimeError(
                    f"Copied {copied_masks}/{len(frame_old_to_new)} masks for cam {cam_id} track '{track_dir.name}'."
                )

        dst_cam_dir = dst_scene_dir / "all_cameras" / str(cam_id)
        src_cam_map = _numeric_stem_to_file(src_scene_dir / "all_cameras" / str(cam_id), {".npz"})
        if src_cam_map:
            copied_cams = _copy_frames_with_map(
                src_cam_map,
                frame_old_to_new,
                dst_cam_dir,
                cfg.frame_name_num_digits,
            )
            if copied_cams != len(frame_old_to_new):
                raise RuntimeError(
                    f"Copied {copied_cams}/{len(frame_old_to_new)} camera files for cam {cam_id}."
                )
        else:
            if cam_id not in rgb_calib:
                raise RuntimeError(
                    f"Cannot materialize cameras for cam {cam_id}: "
                    "missing all_cameras files and missing cameras/rgb_cameras.npz calibration."
                )
            dst_cam_dir.mkdir(parents=True, exist_ok=True)
            intr = rgb_calib[cam_id]["intrinsics"]
            extr = rgb_calib[cam_id]["extrinsics"]
            for old_id, new_id in frame_old_to_new.items():
                dst_file = dst_cam_dir / f"{new_id:0{cfg.frame_name_num_digits}d}.npz"
                np.savez(dst_file, intrinsics=intr, extrinsics=extr)

        if cfg.include_depths_if_available:
            src_depth_map = _numeric_stem_to_file(src_scene_dir / "depths" / str(cam_id), {".npy"})
            if src_depth_map:
                _copy_frames_with_map(
                    src_depth_map,
                    frame_old_to_new,
                    dst_scene_dir / "depths" / str(cam_id),
                    cfg.frame_name_num_digits,
                )

    # Copy root-level SMPL params. SMPL-X is generated afterwards from these canonical SMPL files.
    copied_smpl = _copy_frames_with_map(
        smpl_map,
        frame_old_to_new,
        dst_scene_dir / "smpl",
        cfg.frame_name_num_digits,
    )
    if copied_smpl != len(frame_old_to_new):
        raise RuntimeError(f"Copied {copied_smpl}/{len(frame_old_to_new)} SMPL files.")

    # Optional metadata.
    if cfg.include_meta and (src_scene_dir / "meta.npz").exists():
        shutil.copy2(src_scene_dir / "meta.npz", dst_scene_dir / "meta.npz")

    # Keep traceability between canonical names and raw ids.
    frame_map = {
        "num_frames": len(frame_old_to_new),
        "num_digits": cfg.frame_name_num_digits,
        "source_scene_dir": str(src_scene_dir),
        "canonical_scene_dir": str(dst_scene_dir),
        "camera_ids": cam_ids,
        "reference_camera_id": scene.cam_id,
        "old_to_new": {
            f"{old_id:0{cfg.frame_name_num_digits}d}": f"{new_id:0{cfg.frame_name_num_digits}d}"
            for old_id, new_id in frame_old_to_new.items()
        },
        "new_to_old": {
            f"{new_id:0{cfg.frame_name_num_digits}d}": f"{old_id:0{cfg.frame_name_num_digits}d}"
            for new_id, old_id in frame_new_to_old.items()
        },
    }
    with (dst_scene_dir / "frame_map.json").open("w") as f:
        json.dump(frame_map, f, indent=2)

    print(
        f"Materialized {scene.seq_name}: {len(frame_old_to_new)} frames, "
        f"cams={cam_ids}, src={src_scene_dir}, dst={dst_scene_dir}"
    )
    return dst_scene_dir


def _maybe_generate_smplx_from_smpl(cfg: Config, dst_scene_dir: Path) -> None:
    smplx_dir = dst_scene_dir / "smplx"
    if smplx_dir.exists() and any(smplx_dir.glob("*.npz")):
        return
    smpl_dir = dst_scene_dir / "smpl"
    if not smpl_dir.exists() or not any(smpl_dir.glob("*.npz")):
        raise RuntimeError(
            f"No SMPL-X files and no SMPL files to convert in {dst_scene_dir}."
        )

    smpl2smplx_script = _resolve_repo_path(cfg.repo_dir, cfg.smpl2smplx_script)
    if not smpl2smplx_script.exists():
        raise FileNotFoundError(f"SMPL conversion script not found: {smpl2smplx_script}")
    convert_cmd = [
        "bash",
        str(smpl2smplx_script),
        str(dst_scene_dir),
        "smpl",
        "smplx",
    ]
    subprocess.run(convert_cmd, check=True, cwd=str(cfg.repo_dir))


def _maybe_update_scene_registry(cfg: Config, scene: Scene, dst_scene_dir: Path) -> None:
    if not cfg.update_scene_registry:
        return
    if cfg.scenes_dir is None:
        return
    scenes_dir = _resolve_repo_path(cfg.repo_dir, cfg.scenes_dir)
    json_path = scenes_dir / f"{scene.seq_name}.json"
    if not json_path.exists():
        print(f"Warning: scene json not found for registry update: {json_path}")
        return
    with json_path.open() as f:
        data = json.load(f)
    data["raw_gt_dir_path"] = scene.raw_gt_dir_path
    data.pop("gt_dir_path", None)
    with json_path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"Updated scene registry: {json_path}")


def _infer_total_frames(scene: Scene, dst_scene_dir: Path) -> int:
    frame_map_path = dst_scene_dir / "frame_map.json"
    if frame_map_path.exists():
        try:
            with frame_map_path.open("r", encoding="utf-8") as f:
                frame_map = json.load(f)
            if isinstance(frame_map, dict) and "num_frames" in frame_map:
                return int(frame_map["num_frames"])
        except Exception:
            pass

    canonical_images = _numeric_stem_to_file(
        dst_scene_dir / "images" / str(scene.cam_id),
        IMAGE_SUFFIXES,
    )
    if canonical_images:
        return len(canonical_images)

    raw_images = _numeric_stem_to_file(
        Path(scene.raw_gt_dir_path) / "images" / str(scene.cam_id),
        IMAGE_SUFFIXES,
    )
    return len(raw_images)


def _write_preprocess_info(
    *,
    scene: Scene,
    dst_scene_dir: Path,
    success: bool,
    elapsed_seconds: float,
    total_frames: int,
    error_text: Optional[str],
) -> None:
    misc_dir = dst_scene_dir / "misc"
    misc_dir.mkdir(parents=True, exist_ok=True)
    info_path = misc_dir / "preprocess_info.txt"

    lines: List[str] = []
    lines.append(f"scene_name: {scene.seq_name}")
    lines.append(f"raw_gt_dir_path: {scene.raw_gt_dir_path}")
    lines.append(f"output_scene_dir: {dst_scene_dir}")
    lines.append(f"reference_camera_id: {scene.cam_id}")
    lines.append(f"status: {'success' if success else 'failed'}")
    lines.append(f"elapsed_seconds: {elapsed_seconds:.3f}")
    lines.append(f"total_frames: {total_frames}")
    if error_text is not None:
        lines.append("")
        lines.append("error:")
        lines.append(error_text.rstrip())
    lines.append("")

    with info_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _run_scene(cfg: Config, scene: Scene) -> None:
    dst_scene_dir = cfg.output_root_dir / scene.seq_name
    print(f"Materializing {scene.seq_name}")
    print(f"  raw_gt_dir_path={scene.raw_gt_dir_path}")
    print(f"  output_scene_dir={dst_scene_dir}")
    started_at = time.perf_counter()
    success = False
    error_text: Optional[str] = None
    total_frames = 0
    try:
        if cfg.ensure_raw_data:
            ensure_script = _resolve_repo_path(cfg.repo_dir, cfg.ensure_raw_data_script)
            if not ensure_script.exists():
                raise FileNotFoundError(f"Raw data ensure script not found: {ensure_script}")
            ensure_cmd = [
                "python",
                str(ensure_script),
                "--raw-scene-dir",
                str(scene.raw_gt_dir_path),
                "--seq-name",
                scene.seq_name,
                "--hf-repo-id",
                cfg.hf_repo_id,
            ]
            if cfg.dry_run:
                ensure_cmd.append("--dry-run")
            subprocess.run(ensure_cmd, check=True, cwd=str(cfg.repo_dir))
        if cfg.dry_run:
            return
        materialized_scene_dir = _materialize_scene(cfg, scene)
        _maybe_generate_smplx_from_smpl(cfg, materialized_scene_dir)
        _maybe_update_scene_registry(cfg, scene, materialized_scene_dir)
        success = True
    except Exception:
        error_text = traceback.format_exc()
        raise
    finally:
        if not cfg.dry_run:
            elapsed = time.perf_counter() - started_at
            try:
                total_frames = _infer_total_frames(scene, dst_scene_dir)
            except Exception:
                total_frames = 0
            _write_preprocess_info(
                scene=scene,
                dst_scene_dir=dst_scene_dir,
                success=success,
                elapsed_seconds=elapsed,
                total_frames=total_frames,
                error_text=error_text,
            )


def main() -> None:
    cfg = tyro.cli(Config)
    scenes = _filter_scenes(cfg, _load_scenes(cfg))

    if cfg.submit:
        _submit_array(cfg, scenes)
        return

    scene_index = _resolve_scene_index()
    if cfg.run_all:
        for scene in scenes:
            _run_scene(cfg, scene)
        return

    if scene_index is None:
        if len(scenes) == 1:
            _run_scene(cfg, scenes[0])
            return
        if not scenes:
            print("No scenes matched the filter.")
            sys.exit(1)
        print("Multiple scenes matched. Use --run-all or --submit.")
        sys.exit(1)

    if scene_index < 0 or scene_index >= len(scenes):
        raise IndexError(f"Scene index {scene_index} out of range (0..{len(scenes)-1}).")
    _run_scene(cfg, scenes[scene_index])


if __name__ == "__main__":
    main()
