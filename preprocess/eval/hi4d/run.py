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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import tyro
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess.identity_alignment import IdentityMatchResult, align_identities_from_masks
from utils.path_config import ensure_runtime_dirs, load_runtime_paths

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class Scene:
    seq_name: str
    cam_id: int
    raw_gt_dir_path: str
    identity_manual_preproc_to_gt: Optional[List[int]] = None


@dataclass
class SlurmConfig:
    job_name: str = "preprocess_eval_hi4d"
    slurm_script: Path = Path("preprocess/eval/hi4d/submit.slurm")
    array_parallelism: Optional[int] = None


@dataclass
class IdentityAlignmentDebugContext:
    preproc_cam_id: int
    gt_cam_id: int
    preproc_mask_root: Path
    gt_mask_root: Path
    preproc_track_ids: List[int]
    gt_track_ids: List[int]
    frame_pairs: List[Tuple[int, int]]
    preproc_track_maps: Dict[int, Dict[int, Path]]
    gt_track_maps: Dict[int, Dict[int, Path]]


@dataclass
class Config:
    repo_dir: Path = REPO_ROOT
    paths_config: Path = Path("configs/paths.yaml")
    scenes_dir: Optional[Path] = Path("preprocess/scenes")
    output_root_dir: Optional[Path] = None
    smpl2smplx_script: Path = Path("submodules/smplx/tools/run_conversion.sh")
    ensure_raw_data_script: Path = Path("preprocess/eval/hi4d/helpers/ensure_raw_hi4d_data.py")
    hf_repo_id: str = "ludekcizinsky/hi4d"
    ensure_raw_data: bool = True
    require_preprocessed_scene_dir: bool = True
    preprocessing_root_dir: Optional[Path] = None
    identity_alignment_enabled: bool = True
    identity_alignment_min_confidence: float = 0.5
    identity_alignment_max_frames: int = 50
    identity_alignment_frame_stride: int = 1
    identity_alignment_only: bool = False
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

    identity_cfg = data.get("identity_alignment", {})
    if identity_cfg is None:
        identity_cfg = {}
    if not isinstance(identity_cfg, dict):
        raise ValueError(
            f"Expected 'identity_alignment' to be an object in {path}, got {type(identity_cfg).__name__}."
        )
    manual_map = identity_cfg.get("manual_preproc_to_gt")
    if manual_map is not None:
        if not isinstance(manual_map, list) or not all(isinstance(v, int) for v in manual_map):
            raise ValueError(
                f"'identity_alignment.manual_preproc_to_gt' must be a list[int] in {path}."
            )

    return Scene(
        seq_name=str(data["seq_name"]),
        cam_id=int(data["cam_id"]),
        raw_gt_dir_path=str(raw_gt_dir_path),
        identity_manual_preproc_to_gt=manual_map,
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
    print(f"  Preprocessing root: {cfg.preprocessing_root_dir}")
    print(f"  Require preprocessing scene: {cfg.require_preprocessed_scene_dir}")
    print(f"  Identity alignment enabled: {cfg.identity_alignment_enabled}")
    print(f"  Identity alignment only: {cfg.identity_alignment_only}")
    if cfg.identity_alignment_enabled:
        print(f"  Identity min confidence: {cfg.identity_alignment_min_confidence}")
        print(f"  Identity frame stride/max: {cfg.identity_alignment_frame_stride}/{cfg.identity_alignment_max_frames}")
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
    runtime_paths = load_runtime_paths(_resolve_repo_path(cfg.repo_dir, cfg.paths_config))
    ensure_runtime_dirs(runtime_paths)

    _print_submission_summary(cfg, scenes, slurm_script, array_spec)
    if not _confirm_submit():
        print("Submission cancelled.")
        return

    cmd: List[str] = [
        "sbatch",
        "--job-name",
        cfg.slurm.job_name,
        "--output",
        str(runtime_paths.slurm_dir / "%x.%A_%a.out"),
        "--error",
        str(runtime_paths.slurm_dir / "%x.%A_%a.err"),
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


def _resolve_preprocessed_scene_dir(cfg: Config, scene: Scene) -> Path:
    if cfg.preprocessing_root_dir is None:
        raise ValueError(
            "Missing preprocessing_root_dir. Set cfg.preprocessing_root_dir before running."
        )
    return cfg.preprocessing_root_dir / scene.seq_name


def _collect_numeric_track_ids(mask_root: Path) -> List[int]:
    if not mask_root.exists():
        raise FileNotFoundError(f"Missing mask root: {mask_root}")
    track_ids = sorted(
        int(p.name)
        for p in mask_root.iterdir()
        if p.is_dir() and p.name.isdigit()
    )
    if len(track_ids) == 0:
        raise RuntimeError(f"No numeric person track ids found in {mask_root}")
    return track_ids


def _collect_cam_ids_under_mask_root(mask_root: Path) -> List[int]:
    if not mask_root.exists():
        return []
    return sorted(
        int(p.name)
        for p in mask_root.iterdir()
        if p.is_dir() and p.name.isdigit()
    )


def _resolve_preproc_cam_id(preproc_scene_dir: Path, gt_cam_id: int) -> int:
    preproc_mask_root = preproc_scene_dir / "seg" / "img_seg_mask"
    cam_ids = _collect_cam_ids_under_mask_root(preproc_mask_root)
    if len(cam_ids) == 0:
        raise RuntimeError(f"No camera masks found in preprocessing scene dir: {preproc_mask_root}")
    if gt_cam_id not in cam_ids:
        raise RuntimeError(
            "Preprocessing/GT camera mismatch for identity alignment. "
            f"Expected preprocessing masks for cam_id={gt_cam_id}, "
            f"but available preprocessing cam ids are {cam_ids} in {preproc_mask_root}."
        )
    return gt_cam_id


def _numeric_stem_to_any_file(path_dir: Path) -> Dict[int, Path]:
    if not path_dir.exists():
        return {}
    out: Dict[int, Path] = {}
    for p in sorted(path_dir.iterdir()):
        if not p.is_file():
            continue
        if not p.stem.isdigit():
            continue
        out[int(p.stem)] = p
    return out


def _load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.array(img.convert("L"))
    return arr > 0


def _sample_aligned_frame_pairs_for_identity_alignment(
    *,
    preproc_mask_root: Path,
    preproc_track_ids: Sequence[int],
    gt_mask_root: Path,
    gt_track_ids: Sequence[int],
    frame_stride: int,
    max_frames: int,
) -> List[Tuple[int, int]]:
    preproc_all = _numeric_stem_to_any_file(preproc_mask_root / "all")
    gt_all = _numeric_stem_to_any_file(gt_mask_root / "all")
    preproc_all_ids = sorted(preproc_all.keys())
    gt_all_ids = sorted(gt_all.keys())
    if len(preproc_all_ids) == 0 or len(gt_all_ids) == 0:
        raise RuntimeError(
            "Missing frame ids in 'all' masks for identity alignment. "
            f"preproc_count={len(preproc_all_ids)}, gt_count={len(gt_all_ids)}."
        )

    # Keep only frames where every compared track exists in each source.
    preproc_track_maps = {
        int(tid): _numeric_stem_to_any_file(preproc_mask_root / str(tid)) for tid in preproc_track_ids
    }
    gt_track_maps = {
        int(tid): _numeric_stem_to_any_file(gt_mask_root / str(tid)) for tid in gt_track_ids
    }

    valid_preproc_ids: List[int] = []
    for fid in preproc_all_ids:
        if all(fid in preproc_track_maps[int(tid)] for tid in preproc_track_ids):
            valid_preproc_ids.append(fid)

    valid_gt_ids: List[int] = []
    for fid in gt_all_ids:
        if all(fid in gt_track_maps[int(tid)] for tid in gt_track_ids):
            valid_gt_ids.append(fid)

    if len(valid_preproc_ids) == 0 or len(valid_gt_ids) == 0:
        raise RuntimeError(
            "No valid frames found for identity alignment after requiring per-track mask availability. "
            f"valid_preproc={len(valid_preproc_ids)}, valid_gt={len(valid_gt_ids)}."
        )

    # Temporal alignment by order: k-th valid preproc frame with k-th valid GT frame.
    n_pairs = min(len(valid_preproc_ids), len(valid_gt_ids))
    aligned_pairs = list(zip(valid_preproc_ids[:n_pairs], valid_gt_ids[:n_pairs]))

    frame_stride = max(1, int(frame_stride))
    aligned_pairs = aligned_pairs[::frame_stride]
    if max_frames > 0:
        aligned_pairs = aligned_pairs[:max_frames]
    if len(aligned_pairs) == 0:
        raise RuntimeError("Identity alignment frame sampling produced zero aligned frame pairs.")
    return aligned_pairs


def _compute_person_id_alignment(
    *,
    cfg: Config,
    scene: Scene,
    src_scene_dir: Path,
    preproc_scene_dir: Path,
) -> Tuple[IdentityMatchResult, IdentityAlignmentDebugContext]:
    preproc_cam_id = _resolve_preproc_cam_id(preproc_scene_dir, scene.cam_id)
    print(
        "Identity alignment camera selection: "
        f"preproc_cam_id={preproc_cam_id}, gt_cam_id={scene.cam_id}"
    )
    preproc_mask_root = preproc_scene_dir / "seg" / "img_seg_mask" / str(preproc_cam_id)
    gt_mask_root = src_scene_dir / "seg" / "img_seg_mask" / str(scene.cam_id)
    preproc_track_ids = _collect_numeric_track_ids(preproc_mask_root)
    gt_track_ids = _collect_numeric_track_ids(gt_mask_root)
    if len(preproc_track_ids) != len(gt_track_ids):
        raise RuntimeError(
            "Preprocessing and GT person counts differ. "
            f"preproc={len(preproc_track_ids)} ({preproc_track_ids}), "
            f"gt={len(gt_track_ids)} ({gt_track_ids})."
        )

    frame_pairs = _sample_aligned_frame_pairs_for_identity_alignment(
        preproc_mask_root=preproc_mask_root,
        preproc_track_ids=preproc_track_ids,
        gt_mask_root=gt_mask_root,
        gt_track_ids=gt_track_ids,
        frame_stride=cfg.identity_alignment_frame_stride,
        max_frames=cfg.identity_alignment_max_frames,
    )
    print(
        "Identity alignment temporal pairing: "
        f"n_pairs={len(frame_pairs)}, "
        f"first=(preproc:{frame_pairs[0][0]:06d}, gt:{frame_pairs[0][1]:06d}), "
        f"last=(preproc:{frame_pairs[-1][0]:06d}, gt:{frame_pairs[-1][1]:06d})"
    )
    frame_tokens = [str(idx) for idx in range(len(frame_pairs))]

    preproc_track_maps = {
        int(tid): _numeric_stem_to_any_file(preproc_mask_root / str(tid)) for tid in preproc_track_ids
    }
    gt_track_maps = {
        int(tid): _numeric_stem_to_any_file(gt_mask_root / str(tid)) for tid in gt_track_ids
    }

    def _get_preproc_mask(person_id: int, frame_name: str) -> np.ndarray:
        pair_idx = int(frame_name)
        preproc_frame_id, _ = frame_pairs[pair_idx]
        return _load_binary_mask(preproc_track_maps[person_id][preproc_frame_id])

    def _get_gt_mask(person_id: int, frame_name: str) -> np.ndarray:
        pair_idx = int(frame_name)
        _, gt_frame_id = frame_pairs[pair_idx]
        return _load_binary_mask(gt_track_maps[person_id][gt_frame_id])

    result = align_identities_from_masks(
        preproc_person_ids=preproc_track_ids,
        gt_person_ids=gt_track_ids,
        frame_names=frame_tokens,
        get_preproc_mask=_get_preproc_mask,
        get_gt_mask=_get_gt_mask,
        min_confidence=float(cfg.identity_alignment_min_confidence),
        manual_preproc_to_gt=scene.identity_manual_preproc_to_gt,
        raise_on_low_confidence=False,
    )
    debug_context = IdentityAlignmentDebugContext(
        preproc_cam_id=preproc_cam_id,
        gt_cam_id=scene.cam_id,
        preproc_mask_root=preproc_mask_root,
        gt_mask_root=gt_mask_root,
        preproc_track_ids=[int(v) for v in preproc_track_ids],
        gt_track_ids=[int(v) for v in gt_track_ids],
        frame_pairs=frame_pairs,
        preproc_track_maps=preproc_track_maps,
        gt_track_maps=gt_track_maps,
    )
    return result, debug_context


def _compute_mask_iou(path_a: Path, path_b: Path) -> float:
    a = _load_binary_mask(path_a)
    b = _load_binary_mask(path_b)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(union)


def _write_identity_alignment_debug(
    *,
    scene: Scene,
    misc_dir: Path,
    result: IdentityMatchResult,
    debug_context: IdentityAlignmentDebugContext,
) -> Path:
    misc_dir.mkdir(parents=True, exist_ok=True)
    debug_root = misc_dir / "identity_alignment_debug"
    if debug_root.exists():
        shutil.rmtree(debug_root)
    debug_root.mkdir(parents=True, exist_ok=True)

    pair_debug: List[Dict[str, Any]] = []
    for row_idx, matched_gt_id in enumerate(result.preproc_to_gt):
        preproc_pid = int(result.preproc_person_ids[row_idx])
        iou_records: List[Tuple[int, int, float, Path, Path]] = []
        for preproc_frame_id, gt_frame_id in debug_context.frame_pairs:
            preproc_mask_path = debug_context.preproc_track_maps[preproc_pid][preproc_frame_id]
            gt_mask_path = debug_context.gt_track_maps[int(matched_gt_id)][gt_frame_id]
            iou = _compute_mask_iou(preproc_mask_path, gt_mask_path)
            iou_records.append((preproc_frame_id, gt_frame_id, iou, preproc_mask_path, gt_mask_path))

        iou_records.sort(key=lambda x: x[2])
        worst = iou_records[0]
        best = iou_records[-1]

        pair_dir = debug_root / f"preproc_{preproc_pid}_to_gt_{matched_gt_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        worst_preproc_copy = pair_dir / f"worst_preproc_{worst[0]:06d}_gt_{worst[1]:06d}.png"
        worst_gt_copy = pair_dir / f"worst_gt_{worst[1]:06d}_preproc_{worst[0]:06d}.png"
        best_preproc_copy = pair_dir / f"best_preproc_{best[0]:06d}_gt_{best[1]:06d}.png"
        best_gt_copy = pair_dir / f"best_gt_{best[1]:06d}_preproc_{best[0]:06d}.png"
        shutil.copy2(worst[3], worst_preproc_copy)
        shutil.copy2(worst[4], worst_gt_copy)
        shutil.copy2(best[3], best_preproc_copy)
        shutil.copy2(best[4], best_gt_copy)

        pair_debug.append(
            {
                "preproc_person_id": preproc_pid,
                "matched_gt_person_id": int(matched_gt_id),
                "pair_confidence": float(result.pair_confidences[row_idx]),
                "worst_match": {
                    "preproc_frame_name": f"{worst[0]:06d}",
                    "gt_frame_name": f"{worst[1]:06d}",
                    "iou": float(worst[2]),
                    "preproc_mask_path": str(worst[3]),
                    "gt_mask_path": str(worst[4]),
                    "debug_preproc_mask_copy": str(worst_preproc_copy),
                    "debug_gt_mask_copy": str(worst_gt_copy),
                },
                "best_match": {
                    "preproc_frame_name": f"{best[0]:06d}",
                    "gt_frame_name": f"{best[1]:06d}",
                    "iou": float(best[2]),
                    "preproc_mask_path": str(best[3]),
                    "gt_mask_path": str(best[4]),
                    "debug_preproc_mask_copy": str(best_preproc_copy),
                    "debug_gt_mask_copy": str(best_gt_copy),
                },
            }
        )

    out_path = misc_dir / "identity_alignment_debug.json"
    payload = {
        "scene_name": scene.seq_name,
        "preproc_cam_id": debug_context.preproc_cam_id,
        "gt_cam_id": debug_context.gt_cam_id,
        "method": result.method,
        "preproc_person_ids": result.preproc_person_ids,
        "gt_person_ids": result.gt_person_ids,
        "preproc_to_gt": result.preproc_to_gt,
        "pair_confidences": result.pair_confidences,
        "confidence_min": result.confidence_min,
        "confidence_mean": result.confidence_mean,
        "n_frames_used": result.n_frames_used,
        "frame_pairs": [
            {"preproc_frame_name": f"{pre_id:06d}", "gt_frame_name": f"{gt_id:06d}"}
            for pre_id, gt_id in debug_context.frame_pairs
        ],
        "similarity_matrix": result.similarity_matrix,
        "pair_debug": pair_debug,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return out_path


def _copy_smpl_with_person_map(
    smpl_map: Dict[int, Path],
    frame_old_to_new: Dict[int, int],
    dst_dir: Path,
    num_digits: int,
    preproc_to_gt: Optional[Sequence[int]],
) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    person_keys = {
        "betas",
        "root_pose",
        "global_orient",
        "body_pose",
        "trans",
        "transl",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "lhand_pose",
        "rhand_pose",
        "expr",
        "expression",
        "contact",
    }
    person_index = None if preproc_to_gt is None else np.asarray(preproc_to_gt, dtype=np.int64)
    for old_id, new_id in frame_old_to_new.items():
        src = smpl_map.get(old_id)
        if src is None:
            continue
        with np.load(src, allow_pickle=True) as data:
            payload: Dict[str, Any] = {}
            for key in data.files:
                value = data[key]
                if (
                    person_index is not None
                    and key in person_keys
                    and isinstance(value, np.ndarray)
                    and value.ndim >= 1
                    and value.shape[0] == person_index.shape[0]
                ):
                    payload[key] = value[person_index]
                else:
                    payload[key] = value
        dst = dst_dir / f"{new_id:0{num_digits}d}.npz"
        np.savez(dst, **payload)
        copied += 1
    return copied


def _copy_meta_with_person_map(
    src_meta_path: Path,
    dst_meta_path: Path,
    preproc_to_gt: Optional[Sequence[int]],
) -> None:
    if preproc_to_gt is None:
        shutil.copy2(src_meta_path, dst_meta_path)
        return
    person_index = np.asarray(preproc_to_gt, dtype=np.int64)
    person_meta_keys = {"genders", "gender", "person_ids", "person_id"}
    with np.load(src_meta_path, allow_pickle=True) as data:
        payload: Dict[str, Any] = {}
        for key in data.files:
            value = data[key]
            if (
                key in person_meta_keys
                and isinstance(value, np.ndarray)
                and value.ndim >= 1
                and value.shape[0] == person_index.shape[0]
            ):
                payload[key] = value[person_index]
            else:
                payload[key] = value
    np.savez(dst_meta_path, **payload)


def _write_person_id_map(dst_scene_dir: Path, result: IdentityMatchResult) -> None:
    misc_dir = dst_scene_dir / "misc"
    misc_dir.mkdir(parents=True, exist_ok=True)
    out_path = misc_dir / "person_id_map.json"
    payload = {
        "preproc_person_ids": result.preproc_person_ids,
        "gt_person_ids": result.gt_person_ids,
        "preproc_to_gt": result.preproc_to_gt,
        "gt_to_preproc": result.gt_to_preproc,
        "pair_confidences": result.pair_confidences,
        "confidence_min": result.confidence_min,
        "confidence_mean": result.confidence_mean,
        "n_frames_used": result.n_frames_used,
        "used_manual_override": result.used_manual_override,
        "method": result.method,
        "similarity_matrix": result.similarity_matrix,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _materialize_scene(
    cfg: Config,
    scene: Scene,
    preproc_to_gt: Optional[Sequence[int]] = None,
) -> Path:
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
        if preproc_to_gt is None:
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
        else:
            numeric_track_dirs = {
                int(p.name): p for p in track_dirs if p.name.isdigit()
            }
            if set(numeric_track_dirs.keys()) != set(int(v) for v in preproc_to_gt):
                raise RuntimeError(
                    f"GT numeric track ids in {src_mask_root} do not match identity mapping. "
                    f"tracks={sorted(numeric_track_dirs.keys())}, mapping={list(preproc_to_gt)}"
                )

            all_track_dir = src_mask_root / "all"
            if all_track_dir.exists():
                all_map = _numeric_stem_to_file(all_track_dir, {".png", ".jpg", ".jpeg"})
                copied_all = _copy_frames_with_map(
                    all_map,
                    frame_old_to_new,
                    dst_scene_dir / "seg" / "img_seg_mask" / str(cam_id) / "all",
                    cfg.frame_name_num_digits,
                )
                if copied_all != len(frame_old_to_new):
                    raise RuntimeError(
                        f"Copied {copied_all}/{len(frame_old_to_new)} masks for cam {cam_id} track 'all'."
                    )

            for preproc_pid, gt_pid in enumerate(preproc_to_gt):
                src_track_map = _numeric_stem_to_file(
                    numeric_track_dirs[int(gt_pid)], {".png", ".jpg", ".jpeg"}
                )
                copied_masks = _copy_frames_with_map(
                    src_track_map,
                    frame_old_to_new,
                    dst_scene_dir / "seg" / "img_seg_mask" / str(cam_id) / str(preproc_pid),
                    cfg.frame_name_num_digits,
                )
                if copied_masks != len(frame_old_to_new):
                    raise RuntimeError(
                        f"Copied {copied_masks}/{len(frame_old_to_new)} masks for cam {cam_id} "
                        f"track '{preproc_pid}' (from GT '{gt_pid}')."
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
    copied_smpl = _copy_smpl_with_person_map(
        smpl_map,
        frame_old_to_new,
        dst_scene_dir / "smpl",
        cfg.frame_name_num_digits,
        preproc_to_gt=preproc_to_gt,
    )
    if copied_smpl != len(frame_old_to_new):
        raise RuntimeError(f"Copied {copied_smpl}/{len(frame_old_to_new)} SMPL files.")

    # Optional metadata.
    if cfg.include_meta and (src_scene_dir / "meta.npz").exists():
        _copy_meta_with_person_map(
            src_scene_dir / "meta.npz",
            dst_scene_dir / "meta.npz",
            preproc_to_gt=preproc_to_gt,
        )

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
    preproc_scene_dir = _resolve_preprocessed_scene_dir(cfg, scene)
    if cfg.require_preprocessed_scene_dir and not preproc_scene_dir.exists():
        raise FileNotFoundError(
            f"Missing preprocessing scene directory for '{scene.seq_name}': {preproc_scene_dir}"
        )
    print(f"Materializing {scene.seq_name}")
    print(f"  raw_gt_dir_path={scene.raw_gt_dir_path}")
    print(f"  preprocessing_scene_dir={preproc_scene_dir}")
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
        identity_result: Optional[IdentityMatchResult] = None
        identity_debug_context: Optional[IdentityAlignmentDebugContext] = None
        if cfg.identity_alignment_enabled:
            src_scene_dir = Path(scene.raw_gt_dir_path)
            identity_result, identity_debug_context = _compute_person_id_alignment(
                cfg=cfg,
                scene=scene,
                src_scene_dir=src_scene_dir,
                preproc_scene_dir=preproc_scene_dir,
            )
            if (
                not identity_result.used_manual_override
                and identity_result.confidence_min < float(cfg.identity_alignment_min_confidence)
            ):
                debug_path = _write_identity_alignment_debug(
                    scene=scene,
                    misc_dir=dst_scene_dir / "misc",
                    result=identity_result,
                    debug_context=identity_debug_context,
                )
                raise RuntimeError(
                    "Automatic identity matching confidence below threshold. "
                    f"confidence_min={identity_result.confidence_min:.4f} < "
                    f"min_confidence={float(cfg.identity_alignment_min_confidence):.4f}. "
                    f"Suggested manual_preproc_to_gt={identity_result.preproc_to_gt}. "
                    f"Debug saved to {debug_path}."
                )
            print(
                "Identity alignment resolved: "
                f"preproc_to_gt={identity_result.preproc_to_gt}, "
                f"confidence_min={identity_result.confidence_min:.4f}, "
                f"method={identity_result.method}"
            )

        materialized_scene_dir = _materialize_scene(
            cfg,
            scene,
            preproc_to_gt=identity_result.preproc_to_gt if identity_result is not None else None,
        )
        if identity_result is not None:
            _write_person_id_map(materialized_scene_dir, identity_result)
            if identity_debug_context is not None:
                _write_identity_alignment_debug(
                    scene=scene,
                    misc_dir=materialized_scene_dir / "misc",
                    result=identity_result,
                    debug_context=identity_debug_context,
                )
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


def _run_scene_identity_alignment_check(cfg: Config, scene: Scene) -> None:
    preproc_scene_dir = _resolve_preprocessed_scene_dir(cfg, scene)
    if cfg.require_preprocessed_scene_dir and not preproc_scene_dir.exists():
        raise FileNotFoundError(
            f"Missing preprocessing scene directory for '{scene.seq_name}': {preproc_scene_dir}"
        )

    print(f"Identity check: {scene.seq_name}")
    print(f"  raw_gt_dir_path={scene.raw_gt_dir_path}")
    print(f"  preprocessing_scene_dir={preproc_scene_dir}")

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

    if not cfg.identity_alignment_enabled:
        raise ValueError(
            "identity_alignment_only requires identity_alignment_enabled=true."
        )

    result, debug_context = _compute_person_id_alignment(
        cfg=cfg,
        scene=scene,
        src_scene_dir=Path(scene.raw_gt_dir_path),
        preproc_scene_dir=preproc_scene_dir,
    )
    debug_path = _write_identity_alignment_debug(
        scene=scene,
        misc_dir=(cfg.output_root_dir / scene.seq_name / "misc"),
        result=result,
        debug_context=debug_context,
    )
    if (
        not result.used_manual_override
        and result.confidence_min < float(cfg.identity_alignment_min_confidence)
    ):
        raise RuntimeError(
            "Automatic identity matching confidence below threshold. "
            f"confidence_min={result.confidence_min:.4f} < "
            f"min_confidence={float(cfg.identity_alignment_min_confidence):.4f}. "
            f"Suggested manual_preproc_to_gt={result.preproc_to_gt}. "
            f"Debug saved to {debug_path}."
        )
    print(
        "Identity alignment check passed: "
        f"preproc_to_gt={result.preproc_to_gt}, "
        f"confidence_min={result.confidence_min:.4f}, "
        f"confidence_mean={result.confidence_mean:.4f}, "
        f"n_frames_used={result.n_frames_used}, "
        f"method={result.method}, "
        f"debug={debug_path}"
    )
    print("Similarity matrix (rows=preproc ids, cols=gt ids):")
    for row in result.similarity_matrix:
        print("  " + " ".join(f"{v:.4f}" for v in row))


def main() -> None:
    cfg = tyro.cli(Config)
    runtime_paths = load_runtime_paths(_resolve_repo_path(cfg.repo_dir, cfg.paths_config))
    if cfg.output_root_dir is None:
        cfg.output_root_dir = runtime_paths.canonical_gt_root_dir
    if cfg.preprocessing_root_dir is None:
        cfg.preprocessing_root_dir = runtime_paths.preprocessing_root_dir
    assert cfg.output_root_dir is not None
    scenes = _filter_scenes(cfg, _load_scenes(cfg))

    if cfg.submit:
        _submit_array(cfg, scenes)
        return

    scene_index = _resolve_scene_index()
    run_one_scene = _run_scene_identity_alignment_check if cfg.identity_alignment_only else _run_scene
    if cfg.run_all:
        for scene in scenes:
            run_one_scene(cfg, scene)
        return

    if scene_index is None:
        if len(scenes) == 1:
            run_one_scene(cfg, scenes[0])
            return
        if not scenes:
            print("No scenes matched the filter.")
            sys.exit(1)
        print("Multiple scenes matched. Use --run-all or --submit.")
        sys.exit(1)

    if scene_index < 0 or scene_index >= len(scenes):
        raise IndexError(f"Scene index {scene_index} out of range (0..{len(scenes)-1}).")
    run_one_scene(cfg, scenes[scene_index])


if __name__ == "__main__":
    main()
