from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class Args:
    scene_dir_path: Path
    scenes_output_dir: Path = REPO_ROOT / "preprocess" / "scenes"
    ref_frame_idx: int = 0
    overwrite: bool = True
    dry_run: bool = False


def _derive_pair_name(scene_dir: Path) -> str:
    parent_name = scene_dir.parent.name
    if re.fullmatch(r"pair\d+", parent_name):
        return parent_name

    suffix_match = re.search(r"(\d+)$", scene_dir.name)
    if suffix_match is None:
        raise ValueError(
            f"Could not infer pair id from parent '{parent_name}' or scene name '{scene_dir.name}'."
        )

    pair_id = int(suffix_match.group(1))
    return f"pair{pair_id:02d}"


def _derive_action_name(scene_dir: Path) -> str:
    action = re.sub(r"\d+$", "", scene_dir.name)
    if not action:
        raise ValueError(f"Could not infer action name from scene dir '{scene_dir.name}'.")
    return action


def _load_mono_cam(scene_dir: Path) -> int:
    meta_path = scene_dir / "meta.npz"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta.npz in scene dir: {scene_dir}")

    with np.load(meta_path, allow_pickle=True) as data:
        if "mono_cam" not in data:
            raise KeyError(f"Missing 'mono_cam' key in {meta_path}")
        mono_cam = int(data["mono_cam"])

    return mono_cam


def _build_scene_record(scene_dir: Path, ref_frame_idx: int) -> dict:
    pair_name = _derive_pair_name(scene_dir)
    action_name = _derive_action_name(scene_dir)
    seq_name = f"hi4d_{pair_name}_{action_name}"

    cam_id = _load_mono_cam(scene_dir)
    video_path = scene_dir / "images" / str(cam_id)

    if not video_path.is_dir():
        raise FileNotFoundError(
            f"Expected source frames directory does not exist: {video_path}"
        )

    return {
        "seq_name": seq_name,
        "cam_id": cam_id,
        "raw_gt_dir_path": str(scene_dir),
        "video_path": str(video_path),
        "ref_frame_idx": int(ref_frame_idx),
    }


def _run(args: Args) -> Path:
    scene_dir = args.scene_dir_path.resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")

    record = _build_scene_record(scene_dir, ref_frame_idx=args.ref_frame_idx)
    out_dir = args.scenes_output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record['seq_name']}.json"

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists and overwrite is disabled: {out_path}"
        )

    print("Scene JSON preview:")
    print(json.dumps(record, indent=2))
    print(f"Output path: {out_path}")

    if args.dry_run:
        print("Dry run enabled. No file written.")
        return out_path

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")

    print("Wrote scene JSON.")
    return out_path


def main() -> None:
    args = tyro.cli(Args)
    _run(args)


if __name__ == "__main__":
    main()
