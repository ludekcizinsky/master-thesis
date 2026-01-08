from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import pyrender
import torch
from tqdm import tqdm
import trimesh
import tyro

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from submodules.smplx import smplx


@dataclass
class Config:
    scene_dir: Path
    frames_dir: str = "frames"
    human3r_dir: str = "motion_human3r"
    smplx_params_dir: str = "smplx_params"
    cameras_filename: str = "cameras.npz"
    output_subdir: str = "debug"
    model_folder: Path = Path("/home/cizinsky/body_models")
    smplx_model_ext: str = "npz"
    gender: str = "neutral"
    device: str = "cuda"
    mesh_alpha: float = 1.0


_PALETTE = np.array(
    [
        [255, 80, 80],
        [80, 200, 255],
        [255, 200, 80],
        [160, 255, 120],
        [200, 120, 255],
        [255, 140, 200],
        [120, 255, 220],
        [255, 120, 120],
        [120, 160, 255],
        [255, 220, 140],
    ],
    dtype=np.float32,
)


def _sorted_images(frames_dir: Path) -> List[Path]:
    images = [
        p
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return sorted(images, key=lambda p: p.name)


def _sorted_jsons(params_dir: Path) -> List[Path]:
    files = [p for p in params_dir.iterdir() if p.is_file() and p.suffix == ".json"]
    return sorted(files, key=lambda p: p.name)


def _parse_smplx_indices(track_name: str, jsons: List[Path]) -> List[int]:
    stems = [p.stem for p in jsons]
    non_numeric = [s for s in stems if not s.isdigit()]
    if non_numeric:
        raise ValueError(f"Non-numeric smplx frame names for {track_name}: {non_numeric}")
    return sorted(int(s) for s in stems)


def _fix_missing_smplx_frames(
    track_name: str,
    params_dir: Path,
    jsons: List[Path],
    expected_count: int,
    width: int = 5,
) -> List[int]:
    if len(jsons) == expected_count:
        return []

    indices = _parse_smplx_indices(track_name, jsons)
    present = set(indices)
    extra = sorted(idx for idx in present if idx < 0 or idx >= expected_count)
    if extra:
        extra_names = [f"{idx:0{width}d}.json" for idx in extra]
        raise AssertionError(
            f"Extra smplx frames for {track_name} outside expected range: {extra_names}"
        )

    missing = sorted(idx for idx in range(expected_count) if idx not in present)
    for idx in missing:
        missing_path = params_dir / f"{idx:0{width}d}.json"
        if not missing_path.exists():
            missing_path.write_text("{}", encoding="utf-8")
    return missing


def _load_intrinsics(camera_path: Path) -> np.ndarray:
    with np.load(camera_path) as cams:
        if "K" not in cams.files:
            raise KeyError(f"Missing K in {camera_path}")
        return np.asarray(cams["K"])


def _load_smplx_params(json_path: Path) -> Optional[Dict[str, torch.Tensor]]:
    if json_path.stat().st_size == 0:
        return None
    with json_path.open("r") as f:
        data = json.load(f)
    required = {
        "betas",
        "root_pose",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "lhand_pose",
        "rhand_pose",
        "trans",
    }
    if not required.issubset(data.keys()):
        return None
    to_tensor = lambda v: torch.tensor(v, dtype=torch.float32)

    params = {
        "betas": to_tensor(data["betas"]).view(1, -1),
        "root_pose": to_tensor(data["root_pose"]).view(1, 3),
        "body_pose": to_tensor(data["body_pose"]).view(1, 21, 3),
        "jaw_pose": to_tensor(data["jaw_pose"]).view(1, 3),
        "leye_pose": to_tensor(data["leye_pose"]).view(1, 3),
        "reye_pose": to_tensor(data["reye_pose"]).view(1, 3),
        "lhand_pose": to_tensor(data["lhand_pose"]).view(1, 15, 3),
        "rhand_pose": to_tensor(data["rhand_pose"]).view(1, 15, 3),
        "trans": to_tensor(data["trans"]).view(1, 3),
    }
    return params


def _color_for_person(pid: int) -> Tuple[float, float, float]:
    color = _PALETTE[pid % len(_PALETTE)] / 255.0
    return float(color[0]), float(color[1]), float(color[2])


def _pad_or_truncate(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_dim = int(vec.shape[-1])
    if current_dim == target_dim:
        return vec
    if current_dim > target_dim:
        return vec[..., :target_dim]
    pad = torch.zeros((*vec.shape[:-1], target_dim - current_dim), device=vec.device, dtype=vec.dtype)
    return torch.cat([vec, pad], dim=-1)


def _build_smplx_layer(cfg: Config, device: torch.device):
    layer = smplx.create(
        str(cfg.model_folder),
        model_type="smplx",
        gender=cfg.gender,
        ext=cfg.smplx_model_ext,
        use_pca=False,
        use_face_contour=True,
        flat_hand_mean=True,
    )
    return layer.to(device)


def _render_frame(
    image: np.ndarray,
    intrinsics: np.ndarray,
    people_params: List[Tuple[Dict[str, torch.Tensor], Tuple[float, float, float]]],
    smplx_layer,
    mesh_alpha: float,
    device: torch.device,
    renderer: pyrender.OffscreenRenderer,
) -> np.ndarray:
    H, W = image.shape[:2]
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    faces = np.asarray(smplx_layer.faces, dtype=np.int64)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.4, 0.4, 0.4))

    for params, color in people_params:
        params = {k: v.to(device) for k, v in params.items()}
        expected_betas = int(getattr(smplx_layer, "num_betas", params["betas"].shape[-1]))
        params["betas"] = _pad_or_truncate(params["betas"], expected_betas)

        expr_dim = int(getattr(smplx_layer, "num_expression_coeffs", 0))
        expr = None
        if expr_dim > 0:
            expr = torch.zeros((1, expr_dim), device=device, dtype=params["betas"].dtype)

        call_args = dict(
            global_orient=params["root_pose"],
            body_pose=params["body_pose"],
            jaw_pose=params["jaw_pose"],
            leye_pose=params["leye_pose"],
            reye_pose=params["reye_pose"],
            left_hand_pose=params["lhand_pose"],
            right_hand_pose=params["rhand_pose"],
            betas=params["betas"],
            transl=params["trans"],
        )
        if expr is not None:
            call_args["expression"] = expr

        with torch.no_grad():
            output = smplx_layer(**call_args)
        verts = output.vertices[0].detach().cpu().numpy()
        verts[:, 1] *= -1.0
        verts[:, 2] *= -1.0

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.8,
            alphaMode="OPAQUE",
            baseColorFactor=(color[0], color[1], color[2], float(mesh_alpha)),
        )
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e4)
    scene.add(camera, pose=np.eye(4))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=np.eye(4))

    color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color_rgb = color_rgba[..., :3].astype(np.float32)
    alpha = color_rgba[..., 3:4].astype(np.float32) / 255.0

    base = image.astype(np.float32)
    blended = color_rgb * alpha + base * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def main() -> None:
    cfg = tyro.cli(Config)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")

    scene_dir = cfg.scene_dir
    frames_dir = scene_dir / cfg.frames_dir
    human3r_dir = scene_dir / cfg.human3r_dir
    cameras_path = human3r_dir / cfg.cameras_filename

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
    if not human3r_dir.exists():
        raise FileNotFoundError(f"Human3R dir not found: {human3r_dir}")
    if not cameras_path.exists():
        raise FileNotFoundError(f"Cameras file not found: {cameras_path}")

    frame_paths = _sorted_images(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found under {frames_dir}")

    track_dirs = [
        p for p in human3r_dir.iterdir() if p.is_dir() and (p / cfg.smplx_params_dir).is_dir()
    ]
    track_dirs = sorted(track_dirs, key=lambda p: p.name)
    if not track_dirs:
        raise FileNotFoundError(f"No track dirs with smplx params found under {human3r_dir}")

    track_frames: Dict[str, List[Path]] = {}
    missing_indices_all: List[int] = []
    for track_dir in track_dirs:
        params_dir = track_dir / cfg.smplx_params_dir
        jsons = _sorted_jsons(params_dir)
        if len(jsons) != len(frame_paths):
            missing = _fix_missing_smplx_frames(
                track_dir.name, params_dir, jsons, len(frame_paths)
            )
            missing_indices_all.extend(missing)
            jsons = _sorted_jsons(params_dir)
        if len(jsons) != len(frame_paths):
            raise AssertionError(
                f"Frame count mismatch for {track_dir.name}: "
                f"{len(jsons)} smplx frames vs {len(frame_paths)} image frames"
            )
        track_frames[track_dir.name] = jsons

    skip_frame_numbers: List[int] = []
    if missing_indices_all:
        for idx in sorted(set(missing_indices_all)):
            frame_stem = frame_paths[idx].stem
            if not frame_stem.isdigit():
                raise ValueError(f"Non-numeric frame name: {frame_paths[idx].name}")
            skip_frame_numbers.append(int(frame_stem))
        skip_path = scene_dir / "skip_frames.csv"
        skip_path.write_text(",".join(str(x) for x in skip_frame_numbers), encoding="utf-8")

    Ks = _load_intrinsics(cameras_path)
    if Ks.shape[0] != len(frame_paths):
        raise AssertionError(
            f"Camera count mismatch: {Ks.shape[0]} cameras vs {len(frame_paths)} image frames"
        )

    output_dir = human3r_dir / cfg.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    smplx_layer = _build_smplx_layer(cfg, device)

    sample_image = np.array(Image.open(frame_paths[0]).convert("RGB"))
    H, W = sample_image.shape[:2]
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

    try:
        track_names = [track_dir.name for track_dir in track_dirs]
        for idx, frame_path in enumerate(tqdm(frame_paths, desc="Rendering")):
            image = np.array(Image.open(frame_path).convert("RGB"))
            if image.shape[0] != H or image.shape[1] != W:
                renderer.delete()
                H, W = image.shape[:2]
                renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

            people_params = []
            for track_idx, track_name in enumerate(track_names):
                json_path = track_frames[track_name][idx]
                params = _load_smplx_params(json_path)
                if params is None:
                    continue
                people_params.append((params, _color_for_person(track_idx)))

            overlay = _render_frame(
                image=image,
                intrinsics=Ks[idx],
                people_params=people_params,
                smplx_layer=smplx_layer,
                mesh_alpha=cfg.mesh_alpha,
                device=device,
                renderer=renderer,
            )

            out_path = output_dir / f"{frame_path.stem}.png"
            Image.fromarray(overlay).save(out_path)
    finally:
        renderer.delete()

    if skip_frame_numbers:
        print(f"Skipped frames ({len(skip_frame_numbers)}): {skip_frame_numbers}")
    else:
        print("Skipped frames: none")


if __name__ == "__main__":
    main()
