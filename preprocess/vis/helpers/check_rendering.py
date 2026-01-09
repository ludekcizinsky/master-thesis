from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import pyrender
import torch
from tqdm import tqdm
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from submodules.smplx import smplx


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MASK_EXT = ".png"

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


@dataclass
class Config:
    scene_dir: Path
    camera_id: int
    device: str = "cuda"
    model_folder: Path = Path("/home/cizinsky/body_models")
    smplx_model_ext: str = "npz"
    gender: str = "neutral"
    mesh_alpha: float = 1.0


def _parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    parser = argparse.ArgumentParser(description="Render SMPL-X overlays for sanity checks.")
    parser.add_argument("--scenes-dir", "--scene-dir", dest="scene_dir", required=True, type=Path)
    parser.add_argument("--camera_id", "--camera-id", dest="camera_id", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-folder", dest="model_folder", default="/home/cizinsky/body_models")
    parser.add_argument("--smplx-model-ext", dest="smplx_model_ext", default="npz")
    parser.add_argument("--gender", default="neutral")
    parser.add_argument("--mesh-alpha", dest="mesh_alpha", type=float, default=1.0)
    args = parser.parse_args(argv)
    return Config(
        scene_dir=args.scene_dir,
        camera_id=args.camera_id,
        device=args.device,
        model_folder=Path(args.model_folder),
        smplx_model_ext=args.smplx_model_ext,
        gender=args.gender,
        mesh_alpha=args.mesh_alpha,
    )


def _sorted_images(images_dir: Path) -> List[Path]:
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    def sort_key(p: Path) -> Tuple[int, str]:
        return (int(p.stem), p.name) if p.stem.isdigit() else (1_000_000_000, p.name)
    return sorted(images, key=sort_key)


def _load_skip_frames(scene_dir: Path) -> set[int]:
    skip_path = scene_dir / "skip_frames.csv"
    if not skip_path.exists():
        return set()
    content = skip_path.read_text(encoding="utf-8").strip()
    if not content:
        return set()
    return {int(x.strip()) for x in content.split(",") if x.strip().isdigit()}


def _load_camera_params(camera_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(camera_path, allow_pickle=True) as data:
        intrinsics = np.asarray(data["intrinsics"])
        extrinsics = np.asarray(data["extrinsics"])

    if intrinsics.shape == (3, 3):
        intrinsics = intrinsics[None, ...]
    if extrinsics.shape == (3, 4):
        extrinsics = extrinsics[None, ...]
    if intrinsics.shape != (1, 3, 3):
        raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")
    if extrinsics.shape != (1, 3, 4):
        raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")
    return intrinsics[0], extrinsics[0]


def _load_smplx_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=True)
    if not npz.files:
        return None
    return {k: np.asarray(npz[k]) for k in npz.files}


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


def _extract_person_params(smplx_data: Dict[str, np.ndarray], pid: int) -> Optional[Dict[str, np.ndarray]]:
    required = [
        "betas",
        "root_pose",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "lhand_pose",
        "rhand_pose",
        "trans",
    ]
    if any(k not in smplx_data for k in required):
        return None
    try:
        params = {k: smplx_data[k][pid] for k in required}
    except IndexError:
        return None
    if "expression" in smplx_data:
        params["expression"] = smplx_data["expression"][pid]
    return params


def _render_overlay(
    image: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    smplx_data: Dict[str, np.ndarray],
    smplx_layer,
    mesh_alpha: float,
    device: torch.device,
    renderer: pyrender.OffscreenRenderer,
) -> np.ndarray:
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    R_w2c = extrinsics[:3, :3]
    t_w2c = extrinsics[:3, 3]

    faces = np.asarray(smplx_layer.faces, dtype=np.int64)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.4, 0.4, 0.4))

    num_people = int(smplx_data["trans"].shape[0])
    expr_dim = int(getattr(smplx_layer, "num_expression_coeffs", 0))
    expected_betas = int(getattr(smplx_layer, "num_betas", smplx_data["betas"].shape[-1]))

    for pid in range(num_people):
        params_np = _extract_person_params(smplx_data, pid)
        if params_np is None:
            continue
        params = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in params_np.items()}

        params["betas"] = _pad_or_truncate(params["betas"].view(1, -1), expected_betas)
        call_args = dict(
            global_orient=params["root_pose"].view(1, 3),
            body_pose=params["body_pose"].view(1, 21, 3),
            jaw_pose=params["jaw_pose"].view(1, 3),
            leye_pose=params["leye_pose"].view(1, 3),
            reye_pose=params["reye_pose"].view(1, 3),
            left_hand_pose=params["lhand_pose"].view(1, 15, 3),
            right_hand_pose=params["rhand_pose"].view(1, 15, 3),
            betas=params["betas"],
            transl=params["trans"].view(1, 3),
        )
        if expr_dim > 0:
            expr = params.get("expression")
            if expr is None:
                expr = torch.zeros((1, expr_dim), device=device, dtype=params["betas"].dtype)
            else:
                expr = _pad_or_truncate(expr.view(1, -1), expr_dim)
            call_args["expression"] = expr

        with torch.no_grad():
            output = smplx_layer(**call_args)
        verts = output.vertices[0].detach().cpu().numpy()
        verts = (R_w2c @ verts.T).T + t_w2c.reshape(1, 3)
        verts[:, 1] *= -1.0
        verts[:, 2] *= -1.0

        color = _color_for_person(pid)
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
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cfg = _parse_args()
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")

    images_dir = cfg.scene_dir / "images" / str(cfg.camera_id)
    masks_dir = cfg.scene_dir / "seg" / "img_seg_mask" / str(cfg.camera_id) / "all"
    cameras_dir = cfg.scene_dir / "all_cameras" / str(cfg.camera_id)
    smplx_dir = cfg.scene_dir / "smplx"
    output_dir = cfg.scene_dir / "quality_checks" / "rendering"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks dir not found: {masks_dir}")
    if not cameras_dir.exists():
        raise FileNotFoundError(f"Cameras dir not found: {cameras_dir}")
    if not smplx_dir.exists():
        raise FileNotFoundError(f"SMPLX dir not found: {smplx_dir}")

    frame_paths = _sorted_images(images_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No images found under {images_dir}")

    skip_frames = _load_skip_frames(cfg.scene_dir)

    smplx_layer = _build_smplx_layer(cfg, device)

    sample_image = np.array(Image.open(frame_paths[0]).convert("RGB"))
    H, W = sample_image.shape[:2]
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

    try:
        for frame_path in tqdm(frame_paths, desc="Rendering"):
            stem = frame_path.stem
            if not stem.isdigit():
                continue
            frame_idx = int(stem)
            if frame_idx in skip_frames:
                continue

            mask_path = masks_dir / f"{stem}{MASK_EXT}"
            camera_path = cameras_dir / f"{stem}.npz"
            smplx_path = smplx_dir / f"{stem}.npz"
            if not (mask_path.exists() and camera_path.exists() and smplx_path.exists()):
                continue

            image = np.array(Image.open(frame_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))
            if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
                raise ValueError(f"Mask size mismatch for frame {stem}")

            if image.shape[0] != H or image.shape[1] != W:
                renderer.delete()
                H, W = image.shape[:2]
                renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

            mask_bool = mask > 0
            masked = image.copy()
            masked[~mask_bool] = 0

            intrinsics, extrinsics = _load_camera_params(camera_path)
            smplx_data = _load_smplx_npz(smplx_path)
            if smplx_data is None:
                continue
            if "trans" not in smplx_data or smplx_data["trans"].shape[0] == 0:
                continue

            overlay = _render_overlay(
                image=masked,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                smplx_data=smplx_data,
                smplx_layer=smplx_layer,
                mesh_alpha=cfg.mesh_alpha,
                device=device,
                renderer=renderer,
            )

            out_path = output_dir / f"{stem}.png"
            Image.fromarray(overlay).save(out_path)
    finally:
        renderer.delete()

    print(f"Rendering check images saved to: {output_dir}")

if __name__ == "__main__":
    main()
