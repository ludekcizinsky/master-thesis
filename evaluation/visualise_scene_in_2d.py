from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List, Tuple

import numpy as np
from PIL import Image
import pyrender
from tqdm import tqdm
import trimesh
import tyro


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
MESH_EXT = ".obj"
SMPLX_COLORS = [
    (0, 114, 189),  # blue
    (140, 58, 42),  # custom
    (217, 83, 25),  # orange
    (126, 47, 142),  # purple
    (119, 172, 48),  # green
    (77, 190, 238),  # cyan
    (255, 215, 0),  # yellow
    (0, 128, 128),  # teal
    (128, 0, 0),  # maroon
    (128, 128, 128),  # gray
]


@dataclass
class Config:
    exp_eval_dir: Annotated[Path, tyro.conf.arg(aliases=["--exp-eval-dir"])]
    cam_id: Annotated[int, tyro.conf.arg(aliases=["--cam-id"])]
    max_frames: int = 10
    mesh_alpha: float = 1.0


def _load_skip_frames(scene_dir: Path) -> set[str]:
    skip_path = scene_dir / "skip_frames.csv"
    if not skip_path.exists():
        return set()
    content = skip_path.read_text(encoding="utf-8").strip()
    if not content:
        return set()
    return {f"{int(x.strip()):06d}" for x in content.split(",") if x.strip().isdigit()}


def _sorted_meshes(mesh_dir: Path) -> List[Path]:
    meshes = [p for p in mesh_dir.iterdir() if p.suffix.lower() == MESH_EXT]

    def sort_key(p: Path) -> Tuple[int, str]:
        return (int(p.stem), p.name) if p.stem.isdigit() else (1_000_000_000, p.name)

    return sorted(meshes, key=sort_key)


def _sorted_person_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]

    def sort_key(p: Path) -> Tuple[int, str]:
        return (0, f"{int(p.name):06d}") if p.name.isdigit() else (1, p.name)

    return sorted(dirs, key=sort_key)


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_camera_npz(cameras_dir: Path, stem: str) -> Tuple[np.ndarray, np.ndarray]:
    cam_path = cameras_dir / f"{stem}.npz"
    if not cam_path.exists():
        raise FileNotFoundError(f"Camera file not found: {cam_path}")
    with np.load(cam_path) as data:
        intr = np.asarray(data["intrinsics"])[0]
        extr = np.asarray(data["extrinsics"])[0]
    return intr, extr


def _w2c_to_c2w_gl(w2c_cv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cv_to_gl = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    w2c_gl = cv_to_gl @ w2c_cv
    c2w_gl = np.linalg.inv(w2c_gl)
    c2w_gl[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return w2c_gl, c2w_gl


def _normals_to_colors(normals: np.ndarray) -> np.ndarray:
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms > 1e-8
    mask = valid.squeeze(-1)
    if np.any(mask):
        denom = norms[mask]
        normals[mask] /= denom
    normals[~mask] = np.array([0.0, 0.0, 1.0], dtype=normals.dtype)
    colors = ((normals + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    if colors.shape[1] == 3:
        alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
        colors = np.concatenate([colors, alpha], axis=1)
    return colors


def _render_normal_map(
    mesh: trimesh.Trimesh,
    renderer: pyrender.OffscreenRenderer,
    intr: np.ndarray,
    w2c_gl: np.ndarray,
    c2w_gl: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray] | None:
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        return None
    normals_world = mesh.vertex_normals
    normals_cam = (w2c_gl[:3, :3] @ normals_world.T).T
    mesh.visual.vertex_colors = _normals_to_colors(normals_cam)
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

    fx, fy, cx, cy = (
        float(intr[0, 0]),
        float(intr[1, 1]),
        float(intr[0, 2]),
        float(intr[1, 2]),
    )
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.4, 0.4, 0.4))
    scene.add(pr_mesh)
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e4)
    scene.add(camera, pose=c2w_gl)
    normal_rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    return normal_rgb[..., :3], depth


def _render_colored_meshes(
    meshes: List[trimesh.Trimesh],
    colors: List[Tuple[int, int, int]],
    renderer: pyrender.OffscreenRenderer,
    intr: np.ndarray,
    c2w_gl: np.ndarray,
    mesh_alpha: float,
) -> Tuple[np.ndarray, np.ndarray] | None:
    if not meshes:
        return None
    fx, fy, cx, cy = (
        float(intr[0, 0]),
        float(intr[1, 1]),
        float(intr[0, 2]),
        float(intr[1, 2]),
    )
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(1.0, 1.0, 1.0))
    added = False
    for mesh, color in zip(meshes, colors):
        if mesh.vertices.size == 0 or mesh.faces.size == 0:
            continue
        base_color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, float(mesh_alpha))
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.8,
            alphaMode="OPAQUE",
            baseColorFactor=base_color,
        )
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene.add(pr_mesh)
        added = True
    if not added:
        return None
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e4)
    scene.add(camera, pose=c2w_gl)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
        pose=c2w_gl,
    )
    color_rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    return color_rgba, depth


def _smplx_color(idx: int) -> Tuple[int, int, int]:
    return SMPLX_COLORS[idx % len(SMPLX_COLORS)]


def main() -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cfg = tyro.cli(Config)

    mesh_dir = cfg.exp_eval_dir / "posed_meshes_per_frame"
    if not mesh_dir.exists():
        print(f"Mesh directory not found: {mesh_dir}")
        sys.exit(1)

    images_dir = cfg.exp_eval_dir / "images" / f"{cfg.cam_id}"
    cameras_dir = cfg.exp_eval_dir / "all_cameras" / f"{cfg.cam_id}"
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        sys.exit(1)
    if not cameras_dir.exists():
        print(f"Camera directory not found: {cameras_dir}")
        sys.exit(1)

    out_dir = cfg.exp_eval_dir / "quality_checks" / "meshes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_individual = cfg.exp_eval_dir / "quality_checks" / "individual_meshes"
    out_dir_individual.mkdir(parents=True, exist_ok=True)
    out_dir_smplx = cfg.exp_eval_dir / "quality_checks" / "smplx_meshes"
    out_dir_smplx.mkdir(parents=True, exist_ok=True)

    posed_root = cfg.exp_eval_dir / "posed_meshes_per_frame"
    posed_dirs = _sorted_person_dirs(posed_root)
    person_names = {p.name for p in posed_dirs}
    if person_names:
        def _name_key(name: str) -> Tuple[int, str]:
            return (0, f"{int(name):06d}") if name.isdigit() else (1, name)
        person_ids = sorted(person_names, key=_name_key)
    else:
        person_ids = []
    for pid in person_ids:
        (out_dir_individual / pid).mkdir(parents=True, exist_ok=True)

    smplx_root = cfg.exp_eval_dir / "posed_smplx_meshes_per_frame"
    smplx_person_dirs = _sorted_person_dirs(smplx_root)

    skip_frames = _load_skip_frames(cfg.exp_eval_dir)
    mesh_paths = _sorted_meshes(mesh_dir)

    renderer = None
    current_hw = None

    processed = 0
    total = len(mesh_paths) if cfg.max_frames <= 0 else min(len(mesh_paths), cfg.max_frames)
    for mesh_path in tqdm(mesh_paths, desc="Rendering normals", total=total):
        if cfg.max_frames > 0 and processed >= cfg.max_frames:
            break
        stem = mesh_path.stem
        if stem in skip_frames:
            continue

        image_path = _find_image(images_dir, stem)
        if image_path is None:
            print(f"[WARN] Missing image for frame {stem}")
            continue

        with Image.open(image_path) as img:
            rgb = np.array(img.convert("RGB"))
        H, W = rgb.shape[0], rgb.shape[1]

        if current_hw != (H, W):
            if renderer is not None:
                renderer.delete()
            renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
            current_hw = (H, W)

        intr, extr = _load_camera_npz(cameras_dir, stem)

        w2c_cv = np.eye(4, dtype=np.float32)
        w2c_cv[:3, :4] = extr
        w2c_gl, c2w_gl = _w2c_to_c2w_gl(w2c_cv)

        mesh = trimesh.load(mesh_path, force="mesh")
        render_out = _render_normal_map(mesh, renderer, intr, w2c_gl, c2w_gl)
        if render_out is None:
            continue
        normal_rgb, depth = render_out
        normal_rgb = np.clip(normal_rgb.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
        mask = depth > 0
        if mask.any():
            out = rgb.copy()
            out[mask] = normal_rgb[mask]
            out_path = out_dir / f"{stem}.png"
            Image.fromarray(out).save(out_path)

        for pid in person_ids:
            posed_mesh_path = posed_root / pid / f"{stem}.obj"
            if posed_mesh_path.exists():
                person_mesh_path = posed_mesh_path
            else:
                print(f"[WARN] Missing individual mesh for pid={pid} frame={stem}")
                continue
            if person_mesh_path.stat().st_size == 0:
                print(f"[WARN] Empty individual mesh for pid={pid} frame={stem}: {person_mesh_path}")
                continue
            person_mesh = trimesh.load(person_mesh_path, force="mesh")
            person_render = _render_normal_map(person_mesh, renderer, intr, w2c_gl, c2w_gl)
            if person_render is None:
                continue
            person_normals, person_depth = person_render
            person_normals = np.clip(
                person_normals.astype(np.float32) * 1.25, 0, 255
            ).astype(np.uint8)
            person_mask = person_depth > 0
            if not person_mask.any():
                continue
            person_out = np.full_like(rgb, 255)
            person_out[person_mask] = person_normals[person_mask]
            person_out_path = out_dir_individual / pid / f"{stem}.png"
            Image.fromarray(person_out).save(person_out_path)

        smplx_meshes: List[trimesh.Trimesh] = []
        smplx_colors: List[Tuple[int, int, int]] = []
        for idx, pdir in enumerate(smplx_person_dirs):
            person_mesh_path = pdir / f"{stem}.obj"
            if not person_mesh_path.exists():
                continue
            if person_mesh_path.stat().st_size == 0:
                continue
            person_mesh = trimesh.load(person_mesh_path, force="mesh")
            if person_mesh.vertices.size == 0 or person_mesh.faces.size == 0:
                continue
            if (
                hasattr(person_mesh.visual, "vertex_colors")
                and person_mesh.visual.vertex_colors is not None
                and len(person_mesh.visual.vertex_colors) > 0
            ):
                person_mesh = trimesh.Trimesh(
                    vertices=person_mesh.vertices,
                    faces=person_mesh.faces,
                    process=False,
                )
            smplx_meshes.append(person_mesh)
            smplx_colors.append(_smplx_color(idx))

        smplx_out = rgb.copy()
        if smplx_meshes:
            smplx_render = _render_colored_meshes(
                smplx_meshes, smplx_colors, renderer, intr, c2w_gl, cfg.mesh_alpha
            )
            if smplx_render is not None:
                smplx_rgba, _ = smplx_render
                smplx_rgb = smplx_rgba[..., :3].astype(np.float32)
                smplx_alpha = smplx_rgba[..., 3:4].astype(np.float32) / 255.0
                base = smplx_out.astype(np.float32)
                smplx_out = np.clip(
                    smplx_rgb * smplx_alpha + base * (1.0 - smplx_alpha),
                    0,
                    255,
                ).astype(np.uint8)
        smplx_out_path = out_dir_smplx / f"{stem}.png"
        Image.fromarray(smplx_out).save(smplx_out_path)
        processed += 1

    if renderer is not None:
        renderer.delete()


if __name__ == "__main__":
    main()
