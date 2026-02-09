from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pytorch3d.transforms import quaternion_to_matrix
import torch
from tqdm import tqdm
import tyro
import viser


def _sorted_files(root: Path, pattern: str) -> List[Path]:
    files = list(root.glob(pattern))

    def _key(path: Path) -> Tuple[int, str]:
        stem = path.stem
        if stem.isdigit():
            return (0, f"{int(stem):012d}")
        return (1, stem)

    return sorted(files, key=_key)


def _sorted_person_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    dirs = [path for path in root.iterdir() if path.is_dir()]

    def _key(path: Path) -> Tuple[int, str]:
        name = path.name
        if name.isdigit():
            return (0, f"{int(name):012d}")
        return (1, name)

    return sorted(dirs, key=_key)


def _find_rgb_image(scene_dir: Path, src_cam_id: int, frame_stem: str) -> Optional[Path]:
    image_dir = scene_dir / "images" / str(src_cam_id)
    for ext in (".jpg", ".jpeg", ".png"):
        path = image_dir / f"{frame_stem}{ext}"
        if path.exists():
            return path
    return None


def _load_rgb_image(scene_dir: Path, src_cam_id: int, frame_stem: str) -> np.ndarray:
    path = _find_rgb_image(scene_dir, src_cam_id, frame_stem)
    if path is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _downscale_rgb(rgb: np.ndarray, *, max_side: int) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if max(h, w) <= int(max_side):
        return rgb
    scale = float(max_side) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return np.asarray(
        Image.fromarray(rgb).resize((new_w, new_h), resample=Image.BILINEAR), dtype=np.uint8
    )


def _set_gui_image(handle: object, image: np.ndarray) -> None:
    if hasattr(handle, "image"):
        handle.image = image
    elif hasattr(handle, "value"):
        handle.value = image


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_to_splat_arrays(
    state: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    xyz = state["xyz"].float()
    opacity = state["opacity"].float()
    rotation = state["rotation"].float()
    scaling = state["scaling"].float()
    shs = state["shs"].float()

    opacity = opacity.squeeze(-1).clamp(0.0, 1.0).unsqueeze(-1)
    rotation = torch.nn.functional.normalize(rotation, dim=-1)
    scales = scaling.clamp(min=1e-8)
    rgbs = shs.squeeze(1).clamp(0.0, 1.0)

    rotation_mats = quaternion_to_matrix(rotation)
    covariances = rotation_mats @ torch.diag_embed(scales**2) @ rotation_mats.transpose(-1, -2)

    return {
        "centers": xyz.detach().cpu().numpy().astype(np.float32),
        "opacities": opacity.detach().cpu().numpy().astype(np.float32),
        "rgbs": rgbs.detach().cpu().numpy().astype(np.float32),
        "covariances": covariances.detach().cpu().numpy().astype(np.float32),
    }


def _frame_stem_key(stem: str) -> Tuple[int, str]:
    if stem.isdigit():
        return (0, f"{int(stem):012d}")
    return (1, stem)


def _common_frame_stems(person_dirs: List[Path]) -> List[str]:
    common: Optional[set[str]] = None
    for person_dir in person_dirs:
        stems = {path.stem for path in _sorted_files(person_dir, "*.pt")}
        if not stems:
            continue
        if common is None:
            common = set(stems)
        else:
            common &= stems
    if common is None:
        return []
    return sorted(common, key=_frame_stem_key)


@dataclass
class Args:
    eval_scene_dir: Path
    frame_idx_range: Tuple[int, int] = (0, 20)
    subsample_rate: int = 5
    src_cam_id: int = 4


def main() -> None:
    args = tyro.cli(Args)

    posed_3dgs_dir = args.eval_scene_dir / "posed_3dgs_per_frame"
    if not posed_3dgs_dir.exists():
        raise FileNotFoundError(f"Missing directory: {posed_3dgs_dir}")

    person_dirs = _sorted_person_dirs(posed_3dgs_dir)
    if not person_dirs:
        raise FileNotFoundError(f"No person subdirectories found in {posed_3dgs_dir}")

    all_frame_stems = _common_frame_stems(person_dirs)
    if not all_frame_stems:
        raise FileNotFoundError("No common frame files found across 3DGS person folders.")

    start_idx, end_idx = int(args.frame_idx_range[0]), int(args.frame_idx_range[1])
    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError("frame_idx_range must be valid: start >= 0 and end > start.")
    if start_idx >= len(all_frame_stems):
        raise ValueError(
            f"frame_idx_range start {start_idx} out of bounds for {len(all_frame_stems)} frames."
        )
    end_idx = min(end_idx, len(all_frame_stems))

    subsample_rate = max(int(args.subsample_rate), 1)
    frame_stems = all_frame_stems[start_idx:end_idx:subsample_rate]
    if not frame_stems:
        raise FileNotFoundError("No frames selected after applying frame_idx_range and subsample_rate.")

    server = viser.ViserServer(port=8080)
    server.scene.set_up_direction("+y")
    server.scene.add_frame("/scene", show_axes=False)

    rgb_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
    rgb_cache_size = 32
    rgb_max_side = 1280

    def _get_rgb(frame_stem: str) -> np.ndarray:
        cached = rgb_cache.get(frame_stem)
        if cached is not None:
            rgb_cache.move_to_end(frame_stem)
            return cached
        rgb = _load_rgb_image(args.eval_scene_dir, args.src_cam_id, frame_stem)
        rgb = _downscale_rgb(rgb, max_side=rgb_max_side)
        rgb_cache[frame_stem] = rgb
        if len(rgb_cache) > rgb_cache_size:
            rgb_cache.popitem(last=False)
        return rgb

    with server.gui.add_folder("Frame"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=len(frame_stems) - 1,
            step=1,
            initial_value=0,
        )
        frame_name_text = server.gui.add_text("Current frame", frame_stems[0])

    initial_rgb = _get_rgb(frame_stems[0])
    with server.gui.add_folder("RGB"):
        rgb_handle = server.gui.add_image(initial_rgb, label=f"Cam {args.src_cam_id}")

    frame_nodes: List[viser.FrameHandle] = []
    for frame_stem in tqdm(frame_stems, desc="Loading 3DGS frames", total=len(frame_stems)):
        frame_node = server.scene.add_frame(f"/scene/3dgs/f_{frame_stem}", show_axes=False)
        frame_nodes.append(frame_node)

        centers_all: List[np.ndarray] = []
        rgbs_all: List[np.ndarray] = []
        opacities_all: List[np.ndarray] = []
        covariances_all: List[np.ndarray] = []

        for person_dir in person_dirs:
            state_path = person_dir / f"{frame_stem}.pt"
            if not state_path.exists():
                continue
            state = _torch_load(state_path)
            splat = _state_to_splat_arrays(state)
            if splat["centers"].size != 0:
                centers_all.append(splat["centers"])
                rgbs_all.append(splat["rgbs"])
                opacities_all.append(splat["opacities"])
                covariances_all.append(splat["covariances"])

        if centers_all:
            server.scene.add_gaussian_splats(
                f"/scene/3dgs/f_{frame_stem}/splats",
                centers=np.concatenate(centers_all, axis=0),
                rgbs=np.concatenate(rgbs_all, axis=0),
                opacities=np.concatenate(opacities_all, axis=0),
                covariances=np.concatenate(covariances_all, axis=0),
            )
        frame_node.visible = False

    current_idx = 0
    frame_nodes[current_idx].visible = True

    def _set_frame(frame_idx: int) -> None:
        nonlocal current_idx
        if frame_idx == current_idx:
            return
        frame_nodes[current_idx].visible = False
        current_idx = frame_idx
        frame_nodes[current_idx].visible = True
        frame_stem = frame_stems[current_idx]
        frame_name_text.value = frame_stem
        _set_gui_image(rgb_handle, _get_rgb(frame_stem))

    @frame_slider.on_update
    def _(_event) -> None:
        _set_frame(int(frame_slider.value))

    print("Viser server is running. Use slider to change frame (Ctrl+C to exit).")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main()
