from __future__ import annotations

import inspect
import logging
import os
import re
import signal
import socket
import subprocess
import time
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
import torch
from tqdm import tqdm
import tyro

_CONSOLE = Console()
_LOGGER = logging.getLogger("rerun_dynamic_3dgs")


def _setup_logging() -> None:
    if _LOGGER.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=_CONSOLE, markup=True, rich_tracebacks=True)],
    )
    _LOGGER.setLevel(logging.INFO)


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


def _find_rgb_image(scene_dir: Path, src_cam_id: int, frame_stem: str) -> Optional[Path]:
    image_dir = scene_dir / "images" / str(src_cam_id)
    for ext in (".jpg", ".jpeg", ".png"):
        path = image_dir / f"{frame_stem}{ext}"
        if path.exists():
            return path
    return None


def _load_rgb_image(scene_dir: Path, src_cam_id: int, frame_stem: str) -> Optional[np.ndarray]:
    path = _find_rgb_image(scene_dir, src_cam_id, frame_stem)
    if path is None:
        return None
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


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_camera_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as cams:
        if "intrinsics" not in cams.files or "extrinsics" not in cams.files:
            raise KeyError(f"Missing intrinsics/extrinsics in {path}")
        intr = cams["intrinsics"]
        extr = cams["extrinsics"]
    if intr.ndim == 3:
        intr = intr[0]
    if extr.ndim == 3:
        extr = extr[0]
    if intr.shape != (3, 3) or extr.shape != (3, 4):
        raise ValueError(f"Unexpected camera shapes in {path}: {intr.shape}, {extr.shape}")
    return intr.astype(np.float32), extr.astype(np.float32)


def _infer_resolution(intr: np.ndarray, fallback_rgb: Optional[np.ndarray]) -> Tuple[int, int]:
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    width = int(round(cx * 2.0))
    height = int(round(cy * 2.0))
    if width > 0 and height > 0:
        return width, height
    if fallback_rgb is not None and fallback_rgb.ndim == 3:
        return int(fallback_rgb.shape[1]), int(fallback_rgb.shape[0])
    width = max(1, width)
    height = max(1, height)
    return width, height


def _scale_intrinsics(intr: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = intr.copy()
    out[0, 0] *= float(sx)
    out[1, 1] *= float(sy)
    out[0, 2] *= float(sx)
    out[1, 2] *= float(sy)
    return out


def _w2c_3x4_to_4x4(extr: np.ndarray) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :4] = extr
    return w2c


def _resolve_gsplat_rasterization(gs: Any) -> Any:
    if hasattr(gs, "rasterization"):
        return gs.rasterization
    if hasattr(gs, "rendering") and hasattr(gs.rendering, "rasterization"):
        return gs.rendering.rasterization
    return None


def _state_to_gsplat_arrays(
    state: Dict[str, Any],
    *,
    min_opacity: float,
    max_gaussians: Optional[int],
    seed: int,
    rotation_format: Literal["wxyz", "xyzw"],
) -> Dict[str, torch.Tensor]:
    xyz = state["xyz"].float()
    opacity = state["opacity"].float().squeeze(-1).clamp(0.0, 1.0)
    scaling = state["scaling"].float().clamp(min=1e-8)
    rotation = state["rotation"].float()
    shs = state["shs"].float()

    rgb = shs.squeeze(1).clamp(0.0, 1.0)

    keep = opacity >= float(min_opacity)
    xyz = xyz[keep]
    opacity = opacity[keep]
    scaling = scaling[keep]
    rotation = rotation[keep]
    rgb = rgb[keep]

    if max_gaussians is not None and xyz.shape[0] > int(max_gaussians):
        g = torch.Generator(device=xyz.device)
        g.manual_seed(int(seed))
        idx = torch.randperm(xyz.shape[0], generator=g)[: int(max_gaussians)]
        xyz = xyz[idx]
        opacity = opacity[idx]
        scaling = scaling[idx]
        rotation = rotation[idx]
        rgb = rgb[idx]

    quats = torch.nn.functional.normalize(rotation, dim=-1)
    if rotation_format == "xyzw":
        quats = quats[:, [3, 0, 1, 2]]

    return {
        "means": xyz,
        "scales": scaling,
        "quats": quats,
        "opacities": opacity,
        "colors": rgb,
    }


def _render_gsplat_image(
    *,
    rasterization: Any,
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    width: int,
    height: int,
    device: torch.device,
    packed: bool,
    rasterize_mode: Literal["classic", "antialiased"],
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    white_background: bool,
) -> np.ndarray:
    means = means.to(device=device, dtype=torch.float32, non_blocking=True)
    scales = scales.to(device=device, dtype=torch.float32, non_blocking=True)
    quats = quats.to(device=device, dtype=torch.float32, non_blocking=True)
    opacities = opacities.to(device=device, dtype=torch.float32, non_blocking=True)
    colors = colors.to(device=device, dtype=torch.float32, non_blocking=True)

    K = torch.from_numpy(intrinsics.astype(np.float32)).to(device)
    viewmat = torch.from_numpy(_w2c_3x4_to_4x4(extrinsics)).to(device)

    backgrounds = None
    if white_background:
        backgrounds = torch.ones((1, 3), dtype=torch.float32, device=device)

    with torch.no_grad():
        renders, _alphas, _meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None, ...],
            Ks=K[None, ...],
            width=int(width),
            height=int(height),
            packed=bool(packed),
            near_plane=float(near_plane),
            far_plane=float(far_plane),
            radius_clip=float(radius_clip),
            rasterize_mode=rasterize_mode,
            render_mode="RGB",
            backgrounds=backgrounds,
        )
        rgb = renders[0, :, :, :3].clamp(0.0, 1.0).detach().cpu().numpy()
    return (rgb * 255.0).astype(np.uint8)


def _state_to_point_arrays(
    state: Dict[str, Any],
    *,
    min_opacity: float,
    point_radius_scale: float,
    max_point_radius: Optional[float],
    max_gaussians: Optional[int],
    seed: int,
    color_by_opacity: bool,
) -> Dict[str, np.ndarray]:
    xyz = state["xyz"].float()
    opacity = state["opacity"].float().squeeze(-1).clamp(0.0, 1.0)
    scaling = state["scaling"].float().clamp(min=1e-8)
    shs = state["shs"].float()

    rgb = shs.squeeze(1).clamp(0.0, 1.0)

    keep = opacity >= float(min_opacity)
    xyz = xyz[keep]
    opacity = opacity[keep]
    scaling = scaling[keep]
    rgb = rgb[keep]

    if max_gaussians is not None and xyz.shape[0] > int(max_gaussians):
        g = torch.Generator(device=xyz.device)
        g.manual_seed(int(seed))
        idx = torch.randperm(xyz.shape[0], generator=g)[: int(max_gaussians)]
        xyz = xyz[idx]
        opacity = opacity[idx]
        scaling = scaling[idx]
        rgb = rgb[idx]

    radii = scaling.mean(dim=-1) * float(point_radius_scale)
    if max_point_radius is not None:
        radii = radii.clamp(max=float(max_point_radius))

    if color_by_opacity:
        rgb = rgb * opacity.unsqueeze(-1)

    return {
        "positions": xyz.detach().cpu().numpy().astype(np.float32),
        "colors": (rgb.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8),
        "radii": radii.detach().cpu().numpy().astype(np.float32),
    }


def _state_to_ellipsoid_arrays(
    state: Dict[str, Any],
    *,
    min_opacity: float,
    ellipsoid_half_size_scale: float,
    max_ellipsoid_half_size: Optional[float],
    max_gaussians: Optional[int],
    seed: int,
    rotation_format: Literal["wxyz", "xyzw"],
    min_alpha: int,
    alpha_scale: float,
    color_by_opacity: bool,
) -> Dict[str, np.ndarray]:
    xyz = state["xyz"].float()
    opacity = state["opacity"].float().squeeze(-1).clamp(0.0, 1.0)
    scaling = state["scaling"].float().clamp(min=1e-8)
    rotation = state["rotation"].float()
    shs = state["shs"].float()

    rgb = shs.squeeze(1).clamp(0.0, 1.0)

    keep = opacity >= float(min_opacity)
    xyz = xyz[keep]
    opacity = opacity[keep]
    scaling = scaling[keep]
    rotation = rotation[keep]
    rgb = rgb[keep]

    if max_gaussians is not None and xyz.shape[0] > int(max_gaussians):
        g = torch.Generator(device=xyz.device)
        g.manual_seed(int(seed))
        idx = torch.randperm(xyz.shape[0], generator=g)[: int(max_gaussians)]
        xyz = xyz[idx]
        opacity = opacity[idx]
        scaling = scaling[idx]
        rotation = rotation[idx]
        rgb = rgb[idx]

    half_sizes = scaling * float(ellipsoid_half_size_scale)
    if max_ellipsoid_half_size is not None:
        half_sizes = half_sizes.clamp(max=float(max_ellipsoid_half_size))

    rotation = torch.nn.functional.normalize(rotation, dim=-1)
    if rotation_format == "wxyz":
        quaternions = rotation[:, [1, 2, 3, 0]]
    else:
        quaternions = rotation

    rgba = torch.cat(
        [
            (
                (rgb * opacity.unsqueeze(-1)).clip(0.0, 1.0) * 255.0
                if color_by_opacity
                else rgb.clip(0.0, 1.0) * 255.0
            ),
            (opacity * 255.0 * float(alpha_scale))
            .clamp(min=float(min_alpha), max=255.0)
            .unsqueeze(-1),
        ],
        dim=-1,
    )

    return {
        "centers": xyz.detach().cpu().numpy().astype(np.float32),
        "half_sizes": half_sizes.detach().cpu().numpy().astype(np.float32),
        "quaternions": quaternions.detach().cpu().numpy().astype(np.float32),
        "colors": rgba.detach().cpu().numpy().astype(np.uint8),
    }


def _call_with_supported_kwargs(func: Any, kwargs: Dict[str, Any]) -> Any:
    try:
        sig = inspect.signature(func)
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(**supported)
    except (TypeError, ValueError):
        return func()


def _pids_listening_on_port(port: int) -> List[int]:
    try:
        proc = subprocess.run(
            ["ss", "-ltnp"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []

    pids: set[int] = set()
    port_tag = f":{int(port)}"
    for line in proc.stdout.splitlines():
        if port_tag not in line:
            continue
        for match in re.finditer(r"pid=(\d+)", line):
            pids.add(int(match.group(1)))
    return sorted(pids)


def _can_bind_port(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", int(port)))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_pids(pids: List[int], *, grace_seconds: float) -> None:
    current_pid = os.getpid()
    parent_pid = os.getppid()
    targets = [pid for pid in pids if pid not in {current_pid, parent_pid}]
    if not targets:
        return

    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    deadline = time.time() + float(grace_seconds)
    while time.time() < deadline:
        alive = [pid for pid in targets if _pid_alive(pid)]
        if not alive:
            return
        time.sleep(0.1)

    for pid in targets:
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue


def _ensure_port_available(port: int, *, kill_occupied: bool, grace_seconds: float) -> None:
    if _can_bind_port(port):
        return
    pids = _pids_listening_on_port(port)
    if not pids:
        raise RuntimeError(f"Port {port} is unavailable and owning PID could not be determined.")
    if not kill_occupied:
        raise RuntimeError(f"Port {port} is occupied by PID(s): {pids}")

    _LOGGER.warning(f"Port {port} is occupied by PID(s) {pids}. Terminating them...")
    _terminate_pids(pids, grace_seconds=grace_seconds)
    if not _can_bind_port(port):
        remaining = _pids_listening_on_port(port)
        raise RuntimeError(
            f"Failed to reclaim port {port}. Remaining PID(s): {remaining or 'unknown'}"
        )


def _setup_rerun_sink(rr: Any, args: "Args") -> None:
    sink_mode = args.sink_mode.lower()
    if sink_mode == "spawn":
        if not hasattr(rr, "spawn"):
            raise RuntimeError("rerun SDK does not expose rr.spawn().")
        rr.spawn()
        _LOGGER.info("Rerun sink: local spawned viewer process")
        return

    if sink_mode == "connect-grpc":
        if hasattr(rr, "connect_grpc"):
            rr.connect_grpc(args.grpc_url)
        elif hasattr(rr, "connect"):
            rr.connect(args.grpc_url)
        else:
            raise RuntimeError("rerun SDK does not expose rr.connect_grpc() / rr.connect().")
        _LOGGER.info(f"Rerun sink: connected to gRPC endpoint: {args.grpc_url}")
        return

    if sink_mode == "serve-web":
        for port in sorted({int(args.web_port), int(args.grpc_port)}):
            _ensure_port_available(
                port,
                kill_occupied=bool(args.kill_occupied_ports),
                grace_seconds=float(args.port_kill_grace_seconds),
            )

        grpc_uri: Optional[str] = None
        if hasattr(rr, "serve_grpc"):
            grpc_uri = rr.serve_grpc(grpc_port=args.grpc_port)
        elif hasattr(rr, "connect_grpc"):
            rr.connect_grpc(args.grpc_url)
            grpc_uri = args.grpc_url

        if hasattr(rr, "serve_web_viewer"):
            rr.serve_web_viewer(
                web_port=args.web_port,
                open_browser=args.open_browser,
                connect_to=grpc_uri,
            )
            viewer_host = "localhost" if args.web_host in ("0.0.0.0", "::") else args.web_host
            viewer_url = f"http://{viewer_host}:{args.web_port}"
            _LOGGER.info(f"Rerun sink: web viewer at {viewer_url}")
            if grpc_uri is not None:
                encoded = urllib.parse.quote(grpc_uri, safe="")
                auto_connect_url = f"{viewer_url}/?url={encoded}"
                _LOGGER.info(f"Web auto-connect URL: {auto_connect_url}")
                _LOGGER.info(f"Data source URI (manual fallback): {grpc_uri}")
            return

        # Backward compatibility for old SDKs.
        for name in ("serve_web", "serve"):
            if hasattr(rr, name):
                fn = getattr(rr, name)
                _call_with_supported_kwargs(
                    fn,
                    {
                        "host": args.web_host,
                        "port": args.web_port,
                        "web_port": args.web_port,
                        "open_browser": args.open_browser,
                    },
                )
                _LOGGER.info(
                    "Rerun sink: web server enabled "
                    f"(open http://{args.web_host}:{args.web_port})"
                )
                return

        raise RuntimeError("No web-serving API found in rerun SDK.")

    raise ValueError(
        f"Unsupported sink_mode={args.sink_mode!r}. "
        "Expected one of: serve-web, spawn, connect-grpc."
    )


def _set_y_up_view_coordinates(rr: Any) -> None:
    if hasattr(rr, "ViewCoordinates") and hasattr(rr.ViewCoordinates, "RIGHT_HAND_Y_UP"):
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        _LOGGER.info("View coordinates set: RIGHT_HAND_Y_UP (+Y up).")
        return
    if hasattr(rr, "log_view_coordinates"):
        rr.log_view_coordinates("world", up_vector=[0, 1, 0], right_vector=[1, 0, 0])
        _LOGGER.info("View coordinates set via log_view_coordinates (+Y up).")
        return
    _LOGGER.warning("Could not set view coordinates to +Y up (API not available).")


def _resolve_ellipsoid_fill_mode(rr: Any, fill_mode: str) -> Any:
    if not (hasattr(rr, "components") and hasattr(rr.components, "FillMode")):
        return None
    mode_enum = rr.components.FillMode
    normalized = fill_mode.strip().lower()
    mapping = {
        "solid": "Solid",
        "major_wireframe": "MajorWireframe",
        "dense_wireframe": "DenseWireframe",
    }
    name = mapping.get(normalized)
    if name is None:
        return None
    return getattr(mode_enum, name, None)


def _apply_ellipsoid_preset(args: "Args") -> None:
    preset = args.ellipsoid_preset.lower()
    if preset == "custom":
        return

    if preset == "smooth":
        args.max_gaussians_per_person = 30000
        args.ellipsoid_half_size_scale = 2.2
        args.max_ellipsoid_half_size = 1.0
        args.ellipsoid_alpha_scale = 0.35
        args.ellipsoid_fill_mode = "solid"
        return

    if preset == "balanced":
        args.max_gaussians_per_person = 20000
        args.ellipsoid_half_size_scale = 1.8
        args.max_ellipsoid_half_size = 0.8
        args.ellipsoid_alpha_scale = 0.30
        args.ellipsoid_fill_mode = "major_wireframe"
        return

    if preset == "fast":
        args.max_gaussians_per_person = 12000
        args.ellipsoid_half_size_scale = 1.5
        args.max_ellipsoid_half_size = 0.6
        args.ellipsoid_alpha_scale = 0.45
        args.ellipsoid_fill_mode = "solid"
        return

    raise ValueError(
        f"Unsupported ellipsoid_preset={args.ellipsoid_preset!r}. "
        "Expected one of: custom, smooth, balanced, fast."
    )


@dataclass
class Args:
    eval_scene_dir: Path
    frame_idx_range: Tuple[int, int] = (0, 20)
    subsample_rate: int = 5
    src_cam_id: int = 4

    sink_mode: str = "serve-web"
    web_host: str = "0.0.0.0"
    web_port: int = 9876
    grpc_port: int = 9877
    open_browser: bool = False
    grpc_url: str = "rerun+http://127.0.0.1:9877/proxy"
    kill_occupied_ports: bool = True
    port_kill_grace_seconds: float = 2.0

    app_id: str = "dynamic_3dgs"
    splat_primitive: Literal["ellipsoids", "points"] = "ellipsoids"
    ellipsoid_preset: Literal["custom", "smooth", "balanced", "fast"] = "custom"
    rotation_format: Literal["wxyz", "xyzw"] = "wxyz"
    max_gaussians_per_person: Optional[int] = None
    min_opacity: float = 0.0
    point_radius_scale: float = 0.08
    max_point_radius: Optional[float] = 0.2
    point_use_radii: bool = True
    point_color_by_opacity: bool = True
    ellipsoid_half_size_scale: float = 1.5
    max_ellipsoid_half_size: Optional[float] = 0.6
    min_ellipsoid_alpha: int = 24
    ellipsoid_alpha_scale: float = 0.45
    ellipsoid_color_by_opacity: bool = True
    ellipsoid_fill_mode: Literal["solid", "major_wireframe", "dense_wireframe"] = "solid"

    include_rgb: bool = True
    rgb_max_side: int = 1280
    include_gsplat_render: bool = False
    gsplat_max_side: int = 960
    gsplat_device: Literal["auto", "cuda", "cpu"] = "auto"
    gsplat_background: Literal["black", "white"] = "black"
    gsplat_packed: bool = False
    gsplat_rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    gsplat_near_plane: float = 0.01
    gsplat_far_plane: float = 1.0e10
    gsplat_radius_clip: float = 0.0
    seed: int = 0
    hold_open: bool = True


def main() -> None:
    _setup_logging()
    args = tyro.cli(Args)
    if args.splat_primitive == "ellipsoids":
        _apply_ellipsoid_preset(args)

    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError(
            "rerun-sdk is required. Install with: pip install rerun-sdk"
        ) from exc

    gsplat_rasterization = None
    gsplat_device: Optional[torch.device] = None
    if args.include_gsplat_render:
        try:
            import gsplat as gs
        except ImportError as exc:
            raise ImportError(
                "gsplat is required for --include-gsplat-render. Install gsplat in the active env."
            ) from exc

        gsplat_rasterization = _resolve_gsplat_rasterization(gs)
        if gsplat_rasterization is None:
            raise RuntimeError("Could not find gsplat rasterization() API in installed gsplat.")

        requested_device = args.gsplat_device.lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            _LOGGER.warning("Requested gsplat_device='cuda' but CUDA is unavailable. Falling back to CPU.")
            requested_device = "cpu"
        gsplat_device = torch.device(requested_device)
        _LOGGER.info(f"gsplat render enabled on device={gsplat_device}")

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

    _LOGGER.info(
        (
            "Scene: %s | frames: %d (range=%s, subsample=%d) | people: %d | "
            "sink=%s | primitive=%s | preset=%s | fill_mode=%s"
        ),
        str(args.eval_scene_dir),
        len(frame_stems),
        args.frame_idx_range,
        subsample_rate,
        len(person_dirs),
        args.sink_mode,
        args.splat_primitive,
        args.ellipsoid_preset,
        args.ellipsoid_fill_mode,
    )

    rr.init(args.app_id)
    _setup_rerun_sink(rr, args)
    _set_y_up_view_coordinates(rr)

    rgb_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
    rgb_cache_size = 32

    def _get_rgb(frame_stem: str) -> Optional[np.ndarray]:
        cached = rgb_cache.get(frame_stem)
        if cached is not None:
            rgb_cache.move_to_end(frame_stem)
            return cached
        rgb = _load_rgb_image(args.eval_scene_dir, args.src_cam_id, frame_stem)
        if rgb is None:
            return None
        rgb = _downscale_rgb(rgb, max_side=args.rgb_max_side)
        rgb_cache[frame_stem] = rgb
        if len(rgb_cache) > rgb_cache_size:
            rgb_cache.popitem(last=False)
        return rgb

    for frame_idx, frame_stem in enumerate(
        tqdm(frame_stems, desc="Logging frames to Rerun", total=len(frame_stems))
    ):
        rr.set_time("frame", sequence=frame_idx)

        rgb_for_resolution: Optional[np.ndarray] = None
        if args.include_rgb:
            rgb = _get_rgb(frame_stem)
            if rgb is not None:
                rgb_for_resolution = rgb
                rr.log(f"world/rgb/cam_{args.src_cam_id}", rr.Image(rgb))

        gs_means: List[torch.Tensor] = []
        gs_scales: List[torch.Tensor] = []
        gs_quats: List[torch.Tensor] = []
        gs_opacities: List[torch.Tensor] = []
        gs_colors: List[torch.Tensor] = []

        for person_dir in person_dirs:
            state_path = person_dir / f"{frame_stem}.pt"
            person_path = f"world/3dgs/person_{person_dir.name}"
            if not state_path.exists():
                if args.splat_primitive == "ellipsoids":
                    rr.log(
                        person_path,
                        rr.Ellipsoids3D(
                            centers=np.zeros((0, 3), dtype=np.float32),
                            half_sizes=np.zeros((0, 3), dtype=np.float32),
                        ),
                    )
                else:
                    rr.log(person_path, rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
                continue

            state = _torch_load(state_path)
            if gsplat_rasterization is not None:
                gs_data = _state_to_gsplat_arrays(
                    state,
                    min_opacity=args.min_opacity,
                    max_gaussians=args.max_gaussians_per_person,
                    seed=args.seed + frame_idx,
                    rotation_format=args.rotation_format,
                )
                if gs_data["means"].numel() > 0:
                    gs_means.append(gs_data["means"])
                    gs_scales.append(gs_data["scales"])
                    gs_quats.append(gs_data["quats"])
                    gs_opacities.append(gs_data["opacities"])
                    gs_colors.append(gs_data["colors"])

            if args.splat_primitive == "ellipsoids":
                ellipsoid_data = _state_to_ellipsoid_arrays(
                    state,
                    min_opacity=args.min_opacity,
                    ellipsoid_half_size_scale=args.ellipsoid_half_size_scale,
                    max_ellipsoid_half_size=args.max_ellipsoid_half_size,
                    max_gaussians=args.max_gaussians_per_person,
                    seed=args.seed + frame_idx,
                    rotation_format=args.rotation_format,
                    min_alpha=args.min_ellipsoid_alpha,
                    alpha_scale=args.ellipsoid_alpha_scale,
                    color_by_opacity=args.ellipsoid_color_by_opacity,
                )
                ellipsoid_kwargs: Dict[str, Any] = {
                    "centers": ellipsoid_data["centers"],
                    "half_sizes": ellipsoid_data["half_sizes"],
                    "quaternions": ellipsoid_data["quaternions"],
                    "colors": ellipsoid_data["colors"],
                }
                fill_mode = _resolve_ellipsoid_fill_mode(rr, args.ellipsoid_fill_mode)
                if fill_mode is not None:
                    ellipsoid_kwargs["fill_mode"] = fill_mode
                rr.log(person_path, rr.Ellipsoids3D(**ellipsoid_kwargs))
            else:
                point_data = _state_to_point_arrays(
                    state,
                    min_opacity=args.min_opacity,
                    point_radius_scale=args.point_radius_scale,
                    max_point_radius=args.max_point_radius,
                    max_gaussians=args.max_gaussians_per_person,
                    seed=args.seed + frame_idx,
                    color_by_opacity=args.point_color_by_opacity,
                )
                point_kwargs: Dict[str, Any] = {
                    "positions": point_data["positions"],
                    "colors": point_data["colors"],
                }
                if args.point_use_radii:
                    point_kwargs["radii"] = point_data["radii"]
                rr.log(person_path, rr.Points3D(**point_kwargs))

        if gsplat_rasterization is not None and gs_means:
            camera_path = (
                args.eval_scene_dir / "all_cameras" / f"{args.src_cam_id}" / f"{frame_stem}.npz"
            )
            if camera_path.exists():
                try:
                    intr, extr = _load_camera_npz(camera_path)
                    width, height = _infer_resolution(intr, rgb_for_resolution)
                    max_side = int(args.gsplat_max_side)
                    if max_side > 0 and max(width, height) > max_side:
                        scale = float(max_side) / float(max(width, height))
                        new_w = max(1, int(round(width * scale)))
                        new_h = max(1, int(round(height * scale)))
                        sx = float(new_w) / float(width)
                        sy = float(new_h) / float(height)
                        intr = _scale_intrinsics(intr, sx, sy)
                        width, height = new_w, new_h

                    gsplat_rgb = _render_gsplat_image(
                        rasterization=gsplat_rasterization,
                        means=torch.cat(gs_means, dim=0),
                        scales=torch.cat(gs_scales, dim=0),
                        quats=torch.cat(gs_quats, dim=0),
                        opacities=torch.cat(gs_opacities, dim=0),
                        colors=torch.cat(gs_colors, dim=0),
                        intrinsics=intr,
                        extrinsics=extr,
                        width=width,
                        height=height,
                        device=gsplat_device if gsplat_device is not None else torch.device("cpu"),
                        packed=args.gsplat_packed,
                        rasterize_mode=args.gsplat_rasterize_mode,
                        near_plane=args.gsplat_near_plane,
                        far_plane=args.gsplat_far_plane,
                        radius_clip=args.gsplat_radius_clip,
                        white_background=args.gsplat_background == "white",
                    )
                    rr.log(f"world/gsplat/cam_{args.src_cam_id}/rgb", rr.Image(gsplat_rgb))
                except Exception as exc:
                    _LOGGER.warning(
                        "Skipping gsplat render for frame %s due to: %s", frame_stem, str(exc)
                    )
            else:
                _LOGGER.warning("Missing camera file for gsplat render: %s", str(camera_path))

    _LOGGER.info(
        "Logged %d frames across %d people to Rerun (timeline='frame').",
        len(frame_stems),
        len(person_dirs),
    )
    if args.sink_mode.lower() == "serve-web":
        _LOGGER.info(
            "If running remotely: tunnel with "
            f"`ssh -L {args.web_port}:127.0.0.1:{args.web_port} "
            f"-L {args.grpc_port}:127.0.0.1:{args.grpc_port} <user>@<server>` "
            f"and open http://localhost:{args.web_port}"
        )
    if args.hold_open and args.sink_mode.lower() == "serve-web":
        _LOGGER.info("Serving web viewer. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            _LOGGER.info("Shutting down web viewer...")


if __name__ == "__main__":
    main()
