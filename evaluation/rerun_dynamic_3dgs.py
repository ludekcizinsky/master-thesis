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
from typing import Any, Dict, List, Optional, Tuple

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


def _state_to_point_arrays(
    state: Dict[str, Any],
    *,
    min_opacity: float,
    point_radius_scale: float,
    max_point_radius: Optional[float],
    max_gaussians: Optional[int],
    seed: int,
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

    rgb = rgb * opacity.unsqueeze(-1)

    return {
        "positions": xyz.detach().cpu().numpy().astype(np.float32),
        "colors": (rgb.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8),
        "radii": radii.detach().cpu().numpy().astype(np.float32),
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
    max_gaussians_per_person: Optional[int] = 200000
    min_opacity: float = 0.01
    point_radius_scale: float = 0.02
    max_point_radius: Optional[float] = 0.05

    include_rgb: bool = True
    rgb_max_side: int = 1280
    seed: int = 0
    hold_open: bool = True


def main() -> None:
    _setup_logging()
    args = tyro.cli(Args)

    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError(
            "rerun-sdk is required. Install with: pip install rerun-sdk"
        ) from exc

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
        "Scene: %s | frames: %d (range=%s, subsample=%d) | people: %d | sink=%s",
        str(args.eval_scene_dir),
        len(frame_stems),
        args.frame_idx_range,
        subsample_rate,
        len(person_dirs),
        args.sink_mode,
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

        if args.include_rgb:
            rgb = _get_rgb(frame_stem)
            if rgb is not None:
                rr.log(f"world/rgb/cam_{args.src_cam_id}", rr.Image(rgb))

        for person_dir in person_dirs:
            state_path = person_dir / f"{frame_stem}.pt"
            person_path = f"world/3dgs/person_{person_dir.name}"
            if not state_path.exists():
                rr.log(person_path, rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
                continue

            state = _torch_load(state_path)
            point_data = _state_to_point_arrays(
                state,
                min_opacity=args.min_opacity,
                point_radius_scale=args.point_radius_scale,
                max_point_radius=args.max_point_radius,
                max_gaussians=args.max_gaussians_per_person,
                seed=args.seed + frame_idx,
            )
            rr.log(
                person_path,
                rr.Points3D(
                    point_data["positions"],
                    colors=point_data["colors"],
                    radii=point_data["radii"],
                ),
            )

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
