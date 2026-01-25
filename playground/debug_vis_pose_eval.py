from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tyro

import viser
import viser.transforms as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submodules.smplx import smplx  # noqa: E402


def _sorted_frame_files(root: Path, pattern: str) -> List[Path]:
    paths = [p for p in root.glob(pattern) if p.is_file()]
    if not paths:
        return []

    def _key(p: Path) -> Tuple[int, str]:
        stem = p.stem
        if stem.isdigit():
            return (0, f"{int(stem):012d}")
        return (1, stem)

    return sorted(paths, key=_key)


def _select_frame(frame_files: List[Path], frame_index: int, frame_name: Optional[str]) -> Path:
    if frame_name:
        if "." in frame_name:
            matches = [p for p in frame_files if p.name == frame_name]
        else:
            matches = [p for p in frame_files if p.stem == frame_name]
        if not matches and frame_name.isdigit():
            target = int(frame_name)
            matches = [p for p in frame_files if p.stem.isdigit() and int(p.stem) == target]
        if not matches:
            raise FileNotFoundError(f"No frame named {frame_name!r}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple frames matched {frame_name!r}")
        return matches[0]
    for stem in (str(frame_index), f"{frame_index:06d}", f"{frame_index:08d}"):
        for path in frame_files:
            if path.stem == stem:
                return path
    if frame_index < 0 or frame_index >= len(frame_files):
        raise IndexError(f"frame_index {frame_index} is out of range [0, {len(frame_files) - 1}]")
    return frame_files[frame_index]


def _load_metrics_csv(path: Path) -> Tuple[Dict[int, Dict[str, float]], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Metrics file has no header: {path}")
        if "frame_id" not in reader.fieldnames:
            raise ValueError(f"Metrics file missing frame_id column: {path}")
        metric_names = [name for name in reader.fieldnames if name != "frame_id"]
        if not metric_names:
            raise ValueError(f"No metric columns found in {path}")
        metrics: Dict[int, Dict[str, float]] = {}
        for row in reader:
            raw_frame = row.get("frame_id")
            if raw_frame is None:
                continue
            raw_frame = raw_frame.strip()
            if raw_frame == "":
                continue
            if not raw_frame.lstrip("-").isdigit():
                raise ValueError(f"Non-numeric frame_id {raw_frame!r} in {path}")
            frame_id = int(raw_frame)
            frame_metrics: Dict[str, float] = {}
            for name in metric_names:
                raw_val = row.get(name, "")
                if raw_val is None or raw_val == "":
                    raise ValueError(f"Missing value for {name} at frame {frame_id} in {path}")
                try:
                    frame_metrics[name] = float(raw_val)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric value {raw_val!r} for {name} at frame {frame_id} in {path}"
                    ) from exc
            metrics[frame_id] = frame_metrics
    if not metrics:
        raise ValueError(f"No metric rows found in {path}")
    return metrics, metric_names


def _format_metrics(metrics: Dict[str, float], metric_names: List[str], *, signed: bool = False) -> str:
    parts = []
    fmt = "{:+.3f}" if signed else "{:.3f}"
    for name in metric_names:
        val = metrics.get(name)
        if val is None:
            parts.append(f"{name}=nan")
        else:
            parts.append(f"{name}=" + fmt.format(val))
    return ", ".join(parts)


def _metrics_table_markdown(
    baseline: Dict[str, float],
    comparison: Dict[str, float],
    delta: Dict[str, float],
    metric_names: List[str],
) -> str:
    header = "| | " + " | ".join(metric_names) + " |"
    separator = "|" + " --- |" * (len(metric_names) + 1)

    def _row(label: str, values: Dict[str, float], signed: bool = False) -> str:
        fmt = "{:+.3f}" if signed else "{:.3f}"
        cells = []
        for name in metric_names:
            val = values.get(name)
            cells.append("nan" if val is None else fmt.format(val))
        return "| " + label + " | " + " | ".join(cells) + " |"

    return "\n".join(
        [
            header,
            separator,
            _row("Baseline", baseline),
            _row("Comparison", comparison),
            _row("Delta", delta, signed=True),
        ]
    )


def _discover_pose_types(scene_dir: Path) -> List[str]:
    suffix = "_pose_estimation_metrics_per_frame.csv"
    types: List[str] = []
    for path in scene_dir.glob(f"*{suffix}"):
        name = path.name
        if not name.endswith(suffix):
            continue
        prefix = name[: -len(suffix)]
        if prefix:
            types.append(prefix)
    return sorted(set(types))

def _load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        if not data.files:
            return None
        return {k: np.asarray(data[k]) for k in data.files}


def _pad_or_truncate(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_dim = int(vec.shape[-1])
    if current_dim == target_dim:
        return vec
    if current_dim > target_dim:
        return vec[..., :target_dim]
    pad = torch.zeros((*vec.shape[:-1], target_dim - current_dim), device=vec.device, dtype=vec.dtype)
    return torch.cat([vec, pad], dim=-1)


def _reshape_pose(pose: torch.Tensor, joints: int) -> torch.Tensor:
    if pose.ndim == 3 and pose.shape[-2:] == (joints, 3):
        return pose[:1]
    if pose.ndim == 2 and pose.shape == (joints, 3):
        return pose.unsqueeze(0)
    flat = pose.reshape(-1)
    expected = joints * 3
    if flat.numel() != expected:
        raise ValueError(f"Expected pose with {expected} values, got {flat.numel()}")
    return flat.view(1, joints, 3)


def _reshape_vec3(vec: torch.Tensor) -> torch.Tensor:
    flat = vec.reshape(-1)
    if flat.numel() != 3:
        raise ValueError(f"Expected 3 values, got {flat.numel()}")
    return flat.view(1, 3)


def _reshape_body_pose_flat(pose: torch.Tensor, joints: int) -> torch.Tensor:
    if pose.ndim == 2 and pose.shape == (joints, 3):
        return pose.reshape(1, joints * 3)
    if pose.ndim == 3 and pose.shape[-2:] == (joints, 3):
        return pose.reshape(-1, joints * 3)
    if pose.ndim == 2 and pose.shape[1] == joints * 3:
        return pose[:1]
    flat = pose.reshape(-1)
    expected = joints * 3
    if flat.numel() != expected:
        raise ValueError(f"Expected body pose with {expected} values, got {flat.numel()}")
    return flat.view(1, expected)


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


def _extract_smpl_person_params(smpl_data: Dict[str, np.ndarray], pid: int) -> Optional[Dict[str, np.ndarray]]:
    required = ["betas", "global_orient", "body_pose"]
    if any(k not in smpl_data for k in required):
        return None
    transl_key = "transl" if "transl" in smpl_data else "trans" if "trans" in smpl_data else None
    if transl_key is None:
        return None
    try:
        params = {
            "betas": smpl_data["betas"][pid],
            "global_orient": smpl_data["global_orient"][pid],
            "body_pose": smpl_data["body_pose"][pid],
            "transl": smpl_data[transl_key][pid],
        }
    except IndexError:
        return None
    return params


def _smplx_vertices(
    layer,
    smplx_data: Dict[str, np.ndarray],
    device: torch.device,
) -> Optional[np.ndarray]:
    if "trans" not in smplx_data:
        return None
    num_people = int(smplx_data["trans"].shape[0])
    if num_people <= 0:
        return None

    expr_dim = int(getattr(layer, "num_expression_coeffs", 0))
    expected_betas = int(getattr(layer, "num_betas", smplx_data["betas"].shape[-1]))

    verts_out = []
    for pid in range(num_people):
        params_np = _extract_person_params(smplx_data, pid)
        if params_np is None:
            continue
        params = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in params_np.items()}

        betas = _pad_or_truncate(params["betas"].view(1, -1), expected_betas)
        call_args = dict(
            global_orient=_reshape_vec3(params["root_pose"]),
            body_pose=_reshape_pose(params["body_pose"], 21),
            jaw_pose=_reshape_vec3(params["jaw_pose"]),
            leye_pose=_reshape_vec3(params["leye_pose"]),
            reye_pose=_reshape_vec3(params["reye_pose"]),
            left_hand_pose=_reshape_pose(params["lhand_pose"], 15),
            right_hand_pose=_reshape_pose(params["rhand_pose"], 15),
            betas=betas,
            transl=_reshape_vec3(params["trans"]),
        )
        if expr_dim > 0:
            expr = params.get("expression")
            if expr is None:
                expr = torch.zeros((1, expr_dim), device=device, dtype=betas.dtype)
            else:
                expr = _pad_or_truncate(expr.view(1, -1), expr_dim)
            call_args["expression"] = expr

        with torch.no_grad():
            output = layer(**call_args)
        verts_out.append(output.vertices[0].detach().cpu().numpy())

    if not verts_out:
        return None
    return np.stack(verts_out, axis=0)


def _smpl_vertices(
    layer,
    smpl_data: Dict[str, np.ndarray],
    device: torch.device,
) -> Optional[np.ndarray]:
    transl_key = "transl" if "transl" in smpl_data else "trans" if "trans" in smpl_data else None
    if transl_key is None or "betas" not in smpl_data:
        return None
    num_people = int(smpl_data[transl_key].shape[0])
    if num_people <= 0:
        return None

    expected_betas = int(getattr(layer, "num_betas", smpl_data["betas"].shape[-1]))

    verts_out = []
    for pid in range(num_people):
        params_np = _extract_smpl_person_params(smpl_data, pid)
        if params_np is None:
            continue
        params = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in params_np.items()}

        betas = _pad_or_truncate(params["betas"].view(1, -1), expected_betas)
        call_args = dict(
            global_orient=_reshape_vec3(params["global_orient"]),
            body_pose=_reshape_body_pose_flat(params["body_pose"], 23),
            betas=betas,
            transl=_reshape_vec3(params["transl"]),
        )
        with torch.no_grad():
            output = layer(**call_args)
        verts_out.append(output.vertices[0].detach().cpu().numpy())

    if not verts_out:
        return None
    return np.stack(verts_out, axis=0)


def _build_smplx_layer(model_folder: Path, gender: str, ext: str, device: torch.device):
    layer = smplx.create(
        str(model_folder),
        model_type="smplx",
        gender=gender,
        ext=ext,
        use_pca=False,
        use_face_contour=True,
        flat_hand_mean=True,
    )
    return layer.to(device)


def _build_smpl_layer(model_folder: Path, gender: str, ext: str, device: torch.device):
    layer = smplx.create(
        str(model_folder),
        model_type="smpl",
        gender=gender,
        ext=ext,
        use_pca=False,
        use_face_contour=False,
        flat_hand_mean=True,
    )
    return layer.to(device)


def _color_for_person(base: np.ndarray, pid: int) -> Tuple[int, int, int]:
    scale = 0.6 + 0.4 * ((pid % 5) / 4.0)
    color = np.clip(base * scale, 0, 255)
    return tuple(int(c) for c in color)


def _update_bounds(
    verts: np.ndarray, bounds: Tuple[Optional[np.ndarray], Optional[np.ndarray]]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if verts.size == 0:
        return bounds
    verts_flat = verts.reshape(-1, 3)
    vmin = verts_flat.min(axis=0)
    vmax = verts_flat.max(axis=0)
    min_bound, max_bound = bounds
    if min_bound is None:
        return vmin, vmax
    return np.minimum(min_bound, vmin), np.maximum(max_bound, vmax)


@dataclass
class Args:
    gt_scene_dir: Path
    pred_scene_dir: Path
    comp_pred_scene_dir: Optional[Path] = None
    sort_metric: str = "mpjpe_mm"
    top_k: int = 10
    frame_index: int = 0
    frame_name: Optional[str] = None
    port: int = 8080
    center_scene: bool = True
    is_minus_y_up: bool = True
    device: str = "cuda"
    model_folder: Path = Path("/home/cizinsky/body_models")
    smpl_model_ext: str = "pkl"
    smplx_model_ext: str = "npz"
    gender: str = "neutral"
    smpl_pattern: str = "*.npz"
    smplx_pattern: str = "*.npz"
    mesh_opacity: float = 0.85


def _visualize_comparison_mode(args: Args) -> None:
    if args.comp_pred_scene_dir is None:
        raise ValueError("comp_pred_scene_dir must be provided for comparison mode.")

    baseline_pose_types = _discover_pose_types(args.pred_scene_dir)
    comp_pose_types = _discover_pose_types(args.comp_pred_scene_dir)
    available_pose_types = sorted(set(baseline_pose_types) & set(comp_pose_types))
    if not available_pose_types:
        raise ValueError(
            "No common pose types found between baseline and comparison metrics files."
        )
    current_pose_type = "smpl" if "smpl" in available_pose_types else available_pose_types[0]

    allowed_metrics = ["mpjpe_mm", "mve_mm"]

    def _load_comparison_metrics(pose_type: str):
        metrics_filename = f"{pose_type}_pose_estimation_metrics_per_frame.csv"
        baseline_metrics_path = args.pred_scene_dir / metrics_filename
        comp_metrics_path = args.comp_pred_scene_dir / metrics_filename

        baseline_metrics, metric_names = _load_metrics_csv(baseline_metrics_path)
        comp_metrics, comp_metric_names = _load_metrics_csv(comp_metrics_path)
        if comp_metric_names != metric_names:
            raise ValueError(
                "Metric columns mismatch between baseline and comparison files: "
                f"{metric_names} vs {comp_metric_names}"
            )

        metric_names = [m for m in metric_names if m in allowed_metrics]
        if not metric_names:
            raise ValueError(
                f"None of the allowed metrics {allowed_metrics} found in {baseline_metrics_path}"
            )
        for frame_id in list(baseline_metrics.keys()):
            baseline_metrics[frame_id] = {
                k: v for k, v in baseline_metrics[frame_id].items() if k in metric_names
            }
        for frame_id in list(comp_metrics.keys()):
            comp_metrics[frame_id] = {
                k: v for k, v in comp_metrics[frame_id].items() if k in metric_names
            }
        if args.sort_metric not in metric_names:
            raise ValueError(
                f"sort_metric {args.sort_metric!r} must be one of {metric_names}"
            )

        baseline_frames = set(baseline_metrics.keys())
        comp_frames = set(comp_metrics.keys())
        if baseline_frames != comp_frames:
            missing_in_comp = sorted(baseline_frames - comp_frames)
            missing_in_base = sorted(comp_frames - baseline_frames)
            raise ValueError(
                "Frame mismatch between baseline and comparison metrics. "
                f"Missing in comp: {missing_in_comp[:10]} "
                f"Missing in baseline: {missing_in_base[:10]}"
            )

        delta_metrics: Dict[int, Dict[str, float]] = {}
        for frame_id in baseline_frames:
            delta_metrics[frame_id] = {
                name: comp_metrics[frame_id][name] - baseline_metrics[frame_id][name]
                for name in metric_names
            }

        sorted_frames = sorted(
            baseline_frames,
            key=lambda fid: baseline_metrics[fid][args.sort_metric]
            - comp_metrics[fid][args.sort_metric],
            reverse=True,
        )
        if args.top_k > 0:
            sorted_frames = sorted_frames[: min(args.top_k, len(sorted_frames))]
        if not sorted_frames:
            raise ValueError("No frames available for visualization after sorting.")

        return baseline_metrics, comp_metrics, delta_metrics, metric_names, sorted_frames

    (
        baseline_metrics,
        comp_metrics,
        delta_metrics,
        metric_names,
        sorted_frames,
    ) = _load_comparison_metrics(current_pose_type)

    gt_smplx_dir = args.gt_scene_dir / "smplx"
    pred_smplx_dir = args.pred_scene_dir / "smplx"
    comp_pred_smplx_dir = args.comp_pred_scene_dir / "smplx"
    gt_smpl_dir = args.gt_scene_dir / "smpl"
    pred_smpl_dir = args.pred_scene_dir / "smpl"
    comp_pred_smpl_dir = args.comp_pred_scene_dir / "smpl"

    if not gt_smplx_dir.exists():
        raise FileNotFoundError(f"GT smplx dir not found: {gt_smplx_dir}")
    if not pred_smplx_dir.exists():
        raise FileNotFoundError(f"Pred smplx dir not found: {pred_smplx_dir}")
    if not comp_pred_smplx_dir.exists():
        raise FileNotFoundError(f"Comparison pred smplx dir not found: {comp_pred_smplx_dir}")

    gt_frames = _sorted_frame_files(gt_smplx_dir, args.smplx_pattern)
    pred_frames = _sorted_frame_files(pred_smplx_dir, args.smplx_pattern)
    comp_pred_frames = _sorted_frame_files(comp_pred_smplx_dir, args.smplx_pattern)
    if not gt_frames:
        raise FileNotFoundError(f"No GT smplx files found in {gt_smplx_dir} with {args.smplx_pattern}")
    if not pred_frames:
        raise FileNotFoundError(f"No pred smplx files found in {pred_smplx_dir} with {args.smplx_pattern}")
    if not comp_pred_frames:
        raise FileNotFoundError(
            f"No comparison pred smplx files found in {comp_pred_smplx_dir} with {args.smplx_pattern}"
        )

    gt_smpl_frames = _sorted_frame_files(gt_smpl_dir, args.smpl_pattern) if gt_smpl_dir.exists() else []
    pred_smpl_frames = _sorted_frame_files(pred_smpl_dir, args.smpl_pattern) if pred_smpl_dir.exists() else []
    comp_pred_smpl_frames = (
        _sorted_frame_files(comp_pred_smpl_dir, args.smpl_pattern) if comp_pred_smpl_dir.exists() else []
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    smplx_layer = _build_smplx_layer(args.model_folder, args.gender, args.smplx_model_ext, device)
    smpl_layer = (
        _build_smpl_layer(args.model_folder, args.gender, args.smpl_model_ext, device)
        if gt_smpl_frames or pred_smpl_frames or comp_pred_smpl_frames
        else None
    )
    smplx_faces = np.asarray(smplx_layer.faces, dtype=np.int32)
    smpl_faces = np.asarray(smpl_layer.faces, dtype=np.int32) if smpl_layer is not None else None

    gt_nodes: Dict[int, viser.FrameHandle] = {}
    pred_nodes: Dict[int, viser.FrameHandle] = {}
    comp_nodes: Dict[int, viser.FrameHandle] = {}
    gt_smpl_nodes: Dict[int, viser.FrameHandle] = {}
    pred_smpl_nodes: Dict[int, viser.FrameHandle] = {}
    comp_smpl_nodes: Dict[int, viser.FrameHandle] = {}

    bounds: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)

    # Preload meshes for the sorted frames.
    for frame_id in sorted_frames:
        gt_frame_path = _select_frame(gt_frames, 0, str(frame_id))
        pred_frame_path = _select_frame(pred_frames, 0, str(frame_id))
        comp_frame_path = _select_frame(comp_pred_frames, 0, str(frame_id))

        gt_data = _load_npz(gt_frame_path)
        pred_data = _load_npz(pred_frame_path)
        comp_data = _load_npz(comp_frame_path)
        if gt_data is None or pred_data is None or comp_data is None:
            raise FileNotFoundError(f"Missing SMPL-X params for frame {frame_id}")

        gt_verts = _smplx_vertices(smplx_layer, gt_data, device)
        pred_verts = _smplx_vertices(smplx_layer, pred_data, device)
        comp_verts = _smplx_vertices(smplx_layer, comp_data, device)

        if gt_verts is not None:
            bounds = _update_bounds(gt_verts, bounds)
        if pred_verts is not None:
            bounds = _update_bounds(pred_verts, bounds)
        if comp_verts is not None:
            bounds = _update_bounds(comp_verts, bounds)

        gt_smpl_verts = None
        pred_smpl_verts = None
        comp_smpl_verts = None
        if smpl_layer is not None and smpl_faces is not None:
            if gt_smpl_frames:
                gt_smpl_path = _select_frame(gt_smpl_frames, 0, str(frame_id))
                gt_smpl_data = _load_npz(gt_smpl_path)
                if gt_smpl_data is not None:
                    gt_smpl_verts = _smpl_vertices(smpl_layer, gt_smpl_data, device)
            if pred_smpl_frames:
                pred_smpl_path = _select_frame(pred_smpl_frames, 0, str(frame_id))
                pred_smpl_data = _load_npz(pred_smpl_path)
                if pred_smpl_data is not None:
                    pred_smpl_verts = _smpl_vertices(smpl_layer, pred_smpl_data, device)
            if comp_pred_smpl_frames:
                comp_smpl_path = _select_frame(comp_pred_smpl_frames, 0, str(frame_id))
                comp_smpl_data = _load_npz(comp_smpl_path)
                if comp_smpl_data is not None:
                    comp_smpl_verts = _smpl_vertices(smpl_layer, comp_smpl_data, device)

        if gt_smpl_verts is not None:
            bounds = _update_bounds(gt_smpl_verts, bounds)
        if pred_smpl_verts is not None:
            bounds = _update_bounds(pred_smpl_verts, bounds)
        if comp_smpl_verts is not None:
            bounds = _update_bounds(comp_smpl_verts, bounds)

    center_offset = np.zeros(3, dtype=np.float32)
    if args.center_scene and bounds[0] is not None and bounds[1] is not None:
        center_offset = (bounds[0] + bounds[1]) * 0.5

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_width="large")
    angle = -np.pi / 2 if args.is_minus_y_up else np.pi / 2
    R_fix = tf.SO3.from_x_radians(angle)
    server.scene.add_frame(
        "/scene",
        show_axes=False,
        wxyz=tuple(R_fix.wxyz),
        position=tuple((-R_fix.apply(center_offset)).tolist()),
    )

    gt_root = server.scene.add_frame("/scene/gt_smplx", show_axes=False)
    pred_root = server.scene.add_frame("/scene/pred_smplx", show_axes=False)
    comp_root = server.scene.add_frame("/scene/comp_pred_smplx", show_axes=False)
    gt_smpl_root = server.scene.add_frame("/scene/gt_smpl", show_axes=False) if smpl_faces is not None else None
    pred_smpl_root = server.scene.add_frame("/scene/pred_smpl", show_axes=False) if smpl_faces is not None else None
    comp_smpl_root = server.scene.add_frame("/scene/comp_pred_smpl", show_axes=False) if smpl_faces is not None else None

    gt_base = np.array([70, 130, 255], dtype=np.float32)
    pred_base = np.array([255, 140, 70], dtype=np.float32)
    comp_pred_base = np.array([140, 200, 90], dtype=np.float32)
    gt_smpl_base = np.array([60, 200, 200], dtype=np.float32)
    pred_smpl_base = np.array([220, 80, 80], dtype=np.float32)
    comp_smpl_base = np.array([200, 140, 40], dtype=np.float32)

    def _create_frame_nodes(frame_id: int) -> None:
        if frame_id in gt_nodes:
            return
        gt_frame_path = _select_frame(gt_frames, 0, str(frame_id))
        pred_frame_path = _select_frame(pred_frames, 0, str(frame_id))
        comp_frame_path = _select_frame(comp_pred_frames, 0, str(frame_id))

        gt_data = _load_npz(gt_frame_path)
        pred_data = _load_npz(pred_frame_path)
        comp_data = _load_npz(comp_frame_path)

        gt_verts = _smplx_vertices(smplx_layer, gt_data, device) if gt_data is not None else None
        pred_verts = _smplx_vertices(smplx_layer, pred_data, device) if pred_data is not None else None
        comp_verts = _smplx_vertices(smplx_layer, comp_data, device) if comp_data is not None else None

        gt_node = server.scene.add_frame(f"/scene/gt_smplx/f_{frame_id}", show_axes=False)
        pred_node = server.scene.add_frame(f"/scene/pred_smplx/f_{frame_id}", show_axes=False)
        comp_node = server.scene.add_frame(f"/scene/comp_pred_smplx/f_{frame_id}", show_axes=False)

        if gt_verts is not None:
            for pid in range(gt_verts.shape[0]):
                color = _color_for_person(gt_base, pid)
                handle = server.scene.add_mesh_simple(
                    f"/scene/gt_smplx/f_{frame_id}/person_{pid}",
                    vertices=gt_verts[pid],
                    faces=smplx_faces,
                    color=color,
                )
                if hasattr(handle, "opacity"):
                    handle.opacity = float(args.mesh_opacity)

        if pred_verts is not None:
            for pid in range(pred_verts.shape[0]):
                color = _color_for_person(pred_base, pid)
                handle = server.scene.add_mesh_simple(
                    f"/scene/pred_smplx/f_{frame_id}/person_{pid}",
                    vertices=pred_verts[pid],
                    faces=smplx_faces,
                    color=color,
                )
                if hasattr(handle, "opacity"):
                    handle.opacity = float(args.mesh_opacity)

        if comp_verts is not None:
            for pid in range(comp_verts.shape[0]):
                color = _color_for_person(comp_pred_base, pid)
                handle = server.scene.add_mesh_simple(
                    f"/scene/comp_pred_smplx/f_{frame_id}/person_{pid}",
                    vertices=comp_verts[pid],
                    faces=smplx_faces,
                    color=color,
                )
                if hasattr(handle, "opacity"):
                    handle.opacity = float(args.mesh_opacity)

        gt_node.visible = False
        pred_node.visible = False
        comp_node.visible = False
        gt_nodes[frame_id] = gt_node
        pred_nodes[frame_id] = pred_node
        comp_nodes[frame_id] = comp_node

        if smpl_faces is not None:
            gt_smpl_node = server.scene.add_frame(f"/scene/gt_smpl/f_{frame_id}", show_axes=False)
            pred_smpl_node = server.scene.add_frame(f"/scene/pred_smpl/f_{frame_id}", show_axes=False)
            comp_smpl_node = server.scene.add_frame(f"/scene/comp_pred_smpl/f_{frame_id}", show_axes=False)

            gt_smpl_verts = None
            pred_smpl_verts = None
            comp_smpl_verts = None

            if gt_smpl_frames:
                gt_smpl_path = _select_frame(gt_smpl_frames, 0, str(frame_id))
                gt_smpl_data = _load_npz(gt_smpl_path)
                if gt_smpl_data is not None:
                    gt_smpl_verts = _smpl_vertices(smpl_layer, gt_smpl_data, device)
            if pred_smpl_frames:
                pred_smpl_path = _select_frame(pred_smpl_frames, 0, str(frame_id))
                pred_smpl_data = _load_npz(pred_smpl_path)
                if pred_smpl_data is not None:
                    pred_smpl_verts = _smpl_vertices(smpl_layer, pred_smpl_data, device)
            if comp_pred_smpl_frames:
                comp_smpl_path = _select_frame(comp_pred_smpl_frames, 0, str(frame_id))
                comp_smpl_data = _load_npz(comp_smpl_path)
                if comp_smpl_data is not None:
                    comp_smpl_verts = _smpl_vertices(smpl_layer, comp_smpl_data, device)

            if gt_smpl_verts is not None:
                for pid in range(gt_smpl_verts.shape[0]):
                    color = _color_for_person(gt_smpl_base, pid)
                    handle = server.scene.add_mesh_simple(
                        f"/scene/gt_smpl/f_{frame_id}/person_{pid}",
                        vertices=gt_smpl_verts[pid],
                        faces=smpl_faces,
                        color=color,
                    )
                    if hasattr(handle, "opacity"):
                        handle.opacity = float(args.mesh_opacity)

            if pred_smpl_verts is not None:
                for pid in range(pred_smpl_verts.shape[0]):
                    color = _color_for_person(pred_smpl_base, pid)
                    handle = server.scene.add_mesh_simple(
                        f"/scene/pred_smpl/f_{frame_id}/person_{pid}",
                        vertices=pred_smpl_verts[pid],
                        faces=smpl_faces,
                        color=color,
                    )
                    if hasattr(handle, "opacity"):
                        handle.opacity = float(args.mesh_opacity)

            if comp_smpl_verts is not None:
                for pid in range(comp_smpl_verts.shape[0]):
                    color = _color_for_person(comp_smpl_base, pid)
                    handle = server.scene.add_mesh_simple(
                        f"/scene/comp_pred_smpl/f_{frame_id}/person_{pid}",
                        vertices=comp_smpl_verts[pid],
                        faces=smpl_faces,
                        color=color,
                    )
                    if hasattr(handle, "opacity"):
                        handle.opacity = float(args.mesh_opacity)

            gt_smpl_node.visible = False
            pred_smpl_node.visible = False
            comp_smpl_node.visible = False
            gt_smpl_nodes[frame_id] = gt_smpl_node
            pred_smpl_nodes[frame_id] = pred_smpl_node
            comp_smpl_nodes[frame_id] = comp_smpl_node

    for frame_id in sorted_frames:
        _create_frame_nodes(frame_id)

    initial_frame_id = sorted_frames[0]
    if args.frame_name and args.frame_name.isdigit():
        requested = int(args.frame_name)
        if requested in sorted_frames:
            initial_frame_id = requested

    current_frame_id = initial_frame_id

    with server.gui.add_folder("Frame selection"):
        pose_type_dropdown = server.gui.add_dropdown(
            "Eval pose type",
            available_pose_types,
            initial_value=current_pose_type,
        )
        frame_dropdown = server.gui.add_dropdown(
            "Frame", [str(fid) for fid in sorted_frames], initial_value=str(initial_frame_id)
        )
        metrics_markdown = server.gui.add_markdown(
            f"**Eval pose type:** `{current_pose_type}`\n\n"
            + _metrics_table_markdown(
                baseline_metrics[current_frame_id],
                comp_metrics[current_frame_id],
                delta_metrics[current_frame_id],
                metric_names,
            )
        )

    with server.gui.add_folder("GT Scene dir"):
        gt_checkbox = server.gui.add_checkbox("Show SMPL-X", True)
        if gt_smpl_root is not None:
            gt_smpl_checkbox = server.gui.add_checkbox("Show SMPL", True)

    with server.gui.add_folder("Prediction Scene Dir"):
        pred_checkbox = server.gui.add_checkbox("Show SMPL-X", True)
        if pred_smpl_root is not None:
            pred_smpl_checkbox = server.gui.add_checkbox("Show SMPL", True)

    with server.gui.add_folder("Comparison Pred Scene Dir"):
        comp_checkbox = server.gui.add_checkbox("Show SMPL-X", True)
        if comp_smpl_root is not None:
            comp_smpl_checkbox = server.gui.add_checkbox("Show SMPL", True)

    def _apply_visibility(frame_id: int, *, force: bool = False) -> None:
        nonlocal current_frame_id
        if not force and frame_id == current_frame_id:
            return
        if frame_id not in gt_nodes:
            _create_frame_nodes(frame_id)
        if current_frame_id in gt_nodes:
            gt_nodes[current_frame_id].visible = False
        if current_frame_id in pred_nodes:
            pred_nodes[current_frame_id].visible = False
        if current_frame_id in comp_nodes:
            comp_nodes[current_frame_id].visible = False
        if current_frame_id in gt_smpl_nodes:
            gt_smpl_nodes[current_frame_id].visible = False
        if current_frame_id in pred_smpl_nodes:
            pred_smpl_nodes[current_frame_id].visible = False
        if current_frame_id in comp_smpl_nodes:
            comp_smpl_nodes[current_frame_id].visible = False

        current_frame_id = frame_id
        metrics_markdown.content = (
            f"**Eval pose type:** `{current_pose_type}`\n\n"
            + _metrics_table_markdown(
                baseline_metrics[frame_id],
                comp_metrics[frame_id],
                delta_metrics[frame_id],
                metric_names,
            )
        )

        if frame_id in gt_nodes:
            gt_nodes[frame_id].visible = gt_checkbox.value
        if frame_id in pred_nodes:
            pred_nodes[frame_id].visible = pred_checkbox.value
        if frame_id in comp_nodes:
            comp_nodes[frame_id].visible = comp_checkbox.value
        if frame_id in gt_smpl_nodes:
            gt_smpl_nodes[frame_id].visible = gt_smpl_checkbox.value  # type: ignore[name-defined]
        if frame_id in pred_smpl_nodes:
            pred_smpl_nodes[frame_id].visible = pred_smpl_checkbox.value  # type: ignore[name-defined]
        if frame_id in comp_smpl_nodes:
            comp_smpl_nodes[frame_id].visible = comp_smpl_checkbox.value  # type: ignore[name-defined]

    def _refresh_current() -> None:
        if current_frame_id in gt_nodes:
            gt_nodes[current_frame_id].visible = gt_checkbox.value
        if current_frame_id in pred_nodes:
            pred_nodes[current_frame_id].visible = pred_checkbox.value
        if current_frame_id in comp_nodes:
            comp_nodes[current_frame_id].visible = comp_checkbox.value
        if current_frame_id in gt_smpl_nodes:
            gt_smpl_nodes[current_frame_id].visible = gt_smpl_checkbox.value  # type: ignore[name-defined]
        if current_frame_id in pred_smpl_nodes:
            pred_smpl_nodes[current_frame_id].visible = pred_smpl_checkbox.value  # type: ignore[name-defined]
        if current_frame_id in comp_smpl_nodes:
            comp_smpl_nodes[current_frame_id].visible = comp_smpl_checkbox.value  # type: ignore[name-defined]

    # Initialize visibility
    gt_nodes[current_frame_id].visible = gt_checkbox.value
    pred_nodes[current_frame_id].visible = pred_checkbox.value
    comp_nodes[current_frame_id].visible = comp_checkbox.value
    if current_frame_id in gt_smpl_nodes:
        gt_smpl_nodes[current_frame_id].visible = gt_smpl_checkbox.value  # type: ignore[name-defined]
    if current_frame_id in pred_smpl_nodes:
        pred_smpl_nodes[current_frame_id].visible = pred_smpl_checkbox.value  # type: ignore[name-defined]
    if current_frame_id in comp_smpl_nodes:
        comp_smpl_nodes[current_frame_id].visible = comp_smpl_checkbox.value  # type: ignore[name-defined]

    def _set_pose_type(pose_type: str) -> None:
        nonlocal baseline_metrics, comp_metrics, delta_metrics, metric_names, sorted_frames, current_pose_type
        if pose_type == current_pose_type:
            return
        (
            baseline_metrics,
            comp_metrics,
            delta_metrics,
            metric_names,
            sorted_frames,
        ) = _load_comparison_metrics(pose_type)
        current_pose_type = pose_type
        frame_dropdown.options = [str(fid) for fid in sorted_frames]
        for frame_id in sorted_frames:
            _create_frame_nodes(frame_id)
        _apply_visibility(int(frame_dropdown.value), force=True)

    @frame_dropdown.on_update
    def _(_event=None) -> None:
        _apply_visibility(int(frame_dropdown.value))

    @pose_type_dropdown.on_update
    def _(_event=None) -> None:
        _set_pose_type(str(pose_type_dropdown.value))

    @gt_checkbox.on_update
    def _(_event=None) -> None:
        _refresh_current()

    @pred_checkbox.on_update
    def _(_event=None) -> None:
        _refresh_current()

    @comp_checkbox.on_update
    def _(_event=None) -> None:
        _refresh_current()

    if gt_smpl_root is not None:
        @gt_smpl_checkbox.on_update  # type: ignore[name-defined]
        def _(_event=None) -> None:
            _refresh_current()

    if pred_smpl_root is not None:
        @pred_smpl_checkbox.on_update  # type: ignore[name-defined]
        def _(_event=None) -> None:
            _refresh_current()

    if comp_smpl_root is not None:
        @comp_smpl_checkbox.on_update  # type: ignore[name-defined]
        def _(_event=None) -> None:
            _refresh_current()

    print(
        "Viser server running. "
        f"Comparison mode for GT {args.gt_scene_dir}, "
        f"baseline {args.pred_scene_dir}, comparison {args.comp_pred_scene_dir}."
    )

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


def main(args: Args) -> None:
    if args.comp_pred_scene_dir is not None:
        _visualize_comparison_mode(args)
        return
    gt_smplx_dir = args.gt_scene_dir / "smplx"
    pred_smplx_dir = args.pred_scene_dir / "smplx"
    comp_pred_smplx_dir = args.comp_pred_scene_dir / "smplx" if args.comp_pred_scene_dir else None
    gt_smpl_dir = args.gt_scene_dir / "smpl"
    pred_smpl_dir = args.pred_scene_dir / "smpl"
    comp_pred_smpl_dir = args.comp_pred_scene_dir / "smpl" if args.comp_pred_scene_dir else None

    if not gt_smplx_dir.exists():
        raise FileNotFoundError(f"GT smplx dir not found: {gt_smplx_dir}")
    if not pred_smplx_dir.exists():
        raise FileNotFoundError(f"Pred smplx dir not found: {pred_smplx_dir}")
    if comp_pred_smplx_dir is not None and not comp_pred_smplx_dir.exists():
        raise FileNotFoundError(f"Comparison pred smplx dir not found: {comp_pred_smplx_dir}")

    gt_frames = _sorted_frame_files(gt_smplx_dir, args.smplx_pattern)
    pred_frames = _sorted_frame_files(pred_smplx_dir, args.smplx_pattern)
    comp_pred_frames = (
        _sorted_frame_files(comp_pred_smplx_dir, args.smplx_pattern)
        if comp_pred_smplx_dir is not None
        else []
    )
    if not gt_frames:
        raise FileNotFoundError(f"No GT smplx files found in {gt_smplx_dir} with {args.smplx_pattern}")
    if not pred_frames:
        raise FileNotFoundError(
            f"No pred smplx files found in {pred_smplx_dir} with {args.smplx_pattern}"
        )
    if comp_pred_smplx_dir is not None and not comp_pred_frames:
        raise FileNotFoundError(
            f"No comparison pred smplx files found in {comp_pred_smplx_dir} with {args.smplx_pattern}"
        )

    gt_smpl_frames = (
        _sorted_frame_files(gt_smpl_dir, args.smpl_pattern) if gt_smpl_dir.exists() else []
    )
    pred_smpl_frames = (
        _sorted_frame_files(pred_smpl_dir, args.smpl_pattern) if pred_smpl_dir.exists() else []
    )
    comp_pred_smpl_frames = (
        _sorted_frame_files(comp_pred_smpl_dir, args.smpl_pattern)
        if comp_pred_smpl_dir is not None and comp_pred_smpl_dir.exists()
        else []
    )

    gt_frame_path = _select_frame(gt_frames, args.frame_index, args.frame_name)
    pred_frame_path = _select_frame(pred_frames, args.frame_index, args.frame_name)
    comp_pred_frame_path = (
        _select_frame(comp_pred_frames, args.frame_index, args.frame_name)
        if comp_pred_frames
        else None
    )
    gt_smpl_frame_path = (
        _select_frame(gt_smpl_frames, args.frame_index, args.frame_name) if gt_smpl_frames else None
    )
    pred_smpl_frame_path = (
        _select_frame(pred_smpl_frames, args.frame_index, args.frame_name) if pred_smpl_frames else None
    )
    comp_pred_smpl_frame_path = (
        _select_frame(comp_pred_smpl_frames, args.frame_index, args.frame_name)
        if comp_pred_smpl_frames
        else None
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    smplx_layer = _build_smplx_layer(args.model_folder, args.gender, args.smplx_model_ext, device)
    smpl_layer = (
        _build_smpl_layer(args.model_folder, args.gender, args.smpl_model_ext, device)
        if gt_smpl_frame_path is not None
        or pred_smpl_frame_path is not None
        or comp_pred_smpl_frame_path is not None
        else None
    )
    faces = np.asarray(smplx_layer.faces, dtype=np.int32)
    smpl_faces = np.asarray(smpl_layer.faces, dtype=np.int32) if smpl_layer is not None else None

    gt_data = _load_npz(gt_frame_path)
    pred_data = _load_npz(pred_frame_path)
    comp_pred_data = _load_npz(comp_pred_frame_path) if comp_pred_frame_path is not None else None
    if gt_data is None:
        raise FileNotFoundError(f"Failed to load GT smplx params: {gt_frame_path}")
    if pred_data is None:
        raise FileNotFoundError(f"Failed to load pred smplx params: {pred_frame_path}")
    if comp_pred_frame_path is not None and comp_pred_data is None:
        raise FileNotFoundError(f"Failed to load comparison pred smplx params: {comp_pred_frame_path}")

    gt_verts = _smplx_vertices(smplx_layer, gt_data, device)
    pred_verts = _smplx_vertices(smplx_layer, pred_data, device)
    comp_pred_verts = (
        _smplx_vertices(smplx_layer, comp_pred_data, device)
        if comp_pred_data is not None
        else None
    )

    gt_smpl_verts = None
    pred_smpl_verts = None
    comp_pred_smpl_verts = None
    if smpl_layer is not None and smpl_faces is not None:
        if gt_smpl_frame_path is not None:
            gt_smpl_data = _load_npz(gt_smpl_frame_path)
            if gt_smpl_data is not None:
                gt_smpl_verts = _smpl_vertices(smpl_layer, gt_smpl_data, device)
        if pred_smpl_frame_path is not None:
            pred_smpl_data = _load_npz(pred_smpl_frame_path)
            if pred_smpl_data is not None:
                pred_smpl_verts = _smpl_vertices(smpl_layer, pred_smpl_data, device)
        if comp_pred_smpl_frame_path is not None:
            comp_pred_smpl_data = _load_npz(comp_pred_smpl_frame_path)
            if comp_pred_smpl_data is not None:
                comp_pred_smpl_verts = _smpl_vertices(smpl_layer, comp_pred_smpl_data, device)

    if (
        gt_verts is None
        and pred_verts is None
        and comp_pred_verts is None
        and gt_smpl_verts is None
        and pred_smpl_verts is None
        and comp_pred_smpl_verts is None
    ):
        raise FileNotFoundError("No SMPL-X vertices could be computed for the selected frame.")

    bounds: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
    if gt_verts is not None:
        bounds = _update_bounds(gt_verts, bounds)
    if pred_verts is not None:
        bounds = _update_bounds(pred_verts, bounds)
    if comp_pred_verts is not None:
        bounds = _update_bounds(comp_pred_verts, bounds)
    if gt_smpl_verts is not None:
        bounds = _update_bounds(gt_smpl_verts, bounds)
    if pred_smpl_verts is not None:
        bounds = _update_bounds(pred_smpl_verts, bounds)
    if comp_pred_smpl_verts is not None:
        bounds = _update_bounds(comp_pred_smpl_verts, bounds)

    center_offset = np.zeros(3, dtype=np.float32)
    if args.center_scene and bounds[0] is not None and bounds[1] is not None:
        center_offset = (bounds[0] + bounds[1]) * 0.5

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_width="large")
    angle = -np.pi / 2 if args.is_minus_y_up else np.pi / 2
    R_fix = tf.SO3.from_x_radians(angle)
    server.scene.add_frame(
        "/scene",
        show_axes=False,
        wxyz=tuple(R_fix.wxyz),
        position=tuple((-R_fix.apply(center_offset)).tolist()),
    )

    gt_root = server.scene.add_frame("/scene/gt_smplx", show_axes=False)
    pred_root = server.scene.add_frame("/scene/pred_smplx", show_axes=False)
    comp_pred_root = (
        server.scene.add_frame("/scene/comp_pred_smplx", show_axes=False)
        if comp_pred_verts is not None
        else None
    )
    gt_smpl_root = server.scene.add_frame("/scene/gt_smpl", show_axes=False) if gt_smpl_verts is not None else None
    pred_smpl_root = (
        server.scene.add_frame("/scene/pred_smpl", show_axes=False) if pred_smpl_verts is not None else None
    )
    comp_pred_smpl_root = (
        server.scene.add_frame("/scene/comp_pred_smpl", show_axes=False)
        if comp_pred_smpl_verts is not None
        else None
    )

    gt_base = np.array([70, 130, 255], dtype=np.float32)
    pred_base = np.array([255, 140, 70], dtype=np.float32)
    comp_pred_base = np.array([140, 200, 90], dtype=np.float32)
    gt_smpl_base = np.array([60, 200, 200], dtype=np.float32)
    pred_smpl_base = np.array([220, 80, 80], dtype=np.float32)
    comp_pred_smpl_base = np.array([200, 140, 40], dtype=np.float32)

    if gt_verts is not None:
        for pid in range(gt_verts.shape[0]):
            color = _color_for_person(gt_base, pid)
            handle = server.scene.add_mesh_simple(
                f"/scene/gt_smplx/person_{pid}",
                vertices=gt_verts[pid],
                faces=faces,
                color=color,
            )
            if hasattr(handle, "opacity"):
                handle.opacity = float(args.mesh_opacity)

    if pred_verts is not None:
        for pid in range(pred_verts.shape[0]):
            color = _color_for_person(pred_base, pid)
            handle = server.scene.add_mesh_simple(
                f"/scene/pred_smplx/person_{pid}",
                vertices=pred_verts[pid],
                faces=faces,
                color=color,
            )
            if hasattr(handle, "opacity"):
                handle.opacity = float(args.mesh_opacity)

    if comp_pred_verts is not None:
        for pid in range(comp_pred_verts.shape[0]):
            color = _color_for_person(comp_pred_base, pid)
            handle = server.scene.add_mesh_simple(
                f"/scene/comp_pred_smplx/person_{pid}",
                vertices=comp_pred_verts[pid],
                faces=faces,
                color=color,
            )
            if hasattr(handle, "opacity"):
                handle.opacity = float(args.mesh_opacity)

    if gt_smpl_verts is not None and smpl_faces is not None:
        for pid in range(gt_smpl_verts.shape[0]):
            color = _color_for_person(gt_smpl_base, pid)
            handle = server.scene.add_mesh_simple(
                f"/scene/gt_smpl/person_{pid}",
                vertices=gt_smpl_verts[pid],
                faces=smpl_faces,
                color=color,
            )
            if hasattr(handle, "opacity"):
                handle.opacity = float(args.mesh_opacity)

    if pred_smpl_verts is not None and smpl_faces is not None:
        for pid in range(pred_smpl_verts.shape[0]):
            color = _color_for_person(pred_smpl_base, pid)
            handle = server.scene.add_mesh_simple(
                f"/scene/pred_smpl/person_{pid}",
                vertices=pred_smpl_verts[pid],
                faces=smpl_faces,
                color=color,
            )
            if hasattr(handle, "opacity"):
                handle.opacity = float(args.mesh_opacity)

    if comp_pred_smpl_verts is not None and smpl_faces is not None:
        for pid in range(comp_pred_smpl_verts.shape[0]):
            color = _color_for_person(comp_pred_smpl_base, pid)
            handle = server.scene.add_mesh_simple(
                f"/scene/comp_pred_smpl/person_{pid}",
                vertices=comp_pred_smpl_verts[pid],
                faces=smpl_faces,
                color=color,
            )
            if hasattr(handle, "opacity"):
                handle.opacity = float(args.mesh_opacity)

    with server.gui.add_folder("GT Scene dir"):
        gt_checkbox = server.gui.add_checkbox("Show SMPL-X", True)

        @gt_checkbox.on_update
        def _(_event=None, checkbox=gt_checkbox, handle=gt_root) -> None:
            handle.visible = bool(checkbox.value)

        if gt_smpl_root is not None:
            gt_smpl_checkbox = server.gui.add_checkbox("Show SMPL", True)

            @gt_smpl_checkbox.on_update
            def _(_event=None, checkbox=gt_smpl_checkbox, handle=gt_smpl_root) -> None:
                handle.visible = bool(checkbox.value)

    with server.gui.add_folder("Prediction Scene Dir"):
        pred_checkbox = server.gui.add_checkbox("Show SMPL-X", True)

        @pred_checkbox.on_update
        def _(_event=None, checkbox=pred_checkbox, handle=pred_root) -> None:
            handle.visible = bool(checkbox.value)

        if pred_smpl_root is not None:
            pred_smpl_checkbox = server.gui.add_checkbox("Show SMPL", True)

            @pred_smpl_checkbox.on_update
            def _(_event=None, checkbox=pred_smpl_checkbox, handle=pred_smpl_root) -> None:
                handle.visible = bool(checkbox.value)

    if comp_pred_root is not None or comp_pred_smpl_root is not None:
        with server.gui.add_folder("Comparison Pred Scene Dir"):
            if comp_pred_root is not None:
                comp_pred_checkbox = server.gui.add_checkbox("Show SMPL-X", True)

                @comp_pred_checkbox.on_update
                def _(_event=None, checkbox=comp_pred_checkbox, handle=comp_pred_root) -> None:
                    handle.visible = bool(checkbox.value)

            if comp_pred_smpl_root is not None:
                comp_pred_smpl_checkbox = server.gui.add_checkbox("Show SMPL", True)

                @comp_pred_smpl_checkbox.on_update
                def _(_event=None, checkbox=comp_pred_smpl_checkbox, handle=comp_pred_smpl_root) -> None:
                    handle.visible = bool(checkbox.value)
    frame_desc = args.frame_name if args.frame_name is not None else str(args.frame_index)
    print(
        "Viser server running. "
        f"Showing frame {frame_desc} from GT {args.gt_scene_dir} and Pred {args.pred_scene_dir}"
        f"{' and Comp Pred ' + str(args.comp_pred_scene_dir) if args.comp_pred_scene_dir else ''}."
    )

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main(tyro.cli(Args))
