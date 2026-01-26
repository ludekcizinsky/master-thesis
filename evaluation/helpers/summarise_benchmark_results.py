from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Literal

import pandas as pd
import tyro

@dataclass
class Args:
    exp_name: str
    epoch_str: str
    scene_names: List[str]
    src_cam_ids: List[int] = field(default_factory=list)
    include_segmentation_novel_cams: bool = False
    results_root: Path = Path("/scratch/izar/cizinsky/thesis/results")
    output_dir: Path = Path("/home/cizinsky/master-thesis/results")


@dataclass
class TaskSpec:
    name: str
    title: str
    kind: Literal["kv", "csv_camera"]
    filename: str
    columns: Sequence[Tuple[str, str]]
    avg_formats: Dict[str, str] = field(default_factory=dict)

@dataclass
class OursSpec:
    label: str
    task: TaskSpec
    task_metric_map: Dict[str, str]
    extra_values: Dict[str, object] = field(default_factory=dict)

@dataclass
class BaselineSpec:
    name: str
    title: str
    description: str
    baseline_filename: str
    task: TaskSpec
    metric_order: Sequence[str]
    metric_headers: Dict[str, str]
    best_directions: Dict[str, Literal["min", "max"]]
    formats: Dict[str, str]
    task_metric_map: Dict[str, str]
    extra_columns: Sequence[str] = field(default_factory=list)
    extra_headers: Dict[str, str] = field(default_factory=dict)
    baseline_defaults: Dict[str, object] = field(default_factory=dict)
    ours_specs: Sequence[OursSpec] = field(default_factory=list)

DATASET_LABELS = {"hi4d": "Hi4D", "mmm": "MMM"}


def _parse_metrics(results_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if not value:
                continue
            metrics[key] = float(value)
    return metrics


def _format_table(df: pd.DataFrame, avg_formats: Dict[str, str]) -> pd.DataFrame:
    records = []
    for scene, row in df.iterrows():
        record: Dict[str, str] = {"scene": str(scene)}
        for col in df.columns:
            value = row[col]
            if scene == "avg":
                fmt = avg_formats.get(col, ".4f")
                record[col] = f"{value:{fmt}}"
            else:
                record[col] = f"{value:.4f}"
        records.append(record)
    return pd.DataFrame(records).set_index("scene")


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = ["scene", *df.columns]
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for scene, row in df.iterrows():
        values: Iterable[str] = [str(scene), *[str(row[col]) for col in df.columns]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _results_path(args: Args, scene: str, task: TaskSpec) -> Path:
    return (
        args.results_root
        / scene
        / "evaluation"
        / args.exp_name
        / f"epoch_{args.epoch_str}"
        / task.filename
    )


def _collect_task_results_numeric(
    args: Args,
    task: TaskSpec,
    scene_names: Sequence[str],
    src_cam_ids: Sequence[int],
) -> pd.DataFrame:
    if task.kind == "csv_camera" and len(src_cam_ids) != len(scene_names):
        raise ValueError(
            f"--src-cam-ids must have the same length as --scene-names "
            f"({len(src_cam_ids)} vs {len(scene_names)})."
        )

    rows = []
    for idx, scene in enumerate(scene_names):
        results_path = _results_path(args, scene, task)
        if task.kind == "kv":
            metrics = _parse_metrics(results_path)
        else:
            src_cam_id = int(src_cam_ids[idx])
            df = pd.read_csv(results_path)
            if "camera_id" not in df.columns:
                raise ValueError(
                    f"Expected a 'camera_id' column in {results_path}, got {list(df.columns)}"
                )
            row_match = df.loc[df["camera_id"] == src_cam_id]
            if row_match.empty:
                raise ValueError(
                    f"No row found for camera_id={src_cam_id} in {results_path}"
                )
            metrics = row_match.iloc[0].to_dict()

        row: Dict[str, float] = {"scene": scene}
        for column, metric_key in task.columns:
            row[column] = metrics[metric_key]
        rows.append(row)

    column_names = [column for column, _ in task.columns]
    df = pd.DataFrame(rows).set_index("scene")[column_names]
    df.loc["avg"] = df.mean(numeric_only=True)
    return df


def _collect_task_results(
    args: Args,
    task: TaskSpec,
    scene_names: Sequence[str],
    src_cam_ids: Sequence[int],
) -> pd.DataFrame:
    df = _collect_task_results_numeric(args, task, scene_names, src_cam_ids)
    return _format_table(df, avg_formats=task.avg_formats)


def _scene_dataset(scene: str) -> str:
    return scene.split("_", 1)[0]

def _dataset_label(ds_name: str) -> str:
    return DATASET_LABELS.get(ds_name, ds_name)

def _load_baseline_table(
    baseline_path: Path,
    required_columns: Sequence[str],
    name: str,
    defaults: Dict[str, object],
) -> pd.DataFrame:
    if not baseline_path.is_file():
        print(f"Warning: {name} baseline file not found at {baseline_path}")
        return pd.DataFrame(columns=required_columns)
    df = pd.read_csv(baseline_path)
    missing = [col for col in required_columns if col not in df.columns]
    fillable = [col for col in missing if col in defaults]
    for col in fillable:
        df[col] = defaults[col]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{name} baseline file is missing columns: {', '.join(missing)}"
        )
    return df[list(required_columns)]


def _format_baseline_value(value: float, fmt: str) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:{fmt}}"


def _baseline_markdown_table(df: pd.DataFrame, spec: BaselineSpec) -> str:
    best_values: Dict[str, float] = {}
    for metric, direction in spec.best_directions.items():
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            best_values[metric] = float("nan")
        else:
            best_values[metric] = series.max() if direction == "max" else series.min()

    headers = [
        "Method",
        *[spec.extra_headers.get(col, col) for col in spec.extra_columns],
        *[spec.metric_headers[m] for m in spec.metric_order],
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        values: List[str] = [str(row["method"])]
        for col in spec.extra_columns:
            values.append(str(row.get(col, "")))
        for metric in spec.metric_order:
            value = pd.to_numeric(row.get(metric), errors="coerce")
            display = _format_baseline_value(value, spec.formats[metric])
            best = best_values.get(metric)
            if pd.notna(value) and pd.notna(best) and abs(float(value) - float(best)) <= 1e-9:
                display = f"**{display}**"
            values.append(display)
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _baseline_dataset_order(
    scene_indices_by_ds: Dict[str, List[int]], baseline_df: pd.DataFrame
) -> List[str]:
    ordered: List[str] = []
    for ds_name in scene_indices_by_ds:
        label = _dataset_label(ds_name)
        if label not in ordered:
            ordered.append(label)
    if "dataset" in baseline_df.columns:
        for label in baseline_df["dataset"].dropna().tolist():
            if label not in ordered:
                ordered.append(label)
    return ordered


def _ours_row_for_dataset(
    args: Args,
    spec: BaselineSpec,
    ours_spec: OursSpec,
    dataset_label: str,
    label_to_scene_key: Dict[str, str],
    scene_indices_by_ds: Dict[str, List[int]],
) -> Dict[str, object]:
    ours_row: Dict[str, object] = {
        "dataset": dataset_label,
        "method": ours_spec.label,
    }
    for col in spec.extra_columns:
        ours_row[col] = ours_spec.extra_values.get(col, spec.baseline_defaults.get(col))
    ds_name = label_to_scene_key.get(dataset_label)
    if ds_name is None:
        print(
            f"Warning: no scenes listed for dataset {dataset_label} when "
            f"building {spec.name} baselines."
        )
        for metric in spec.metric_order:
            ours_row[metric] = float("nan")
        return ours_row

    indices = scene_indices_by_ds[ds_name]
    dataset_scenes = [args.scene_names[i] for i in indices]
    missing = [
        scene
        for scene in dataset_scenes
        if not _results_path(args, scene, spec.task).is_file()
    ]

    if missing:
        print(
            f"Warning: could not find {spec.name} results for {dataset_label}: "
            f"{', '.join(missing)}"
        )
        for metric in spec.metric_order:
            ours_row[metric] = float("nan")
        return ours_row

    if ours_spec.task.kind == "csv_camera":
        dataset_src_cam_ids = (
            [args.src_cam_ids[i] for i in indices] if args.src_cam_ids else []
        )
    else:
        dataset_src_cam_ids = []

    numeric_df = _collect_task_results_numeric(
        args, ours_spec.task, dataset_scenes, dataset_src_cam_ids
    )
    avg = numeric_df.loc["avg"]
    for metric in spec.metric_order:
        source_col = ours_spec.task_metric_map.get(metric)
        if source_col is None:
            ours_row[metric] = float("nan")
        else:
            ours_row[metric] = float(avg[source_col])
    return ours_row


def _baseline_sections_for_spec(
    args: Args,
    spec: BaselineSpec,
    scene_indices_by_ds: Dict[str, List[int]],
) -> List[str]:
    baseline_dir = Path(__file__).resolve().parents[1] / "baselines"
    required_columns = [
        "dataset",
        "method",
        *spec.extra_columns,
        *spec.metric_order,
    ]
    baseline_path = baseline_dir / spec.baseline_filename
    baseline_df = _load_baseline_table(
        baseline_path, required_columns, spec.name, spec.baseline_defaults
    )
    ours_specs = (
        list(spec.ours_specs)
        if spec.ours_specs
        else [OursSpec("Ours", spec.task, spec.task_metric_map)]
    )

    label_to_scene_key = {
        _dataset_label(ds_name): ds_name for ds_name in scene_indices_by_ds
    }
    dataset_order = _baseline_dataset_order(scene_indices_by_ds, baseline_df)

    sections: List[str] = []
    for dataset_label in dataset_order:
        ours_rows = [
            _ours_row_for_dataset(
                args,
                spec,
                ours_spec,
                dataset_label,
                label_to_scene_key,
                scene_indices_by_ds,
            )
            for ours_spec in ours_specs
        ]

        baseline_rows = baseline_df.loc[
            baseline_df["dataset"] == dataset_label
        ].copy()
        if baseline_rows.empty:
            print(
                f"Warning: no {spec.name} baselines found for dataset {dataset_label} "
                f"in {baseline_path}"
            )

        combined_records = baseline_rows.to_dict("records")
        combined_records.extend(ours_rows)
        combined_df = pd.DataFrame(combined_records, columns=required_columns)
        for metric in spec.metric_order:
            combined_df[metric] = pd.to_numeric(combined_df[metric], errors="coerce")

        table_md = _baseline_markdown_table(combined_df, spec)
        sections.append(f"#### {dataset_label}\n{table_md}")

    if not sections:
        print(f"Warning: no datasets available to build {spec.name} baselines table.")

    return sections


def _write_baselines_markdown(
    args: Args,
    specs: Sequence[BaselineSpec],
    scene_indices_by_ds: Dict[str, List[int]],
    out_dir: Path,
    filename: str,
) -> None:
    sections: List[str] = []
    for spec in specs:
        task_sections = _baseline_sections_for_spec(args, spec, scene_indices_by_ds)
        if not task_sections:
            continue
        sections.append(
            f"### {spec.title} (Baselines + Ours)\n"
            f"{spec.description}\n\n"
            + "\n\n".join(task_sections)
        )

    if not sections:
        print(f"Warning: no baseline sections generated for {filename}.")
        return

    md_path = out_dir / filename
    md_path.write_text("\n\n".join(sections), encoding="utf-8")
    print(f"Wrote {md_path}")

def main(args: Args) -> None:
    tasks = [
        TaskSpec(
            name="nvs",
            title="NVS",
            kind="kv",
            filename="novel_view_results.txt",
            columns=[("SSIM", "ssim"), ("PSNR", "psnr"), ("LPIPS", "lpips")],
            avg_formats={"SSIM": ".3f", "PSNR": ".1f", "LPIPS": ".4f"},
        ),
        TaskSpec(
            name="pose_estimation_smpl",
            title="Pose Estimation (SMPL)",
            kind="kv",
            filename="smpl_pose_estimation_overall_results.txt",
            columns=[
                ("MPJPE_mm", "mpjpe_mm"),
                ("MVE_mm", "mve_mm"),
                ("CD_mm", "cd_mm"),
                ("PCDR", "pcdr"),
            ],
        ),
        TaskSpec(
            name="pose_estimation_smplx",
            title="Pose Estimation (SMPL-X)",
            kind="kv",
            filename="smplx_pose_estimation_overall_results.txt",
            columns=[
                ("MPJPE_mm", "mpjpe_mm"),
                ("MVE_mm", "mve_mm"),
                ("PCDR", "pcdr"),
            ],
        ),
        TaskSpec(
            name="reconstruction",
            title="Reconstruction",
            kind="kv",
            filename="reconstruction_overall_results.txt",
            columns=[
                ("V_IoU", "v_iou"),
                ("Chamfer_cm", "chamfer_cm"),
                ("P2S_cm", "p2s_cm"),
                ("Normal_Consistency", "normal_consistency"),
            ],
        ),
    ]

    if args.include_segmentation_novel_cams:
        tasks.append(
            TaskSpec(
                name="segmentation",
                title="Segmentation",
                kind="kv",
                filename="segmentation_overall_results.txt",
                columns=[("IoU", "iou"), ("Recall", "recall"), ("F1", "f1")],
            )
        )

    if args.src_cam_ids:
        tasks.append(
            TaskSpec(
                name="segmentation_src_cam",
                title="Segmentation (Src Cam)",
                kind="csv_camera",
                filename="segmentation_metrics_avg_per_camera.csv",
                columns=[("IoU", "iou"), ("Recall", "recall"), ("F1", "f1")],
            )
        )

    if any(task.kind == "csv_camera" for task in tasks):
        if not args.src_cam_ids:
            raise ValueError(
                "CSV camera tasks require --src-cam-ids (one per scene)."
            )
        if len(args.src_cam_ids) != len(args.scene_names):
            raise ValueError(
                f"--src-cam-ids must have the same length as --scene-names "
                f"({len(args.src_cam_ids)} vs {len(args.scene_names)})."
            )

    task_by_name = {task.name: task for task in tasks}
    segmentation_task = task_by_name.get("segmentation_src_cam")
    if segmentation_task is None:
        segmentation_task = task_by_name.get(
            "segmentation",
            TaskSpec(
                name="segmentation",
                title="Segmentation",
                kind="kv",
                filename="segmentation_overall_results.txt",
                columns=[("IoU", "iou"), ("Recall", "recall"), ("F1", "f1")],
            ),
        )
        if args.src_cam_ids:
            print(
                "Warning: segmentation source-view metrics not available; "
                "falling back to segmentation_overall_results.txt."
            )

    reconstruction_description = (
        "We consider the following metrics for human mesh reconstruction evaluation: "
        "volumetric IoU (V-IoU), Chamfer distance (C−l2) [cm], "
        "point-to-surface distance (P2S) [cm], and normal consistency (NC). "
        "We highlight the best results for each dataset and metric in **bold**."
    )
    nvs_description = (
        "Rendering quality is measured via three metrics: PSNR, SSIM, and LPIPS. "
        "We highlight the best results for each dataset and metric in **bold**."
    )
    pose_description = (
        "We assess human pose estimation using MPJPE [mm], MVE [mm], Contact Distance "
        "(CD) [mm], and Percentage of Correct Depth Relations (PCDR) with a threshold "
        "of 0.15m. Baselines are reported in SMPL space. SMPL-X does not include CD, "
        "which is shown as N/A. We highlight the best results for each dataset and "
        "metric in **bold**."
    )
    segmentation_description = (
        "We report IoU, Recall, and F1 score for human instance segmentation, and "
        "highlight the best results for each dataset and metric in **bold**."
    )

    baseline_specs = [
        BaselineSpec(
            name="nvs",
            title="NVS",
            description=nvs_description,
            baseline_filename="nvs.csv",
            task=task_by_name["nvs"],
            metric_order=["ssim", "psnr", "lpips"],
            metric_headers={
                "ssim": "SSIM ↑",
                "psnr": "PSNR ↑",
                "lpips": "LPIPS ↓",
            },
            best_directions={"ssim": "max", "psnr": "max", "lpips": "min"},
            formats={"ssim": ".3f", "psnr": ".1f", "lpips": ".4f"},
            task_metric_map={"ssim": "SSIM", "psnr": "PSNR", "lpips": "LPIPS"},
        ),
        BaselineSpec(
            name="pose",
            title="Pose Estimation",
            description=pose_description,
            baseline_filename="pose.csv",
            task=task_by_name["pose_estimation_smpl"],
            metric_order=["mpjpe", "mve", "cd", "pcdr"],
            metric_headers={
                "mpjpe": "MPJPE ↓",
                "mve": "MVE ↓",
                "cd": "CD ↓",
                "pcdr": "PCDR ↑",
            },
            best_directions={
                "mpjpe": "min",
                "mve": "min",
                "cd": "min",
                "pcdr": "max",
            },
            formats={"mpjpe": ".1f", "mve": ".1f", "cd": ".1f", "pcdr": ".3f"},
            task_metric_map={
                "mpjpe": "MPJPE_mm",
                "mve": "MVE_mm",
                "cd": "CD_mm",
                "pcdr": "PCDR",
            },
            extra_columns=["space"],
            extra_headers={"space": "Space"},
            baseline_defaults={"space": "SMPL"},
            ours_specs=[
                OursSpec(
                    label="Ours (SMPL)",
                    task=task_by_name["pose_estimation_smpl"],
                    task_metric_map={
                        "mpjpe": "MPJPE_mm",
                        "mve": "MVE_mm",
                        "cd": "CD_mm",
                        "pcdr": "PCDR",
                    },
                    extra_values={"space": "SMPL"},
                ),
                OursSpec(
                    label="Ours (SMPL-X)",
                    task=task_by_name["pose_estimation_smplx"],
                    task_metric_map={
                        "mpjpe": "MPJPE_mm",
                        "mve": "MVE_mm",
                        "pcdr": "PCDR",
                    },
                    extra_values={"space": "SMPL-X"},
                ),
            ],
        ),
        BaselineSpec(
            name="reconstruction",
            title="Reconstruction",
            description=reconstruction_description,
            baseline_filename="reconstruction.csv",
            task=task_by_name["reconstruction"],
            metric_order=["v-iou", "c-l2", "p2s", "nc"],
            metric_headers={
                "v-iou": "V-IoU ↑",
                "c-l2": "C-l2 ↓",
                "p2s": "P2S ↓",
                "nc": "NC ↑",
            },
            best_directions={"v-iou": "max", "c-l2": "min", "p2s": "min", "nc": "max"},
            formats={"v-iou": ".3f", "c-l2": ".2f", "p2s": ".2f", "nc": ".3f"},
            task_metric_map={
                "v-iou": "V_IoU",
                "c-l2": "Chamfer_cm",
                "p2s": "P2S_cm",
                "nc": "Normal_Consistency",
            },
        ),
        BaselineSpec(
            name="segmentation",
            title="Segmentation",
            description=segmentation_description,
            baseline_filename="segmentation.csv",
            task=segmentation_task,
            metric_order=["iou", "recall", "f1"],
            metric_headers={"iou": "IoU ↑", "recall": "Recall ↑", "f1": "F1 ↑"},
            best_directions={"iou": "max", "recall": "max", "f1": "max"},
            formats={"iou": ".3f", "recall": ".3f", "f1": ".3f"},
            task_metric_map={"iou": "IoU", "recall": "Recall", "f1": "F1"},
        ),
    ]

    scene_indices_by_ds: Dict[str, List[int]] = {}
    for idx, scene in enumerate(args.scene_names):
        ds_name = _scene_dataset(scene)
        scene_indices_by_ds.setdefault(ds_name, []).append(idx)

    out_dir = args.output_dir / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_sections = []
    for task in tasks:
        task_sections = []
        for ds_name, indices in scene_indices_by_ds.items():
            dataset_scenes = [args.scene_names[i] for i in indices]
            dataset_src_cam_ids = (
                [args.src_cam_ids[i] for i in indices] if args.src_cam_ids else []
            )
            missing = [
                scene
                for scene in dataset_scenes
                if not _results_path(args, scene, task).is_file()
            ]
            if missing:
                print(f"Skipping task {task.name} for dataset {ds_name}")
                continue
            formatted_df = _collect_task_results(
                args, task, dataset_scenes, dataset_src_cam_ids
            )
            csv_path = out_dir / f"{ds_name}_{task.name}.csv"
            formatted_df.to_csv(csv_path)
            md_path = out_dir / f"{ds_name}_{task.name}.md"
            md_path.write_text(_to_markdown_table(formatted_df), encoding="utf-8")
            task_sections.append(
                f"#### {ds_name}\n{_to_markdown_table(formatted_df)}"
            )
        if task_sections:
            combined_sections.append(f"### {task.title}\n" + "\n\n".join(task_sections))

    combined_path = out_dir / "all_results.md"
    combined_path.write_text("\n\n".join(combined_sections), encoding="utf-8")

    _write_baselines_markdown(
        args, baseline_specs, scene_indices_by_ds, out_dir, "baselines_with_ours.md"
    )

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
