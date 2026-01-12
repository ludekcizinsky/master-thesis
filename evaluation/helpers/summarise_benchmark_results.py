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

RECON_BASELINE_COLUMNS = ("dataset", "method", "v-iou", "c-l2", "p2s", "nc")
RECON_METRIC_MAP = {
    "v-iou": "V_IoU",
    "c-l2": "Chamfer_cm",
    "p2s": "P2S_cm",
    "nc": "Normal_Consistency",
}
RECON_METRIC_ORDER = ["v-iou", "c-l2", "p2s", "nc"]
RECON_BEST_DIRECTIONS = {"v-iou": "max", "c-l2": "min", "p2s": "min", "nc": "max"}
RECON_FORMATS = {"v-iou": ".3f", "c-l2": ".2f", "p2s": ".2f", "nc": ".3f"}
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

def _load_reconstruction_baselines(path: Path) -> pd.DataFrame:
    if not path.is_file():
        print(f"Warning: reconstruction baseline file not found at {path}")
        return pd.DataFrame(columns=RECON_BASELINE_COLUMNS)
    df = pd.read_csv(path)
    missing = [col for col in RECON_BASELINE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Reconstruction baseline file is missing columns: {', '.join(missing)}"
        )
    return df[list(RECON_BASELINE_COLUMNS)]


def _format_reconstruction_value(value: float, metric: str) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:{RECON_FORMATS[metric]}}"


def _reconstruction_markdown_table(df: pd.DataFrame) -> str:
    best_values: Dict[str, float] = {}
    for metric, direction in RECON_BEST_DIRECTIONS.items():
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            best_values[metric] = float("nan")
        else:
            best_values[metric] = series.max() if direction == "max" else series.min()

    headers = ["Method", "V-IoU ↑", "C-l2 ↓", "P2S ↓", "NC ↑"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        values: List[str] = [str(row["method"])]
        for metric in RECON_METRIC_ORDER:
            value = row.get(metric)
            display = _format_reconstruction_value(value, metric)
            best = best_values.get(metric)
            if pd.notna(value) and pd.notna(best) and abs(float(value) - float(best)) <= 1e-9:
                display = f"**{display}**"
            values.append(display)
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _write_reconstruction_with_baselines(
    args: Args,
    task: TaskSpec,
    scene_indices_by_ds: Dict[str, List[int]],
    out_dir: Path,
) -> None:
    baseline_path = Path(__file__).resolve().parents[1] / "baselines" / "reconstruction.csv"
    baseline_df = _load_reconstruction_baselines(baseline_path)
    sections = []

    for ds_name, indices in scene_indices_by_ds.items():
        dataset_label = _dataset_label(ds_name)
        dataset_scenes = [args.scene_names[i] for i in indices]
        missing = [
            scene
            for scene in dataset_scenes
            if not _results_path(args, scene, task).is_file()
        ]

        ours_row: Dict[str, object] = {"dataset": dataset_label, "method": "Ours"}
        if missing:
            print(
                f"Warning: could not find reconstruction results for {dataset_label}: "
                f"{', '.join(missing)}"
            )
            for metric in RECON_METRIC_ORDER:
                ours_row[metric] = float("nan")
        else:
            numeric_df = _collect_task_results_numeric(
                args, task, dataset_scenes, src_cam_ids=[]
            )
            avg = numeric_df.loc["avg"]
            for metric, source_col in RECON_METRIC_MAP.items():
                ours_row[metric] = float(avg[source_col])

        baseline_rows = baseline_df.loc[baseline_df["dataset"] == dataset_label].copy()
        if baseline_rows.empty:
            print(
                f"Warning: no reconstruction baselines found for dataset {dataset_label} "
                f"in {baseline_path}"
            )

        combined_records = baseline_rows.to_dict("records")
        combined_records.append(ours_row)
        combined_df = pd.DataFrame(combined_records, columns=RECON_BASELINE_COLUMNS)
        for metric in RECON_METRIC_ORDER:
            combined_df[metric] = pd.to_numeric(combined_df[metric], errors="coerce")

        table_md = _reconstruction_markdown_table(combined_df)
        sections.append(f"#### {dataset_label}\n{table_md}")

    if not sections:
        print("Warning: no datasets available to build reconstruction baselines table.")
        return

    md_path = out_dir / "reconstruction_with_baselines.md"
    description = (
        "We consider the following metrics for human mesh reconstruction evaluation: "
        "volumetric IoU (V-IoU), Chamfer distance (C−l2) [cm], "
        "point-to-surface distance (P2S) [cm], and normal consistency (NC). "
        "We highlight the best results for each dataset and metric in **bold**."
    )
    md_path.write_text(
        "### Reconstruction (Baselines + Ours)\n"
        f"{description}\n\n"
        + "\n\n".join(sections),
        encoding="utf-8",
    )
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
            name="pose_estimation",
            title="Pose Estimation",
            kind="kv",
            filename="pose_estimation_overall_results.txt",
            columns=[
                ("MPJPE_mm", "mpjpe_mm"),
                ("MVE_mm", "mve_mm"),
                ("CD_mm", "cd_mm"),
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
            print(f"Wrote {csv_path}")
            print(f"Wrote {md_path}")
        if task_sections:
            combined_sections.append(f"### {task.title}\n" + "\n\n".join(task_sections))

    combined_path = out_dir / "all_results.md"
    combined_path.write_text("\n\n".join(combined_sections), encoding="utf-8")
    print(f"Wrote {combined_path}")

    reconstruction_task = next(
        (task for task in tasks if task.name == "reconstruction"), None
    )
    if reconstruction_task is not None:
        _write_reconstruction_with_baselines(
            args, reconstruction_task, scene_indices_by_ds, out_dir
        )

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
