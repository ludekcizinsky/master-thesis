# Qualitative Output Schema Manifest

This document defines the target output layout for one experiment.

Root:

- `<output_dir>/<exp_name>/`

The schema is task-oriented and intended to make metric inputs/outputs easy to inspect and visualize.

## Version

- `output_layout_version: v1`
- This schema is a clean-break target. Backward compatibility is out of scope.

## Top-Level Structure

```text
<output_dir>/<exp_name>/
├── manifest.json
├── training_artifacts/
├── input_data/
└── evaluation/
```

## Top-Level Folders

### 1) `training_artifacts/`

Replaces old `debug/`.

Purpose:

- Training-time diagnostics and loss-input visual checks.
- Epoch snapshots used for optimization debugging (RGB/depth/mask overlays, etc.).

Example:

```text
training_artifacts/
└── epoch_0002/
    ├── rgb/
    ├── depth/
    └── gt_input/
```

### 2) `input_data/`

Purpose:

- Exact data used by training/evaluation.
- Keeps training/eval inputs reproducible.

Structure:

```text
input_data/
├── train/
│   ├── images/
│   ├── seg/
│   ├── all_cameras/
│   ├── smplx/
│   ├── depths/
│   └── misc/
│       └── nv_generation/
│           ├── est_images_debug/
│           ├── est_masks_debug/
│           └── est_depths_debug/
└── test/
    ├── images/
    ├── seg/
    ├── all_cameras/
    ├── smpl/
    └── smplx/
```

Rules:

- `input_data/train` should stay clean (core train inputs only).
- Novel-view synthesized training data should be stored in the standard train folders (`images/`, `seg/`, `depths/`, `all_cameras/`) for the corresponding novel camera ids.
- Only novel-view generation debug artifacts (`est_*`) should live under `input_data/train/misc/nv_generation/`.

### 3) `evaluation/`

Purpose:

- Task outputs at evaluation epochs.
- Organized by epoch, then by task.

Structure:

```text
evaluation/
└── epoch_0030/
    ├── nvs/
    │   ├── inputs/
    │   ├── pred/
    │   ├── gt_inputs/
    │   ├── metrics/
    │   └── viz/
    ├── pose_smplx/
    │   ├── inputs/
    │   ├── pred/
    │   ├── gt_inputs/
    │   ├── metrics/
    │   └── viz/
    └── pose_smpl/
        ├── inputs/
        ├── pred/
        ├── gt_inputs/
        ├── metrics/
        └── viz/
```

## Task Contracts

### NVS (`evaluation/epoch_xxxx/nvs/`)

- `inputs/`: camera/image/mask inputs used to compute NVS metrics.
- `pred/`: renders used for evaluation.
- `gt_inputs/`: GT views compared against predictions.
- `metrics/`: `novel_view_results.txt`, per-camera CSVs, optional per-frame CSV.
- `viz/`: render-vs-gt comparisons, white-bg renders, optional videos.

### Pose SMPL-X (`evaluation/epoch_xxxx/pose_smplx/`)

- `inputs/`: pose-tuned predicted parameter files used in metric computation.
- `pred/`: predicted meshes/pose artifacts used for metrics.
- `gt_inputs/`: GT SMPL-X parameters/meshes transformed into the chosen comparison frame.
- `metrics/`: per-frame and overall SMPL-X pose metrics files.
- `viz/`: qualitative overlays or viewer-ready assets.

### Pose SMPL (`evaluation/epoch_xxxx/pose_smpl/`)

- Same contract as `pose_smplx`, but for SMPL-space evaluation.

## Minimal `manifest.json` (at experiment root)

```json
{
  "output_layout_version": "v1",
  "exp_name": "<exp_name>",
  "scene_name": "<scene_name>",
  "tasks": ["nvs", "pose_smplx", "pose_smpl"],
  "eval_epochs": ["epoch_0000", "epoch_0015", "epoch_0030"]
}
```

Required keys:

- `output_layout_version`
- `exp_name`
- `scene_name`
- `tasks`
- `eval_epochs`

Optional keys (recommended):

- `nvs_cameras`
- `source_camera_id`
- `pose_eval_world_alignment_method`
- `notes`

## Required File-Level Contracts

The directory contract above is not enough by itself. For deterministic tooling, these files should always exist when the corresponding task is enabled.

### NVS Required Outputs (`evaluation/epoch_xxxx/nvs/metrics/`)

- `novel_view_results.txt` (overall NVS metrics)
- `metrics_avg_per_camera.csv`
- `metrics_all_cams_per_frame.csv`

### Pose SMPL-X Required Outputs (`evaluation/epoch_xxxx/pose_smplx/metrics/`)

- `smplx_pose_estimation_overall_results.txt`
- `smplx_pose_estimation_metrics_per_frame.csv`

### Pose SMPL Required Outputs (`evaluation/epoch_xxxx/pose_smpl/metrics/`)

- `smpl_pose_estimation_overall_results.txt`
- `smpl_pose_estimation_metrics_per_frame.csv`

### Optional Visualization Assets

- NVS videos and image comparisons
- Pose meshes (`pred`, `gt_inputs`) prepared for viewer consumption
- Camera dumps used by qualitative viewers

## Design Goals

- Make metric inputs explicit (`pred` vs `gt_inputs`) for every task.
- Keep train/eval data snapshots reproducible.
- Keep NV-generation intermediate outputs available but separated from core train input data.
- Keep paths deterministic for downstream visualization tooling (e.g., Rerun).

## Rerun Visualization Spec

The schema is designed so Rerun exporters can be simple and deterministic.

### Recommended Script Split

Use two exporter scripts (plus one convenience wrapper):

1. `evaluation/qualitative/rerun_export_input_data.py`
2. `evaluation/qualitative/rerun_export_evaluation.py`
3. `evaluation/qualitative/rerun_export_all.py` (optional wrapper calling both)

Rationale:

- `input_data` and `evaluation` have different timelines and semantics.
- Separate scripts keep code modular and allow selective export.
- Wrapper script provides one-command export when needed.

### Rerun Output Files

Store exported recordings under:

```text
<output_dir>/<exp_name>/rerun/
├── input_data.rrd
├── evaluation_epoch_0000.rrd
├── evaluation_epoch_0015.rrd
└── evaluation_epoch_0030.rrd
```

### Entity Path Contract (Rerun)

Use stable entity roots:

- `/input_data/train/...`
- `/input_data/test/...`
- `/evaluation/epoch_0030/nvs/...`
- `/evaluation/epoch_0030/pose_smplx/...`
- `/evaluation/epoch_0030/pose_smpl/...`

Within each path:

- `inputs` entities for metric inputs
- `pred` entities for predictions
- `gt_inputs` entities for ground truth
- `metrics` entities for scalar/time-series logging

### Timeline Contract (Rerun)

- Use `frame` sequence for per-frame data.
- Use `epoch` sequence for epoch-level assets.
- For static assets (e.g., camera intrinsics that do not change over frame), log once at the corresponding entity path.

### Export/Consume Workflow

Python exporter (Rerun SDK):

- `rr.init("<app_id>")`
- `rr.save("<path>.rrd")`
- log entities by the path contract above

Viewer:

- `rerun <path>.rrd`

### Sharing `.rrd` via Hugging Face

Public sharing flow:

1. Export `.rrd` files under `<output_dir>/<exp_name>/rerun/`
2. Upload with `hf upload`
3. Share public Hub URL

Example commands:

```bash
hf upload <user_or_org>/<repo> /path/to/input_data.rrd rerun/input_data.rrd --repo-type=dataset
hf upload <user_or_org>/<repo> /path/to/evaluation_epoch_0030.rrd rerun/evaluation_epoch_0030.rrd --repo-type=dataset
```

### Notes for Large Recordings

- Prefer per-epoch `.rrd` for evaluation to keep files manageable.
- Keep `input_data.rrd` separate from evaluation `.rrd`.
- If needed, also provide task-specific `.rrd` (`nvs_epoch_0030.rrd`, `pose_smplx_epoch_0030.rrd`).

## Scope

- This document defines the target schema and Rerun-facing contract.
- Trainer/exporter implementation details are defined in code-specific docs and scripts.
