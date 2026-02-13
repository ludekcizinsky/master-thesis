# Qualitative Export + Upload

This folder provides one wrapper entrypoint for qualitative export and upload:

- `evaluation/qualitative/run.py`
- `evaluation/qualitative/submit.slurm` (for scheduled array runs)

Helper scripts live under:

- `evaluation/qualitative/helpers/`

## What It Does

`run.py` orchestrates two pipeline stages:

1. Export `.rrd` files from experiment outputs.
2. Upload selected `.rrd` files to Hugging Face (with scene-level README generation).

You can run export only, upload only, or both.

## Supported Tasks

Use `--tasks` with one or more comma-separated values:

- `nvs`: novel view synthesis evaluation export
- `pose`: pose estimation evaluation export (SMPL-X)
- `trn_nv_generation`: training-time novel-view generation debug export

Example:

```bash
--tasks nvs,pose,trn_nv_generation
```

## Inputs

Single-scene mode expects:

- `--exp-dir <results_root>/<scene_name>/<exp_name>`

Example:

- `/scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/105_major_refactor_debug`

Multi-scene mode expects:

- `--exp-name <exp_name>`
- optional `--results-root` (default: `/scratch/izar/cizinsky/thesis/results`)

## Local Output

By default, exported files are written to:

- `<exp_dir>/rerun/`

Current output file names are task-specific:

- `evaluation_nvs_epoch_XXXX.rrd`
- `evaluation_pose_epoch_XXXX.rrd`
- `trn_nv_generation_epoch_XXXX.rrd`

## Main Usage

Export only (single scene):

```bash
python evaluation/qualitative/run.py \
  --exp-dir /scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/105_major_refactor_debug \
  --tasks nvs,pose,trn_nv_generation \
  --epoch all \
  --no-upload
```

Export + upload (single scene):

```bash
python evaluation/qualitative/run.py \
  --exp-dir /scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/105_major_refactor_debug \
  --tasks nvs,pose,trn_nv_generation \
  --epoch all \
  --upload
```

Upload-only (if `.rrd` already exists):

```bash
python evaluation/qualitative/run.py \
  --exp-dir /scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/105_major_refactor_debug \
  --tasks nvs,pose,trn_nv_generation \
  --no-export \
  --upload
```

The upload step is non-interactive (safe for Slurm execution).
For boolean options, use flags like `--upload`, `--no-upload`, `--export`, `--no-export` (do not pass `true/false` values).

## Multi-Scene Scheduling (Slurm Array)

Schedule all matching scenes for one experiment:

```bash
python evaluation/qualitative/run.py \
  --exp-name 105_major_refactor_debug \
  --tasks nvs,pose,trn_nv_generation \
  --upload \
  --schedule
```

Optional scene filter:

```bash
python evaluation/qualitative/run.py \
  --exp-name 105_major_refactor_debug \
  --scene-name-includes hi4d_pair16 \
  --tasks nvs,pose \
  --upload \
  --schedule
```

Scheduling workflow:

1. Discovers scene experiment dirs as `results_root/<scene_name>/<exp_name>`.
2. Prints the submission plan.
3. Waits for confirmation (`Enter` submits, any other input cancels).
4. Submits one array task per scene using `evaluation/qualitative/submit.slurm`.

## Upload Layout on Hugging Face

When upload is enabled, files are pushed as:

- `experiments/<exp_name>/<scene_name>/rerun_files/*.rrd`
- `experiments/<exp_name>/<scene_name>/README.md`

The scene-level `README.md` is generated and uploaded automatically together with the selected `.rrd` files.

Default upload target:

- repo id: `ludekcizinsky/rerun-exp-eval`
- repo type: `dataset`
- branch: `main`

## Upload Selection Behavior

`run.py` maps tasks to upload selection automatically:

- if tasks include `nvs` and/or `pose`: selection defaults to `eval_all` (`eval_latest` if `--epoch latest`)
- if tasks only include `trn_nv_generation`: eval selection is disabled, and upload relies on `--rrd-glob trn_nv_generation_epoch_*.rrd`

You can still override upload selection with `--upload-selection`.

## Viewer Notes

### NVS

- one grid with one `Spatial2DView` per camera
- each camera stream is stacked `pred|gt`
- source camera is visible by default; others are toggleable in Blueprint panel
- timeline defaults: `frame`, `fps=20`, `loop_mode=all`

### Pose

- pose meshes exported for SMPL-X task
- root-aligned and world-aligned spaces are logged
- source camera and source-view RGB are included
- timeline defaults: `frame`, `fps=20`, `loop_mode=all`

### Training NV Generation

- left: world-space meshes + cameras over time
- right: per-virtual-camera debug streams (triplet and split views)
- timeline defaults: `frame`, `fps=20`, `loop_mode=all`

Source camera id is inferred from:

- `preprocess/scenes/<scene_name>.json`
- key `cam_id`

## Open in Rerun

```bash
rerun /scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/105_major_refactor_debug/rerun/evaluation_nvs_epoch_0010.rrd
```

```bash
rerun /scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/105_major_refactor_debug/rerun/evaluation_pose_epoch_0010.rrd
```

```bash
rerun /scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/105_major_refactor_debug/rerun/trn_nv_generation_epoch_0006.rrd
```
