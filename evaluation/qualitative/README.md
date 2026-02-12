# Qualitative Export + Upload

This folder provides one wrapper entrypoint for qualitative pipelines:

- `evaluation/qualitative/run.py`
- `evaluation/qualitative/submit.slurm` (for scheduled array runs)

Helper scripts live under:

- `evaluation/qualitative/helpers/`

## Folder Structure

- `run.py`: orchestrates export and optional upload.
- `helpers/rerun_export_evaluation.py`: exports `evaluation_epoch_*.rrd`.
- `helpers/upload_rerun_to_hf.py`: uploads `.rrd` files + generated README to Hugging Face.

## Single Scene Input

Pass one experiment directory:

- `--exp-dir <results_root>/<scene_name>/<exp_name>`

Example:

- `/scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/105_major_refactor_debug`

## Output

By default, outputs are written to:

- `<exp_dir>/rerun/`

Files created:

- `evaluation_epoch_0000.rrd`, `evaluation_epoch_0005.rrd`, ...

## Main Usage

Export only evaluation NVS:

```bash
python evaluation/qualitative/run.py \
  --exp-dir /scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/105_major_refactor_debug \
  --tasks nvs \
  --epoch all \
  --export true \
  --upload false
```

Export + upload in one call:

```bash
python evaluation/qualitative/run.py \
  --exp-dir /scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/105_major_refactor_debug \
  --tasks nvs \
  --epoch latest \
  --upload true
```

The upload step now runs non-interactively (no confirmation prompt), which is safe for Slurm jobs.

## Multi-Scene Scheduling (SLURM Array)

Use `--schedule` with `--exp-name` to process all matching scenes under `results_root`:

```bash
python evaluation/qualitative/run.py \
  --exp-name 105_major_refactor_debug \
  --results-root /scratch/izar/cizinsky/thesis/results \
  --tasks nvs,pose,trn_nv_generation \
  --export true \
  --upload true \
  --schedule
```

Workflow:

1. Script discovers scene experiment dirs as:
   `results_root/<scene_name>/<exp_name>`.
2. Prints scene list and Slurm submission plan.
3. Waits for confirmation:
   `Enter` submits, any other input cancels.
4. Submits one array task per scene.

Optional scene filter:

```bash
python evaluation/qualitative/run.py \
  --exp-name 105_major_refactor_debug \
  --scene-name-includes hi4d_pair16 \
  --tasks nvs,pose \
  --upload true \
  --schedule
```

## Viewer Notes (Evaluation)

Current NVS layout in each `evaluation_*.rrd`:

- one `Grid` with one `Spatial2DView` per camera (`cam_*`)
- each camera stream shows stacked `pred|gt`
- only source camera is visible by default; others are toggleable in Blueprint panel
- timeline defaults: `frame`, `fps=20`, `loop_mode=all`

Source camera is inferred automatically from:

- `preprocess/scenes/<scene_name>.json`
- key `cam_id`

where `scene_name` comes from `<results_root>/<scene_name>/<exp_name>`.

## Open in Rerun

```bash
rerun /scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/105_major_refactor_debug/rerun/evaluation_epoch_0010.rrd
```
