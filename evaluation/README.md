# Evaluation utilities

This folder contains small scripts for (a) summarising benchmark results and (b) visual sanity-checking outputs (2D overlays + an interactive 3D viewer).

## Contents (scripts at the root of `evaluation/`)

- `summarise_results.sh`: convenience wrapper for `evaluation/helpers/summarise_benchmark_results.py`
- `visualise_scene_in_2d.py`: renders mesh normal-map overlays onto RGB frames and writes PNGs
- `visualise_scene_in_3d.py`: launches an interactive Viser viewer for a single frame (3DGS + meshes + cameras + optional background point cloud)

## Expected evaluation directory layout

Both visualization scripts expect an “evaluation scene directory” (passed as `--exp-eval-dir` or `--eval-scene-dir`) that contains some of the following subfolders:

```text
<EVAL_SCENE_DIR>/
  images/<cam_id>/<frame>.jpg|.png
  all_cameras/<cam_id>/<frame>.npz              # contains intrinsics, extrinsics
  posed_meshes_per_frame/
    <frame>.obj                                 # optional: combined mesh per frame
    <person_id>/<frame>.obj                     # optional: per-person meshes
  posed_smplx_meshes_per_frame/<person_id>/<frame>.obj   # optional
  posed_3dgs_per_frame/<frame>.pt               # for `visualise_scene_in_3d.py`
  masked_depth_maps/<frame>.npy                 # optional: for background point cloud in 3D viewer
```

Notes:
- `visualise_scene_in_2d.py` requires: `images/<cam_id>/...`, `all_cameras/<cam_id>/...`, and at least one mesh in `posed_meshes_per_frame/`.
- `visualise_scene_in_3d.py` can run with just meshes + cameras, but is most useful when `posed_3dgs_per_frame/` is present; it can also show a background point cloud if `masked_depth_maps/` exists.

## 1) Summarise benchmark results

### Quick start (wrapper script)

```bash
bash evaluation/summarise_results.sh <exp_name>
```

What it does:
- Activates a hard-coded conda env (`thesis`) and loads modules (`gcc`, `ffmpeg`).
- Uses a fixed `epoch_str="0030"` and a fixed list of scenes + source camera IDs.
- Calls `evaluation/helpers/summarise_benchmark_results.py` with:
  - `--exp-name <exp_name>`
  - `--epoch-str 0030`
  - `--scene-names ...`
  - `--src-cam-ids ...`
  - `--results-root /scratch/izar/cizinsky/thesis/results`

If you’re not on the same machine/HPC environment, you will likely need to edit:
- conda path, module loads, `results_root`, `epoch_str`, `scene_names`, `src_cam_ids`.

### Direct usage (recommended for portability)

```bash
python evaluation/helpers/summarise_benchmark_results.py \
  --exp-name "<exp_name>" \
  --epoch-str "0030" \
  --scene-names hi4d_pair15_fight hi4d_pair16_jump \
  --src-cam-ids 4 4 \
  --results-root "/path/to/results" \
  --output-dir "/path/to/output"
```

Outputs:
- `<output-dir>/<exp_name>/all_results.md` (per-task markdown tables, per dataset)
- `<output-dir>/<exp_name>/baselines_with_ours.md` (joins baselines from `evaluation/baselines/*.csv` with your results)
- Also writes per-dataset CSV/MD files like `<dataset>_nvs.csv`, `<dataset>_nvs.md`, etc.

## 2) 2D mesh overlays (normal maps + per-person + SMPL-X coloring)

Renders:
- RGB frame overlaid with a normal map of the combined mesh
- Per-person normal maps (white background)
- Optional SMPL-X meshes colored per person and alpha-blended onto RGB

Run:

```bash
python evaluation/visualise_scene_in_2d.py \
  --exp-eval-dir "<EVAL_SCENE_DIR>" \
  --cam-id 4 \
  --max-frames 50
```

Key inputs:
- `<EVAL_SCENE_DIR>/images/<cam_id>/<frame>.(jpg|png|...)`
- `<EVAL_SCENE_DIR>/all_cameras/<cam_id>/<frame>.npz` containing `intrinsics` and `extrinsics`
- `<EVAL_SCENE_DIR>/posed_meshes_per_frame/` meshes (`.obj`)

Outputs (created under the eval directory):
- `<EVAL_SCENE_DIR>/quality_checks/meshes/<frame>.png`
- `<EVAL_SCENE_DIR>/quality_checks/individual_meshes/<person_id>/<frame>.png`
- `<EVAL_SCENE_DIR>/quality_checks/smplx_meshes/<frame>.png`

Headless rendering note:
- If you hit OpenGL context errors, try running with `PYOPENGL_PLATFORM=egl`.

## 3) Interactive 3D viewer (Viser)

Launch a viewer for a single frame:

```bash
python evaluation/visualise_scene_in_3d.py \
  --eval-scene-dir "<EVAL_SCENE_DIR>" \
  --frame-index 0 \
  --port 8080
```

Then open:
- `http://localhost:8080`

What it shows (depending on what exists on disk):
- posed 3D Gaussians (from `posed_3dgs_per_frame/`)
- posed meshes (from `posed_meshes_per_frame/`)
- posed SMPL-X meshes (from `posed_smplx_meshes_per_frame/`)
- cameras (from `all_cameras/`)
- optional background point cloud (from `masked_depth_maps/` + `images/<source_camera_id>/...`)

Useful knobs (CLI args):
- `--frame-name <stem>` (select by filename stem instead of index)
- `--source-camera-id <id>` (used for background RGB/depth pairing)
- `--max-gaussians <N>` / `--max-scale <S>` (performance / visualization stability)
- `--background-max-depth <meters>` / `--depth-stride <k>` (background point cloud density)
- `--background-filter-sparse-points` (enable sparse outlier removal for background depth points)
- `--background-filter-voxel-size <meters>` (3D neighborhood radius proxy; larger = more aggressive filtering)
- `--background-filter-min-neighbors <N>` (minimum local 3D neighbor count to keep points)
