# Visualization Helpers

This folder contains interactive visualization utilities for inspecting a preprocessed `scene_dir`.

## check_scene_in_3d.py

Launches an interactive Viser UI that shows:
- SMPL / SMPL-X meshes
- Per-frame cameras
- Masked RGB view
- SMPL-X overlay view
- Depth map view (dynamic percentile range)
- Optional 3D Gaussians (3DGS)

### Basic usage

```bash
python preprocess/vis/check_scene_in_3d.py \
  --scene-dir /path/to/scene_dir
```

### Common options

```bash
python preprocess/vis/check_scene_in_3d.py \
  --scene-dir /path/to/scene_dir \
  --frame-idx-range 0 10 \
  --port 8080 \
  --vis-3dgs
```

### Single-frame visualization

The script expects a range, so use `start` and `end = start + 1`:

```bash
python preprocess/vis/check_scene_in_3d.py \
  --scene-dir /path/to/scene_dir \
  --frame-idx-range 25 26
```

### Notes

- Depth map visualization uses a per-scene percentile range (10–90) computed from valid depth values.
- If `meta.npz` contains `genders`, those are used to load gender-specific body models.
- Source camera is inferred automatically from `preprocess/scenes/<scene_dir_name>.json` (`cam_id`).
- The UI runs on the specified `--port` (default `8080`).

### Example command

Here is an example command to visualize frames 80 to 100 of a specific estimated scene:

```bash
estimated_scene_dir=/scratch/izar/cizinsky/thesis/v2_preprocessing/hi4d_pair15_fight
python preprocess/vis/check_scene_in_3d.py --scenes-dir $estimated_scene_dir --frame-idx-range 80 100
```
