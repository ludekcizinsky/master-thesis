# HI4D Eval GT Materialization

This folder provides a canonical GT preprocessing wrapper for HI4D:

- Entry point: `preprocess/eval/hi4d/run.py`
- Slurm wrapper: `preprocess/eval/hi4d/submit.slurm`

The goal is to convert raw HI4D scene folders into the repository's canonical scene format under:

`/scratch/izar/cizinsky/thesis/gt_scene_data/<scene_name>`

## What `run.py` does

For each selected scene:

1. Reads scene metadata from `preprocess/scenes/*.json`.
2. Ensures raw GT is available:
   - discovers all matching pair archives from Hugging Face (`pairXX*.tar.gz`)
   - downloads missing archives
   - extracts archives into the raw root directory
   - resolves non-standard extracted layouts by linking to expected `raw_gt_dir_path`
3. Loads raw GT from `raw_gt_dir_path`.
4. Builds a shared frame set from the intersection of:
   - `images/<cam_id>`
   - `seg/img_seg_mask/<cam_id>/all`
   - `all_cameras/<cam_id>`
   - `smpl/`
5. Renames frames to contiguous canonical names starting at `000001`.
6. Copies canonicalized data into output scene dir:
   - `images/<cam_id>/`
   - `seg/img_seg_mask/<cam_id>/{all,0,1,...}/`
   - `all_cameras/<cam_id>/`
   - `smpl/`
   - `meta.npz` (if present)
7. Runs SMPL -> SMPL-X conversion in the canonical scene dir.
8. Writes `frame_map.json` with `old_to_new` and `new_to_old` frame mappings.

## Scene JSON requirements

Each scene JSON in `preprocess/scenes/` should define:

- `seq_name`
- `cam_id` (reference/source camera)
- `raw_gt_dir_path` (required)
- `video_path` (used by train preprocessing)
- `ref_frame_idx` (typically `0`)

Example:

```json
{
  "seq_name": "hi4d_pair16_jump",
  "cam_id": 4,
  "raw_gt_dir_path": "/scratch/izar/cizinsky/raw_datasets/hi4d/pair16/jump16",
  "video_path": "/scratch/izar/cizinsky/raw_datasets/hi4d/pair16/jump16/images/4",
  "ref_frame_idx": 0
}
```

## Scene JSON Helper

Use `preprocess/eval/hi4d/helpers/scene_to_json.py` to generate a scene JSON from a raw scene directory.

The helper:

- infers `seq_name` as `hi4d_<pair>_<action>` from directory names
- reads `cam_id` from `meta.npz` key `mono_cam`
- sets `raw_gt_dir_path` to the provided scene dir
- sets `video_path` to `<raw_gt_dir_path>/images/<cam_id>`
- sets `ref_frame_idx` (default `0`)

Example:

```bash
python preprocess/eval/hi4d/helpers/scene_to_json.py \
  --scene-dir-path /scratch/izar/cizinsky/raw_datasets/hi4d/pair00/yoga00
```

This writes:

`preprocess/scenes/hi4d_pair00_yoga.json`

Useful flags:

- `--dry-run`
- `--overwrite false`
- `--ref-frame-idx <int>`
- `--scenes-output-dir <path>`

## Usage

Run one scene locally:

```bash
python preprocess/eval/hi4d/run.py --seq-name-includes hi4d_pair16_jump
```

Dry-run one scene:

```bash
python preprocess/eval/hi4d/run.py --seq-name-includes hi4d_pair16_jump --dry-run --run-all
```

Run all HI4D scenes locally:

```bash
python preprocess/eval/hi4d/run.py --run-all
```

Submit all HI4D scenes to Slurm array:

```bash
python preprocess/eval/hi4d/run.py --submit --run-all
```

Submit one specific scene to Slurm:

```bash
python preprocess/eval/hi4d/run.py --submit --seq-name-includes hi4d_pair16_jump
```

Normalize scene registry entries (refresh `raw_gt_dir_path`, remove stale `gt_dir_path`):

```bash
python preprocess/eval/hi4d/run.py --run-all --update-scene-registry
```

## Important config flags

- `output_root_dir`: canonical GT root (default `/scratch/izar/cizinsky/thesis/gt_scene_data`)
- `ensure_raw_data`: auto-download/extract raw scene before materialization
- `ensure_raw_data_script`: helper invoked by `run.py`
- `hf_repo_id`: default `ludekcizinsky/hi4d`
- `seq_name_prefix`: default `hi4d_`
- `overwrite_output_scene`: if `true`, rebuilds existing canonical scene dir
- `include_depths_if_available`: copy depth maps if present
- `update_scene_registry`: write canonical `gt_dir_path` back to scene JSONs

## Output structure

Canonical output for scene `hi4d_pair16_jump`:

```text
/scratch/izar/cizinsky/thesis/gt_scene_data/hi4d_pair16_jump/
├── images/<cam_id>/*.jpg
├── seg/img_seg_mask/<cam_id>/{all,0,1,...}/*.png
├── all_cameras/<cam_id>/*.npz
├── smpl/*.npz
├── smplx/*.npz
├── misc/preprocess_info.txt      # run status + timing + frame count (+ traceback on failure)
├── meta.npz                      # optional
└── frame_map.json
```

## Notes

- Raw GT data is not modified in place.
- The canonical scene is the expected input for trainer evaluation via:
  `test_scene_dir: ${canonical_gt_scene_root_dir}/${scene_name}` in `training/configs/train.yaml`.
