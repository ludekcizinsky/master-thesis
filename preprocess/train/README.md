# Preprocess Train: Scene Runner + Slurm Array

This directory provides a simple, scalable way to preprocess scenes locally or on a Slurm cluster.

## Files
- `run.py`: Tyro-based CLI to run one scene, all scenes, or submit a Slurm array.
- `submit.slurm`: Slurm template script used by `run.py` for array submission.
- `scenes/`: One JSON file per scene (loaded in lexical order).

## Scene JSON format
Each scene file must contain:

```json
{
  "video_path": "/abs/path/to/frames",
  "seq_name": "scene_name",
  "cam_id": 4,
  "ref_frame_idx": 0
}
```

Only `ref_frame_idx` is optional (defaults to 0).

## Local usage
Run a single scene by filtering to one match:

```bash
python preprocess/train/run.py --seq-name-includes hi4d_pair15_fight
```

Run all scenes serially:

```bash
python preprocess/train/run.py --run-all
```

## Slurm usage
Submit a job array (one task per scene, with optional filtering):

```bash
python preprocess/train/run.py --submit
```

```bash
python preprocess/train/run.py --submit --seq-name-includes hi4d_pair16
```

Limit parallelism (e.g. max 4 running at once):

```bash
python preprocess/train/run.py --submit --slurm.array-parallelism 4
```

## Customizing Slurm resources
Edit `preprocess/train/submit.slurm` to change default resources:
- `--cpus-per-task`
- `--mem`
- `--gres`
- `--time`
- `--account`

You can also pass extra `sbatch` flags at submit time, for example:

```bash
python preprocess/train/run.py --submit --partition=gpuA --qos=normal
```

These extra args are forwarded to `sbatch`.

## Notes
- The scene list is loaded from `preprocess/scenes/*.json` if present.
- If the `scenes/` directory is empty or missing, `run.py` falls back to its internal defaults.
- Filtering uses substring matching on `seq_name`.
