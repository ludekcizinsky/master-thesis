# Training Scheduler

This folder uses `training/run.py` as the single entrypoint to launch training jobs.

- Local run (single scene or all scenes)
- Slurm array submission
- Hydra overrides forwarding

Scene definitions are loaded from `preprocess/scenes/*.json`.

## What `run.py` does

`training/run.py`:

1. Loads scenes from `preprocess/scenes`.
2. Optionally filters scenes by substring with `--scene-name-includes`.
3. Runs training directly (`python training/simple_multi_human_trainer.py ...`) or submits Slurm jobs.
4. In Slurm array mode, each task index maps to one scene via `SLURM_ARRAY_TASK_ID`.

## Basic usage

Run one scene locally:

```bash
python training/run.py \
  --scene-name-includes hi4d_pair17_dance \
  --exp-name my_exp
```

Run all scenes locally:

```bash
python training/run.py \
  --run-all \
  --exp-name my_exp
```

Dry run (print commands only):

```bash
python training/run.py \
  --scene-name-includes hi4d_pair17_dance \
  --exp-name my_exp \
  --dry-run
```

## Hydra overrides

Forward any Hydra overrides using `--overrides`:

```bash
python training/run.py \
  --scene-name-includes hi4d_pair17_dance \
  --exp-name my_exp \
  --overrides shared.wandb.enable=false train.evaluation.pose_eval.eval_smpl=false
```

## Slurm submission

Submit one scene:

```bash
python training/run.py \
  --submit \
  --scene-name-includes hi4d_pair17_dance \
  --exp-name my_exp \
  --overrides shared.wandb.enable=false
```

Submit all scenes as an array:

```bash
python training/run.py \
  --submit \
  --run-all \
  --exp-name my_exp
```

Submit all scenes with max parallel tasks:

```bash
python training/run.py \
  --submit \
  --run-all \
  --exp-name my_exp \
  --slurm.array-parallelism 2
```

By default, submissions use `training/submit.slurm`.

## DiFix Data Generation (One Command)

If you submit DiFix shard generation with:

- `--submit`
- `run_mode=difix_data_generation_generate` in `--overrides`

then `training/run.py` will automatically submit a dependent aggregate job with
Slurm dependency `afterok:<array_job_id>`.

Example:

```bash
python training/run.py \
  --submit \
  --run-all \
  --scenes-dir /scratch/izar/cizinsky/thesis/misc/scene_subsets/difix_train_v1 \
  --exp-name v1_difix_data \
  --slurm.array-parallelism 6 \
  --overrides run_mode=difix_data_generation_generate shared.wandb.enable=false
```

You can control this behavior with:

- `--auto-schedule-aggregate` / `--no-auto-schedule-aggregate`
- `--aggregate-job-name`
- `--aggregate-time`

## Important flags

- `--exp-name`: experiment name passed as `shared.exp_name=<value>`
- `--scene-name-includes`: substring match on `seq_name`
- `--run-all`: run all matched scenes
- `--submit`: submit through Slurm (array mode)
- `--dry-run`: print commands without executing
- `--overrides`: extra Hydra key/value overrides
- `--slurm.job-name`: Slurm job name
- `--slurm.slurm-script`: path to Slurm wrapper script
- `--slurm.array-parallelism`: array concurrency limit
- `--auto-schedule-aggregate`: auto-submit dependent DiFix aggregate job when submitting DiFix generate arrays
- `--aggregate-job-name`: Slurm job name for aggregate stage
- `--aggregate-time`: Slurm time limit for aggregate stage

## Novel-View Camera Config

Novel-view training cameras are now configured by count, not by explicit id list.

In `training/configs/train.yaml`:

- `train.nv_generation.trn_nv_gen.num_cameras`: number of virtual cameras to synthesize
- `train.nv_generation.trn_nv_gen.start_camera_id`: first virtual camera id (generated ids are contiguous)
- `train.nv_generation.trn_nv_gen.runtime_camera.*`: strategy parameters used to place those cameras

Behavior in trainer:

- Source camera id is resolved from scene metadata (`preprocess/scenes/<scene>.json`, key `cam_id`).
- Virtual ids are generated as `start_camera_id + i` for `i in [0, num_cameras)`.
- Source camera id must not collide with generated virtual ids.

Example override:

```bash
python training/run.py \
  --scene-name-includes hi4d_pair17_dance \
  --exp-name my_exp \
  --overrides train.nv_generation.trn_nv_gen.num_cameras=5 train.nv_generation.trn_nv_gen.start_camera_id=200
```
