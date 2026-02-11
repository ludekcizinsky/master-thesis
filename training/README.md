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
  --overrides wandb.enable=false pose_eval.eval_smpl=false
```

## Slurm submission

Submit one scene:

```bash
python training/run.py \
  --submit \
  --scene-name-includes hi4d_pair17_dance \
  --exp-name my_exp \
  --overrides wandb.enable=false
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

## Important flags

- `--exp-name`: experiment name passed as `exp_name=<value>`
- `--scene-name-includes`: substring match on `seq_name`
- `--run-all`: run all matched scenes
- `--submit`: submit through Slurm (array mode)
- `--dry-run`: print commands without executing
- `--overrides`: extra Hydra key/value overrides
- `--slurm.job-name`: Slurm job name
- `--slurm.slurm-script`: path to Slurm wrapper script
- `--slurm.array-parallelism`: array concurrency limit
