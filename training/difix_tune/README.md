# DiFix Tuning Scheduler

This folder provides a dedicated launcher for DiFix fine-tuning:

- `training/difix_tune/run.py`
- `training/difix_tune/submit.slurm`

The launcher is designed for **single-node training** and uses:

1. Slurm for resource allocation.
2. `accelerate launch` for process spawning across GPUs on that node.

## What `run.py` does

`training/difix_tune/run.py`:

1. Resolves dataset JSON from:
   - `--dataset-path`, or
   - `${THESIS_DIFIX_DATA_ROOT_DIR}/<exp_name>/difix/data.json`
2. Resolves output dir from:
   - `--output-dir`, or
   - `${THESIS_DIFIX_DATA_ROOT_DIR}/<exp_name>/tuning_runs/<run_name>`
3. Builds an `accelerate launch` command for `submodules/difix3d/src/train_difix.py`.
4. Runs locally, or submits one Slurm job with `--submit`.

## Basic usage

### Local run (single GPU)

```bash
python training/difix_tune/run.py \
  --exp-name v0_difix_data_full \
  --run-name run_000 \
  --num-processes 1
```

### Slurm run (single GPU)

```bash
python training/difix_tune/run.py \
  --submit \
  --exp-name v0_difix_data_full \
  --run-name run_000 \
  --num-processes 1 \
  --slurm.gpus 1
```

### Slurm run (2 GPUs on one node)

```bash
python training/difix_tune/run.py \
  --submit \
  --exp-name v0_difix_data_full \
  --run-name run_001_2gpu \
  --num-processes 2 \
  --slurm.gpus 2 \
  --slurm.cpus-per-task 16 \
  --slurm.mem 128G
```

### Slurm run (4 GPUs on one node, typically XL node)

```bash
python training/difix_tune/run.py \
  --submit \
  --exp-name v0_difix_data_full \
  --run-name run_002_4gpu \
  --num-processes 4 \
  --slurm.gpus 4 \
  --slurm.cpus-per-task 24 \
  --slurm.mem 192G
```

## Important notes

1. In submit mode, `--num-processes` must equal `--slurm.gpus`.
2. This scheduler currently targets single-node setups.
3. For multi-node training, add explicit rendezvous config (`main_process_ip`, `machine_rank`, etc.).

## Useful flags

- `--exp-name`: data-generation experiment name under the DiFix data root
- `--run-name`: tuning run name under `tuning_runs/`
- `--dataset-path`: optional explicit `data.json`
- `--output-dir`: optional explicit output dir
- `--base-model-id`: HF model id or local diffusers dir used as warm-start base (default `stabilityai/sd-turbo`)
- `--resume`: optional local checkpoint file (`model_*.pkl`) or checkpoint dir to continue training state
- `--submit`: submit via Slurm
- `--dry-run`: print command(s) only
- `--num-processes`: processes for `accelerate` (single-node)
- `--conda-env`: conda env name used in submitted jobs (default `thesis`)
- `--slurm.*`: Slurm resource settings (`gpus`, `cpus-per-task`, `mem`, `time`, etc.)
- `--extra-train-args`: raw passthrough args appended to `train_difix.py`

## Defaults copied from DiFix README baseline

By default, `run.py` launches `train_difix.py` with:

- `--max_train_steps 10000`
- `--resolution 512`
- `--learning_rate 2e-5`
- `--train_batch_size 1`
- `--dataloader_num_workers 8`
- `--enable_xformers_memory_efficient_attention`
- `--checkpointing_steps 1000`
- `--eval_freq 1000`
- `--viz_freq 100`
- `--lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0`
- `--gram_loss_warmup_steps 2000`
- `--report_to wandb`
- `--tracker_project_name difix`
- `--timestep 199`

You can override these via CLI flags.

## Warm-start vs resume

1. Use `--base-model-id nvidia/difix_ref` to initialize from the published DiFix model on HF.
2. Use `--resume <local_path>` to continue from a previously saved local training checkpoint.
3. `--resume` restores optimizer + step state; `--base-model-id` sets the model initialization base.
