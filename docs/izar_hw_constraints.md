# IZAR HW + Quota Constraints

Last checked: 2026-02-15
User: `cizinsky`
Cluster: `izar`

## 1. Account / QOS Limits (for `master`)

From Slurm accounting (`sacctmgr`), your effective QOS options on `master` are:
- `normal`
- `debug`
- `long`
- `build`

### QOS caps

| QOS | Max Walltime | GPU cap per user (`MaxTRESPU`) | GPU cap total per user (`MaxTRESPerUser`) | Submit cap |
| --- | --- | --- | --- | --- |
| `normal` | `3-00:00:00` | `gres/gpu=16` | `gres/gpu=16` | `MaxSubmitPU=10000` |
| `long` | `7-00:00:00` | `gres/gpu=16` | `gres/gpu=16` | `MaxSubmitPU=10000` |
| `debug` | `01:00:00` | not explicitly set | not explicitly set | `MaxSubmitPU=5` |
| `build` | `10-00:00:00` | not explicitly set | not explicitly set | `MaxSubmitPU=1` |

Practical interpretation:
- For standard training, assume a hard upper bound of **16 GPUs total concurrently** on `normal/long`.
- `debug` is short and useful for smoke tests.

## 2. GPU Partition Snapshot

`gpu` partition (`scontrol show partition gpu`):
- Nodes: `71`
- Total GPUs: `146` (`gres/gpu:volta=146`)
- Total CPUs: `2840`
- Total memory: `~21.4 TB` (`21412914M`)
- Default time: `00:05:00`
- Default mem-per-cpu: `4500 MB`

## 3. Node Classes Observed

From live node inspection:
- Standard GPU nodes (example `i01`):
  - `40` CPUs
  - `~192 GB` RAM (`RealMemory=192556M`)
  - `2x` Volta GPUs (`gres/gpu:volta:2`)
- XL GPU nodes (example `ixl01`):
  - `40` CPUs
  - `~773 GB` RAM (`RealMemory=773163M`)
  - `4x` Volta GPUs (`gres/gpu:volta:4`)

Additional hardware note from local cluster docs/user context:
- ProLiant XL190r Gen10 (U38), Intel Xeon Gold 6230 @ 2.10 GHz, ~196 GB RAM class.

## 4. Practical Defaults For This Repo

- Data generation / eval smoke runs:
  - prefer `debug` (<= 1h) or short `normal`.
- DiFix/3DGS training:
  - use `normal` by default.
  - start with moderate parallelism (e.g. `1-4` GPUs) and scale up.
- Keep total concurrent GPU requests under your effective per-user cap to avoid `QOSMinGRES` / policy rejections.

## 5. Recommended Slurm Presets

Use these as practical defaults, then tune based on runtime and memory behavior.

| Profile | QOS | Time | GPUs | CPUs/Task | Mem | Typical Use |
| --- | --- | --- | --- | --- | --- | --- |
| `smoke` | `debug` | `00:30:00` | `1` | `4` | `32G` | quick config/data sanity check |
| `data_gen` | `normal` | `03:00:00` | `1` | `8` | `96G` | DiFix dataset generation per scene |
| `tune_1gpu` | `normal` | `12:00:00` | `1` | `8` | `96G` | first DiFix tuning baseline |
| `tune_2gpu` | `normal` | `12:00:00` | `2` | `16` | `128G` | faster tuning on standard 2-GPU nodes |
| `tune_4gpu` | `normal` | `12:00:00` | `4` | `24` | `192G` | accelerated tuning on XL nodes |

Notes:
- `debug` is capped at `01:00:00`.
- Keep aggregate concurrent GPU usage under your per-user cap (`16` on `normal/long`).
- `4`-GPU jobs typically require XL nodes (`ixl*`).
- For array jobs, effective total GPUs = `array_parallelism * gpus_per_task`.

## 6. Useful Commands

```bash
# Show your queued/running jobs
squeue -u $USER

# Show account/QOS associations
sacctmgr -P -n show assoc where user=$USER format=Cluster,Account,User,QOS,MaxWall

# Show QOS limits
sacctmgr -P -n show qos format=Name,MaxSubmitPU,MaxTRESPU,MaxTRESPerUser,MaxWall

# Show partition resources
scontrol show partition gpu
```
