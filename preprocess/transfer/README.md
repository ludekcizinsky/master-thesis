# Preprocess Data Transfer (Scratch <-> HF)

This folder provides one CLI wrapper for transferring preprocessing artifacts
between local scratch storage and Hugging Face.

- Script: `preprocess/transfer/run.py`
- Defaults:
  - GT repo: `ludekcizinsky/thesis-gt-scene-data`
  - Preprocessed repo: `ludekcizinsky/thesis-preprocessed-scenes`
  - `repo_type=dataset`
  - `private=true`

## Authentication

`upload` and `sync` require a write token.

Recommended:

```bash
export HF_TOKEN=hf_xxx
```

For scheduled jobs (`--schedule`), `HF_TOKEN` must be set in the submit shell.
It is forwarded to array tasks via `--export ALL`.

## Data Roots

If `--local-root` is not provided, local paths come from `configs/paths.yaml`:

- `data_kind=gt` -> `canonical_gt_root_dir`
- `data_kind=preprocessed` -> `preprocessing_root_dir`

## Actions

- `upload`: local scene dirs -> HF (per-scene upload)
- `download`: HF scene dirs -> local root
- `sync`: upload + prune stale remote files not present locally

All actions print a plan and require Enter confirmation (unless `--yes`).
For `upload` and `sync`, the plan includes per-scene file counts and sizes,
plus total files and total upload size.
By default, uploads use `upload_large_folder` for better resilience on large scenes.
You can disable this with `--no-use-large-upload`.
By default, `misc/` is excluded from HF `upload`/`sync` (`--exclude-misc`).
Use `--no-exclude-misc` if you explicitly want to transfer debug artifacts.
Local data is unchanged; this only affects what is transferred to HF.
When `upload_large_folder` is used, worker count defaults to:
`min(max(4, nproc // 2), 16)`.
Override with `--num-upload-workers <int>`.
By default, Hugging Face internal progress bars are disabled for cleaner logs.
Enable with `--hf-progress-bars`.

## Scheduling (Slurm Array)

Use `--schedule` to submit one scene per array task via:

- `preprocess/transfer/submit.slurm`

The script prints the full plan first; press Enter to submit, any other input cancels.
Hidden folders (e.g. `.cache`) under local roots are ignored.

Useful Slurm overrides:

- `--slurm.job-name <name>`
- `--slurm.time <HH:MM:SS>`
- `--slurm.array-parallelism <N>`
- `--slurm.gres <resource>` (default `gpu:1` for clusters with QoS min-GRES)
Default transfer schedule time is `01:00:00`.

## Scene Selection

Use one of:

- `--run-all`
- `--scene-names sceneA,sceneB`
- `--scene-name-includes hi4d_pair01`

## Examples

Upload all canonical GT scenes:

```bash
python preprocess/transfer/run.py \
  --action upload \
  --data-kind gt \
  --run-all
```

Upload selected preprocessed scenes:

```bash
python preprocess/transfer/run.py \
  --action upload \
  --data-kind preprocessed \
  --scene-names hi4d_pair15_fight,hi4d_pair16_jump
```

Download one scene from HF:

```bash
python preprocess/transfer/run.py \
  --action download \
  --data-kind gt \
  --scene-names hi4d_pair00_dance
```

Sync all local preprocessed scenes (upload + prune remote extras):

```bash
python preprocess/transfer/run.py \
  --action sync \
  --data-kind preprocessed \
  --run-all
```

Dry run:

```bash
python preprocess/transfer/run.py \
  --action upload \
  --data-kind gt \
  --scene-name-includes pair01 \
  --dry-run
```

Schedule upload for all GT scenes:

```bash
python preprocess/transfer/run.py \
  --action upload \
  --data-kind gt \
  --run-all \
  --schedule
```

If you hit rate limits (HTTP 429), reduce concurrency:

Operational note: plan around an HF API cap of roughly `2500` calls per `5` minutes
(can vary by auth/account tier). Large multi-scene uploads can hit this quickly.

```bash
python preprocess/transfer/run.py \
  --action upload \
  --data-kind preprocessed \
  --run-all \
  --schedule \
  --slurm.array-parallelism 1 \
  --num-upload-workers 4
```

Schedule sync for selected preprocessed scenes with parallelism cap:

```bash
python preprocess/transfer/run.py \
  --action sync \
  --data-kind preprocessed \
  --scene-name-includes hi4d_pair0 \
  --schedule \
  --slurm.array-parallelism 4
```
