# Quantitative Aggregation

This folder contains a single entrypoint:

- `evaluation/quantitative/run.py`

It aggregates quantitative metrics for one experiment across all scenes found under:

- `/scratch/izar/cizinsky/thesis/results`

and writes a single markdown report to:

- `/home/cizinsky/master-thesis/docs/results/<exp_name>/quan_results.md`

## What it reads

For each scene, the script looks for the experiment in either layout:

- New: `<results_root>/<scene_name>/<exp_name>/evaluation/epoch_xxxx/...`
- Legacy: `<results_root>/<scene_name>/evaluation/<exp_name>/epoch_xxxx/...`

Then it reads task files (if present):

- `novel_view_results.txt` -> `Novel View Synthesis`
- `smplx_pose_estimation_overall_results.txt` -> `Pose Estimation (SMPL-X)`
- `smpl_pose_estimation_overall_results.txt` -> `Pose Estimation (SMPL)`

Dataset name is inferred from scene name prefix before the first `_`.

## Output structure

The generated markdown is organized as:

- `## <Task Name>`
- `### <Dataset Name>`
- table:
  - one row per scene
  - one column per metric (dynamic union for that task+dataset)
  - final `avg` row (mean per metric across available scene values)

Missing values are shown as `-`.

## Usage

From repo root:

```bash
python evaluation/quantitative/run.py --exp-name v104_testing_new_code
```

Use a specific epoch (instead of latest per scene):

```bash
python evaluation/quantitative/run.py --exp-name v104_testing_new_code --epoch 30
```

or:

```bash
python evaluation/quantitative/run.py --exp-name v104_testing_new_code --epoch epoch_0030
```

Optional custom roots:

```bash
python evaluation/quantitative/run.py \
  --exp-name v104_testing_new_code \
  --results-root /scratch/izar/cizinsky/thesis/results \
  --docs-root /home/cizinsky/master-thesis/docs/results
```
