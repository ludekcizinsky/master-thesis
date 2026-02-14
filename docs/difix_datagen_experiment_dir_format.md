# DiFix DataGen Experiment Dir Format

This document defines the target format for DiFix tuning data generation experiments.

Primary goals:
1. keep data export deterministic and reproducible,
2. support robust train/eval splitting without leakage,
3. scale to multiple datasets/sources,
4. make DiFix fine-tuning and downstream 3DGS usage straightforward.

## Root

```text
/scratch/izar/cizinsky/thesis/difix_tuning_data/<exp_name>/
```

`<exp_name>` identifies one complete data-generation experiment.

## Export Format (Sample/Triplet-Oriented)

```text
<root>/<exp_name>/
├── samples/
│   ├── <sample_id>/
│   │   ├── image.png
│   │   ├── target_image.jpg
│   │   ├── ref_image.jpg
│   │   └── meta.json
│   └── ...
├── manifests/
│   ├── all_samples.jsonl
│   ├── split_train.jsonl
│   ├── split_test.jsonl
│   └── split_summary.json
├── difix/
│   └── data.json
├── tuning_runs/
│   └── <run_name>/
│       ├── checkpoints/
│       │   ├── model_<step>.pkl
│       │   └── ...
│       ├── eval/
│       └── train_config.json
└── meta/
    ├── generation_config.yaml
    ├── generation_stats.json
    └── dropped_samples.csv
```

### Sample Files

For each sample:
1. `image.png`: imperfect render to refine,
2. `target_image.jpg`: GT target camera image (supervision),
3. `ref_image.jpg`: reference view image,
4. `meta.json`: provenance and split metadata.

### `meta.json` (required keys)

```json
{
  "sample_id": "hi4d_pair00_dance__tcam16__rccam4__tf000123__rf000123",
  "dataset": "hi4d",
  "scene_name": "hi4d_pair00_dance",
  "group_id": "pair00",
  "target_cam_id": 16,
  "reference_cam_id": 4,
  "target_fname": "000123",
  "reference_fname": "000123",
  "image_path": "samples/<sample_id>/image.png",
  "target_image_path": "samples/<sample_id>/target_image.jpg",
  "ref_image_path": "samples/<sample_id>/ref_image.jpg"
}
```

Notes:
1. `reference_cam_id` is the camera used to fetch `ref_image`.
2. `target_fname` and `reference_fname` are explicit and may differ.
3. This supports future strategies where reference view comes from a different timestamp and/or camera.

## Split Strategy

Split is done from `manifests/all_samples.jsonl` into train/test manifests.

### Key Rule

Never split samples from the same `group_id` across train and test.

### Dataset-Specific `group_id`

1. HI4D: `group_id = pair_id` (e.g., `pair00`), to avoid training and evaluating on same pair/persons.
2. Datasets where each scene has different people: `group_id = scene_name`.
3. New datasets can define custom `group_id` extraction without changing the global split engine.

This makes the split mechanism scalable across sources while preserving non-leakage semantics.

## DiFix Trainer Compatibility

`submodules/difix3d/src/train_difix.py` expects one JSON file with:
1. `train` map
2. `test` map
3. each sample entry containing:
   - `image`
   - `target_image`
   - `ref_image`
   - `prompt`

We generate this adapter file at:

```text
<root>/<exp_name>/difix/data.json
```

So DiFix training can be launched directly with:

```bash
accelerate launch --mixed_precision=bf16 submodules/difix3d/src/train_difix.py \
  --dataset_path /scratch/izar/cizinsky/thesis/difix_tuning_data/<exp_name>/difix/data.json \
  --output_dir /scratch/izar/cizinsky/thesis/difix_tuning_data/<exp_name>/tuning_runs/<run_name> \
  --tracker_run_name <run_name>
```

## Tuned Checkpoint Handoff Back to 3DGS Pipeline

After tuning, checkpoints live in:

```text
/scratch/izar/cizinsky/thesis/difix_tuning_data/<exp_name>/tuning_runs/<run_name>/checkpoints/model_<step>.pkl
```

`<exp_name>` identifies the generated data snapshot.  
`<run_name>` identifies one specific fine-tuning run (hyperparameter setup) on top of that same data.

This allows multiple tuning runs per same data export, for example:
1. different learning rates,
2. different batch sizes,
3. different max steps/checkpoint cadence.

Planned trainer contract (3DGS pipeline):
1. training config supports an explicit tuned checkpoint path (e.g. `train.difix.model_checkpoint_path`),
2. if provided, refinement uses this tuned checkpoint,
3. if not provided, refinement falls back to base pretrained model config.

This enables a clean loop:
1. define data generation strategy,
2. generate triplets,
3. tune DiFix,
4. pick checkpoint,
5. run 3DGS pipeline with tuned DiFix.

## Reproducibility Notes

Each `<exp_name>` must persist:
1. generation config snapshot (`meta/generation_config.yaml`),
2. split summary (`manifests/split_summary.json`),
3. generation stats and dropped reasons.

`generation_config.yaml` should store experiment-global generation settings only (not single-scene coverage fields).  
Scene/dataset/group coverage belongs in `generation_stats.json` and manifests.

This guarantees that data generation, split, and tuning inputs are auditable and repeatable.
