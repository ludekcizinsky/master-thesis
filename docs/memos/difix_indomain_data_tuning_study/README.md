## Study Versions

This memo is the single source of truth for study designs and iteration history.
Each version below should document:
1. objective,
2. data split (train/test scenes),
3. key decisions,
4. expected outputs.

## Fixed Scene Policy (Important)

These scenes are reserved for 3DGS tuning/evaluation and must be excluded from DiFix tuning data:
1. `hi4d_pair15_fight`
2. `hi4d_pair16_jump`
3. `hi4d_pair17_dance`
4. `hi4d_pair19_piggyback`

Interpretation:
1. we still preprocess/materialize them (needed by the trainer pipeline),
2. but we do **not** use them as DiFix train samples.

## V0 Plan (Ultra-Fast Sanity Check)

Hypothesis:
> with a tiny in-domain train subset, we can validate the full DiFix tuning pipeline end-to-end and get first directional signal quickly

### Objective
Run the fastest possible training/eval loop to validate:
1. data generation pipeline,
2. DiFix tuning integration,
3. downstream NVS evaluation wiring.

### DiFix Train Split (V0)
Use only 2 scenes (non-reserved pairs):
1. `hi4d_pair00_dance`
2. `hi4d_pair01_talk`

### DiFix Internal Eval Split (V0)
Use one small held-out scene from a different pair:
1. `hi4d_pair02_dance`

### Downstream 3DGS Eval Split (Fixed)
Keep the fixed reserved evaluation scenes:
1. `hi4d_pair15_fight`
2. `hi4d_pair16_jump`
3. `hi4d_pair17_dance`
4. `hi4d_pair19_piggyback`

### Why this split
1. no leakage with reserved downstream eval scenes,
2. minimal time to first result,
3. includes two different motion profiles for tuning (`dance` vs `talk`),
4. has a tiny but separate DiFix internal eval scene (`pair02`) for quick sanity checks.

### Scenes to Preprocess First (Pragmatic Order)
1. DiFix tuning: `hi4d_pair00_dance`, `hi4d_pair01_talk`
2. DiFix internal eval: `hi4d_pair02_dance`
3. Downstream fixed eval: `hi4d_pair15_fight`, `hi4d_pair16_jump`, `hi4d_pair17_dance`, `hi4d_pair19_piggyback`

### Expected Output
1. successful generation of DiFix training samples (`image`, `target_image`, `ref_image`),
2. successful DiFix fine-tuning run,
3. NVS metrics on the fixed eval scenes for baseline comparison.

### Key Design Decisions
1. Use **imperfect poses/rendering pipeline output** to build the input image (`image`), because this is what we have at inference time.
2. Use **same-timestamp source-camera GT** as reference (`ref_image`):
   - target sample is `(cam_target, t)`
   - reference is `(cam_source, t)`

### Out of Scope (For Later Ablations)
1. GT-pose vs imperfect-pose rendering for data generation.
2. Alternative reference selection:
   - temporal neighbor reference (`t±k`),
   - mixed strategy (same-time + temporal offsets).

### Tuning Enablement Notes (2026-02-15)

The first successful DiFix tuning launch required a few stability fixes and runtime adjustments:

1. Added a dedicated tuning scheduler entrypoint:
   - `training/difix_tune/run.py`
   - `training/difix_tune/submit.slurm`
2. Enforced model-safe resize alignment in DiFix training code:
   - target resize now snaps to multiples of 16 (prevents VAE skip-connection shape mismatch).
3. Fixed dataset validation shuffle bug:
   - `dataset_val.img_names` -> `dataset_val.img_ids`.
4. Wired train/val dataset dimensions to the configured `--resolution`.
5. Routed Hugging Face cache to scratch-backed paths to avoid home quota pressure during model downloads.
6. Exposed `max_grad_norm` in the scheduler wrapper and passed it to `train_difix.py`.

Observed runtime behavior during bring-up:

1. `fp16` runs hit GradScaler failure (`Attempting to unscale FP16 gradients`) in current upstream training path.
2. `mixed_precision=no` at higher resolution could hit OOM.
3. A stable configuration was found with:
   - `num_processes=2` (single node, 2 GPUs),
   - `mixed_precision=no`,
   - `resolution=320`,
   - `max_grad_norm=0`.

Current status:

1. Tuning runs can be launched and kept running under the stable config above.
2. This is sufficient for V0 to move forward and get first checkpoint outputs.
3. Optimization for speed/quality (e.g., restoring mixed precision) is deferred to later ablation/engineering rounds.

### Interim TODOs (High-Level)

- [ ] Run first DiFix fine-tuning run on V0 data and save checkpoints under the experiment directory.
- [ ] Connect tuned DiFix checkpoint to downstream 3DGS pipeline (explicit checkpoint path).
- [ ] Run downstream evaluation on reserved scenes (`hi4d_pair15_fight`, `hi4d_pair16_jump`, `hi4d_pair17_dance`, `hi4d_pair19_piggyback`).
- [ ] Record baseline vs tuned deltas (PSNR, SSIM, LPIPS) as the first decision snapshot.
- [ ] Decide the next ablation direction (more scenes, camera/frame sampling policy, reference-view strategy).
