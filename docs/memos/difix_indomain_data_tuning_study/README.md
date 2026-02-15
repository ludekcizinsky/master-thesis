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

### Interim TODOs (High-Level)

- [ ] Finish V0 multi-scene DiFix data generation for the selected V0 split.
- [ ] Confirm generated dataset integrity (split files, sample counts, and expected scene coverage).
- [ ] Run first DiFix fine-tuning run on V0 data and save checkpoints under the experiment directory.
- [ ] Connect tuned DiFix checkpoint to downstream 3DGS pipeline (explicit checkpoint path).
- [ ] Run downstream evaluation on reserved scenes (`hi4d_pair15_fight`, `hi4d_pair16_jump`, `hi4d_pair17_dance`, `hi4d_pair19_piggyback`).
- [ ] Record baseline vs tuned deltas (PSNR, SSIM, LPIPS) as the first decision snapshot.
- [ ] Decide the next ablation direction (more scenes, camera/frame sampling policy, reference-view strategy).
