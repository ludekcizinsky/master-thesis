## Study Versions

This memo is the single source of truth for study designs and iteration history.
Each version below should document:
1. objective,
2. data split (train/test scenes),
3. key decisions,
4. expected outputs.

## V0 Plan (Ultra-Fast Sanity Check)

Hypothesis:
> with a tiny in-domain train subset, we can validate the full DiFix tuning pipeline end-to-end and get first directional signal quickly

### Objective
Run the fastest possible training/eval loop to validate:
1. data generation pipeline,
2. DiFix tuning integration,
3. downstream NVS evaluation wiring.

### Train Split (V0)
Use only 2 scenes (non-eval pairs):
1. `hi4d_pair00_dance`
2. `hi4d_pair01_talk`

### Test Split (V0)
Keep fixed evaluation scenes:
1. `hi4d_pair15_fight`
2. `hi4d_pair16_jump`
3. `hi4d_pair17_dance`
4. `hi4d_pair19_piggyback`

### Why this split
1. no leakage with eval scenes,
2. minimal time to first result,
3. includes two different motion profiles (`dance` vs `talk`).

### Expected Output
1. successful generation of DiFix training samples (`image`, `target_image`, `ref_image`),
2. successful DiFix fine-tuning run,
3. NVS metrics on the fixed eval scenes for baseline comparison.

## V1 Plan (First End-to-End Version)

Hypothesis:
> we should be able to obtain better nvs results on the test subset of hi4d data by tunning on the train subset of hi4d data

### 1. Core Goal
Train a DiFix model on HI4D in-domain pairs so that NVS quality improves on held-out HI4D test scenes/cameras.

### 2. V1 Decisions
1. Use **imperfect poses/rendering pipeline output** to build the input image (`image`), because this is what we have at inference time.
2. Use **same-timestamp source-camera GT** as reference (`ref_image`):
   - target sample is `(cam_target, t)`
   - reference is `(cam_source, t)`

### 3. Sample Construction
For each valid `(scene, t, cam_target)`:
1. `target_image`: GT frame from `cam_target` at time `t`.
2. `image`: imperfect rendered frame at same `cam_target, t` (from current reconstruction state / pipeline).
3. `ref_image`: GT source view at `cam_source, t`.
4. `prompt`: `"remove degradation"`.

JSON entry format:
```json
{
  "image": ".../imperfect/cam_target/000123.png",
  "target_image": ".../gt/cam_target/000123.jpg",
  "ref_image": ".../gt/cam_source/000123.jpg",
  "prompt": "remove degradation"
}
```

### 4. Data Split (V1)
Use a strict split to avoid leakage:
1. Train split: HI4D train subset scenes.
2. Test split: HI4D test subset scenes (no scene overlap with train).
3. Current evaluation scenes:
   - `hi4d_pair15_fight`
   - `hi4d_pair16_jump`
   - `hi4d_pair17_dance`
   - `hi4d_pair19_piggyback`

### 5. Filtering Rules (V1)
Keep only samples where:
1. all three files exist (`image`, `target_image`, `ref_image`),
2. frame is not in `skip_frames`,
3. masks/people visibility are valid enough for supervision.

### 6. Evaluation Protocol
After tuning:
1. plug tuned DiFix into the same NVS pipeline,
2. evaluate on HI4D test subset,
3. compare against baseline DiFix (or no tuning) with same metrics:
   - PSNR, SSIM, LPIPS (and downstream task metrics if needed).

### 7. Out of Scope for V1 (Later Ablations)
1. GT-pose vs imperfect-pose rendering for data generation.
2. Alternative reference selection:
   - temporal neighbor reference (`t±k`),
   - mixed strategy (same-time + temporal offsets).
