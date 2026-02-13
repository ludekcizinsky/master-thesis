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
