# On Limitations of PromptHMR-Vid (Observed in HI4D `pair19_2/piggyback19`)

## Context
We investigated two separate failure modes in a two-person interaction scene where one person jumps on the other's back.
Important scope note: these issues were observed in this single scene and were not reproduced in other tested scenes.

### A) Translation/depth issue (primary)
- In 2D overlays, silhouettes can look acceptable for both persons.
- In 3D, the jumping person is consistently placed too far from the camera (depth error).
- The issue appears already in camera-space outputs (`smplx_cam`), before world conversion.
- The other person is generally translated reasonably.
- Similar failure was **not** observed in easier interactions (hugging, dancing, etc.).

### B) Shape issue (separate)
- There is also a shape mismatch for one person in this scene.
- This was analyzed separately from translation, and should not be conflated with the depth error above.

---

## What We Tested

### 1) Translation source ablation: image model vs video model
- Change: use PromptHMR image-model translation (`tracks[k]['smplx_transl']`) instead of GVHMR video translation (`pred_smpl_params_incam['transl']`).
- Result: **slight but consistent improvement** for the jumping person.
- Interpretation: a significant part of the error is introduced by video translation refinement (or by its conditioning in this scene).

### 2) Intrinsics ablation: estimated vs GT intrinsics
- Estimated intrinsics were notably different from GT (focal was higher by ~14.45%).
- Injecting GT intrinsics improved translation quality.
- Result: **clear positive impact**, though not a complete fix.
- Interpretation: translation parameterization is sensitive to focal accuracy; calibration error amplifies depth error.

### 3) Beta aggregation in world stage (shape-focused ablation)
- Compared temporal-mean beta vs first-frame beta.
- Result: this did **not** materially resolve the observed shape mismatch for the problematic person.
- Interpretation: the shape issue is unlikely to be explained only by the world-stage beta aggregation choice.

### 4) Disable bbox prompt to image model
- Added a config toggle and tested with bbox prompt disabled.
- Result: did **not** resolve the jumping-person translation issue.
- Interpretation: image-model bbox prompting is not the dominant root cause here.

---

## What Worked Best (for translation issue)
The two highest-impact changes were:
1. Using image-model translation instead of video-model translation.
2. Using GT intrinsics when available.

These gave the strongest practical improvement among tested options.

---

## What Did Not Meaningfully Fix It
For translation:
- Disabling bbox prompt in image model inputs.

For shape:
- Switching beta aggregation from mean to first frame.

These did not remove the corresponding failure modes above.

---

## Known Limitations (Based on This Debugging)
Scope caveat: all limitations below are inferred from this one scene and should be treated as scene-specific evidence, not universal behavior.

1. **Translation fragility under close-contact, asymmetric motion**
   - PromptHMR-Vid translation can fail for one actor in heavy occlusion/contact scenarios.
   - The model appears more reliable on simpler interactions.

2. **High sensitivity to camera intrinsics (especially focal length)**
   - Translation decoding uses camera parameters; calibration bias directly affects depth.

3. **Per-track processing lacks explicit inter-person geometric constraints**
   - Each person is refined largely independently.
   - No explicit constraint enforces physically plausible relative depth/contact during piggyback-like motion.

4. **2D-consistent but 3D-inconsistent outcomes are possible**
   - Good silhouette overlap in image space does not guarantee correct 3D relative placement.

5. **A second, partly independent shape-estimation failure can co-occur**
   - In this scene, one-person shape mismatch persisted even when changing beta aggregation policy in world conversion.
   - This points to upstream estimation limits (image/video model outputs), not just world-stage post-processing.

---

## Practical Conclusion
For challenging contact interactions in this pipeline, translation estimates should be treated as uncertain.
For this specific translation failure mode, a practical operating point is:
- Prefer image-model translation when this failure mode appears.
- Use trusted intrinsics (GT/calibrated) whenever possible.

For the separate shape mismatch issue, changing world-stage beta averaging alone was not sufficient, suggesting the limitation is upstream in the pose/shape estimation stack.
