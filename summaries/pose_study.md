# Init Ablation Study Summary

## NVS (Hi4D avg), sorted by PSNR (higher is better)

| rank | exp | PSNR | SSIM | LPIPS | status |
|---:|---|---:|---:|---:|---|
| 1 | v925_depth10 | 20.3000 | 0.9270 | 0.0867 | ok |
| 2 | v934_acapM005 | 20.3000 | 0.9270 | 0.0876 | ok |
| 3 | v936_lrLow | 20.3000 | 0.9270 | 0.0852 | ok |
| 4 | v920_conf075 | 20.2000 | 0.9260 | 0.0862 | ok |
| 5 | v921_conf085 | 20.2000 | 0.9270 | 0.0849 | ok |
| 6 | v922_confupd1 | 20.2000 | 0.9270 | 0.0862 | ok |
| 7 | v923_confupd10 | 20.2000 | 0.9270 | 0.0863 | ok |
| 8 | v924_depth00 | 20.2000 | 0.9270 | 0.0863 | ok |
| 9 | v926_sil2 | 20.2000 | 0.9260 | 0.0859 | ok |
| 10 | v927_sil8 | 20.2000 | 0.9270 | 0.0866 | ok |
| 11 | v928_poseReg1e5 | 20.2000 | 0.9270 | 0.0862 | ok |
| 12 | v929_poseReg5e4 | 20.2000 | 0.9270 | 0.0862 | ok |
| 13 | v932_inter00 | 20.2000 | 0.9270 | 0.0862 | ok |
| 14 | v933_inter005 | 20.2000 | 0.9270 | 0.0862 | ok |
| 15 | v935_acapM02 | 20.1000 | 0.9260 | 0.0839 | ok |
| 16 | v937_lrHigh | 20.1000 | 0.9250 | 0.0858 | ok |

## Pose (SMPL-X, Hi4D avg), sorted by MPJPE (lower is better)

| rank | exp | MPJPE_mm | MVE_mm | PCDR | status |
|---:|---|---:|---:|---:|---|
| 1 | v937_lrHigh | 77.4559 | 63.8353 | 0.6173 | ok |
| 2 | v926_sil2 | 77.6030 | 63.4122 | 0.6103 | ok |
| 3 | v920_conf075 | 77.6660 | 63.4545 | 0.6123 | ok |
| 4 | v933_inter005 | 77.6733 | 63.4739 | 0.6123 | ok |
| 5 | v935_acapM02 | 77.6948 | 63.4819 | 0.6123 | ok |
| 6 | v924_depth00 | 77.6949 | 63.4851 | 0.6123 | ok |
| 7 | v934_acapM005 | 77.6949 | 63.4816 | 0.6123 | ok |
| 8 | v922_confupd1 | 77.6970 | 63.4854 | 0.6123 | ok |
| 9 | v923_confupd10 | 77.6970 | 63.4880 | 0.6123 | ok |
| 10 | v932_inter00 | 77.6972 | 63.4836 | 0.6123 | ok |
| 11 | v928_poseReg1e5 | 77.6987 | 63.4866 | 0.6123 | ok |
| 12 | v929_poseReg5e4 | 77.6994 | 63.4874 | 0.6123 | ok |
| 13 | v921_conf085 | 77.7188 | 63.5067 | 0.6123 | ok |
| 14 | v927_sil8 | 77.8420 | 63.6250 | 0.6123 | ok |
| 15 | v925_depth10 | 78.0211 | 63.7859 | 0.6123 | ok |
| 16 | v936_lrLow | 78.5362 | 64.0736 | 0.6173 | ok |

## Interpretation

This section ties each ablation to the hypotheses in `schedule.sh` and summarizes what the numbers support (or fail to support).

**Overall best/worst**: NVS PSNR peaks at v925_depth10 (20.3000) and bottoms at v935_acapM02 (20.1000), while pose MPJPE ranges from best v937_lrHigh (77.4559 mm) to worst v936_lrLow (78.5362 mm). The pose spread (~1.08 mm) is larger than NVS spread (~0.20 dB), consistent with “pose is the main lever” right now.

**Hypothesis 1: reliability threshold sweep.**  
Thresholds 0.75 vs 0.85 produce identical NVS PSNR (20.2000) and only a tiny MPJPE shift (77.6660 → 77.7188, +0.0528 mm). This does not support a strong sensitivity to threshold within [0.75, 0.85]; the reliability signal may be too weak/flat or other losses dominate.

**Hypothesis 2: reliability update cadence.**  
Updating confidence every 1 vs 10 epochs yields no measurable difference in PSNR or MPJPE (both identical within rounding). This suggests cadence is not a key knob for pose quality under current scoring.

**Hypothesis 3: depth supervision strength.**  
Depth weight 0.0 vs 1.0 leaves NVS unchanged (20.2000 vs 20.3000) but worsens pose slightly at 1.0 (77.6949 → 78.0211, +0.3262 mm). This weakly supports “depth loss can hurt pose when overweighted,” though effect is modest.

**Hypothesis 4: silhouette loss strength.**  
Increasing silhouette weight from 2.0 to 8.0 does not move NVS but degrades MPJPE (77.6030 → 77.8420, +0.2390 mm). This supports the hypothesis that too-strong silhouette over-constrains pose.

**Hypothesis 5: pose regularization.**  
Reg weight 1e-5 vs 5e-4 yields essentially identical MPJPE (77.6987 → 77.6994). This does not support a strong effect from L2 regularization in this range.

**Hypothesis 7: interpenetration loss.**  
Adding interpenetration weight 0.05 gives a tiny MPJPE improvement (77.6972 → 77.6733, -0.0239 mm). This is directionally positive but extremely small, suggesting the weight or formulation is still too weak.

**Hypothesis 8: ACAP margin.**  
Changing margin 0.005 → 0.02 leaves MPJPE unchanged (~77.6949). This suggests ACAP margin is not a primary driver of pose in this range.

**Hypothesis 9: learning rates.**  
Lowering both pose/GS LR (5e-5) improves NVS (20.3000) but substantially worsens pose (78.5362). Higher LR (2e-4) gives the best pose but slightly worse NVS (20.1000). This supports the idea that pose benefits from higher LR, while NVS is less sensitive or may prefer lower LR for stability.

## New hypotheses (focus on pose)

### Knobs we can tune now (train.yaml only)

1) **Per-scene reliability variance needs adaptive thresholds.**  
   Hypothesis: fixed confidence threshold under-selects reliable frames for pair19 and over-selects for others.  
   Ideas: use per-scene (or per-epoch) adaptive thresholds targeting a fixed reliable ratio (e.g., 70–85%).

2) **Betas updates may be too aggressive for low-reliability scenes.**  
   Hypothesis: in scenes with low initial reliability, betas updates amplify errors.  
   Ideas: freeze betas for first N pose epochs, or scale betas LR by reliability ratio.

3) **Stronger interpenetration loss may be needed to affect pose.**  
   Hypothesis: interpenetration is too small relative to total loss (~0.04–0.06 vs 0.7–0.9), so it does not shape pose.  
   Ideas: increase loss_weights.interpenetration (e.g., 5–10×) and/or adjust interpenetration_margin/max_points.

4) **Pose tuning schedule could benefit from a contact-focused second pass.**  
   Hypothesis: once global pose improves, a short additional pose-only fine-tuning on contact-heavy frames improves MPJPE and PCDR.  
   Ideas: add a short second pose phase before GS with higher depth-order penalty (if implemented) or re-run pose tuning later with stricter reliability.

### Requires adding new logic / loss terms

1) **Contact-aware pose refinement is the bottleneck.**  
   Hypothesis: frames with close contact/occlusion drive most MPJPE error and hurt PCDR; explicitly handling these frames will improve pose and PCDR.  
   Ideas: up-weight contact frames, add occlusion-aware silhouette loss, or add a contact-consistency term when annotations exist.

2) **Reliability should be contact-aware, not just mask IoU.**  
   Hypothesis: current reliability scoring is too lenient for occluded/contact frames (especially pair19), so unreliable contact frames still update pose/betas.  
   Ideas: augment reliability with per-person depth ordering consistency or silhouette boundary mismatch; apply stricter thresholds for contact frames.

3) **PCDR can be improved by emphasizing depth ordering during pose phase.**  
   Hypothesis: PCDR degrades because pose tuning optimizes 2D appearance without explicitly preserving depth relations.  
   Ideas: introduce/strengthen a depth-order or PCDR-style penalty using posed SMPL-X depth, gated to contact frames.

4) **Use temporal consistency on pose deltas in occluded frames.**  
   Hypothesis: occluded/contact frames are unstable; a temporal smoothness prior on pose deltas will reduce jitter and MPJPE.  
   Ideas: add an L2 temporal smoothness loss on pose deltas for adjacent frames.

## Hypothesis-driven ablation plan (config-only, train.yaml)

This plan is structured to mirror the thesis narrative:
1) establish a naive baseline that **does not** use interpenetration, depth-order, or confidence-guided optimization,  
2) add each component **one at a time**,  
3) then test SMPL-X parameter subsets to understand which degrees of freedom matter.

**Note:** avoid repeating ablations already run (e.g., confidence_update_every sweep). Focus on new combinations or unexplored ranges.

### A) Naive baseline (no interpenetration, no depth-order, no confidence guidance)
**Hypothesis A:** Even without additional constraints, pose tuning should improve over initial estimates when trained on monocular RGB + masks + depth.  
**Expected outcome + metric:** MPJPE ↓, MVE ↓ (primary); PCDR ↑ (secondary).  
**Config:**  
- `enable_alternate_opt=false`  
- `loss_weights.interpenetration=0.0`  
- `loss_weights.depth_order=0.0`  
- Suggested stable core knobs: `pose_tuning.lr=2e-4`, `lr=2e-4`, `loss_weights.sil=2.0`, `loss_weights.depth=0.5`  

**Optional stability knobs for A (only if needed):**  
- `pose_tuning.reg_w`: `{1e-5, 1e-4, 3e-4}`  
- `grad_clip`: `{0.05, 0.1, 0.2}`  
- `loss_weights.rgb`: `{10.0, 20.0, 30.0}`  
- `loss_weights.ssim`: `{0.2, 0.4, 0.8}`

### B) Add interpenetration loss
**Hypothesis B:** A nonzero interpenetration loss reduces collisions and improves pose (MPJPE/PCDR) without hurting stability.  
**Expected outcome + metric:** MPJPE ↓ and PCDR ↑ (primary), interpenetration loss ↓ (aux).  
**Config:**  
- same as A, plus `loss_weights.interpenetration={0.05, 0.1}`  
- optional shape of loss: `interpenetration_margin={0.005, 0.01, 0.02}`, `interpenetration_max_points={1000, 2000, 5000}`

### C) Add depth-order loss
**Hypothesis C:** Enforcing per-person depth ordering improves PCDR and pose on contact/occlusion frames.  
**Expected outcome + metric:** PCDR ↑ (primary), MPJPE ↓ on contact frames (secondary).  
**Config:**  
- same as A, plus `loss_weights.depth_order={0.05, 0.1}`  
- keep `loss_weights.depth` low to isolate ordering effects (e.g., `depth=0.0` or `0.1`)

### D) Add confidence-guided optimization (alternate opt)
**Hypothesis D:** Filtering unreliable frames improves pose by preventing bad updates from noisy masks/occlusions.  
**Expected outcome + metric:** MPJPE ↓ and variance across scenes ↓ (primary); reliable ratio ↑ (aux).  
**Config:**  
- same as A, plus:  
  - `enable_alternate_opt=true`  
  - `confidence_threshold={0.75, 0.85}` (skip values already tested if rerun)  
  - `confidence_update_every=5` (keep fixed to avoid repeating cadence sweep)

### E) SMPL-X params ablation (config-only)
**Hypothesis E:** Only a subset of SMPL-X parameters is necessary; removing low-impact DOFs improves stability without hurting pose.
**Expected outcome + metric:** MPJPE ↓ or ≈ (primary) with lower variance across runs (secondary).  

Use `pose_tuning.params` variants (all config-only):
1) **Core (current default)**  
   `["root_pose", "body_pose", "trans", "betas"]`
2) **No betas**  
   `["root_pose", "body_pose", "trans"]`
3) **No hands**  
   `["root_pose", "body_pose", "trans", "betas"]` (exclude `lhand_pose`, `rhand_pose`)
4) **Add hands**  
   `["root_pose", "body_pose", "lhand_pose", "rhand_pose", "trans", "betas"]`
5) **Full pose (no betas)**  
   `["root_pose", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "lhand_pose", "rhand_pose", "trans"]`
6) **Translation-only sanity**  
   `["trans"]`

### F) Combine components (best observed settings)
**Hypothesis F:** Combining the strongest individual components yields the best pose overall.  
**Expected outcome + metric:** Best MPJPE/MVE and highest PCDR among all runs.  
**Config (example):**  
- `enable_alternate_opt=true`  
- `loss_weights.interpenetration=0.1`  
- `loss_weights.depth_order=0.1`  
- `confidence_threshold=0.85`, `confidence_update_every=5`  
- keep stable core knobs: `pose_tuning.lr=2e-4`, `lr=2e-4`, `loss_weights.sil=2.0`
