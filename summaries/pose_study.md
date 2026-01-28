## Pose ablation plan (config-only, train.yaml)

We first fix the novel-view synthesis pipeline to the best configuration found in the NVS ablations (DiFix refinement with the previous camera as source), and then vary pose-tuning components. This keeps the NVS method constant so any changes in NVS can be attributed to improved pose rather than pipeline changes.
We also hypothesize that improved pose should translate into improved NVS quality, so we will report both pose metrics (MPJPE/MVE/PCDR) and NVS metrics (PSNR/SSIM/LPIPS) for each experiment.

This plan is structured to mirror the thesis narrative:
1) establish a naive baseline that **does not** use interpenetration, depth-order, or confidence-guided optimization,  
2) add each component **one at a time**,  
3) then test SMPL-X parameter subsets to understand which degrees of freedom matter.

---

### A) Naive baseline (no interpenetration, no depth-order, no confidence guidance)
**Hypothesis A:** Even without additional constraints, pose tuning should improve over initial estimates when trained on monocular RGB + masks + depth.  
**Expected outcome + metric:** MPJPE ↓, MVE ↓ (primary); PCDR ↑ (secondary).  
**Baseline Config:**  
- `enable_alternate_opt=false`  
- `loss_weights.interpenetration=0.0`  
- `loss_weights.depth_order=0.0`  
- `pose_tuning.lr=2e-4`
- `lr=2e-4`
- `loss_weights.sil=2.0`
- `loss_weights.depth=0.5`  

**Parameters and ranges covered by the scheduled experiments (Phase A):**  
- `loss_weights.rgb`: `{10.0, 20.0, 30.0}`  
- `loss_weights.ssim`: `{0.2, 0.4, 0.8}`  
- `regularization.acap_margin`: `{0.03, 0.05}` (expanded beyond the default 0.01)  
- `lr` (3DGS): `{5e-5, 1e-4, 2e-4}`  
- `pose_tuning.lr`: `{1e-4, 1.5e-4, 2e-4}`  


---

### B) Add interpenetration loss
**Hypothesis B:** A nonzero interpenetration loss reduces collisions and improves pose (MPJPE/PCDR) without hurting stability.  
**Expected outcome + metric:** MPJPE ↓ and PCDR ↑ (primary), interpenetration loss ↓ (aux).  
**Config:**  
- same as A, plus `loss_weights.interpenetration={0.05, 0.1}`  
- optional shape of loss: `interpenetration_margin={0.005, 0.01, 0.02}`, `interpenetration_max_points={1000, 2000, 5000}`

---

### C) Add depth-order loss
**Hypothesis C:** Enforcing per-person depth ordering improves PCDR and pose on contact/occlusion frames.  
**Expected outcome + metric:** PCDR ↑ (primary), MPJPE ↓ on contact frames (secondary).  
**Config:**  
- same as A, plus `loss_weights.depth_order={0.05, 0.1}`  
- keep `loss_weights.depth` low to isolate ordering effects (e.g., `depth=0.0` or `0.1`)

---

### D) Add confidence-guided optimization (alternate opt)
**Hypothesis D:** Filtering unreliable frames improves pose by preventing bad updates from noisy masks/occlusions.  
**Expected outcome + metric:** MPJPE ↓ and variance across scenes ↓ (primary); reliable ratio ↑ (aux).  
**Config:**  
- same as A, plus:  
  - `enable_alternate_opt=true`  
  - `confidence_threshold={0.75, 0.85}` (skip values already tested if rerun)  
  - `confidence_update_every=5` (keep fixed to avoid repeating cadence sweep)

---

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

---

### F) Combine components (best observed settings)
**Hypothesis F:** Combining the strongest individual components yields the best pose overall.  
**Expected outcome + metric:** Best MPJPE/MVE and highest PCDR among all runs.  
**Config (example):**  
- `enable_alternate_opt=true`  
- `loss_weights.interpenetration=0.1`  
- `loss_weights.depth_order=0.1`  
- `confidence_threshold=0.85`, `confidence_update_every=5`  
- keep stable core knobs: `pose_tuning.lr=2e-4`, `lr=2e-4`, `loss_weights.sil=2.0`

## Results of the ablation study

### Section A results

#### NVS (Hi4D avg across scenes, sorted by PSNR)
| Rank | Exp | PSNR | SSIM | LPIPS | Status |
|---:|:---|---:|---:|---:|:---|
| 1 | v969_A10_lr5e5 | 20.3000 | 0.9270 | 0.0869 | ok |
| 2 | v961_A3_rgb10 | 20.2000 | 0.9260 | 0.0855 | ok |
| 3 | v970_A10_lr1e4 | 20.2000 | 0.9260 | 0.0862 | ok |
| 4 | v960_A0_baseline | 20.1000 | 0.9260 | 0.0857 | ok |
| 5 | v962_A3_rgb20 | 20.1000 | 0.9260 | 0.0858 | ok |
| 6 | v963_A3_rgb30 | 20.1000 | 0.9250 | 0.0858 | ok |
| 7 | v964_A4_ssim02 | 20.1000 | 0.9250 | 0.0871 | ok |
| 8 | v965_A4_ssim04 | 20.1000 | 0.9250 | 0.0859 | ok |
| 9 | v966_A4_ssim08 | 20.1000 | 0.9260 | 0.0844 | ok |
| 10 | v971_A10_lr2e4 | 20.1000 | 0.9250 | 0.0859 | ok |
| 11 | v972_A11_poseLR1e4 | 20.1000 | 0.9250 | 0.0862 | ok |
| 12 | v973_A11_poseLR15e4 | 20.1000 | 0.9260 | 0.0857 | ok |
| 13 | v974_A11_poseLR2e4 | 20.1000 | 0.9250 | 0.0858 | ok |
| 14 | v967_A9_acapM03 | 20.0000 | 0.9250 | 0.0818 | ok |
| 15 | v968_A9_acapM05 | 19.9000 | 0.9250 | 0.0800 | ok |

#### Pose (SMPL-X, Hi4D avg across scenes, sorted by MPJPE)
| Rank | Exp | MPJPE_mm | MVE_mm | PCDR | Status |
|---:|:---|---:|---:|---:|:---|
| 1 | v963_A3_rgb30 | 77.3154 | 63.6937 | 0.6113 | ok |
| 2 | v966_A4_ssim08 | 77.4432 | 63.7891 | 0.6153 | ok |
| 3 | v962_A3_rgb20 | 77.4816 | 63.8327 | 0.6096 | ok |
| 4 | v969_A10_lr5e5 | 77.4925 | 63.8471 | 0.6153 | ok |
| 5 | v965_A4_ssim04 | 77.4990 | 63.8509 | 0.6096 | ok |
| 6 | v967_A9_acapM03 | 77.5001 | 63.8537 | 0.6096 | ok |
| 7 | v971_A10_lr2e4 | 77.5023 | 63.8526 | 0.6096 | ok |
| 8 | v970_A10_lr1e4 | 77.5044 | 63.8567 | 0.6133 | ok |
| 9 | v974_A11_poseLR2e4 | 77.5060 | 63.8556 | 0.6153 | ok |
| 10 | v960_A0_baseline | 77.5081 | 63.8628 | 0.6116 | ok |
| 11 | v968_A9_acapM05 | 77.5126 | 63.8689 | 0.6113 | ok |
| 12 | v973_A11_poseLR15e4 | 77.5396 | 63.6046 | 0.6109 | ok |
| 13 | v964_A4_ssim02 | 77.5404 | 63.8957 | 0.6116 | ok |
| 14 | v972_A11_poseLR1e4 | 77.7286 | 63.5130 | 0.6103 | ok |
| 15 | v961_A3_rgb10 | 77.9382 | 64.2321 | 0.6133 | ok |

**Baseline selection for downstream ablations (joint NVS + pose):**
- We pick settings that are strong on **both** NVS (PSNR/LPIPS) and pose (MPJPE/MVE), not just one side.
- `v969_A10_lr5e5` is the **top PSNR** and also ranks within the top pose performers → strong joint candidate.
- `v966_A4_ssim08` is top‑2 for pose and has competitive NVS → supports higher SSIM weight without NVS penalty.
- `v963_A3_rgb30` is best MPJPE but doesn’t improve NVS vs baseline → keep as a pose‑only reference, not the joint baseline.
- `pose_tuning.lr` sweep shows no improvement over 2e-4 → keep `pose_tuning.lr=2e-4`.
- `acap_margin=0.03` improves LPIPS and is neutral on pose → safe for joint baseline.

**Recommended baseline knobs to carry forward (joint objective):**
- `lr=5e-5` (3DGS LR)
- `loss_weights.ssim=0.8`
- `regularization.acap_margin=0.03`
- keep: `pose_tuning.lr=2e-4`, `loss_weights.rgb=20.0` (default), `loss_weights.sil=2.0`, `loss_weights.depth=0.5`, `enable_alternate_opt=false`

**Pose‑only reference (for comparison):**
- `loss_weights.rgb=30.0` (v963_A3_rgb30)

If you want the most conservative narrative, you can keep `v960_A0_baseline` as the primary reference and treat the joint baseline above as an informed upgrade.
