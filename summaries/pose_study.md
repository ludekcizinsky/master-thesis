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
| 14 | v975_A0b_baseline_combo | 20.1000 | 0.9260 | 0.0809 | ok |
| 15 | v967_A9_acapM03 | 20.0000 | 0.9250 | 0.0818 | ok |
| 16 | v968_A9_acapM05 | 19.9000 | 0.9250 | 0.0800 | ok |

#### Pose (SMPL-X, Hi4D avg across scenes, sorted by MPJPE)
| Rank | Exp | MPJPE_mm | MVE_mm | PCDR | Status |
|---:|:---|---:|---:|---:|:---|
| 1 | v963_A3_rgb30 | 77.3154 | 63.6937 | 0.6113 | ok |
| 2 | v975_A0b_baseline_combo | 77.4410 | 63.7939 | 0.6133 | ok |
| 3 | v966_A4_ssim08 | 77.4432 | 63.7891 | 0.6153 | ok |
| 4 | v962_A3_rgb20 | 77.4816 | 63.8327 | 0.6096 | ok |
| 5 | v969_A10_lr5e5 | 77.4925 | 63.8471 | 0.6153 | ok |
| 6 | v965_A4_ssim04 | 77.4990 | 63.8509 | 0.6096 | ok |
| 7 | v967_A9_acapM03 | 77.5001 | 63.8537 | 0.6096 | ok |
| 8 | v971_A10_lr2e4 | 77.5023 | 63.8526 | 0.6096 | ok |
| 9 | v970_A10_lr1e4 | 77.5044 | 63.8567 | 0.6133 | ok |
| 10 | v974_A11_poseLR2e4 | 77.5060 | 63.8556 | 0.6153 | ok |
| 11 | v960_A0_baseline | 77.5081 | 63.8628 | 0.6116 | ok |
| 12 | v968_A9_acapM05 | 77.5126 | 63.8689 | 0.6113 | ok |
| 13 | v973_A11_poseLR15e4 | 77.5396 | 63.6046 | 0.6109 | ok |
| 14 | v964_A4_ssim02 | 77.5404 | 63.8957 | 0.6116 | ok |
| 15 | v972_A11_poseLR1e4 | 77.7286 | 63.5130 | 0.6103 | ok |
| 16 | v961_A3_rgb10 | 77.9382 | 64.2321 | 0.6133 | ok |

**Section A recommendation update (baseline-combo verification):**
- `v975_A0b_baseline_combo` is **jointly strong**: MPJPE **77.4410** (better than baseline 77.5081) and PSNR **20.1** (ties top cluster), with improved LPIPS **0.0809**.
- This beats or matches the top single‑knob candidates on the joint objective without sacrificing pose, so we **accept the combo as the new baseline** for downstream ablations.

**Baseline to carry forward:**
- `lr=5e-5`, `loss_weights.ssim=0.8`, `regularization.acap_margin=0.03`
- keep: `pose_tuning.lr=2e-4`, `loss_weights.rgb=20.0`, `loss_weights.sil=2.0`, `loss_weights.depth=0.5`, `enable_alternate_opt=false`

### Sections B, C, D, E results

**Baseline reference (v975_A0b_baseline_combo):**
| Exp | PSNR | SSIM | LPIPS | MPJPE_mm | MVE_mm | PCDR |
|:---|---:|---:|---:|---:|---:|---:|
| v975_A0b_baseline_combo | 20.1000 | 0.9260 | 0.0809 | 77.4410 | 63.7939 | 0.6133 |

#### B) Interpenetration loss
| Exp | PSNR | SSIM | LPIPS | MPJPE_mm | MVE_mm | PCDR |
|:---|---:|---:|---:|---:|---:|---:|
| v976_B_inter01 | 20.1000 | 0.9260 | 0.0809 | 77.5931 | 63.4099 | 0.6123 |
| v977_B_inter02 | 20.1000 | 0.9260 | 0.0810 | 77.6678 | 63.4741 | 0.6083 |
| v978_B_inter04 | 20.1000 | 0.9260 | 0.0809 | 77.7702 | 63.5668 | 0.6083 |

**Interpretation (B):**
- Best pose (MPJPE): `v976_B_inter01` (77.5931 vs baseline 77.4410).
- Best NVS PSNR: `v976_B_inter01` (20.1000 vs baseline 20.1000).
- Best LPIPS: `v976_B_inter01` (0.0809 vs baseline 0.0809).
- **Net effect:** no improvement over the v975 baseline; keep this component off for now.

#### C) Depth-order loss
| Exp | PSNR | SSIM | LPIPS | MPJPE_mm | MVE_mm | PCDR |
|:---|---:|---:|---:|---:|---:|---:|
| v979_C_dorder01 | 20.0000 | 0.9250 | 0.0818 | 77.9943 | 63.6916 | 0.6099 |
| v980_C_dorder02 | 20.0000 | 0.9250 | 0.0818 | 78.3018 | 63.9591 | 0.6099 |
| v981_C_dorder04 | 20.0000 | 0.9240 | 0.0819 | 78.5956 | 64.2230 | 0.6113 |

**Interpretation (C):**
- Best pose (MPJPE): `v979_C_dorder01` (77.9943 vs baseline 77.4410).
- Best NVS PSNR: `v979_C_dorder01` (20.0000 vs baseline 20.1000).
- Best LPIPS: `v979_C_dorder01` (0.0818 vs baseline 0.0809).
- **Net effect:** no improvement over the v975 baseline; keep this component off for now.

#### D) Confidence-guided optimization
| Exp | PSNR | SSIM | LPIPS | MPJPE_mm | MVE_mm | PCDR |
|:---|---:|---:|---:|---:|---:|---:|
| v982_D_conf075_fix | 20.1000 | 0.9260 | 0.0808 | 77.6290 | 63.4303 | 0.6123 |
| v983_D_conf085_fix | 20.1000 | 0.9260 | 0.0803 | 77.6801 | 63.4801 | 0.6123 |
| v984_D_conf_median | 20.1000 | 0.9260 | 0.0810 | 77.6414 | 63.4455 | 0.6123 |

**Interpretation (D):**
- Best pose (MPJPE): `v982_D_conf075_fix` (77.6290 vs baseline 77.4410).
- Best NVS PSNR: `v982_D_conf075_fix` (20.1000 vs baseline 20.1000).
- Best LPIPS: `v983_D_conf085_fix` (0.0803 vs baseline 0.0809).
- **Net effect:** slight LPIPS gain but worse pose; not a clear win over baseline.

#### E) SMPL-X params ablation
| Exp | PSNR | SSIM | LPIPS | MPJPE_mm | MVE_mm | PCDR |
|:---|---:|---:|---:|---:|---:|---:|
| v985_E_params_nobetas | 20.1000 | 0.9260 | 0.0811 | 77.6851 | 63.4863 | 0.6123 |
| v986_E_params_addhands | 20.1000 | 0.9260 | 0.0809 | 77.5986 | 63.4009 | 0.6123 |
| v987_E_params_fullpose_nobetas | 20.1000 | 0.9260 | 0.0810 | 77.6383 | 63.4802 | 0.6123 |

**Interpretation (E):**
- Best pose (MPJPE): `v986_E_params_addhands` (77.5986 vs baseline 77.4410).
- Best NVS PSNR: `v985_E_params_nobetas` (20.1000 vs baseline 20.1000).
- Best LPIPS: `v986_E_params_addhands` (0.0809 vs baseline 0.0809).
- **Net effect:** no improvement over the v975 baseline; keep this component off for now.

**Overall B–E takeaway:** none of the tested additions beat the v975 baseline on the joint objective (pose + NVS). Keep the v975 defaults and only revisit these components if we expand to different scenes or adjust loss scaling.
