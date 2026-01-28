## Pose ablation plan (config-only, train.yaml)

This plan is structured to mirror the thesis narrative:
1) establish a naive baseline that **does not** use interpenetration, depth-order, or confidence-guided optimization,  
2) add each component **one at a time**,  
3) then test SMPL-X parameter subsets to understand which degrees of freedom matter.

---

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

### A) Naive baseline search