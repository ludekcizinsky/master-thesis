# Master’s Thesis Defense (25–30 min) — Presentation Outline

Thesis topic (1 sentence):
Hybrid monocular 4D reconstruction of **multi-person, human-centric dynamic scenes** using **feed-forward priors** (camera + SMPL-X + masks + depth + canonical 3DGS init) plus **lightweight per-scene optimization** of an explicit **3D Gaussian Splatting** representation, with **DiFix**-refined synthesized training views for denser multi-view supervision.

## Target length and pacing

- **25 min talk + ~5 min Q&A**: prioritize *problem → method overview → key results → limitations/future work*.
- **30 min talk + ~5–10 min Q&A**: add *ablations + deeper method details + more qualitative*.

Rule of thumb: **~1 minute/slide**, so aim for **~22–26 slides total** including backup; **~16–20 main slides**.

## Single-sentence narrative (keep repeating)

Monocular multi-person 4D is ill-posed; pure feed-forward is fast but imperfect, pure optimization is accurate but slow—**a hybrid pipeline can reach competitive novel-view quality at tens-of-minutes runtime** by anchoring explicit 3DGS in SMPL-X motion priors and densifying supervision with geometry-grounded diffusion refinement.

## Agenda (time budget)

### 25-minute version (recommended)
- 0:00–2:00 — Motivation + problem definition + why it’s hard
- 2:00–4:00 — Related work + where the gap is (feed-forward vs optimization)
- 4:00–6:00 — Contributions + high-level method overview
- 6:00–15:00 — Method (representation + preprocessing + training schedule + DiFix view synthesis)
- 15:00–21:00 — Experiments (setup + results across NVS / pose / mesh / speed)
- 21:00–23:30 — Limitations + failure modes
- 23:30–25:00 — Conclusions + next steps

### 30-minute version (add-ons)
- +2–3 min: ablations (view synthesis components; # virtual cameras; pose tuning variants)
- +1–2 min: mesh extraction ablation (TSDF vs Marching Cubes) + qualitative

## Slide-by-slide outline (with suggested figures/tables)

### 1) Title (0:30)
- Title, name, advisors/lab, one-line thesis claim.

### 2) Motivation: “2D everywhere → 3D needed” (1:00)
- Applications: robotics / free-viewpoint video / content creation.
- Constraint: real-world data is mostly **monocular video**.

### 3) Problem statement + challenges (1:30)
- Input: monocular RGB video; output: **renderable 4D** multi-person scene in metric scale.
- Why hard: camera/object motion entanglement, occlusions, unseen regions, temporal stability, multi-person identity.

### 4) Where current methods sit (1:30)
- Feed-forward: fast, but artifacts in challenging interactions.
- Optimization-heavy: high quality but **hours→days** per scene.
- Show qualitative contrast: `report/figures/qual_feedforward_vs_ours.drawio.png`.

### 5) Goal + evaluation axes (0:45)
- What “success” means here:
  1) Novel view synthesis (appearance + consistency)
  2) Pose/motion quality (motion analysis)
  3) Mesh reconstruction (geometry proxy)
  4) Practical runtime

### 6) Contributions (1:00)
- Hybrid pipeline: priors + targeted optimization.
- Explicit per-person canonical **3DGS** aligned to **SMPL-X** motion.
- Geometry-grounded view densification: virtual cameras + **DiFix** refinement.
- Evaluation on Hi4D + MMM across the 4 axes.

### 7) Method overview (1:00)
- One picture slide: `report/figures/what_our_method_does_impress_fig.drawio.png`.
- Emphasize: “initialize strong; optimize light; supervise with synthesized views”.

### 8) Preprocessing outputs (1:30)
- What’s estimated off-the-shelf:
  - Human3R: cameras + SMPL-X in shared world frame
  - SAM3: instance masks + tracking
  - Depth Anything 3: depth (source view)
  - LHM: canonical neutral-pose per-person 3DGS init
- Show: `report/figures/preprocessing_overview.drawio.png`.

### 9) Representation choice: why explicit 3DGS (1:00)
- Pros: fast rendering, composability (humans + background), optimization efficiency.
- Con: needs multi-view supervision to avoid “single-view overfit”.
- Show failure mode: `report/figures/qual_poor_nvs_from_mono_trn.drawio.png`.

### 10) Canonical per-person 3DGS aligned to SMPL-X (2:00)
- Canonical (neutral pose) Gaussians anchored to dense SMPL-X surface points (offsets allow clothing deviation).
- Deformation via LBS/kinematics to each frame’s pose.
- Key message: **motion comes from SMPL-X**, appearance/shape from Gaussians.

### 11) Training view synthesis: why + what (1:00)
- Single video → sparse viewpoints → poor novel-view quality for explicit Gaussians.
- Strategy: synthesize **7 virtual cameras** around scene, progressive traversal.
- Show: `report/figures/example_of_progressive_trn_nv_gen.drawio.png` and `report/figures/example_of_all_cams_in_3d.png`.

### 12) DiFix refinement (1:30)
- For each target camera and timestamp:
  - render geometry-grounded initial novel view
  - refine with DiFix (prompt: “remove degradation”)
- Key claim: diffusion is used as *refiner*, not a geometry generator → fewer hallucinations.
- Show: `report/figures/single_nv_train_view_synthesis_diagram.drawio.png` (optionally `report/figures/qual_difix_comp.drawio.png` as extra).

### 13) Optimization schedule (1:30)
- Two-stage:
  1) **Pose tuning** (subset of SMPL-X params) with 3DGS fixed
  2) **3DGS appearance optimization** with poses/cameras fixed + synthesized views
- Losses: masked RGB + SSIM; silhouette/depth on source; canonical regularizers (ASAP/ACAP).
- Show (optional): `report/figures/train_step_example.drawio.png`.

### 14) Experimental setup (1:15)
- Datasets:
  - Hi4D: multi-view + GT meshes/poses; evaluate NVS + pose + recon
  - MMM: moving camera + 3–4 people; primarily recon
- Metrics (1 slide):
  - NVS: SSIM/PSNR/LPIPS
  - Pose: MPJPE/MVE + interaction metrics CD + PCDR
  - Recon: V-IoU, C-l2, P2S, NC (mesh from TSDF fusion)

### 15) Results: Novel view synthesis (2:00)
- Key quantitative: on Hi4D, **SSIM best (0.926)**; MultiPly best PSNR/LPIPS; overall “on par”.
- Table to show: Table `tab:nvs_results_hi4d`.
- Qualitative: `report/figures/qual_nvs_comp.drawio.png`.
- Talking point: sharper details can hurt PSNR while looking better.

### 16) Results: Pose quality (1:30)
- In SMPL-space, MPJPE gap vs MultiPly; competitive on MVE and **best CD (contact distance)**.
- In SMPL-X-space, pose tuning improves over Human3R init (MPJPE 80.7→77.4; MVE 65.9→63.8).
- Table to show: `tab:pose_results_hi4d`.
- Qualitative: `report/figures/qual_pose_comp.drawio.png`.

### 17) Results: Mesh reconstruction (1:30)
- Mesh extraction via **TSDF fusion of rendered depth** (approximate; sensitive to pose).
- Compared to MultiPly (implicit SDF), reconstruction metrics trail, especially on MMM.
- Table to show: `tab:reconstruction_results` (note: Hi4D P2S best for ours, but MMM degrades).
- Qualitative: `report/figures/qual_recon_comp.drawio.png`.

### 18) Results: Speed / practicality (1:00)
- On single V100:
  - ~14 min (50 frames), ~29 min (100), ~57 min (200) for DiFix+optimization (excluding preprocessing).
- Table to show: `tab:training_speed_breakdown`.
- Contrast: MultiPly reports ~day/person (not controlled, but highlights order-of-magnitude gap).

### 19) Takeaways + what the hybrid choice buys (0:45)
- Competitive NVS at practical runtime.
- Explicit representation is composable; can plug into larger systems (e.g., add static background).

### 20) Limitations (1:30)
- Multi-stage preprocessing brittleness (Human3R/SAM3/Depth/LHM failures propagate).
- Geometry: 3DGS optimized for rendering; mesh via TSDF is indirect → weaker recon metrics.
- Scope: no dynamic non-human objects/props; short sequences; heavy occlusions still hard.

### 21) Most direct next steps (1:00)
- Stronger pose estimation/refinement (e.g., add 2D keypoint guided refinement; use PromptHMR).
- Better view synthesis supervision (multi-reference DiFix; visibility-aware reference selection).
- Better canonical init (select/merge multiple LHM frames).

### 22) Conclusion (0:45)
- Restate: hybrid monocular 4D multi-person can be **fast** and **NVS-competitive**, but geometry + pose remain main bottlenecks.
- “If you remember one thing…”: geometry-grounded refinement + explicit 3DGS + SMPL-X priors = practical pipeline.

## Suggested backup slides (pick 4–8)

- Ablation: LHM init / DiFix contribution (Table around `tab:ablation...` in Experiments).
- Ablation: number of novel training cameras (show diminishing returns).
- Ablation: pose optimization variants (what helped vs didn’t).
- Mesh extraction: TSDF vs Marching Cubes + qualitative:
  - `report/figures/qual_3dgs_to_mesh.drawio.png`
- Failure cases / preprocessing failure modes (identity swaps, occlusion).
- “Why SMPL-X?” hand articulation example (from qualitative pose figure).

## Q&A prep: likely questions (with short answers)

- “Why not pure feed-forward?” → data scarcity + quality gap in multi-person interactions; hybrid is practical stepping stone + supervision generator.
- “Why 3DGS instead of implicit?” → rendering speed + composability + efficient optimization; but meshing is weaker.
- “Does DiFix hallucinate?” → geometry-anchored render constrains viewpoint/pose; DiFix mainly restores local texture/details.
- “What’s the main bottleneck today?” → pose under occlusion + mesh extraction; both directly impact geometry metrics.
