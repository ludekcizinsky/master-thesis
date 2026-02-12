# DiFix reference-view ablation study

This memo summarizes the motivation and design space for choosing a **reference view** when using DiFix to refine synthesized training views. The document is structured as a short chain of arguments: we first explain why training-view synthesis is needed, then describe the current pipeline at a high level, then break down the key reference-selection choices (camera and timestep). All of this context leads to the final sections, where we summarize the empirical takeaway and formulate the hypothesis to be tested in the ablation study.

## 1) Why we synthesize training views

We optimize an explicit, renderable 3D Gaussian Splatting (3DGS) scene representation from a **single monocular video**. With only one camera view, the optimization tends to overfit that source view; as a result, **novel-view renders** can show holes, blur, or missing content. The core idea is therefore:

> Create additional (virtual) training views to make supervision more “multi-view-like”, even though the capture is monocular.

## 2) How it works in our pipeline (high level)

We first estimate strong priors (camera + human pose/shape + masks) and initialize a canonical per-person 3DGS. Training is split into two phases:

1. **Pose tuning:** improve the per-frame human parameters (keeping 3DGS fixed).
2. **3DGS tuning:** optimize the 3DGS parameters for appearance (keeping pose/cameras fixed).

Between these two phases, we **generate pseudo ground-truth RGB training images** for a small set of virtual cameras:

- Render the current model from a target virtual camera (“view to refine”).
- Refine that render with **DiFix** (image-to-image diffusion) using a chosen reference image.
- Use the refined result as an extra supervision signal during the 3DGS tuning phase.

## 3) The key choice during view synthesis: the reference image

For each synthesized frame, DiFix takes:

- a **rendered target view** (the image to refine), and
- a **reference image** that provides high-frequency detail.

Choosing the reference decomposes into two practical decisions:

### (a) Which camera should provide the reference?

We have two natural options, each with different trade-offs:

- **Source-camera reference:** always reference the original input camera.
  - **Main benefit:** this is the only view that is truly *observed*. It tends to be the most reliable source of high-frequency details (texture, fine edges) and provides a stable anchor that does not accumulate synthesis errors over time.
  - **Main limitation:** for target cameras that are far from the source, the source view can have **limited overlap** with the view-to-refine. Large viewpoint changes mean DiFix must “invent” more content (previously occluded surfaces, side/back views), which increases hallucination risk and can reduce multi-view consistency.

- **Previous-camera reference (current default):** reference an already-synthesized neighbor in a progressive traversal.
  - **Main benefit:** the heuristic motivation is **higher overlap** (smaller viewpoint change), so DiFix can act more like a *detail transfer / artifact removal* step rather than a generative step. This can also expose surfaces that the source camera never observed.
  - **Main limitation:** the reference is itself **synthetic**. Any mistakes or hallucinations in earlier synthesized views can propagate (error accumulation / drift). In addition, the “previous camera = closest camera” assumption is only an approximation: as we synthesize more cameras, the traversal order does not necessarily guarantee that the chosen “previous” camera is the **best-overlap** reference for the current target.

In the new ablation study, we treat “reference camera selection” as a first-class design choice rather than a fixed heuristic, and we analyze when (and why) source reference, previous reference, or a more overlap-driven strategy is preferable.

### (b) Which timestep should provide the reference?

There are (at least) two reasonable strategies:

- **Same-timestep reference (current default):** when refining the target-camera frame at time `t`, use the reference-camera frame at the *same* time `t`.
  - **Main benefit:** guarantees **pose consistency** across views (the reference and view-to-refine depict the same instant), so DiFix can focus on restoring appearance/detail rather than resolving motion mismatch.
  - **Main limitation:** does *not* guarantee that the chosen reference provides the **best context** for refinement. Even if the camera is “good”, at that specific time `t` the overlap might be small (e.g., self-occlusions, extreme articulation, people turned away), which again increases how much DiFix must hallucinate.

- **Best-overlap timestep search:** allow choosing a different timestep `t'` (and reference camera) such that the selected reference view has the **largest possible overlap** with the view to be refined.
  - Intuition: pick the moment where the reference view contains the most informative pixels for the target viewpoint, minimizing missing content and reducing hallucinations.

## 4) Empirical takeaway so far: DiFix refinement did not help much

**Current reference-view setup (used for the results below):**

- **Reference camera:** *previous synthesized camera* in the progressive traversal (rather than always the source camera).
- **Reference timestep:** *same timestep* (when refining time `t`, we reference time `t`), to enforce pose consistency.

In the thesis NVS ablation (Hi4D protocol), adding DiFix on top of an LHM-initialized reconstruction yields only **marginal** changes:

- With LHM init (no DiFix): SSIM 0.925, PSNR 20.0, LPIPS 0.0818
- With LHM init + DiFix (previous-camera ref): SSIM 0.926, PSNR 20.2, LPIPS 0.0872

So, in our current setting, DiFix behaves more like a *light detail enhancer* than a major driver of reconstruction quality.

## 5) New ablation study hypothesis + experiment outline

We want to test the high-level hypothesis:

> Choosing a reference view with **high overlap** with the view-to-refine reduces how much information DiFix must hallucinate, leading to more faithful refinements and improved downstream novel-view synthesis.

**TODO:** Define what “overlap” means operationally (e.g., a visibility / silhouette / correspondence-based score) and how it is computed in our pipeline.

To make this hypothesis testable, we structure the study as a simple two-factor ablation over the *reference-view selection policy*:

### (a) Ablate the reference **camera** choice (timestep fixed)

Hold the reference timestep fixed to the current default (same timestep `t` for both reference and target), and vary only which camera provides the reference:

- Source-camera reference
- Previous-camera reference (progressive traversal)
- Overlap-driven camera selection (pick the camera that maximizes overlap)

This isolates whether camera selection alone improves refinement quality without introducing pose-mismatch confounds.

### (b) Ablate the reference **timepoint** choice (camera fixed)

Hold the reference camera policy fixed (choose one camera policy and keep it constant), and vary only the timepoint strategy:

- Same-timestep reference: use `t' = t` (pose-consistent baseline)
- Overlap-driven timestep selection: choose `t'` to maximize overlap with the target view at time `t`

This tests whether relaxing the “same timestep” constraint provides better context for refinement.

### (c) Best combined setup

Finally, combine the best-performing camera-selection rule from (a) with the best-performing timepoint-selection rule from (b) to form the overall best **reference-view policy**, and evaluate it against the current baseline (“previous camera, same timestep”).

### Success criterion

We measure success by improved **novel view synthesis (NVS)** metrics on the **Hi4D** dataset under the same evaluation protocol used elsewhere in the thesis (SSIM/PSNR/LPIPS).
