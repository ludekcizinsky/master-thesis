# DiFix tuning study — high-level baseline + motivation for new ablation

This note sets the **red thread** for the DiFix-based *training-view synthesis* component, and motivates a new ablation study where this approach serves as the baseline.

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

We always use the **same timestep** as the frame we are refining. Concretely: when refining the target-camera frame at time `t`, we reference the frame at time `t` from the reference camera. This guarantees that:

> The reference view and the view-to-refine depict the **exact same pose**, so the refinement can focus on appearance/detail rather than motion mismatch.

## 4) Empirical takeaway so far: DiFix refinement did not help much

In the thesis NVS ablation (Hi4D protocol), adding DiFix on top of an LHM-initialized reconstruction yields only **marginal** changes:

- With LHM init (no DiFix): SSIM 0.925, PSNR 20.0, LPIPS 0.0818
- With LHM init + DiFix (previous-camera ref): SSIM 0.926, PSNR 20.2, LPIPS 0.0872

So, in our current setting, DiFix behaves more like a *light detail enhancer* than a major driver of reconstruction quality.

## 5) New ablation study hypothesis

We want to test the hypothesis:

> “We should choose the reference view with the **largest overlap** with the view to be refined. This is the best possible strategy because it minimizes how much information DiFix must hallucinate.”

Concretely, this means rethinking the reference selection rule (camera choice) to be overlap-driven rather than “previous” by traversal order.
