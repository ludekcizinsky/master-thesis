# Limitations Of Visualizing Dynamic 3DGS

## Context

This memo summarizes practical limitations we observed when visualizing **dynamic 3D Gaussian Splatting (3DGS)** in this project, focusing on:

1. `viser`
2. `rerun`

The goal is to remember what each tool is good at, where it breaks down, and why.

---

## What Makes Dynamic 3DGS Hard

Dynamic 3DGS visualization is harder than static 3DGS because we need:

- good per-frame appearance (smooth surfaces, not sparse artifacts),
- temporal navigation (many frames, easy iteration),
- interactive speed (especially remote),
- stable color/opacity behavior,
- minimal friction for server usage.

These goals fight each other: quality settings often hurt speed, and fast settings often look sparse.

---

## 1) `viser`

### Strengths

- Native Gaussian-splat style rendering is visually better for body surfaces than simple point primitives.
- Better perceived continuity of human surfaces compared to point-only views.
- Good for local interactive 3D inspection when scene size is moderate.

### Limitations For Dynamic 3DGS

- Temporal workflows are not first-class in our usage pattern.
- Our dynamic approach requires many per-frame entities and manual frame switching logic.
- Preloading many frames can become heavy in memory and less responsive.
- As sequence length or splat count grows, interaction degrades.
- Not ideal for remote-server-first workflows compared to web-first tools.

### Practical consequence

`viser` gives nicer single-frame appearance, but dynamic sequence UX/scalability is a pain point.

---

## 2) `rerun`

### Strengths

- Excellent temporal logging model (`frame` timeline): clean for dynamic data iteration.
- Very strong remote workflow via web viewer and SSH tunneling.
- Easy to log multi-modal data together: raw 3D primitives, RGB frames, and optional gsplat-rendered reference images.
- Tooling is robust for debugging pipelines over time.

### Limitations For Dynamic 3DGS Appearance

- No native full 3DGS rasterization in the interactive 3D viewport in the way dedicated 3DGS renderers do.
- Point mode is fast, but can look like sparse point cloud instead of smooth human surfaces.
- Ellipsoid mode is closer to splats, but visible ellipsoid boundaries/contours remain, blending is not equivalent to true Gaussian splat rasterization, and high ellipsoid counts become laggy.
- Quality tuning is a tradeoff: more ellipsoids / larger sizes improve smoothness but hurt interactivity, while aggressive filtering/sampling improves speed but hurts fidelity.

### Color/opacity pitfalls we hit

- Low-opacity splats can still create visual clutter unless aggressively filtered.
- Depending on settings, dark/black-looking ellipsoids can appear and dominate visuals.
- Appearance is sensitive to alpha scaling, min opacity thresholds, and color-by-opacity choices.

### Practical consequence

`rerun` is excellent for dynamic-debug workflows and remote operation, but not ideal for photoreal smooth 3DGS surface appearance in the 3D viewport.

---

## Cross-Tool Tradeoff (Short Version)

- `viser`: better local 3DGS appearance, weaker dynamic/temporal workflow ergonomics at scale.
- `rerun`: better dynamic timeline/debug/remote workflow, weaker true 3DGS visual fidelity.

---

## Why This Happens Technically

- Dynamic 3DGS quality relies on proper Gaussian compositing/rasterization.
- Point/ellipsoid approximations are not equivalent to full splat rasterization.
- Temporal interactivity requires efficient data organization and streaming, not just good rendering of one frame.
- Remote use further constrains tool choice toward web/streaming-friendly viewers.

---

## Recommended Use In This Project

- Use `rerun` as the default **dynamic debugging and remote inspection** tool.
- Use `viser` when you need **better qualitative 3D appearance** for a smaller subset (fewer frames / local inspection).
- When final visual quality matters most, prefer a dedicated 3DGS rasterization path (or pre-rendered views) rather than relying on generic 3D primitives.

---

## Open Challenges

- A single tool that is both truly high-fidelity for 3DGS in 3D view and excellent for long dynamic timelines + remote web usage.
- Efficient frame streaming for very large dynamic sequences.
- Better defaults that avoid opacity/color artifacts without heavy manual tuning.
