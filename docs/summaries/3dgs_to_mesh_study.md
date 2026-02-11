## 3DGS → Mesh Conversion Study

### Setup and motivation
This study builds directly on the earlier ablations that established:
1) the **best novel-view training pipeline** (DiFix-refined views, previous-camera source), and
2) the **best pose-tuning baseline** (v975 baseline combo).

With those choices fixed, we now isolate the **3DGS → mesh conversion** step and compare two extraction methods:
- **TSDF fusion** (baseline): multi-view depth fusion with camera geometry.
- **Marching Cubes (MC)**: density-grid extraction from Gaussians.

This study focuses on **reconstruction metrics**, since the conversion method directly affects mesh quality.
Context: **Hi4D has 2 people**, while **MMM typically has 3–4 people**, making MMM more challenging for meshing.

**Experiments**
- TSDF: `v975_A0b_baseline_combo_all_scenes_eval`
- MC: `v984_mc_for_3dgs_to_mesh`

---

## Reconstruction comparison (avg across scenes)

### Hi4D (2 people)
| Method | V_IoU (↑) | Chamfer_cm (↓) | P2S_cm (↓) | Normal_Consistency (↑) |
|:---|---:|---:|---:|---:|
| TSDF | 0.5938 | **4.3276** | **2.2454** | **0.7831** |
| MC | **0.6329** | 4.7074 | 2.9464 | 0.6886 |

### MMM (3–4 people)
| Method | V_IoU (↑) | Chamfer_cm (↓) | P2S_cm (↓) | Normal_Consistency (↑) |
|:---|---:|---:|---:|---:|
| TSDF | 0.3151 | **6.4201** | **3.7659** | **0.6717** |
| MC | **0.4268** | 6.5439 | 3.9907 | 0.6163 |

---

## Interpretation
**Hi4D:**
- **MC improves V-IoU** (0.6329 vs 0.5938), suggesting better volumetric coverage.
- **TSDF is clearly better on surface accuracy and normals** (lower Chamfer/P2S, higher Normal Consistency).

**MMM:**
- MC again shows higher V-IoU (0.4268 vs 0.3151), but **TSDF remains better on Chamfer/P2S and normals**.
- With more people, the mesh fidelity advantage of TSDF becomes even more important.

**Conclusion:**
- If the priority is **surface accuracy and normal consistency**, **TSDF** should remain the default.
- If the priority is **volumetric overlap (V-IoU)**, **MC** can be competitive but at the cost of surface quality.

Given our focus on accurate reconstructions (Chamfer/P2S/Normals), the evidence supports **TSDF as the preferred 3DGS→mesh method**.
