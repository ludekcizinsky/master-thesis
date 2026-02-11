### Reconstruction (Baselines + Ours)
We consider the following metrics for human mesh reconstruction evaluation: volumetric IoU (V-IoU), Chamfer distance (C−l2) [cm], point-to-surface distance (P2S) [cm], and normal consistency (NC). We highlight the best results for each dataset and metric in **bold**.

#### Hi4D
| Method | V-IoU ↑ | C-l2 ↓ | P2S ↓ | NC ↑ |
| --- | --- | --- | --- | --- |
| ECON | 0.787 | 3.72 | 3.59 | 0.746 |
| V2A | 0.783 | 3.02 | 2.46 | 0.775 |
| MultiPly | **0.816** | **2.53** | **2.34** | **0.789** |
| Ours | 0.560 | 4.63 | 2.86 | 0.733 |

#### MMM
| Method | V-IoU ↑ | C-l2 ↓ | P2S ↓ | NC ↑ |
| --- | --- | --- | --- | --- |
| ECON | 0.760 | 4.17 | 3.71 | 0.705 |
| V2A | 0.812 | 3.34 | 2.68 | 0.735 |
| MultiPly | **0.826** | **2.89** | **2.40** | **0.757** |
| Ours | 0.377 | 6.33 | 3.86 | 0.641 |