### NVS (Baselines + Ours)
Rendering quality is measured via three metrics: PSNR, SSIM, and LPIPS. We highlight the best results for each dataset and metric in **bold**.

#### Hi4D
| Method | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
| --- | --- | --- | --- |
| Shuai et al. | 0.898 | 19.6 | 0.1099 |
| MultiPly | 0.915 | **20.7** | **0.0798** |
| Ours | **0.926** | 20.1 | 0.0858 |

#### MMM
| Method | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
| --- | --- | --- | --- |
| Ours | N/A | N/A | N/A |

### Pose Estimation (Baselines + Ours)
We assess human pose estimation using MPJPE [mm], MVE [mm], Contact Distance (CD) [mm], and Percentage of Correct Depth Relations (PCDR) with a threshold of 0.15m. Baselines are reported in SMPL space. SMPL-X does not include CD, which is shown as N/A. We highlight the best results for each dataset and metric in **bold**.

#### Hi4D
| Method | Space | MPJPE ↓ | MVE ↓ | CD ↓ | PCDR ↑ |
| --- | --- | --- | --- | --- | --- |
| CLIFF | SMPL | 85.7 | 102.1 | 351.7 | 0.606 |
| TRACE | SMPL | 95.6 | 115.7 | 249.4 | 0.603 |
| MultiPly | SMPL | **69.4** | **83.6** | **218.4** | **0.709** |
| Ours (SMPL) | SMPL | N/A | N/A | N/A | N/A |
| Ours (SMPL-X) | SMPL-X | N/A | N/A | N/A | N/A |

#### MMM
| Method | Space | MPJPE ↓ | MVE ↓ | CD ↓ | PCDR ↑ |
| --- | --- | --- | --- | --- | --- |
| Ours (SMPL) | SMPL | N/A | N/A | N/A | N/A |
| Ours (SMPL-X) | SMPL-X | N/A | N/A | N/A | N/A |

### Reconstruction (Baselines + Ours)
We consider the following metrics for human mesh reconstruction evaluation: volumetric IoU (V-IoU), Chamfer distance (C−l2) [cm], point-to-surface distance (P2S) [cm], and normal consistency (NC). We highlight the best results for each dataset and metric in **bold**.

#### Hi4D
| Method | V-IoU ↑ | C-l2 ↓ | P2S ↓ | NC ↑ |
| --- | --- | --- | --- | --- |
| ECON | 0.787 | 3.72 | 3.59 | 0.746 |
| V2A | 0.783 | 3.02 | 2.46 | 0.775 |
| MultiPly | **0.816** | **2.53** | **2.34** | **0.789** |
| Ours | N/A | N/A | N/A | N/A |

#### MMM
| Method | V-IoU ↑ | C-l2 ↓ | P2S ↓ | NC ↑ |
| --- | --- | --- | --- | --- |
| ECON | 0.760 | 4.17 | 3.71 | 0.705 |
| V2A | 0.812 | 3.34 | 2.68 | 0.735 |
| MultiPly | **0.826** | **2.89** | **2.40** | **0.757** |
| Ours | N/A | N/A | N/A | N/A |

### Segmentation (Baselines + Ours)
We report IoU, Recall, and F1 score for human instance segmentation, and highlight the best results for each dataset and metric in **bold**.

#### Hi4D
| Method | IoU ↑ | Recall ↑ | F1 ↑ |
| --- | --- | --- | --- |
| SCHP | 0.937 | 0.983 | 0.982 |
| MultiPly (Init.) | 0.943 | 0.975 | 0.984 |
| MultiPly (Progressive) | **0.963** | **0.985** | **0.990** |
| Ours | N/A | N/A | N/A |

#### MMM
| Method | IoU ↑ | Recall ↑ | F1 ↑ |
| --- | --- | --- | --- |
| Ours | N/A | N/A | N/A |