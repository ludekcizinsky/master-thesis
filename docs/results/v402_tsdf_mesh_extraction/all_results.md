### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9151 | 18.6134 | 0.0895 |
| hi4d_pair16_jump | 0.9163 | 18.9366 | 0.0882 |
| hi4d_pair17_dance | 0.9353 | 21.9504 | 0.0773 |
| hi4d_pair19_piggyback | 0.9321 | 20.6112 | 0.0722 |
| avg | 0.925 | 20.0 | 0.0818 |

### Pose Estimation
#### hi4d
| scene | MPJPE_mm | MVE_mm | CD_mm | PCDR |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 73.3670 | 57.4941 | 237.9088 | 0.6200 |
| hi4d_pair16_jump | 72.0638 | 58.7529 | 148.7357 | 0.4051 |
| hi4d_pair17_dance | 109.3535 | 88.4373 | 235.0939 | 0.6640 |
| hi4d_pair19_piggyback | 120.9767 | 106.2348 | 161.3349 | 0.9000 |
| avg | 93.9402 | 77.7298 | 195.7683 | 0.6473 |

### Reconstruction
#### hi4d
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 0.2726 | 4.2369 | 2.2259 | 0.7214 |
| hi4d_pair16_jump | 0.2681 | 4.1656 | 2.0380 | 0.7277 |
| hi4d_pair17_dance | 0.2950 | 4.1310 | 2.0938 | 0.7427 |
| hi4d_pair19_piggyback | 0.2483 | 5.2639 | 3.3133 | 0.6676 |
| avg | 0.2710 | 4.4493 | 2.4177 | 0.7148 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.1818 | 6.9986 | 3.9544 | 0.6337 |
| mmm_lift | 0.1748 | 6.7881 | 4.3761 | 0.6173 |
| mmm_walkdance | 0.2573 | 5.1095 | 2.7081 | 0.6642 |
| avg | 0.2046 | 6.2987 | 3.6795 | 0.6384 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9801 | 0.9910 | 0.9899 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9738 | 0.9815 | 0.9867 |