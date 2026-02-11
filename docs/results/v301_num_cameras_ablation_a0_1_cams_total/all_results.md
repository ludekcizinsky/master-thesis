### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9075 | 18.2878 | 0.1134 |
| hi4d_pair16_jump | 0.9103 | 19.2277 | 0.1112 |
| hi4d_pair17_dance | 0.9058 | 19.3354 | 0.1182 |
| hi4d_pair19_piggyback | 0.9088 | 18.3282 | 0.1070 |
| avg | 0.908 | 18.8 | 0.1124 |

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
| hi4d_pair15_fight | 0.5456 | 4.4986 | 2.7403 | 0.7165 |
| hi4d_pair16_jump | 0.6269 | 4.4680 | 2.7379 | 0.7470 |
| hi4d_pair17_dance | 0.5955 | 4.3764 | 2.6486 | 0.7662 |
| hi4d_pair19_piggyback | 0.4084 | 5.2856 | 3.6862 | 0.6595 |
| avg | 0.5441 | 4.6571 | 2.9532 | 0.7223 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.3211 | 7.3332 | 4.3656 | 0.6161 |
| mmm_lift | 0.3218 | 6.9781 | 4.5773 | 0.5868 |
| mmm_walkdance | 0.4717 | 5.4789 | 3.3242 | 0.6499 |
| avg | 0.3715 | 6.5967 | 4.0890 | 0.6176 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9801 | 0.9910 | 0.9899 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9738 | 0.9815 | 0.9867 |