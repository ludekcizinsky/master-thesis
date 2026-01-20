### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9181 | 19.3671 | 0.1000 |
| hi4d_pair16_jump | 0.9177 | 19.4497 | 0.0961 |
| hi4d_pair17_dance | 0.9246 | 20.9923 | 0.0981 |
| hi4d_pair19_piggyback | 0.9288 | 20.5162 | 0.0870 |
| avg | 0.922 | 20.1 | 0.0953 |

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
| hi4d_pair15_fight | 0.5930 | 4.6124 | 2.9500 | 0.7415 |
| hi4d_pair16_jump | 0.6274 | 4.6883 | 3.0518 | 0.7631 |
| hi4d_pair17_dance | 0.6627 | 4.3370 | 2.6068 | 0.7620 |
| hi4d_pair19_piggyback | 0.5042 | 5.3477 | 3.6628 | 0.6682 |
| avg | 0.5968 | 4.7463 | 3.0678 | 0.7337 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.3340 | 7.0930 | 4.2153 | 0.6411 |
| mmm_lift | 0.3499 | 6.8002 | 4.4823 | 0.6127 |
| mmm_walkdance | 0.5097 | 5.3983 | 3.2612 | 0.6860 |
| avg | 0.3979 | 6.4305 | 3.9863 | 0.6466 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9801 | 0.9910 | 0.9899 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9738 | 0.9815 | 0.9867 |