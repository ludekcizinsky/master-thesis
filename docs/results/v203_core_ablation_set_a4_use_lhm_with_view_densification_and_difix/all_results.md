### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9173 | 18.7849 | 0.0942 |
| hi4d_pair16_jump | 0.9176 | 19.2047 | 0.0941 |
| hi4d_pair17_dance | 0.9347 | 22.0286 | 0.0828 |
| hi4d_pair19_piggyback | 0.9326 | 20.7791 | 0.0779 |
| avg | 0.926 | 20.2 | 0.0872 |

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
| hi4d_pair15_fight | 0.5909 | 4.5585 | 2.8529 | 0.7481 |
| hi4d_pair16_jump | 0.6185 | 4.4516 | 2.6512 | 0.7543 |
| hi4d_pair17_dance | 0.5894 | 4.1816 | 2.3010 | 0.7598 |
| hi4d_pair19_piggyback | 0.4887 | 5.3495 | 3.6475 | 0.6732 |
| avg | 0.5719 | 4.6353 | 2.8632 | 0.7339 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.3208 | 6.9758 | 4.0525 | 0.6356 |
| mmm_lift | 0.3431 | 6.7683 | 4.4579 | 0.6101 |
| mmm_walkdance | 0.4700 | 5.2676 | 3.0587 | 0.6761 |
| avg | 0.3780 | 6.3372 | 3.8564 | 0.6406 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9801 | 0.9910 | 0.9899 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9738 | 0.9815 | 0.9867 |