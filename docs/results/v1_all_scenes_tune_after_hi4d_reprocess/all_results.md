### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9173 | 18.7805 | 0.0942 |
| hi4d_pair16_jump | 0.9176 | 19.1777 | 0.0939 |
| hi4d_pair17_dance | 0.9349 | 22.0389 | 0.0826 |
| hi4d_pair19_piggyback | 0.9325 | 20.7593 | 0.0779 |
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
| hi4d_pair15_fight | 0.5868 | 4.5716 | 2.8716 | 0.7490 |
| hi4d_pair16_jump | 0.6231 | 4.4336 | 2.6355 | 0.7544 |
| hi4d_pair17_dance | 0.5821 | 4.1888 | 2.2945 | 0.7604 |
| hi4d_pair19_piggyback | 0.4952 | 5.3547 | 3.6268 | 0.6746 |
| avg | 0.5718 | 4.6372 | 2.8571 | 0.7346 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.3214 | 6.9818 | 4.0516 | 0.6374 |
| mmm_lift | 0.3394 | 6.7661 | 4.4699 | 0.6096 |
| mmm_walkdance | 0.4694 | 5.2864 | 3.0565 | 0.6754 |
| avg | 0.3767 | 6.3448 | 3.8593 | 0.6408 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9801 | 0.9910 | 0.9899 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9738 | 0.9815 | 0.9867 |