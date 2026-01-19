### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9006 | 16.5997 | 0.1360 |
| hi4d_pair16_jump | 0.8869 | 16.8970 | 0.1390 |
| hi4d_pair17_dance | 0.8808 | 17.5626 | 0.1396 |
| hi4d_pair19_piggyback | 0.8915 | 16.7867 | 0.1224 |
| avg | 0.890 | 17.0 | 0.1343 |

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
| hi4d_pair15_fight | 0.5680 | 4.7825 | 3.1593 | 0.7553 |
| hi4d_pair16_jump | 0.5219 | 4.4867 | 2.6768 | 0.7428 |
| hi4d_pair17_dance | 0.4399 | 4.5797 | 2.8913 | 0.7197 |
| hi4d_pair19_piggyback | 0.3848 | 5.3825 | 3.8321 | 0.6420 |
| avg | 0.4787 | 4.8079 | 3.1399 | 0.7149 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.3460 | 7.4089 | 4.5963 | 0.6343 |
| mmm_lift | 0.2898 | 7.0720 | 4.5515 | 0.5974 |
| mmm_walkdance | 0.4510 | 5.7127 | 3.5239 | 0.6576 |
| avg | 0.3623 | 6.7312 | 4.2239 | 0.6298 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9801 | 0.9910 | 0.9899 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9738 | 0.9815 | 0.9867 |