### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9182 | 19.1265 | 0.0969 |
| hi4d_pair16_jump | 0.9188 | 19.4334 | 0.0955 |
| hi4d_pair17_dance | 0.9325 | 21.8639 | 0.0873 |
| hi4d_pair19_piggyback | 0.9320 | 20.8155 | 0.0816 |
| avg | 0.925 | 20.3 | 0.0903 |

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
| hi4d_pair15_fight | 0.6095 | 4.5409 | 2.8396 | 0.7434 |
| hi4d_pair16_jump | 0.6134 | 4.5220 | 2.7836 | 0.7493 |
| hi4d_pair17_dance | 0.5698 | 4.2022 | 2.3342 | 0.7575 |
| hi4d_pair19_piggyback | 0.4686 | 5.3440 | 3.6156 | 0.6625 |
| avg | 0.5653 | 4.6523 | 2.8933 | 0.7282 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.3220 | 7.0253 | 4.1076 | 0.6337 |
| mmm_lift | 0.3301 | 6.7829 | 4.4074 | 0.6018 |
| mmm_walkdance | 0.4699 | 5.3026 | 3.0627 | 0.6734 |
| avg | 0.3740 | 6.3703 | 3.8592 | 0.6363 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9801 | 0.9910 | 0.9899 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9738 | 0.9815 | 0.9867 |