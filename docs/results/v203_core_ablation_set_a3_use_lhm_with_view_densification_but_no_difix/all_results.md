### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9174 | 18.9931 | 0.0973 |
| hi4d_pair16_jump | 0.9183 | 19.3551 | 0.0945 |
| hi4d_pair17_dance | 0.9355 | 22.0842 | 0.0839 |
| hi4d_pair19_piggyback | 0.9329 | 20.8221 | 0.0784 |
| avg | 0.926 | 20.3 | 0.0885 |

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
| hi4d_pair15_fight | 0.5937 | 4.5570 | 2.8753 | 0.7476 |
| hi4d_pair16_jump | 0.5786 | 4.4455 | 2.6196 | 0.7494 |
| hi4d_pair17_dance | 0.5493 | 4.2035 | 2.3214 | 0.7587 |
| hi4d_pair19_piggyback | 0.4560 | 5.3477 | 3.6334 | 0.6709 |
| avg | 0.5444 | 4.6384 | 2.8624 | 0.7317 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.3244 | 6.9999 | 4.1207 | 0.6310 |
| mmm_lift | 0.3222 | 6.7617 | 4.4417 | 0.5993 |
| mmm_walkdance | 0.4514 | 5.2995 | 3.0628 | 0.6580 |
| avg | 0.3660 | 6.3537 | 3.8751 | 0.6294 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9801 | 0.9910 | 0.9899 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9738 | 0.9815 | 0.9867 |