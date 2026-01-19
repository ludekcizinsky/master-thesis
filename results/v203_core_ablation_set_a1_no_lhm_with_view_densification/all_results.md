### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9068 | 16.6133 | 0.1266 |
| hi4d_pair16_jump | 0.8937 | 16.6613 | 0.1248 |
| hi4d_pair17_dance | 0.8929 | 17.5884 | 0.1238 |
| hi4d_pair19_piggyback | 0.9076 | 16.9804 | 0.1071 |
| avg | 0.900 | 17.0 | 0.1206 |

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
| hi4d_pair15_fight | 0.5843 | 4.6567 | 3.0157 | 0.7706 |
| hi4d_pair16_jump | 0.5254 | 4.3313 | 2.4423 | 0.7685 |
| hi4d_pair17_dance | 0.5001 | 4.2634 | 2.3847 | 0.7432 |
| hi4d_pair19_piggyback | 0.3948 | 5.2589 | 3.4651 | 0.6429 |
| avg | 0.5011 | 4.6276 | 2.8270 | 0.7313 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.3264 | 7.3776 | 4.3634 | 0.6412 |
| mmm_lift | 0.2879 | 7.1483 | 4.3936 | 0.5929 |
| mmm_walkdance | 0.4545 | 5.5308 | 3.2855 | 0.6660 |
| avg | 0.3563 | 6.6856 | 4.0142 | 0.6334 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9801 | 0.9910 | 0.9899 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9738 | 0.9815 | 0.9867 |