### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9077 | 16.7038 | 0.1236 |
| hi4d_pair16_jump | 0.8955 | 16.8192 | 0.1217 |
| hi4d_pair17_dance | 0.8983 | 18.1348 | 0.1217 |
| hi4d_pair19_piggyback | 0.9106 | 16.9419 | 0.1066 |
| avg | 0.903 | 17.1 | 0.1184 |

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
| hi4d_pair15_fight | 0.5826 | 4.6632 | 3.0066 | 0.7715 |
| hi4d_pair16_jump | 0.5183 | 4.2955 | 2.3833 | 0.7626 |
| hi4d_pair17_dance | 0.5108 | 4.2444 | 2.2888 | 0.7392 |
| hi4d_pair19_piggyback | 0.3975 | 5.2450 | 3.4008 | 0.6501 |
| avg | 0.5023 | 4.6120 | 2.7699 | 0.7308 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.3248 | 7.3753 | 4.3394 | 0.6395 |
| mmm_lift | 0.2925 | 7.0793 | 4.3366 | 0.5961 |
| mmm_walkdance | 0.4554 | 5.5023 | 3.2373 | 0.6687 |
| avg | 0.3576 | 6.6523 | 3.9711 | 0.6348 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9801 | 0.9910 | 0.9899 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9738 | 0.9815 | 0.9867 |