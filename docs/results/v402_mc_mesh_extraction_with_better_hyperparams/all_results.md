### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9172 | 18.7572 | 0.0941 |
| hi4d_pair16_jump | 0.9179 | 19.2441 | 0.0935 |
| hi4d_pair17_dance | 0.9349 | 22.0382 | 0.0829 |
| hi4d_pair19_piggyback | 0.9327 | 20.7765 | 0.0776 |
| avg | 0.926 | 20.2 | 0.0870 |

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
| hi4d_pair15_fight | 0.5746 | 4.4512 | 2.6154 | 0.7193 |
| hi4d_pair16_jump | 0.5548 | 4.3101 | 2.3753 | 0.7229 |
| hi4d_pair17_dance | 0.5082 | 4.1633 | 2.2144 | 0.7158 |
| hi4d_pair19_piggyback | 0.4395 | 5.3079 | 3.4763 | 0.6487 |
| avg | 0.5193 | 4.5581 | 2.6704 | 0.7017 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.2901 | 7.0155 | 3.9308 | 0.6161 |
| mmm_lift | 0.3010 | 6.8013 | 4.2978 | 0.5959 |
| mmm_walkdance | 0.4289 | 5.2524 | 2.9151 | 0.6589 |
| avg | 0.3400 | 6.3564 | 3.7146 | 0.6236 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9801 | 0.9910 | 0.9899 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9738 | 0.9815 | 0.9867 |