### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9173 | 18.8046 | 0.0934 |
| hi4d_pair16_jump | 0.9179 | 19.2477 | 0.0935 |
| hi4d_pair17_dance | 0.9349 | 22.0239 | 0.0817 |
| hi4d_pair19_piggyback | 0.9329 | 20.8291 | 0.0774 |
| avg | 0.926 | 20.2 | 0.0865 |

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
| hi4d_pair15_fight | 0.5917 | 4.5549 | 2.8613 | 0.7476 |
| hi4d_pair16_jump | 0.6046 | 4.4427 | 2.6228 | 0.7534 |
| hi4d_pair17_dance | 0.5732 | 4.1840 | 2.2940 | 0.7597 |
| hi4d_pair19_piggyback | 0.4923 | 5.3620 | 3.6450 | 0.6722 |
| avg | 0.5655 | 4.6359 | 2.8558 | 0.7332 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.3209 | 6.9715 | 4.0578 | 0.6365 |
| mmm_lift | 0.3404 | 6.7563 | 4.4543 | 0.6080 |
| mmm_walkdance | 0.4738 | 5.2718 | 3.0507 | 0.6786 |
| avg | 0.3784 | 6.3332 | 3.8543 | 0.6410 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9801 | 0.9910 | 0.9899 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9738 | 0.9815 | 0.9867 |