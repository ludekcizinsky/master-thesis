### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9173 | 18.7727 | 0.0945 |
| hi4d_pair16_jump | 0.9173 | 19.1855 | 0.0943 |
| hi4d_pair17_dance | 0.9350 | 22.0617 | 0.0825 |
| hi4d_pair19_piggyback | 0.9326 | 20.7712 | 0.0779 |
| avg | 0.926 | 20.2 | 0.0873 |

### Pose Estimation
#### hi4d
| scene | MPJPE_mm | MVE_mm | CD_mm | PCDR |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 73.3665 | 57.4937 | 237.9104 | 0.6200 |
| hi4d_pair16_jump | 72.0069 | 58.6678 | 148.7172 | 0.4026 |
| hi4d_pair17_dance | 109.3537 | 88.4375 | 235.0948 | 0.6640 |
| hi4d_pair19_piggyback | 120.9763 | 106.2352 | 161.3354 | 0.9000 |
| avg | 93.9258 | 77.7086 | 195.7645 | 0.6466 |

### Reconstruction
#### hi4d
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 0.5827 | 4.5765 | 2.8570 | 0.7497 |
| hi4d_pair16_jump | 0.5747 | 4.4378 | 2.6476 | 0.7553 |
| hi4d_pair17_dance | 0.5836 | 4.1853 | 2.3062 | 0.7599 |
| hi4d_pair19_piggyback | 0.4984 | 5.3596 | 3.6421 | 0.6726 |
| avg | 0.5599 | 4.6398 | 2.8632 | 0.7344 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9805 | 0.9913 | 0.9902 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9739 | 0.9816 | 0.9867 |