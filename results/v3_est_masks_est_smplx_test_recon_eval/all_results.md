### NVS
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9173 | 18.7641 | 0.0942 |
| hi4d_pair16_jump | 0.9176 | 19.1906 | 0.0939 |
| hi4d_pair17_dance | 0.9349 | 22.0470 | 0.0827 |
| hi4d_pair19_piggyback | 0.9326 | 20.7757 | 0.0779 |
| avg | 0.926 | 20.2 | 0.0872 |

### Pose Estimation
| scene | MPJPE_mm | MVE_mm | CD_mm | PCDR |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 73.3655 | 57.4946 | 237.9949 | 0.6200 |
| hi4d_pair16_jump | 72.0302 | 58.7038 | 148.1010 | 0.4026 |
| hi4d_pair17_dance | 109.3601 | 88.4557 | 235.1298 | 0.6640 |
| hi4d_pair19_piggyback | 120.9660 | 106.2170 | 161.3194 | 0.9000 |
| avg | 93.9304 | 77.7178 | 195.6363 | 0.6466 |

### Reconstruction
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 0.6058 | 3.6371 | 2.6111 | 0.7569 |
| hi4d_pair16_jump | 0.6181 | 3.5204 | 2.4436 | 0.7668 |
| hi4d_pair17_dance | 0.5763 | 3.5875 | 2.2634 | 0.7728 |
| hi4d_pair19_piggyback | 0.4782 | 4.6859 | 3.6543 | 0.7054 |
| avg | 0.5696 | 3.8577 | 2.7431 | 0.7505 |

### Segmentation (Src Cam)
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9805 | 0.9913 | 0.9902 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9739 | 0.9816 | 0.9867 |