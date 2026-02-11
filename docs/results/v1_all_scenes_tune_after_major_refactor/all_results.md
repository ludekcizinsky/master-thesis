### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9175 | 18.7975 | 0.0943 |
| hi4d_pair16_jump | 0.9178 | 19.2517 | 0.0936 |
| hi4d_pair17_dance | 0.9347 | 22.0226 | 0.0831 |
| hi4d_pair19_piggyback | 0.9327 | 20.7888 | 0.0778 |
| avg | 0.926 | 20.2 | 0.0872 |

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
| hi4d_pair15_fight | 0.5906 | 4.5609 | 2.8568 | 0.7489 |
| hi4d_pair16_jump | 0.5766 | 4.4372 | 2.6329 | 0.7515 |
| hi4d_pair17_dance | 0.5856 | 4.1913 | 2.3013 | 0.7586 |
| hi4d_pair19_piggyback | 0.4864 | 5.3400 | 3.6383 | 0.6724 |
| avg | 0.5598 | 4.6324 | 2.8573 | 0.7329 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.3216 | 6.9666 | 4.0450 | 0.6368 |
| mmm_lift | 0.3402 | 6.7532 | 4.4647 | 0.6107 |
| mmm_walkdance | 0.4705 | 5.2670 | 3.0591 | 0.6769 |
| avg | 0.3774 | 6.3289 | 3.8563 | 0.6415 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9805 | 0.9913 | 0.9902 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9739 | 0.9816 | 0.9867 |