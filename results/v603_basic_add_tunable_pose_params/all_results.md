### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9181 | 18.9581 | 0.0953 |
| hi4d_pair16_jump | 0.9190 | 19.3295 | 0.0940 |
| hi4d_pair17_dance | 0.9349 | 22.0424 | 0.0836 |
| hi4d_pair19_piggyback | 0.9340 | 20.9798 | 0.0783 |
| avg | 0.926 | 20.3 | 0.0878 |

### Pose Estimation (SMPL)
#### hi4d
| scene | MPJPE_mm | MVE_mm | CD_mm | PCDR |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 73.2349 | 57.3508 | 237.8821 | 0.6200 |
| hi4d_pair16_jump | 71.9927 | 58.7094 | 148.5409 | 0.4051 |
| hi4d_pair17_dance | 109.3463 | 88.4199 | 235.4069 | 0.6560 |
| hi4d_pair19_piggyback | 120.5726 | 105.7325 | 160.5432 | 0.9000 |
| avg | 93.7866 | 77.5532 | 195.5933 | 0.6453 |

### Pose Estimation (SMPL-X)
#### hi4d
| scene | MPJPE_mm | MVE_mm | PCDR |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 60.0877 | 50.8872 | 0.6667 |
| hi4d_pair16_jump | 53.1952 | 44.2058 | 0.3291 |
| hi4d_pair17_dance | 92.1077 | 67.5434 | 0.5520 |
| hi4d_pair19_piggyback | 116.3733 | 100.0207 | 0.9200 |
| avg | 80.4410 | 65.6643 | 0.6169 |

### Reconstruction
#### hi4d
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 0.5841 | 4.4502 | 2.5896 | 0.7189 |
| hi4d_pair16_jump | 0.5652 | 4.3071 | 2.3919 | 0.7249 |
| hi4d_pair17_dance | 0.5083 | 4.1644 | 2.2217 | 0.7175 |
| hi4d_pair19_piggyback | 0.4629 | 5.3423 | 3.4641 | 0.6487 |
| avg | 0.5301 | 4.5660 | 2.6668 | 0.7025 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.2938 | 6.9809 | 3.9316 | 0.6155 |
| mmm_lift | 0.3015 | 6.8060 | 4.3166 | 0.5939 |
| mmm_walkdance | 0.4297 | 5.2530 | 2.9173 | 0.6581 |
| avg | 0.3417 | 6.3466 | 3.7218 | 0.6225 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9801 | 0.9910 | 0.9899 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9738 | 0.9815 | 0.9867 |