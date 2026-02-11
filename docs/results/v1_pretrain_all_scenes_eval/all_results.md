### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9151 | 18.6129 | 0.0895 |
| hi4d_pair16_jump | 0.9163 | 18.9397 | 0.0882 |
| hi4d_pair17_dance | 0.9353 | 21.9484 | 0.0773 |
| hi4d_pair19_piggyback | 0.9321 | 20.6109 | 0.0722 |
| avg | 0.925 | 20.0 | 0.0818 |

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
| hi4d_pair15_fight | 0.6285 | 4.7791 | 3.2635 | 0.7635 |
| hi4d_pair16_jump | 0.6255 | 4.7324 | 3.1255 | 0.7672 |
| hi4d_pair17_dance | 0.6696 | 4.4687 | 2.7788 | 0.7865 |
| hi4d_pair19_piggyback | 0.5411 | 5.5283 | 4.0229 | 0.6887 |
| avg | 0.6162 | 4.8771 | 3.2977 | 0.7515 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.4024 | 6.9691 | 4.2683 | 0.6580 |
| mmm_lift | 0.3952 | 6.8175 | 4.7881 | 0.6347 |
| mmm_walkdance | 0.5762 | 5.3151 | 3.2657 | 0.7132 |
| avg | 0.4579 | 6.3672 | 4.1074 | 0.6686 |

### Segmentation (Src Cam)
#### hi4d
| scene | IoU | Recall | F1 |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9832 | 0.9923 | 0.9915 |
| hi4d_pair16_jump | 0.9805 | 0.9913 | 0.9902 |
| hi4d_pair17_dance | 0.9766 | 0.9818 | 0.9882 |
| hi4d_pair19_piggyback | 0.9554 | 0.9608 | 0.9771 |
| avg | 0.9739 | 0.9816 | 0.9867 |

