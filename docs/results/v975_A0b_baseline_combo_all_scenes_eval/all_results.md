### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9171 | 18.6234 | 0.0869 |
| hi4d_pair16_jump | 0.9176 | 19.0047 | 0.0873 |
| hi4d_pair17_dance | 0.9356 | 21.9727 | 0.0771 |
| hi4d_pair19_piggyback | 0.9327 | 20.6637 | 0.0722 |
| avg | 0.926 | 20.1 | 0.0809 |

### Pose Estimation (SMPL)
#### hi4d
| scene | MPJPE_mm | MVE_mm | CD_mm | PCDR |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 101.2941 | 79.3433 | 241.4512 | 0.5733 |
| hi4d_pair16_jump | 72.9870 | 60.7708 | 133.5337 | 0.4051 |
| hi4d_pair17_dance | 99.5167 | 79.8630 | 233.5361 | 0.6640 |
| hi4d_pair19_piggyback | 135.4769 | 115.1543 | 148.4254 | 0.8933 |
| avg | 102.3187 | 83.7828 | 189.2366 | 0.6339 |

### Pose Estimation (SMPL-X)
#### hi4d
| scene | MPJPE_mm | MVE_mm | PCDR |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 58.9683 | 50.8201 | 0.6267 |
| hi4d_pair16_jump | 53.5976 | 45.0732 | 0.3291 |
| hi4d_pair17_dance | 91.2709 | 68.0728 | 0.5840 |
| hi4d_pair19_piggyback | 105.8598 | 91.1192 | 0.9133 |
| avg | 77.4241 | 63.7713 | 0.6133 |

### Reconstruction
#### hi4d
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 0.6194 | 4.1670 | 2.0999 | 0.8044 |
| hi4d_pair16_jump | 0.6142 | 4.1477 | 2.0211 | 0.8020 |
| hi4d_pair17_dance | 0.6047 | 3.9846 | 1.9152 | 0.8041 |
| hi4d_pair19_piggyback | 0.5371 | 5.0111 | 2.9452 | 0.7217 |
| avg | 0.5938 | 4.3276 | 2.2454 | 0.7831 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.3348 | 7.2957 | 4.2430 | 0.6583 |
| mmm_lift | 0.3038 | 6.8332 | 4.3306 | 0.6363 |
| mmm_walkdance | 0.3066 | 5.1313 | 2.7240 | 0.7204 |
| avg | 0.3151 | 6.4201 | 3.7659 | 0.6717 |