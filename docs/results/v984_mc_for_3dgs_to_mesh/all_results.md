### NVS
#### hi4d
| scene | SSIM | PSNR | LPIPS |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 0.9169 | 18.6084 | 0.0872 |
| hi4d_pair16_jump | 0.9175 | 18.9996 | 0.0874 |
| hi4d_pair17_dance | 0.9355 | 21.9538 | 0.0770 |
| hi4d_pair19_piggyback | 0.9328 | 20.6806 | 0.0722 |
| avg | 0.926 | 20.1 | 0.0809 |

### Pose Estimation (SMPL)
#### hi4d
| scene | MPJPE_mm | MVE_mm | CD_mm | PCDR |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 101.2763 | 79.3289 | 241.4030 | 0.5733 |
| hi4d_pair16_jump | 73.0016 | 60.7808 | 133.1945 | 0.4051 |
| hi4d_pair17_dance | 99.5332 | 79.8816 | 233.5521 | 0.6720 |
| hi4d_pair19_piggyback | 135.4943 | 115.1809 | 148.5636 | 0.8933 |
| avg | 102.3264 | 83.7931 | 189.1783 | 0.6359 |

### Pose Estimation (SMPL-X)
#### hi4d
| scene | MPJPE_mm | MVE_mm | PCDR |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 58.9527 | 50.8182 | 0.6200 |
| hi4d_pair16_jump | 53.5676 | 45.0572 | 0.3291 |
| hi4d_pair17_dance | 91.2890 | 68.0941 | 0.5760 |
| hi4d_pair19_piggyback | 105.9250 | 91.1793 | 0.9133 |
| avg | 77.4336 | 63.7872 | 0.6096 |

### Reconstruction
#### hi4d
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 0.6541 | 4.6081 | 2.9192 | 0.7115 |
| hi4d_pair16_jump | 0.6332 | 4.5604 | 2.7821 | 0.7060 |
| hi4d_pair17_dance | 0.6601 | 4.3940 | 2.6312 | 0.6950 |
| hi4d_pair19_piggyback | 0.5843 | 5.2671 | 3.4531 | 0.6421 |
| avg | 0.6329 | 4.7074 | 2.9464 | 0.6886 |

#### mmm
| scene | V_IoU | Chamfer_cm | P2S_cm | Normal_Consistency |
| --- | --- | --- | --- | --- |
| mmm_dance | 0.3554 | 7.3334 | 4.2948 | 0.6082 |
| mmm_lift | 0.3787 | 6.8497 | 4.4342 | 0.5873 |
| mmm_walkdance | 0.5462 | 5.4487 | 3.2430 | 0.6535 |
| avg | 0.4268 | 6.5439 | 3.9907 | 0.6163 |