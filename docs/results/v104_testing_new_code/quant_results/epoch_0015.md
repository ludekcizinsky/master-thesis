# Quantitative Results: `v104_testing_new_code`

- results root: `/scratch/izar/cizinsky/thesis/results`
- epoch selection: `epoch_0015`

## Novel View Synthesis

### hi4d

| scene | psnr | ssim | lpips |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 18.6353 | 0.9151 | 0.0897 |
| hi4d_pair16_jump | 18.9513 | 0.9157 | 0.0889 |
| hi4d_pair17_dance | 21.9447 | 0.9352 | 0.0778 |
| hi4d_pair19_piggyback | 20.6285 | 0.9321 | 0.0725 |
| avg | 20.0400 | 0.9245 | 0.0822 |

## Pose Estimation (SMPL-X)

### hi4d

| scene | rr_mpjpe_mm | rr_mve_mm | pcdr |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 74.8688 | 64.4167 | 0.6333 |
| hi4d_pair16_jump | 84.5071 | 73.5036 | 0.6222 |
| hi4d_pair17_dance | 57.9383 | 47.7863 | 0.3760 |
| hi4d_pair19_piggyback | 253.1861 | 242.8181 | 0.0867 |
| avg | 117.6251 | 107.1312 | 0.4295 |

## Pose Estimation (SMPL)

### hi4d

| scene | rr_mpjpe_mm | rr_mve_mm | pcdr | cd_mm |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 101.4375 | 87.6374 | 0.6267 | 235.6607 |
| hi4d_pair16_jump | 81.6929 | 76.0502 | 0.7444 | 133.3444 |
| hi4d_pair17_dance | 59.6333 | 53.1985 | 0.3680 | 138.9188 |
| hi4d_pair19_piggyback | 309.9566 | 253.9106 | 0.0867 | 328.1730 |
| avg | 138.1801 | 117.6992 | 0.4565 | 209.0242 |
