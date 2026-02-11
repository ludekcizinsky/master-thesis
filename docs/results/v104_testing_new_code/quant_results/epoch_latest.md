# Quantitative Results: `v104_testing_new_code`

- results root: `/scratch/izar/cizinsky/thesis/results`
- epoch selection: `latest`

## Novel View Synthesis

### hi4d

| scene | psnr | ssim | lpips |
| --- | --- | --- | --- |
| hi4d_pair15_fight | 18.7335 | 0.9166 | 0.0882 |
| hi4d_pair16_jump | 19.1337 | 0.9166 | 0.0892 |
| hi4d_pair17_dance | 21.9247 | 0.9347 | 0.0793 |
| hi4d_pair19_piggyback | 20.6450 | 0.9317 | 0.0735 |
| avg | 20.1092 | 0.9249 | 0.0825 |

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
| hi4d_pair15_fight | 101.4376 | 87.6376 | 0.6267 | 235.6613 |
| hi4d_pair16_jump | 81.6989 | 76.0563 | 0.7444 | 133.3312 |
| hi4d_pair17_dance | 59.6441 | 53.2050 | 0.3680 | 138.9163 |
| hi4d_pair19_piggyback | 311.6174 | 255.3348 | 0.0867 | 328.0689 |
| avg | 138.5995 | 118.0584 | 0.4565 | 208.9944 |
