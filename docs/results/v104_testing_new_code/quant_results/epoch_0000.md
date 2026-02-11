# Quantitative Results: `v104_testing_new_code`

- results root: `/scratch/izar/cizinsky/thesis/results`
- epoch selection: `epoch_0000`

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
| hi4d_pair15_fight | 88.4082 | 76.8427 | 0.6333 |
| hi4d_pair16_jump | 87.4978 | 76.5173 | 0.6222 |
| hi4d_pair17_dance | 64.0353 | 52.8890 | 0.4000 |
| hi4d_pair19_piggyback | 277.8164 | 267.9655 | 0.0867 |
| avg | 129.4394 | 118.5536 | 0.4355 |

## Pose Estimation (SMPL)

### hi4d

| scene | rr_mpjpe_mm | rr_mve_mm | pcdr | cd_mm |
| --- | --- | --- | --- | --- |
| hi4d_pair15_fight | 108.5728 | 93.9205 | 0.6267 | 257.2888 |
| hi4d_pair16_jump | 82.1093 | 76.3906 | 0.7444 | 160.5740 |
| hi4d_pair17_dance | 59.1360 | 53.5034 | 0.3920 | 134.4893 |
| hi4d_pair19_piggyback | 326.1468 | 269.3301 | 0.0867 | 353.1869 |
| avg | 143.9912 | 123.2862 | 0.4625 | 226.3847 |
