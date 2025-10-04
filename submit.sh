#!/bin/bash

python playground/unidepth_to_cloud.py \
  --preprocess_dir /scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res \
  --unidepth_dir   /scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res/megasam/unidepth \
  --mask_dir       /scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res/mask \
  --every_k 6 \
  --alpha_min 0.01 --alpha_max 1.6 --alpha_steps 15 --eval_stride 10 \
  --output_npz /scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res/unidepth_cloud_static_scaled.npz \
  --depth_scale 0.2
