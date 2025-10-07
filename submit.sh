#!/bin/bash

python playground/unidepth_to_cloud.py \
  --preprocess_dir /scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res \
  --unidepth_dir   /scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res/megasam/unidepth \
  --mask_dir       /scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res/mask \
  --output_npz /scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res/unidepth_cloud_static_scaled.npz \
  --max_frames 30 \
  --depth_scale 0.2
