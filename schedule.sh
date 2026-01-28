#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# ---------------- Phase 1: finding the strongest baseline model (Section A)
# A0b: baseline-combo verification (joint NVS + pose knobs)
exp_name="v975_A0b_baseline_combo"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    pose_tuning.lr=2e-4 lr=5e-5 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    loss_weights.ssim=0.8 regularization.acap_margin=0.03
