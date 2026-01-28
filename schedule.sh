#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# ---------------- Phase 1: finding the strongest baseline model (Section A)
# Baseline A config:
# enable_alternate_opt=false
# loss_weights.interpenetration=0.0
# loss_weights.depth_order=0.0
# pose_tuning.lr=2e-4, lr=2e-4, loss_weights.sil=2.0, loss_weights.depth=0.5

# A0: baseline reference
exp_name="v960_A0_baseline"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    pose_tuning.lr=2e-4 lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5

# A3: loss_weights.rgb sweep (pruned for smaller baseline search)
exp_name="v961_A3_rgb10"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    pose_tuning.lr=2e-4 lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    loss_weights.rgb=10.0

exp_name="v962_A3_rgb20"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    pose_tuning.lr=2e-4 lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    loss_weights.rgb=20.0

exp_name="v963_A3_rgb30"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    pose_tuning.lr=2e-4 lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    loss_weights.rgb=30.0

# A4: loss_weights.ssim sweep (pruned for smaller baseline search)
exp_name="v964_A4_ssim02"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    pose_tuning.lr=2e-4 lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    loss_weights.ssim=0.2

exp_name="v965_A4_ssim04"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    pose_tuning.lr=2e-4 lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    loss_weights.ssim=0.4

exp_name="v966_A4_ssim08"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    pose_tuning.lr=2e-4 lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    loss_weights.ssim=0.8

# A9: regularization.acap_margin sweep (focus on larger margins)
exp_name="v967_A9_acapM03"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    pose_tuning.lr=2e-4 lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    regularization.acap_margin=0.03

exp_name="v968_A9_acapM05"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    pose_tuning.lr=2e-4 lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    regularization.acap_margin=0.05

# A10: lr (3DGS) sweep
exp_name="v969_A10_lr5e5"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    pose_tuning.lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    lr=5e-5

exp_name="v970_A10_lr1e4"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    pose_tuning.lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    lr=1e-4

exp_name="v971_A10_lr2e4"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    pose_tuning.lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    lr=2e-4

# A11: pose_tuning.lr sweep
exp_name="v972_A11_poseLR1e4"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    pose_tuning.lr=1e-4

exp_name="v973_A11_poseLR15e4"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    pose_tuning.lr=1.5e-4

exp_name="v974_A11_poseLR2e4"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=false difix.trn_enable=true \
    loss_weights.interpenetration=0.0 loss_weights.depth_order=0.0 \
    lr=2e-4 loss_weights.sil=2.0 loss_weights.depth=0.5 \
    pose_tuning.lr=2e-4

