#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# =======================================================================================
# Phase A: Finding the strongest naive baseline model (no interpenetration, no depth-order, no confidence guidance)
# Baseline A config:
# enable_alternate_opt=false
# loss_weights.interpenetration=0.0
# loss_weights.depth_order=0.0
# pose_tuning.lr=2e-4, lr=2e-4, loss_weights.sil=2.0, loss_weights.depth=0.5
# =======================================================================================

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


# ---------------- Phase 2: add components on top of the v975 baseline
# Baseline v975 knobs used in all runs below (from train.yaml):
# lr=5e-5, loss_weights.ssim=0.8, regularization.acap_margin=0.03,
# loss_weights.sil=2.0, loss_weights.depth=0.5, loss_weights.rgb=20.0

# B) Interpenetration loss
exp_name="v976_B_inter01"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides \
    loss_weights.interpenetration=0.1

exp_name="v977_B_inter02"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides \
    loss_weights.interpenetration=0.2

exp_name="v978_B_inter04"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides \
    loss_weights.interpenetration=0.4

# C) Depth-order loss
exp_name="v979_C_dorder01"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides \
    loss_weights.depth_order=0.1

exp_name="v980_C_dorder02"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides \
    loss_weights.depth_order=0.2

exp_name="v981_C_dorder04"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides \
    loss_weights.depth_order=0.4

# D) Confidence-guided optimization (alternate opt)
exp_name="v982_D_conf075_fix"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true \
    confidence_threshold=0.75 confidence_update_every=5

exp_name="v983_D_conf085_fix"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true \
    confidence_threshold=0.85 confidence_update_every=5

exp_name="v984_D_conf_median"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true \
    conf_tr_aggregation=median confidence_update_every=5

 # E) SMPL-X params ablation
 exp_name="v985_E_params_nobetas"
 python /home/cizinsky/master-thesis/schedule.py \
     --exp_name $exp_name \
     --job_name_prefix $exp_name \
     --scene_name_includes "hi4d" \
     --overrides \
     pose_tuning.params='[root_pose,body_pose,trans]'

 exp_name="v986_E_params_addhands"
 python /home/cizinsky/master-thesis/schedule.py \
     --exp_name $exp_name \
     --job_name_prefix $exp_name \
     --scene_name_includes "hi4d" \
     --overrides \
     pose_tuning.params='[root_pose,body_pose,lhand_pose,rhand_pose,trans,betas]'

 exp_name="v987_E_params_fullpose_nobetas"
 python /home/cizinsky/master-thesis/schedule.py \
     --exp_name $exp_name \
     --job_name_prefix $exp_name \
     --scene_name_includes "hi4d" \
     --overrides \
     pose_tuning.params='[root_pose,body_pose,jaw_pose,leye_pose,reye_pose,lhand_pose,rhand_pose,trans]'
