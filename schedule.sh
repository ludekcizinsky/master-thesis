#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# ---------------- Phase 2: add components on top of the v975 baseline
# Baseline v975 knobs used in all runs below (from train.yaml):
# lr=5e-5, loss_weights.ssim=0.8, regularization.acap_margin=0.03,
# loss_weights.sil=2.0, loss_weights.depth=0.5, loss_weights.rgb=20.0

# # B) Interpenetration loss
# exp_name="v976_B_inter01"
# python /home/cizinsky/master-thesis/schedule.py \
    # --exp_name $exp_name \
    # --job_name_prefix $exp_name \
    # --scene_name_includes "hi4d" \
    # --overrides \
    # loss_weights.interpenetration=0.1

# exp_name="v977_B_inter02"
# python /home/cizinsky/master-thesis/schedule.py \
    # --exp_name $exp_name \
    # --job_name_prefix $exp_name \
    # --scene_name_includes "hi4d" \
    # --overrides \
    # loss_weights.interpenetration=0.2

# exp_name="v978_B_inter04"
# python /home/cizinsky/master-thesis/schedule.py \
    # --exp_name $exp_name \
    # --job_name_prefix $exp_name \
    # --scene_name_includes "hi4d" \
    # --overrides \
    # loss_weights.interpenetration=0.4

# # C) Depth-order loss
# exp_name="v979_C_dorder01"
# python /home/cizinsky/master-thesis/schedule.py \
    # --exp_name $exp_name \
    # --job_name_prefix $exp_name \
    # --scene_name_includes "hi4d" \
    # --overrides \
    # loss_weights.depth_order=0.1

# exp_name="v980_C_dorder02"
# python /home/cizinsky/master-thesis/schedule.py \
    # --exp_name $exp_name \
    # --job_name_prefix $exp_name \
    # --scene_name_includes "hi4d" \
    # --overrides \
    # loss_weights.depth_order=0.2

# exp_name="v981_C_dorder04"
# python /home/cizinsky/master-thesis/schedule.py \
    # --exp_name $exp_name \
    # --job_name_prefix $exp_name \
    # --scene_name_includes "hi4d" \
    # --overrides \
    # loss_weights.depth_order=0.4

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

#exp_name="v984_D_conf_median"
#python /home/cizinsky/master-thesis/schedule.py \
    #--exp_name $exp_name \
    #--job_name_prefix $exp_name \
    #--scene_name_includes "hi4d" \
    #--overrides enable_alternate_opt=true \
    #conf_tr_aggregation=median confidence_update_every=5

# # E) SMPL-X params ablation
# exp_name="v985_E_params_nobetas"
# python /home/cizinsky/master-thesis/schedule.py \
    # --exp_name $exp_name \
    # --job_name_prefix $exp_name \
    # --scene_name_includes "hi4d" \
    # --overrides \
    # pose_tuning.params='[root_pose,body_pose,trans]'

# exp_name="v986_E_params_addhands"
# python /home/cizinsky/master-thesis/schedule.py \
    # --exp_name $exp_name \
    # --job_name_prefix $exp_name \
    # --scene_name_includes "hi4d" \
    # --overrides \
    # pose_tuning.params='[root_pose,body_pose,lhand_pose,rhand_pose,trans,betas]'

# exp_name="v987_E_params_fullpose_nobetas"
# python /home/cizinsky/master-thesis/schedule.py \
    # --exp_name $exp_name \
    # --job_name_prefix $exp_name \
    # --scene_name_includes "hi4d" \
    # --overrides \
    # pose_tuning.params='[root_pose,body_pose,jaw_pose,leye_pose,reye_pose,lhand_pose,rhand_pose,trans]'
