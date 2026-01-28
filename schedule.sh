#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# Example call:
# exp_name="v802_testing_overrides_via_schedule_alternate_opt"
# python /home/cizinsky/master-thesis/schedule.py \
    # --exp_name $exp_name \
    # --job_name_prefix $exp_name \
    # --scene_name_includes "hi4d" \
    # --overrides enable_alternate_opt=true confidence_threshold=0.7 confidence_update_every=3

# ---------------- Ablations (comments only) ----------------
# 1) Reliability threshold sweep (pose + NVS)
#    Hypothesis: moderate thresholds (0.75–0.85) improve pose stability; too high starves betas and NVs.
#    Params: enable_alternate_opt=true, confidence_threshold={0.75,0.8,0.85}, confidence_update_every=5
#
# 2) Reliability update cadence
#    Hypothesis: frequent updates stabilize early epochs; later updates have diminishing returns.
#    Params: confidence_update_every={1,5,10}, keep threshold fixed (e.g., 0.7)
#
# 3) Depth supervision strength (pose)
#    Hypothesis: small depth loss helps pose if depth is reliable; large weight may hurt.
#    Params: loss_weights.depth={0.0,0.1,1.0}
#
# 4) Silhouette loss strength (pose)
#    Hypothesis: stronger silhouette improves pose until over-constraining masks.
#    Params: loss_weights.sil={2.0,4.0,8.0}
#
# 5) Pose regularization
#    Hypothesis: higher reg_w reduces pose jitter; too high slows convergence.
#    Params: pose_tuning.reg_w={1e-5,1e-4,5e-4}
#
# 6) Pose phase length (must not overlap with GS)
#    Hypothesis: longer pose phase improves pose but may reduce GS learning time.
#    Params: pose_tuning.end_epoch={10,14,18}; gs_tuning.start_epoch=end+1; nv_gen_epoch=gs_start
#
# 7) Interpenetration loss (pose)
#     Hypothesis: small weight reduces collisions; too high can fight silhouette/SSIM.
#     Params: loss_weights.interpenetration={0.0,0.01,0.05,0.1},
#             interpenetration_margin={0.005,0.01,0.02},
#             interpenetration_max_points={1000,2000,5000}
#
# 8) ACAP margin (regularization)
#     Hypothesis: larger margin allows more deformation; smaller margin anchors tighter.
#     Params: regularization.acap_margin={0.005,0.01,0.02}
#
# 9) Learning rates (pose + 3DGS)
#     Hypothesis: pose LR affects convergence speed; GS LR affects detail vs stability.
#     Params: pose_tuning.lr={5e-5,1e-4,2e-4}, lr={5e-5,1e-4,2e-4}

# ---------------- Experiments (v920+) ----------------
# # Reliability threshold sweep
# exp_name="v920_conf075"
# python /home/cizinsky/master-thesis/schedule.py \
    # --exp_name $exp_name \
    # --job_name_prefix $exp_name \
    # --scene_name_includes "hi4d" \
    # --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.75 confidence_update_every=5

exp_name="v921_conf085"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.85 confidence_update_every=5

# Reliability update cadence
exp_name="v922_confupd1"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=1

exp_name="v923_confupd10"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=10

# Depth supervision strength
exp_name="v924_depth00"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 loss_weights.depth=0.0

exp_name="v925_depth10"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 loss_weights.depth=1.0

# Silhouette loss strength
exp_name="v926_sil2"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 loss_weights.sil=2.0

exp_name="v927_sil8"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 loss_weights.sil=8.0

# Pose regularization weight
exp_name="v928_poseReg1e5"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 pose_tuning.reg_w=1e-5

exp_name="v929_poseReg5e4"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 pose_tuning.reg_w=5e-4

# # Pose phase length (no overlap; nv_gen_epoch == gs_start)
# exp_name="v930_poseEnd10"
# python /home/cizinsky/master-thesis/schedule.py \
    # --exp_name $exp_name \
    # --job_name_prefix $exp_name \
    # --scene_name_includes "hi4d" \
    # --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 pose_tuning.end_epoch=10 gs_tuning.start_epoch=11 nv_gen_epoch=11

# exp_name="v931_poseEnd18"
# python /home/cizinsky/master-thesis/schedule.py \
    # --exp_name $exp_name \
    # --job_name_prefix $exp_name \
    # --scene_name_includes "hi4d" \
    # --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 pose_tuning.end_epoch=18 gs_tuning.start_epoch=19 nv_gen_epoch=19

# Interpenetration loss weight
exp_name="v932_inter00"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 loss_weights.interpenetration=0.0

exp_name="v933_inter005"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 loss_weights.interpenetration=0.05

# ACAP margin
exp_name="v934_acapM005"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 regularization.acap_margin=0.005

exp_name="v935_acapM02"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 regularization.acap_margin=0.02

# Learning rates (pose + 3DGS)
exp_name="v936_lrLow"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 pose_tuning.lr=5e-5 lr=5e-5

exp_name="v937_lrHigh"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --scene_name_includes "hi4d" \
    --overrides enable_alternate_opt=true difix.trn_enable=true confidence_threshold=0.8 confidence_update_every=5 pose_tuning.lr=2e-4 lr=2e-4
