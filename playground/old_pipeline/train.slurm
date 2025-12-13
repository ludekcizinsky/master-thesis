#!/bin/bash
#SBATCH --job-name=v8_taichi
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1 
#SBATCH --account=master 

# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis

# Change to the directory where the script is located
cd /home/cizinsky/master-thesis

# Full scene (default setup)
# GROUP_NAME=v3_default
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false logger.tags=[v3] group_name=$GROUP_NAME
# python training/run.py scene_name=taichi tids=[] train_bg=true resume=true logger.tags=[v3] group_name=$GROUP_NAME
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=true logger.tags=[v3] group_name=$GROUP_NAME gs_refine_start_iter=2000 gs_refine_stop_iter=8000

# Early stop iter
# GROUP_NAME=v3_earlystop
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v3, early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000
# python training/run.py scene_name=taichi tids=[] train_bg=true resume=true 'logger.tags=[v3, early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=true 'logger.tags=[v3, early_stop]' group_name=$GROUP_NAME gs_refine_start_iter=2000 gs_refine_stop_iter=8000

# Long stop iter
# GROUP_NAME=v3_longstop
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v3, long_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=15000
# python training/run.py scene_name=taichi tids=[] train_bg=true resume=true 'logger.tags=[v3, long_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=15000
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=true 'logger.tags=[v3, long_stop]' group_name=$GROUP_NAME gs_refine_start_iter=2000 gs_refine_stop_iter=8000

# Human long, static short
# GROUP_NAME=v3_humanlong_staticshort
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v3, human_long, static_short]' group_name=$GROUP_NAME gs_refine_stop_iter=15000
# python training/run.py scene_name=taichi tids=[] train_bg=true resume=true 'logger.tags=[v3, human_long, static_short]' group_name=$GROUP_NAME gs_refine_stop_iter=5000
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=true 'logger.tags=[v3, human_long, static_short]' group_name=$GROUP_NAME gs_refine_start_iter=2000 gs_refine_stop_iter=8000

# Full scene - joined optimisation - loss also on dynamic only
# GROUP_NAME=v3_fullscene_joinedloss
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=false 'logger.tags=[v3, dynamic_only_loss, full_scene_loss]' group_name=$GROUP_NAME gs_refine_stop_iter=15000

# Progressive sam enabled
# GROUP_NAME=v3_psam_rebuild25
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=false 'logger.tags=[v3, psam_rebuild25]' group_name=$GROUP_NAME gs_refine_stop_iter=15000 mask_refinement.rebuild_every_epochs=25

# psam50
# GROUP_NAME=v3_psam_rebuild50
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=false 'logger.tags=[v3, psam_rebuild50]' group_name=$GROUP_NAME gs_refine_stop_iter=15000 mask_refinement.rebuild_every_epochs=50

# psam10
# GROUP_NAME=v3_psam_rebuild10
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=false 'logger.tags=[v3, psam_rebuild10]' group_name=$GROUP_NAME gs_refine_stop_iter=15000 mask_refinement.rebuild_every_epochs=10

# psamm rebuild25 and rebuild mach epoch 1000
# GROUP_NAME=v3_psam_rebuild25_max1000
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=false 'logger.tags=[v3, psam_rebuild25]' group_name=$GROUP_NAME gs_refine_stop_iter=15000 mask_refinement.rebuild_every_epochs=25 mask_refinement.rebuild_max_epoch=1000

# psam early stop
# GROUP_NAME=v3_psam_early_stop_v2
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=false 'logger.tags=[v3, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=3500 mask_refinement.rebuild_every_epochs=25 mask_refinement.rebuild_max_epoch=1000

# psam early stop dynamic only
# GROUP_NAME=v3_psam_early_stop_dynamic_only
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v3, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000 mask_refinement.rebuild_every_epochs=25 mask_refinement.rebuild_max_epoch=1000
# python training/run.py scene_name=taichi tids=[] train_bg=true resume=true 'logger.tags=[v3, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000 mask_refinement.rebuild_every_epochs=10000000 mask_refinement.rebuild_max_epoch=1000 iters=15000
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=true 'logger.tags=[v3, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=400 mask_refinement.rebuild_every_epochs=10000000 mask_refinement.rebuild_max_epoch=1000 iters=15000

# testing new psam
# GROUP_NAME=v4_new_psam
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v4, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000 mask_refinement.rebuild_every_epochs=25 mask_refinement.rebuild_max_epoch=1000

# GROUP_NAME=v4_new_psam_small_n_pos_points
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v4, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000 mask_refinement.rebuild_every_epochs=25 mask_refinement.rebuild_max_epoch=1000 mask_refinement.sam2.max_pos_points=15 mask_refinement.sam2.max_neg_points=15

# GROUP_NAME=v4_new_psam_high_opacity_reg
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v4, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000 mask_refinement.rebuild_every_epochs=25 mask_refinement.rebuild_max_epoch=1000 mask_refinement.sam2.max_pos_points=15 mask_refinement.sam2.max_neg_points=15 opacity_reg=0.01 scale_reg=0.01

# --------------- V5 (better psam and pose optimizations) experiments ----------------
# GROUP_NAME=v5_default
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v5]' group_name=$GROUP_NAME

# GROUP_NAME=v5_default
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v5]' group_name=$GROUP_NAME

# GROUP_NAME=v5_default_high_opacity_scale_reg
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v5]' group_name=$GROUP_NAME opacity_reg=0.1 scale_reg=0.1

# GROUP_NAME=v5_3dgs_refine100
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v5]' group_name=$GROUP_NAME gs_refine_every=100

# GROUP_NAME=v5_3dgs_refine500
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v5]' group_name=$GROUP_NAME gs_refine_every=500

# GROUP_NAME=v5_pose_correction100
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v5]' group_name=$GROUP_NAME pose_correction_epoch=100

# GROUP_NAME=v5_pose_optim_interval5
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v5]' group_name=$GROUP_NAME pose_opt_interval=5

# GROUP_NAME=v5_sam_multimask_off
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v5]' group_name=$GROUP_NAME mask_refinement.sam2.multimask_output=false

# GROUP_NAME=v5_sam_n_iter1
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v5]' group_name=$GROUP_NAME mask_refinement.sam2.n_iterations=1

# GROUP_NAME=v5_sam_n_iter2
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v5]' group_name=$GROUP_NAME mask_refinement.sam2.n_iterations=2

# --------------- V6 reliability threshold adjustment and lower number of gs via regulisation + more frequent adapt. densification
# GROUP_NAME=v6_default
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v6]' group_name=$GROUP_NAME 

# GROUP_NAME=v6_default_rel_tol_003
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v6]' group_name=$GROUP_NAME mask_refinement.reliability_threshold_tolerance=0.03

# GROUP_NAME=v6_default_rel_tol_001
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v6]' group_name=$GROUP_NAME mask_refinement.reliability_threshold_tolerance=0.01

# GROUP_NAME=v6_default_rel_tol_01
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v6]' group_name=$GROUP_NAME mask_refinement.reliability_threshold_tolerance=0.1

# --------------- V7 depth order loss supervision
# GROUP_NAME=v7_default_corrected
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v7]' group_name=$GROUP_NAME

# GROUP_NAME=v7_normalise_by_pixels
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v7]' group_name=$GROUP_NAME depth_loss.normalise_by_pixels=true 

# GROUP_NAME=v7_more_frequent
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v7]' group_name=$GROUP_NAME depth_loss.interval=8

# GROUP_NAME=v7_less_frequent
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v7]' group_name=$GROUP_NAME depth_loss.interval=32

# GROUP_NAME=v7_weight_075
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v7]' group_name=$GROUP_NAME depth_loss.depth_order_weight=0.75

# GROUP_NAME=v7_weight_05
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v7]' group_name=$GROUP_NAME depth_loss.depth_order_weight=0.5

# GROUP_NAME=v7_weight_025
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v7]' group_name=$GROUP_NAME depth_loss.depth_order_weight=0.25

# GROUP_NAME=v7_football
# python training/run.py scene_name=football_high_res tids=[0,1] train_bg=false resume=false 'logger.tags=[v7, football]' group_name=$GROUP_NAME

# --------------- V7 hi4d experiments
# seq_name="hi4d_pair00_dance00_cam76"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00
# cam_id=76
# GROUP_NAME=v7_${seq_name}
# python training/run.py \
    # scene_name=$seq_name \
    # tids=[0,1] \
    # train_bg=false \
    # resume=false \
    # 'logger.tags=[v7]' \
    # group_name=$GROUP_NAME \
    # gt_seg_masks_dir=$gt_dir/seg/img_seg_mask/$cam_id \
    # gt_smpl_dir=$gt_dir/smpl


# seq_name="hi4d_pair01_hug01_cam76"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair01/pair01/hug01
# cam_id=76
# GROUP_NAME=v7_${seq_name}
# python training/run.py \
    # scene_name=$seq_name \
    # tids=[0,1] \
    # train_bg=false \
    # resume=false \
    # 'logger.tags=[v7]' \
    # group_name=$GROUP_NAME \
    # gt_seg_masks_dir=$gt_dir/seg/img_seg_mask/$cam_id \
    # gt_smpl_dir=$gt_dir/smpl


# seq_name="hi4d_pair15_fight15_cam4"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
# cam_id=4
# GROUP_NAME=v7_${seq_name}
# python training/run.py \
    # scene_name=$seq_name \
    # tids=[0,1] \
    # train_bg=false \
    # resume=false \
    # 'logger.tags=[v7]' \
    # group_name=$GROUP_NAME \
    # gt_seg_masks_dir=$gt_dir/seg/img_seg_mask/$cam_id \
    # gt_smpl_dir=$gt_dir/smpl

# seq_name="hi4d_pair16_jump16_cam4"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# cam_id=4
# GROUP_NAME=v7_${seq_name}
# python training/run.py \
    # scene_name=$seq_name \
    # tids=[0,1] \
    # train_bg=false \
    # resume=false \
    # 'logger.tags=[v7]' \
    # group_name=$GROUP_NAME \
    # gt_seg_masks_dir=$gt_dir/seg/img_seg_mask/$cam_id \
    # gt_smpl_dir=$gt_dir/smpl


# seq_name="hi4d_pair17_dance17_cam28"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
# cam_id=28
# GROUP_NAME=v7_${seq_name}
# python training/run.py \
    # scene_name=$seq_name \
    # tids=[0,1] \
    # train_bg=false \
    # resume=false \
    # 'logger.tags=[v7]' \
    # group_name=$GROUP_NAME \
    # gt_seg_masks_dir=$gt_dir/seg/img_seg_mask/$cam_id \
    # gt_smpl_dir=$gt_dir/smpl


# seq_name="hi4d_pair19_piggyback19_cam4"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
# cam_id=4
# GROUP_NAME=v7_${seq_name}
# python training/run.py \
    # scene_name=$seq_name \
    # tids=[0,1] \
    # train_bg=false \
    # resume=false \
    # 'logger.tags=[v7]' \
    # group_name=$GROUP_NAME \
    # gt_seg_masks_dir=$gt_dir/seg/img_seg_mask/$cam_id \
    # gt_smpl_dir=$gt_dir/smpl

# ------- V7 MMM experiments
# GROUP_NAME=v7
# python training/run.py scene_name=mmm_cheer_camorg tids=[0,1,2] train_bg=false resume=false 'logger.tags=[v7, mmm]' group_name=$GROUP_NAME
# python training/run.py scene_name=mmm_dance_camorg tids=[0,1,2,3] train_bg=false resume=false 'logger.tags=[v7, mmm]' group_name=$GROUP_NAME
# python training/run.py scene_name=mmm_lift_camorg tids=[0,1,2] train_bg=false resume=false 'logger.tags=[v7, mmm]' group_name=$GROUP_NAME
# python training/run.py scene_name=mmm_walkdance_camorg tids=[0,1,2] train_bg=false resume=false 'logger.tags=[v7, mmm]' group_name=$GROUP_NAME

# --------------- V8 hi4d pair00_dance00_cam76 experiments to determine best hyper params
# # -- reg start iter 1000
# seq_name="hi4d_pair00_dance00_cam76"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00
# cam_id=76
# GROUP_NAME=v8_regstart1000
# reg_start=1000
# python training/run.py \
    # scene_name=$seq_name \
    # tids=[0,1] \
    # train_bg=false \
    # resume=false \
    # group_name=$GROUP_NAME \
    # gt_seg_masks_dir=$gt_dir/seg/img_seg_mask/$cam_id \
    # gt_smpl_dir=$gt_dir/smpl \
    # dist_loss=true \
    # dist_start_iter=$reg_start \
    # normal_loss=true \
    # normal_start_iter=$reg_start \
    # 'logger.tags=[v8, hi4d]'

# -- reg start iter 5000
# seq_name="hi4d_pair00_dance00_cam76"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00
# cam_id=76
# GROUP_NAME=v8_regstart5000
# reg_start=5000
# python training/run.py \
    # scene_name=$seq_name \
    # tids=[0,1] \
    # train_bg=false \
    # resume=false \
    # group_name=$GROUP_NAME \
    # gt_seg_masks_dir=$gt_dir/seg/img_seg_mask/$cam_id \
    # gt_smpl_dir=$gt_dir/smpl \
    # dist_loss=true \
    # dist_start_iter=$reg_start \
    # normal_loss=true \
    # normal_start_iter=$reg_start \
    # 'logger.tags=[v8, hi4d]'

# -- adaptive densification off
# seq_name="hi4d_pair00_dance00_cam76"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00
# cam_id=76
# GROUP_NAME=v8_adaptiveoff
# reg_start=1000
# super_high_number=1000000
# python training/run.py \
    # scene_name=$seq_name \
    # tids=[0,1] \
    # train_bg=false \
    # resume=false \
    # group_name=$GROUP_NAME \
    # gt_seg_masks_dir=$gt_dir/seg/img_seg_mask/$cam_id \
    # gt_smpl_dir=$gt_dir/smpl \
    # dist_loss=true \
    # dist_start_iter=$reg_start \
    # normal_loss=true \
    # normal_start_iter=$reg_start \
    # 'logger.tags=[v8, hi4d]' \
    # gs_refine_stop_iter=$super_high_number \
    # gs_refine_start_iter=$super_high_number

# -- high distortion loss weight (0.1 instead of 0.01)
# seq_name="hi4d_pair00_dance00_cam76"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00
# cam_id=76
# GROUP_NAME=v8_distlambda01
# reg_start=1000
# python training/run.py \
    # scene_name=$seq_name \
    # tids=[0,1] \
    # train_bg=false \
    # resume=false \
    # group_name=$GROUP_NAME \
    # gt_seg_masks_dir=$gt_dir/seg/img_seg_mask/$cam_id \
    # gt_smpl_dir=$gt_dir/smpl \
    # dist_loss=true \
    # dist_start_iter=$reg_start \
    # normal_loss=true \
    # normal_start_iter=$reg_start \
    # 'logger.tags=[v8, hi4d]' \
    # dist_lambda=0.1 

# -- high normal loss weight (0.5 instead of 0.05)
# seq_name="hi4d_pair00_dance00_cam76"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00
# cam_id=76
# GROUP_NAME=v8_normlambda05
# reg_start=1000
# python training/run.py \
    # scene_name=$seq_name \
    # tids=[0,1] \
    # train_bg=false \
    # resume=false \
    # group_name=$GROUP_NAME \
    # gt_seg_masks_dir=$gt_dir/seg/img_seg_mask/$cam_id \
    # gt_smpl_dir=$gt_dir/smpl \
    # dist_loss=true \
    # dist_start_iter=$reg_start \
    # normal_loss=true \
    # normal_start_iter=$reg_start \
    # 'logger.tags=[v8, hi4d]' \
    # normal_lambda=0.5 

# ----- v8 taichi
# seq_name="taichi"
# GROUP_NAME=v8
# python training/run.py \
    # scene_name=$seq_name \
    # tids=[0,1] \
    # train_bg=false \
    # resume=false \
    # 'logger.tags=[v8, taichi]' \
    # group_name=$GROUP_NAME

# ----- v8 taichi
seq_name="male_3_casual"
GROUP_NAME=v8
python training/run.py \
    scene_name=$seq_name \
    tids=[0] \
    train_bg=false \
    resume=false \
    'logger.tags=[v8, people_snapshot]' \
    group_name=$GROUP_NAME