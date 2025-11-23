#!/bin/bash
set -e
# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis

# Move to project directory
cd /home/cizinsky/master-thesis

# ------ scene: taichi 
# ensure I can run the simple training
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false debug=true group_name=dev max_epochs=10 

# ensure that progressive mask refinment works
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false debug=true group_name=dev max_epochs=1 mask_refinement.rebuild_every_epochs=1 mask_refinement.use_raw_smpl_until_epoch=-1

# ensure i can run the evaluation loop
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false debug=true group_name=dev max_epochs=10 eval_every_epochs=5

# ensure I can save and load checkpoints (resume training)
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false debug=true group_name=dev max_epochs=5 save_freq=5
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=true debug=true group_name=dev max_epochs=10 save_freq=5

# ------ scene: hi4d_pair00_dance00_cam76 
# 1. without dist and normal loss
# seq_name="hi4d_pair00_dance00_cam76"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00
# cam_id=76
# GROUP_NAME=dev
# python training/run.py \
    # scene_name=$seq_name \
    # tids=[0,1] \
    # train_bg=false \
    # resume=false \
    # debug=true \
    # group_name=$GROUP_NAME \
    # gt_seg_masks_dir=$gt_dir/seg/img_seg_mask/$cam_id \
    # gt_smpl_dir=$gt_dir/smpl \
    # max_epochs=10 \
    # eval_every_epochs=5 \
    # save_freq=5

# 2. with dist and normal loss
seq_name="hi4d_pair00_dance00_cam76"
gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00
cam_id=76
GROUP_NAME=dev
python training/run.py \
    scene_name=$seq_name \
    tids=[0,1] \
    train_bg=false \
    resume=false \
    debug=false \
    group_name=$GROUP_NAME \
    gt_seg_masks_dir=$gt_dir/seg/img_seg_mask/$cam_id \
    gt_smpl_dir=$gt_dir/smpl \
    max_epochs=30 \
    eval_every_epochs=15 \
    save_freq=5 \
    dist_loss=true \
    dist_start_iter=500 \
    normal_loss=true \
    normal_start_iter=1000 \
    'logger.tags=[v8, hi4d]'