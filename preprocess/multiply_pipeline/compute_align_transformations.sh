#!/bin/bash
set -e

# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd /home/cizinsky/master-thesis

PREPROCESS_DIR=/scratch/izar/cizinsky/multiply-output/preprocessing/data

# Hi4D
seq_name="hi4d_pair00_dance00_cam76"
gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00
cam_id=76
echo "Aligning: $seq_name"
python preprocess/multiply_pipeline/align_trace_to_gt.py \
    --preprocess_dir $PREPROCESS_DIR/$seq_name \
    --gt_smpl_dir $gt_dir/smpl \
    --gt_camera_file $gt_dir/cameras/rgb_cameras.npz \
    --dataset hi4d \
    --cam_id $cam_id \
    --visualize


seq_name="hi4d_pair01_hug01_cam76"
gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair01/pair01/hug01
cam_id=76
echo "Aligning: $seq_name"
python preprocess/multiply_pipeline/align_trace_to_gt.py \
    --preprocess_dir $PREPROCESS_DIR/$seq_name \
    --gt_smpl_dir $gt_dir/smpl \
    --gt_camera_file $gt_dir/cameras/rgb_cameras.npz \
    --dataset hi4d \
    --cam_id $cam_id \
    --visualize

seq_name="hi4d_pair15_fight15_cam4"
gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
cam_id=4
echo "Aligning: $seq_name"
python preprocess/multiply_pipeline/align_trace_to_gt.py \
    --preprocess_dir $PREPROCESS_DIR/$seq_name \
    --gt_smpl_dir $gt_dir/smpl \
    --gt_camera_file $gt_dir/cameras/rgb_cameras.npz \
    --dataset hi4d \
    --cam_id $cam_id \
    --visualize

seq_name="hi4d_pair16_jump16_cam4"
gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
cam_id=4
echo "Aligning: $seq_name"
python preprocess/multiply_pipeline/align_trace_to_gt.py \
    --preprocess_dir $PREPROCESS_DIR/$seq_name \
    --gt_smpl_dir $gt_dir/smpl \
    --gt_camera_file $gt_dir/cameras/rgb_cameras.npz \
    --dataset hi4d \
    --cam_id $cam_id \
    --visualize

seq_name="hi4d_pair17_dance17_cam28"
gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
cam_id=28
echo "Aligning: $seq_name"
python preprocess/multiply_pipeline/align_trace_to_gt.py \
    --preprocess_dir $PREPROCESS_DIR/$seq_name \
    --gt_smpl_dir $gt_dir/smpl \
    --gt_camera_file $gt_dir/cameras/rgb_cameras.npz \
    --dataset hi4d \
    --cam_id $cam_id \
    --visualize

seq_name="hi4d_pair19_piggyback19_cam4"
gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
cam_id=4
echo "Aligning: $seq_name"
python preprocess/multiply_pipeline/align_trace_to_gt.py \
    --preprocess_dir $PREPROCESS_DIR/$seq_name \
    --gt_smpl_dir $gt_dir/smpl \
    --gt_camera_file $gt_dir/cameras/rgb_cameras.npz \
    --dataset hi4d \
    --cam_id $cam_id \
    --visualize