#!/bin/bash
#SBATCH --job-name=prep_mmm
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=01:00:00
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

# ----

# Hi4D 
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq hi4d_pair00_dance00_cam76 --images-dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00/images/76
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq hi4d_pair01_hug01_cam76 --images-dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair01/pair01/hug01/images/76
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq hi4d_pair15_fight15_cam4 --images-dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15/images/4
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq hi4d_pair16_jump16_cam4 --images-dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16/images/4
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq hi4d_pair17_dance17_cam28 --images-dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17/images/28
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq hi4d_pair19_piggyback19_cam4 --images-dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19/images/4


# MMM
# 1. Downscale the dataset
# bash preprocess/ait_datasets/mmm/downscale_mmm_dataset.sh /scratch/izar/cizinsky/ait_datasets/full/mmm/cheer --org-factor 2 --stage-factor 4 --downsample 2
# bash preprocess/ait_datasets/mmm/downscale_mmm_dataset.sh /scratch/izar/cizinsky/ait_datasets/full/mmm/dance --org-factor 2 --stage-factor 4 --downsample 2
# bash preprocess/ait_datasets/mmm/downscale_mmm_dataset.sh /scratch/izar/cizinsky/ait_datasets/full/mmm/hug --org-factor 2 --stage-factor 4 --downsample 2
# bash preprocess/ait_datasets/mmm/downscale_mmm_dataset.sh /scratch/izar/cizinsky/ait_datasets/full/mmm/lift --org-factor 2 --stage-factor 4 --downsample 2
# bash preprocess/ait_datasets/mmm/downscale_mmm_dataset.sh /scratch/izar/cizinsky/ait_datasets/full/mmm/selfie --org-factor 2 --stage-factor 4 --downsample 2
# bash preprocess/ait_datasets/mmm/downscale_mmm_dataset.sh /scratch/izar/cizinsky/ait_datasets/full/mmm/walkdance --org-factor 2 --stage-factor 4 --downsample 2

# 2. Run the full preprocessing pipeline
multiply_ds_dir=/scratch/izar/cizinsky/ait_datasets/full/mmm
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq mmm_cheer_camorg --images-dir $multiply_ds_dir/cheer/org_img_down2x_subsample2x --num-people 3
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq mmm_dance_camorg --images-dir $multiply_ds_dir/dance/org_image_down2x_subsample2x --num-people 4
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq mmm_hug_camorg --images-dir $multiply_ds_dir/hug/org_img_down2x_subsample2x --num-people 2
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq mmm_lift_camorg --images-dir $multiply_ds_dir/lift/org_image_down2x_subsample2x --num-people 3
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq mmm_selfie_camorg --images-dir $multiply_ds_dir/selfie/org_img_down2x_subsample2x --num-people 2
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq mmm_walkdance_camorg --images-dir $multiply_ds_dir/walkdance/org_image_down2x_subsample2x --num-people 3

# Custom 
bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq taichi --images-dir /scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi/image --num-people 2 

# People SnapShots
# - male 3 casual
bash preprocess/people_snapshots/video_to_frame.sh /scratch/izar/cizinsky/people_snapshot_public/male-3-casual/male-3-casual.mp4 3
bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq male_3_casual --images-dir /scratch/izar/cizinsky/people_snapshot_public/male-3-casual/male-3-casual_frames --num-people 1