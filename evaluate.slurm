#!/bin/bash
#SBATCH --job-name=v7_eval
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1 
#SBATCH --account=master 

# Change to the directory where the script is located
cd /home/cizinsky/master-thesis/evaluation

# exp_version=v7
# # V7 / Hi4d / pair00 / Ours
# seq_name=hi4d_pair00_dance00_cam76
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00
# cam_id=76
# bash eval_hi4d_ours.sh $seq_name $gt_dir $cam_id $exp_version

# # V7 / Hi4d / pair01 / Ours
# seq_name="hi4d_pair01_hug01_cam76"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair01/pair01/hug01
# cam_id=76
# bash eval_hi4d_ours.sh $seq_name $gt_dir $cam_id $exp_version

# # V7 / Hi4d / pair15 / Ours
# seq_name="hi4d_pair15_fight15_cam4"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
# cam_id=4
# bash eval_hi4d_ours.sh $seq_name $gt_dir $cam_id $exp_version

# # V7 / Hi4d / pair16 / Ours
# seq_name="hi4d_pair16_jump16_cam4"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# cam_id=4
# bash eval_hi4d_ours.sh $seq_name $gt_dir $cam_id $exp_version

# # V7 / Hi4d / pair17 / Ours
# seq_name="hi4d_pair17_dance17_cam28"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
# cam_id=28
# bash eval_hi4d_ours.sh $seq_name $gt_dir $cam_id $exp_version

# # V7 / Hi4d / pair19 / Ours
# seq_name="hi4d_pair19_piggyback19_cam4"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
# cam_id=4
# bash eval_hi4d_ours.sh $seq_name $gt_dir $cam_id $exp_version

# -----

# # Hi4d / pair00 / Multiply
# seq_name=hi4d_pair00_dance00_cam76
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00
# cam_id=76
# bash eval_hi4d_multiply.sh $seq_name $gt_dir $cam_id

# # Hi4d / pair01 / Multiply
# seq_name="hi4d_pair01_hug01_cam76"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair01/pair01/hug01
# cam_id=76
# bash eval_hi4d_multiply.sh $seq_name $gt_dir $cam_id

# # Hi4d / pair15 / Multiply
# seq_name="hi4d_pair15_fight15_cam4"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
# cam_id=4
# bash eval_hi4d_multiply.sh $seq_name $gt_dir $cam_id

# # Hi4d / pair16 / Multiply
# seq_name="hi4d_pair16_jump16_cam4"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# cam_id=4
# bash eval_hi4d_multiply.sh $seq_name $gt_dir $cam_id

# # Hi4d / pair17 / Multiply
# seq_name="hi4d_pair17_dance17_cam28"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
# cam_id=28
# bash eval_hi4d_multiply.sh $seq_name $gt_dir $cam_id

# # Hi4d / pair19 / Multiply
# seq_name="hi4d_pair19_piggyback19_cam4"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
# cam_id=4
# bash eval_hi4d_multiply.sh $seq_name $gt_dir $cam_id

# -----

# # Custom / football_high_res / Ours
# seq_name=football_high_res
# exp_version=v7
# bash eval_custom_ours.sh $seq_name $exp_version

# # Custom / taichi / Ours
# seq_name=taichi
# exp_version=v7
# bash eval_custom_ours.sh $seq_name $exp_version

# -----

# Custom / football_high_res / Multiply
# seq_name=football_high_res
# bash eval_custom_multiply.sh $seq_name

# # Custom / taichi / Multiply
# seq_name=taichi01
# bash eval_custom_multiply.sh $seq_name


# -----

# exp_version=v7
# # MMM /  Ours 
# seq_name=mmm_cheer_camorg
# bash eval_mmm_ours.sh $seq_name $exp_version

# seq_name=mmm_lift_camorg
# bash eval_mmm_ours.sh $seq_name $exp_version

# seq_name=mmm_dance_camorg
# bash eval_mmm_ours.sh $seq_name $exp_version

# seq_name=mmm_walkdance_camorg
# bash eval_mmm_ours.sh $seq_name $exp_version

# MMM / MultiPly
seq_name=mmm_cheer_camorg
bash eval_mmm_multiply.sh $seq_name

seq_name=mmm_lift_camorg
bash eval_mmm_multiply.sh $seq_name

seq_name=mmm_dance_camorg
bash eval_mmm_multiply.sh $seq_name

seq_name=mmm_walkdance_camorg
bash eval_mmm_multiply.sh $seq_name

# -----
# Summary
# python summarise.py --metrics-root /scratch/izar/cizinsky/thesis/evaluation/metrics --output-dir summaries
# bash compare_videos.sh /scratch/izar/cizinsky/thesis/evaluation/videos/renders/hi4d
bash compare_videos.sh /scratch/izar/cizinsky/thesis/evaluation/videos/renders/custom
bash mp4_to_gif.sh  /scratch/izar/cizinsky/thesis/evaluation/videos/renders/custom/comparison /scratch/izar/cizinsky/thesis/evaluation/videos/renders/custom/comparison_gif