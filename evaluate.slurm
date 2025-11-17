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
# bash eval_ours_single_experiment.sh $seq_name $gt_dir $cam_id $exp_version

# # V7 / Hi4d / pair01 / Ours
# seq_name="hi4d_pair01_hug01_cam76"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair01/pair01/hug01
# cam_id=76
# bash eval_ours_single_experiment.sh $seq_name $gt_dir $cam_id $exp_version

# # V7 / Hi4d / pair15 / Ours
# seq_name="hi4d_pair15_fight15_cam4"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
# cam_id=4
# bash eval_ours_single_experiment.sh $seq_name $gt_dir $cam_id $exp_version

# # V7 / Hi4d / pair16 / Ours
# seq_name="hi4d_pair16_jump16_cam4"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# cam_id=4
# bash eval_ours_single_experiment.sh $seq_name $gt_dir $cam_id $exp_version

# # V7 / Hi4d / pair17 / Ours
# seq_name="hi4d_pair17_dance17_cam28"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
# cam_id=28
# bash eval_ours_single_experiment.sh $seq_name $gt_dir $cam_id $exp_version

# # V7 / Hi4d / pair19 / Ours
# seq_name="hi4d_pair19_piggyback19_cam4"
# gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
# cam_id=4
# bash eval_ours_single_experiment.sh $seq_name $gt_dir $cam_id $exp_version

# Hi4d / pair00 / Multiply
seq_name=hi4d_pair00_dance00_cam76
gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00
cam_id=76
bash eval_multiply_single_experiment.sh $seq_name $gt_dir $cam_id

# Hi4d / pair01 / Multiply
seq_name="hi4d_pair01_hug01_cam76"
gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair01/pair01/hug01
cam_id=76
bash eval_multiply_single_experiment.sh $seq_name $gt_dir $cam_id

# Hi4d / pair15 / Multiply
seq_name="hi4d_pair15_fight15_cam4"
gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
cam_id=4
bash eval_multiply_single_experiment.sh $seq_name $gt_dir $cam_id

# Hi4d / pair16 / Multiply
seq_name="hi4d_pair16_jump16_cam4"
gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
cam_id=4
bash eval_multiply_single_experiment.sh $seq_name $gt_dir $cam_id

# Hi4d / pair17 / Multiply
seq_name="hi4d_pair17_dance17_cam28"
gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
cam_id=28
bash eval_multiply_single_experiment.sh $seq_name $gt_dir $cam_id

# Hi4d / pair19 / Multiply
seq_name="hi4d_pair19_piggyback19_cam4"
gt_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
cam_id=4
bash eval_multiply_single_experiment.sh $seq_name $gt_dir $cam_id