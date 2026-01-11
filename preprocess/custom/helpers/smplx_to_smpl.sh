#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis
preprocessing_root=/scratch/izar/cizinsky/thesis/preprocessing

# Hi4D
echo "SMPLX to SMPL for hi4d_pair15_fight"
gt_scene_root_dir="/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15"
bash submodules/smplx/tools/run_conversion.sh $preprocessing_root/hi4d_pair15_fight smplx smpl


echo "SMPLX to SMPL for hi4d_pair16_jump"
gt_scene_root_dir="/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16"
bash submodules/smplx/tools/run_conversion.sh $preprocessing_root/hi4d_pair16_jump smplx smpl


echo "SMPLX to SMPL for hi4d_pair17_dance"
gt_scene_root_dir="/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17"
bash submodules/smplx/tools/run_conversion.sh $preprocessing_root/hi4d_pair17_dance smplx smpl


echo "SMPLX to SMPL for hi4d_pair19_piggyback"
gt_scene_root_dir="/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19"
bash submodules/smplx/tools/run_conversion.sh $preprocessing_root/hi4d_pair19_piggyback smplx smpl  