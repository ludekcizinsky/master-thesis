#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

preprocessing_root=/scratch/izar/cizinsky/ait_datasets/full/mmm
echo "Reformatting mmm_dance"
python preprocess/ait_datasets/mmm/other_reformat.py --scene-root-dir $preprocessing_root/dance 
echo "Reformatting mmm_lift"
python preprocess/ait_datasets/mmm/other_reformat.py --scene-root-dir $preprocessing_root/lift
echo "Reformatting mmm_walkdance"
python preprocess/ait_datasets/mmm/other_reformat.py --scene-root-dir $preprocessing_root/walkdance 