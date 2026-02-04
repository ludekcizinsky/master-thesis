#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

reformat_script_path=preprocess/eval/mmm/helpers/reformat.py

preprocessing_root=/scratch/izar/cizinsky/ait_datasets/full/mmm
echo "Reformatting mmm_dance"
python $reformat_script_path --scene-root-dir $preprocessing_root/dance 
echo "Reformatting mmm_lift"
python $reformat_script_path --scene-root-dir $preprocessing_root/lift
echo "Reformatting mmm_walkdance"
python $reformat_script_path --scene-root-dir $preprocessing_root/walkdance 
