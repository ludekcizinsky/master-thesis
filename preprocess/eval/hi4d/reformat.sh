#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

repo_dir=/home/cizinsky/master-thesis
cd $repo_dir

reformat_script_path=preprocess/eval/hi4d/helpers/reformat.py
smpl2smplx_script_path=submodules/smplx/tools/run_conversion.sh

echo "Reformatting hi4d_pair15_fight"
scene_root_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
python $reformat_script_path --scene-root-dir $scene_root_dir 
bash $smpl2smplx_script_path $scene_root_dir smpl smplx

echo "Reformatting hi4d_pair16_jump"
scene_root_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
python $reformat_script_path --scene-root-dir $scene_root_dir --first-frame-number 11
bash $smpl2smplx_script_path $scene_root_dir smpl smplx

echo "Reformatting hi4d_pair17_dance"
scene_root_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
python $reformat_script_path --scene-root-dir $scene_root_dir
bash $smpl2smplx_script_path $scene_root_dir smpl smplx

echo "Reformatting hi4d_pair19_piggyback"
scene_root_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
python $reformat_script_path --scene-root-dir $scene_root_dir 
bash $smpl2smplx_script_path $scene_root_dir smpl smplx
