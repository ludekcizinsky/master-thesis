
#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

echo "refomatting pair15 - fight"
scene_root_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
python preprocess/ait_datasets/hi4d/meshes_reformat.py --scene-root-dir $scene_root_dir 

echo "refomatting pair16 - jump"
scene_root_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
python preprocess/ait_datasets/hi4d/meshes_reformat.py --scene-root-dir $scene_root_dir --first-frame-number 11
echo "refomatting pair17 - dance"
scene_root_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
python preprocess/ait_datasets/hi4d/meshes_reformat.py --scene-root-dir $scene_root_dir 

echo "refomatting pair19 - piggyback"
scene_root_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
python preprocess/ait_datasets/hi4d/meshes_reformat.py --scene-root-dir $scene_root_dir 