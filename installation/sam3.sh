# !/bin/bash

# Things to set
pretrained_model_dir=/scratch/izar/cizinsky/pretrained
root_repo_path=/home/cizinsky/master-thesis

# Activate conda 
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh

# Create conda env
conda create -n sam3 python=3.12

# Navigate to sam3 directory
cd $root_repo_path/submodules/sam3

# Activate conda env
conda activate sam3

# Install dependencies
# - basics
module load gcc cuda/12.1
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install .
pip install opencv-python
pip install einops
pip install decord
pip install pycocotools
pip install psutil
pip install matplotlib pandas 
pip install scikit-learn
pip install tyro