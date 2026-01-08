# !/bin/bash

# Things to set
pretrained_model_dir=/scratch/izar/cizinsky/pretrained
root_repo_path=/home/cizinsky/master-thesis

# Activate conda 
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh

# Create conda env
conda create -n human3r python=3.11

# Navigate to human3r directory
cd $root_repo_path/submodules/human3r

# # Activate conda env
conda activate human3r

# Install dependencies
# - basics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
conda install 'llvm-openmp<16'
pip install --no-build-isolation "git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17"
# - comilation of curope
module load gcc cuda/12.1
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../

# Fetch SMPLX models
bash ./scripts/fetch_smplx.sh

# Human3R checkpoints
huggingface-cli download faneggg/human3r human3r.pth --local-dir $pretrained_model_dir