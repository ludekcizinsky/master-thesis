# !/bin/bash

# Things to set
pretrained_model_dir=/scratch/izar/cizinsky/pretrained
root_repo_path=/home/cizinsky/master-thesis

# Activate conda 
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh

# Create conda env
conda create -n lhm python=3.10
module load gcc cuda/12.1

# Navigate to lhm directory
cd $root_repo_path/submodules/lhm

# Activate conda env
conda activate lhm

# Install dependencies
# install torch 2.3.0
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# install dependencies
pip install -r requirements.txt

pip install --no-build-isolation "git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17"

# install from source code to avoid the conflict with torchvision
pip uninstall basicsr -y
pip install git+https://github.com/XPixelGroup/BasicSR

# install pytorch3d
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# install diff-gaussian-rasterization
pip install --no-build-isolation "git+https://github.com/ashawkey/diff-gaussian-rasterization/"

# install simple-knn
pip install --no-build-isolation "git+https://github.com/camenduru/simple-knn/"

# install onnxruntime
pip install onnxruntime


# Download pretrained model
# cd /scratch/izar/cizinsky/pretrained
# wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_prior_model.tar 
# tar -xvf LHM_prior_model.tar 