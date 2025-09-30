#!/bin/bash
#SBATCH --job-name=bbox_crop_1
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1 
#SBATCH --account=master 

# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis

# Change to the directory where the script is located
cd /home/cizinsky/master-thesis

# # Refinement frequency 100
# python training/run.py scene_name=modric_vs_ribberi gs_refine_every=100 'logger.tags=[h1, refine100]'
# python training/run.py scene_name=modric_vs_ribberi gs_refine_every=200 'logger.tags=[h1, refine200]'
# python training/run.py scene_name=modric_vs_ribberi gs_refine_every=300 'logger.tags=[h1, refine300]'

# # Refinement stop iteration 5000
# python training/run.py scene_name=modric_vs_ribberi gs_refine_stop_iter=5000 'logger.tags=[h1, refstop5k]'
# python training/run.py scene_name=modric_vs_ribberi gs_refine_stop_iter=10000 'logger.tags=[h1, refstop10k]'
# python training/run.py scene_name=modric_vs_ribberi gs_refine_stop_iter=15000 'logger.tags=[h1, refstop15k]'

# # Opacity penalty weight 0.01
# python training/run.py scene_name=modric_vs_ribberi opa_lambda=0.01 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, opa001]'
# python training/run.py scene_name=modric_vs_ribberi opa_lambda=0.005 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, opa0005]'
# python training/run.py scene_name=modric_vs_ribberi opa_lambda=0.001 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, opa0001]'

# python training/run.py scene_name=modric_vs_ribberi opa_lambda=0.01 gs_refine_every=200 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, refine200, opa001]'
# python training/run.py scene_name=modric_vs_ribberi opa_lambda=0.005 gs_refine_every=200 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, refine200, opa0005]'
# python training/run.py scene_name=modric_vs_ribberi opa_lambda=0.001 gs_refine_every=200 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, refine200, opa0001]'

# # MA weight 1.0
# python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, ma1]'
# python training/run.py scene_name=modric_vs_ribberi ma_lambda=5.0 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, ma5]'
# python training/run.py scene_name=modric_vs_ribberi ma_lambda=10.0 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, ma10]'

# python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 gs_refine_every=200 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, refine200, ma1]'
# python training/run.py scene_name=modric_vs_ribberi ma_lambda=5.0 gs_refine_every=200 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, refine200, ma5]'
# python training/run.py scene_name=modric_vs_ribberi ma_lambda=10.0 gs_refine_every=200 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, refine200, ma10]'


# python training/run.py scene_name=modric_vs_ribberi ma_lambda=5.0 opa_lambda=0.01 gs_refine_every=200 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, refine200, opa001, ma10]'
# python training/run.py scene_name=modric_vs_ribberi ma_lambda=5.0 opa_lambda=0.005 gs_refine_every=200 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, refine200, opa0005, ma10]'

# python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 opa_lambda=0.01 gs_refine_every=200 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, refine200, opa001, ma1]'
# python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 opa_lambda=0.005 gs_refine_every=200 gs_refine_stop_iter=10000 'logger.tags=[h1, refinestop10k, refine200, opa0005, ma1]'

# python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 opa_lambda=0.005 gs_refine_every=200 gs_refine_stop_iter=10000 means_lr=0.001 'logger.tags=[h1, refinestop10k, refine200, opa0005, ma1, mlr0001]'
# python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 opa_lambda=0.005 gs_refine_every=200 gs_refine_stop_iter=10000 downscale=1 'logger.tags=[h1, refinestop10k, refine200, opa0005, ma1, downscale1]'


# python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 opa_lambda=0.005 gs_refine_every=200 gs_refine_stop_iter=10000 downscale=1 mask_type="bbox_crop" 'logger.tags=[h1, refinestop10k, refine200, opa0005, ma1, downscale1, bboxcrop]'
# python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 opa_lambda=0.005 gs_refine_every=200 gs_refine_stop_iter=15000 downscale=1 mask_type="bbox_crop" 'logger.tags=[h1, refinestop10k, refine200, opa0005, ma1, downscale1, bboxcrop]'
# python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 opa_lambda=0.005 gs_refine_every=150 gs_refine_stop_iter=10000 downscale=1 mask_type="bbox_crop" 'logger.tags=[h1, refinestop10k, refine150, opa0005, ma1, downscale1, bboxcrop]'

# python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 opa_lambda=0.005 gs_refine_every=200 gs_refine_stop_iter=15000 downscale=1 iters=40000 'logger.tags=[h1, refinestop15k, refine200, opa0005, ma1, downscale1, iters40k]'
# python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 opa_lambda=0.005 gs_refine_every=200 gs_refine_stop_iter=20000 downscale=1 iters=40000 'logger.tags=[h1, refinestop20k, refine200, opa0005, ma1, downscale1, iters40k]'
# python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 opa_lambda=0.005 gs_refine_every=200 gs_refine_stop_iter=25000 downscale=1 iters=40000 'logger.tags=[h1, refinestop25k, refine200, opa0005, ma1, downscale1, iters40k]'

python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 opa_lambda=0.005 gs_refine_every=200 gs_refine_stop_iter=10000 downscale=1 opa_free_radius=0.005 'logger.tags=[h1, refinestop10k, refine200, opa0005, opafr0005, ma1, downscale1]'
python training/run.py scene_name=modric_vs_ribberi ma_lambda=1.0 opa_lambda=0.005 gs_refine_every=200 gs_refine_stop_iter=10000 downscale=1 opa_free_radius=0.001 'logger.tags=[h1, refinestop10k, refine200, opa0005, opafr0001, ma1, downscale1]'

python training/run.py scene_name=modric_vs_ribberi opa_lambda=0.05 gs_refine_every=500 gs_refine_stop_iter=10000 opa_free_radius=0.0001 lbs_knn=30 'logger.tags=[h1, refinestop10k, refine500, opa005, lbs, opafr00001, knn30]'