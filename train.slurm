#!/bin/bash
#SBATCH --job-name=v5_runs
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=03:00:00
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

# Full scene (default setup)
# GROUP_NAME=v3_default
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false logger.tags=[v3] group_name=$GROUP_NAME
# python training/run.py scene_name=taichi tids=[] train_bg=true resume=true logger.tags=[v3] group_name=$GROUP_NAME
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=true logger.tags=[v3] group_name=$GROUP_NAME gs_refine_start_iter=2000 gs_refine_stop_iter=8000

# Early stop iter
# GROUP_NAME=v3_earlystop
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v3, early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000
# python training/run.py scene_name=taichi tids=[] train_bg=true resume=true 'logger.tags=[v3, early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=true 'logger.tags=[v3, early_stop]' group_name=$GROUP_NAME gs_refine_start_iter=2000 gs_refine_stop_iter=8000

# Long stop iter
# GROUP_NAME=v3_longstop
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v3, long_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=15000
# python training/run.py scene_name=taichi tids=[] train_bg=true resume=true 'logger.tags=[v3, long_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=15000
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=true 'logger.tags=[v3, long_stop]' group_name=$GROUP_NAME gs_refine_start_iter=2000 gs_refine_stop_iter=8000

# Human long, static short
# GROUP_NAME=v3_humanlong_staticshort
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v3, human_long, static_short]' group_name=$GROUP_NAME gs_refine_stop_iter=15000
# python training/run.py scene_name=taichi tids=[] train_bg=true resume=true 'logger.tags=[v3, human_long, static_short]' group_name=$GROUP_NAME gs_refine_stop_iter=5000
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=true 'logger.tags=[v3, human_long, static_short]' group_name=$GROUP_NAME gs_refine_start_iter=2000 gs_refine_stop_iter=8000

# Full scene - joined optimisation - loss also on dynamic only
# GROUP_NAME=v3_fullscene_joinedloss
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=false 'logger.tags=[v3, dynamic_only_loss, full_scene_loss]' group_name=$GROUP_NAME gs_refine_stop_iter=15000

# Progressive sam enabled
# GROUP_NAME=v3_psam_rebuild25
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=false 'logger.tags=[v3, psam_rebuild25]' group_name=$GROUP_NAME gs_refine_stop_iter=15000 mask_refinement.rebuild_every_epochs=25

# psam50
# GROUP_NAME=v3_psam_rebuild50
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=false 'logger.tags=[v3, psam_rebuild50]' group_name=$GROUP_NAME gs_refine_stop_iter=15000 mask_refinement.rebuild_every_epochs=50

# psam10
# GROUP_NAME=v3_psam_rebuild10
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=false 'logger.tags=[v3, psam_rebuild10]' group_name=$GROUP_NAME gs_refine_stop_iter=15000 mask_refinement.rebuild_every_epochs=10

# psamm rebuild25 and rebuild mach epoch 1000
# GROUP_NAME=v3_psam_rebuild25_max1000
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=false 'logger.tags=[v3, psam_rebuild25]' group_name=$GROUP_NAME gs_refine_stop_iter=15000 mask_refinement.rebuild_every_epochs=25 mask_refinement.rebuild_max_epoch=1000

# psam early stop
# GROUP_NAME=v3_psam_early_stop_v2
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=false 'logger.tags=[v3, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=3500 mask_refinement.rebuild_every_epochs=25 mask_refinement.rebuild_max_epoch=1000

# psam early stop dynamic only
# GROUP_NAME=v3_psam_early_stop_dynamic_only
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v3, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000 mask_refinement.rebuild_every_epochs=25 mask_refinement.rebuild_max_epoch=1000
# python training/run.py scene_name=taichi tids=[] train_bg=true resume=true 'logger.tags=[v3, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000 mask_refinement.rebuild_every_epochs=10000000 mask_refinement.rebuild_max_epoch=1000 iters=15000
# python training/run.py scene_name=taichi tids=[0,1] train_bg=true resume=true 'logger.tags=[v3, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=400 mask_refinement.rebuild_every_epochs=10000000 mask_refinement.rebuild_max_epoch=1000 iters=15000

# testing new psam
# GROUP_NAME=v4_new_psam
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v4, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000 mask_refinement.rebuild_every_epochs=25 mask_refinement.rebuild_max_epoch=1000

# GROUP_NAME=v4_new_psam_small_n_pos_points
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v4, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000 mask_refinement.rebuild_every_epochs=25 mask_refinement.rebuild_max_epoch=1000 mask_refinement.sam2.max_pos_points=15 mask_refinement.sam2.max_neg_points=15

# GROUP_NAME=v4_new_psam_high_opacity_reg
# python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v4, psam_early_stop]' group_name=$GROUP_NAME gs_refine_stop_iter=5000 mask_refinement.rebuild_every_epochs=25 mask_refinement.rebuild_max_epoch=1000 mask_refinement.sam2.max_pos_points=15 mask_refinement.sam2.max_neg_points=15 opacity_reg=0.01 scale_reg=0.01

# --------------- V5 (better psam and pose optimizations) experiments ----------------
GROUP_NAME=v5_default
python training/run.py scene_name=taichi tids=[0,1] train_bg=false resume=false 'logger.tags=[v5]' group_name=$GROUP_NAME

# --- dev
# python training/run.py scene_name=football_high_res tids=[0,1] train_bg=false resume=true logger.tags=[v2] group_name=$GROUP_NAME debug=true save_freq=30000
