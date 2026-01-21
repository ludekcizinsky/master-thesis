#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# navigate to project directory
cd /home/cizinsky/master-thesis

# ------- Hi4d
# seq_name="hi4d_pair15_fight"
# num_persons=2
# exp_name="v2_of_pose_est_metrics"
# source_cam_id=4
# target_cam_ids="[76,4,28]"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
# python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id  nvs_eval.target_camera_ids=$target_cam_ids  wandb.enable=false eval_pretrain=true difix.trn_enable=false

# seq_name="hi4d_pair16_jump"
# num_persons=2
# exp_name="v6_check_if_we_can_still_run_trn_nv_gen"
# source_cam_id=4
# target_cam_ids="[76,4,28]"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id  nvs_eval.target_camera_ids=$target_cam_ids  wandb.enable=false eval_pretrain=false difix.trn_enable=true

# seq_name="hi4d_pair17_dance"
# num_persons=2
# exp_name="v203_white_bg_check_refine"
# source_cam_id=28
# target_cam_ids="[4,28,52]"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
# python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id  nvs_eval.target_camera_ids=$target_cam_ids  wandb.enable=false eval_pretrain=false difix.trn_enable=true nv_gen_epoch=0 


# ------- MMM
# seq_name="mmm_dance"
# num_persons=4
# exp_name="v5_can_we_run_with_nv_cams_and_difix"
# source_cam_id=0
# target_cam_ids="[]"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/mmm/dance
# preprocessing_dir=/scratch/izar/cizinsky/thesis/preprocessing/mmm_dance
# trn_nv_cam_ids="[0,100,101,102,103,104,105,107]"
# python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id  nvs_eval.target_camera_ids=$target_cam_ids  wandb.enable=false eval_pretrain=false difix.trn_enable=true test_masks_scene_dir=null test_smpl_params_scene_dir=null smpl_params_scene_dir=null test_smplx_params_scene_dir=null cameras_scene_dir=$preprocessing_dir trn_nv_gen.camera_ids=$trn_nv_cam_ids


# seq_name="mmm_lift"
# num_persons=3
# exp_name="v204_save_individual_meshes_eval_pretrain"
# source_cam_id=0
# target_cam_ids="[]"
# trn_nv_gen_camera_ids="[0,100,107]"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/mmm/lift
# preprocessing_dir=/scratch/izar/cizinsky/thesis/preprocessing/mmm_lift
# python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id  nvs_eval.target_camera_ids=$target_cam_ids  wandb.enable=false eval_pretrain=true difix.trn_enable=false test_masks_scene_dir=null test_smpl_params_scene_dir=null smpl_params_scene_dir=null test_smplx_params_scene_dir=null cameras_scene_dir=$preprocessing_dir nv_gen_epoch=-1 trn_nv_gen.camera_ids=$trn_nv_gen_camera_ids

# seq_name="mmm_walkdance"
# num_persons=3
# exp_name="v3_large_refactor_check"
# source_cam_id=0
# target_cam_ids="[]"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/mmm/walkdance
# preprocessing_dir=/scratch/izar/cizinsky/thesis/preprocessing/mmm_walkdance
# python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id  nvs_eval.target_camera_ids=$target_cam_ids  wandb.enable=false eval_pretrain=true difix.trn_enable=false test_masks_scene_dir=null test_smpl_params_scene_dir=null smpl_params_scene_dir=null test_smplx_params_scene_dir=null cameras_scene_dir=$preprocessing_dir

# ------- In-The-Wild
seq_name="taichi"
num_persons=2
exp_name="v102_smplx_meshes_saved_as_well"
source_cam_id=0
target_cam_ids="[]"
root_gt_dir_path=null
preprocessing_dir=/scratch/izar/cizinsky/thesis/preprocessing/taichi
trn_nv_cam_ids="[0,100,101,102,103,104,105,107]"
python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id  nvs_eval.target_camera_ids=$target_cam_ids  wandb.enable=false eval_pretrain=true difix.trn_enable=false smpl_params_scene_dir=null cameras_scene_dir=$preprocessing_dir trn_nv_gen.camera_ids=$trn_nv_cam_ids