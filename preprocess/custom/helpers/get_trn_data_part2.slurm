#!/bin/bash
#SBATCH --job-name=preprocess_part2
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=master

# navigate to the directory containing the preprocessing script
cd /home/cizinsky/master-thesis

path_to_script=preprocess/custom/helpers/get_trn_data_part2.sh

seq_name="${SEQ_NAME:-}"
gt_root_dir="${ROOT_GT_DIR_PATH:-${GT_ROOT_DIR:-}}"

if [[ -z "$seq_name" || -z "$gt_root_dir" ]]; then
    echo "Missing SEQ_NAME or ROOT_GT_DIR_PATH. Provide them via sbatch --export."
    exit 1
fi

bash "$path_to_script" "$seq_name" "$gt_root_dir"
