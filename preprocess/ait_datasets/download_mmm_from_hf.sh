#!/bin/bash
set -euo pipefail

# Directory on the local machine where the downloaded archives should be stored
compressed_dir="/scratch/izar/cizinsky/ait_datasets/compressed/mmm"
full_dir="/scratch/izar/cizinsky/ait_datasets/full/mmm"
mkdir -p "$compressed_dir"
mkdir -p "$full_dir"

# Files available in the HF dataset repo
zip_files_to_download=(
    "cheer.zip"
    "dance.zip"
    "hug.zip"
    "lift.zip"
    "selfie.zip"
    "walkdance.zip"
)

export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
for zip_file in "${zip_files_to_download[@]}"; do
  echo "Downloading $zip_file from Hugging Face..."
  hf download --repo-type dataset ludekcizinsky/mmm "$zip_file" --local-dir "$compressed_dir"

  echo "Extracting $zip_file into $full_dir..."
  unzip "$compressed_dir/$zip_file" -d "$full_dir" 
done