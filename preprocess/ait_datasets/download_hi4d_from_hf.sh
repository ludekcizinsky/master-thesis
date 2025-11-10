#!/bin/bash
set -euo pipefail

# Directories for compressed downloads and extracted contents
compressed_dir="/scratch/izar/cizinsky/ait_datasets/compressed/hi4d"
full_dir="/scratch/izar/cizinsky/ait_datasets/full/hi4d"
mkdir -p "$compressed_dir"
mkdir -p "$full_dir"

# Files available in the HF dataset repo
zip_files_to_download=(
    "pair00_1.tar.gz"
    "pair00_2.tar.gz"
    "pair01.tar.gz"
    "pair15_1.tar.gz"
    "pair15_2.tar.gz"
    "pair16.tar.gz"
    "pair17_1.tar.gz"
    "pair17_2.tar.gz"
    "pair19_1.tar.gz"
    "pair19_2.tar.gz"
)

export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
for zip_file in "${zip_files_to_download[@]}"; do
#   echo "Downloading $zip_file from Hugging Face..."
  # hf download --repo-type dataset ludekcizinsky/hi4d "$zip_file" --local-dir "$compressed_dir"

  # extract filename from zip_file
  base_filename=$(basename "$zip_file" .tar.gz)
  full_dir_subfolder="$full_dir/$base_filename"
  mkdir -p "$full_dir_subfolder"
  echo "Extracting $zip_file into $full_dir..."
  tar -xzvf "$compressed_dir/$zip_file" -C "$full_dir_subfolder"
done
