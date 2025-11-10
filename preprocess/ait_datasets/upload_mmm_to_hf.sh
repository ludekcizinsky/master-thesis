#!/bin/bash
set -euo pipefail

hf repo create ludekcizinsky/mmm --repo-type dataset --private

zip_file_folder="/scratch/izar/cizinsky/ait_datasets/compressed/mmm"
zip_files_to_upload=(
    "cheer.zip"
    "dance.zip"
    "hug.zip"
    "lift.zip"
    "selfie.zip"
    "walkdance.zip"
)

for zip_file in "${zip_files_to_upload[@]}"; do
  echo "Uploading $zip_file to Hugging Face..."
  hf upload --repo-type dataset ludekcizinsky/mmm "$zip_file_folder/$zip_file" "$zip_file"
done
