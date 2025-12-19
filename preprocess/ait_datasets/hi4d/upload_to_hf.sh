#!/bin/bash
set -euo pipefail

# hf repo create ludekcizinsky/hi4d --repo-type dataset --private

zip_file_folder="/scratch/izar/cizinsky/ait_datasets/compressed/hi4d"
# zip_files_to_upload=(
   # "pair00_1.tar.gz"
    # "pair00_2.tar.gz"
    # "pair01.tar.gz"
    # "pair15_1.tar.gz"
    # "pair15_2.tar.gz"
    # "pair16.tar.gz"
    # "pair17_1.tar.gz"
    # "pair17_2.tar.gz"
    # "pair19_1.tar.gz"
    # "pair19_2.tar.gz"
# )
zip_files_to_upload=(
    "pair02_1.tar.gz"
    "pair02_2.tar.gz"
    "pair09.tar.gz"
    "pair10.tar.gz"
    "pair12.tar.gz"
    "pair13_1.tar.gz"
    "pair13_2.tar.gz"
    "pair14.tar.gz"
    "pair18_1.tar.gz"
    "pair18_2.tar.gz"
    "pair21_1.tar.gz"
    "pair21_2.tar.gz"
    "pair22.tar.gz"
    "pair23_1.tar.gz"
    "pair23_2.tar.gz"
    "pair27_1.tar.gz"
    "pair27_2.tar.gz"
    "pair28.tar.gz"
    "pair32_1.tar.gz"
    "pair32_2.tar.gz"
    "pair37_1.tar.gz"
    "pair37_2.tar.gz"
)

for zip_file in "${zip_files_to_upload[@]}"; do
  echo "Uploading $zip_file to Hugging Face..."
  hf upload --repo-type dataset ludekcizinsky/hi4d "$zip_file_folder/$zip_file" "$zip_file"
done
