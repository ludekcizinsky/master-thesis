#!/usr/bin/env bash

set -euo pipefail

print_usage() {
  cat <<EOF
Usage: $(basename "$0") <videos_root>

Combine per-method videos for each scene into side-by-side comparison videos.

Arguments:
  videos_root  Root directory containing subdirectories for each method.
               Example: /scratch/izar/cizinsky/thesis/evaluation/videos/renders/hi4d
EOF
}

if [[ $# -ne 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  print_usage
  exit 1
fi

VIDEOS_ROOT=$1

if [[ ! -d "$VIDEOS_ROOT" ]]; then
  echo "Error: '$VIDEOS_ROOT' is not a directory." >&2
  exit 1
fi

COMPARISON_DIR="$VIDEOS_ROOT/comparison"
mkdir -p "$COMPARISON_DIR"

load_ffmpeg() {
  if ! command -v ffmpeg >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
      module load gcc ffmpeg || {
        echo "Failed to load ffmpeg module." >&2
        exit 1
      }
    else
      echo "ffmpeg not found and environment modules unavailable." >&2
      exit 1
    fi
  fi
}

gather_scenes() {
  local scenes=()
  for method_dir in "$VIDEOS_ROOT"/*; do
    [[ -d "$method_dir" ]] || continue
    [[ "$method_dir" == "$COMPARISON_DIR" ]] && continue
    for video in "$method_dir"/*.mp4; do
      [[ -f "$video" ]] || continue
      scenes+=("$(basename "$video")")
    done
  done

  printf "%s\n" "${scenes[@]}" | sort -u
}

create_comparison() {
  local scene=$1
  shift
  local videos=("$@")

  local input_args=()
  for video in "${videos[@]}"; do
    input_args+=("-i" "$video")
  done

  local filter_parts=""
  local labels=()
  local idx=0
  for _ in "${videos[@]}"; do
    labels+=("v$idx")
    filter_parts+="[${idx}:v]scale=trunc(iw/2)*2:trunc(ih/2)*2[v${idx}];"
    idx=$((idx + 1))
  done

  local stacked_label="stacked"
  local hstack_inputs=""
  for label in "${labels[@]}"; do
    hstack_inputs+="[${label}]"
  done
  filter_parts+="${hstack_inputs}hstack=inputs=${#videos[@]}[${stacked_label}]"

  local output="$COMPARISON_DIR/${scene}.mp4"

  ffmpeg "${input_args[@]}" -filter_complex "$filter_parts" -map "[${stacked_label}]" -c:v libx264 -crf 18 -preset veryfast -y "$output"
}

main() {
  load_ffmpeg

  local scenes
  readarray -t scenes < <(gather_scenes)

  for scene_file in "${scenes[@]}"; do
    local videos=()
    for method_dir in "$VIDEOS_ROOT"/*; do
      [[ -d "$method_dir" ]] || continue
      [[ "$method_dir" == "$COMPARISON_DIR" ]] && continue
      local video_path="$method_dir/$scene_file"
      [[ -f "$video_path" ]] && videos+=("$video_path")
    done

    if [[ ${#videos[@]} -lt 2 ]]; then
      echo "Skipping $scene_file: need at least two videos, found ${#videos[@]}." >&2
      continue
    fi

    local scene_name="${scene_file%.mp4}"
    echo "Creating comparison for $scene_name (${#videos[@]} methods)"
    create_comparison "$scene_name" "${videos[@]}"
  done
}

main "$@"
