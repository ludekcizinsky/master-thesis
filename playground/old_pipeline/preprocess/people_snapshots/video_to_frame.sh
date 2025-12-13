#!/usr/bin/env bash
set -euo pipefail

module load gcc ffmpeg

usage() {
  echo "Usage: $0 /path/to/video.mp4 fps" >&2
  exit 1
}

[[ $# -eq 2 ]] || usage

video_path=$1
fps=$2

if [[ ! -f "$video_path" ]]; then
  echo "Video not found: $video_path" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but not installed or not in PATH" >&2
  exit 1
fi

video_dir=$(dirname "$video_path")
video_base=$(basename "$video_path")
stem=${video_base%.*}
output_dir="$video_dir/${stem}_frames"

rm -rf "$output_dir"
mkdir -p "$output_dir"

ffmpeg -hide_banner -loglevel info -i "$video_path" -vf "fps=$fps" "$output_dir/frame_%05d.png"
echo "Extracted frames to $output_dir"
total_frames=$(ls "$output_dir"/frame_*.png | wc -l)
echo "Total frames extracted: $total_frames"