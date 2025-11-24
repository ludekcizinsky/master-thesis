#!/bin/bash

# set your paths
in_dir=$1
out_dir=$2
# optional third arg to control width (defaults to 1280px for higher-res gifs)
width=${3:-1280}
mkdir -p "$out_dir"

for f in "$in_dir"/*.mp4; do
  [ -e "$f" ] || continue                     # skip if no matches
  base=$(basename "${f%.*}")
  ffmpeg -y -i "$f" -vf "fps=12,scale=${width}:-1:flags=lanczos" "$out_dir/$base.gif"
done
