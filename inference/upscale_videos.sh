#!/bin/bash
# Upscale every generated video (720x480) to 1080x720 with Lanczos, keeping
# frame count and fps. Originals are left untouched; outputs keep the same
# file names in a separate directory. Called at the end of
# 2_run_inference_all.sh, and can also be run on its own.
#
# Usage: ./upscale_videos.sh [src-dir] [dst-dir]
#   e.g., ./upscale_videos.sh                              # results -> results_upscaled
#         ./upscale_videos.sh results results_upscaled

set -euo pipefail

SRC_DIR=${1:-results}
DST_DIR=${2:-results_upscaled}
WIDTH=1080
HEIGHT=720
# Number of ffmpeg processes run at once (CPU only)
PARALLEL=8

if [ ! -d "$SRC_DIR" ]; then
    echo "Source directory not found: $SRC_DIR"
    exit 1
fi
mkdir -p "$DST_DIR"

TOTAL=$(find "$SRC_DIR" -maxdepth 1 -name '*.mp4' | wc -l)
echo "Upscaling $TOTAL videos: $SRC_DIR -> $DST_DIR (${WIDTH}x${HEIGHT}, lanczos)"
START=$(date +%s)

# crf 18 keeps the re-encode visually lossless; yuv420p for player support
find "$SRC_DIR" -maxdepth 1 -name '*.mp4' -print0 \
    | xargs -0 -r -P "$PARALLEL" -I{} sh -c '
        ffmpeg -nostdin -loglevel error -y -i "$1" \
            -vf "scale=$3:$4:flags=lanczos" \
            -c:v libx264 -crf 18 -pix_fmt yuv420p \
            "$2/${1##*/}"' _ {} "$DST_DIR" "$WIDTH" "$HEIGHT"

DONE=$(find "$DST_DIR" -maxdepth 1 -name '*.mp4' | wc -l)
ELAPSED=$(( $(date +%s) - START ))
printf 'Upscale time: %02d:%02d:%02d (%d videos in %s)\n' \
    $((ELAPSED / 3600)) $((ELAPSED % 3600 / 60)) $((ELAPSED % 60)) \
    "$DONE" "$DST_DIR"
