#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  $0 <scale> <fps> <input_video> <output_dir>
  $0 <fps> <input_video> <output_dir>   # keeps original size

  scale (optional):
    - Single number (e.g., 512) => width=512, height auto
    - Or ffmpeg scale expression (e.g., 512:-1, 640:360, 640x360)
    - If omitted, original size is used.

  fps:
    - Frames per second to extract (float allowed).
      Examples: 1, 0.5, 12.5

Examples:
  $0 512 2 input.mp4 frames/
  $0 640:360 0.5 "my video.mov" out_frames/
  $0 1 input.mp4 frames_original_size/

Output frames are named:
  00000.jpg, 00001.jpg, 00002.jpg, ...
EOF
  exit 1
}

prompt_overwrite_dir() {
  local dir="$1"
  echo "Output directory already exists: $dir"
  while true; do
    read -r -p "Remove it and recreate? [y/N]: " ans
    ans="${ans:-N}"
    case "$ans" in
      [yY]|[yY][eE][sS])
        rm -rf "$dir"
        mkdir -p "$dir"
        echo "Recreated directory: $dir"
        return 0
        ;;
      [nN]|[nN][oO])
        echo "Cancelled."
        exit 0
        ;;
      *)
        echo "Please answer y or n."
        ;;
    esac
  done
}

# Parse args
if [[ $# -eq 4 ]]; then
  SCALE_ARG="$1"
  FPS_ARG="$2"
  INPUT_VIDEO="$3"
  OUTPUT_DIR="$4"
  USE_SCALE=1
elif [[ $# -eq 3 ]]; then
  FPS_ARG="$1"
  INPUT_VIDEO="$2"
  OUTPUT_DIR="$3"
  USE_SCALE=0
else
  usage
fi

# Validate FPS is a positive number
if ! [[ "$FPS_ARG" =~ ^[0-9]*\.?[0-9]+$ ]] || \
   [[ "$(awk "BEGIN{print ($FPS_ARG<=0)}")" == "1" ]]; then
  echo "Error: fps must be a positive number." >&2
  exit 1
fi

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Error: input_video not found: $INPUT_VIDEO" >&2
  exit 1
fi

# Handle output directory existence
if [[ -d "$OUTPUT_DIR" ]]; then
  prompt_overwrite_dir "$OUTPUT_DIR"
else
  mkdir -p "$OUTPUT_DIR"
fi

# Build video filter
VF_FILTER="fps=${FPS_ARG}"

if [[ "$USE_SCALE" -eq 1 ]]; then
  # Build scale filter:
  # - If SCALE_ARG contains ":" or "x", treat as full scale expression.
  # - Else treat as width only, height auto (-1).
  if [[ "$SCALE_ARG" == *":"* ]] || [[ "$SCALE_ARG" == *"x"* ]]; then
    # Convert 640x360 -> 640:360 for ffmpeg
    SCALE_EXPR="${SCALE_ARG//x/:}"
  else
    SCALE_EXPR="${SCALE_ARG}:-1"
  fi
  VF_FILTER="${VF_FILTER},scale=${SCALE_EXPR}"
fi

# Extract frames at given FPS, name 00000.jpg, 00001.jpg, ...
ffmpeg -hide_banner -loglevel error -y \
  -i "$INPUT_VIDEO" \
  -vf "$VF_FILTER" \
  -start_number 0 \
  "${OUTPUT_DIR%/}/%05d.jpg"

echo "Done. Extracted frames at ${FPS_ARG} fps to: $OUTPUT_DIR"
if [[ "$USE_SCALE" -eq 0 ]]; then
  echo "Output kept original size."
fi
