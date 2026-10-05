#!/usr/bin/env bash
# 픽스처 .mov → 합성 카메라용 y4m (1280x720, 30fps). 300~700MB라 저장소에 넣지 않는다.
set -euo pipefail
SRC=${CLIP_SRC:-$HOME/Projects/aisw}; OUT=${E2E_OUT:-$(dirname "$0")}/clips; mkdir -p "$OUT"
for c in glass put-airpod increase_temp; do
  ffmpeg -v error -y -i "$SRC/$c.mov" -vf "scale=1280:-2,crop=1280:720,fps=30" -pix_fmt yuv420p "$OUT/$c.y4m" && ls -la "$OUT/$c.y4m"
done
