#!/usr/bin/env sh
# Replay extraction -> classifier -> selection for every video in a folder
# INSIDE the influ-ocr image (its Tesseract and model, no network), and compare
# the frames it would send with tools/baselines/final_test2_frames_sent.json.
#
#   tools/replay_in_docker.sh IMAGE VIDEOS_DIR OUT_DIR [--mount-tree]
#
# --mount-tree bind-mounts the working tree's app over /app, so the image's
# dependencies are exercised with the CURRENT code (attributes a difference to
# code vs image). Output: OUT_DIR/replay.json (+ _processing/ work files).
set -eu
IMAGE=$1; VIDEOS=$(cd "$2" && pwd); OUT=$3; shift 3
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
MOUNT_TREE=""
for arg in "$@"; do [ "$arg" = "--mount-tree" ] && MOUNT_TREE="-v $REPO/instagram_analyzer_app:/app:ro"; done
# shellcheck disable=SC2086
docker run --rm --entrypoint python \
  -e SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  -e LOG_DIR=/work/_logs -e PROCESSING_DIR=/work/_processing \
  -v "$VIDEOS":/videos:ro -v "$OUT":/work -v "$HERE":/app/tools:ro $MOUNT_TREE \
  "$IMAGE" /app/tools/check_outputs.py replay --videos /videos --out /work
