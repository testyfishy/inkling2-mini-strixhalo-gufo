#!/bin/bash
# Build llama.cpp (mainline + PR 32704 loop fix) with the HIP (ROCm) backend, using GUFO's
# development toolchain image so the ROCm version matches what GUFO itself is built with.
#
#   WORK=~/inkling-bench/llama-hip ./llama-hip-build.sh
#
# Output: $WORK/llama.cpp/build/bin
set -e
WORK=${WORK:-$HOME/inkling-bench/llama-hip}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK" && cd "$WORK"
if [ ! -d llama.cpp ]; then
  git clone -q https://github.com/ggml-org/llama.cpp
  (cd llama.cpp && git fetch -q origin pull/32704/head:inkling2-loop-fix && git checkout -q inkling2-loop-fix)
fi
cp "$HERE/llama-hip-inner.sh" "$WORK/llama-hip-inner.sh"
# --memory keeps the compile from squeezing whatever else runs on the box.
podman run --rm --memory=6g --userns=keep-id:uid=1000,gid=1000 --security-opt label=disable \
  -v "$WORK":/workspace -w /workspace/llama.cpp \
  ghcr.io/gufo-org/toolboxes/gufo-dev:edge bash /workspace/llama-hip-inner.sh
