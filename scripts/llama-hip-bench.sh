#!/bin/bash
# Benchmark a GGUF with the HIP build from llama-hip-build.sh.
#
#   MODELS=/path/to/models ./llama-hip-bench.sh Inkling-2-Mini-UD-Q8_K_XL.gguf
#
# The model needs about its file size in free memory, and the numbers are only meaningful when
# nothing else is using the GPU.
set -e
WORK=${WORK:-$HOME/inkling-bench/llama-hip}
MODELS=${MODELS:-$HOME/models}
podman run --rm --network=none --userns=keep-id:uid=1000,gid=1000 --security-opt label=disable \
  --device /dev/kfd --device /dev/dri --group-add keep-groups --ulimit memlock=-1 \
  -v "$MODELS":/models:ro -v "$WORK":/workspace -w /workspace/llama.cpp/build/bin \
  -e LD_LIBRARY_PATH=/workspace/llama.cpp/build/bin \
  ghcr.io/gufo-org/toolboxes/gufo-dev:edge \
  ./llama-bench -m "/models/$1" -ngl 99 -p 512,4096 -n 256 -r 2
