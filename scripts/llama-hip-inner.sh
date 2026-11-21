#!/bin/bash
# Runs inside ghcr.io/gufo-org/toolboxes/gufo-dev:edge, cwd = llama.cpp checkout.
# The image is Nix-built: compilers are amdclang/amdclang++, and CMake packages live in the store.
set -e
P=""
for d in /nix/store/*; do
  if [ -d "$d/lib/cmake" ] || [ -d "$d/lib64/cmake" ] || [ -d "$d/share/cmake" ]; then P="$P$d;"; fi
done
# ar, ranlib and friends are only in the ROCm toolchain's bin directory.
T=$(ls -d /nix/store/*-rocm-toolchain)/bin
export PATH=$T:$PATH
export CC=amdclang CXX=amdclang++ HIPCXX=amdclang++
# -D__MM_MALLOC_H: the toolchain's mm_malloc.h redeclares posix_memalign without noexcept and
# fails against this glibc; llama.cpp does not use _mm_malloc, so the header is skipped.
cmake -B build -G Ninja -DCMAKE_PREFIX_PATH="$P" -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-D__MM_MALLOC_H -DCMAKE_HIP_FLAGS=-D__MM_MALLOC_H \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-bench llama-server llama-cli -j 3
echo HIP_BUILD_DONE
