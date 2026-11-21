#!/bin/bash
# Quality check of the extension against llama.cpp on the same GGUF, the way
# quantization quality is usually reported: perplexity, mean KL divergence and
# top-token agreement on wikitext-2 (test split), second half of each chunk.
#
#   MODEL=/models/Inkling-2-Mini-UD-Q8_K_XL.gguf CHUNKS=8 CTX=512 ./quality.sh
#
# NGL=0 LLAMA_EXTRA="-ctk f32 -ctv f32" makes the reference llama.cpp's CPU path
# with F32 caches instead of its GPU path.
#
# Needs: the HIP build of llama.cpp from llama-hip-build.sh with the
# llama-perplexity target, wikitext-2 in $DATA/wikitext-2-raw/, the extension
# installed in the `inkling-gufo-ext` volume, and nothing else using the GPU.
set -e
WORK=${WORK:-$HOME/inkling-bench/llama-hip}
DATA=${DATA:-$HOME/inkling-bench/data}
OUT=${OUT:-$HOME/inkling-bench/out}
MODELS=${MODELS:-$HOME/models}
MODEL=${MODEL:?set MODEL to the GGUF path under /models}
CHUNKS=${CHUNKS:-8}
CTX=${CTX:-512}
BATCH=${BATCH:-16}
GPU="--device /dev/kfd --device /dev/dri --group-add keep-groups --ulimit memlock=-1"
COMMON="--rm --network=none --userns=keep-id:uid=1000,gid=1000 --security-opt label=disable $GPU -v $MODELS:/models:ro"

if [ -z "$SKIP_BASE" ]; then
  podman run $COMMON -v "$WORK":/workspace -v "$DATA":/data:ro -v "$OUT":/out \
    -w /workspace/llama.cpp/build/bin -e LD_LIBRARY_PATH=/workspace/llama.cpp/build/bin \
    ghcr.io/gufo-org/toolboxes/gufo-dev:edge \
    ./llama-perplexity -m "$MODEL" -f /data/wikitext-2-raw/wiki.test.raw -c "$CTX" --chunks "$CHUNKS" \
    -ngl "${NGL:-99}" $LLAMA_EXTRA --kl-divergence-base /out/kld-base.bin 2>&1 | grep -E "Final estimate|perplexity:|error" || true
fi

podman run $COMMON -v inkling-gufo-ext:/home/gufo/.local/share/gufo/extensions -v "$OUT":/out:ro \
  -e INKLING2_KLD=/out/kld-base.bin -e INKLING2_KLD_CHUNKS="$CHUNKS" -e INKLING2_BATCH="$BATCH" \
  ghcr.io/gufo-org/toolboxes/gufo-runtime:edge gufo prompt --model "$MODEL" --prompt x
