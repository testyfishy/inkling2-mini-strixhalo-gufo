#!/bin/bash
# Downloads the benchmark harnesses and datasets into $EVALS (default ~/inkling-bench/evals)
# on the test machine and builds the container they run in. Run once; rerunning only fetches
# what is missing. Nothing here talks to a model.
set -u
EVALS=${EVALS:-$HOME/inkling-bench/evals}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$EVALS"/{src,hf,data,results}
cd "$EVALS/src"
clone() {  # url, directory
  [ -d "$2/.git" ] || git clone --depth 1 "$1" "$2" || echo "CLONE FAILED: $1"
  (cd "$2" 2>/dev/null && echo "$2 $(git rev-parse --short HEAD)")
}
clone https://github.com/TIGER-AI-Lab/MMLU-Pro.git mmlu-pro
clone https://github.com/openai/simple-evals.git simple-evals
clone https://github.com/LiveCodeBench/LiveCodeBench.git livecodebench
clone https://github.com/THUDM/LongBench.git longbench
clone https://github.com/ShishirPatil/gorilla.git gorilla
clone https://github.com/Aider-AI/aider.git aider
clone https://github.com/Aider-AI/polyglot-benchmark.git polyglot-benchmark

podman image exists localhost/inkling-evals:1 || \
  podman build -t localhost/inkling-evals:1 -f "$HERE/Containerfile" "$HERE"

# Datasets, into the Hugging Face cache under $EVALS/hf, and GPQA's public CSV.
podman run --rm -i --security-opt label=disable -v "$EVALS":/evals localhost/inkling-evals:1 python - <<'EOF'
from huggingface_hub import snapshot_download
import urllib.request, os
for repo in ("TIGER-Lab/MMLU-Pro", "THUDM/LongBench-v2", "livecodebench/code_generation_lite"):
    try:
        path = snapshot_download(repo, repo_type="dataset")
        print("dataset", repo, "->", path)
    except Exception as e:
        print("DATASET FAILED", repo, repr(e)[:300])
url = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
dst = "/evals/data/gpqa_diamond.csv"
if not os.path.exists(dst):
    try:
        urllib.request.urlretrieve(url, dst)
    except Exception as e:
        print("GPQA FAILED", repr(e)[:300])
print("gpqa", os.path.exists(dst) and os.path.getsize(dst))
EOF
du -sh "$EVALS"/src "$EVALS"/hf "$EVALS"/data 2>/dev/null
