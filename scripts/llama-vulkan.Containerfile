# llama.cpp (mainline + PR 32704 loop fix) with the Vulkan backend, plus the GGUF conversion tools.
#   podman build -t localhost/llama-inkling:1 -f llama-vulkan.Containerfile .
#   podman run --rm --device /dev/dri --group-add keep-groups --security-opt label=disable \
#     -v /path/to/models:/models:ro localhost/llama-inkling:1 \
#     llama-bench -m /models/Inkling-2-Mini-UD-Q8_K_XL.gguf -ngl 99 -p 512,4096 -n 256 -r 2
FROM registry.fedoraproject.org/fedora:45
RUN dnf -y install git cmake ninja-build gcc-c++ vulkan-headers vulkan-loader-devel glslc mesa-vulkan-drivers vulkan-tools python3 python3-pip libcurl-devel spirv-headers-devel && dnf clean all
WORKDIR /src
RUN git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && \
    git fetch origin pull/32704/head:inkling2-loop-fix && git checkout inkling2-loop-fix && git log --oneline -3
RUN cd llama.cpp && cmake -B build -G Ninja -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && \
    cmake --build build --target llama-cli llama-server llama-bench llama-quantize llama-mtmd-cli -j 24
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt --extra-index-url https://download.pytorch.org/whl/cpu
ENV PATH=/src/llama.cpp/build/bin:/venv/bin:$PATH
