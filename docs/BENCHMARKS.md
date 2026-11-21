# Benchmarks

## Machine

- AMD Ryzen AI MAX+ 395, Radeon 8060S (`gfx1151`), 125 GiB unified memory
- Fedora 45, kernel 7.2.15
- Vulkan: Mesa RADV, API 1.4.354
- HIP: ROCm 10.1 from the `ghcr.io/gufo-org/toolboxes/gufo-dev:edge` toolchain image

## llama.cpp baseline, 2026-11-20/21

- llama.cpp mainline with [#32704](https://github.com/ggml-org/llama.cpp/pull/32704) applied
  (commit `2a28df0f9`).
- `llama-bench -ngl 99 -p 512,4096 -n 256 -r 2`. Synthetic prompts, no speculative drafter.
- Nothing else was using the GPU during the runs.

| Weights | File size | Backend | pp512 (t/s) | pp4096 (t/s) | tg256 (t/s) |
|---|---:|---|---:|---:|---:|
| Native precision (NVFP4 experts + BF16) | 21.29 GiB | HIP | 519.67 ± 0.66 | 345.66 ± 0.15 | 19.50 ± 0.05 |
| Native precision (NVFP4 experts + BF16) | 21.29 GiB | Vulkan | 344.10 ± 3.24 | 296.40 ± 2.42 | 15.30 ± 0.00 |
| UD-Q4_K_XL | 16.41 GiB | HIP | 508.55 ± 4.46 | 342.97 ± 0.22 | 24.29 ± 0.04 |
| UD-Q4_K_XL | 16.41 GiB | Vulkan | 420.91 ± 5.75 | 350.67 ± 0.38 | 21.68 ± 0.01 |
| BF16 dequant of the whole checkpoint | 48.63 GiB | Vulkan | 206.51 ± 4.36 | 193.91 ± 0.07 | 4.56 ± 0.01 |

Notes:

- "Native precision" is the mainline-layout `UD-Q8_K_XL` file from
  [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)
  (`Mainline_Support/`). A tensor dump shows 40 expert tensors stored as NVFP4 (10.7 GiB) and the
  rest as BF16/F32, which matches the released checkpoint's precision.
- The 48.6 GiB row is `convert_hf_to_gguf.py --outtype auto` on the released checkpoint, which
  expands the NVFP4 experts to BF16. RADV reports no BF16 support, so that row says more about the
  backend than about the model.
- `llama-bench` prints misleading file-type labels for these files (it shows the 21.29 GiB file as
  "BF16"); the tensor dump is the source of truth.

## Reference point: what GUFO does for another model on this machine

Qwen3.8 Flash-Next on GUFO 0.34.0 with MTP drafting, measured over HTTP on the same box:
1,374 t/s at 700 prompt tokens, 1,658 t/s at 5,572 prompt tokens, 34-38 t/s generation.
It is a different model, measured a different way; it only shows the gap between a generic
backend and a model-specific GUFO runner.

## Generation check

`llama-cli --temp 0 --jinja -st` on the native-precision file (HIP) answers two short prompts
correctly and coherently, with reasoning and answer separated, at 19.4 t/s. This is a smoke test,
not a quality evaluation.

## Extension, first GPU decode, 2026-11-21

Native-precision file, greedy decode of 96 tokens after a 5-token prompt, one sequence, measured
inside `gufo prompt` (development driver, fixed token ids, no tokenizer).

| Path | Generation (t/s) |
|---|---:|
| CPU reference, 16 threads | 2.2 |
| HIP kernels for the matrix products, rest on CPU | 13.5 |
| llama.cpp HIP, same file (baseline) | 19.5 |

Where the 74 ms per token goes in the HIP path: about 29 ms in the projection calls, 21 ms in
the expert calls, 24 ms in CPU-side work. Changing the GEMV thread layout (32 to 256 threads per
row, or several rows per block) moves the total by under 7%, so the kernels are not the limit:
each layer visit makes three CPU-GPU round trips (144 per token), each with staging copies and a
stream synchronization. At roughly 54 MB of projection weights and 36 MB of expert weights read
per visit, the kernel calls average about 90-100 GB/s including that overhead.

This is a correctness milestone, not a speed result. The next step moves the remaining
per-visit operations onto the GPU so a token needs one round trip.

### Raw kernel throughput

The same kernels launched 50 times back to back with no round trips in between
(`INKLING2_GPU_BENCH=1`):

| Kernel | Data per pass | Time per pass | Throughput |
|---|---:|---:|---:|
| BF16 projection, output head (201,024 x 2,304) | 926 MB | 4.9 ms | 189 GB/s |
| BF16 experts, 8 selected (layers 20-23) | 127 MB | 0.64 ms | 199 GB/s |
| NVFP4 experts, 8 selected (layers 0-19) | 35.8 MB | 0.33 ms | 108 GB/s |
| BF16 projection, one attention matrix | 14.2 MB | 0.042 ms | not meaningful (repeats hit cache) |

The BF16 kernels already run at about the memory bandwidth of this machine. The NVFP4 expert
kernel is compute-bound at roughly half of it and has room to improve.

Adding these up for one token (48 visits of about 54 MB of projections, 44 NVFP4 and 4 BF16
expert calls, one output head) gives about 36 ms of kernel time, or roughly 28 tokens/s if the
per-call overhead and CPU work were removed entirely, and about 34 tokens/s if the NVFP4 kernel
also reached bandwidth. That is an estimate from kernel timings, not a measured decode speed; it
is the ceiling for single-stream decode at native precision without speculative decoding.

## Not measured yet

- Output quality beyond that smoke test (no perplexity or logit comparison yet).
- Anything on GUFO: there is no Inkling runner yet.
- DSpark speculative decoding.
