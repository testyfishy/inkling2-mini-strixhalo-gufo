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

## Not measured yet

- Output correctness of the native-precision file beyond the benchmark run (a greedy generation
  check is pending).
- Anything on GUFO: there is no Inkling runner yet.
- DSpark speculative decoding.
