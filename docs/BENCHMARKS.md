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

## Extension, whole decode step on the GPU, 2026-11-21

Same file and prompt, greedy decode of 128 tokens, one sequence, one host round trip per token.

| Path | Generation (t/s) | Per token |
|---|---:|---:|
| GPU decoder, first version (two single-block glue kernels) | 17.0 | 58.8 ms |
| GPU decoder, loop controllers and router spread across the GPU | 20.9 | 47.9 ms |
| llama.cpp HIP, same file (baseline) | 19.5 | 51.3 ms |

Per-token profile of the 20.9 t/s version (each kernel group timed with its own round trip, so
the sum, 49.8 ms, is slightly above the unprofiled 47.9 ms):

| Kernel group | ms per token | Note |
|---|---:|---|
| NVFP4 experts (44 visits) | 15.1 | compute-bound: 4-bit decode plus multiply |
| Input projections (48 visits, about 40 MB each) | 12.0 | about 158 GB/s, near bandwidth |
| Output head (201,024 x 2,304 BF16) | 5.6 | about 165 GB/s |
| Attention output projection (48 x 14 MB) | 5.3 | |
| Glue C: convolution, mix, message, norm, router | 3.0 | was 5.9 |
| Glue A: mixes, messages, norms | 2.8 | was 10.7 |
| BF16 experts (4 visits) | 2.6 | about 195 GB/s |
| Attention over the KV cache | 2.2 | short context |
| Glue B and D | 1.1 | |

What this says about single-stream speed at native precision:

- About 25 ms per token is BF16 weight traffic that already runs near the memory bandwidth of
  the machine. It cannot shrink without changing the stored precision.
- The NVFP4 expert kernel is the largest remaining compute cost. It already processes about
  twice as many weights per second as the BF16 kernels; a better decode might save a few ms.
- A fully tuned single-stream decoder is therefore estimated at roughly 25 tokens/s, not
  dramatically above today's 20.9. Larger gains have to come from doing more per pass over the
  weights: speculative decoding (verifying several drafted tokens in one pass) and batching
  several sessions together. Neither is built yet.

Not covered by these numbers: contexts beyond a few hundred tokens (attention cost grows with
context, and the 1,024-token sliding window wrap is untested on the GPU), prompt processing in
batches (prompts are currently fed one token at a time, about 24 tokens/s), sampling, and
anything through GUFO's serving interface.

## Extension, batched engine, 2026-11-21

One step now runs a batch of rows, each a (session, position). Same file; greedy decode.

### Several sessions decoded together

| Sessions | Total (t/s) | Per session (t/s) |
|---:|---:|---:|
| 1 | 19.4 | 19.4 |
| 4 | 35.3 | 8.8 |
| 8 | 39.0 | 4.9 |
| 16 | 44.4 | 2.8 |

The 8-session figure predates the column-major input change below; the others include it.
One session is slightly slower than the unbatched decoder (20.9) because the kernels are now
general over the batch.

### Prompt processing

| Feeding | Prompt (t/s) |
|---|---:|
| One token per step | 21 |
| 16 rows per step | 49 |
| llama.cpp HIP, same file (baseline) | 346 to 520 |

Prompt processing is far behind llama.cpp. Batching more rows does not fix it by itself, for
the reason the profile below shows.

### Where a 16-row step goes (360 ms)

| Kernel group | ms per step | Note |
|---|---:|---|
| NVFP4 experts | 215 | 13 ms per row: every row decodes its own 8 experts |
| Input projections | 39 | was 63 before reading the inputs column-major |
| BF16 experts | 39 | |
| Output head | 28 | decode only; prompts need it once |
| Output projection | 16 | was 37 |
| Everything else | about 23 | |

With several rows the kernels are no longer limited by memory bandwidth but by how many
multiply-adds per second they achieve: about 400 billion per second in the projections and
about 190 billion in the experts. Both are well below what this GPU can do. The experts are
70% of a batched step and are the next target: rows that select the same expert should share
one decode of its weights, and the decode itself needs to be cheaper.

## Extension, expert-major kernels, 2026-11-21

Each expert's NVFP4 weights are now decoded once per step and applied to every row that selected
that expert, instead of each row decoding its own eight experts. Used from 4 rows up.

These runs use real text: each session's prompt is a different 64-token chunk of wikitext-2
(test), then 48 greedy tokens. The earlier tables used a repeating 5-token prompt, which makes
sessions pick the same experts and flatters any kernel that shares work between rows; that
prompt gave 71.6 tokens/s for 16 sessions here, and is not a fair figure.

### Several sessions decoded together (real text)

| Sessions | Total (t/s) | Per session (t/s) | Without expert-major |
|---:|---:|---:|---:|
| 1 | 18.5 | 18.5 | same kernels |
| 4 | 34.7 | 8.7 | |
| 8 | 46.5 | 5.8 | 37.9 |
| 16 (8 distinct chunks, each used twice) | 62.2 | 3.9 | |

The 16-session row repeats prompts, so it overstates what 16 unrelated sessions would get.

### Prompt processing (real text)

| Feeding | Prompt (t/s) | Without expert-major |
|---|---:|---:|
| 64-token prompt, 16 rows per step | 76 | 49 |
| 500-token prompt, 64 rows per step | 83 | 47 |
| llama.cpp HIP, same file (baseline) | 346 to 520 | |

Output is unchanged: the logits and greedy tokens are identical with and without the new
kernels, and the quality harness gives the same result (perplexity 7.7928, mean KLD 0.0021,
98.19% same top token).

### What was tried and rejected

One GPU thread per output value, with no shared-memory reduction and far fewer thread blocks,
was 2 to 25 times slower than one block per output value (correct output, so it is still in the
code behind `INKLING2_EXPERT_VARIANT=1` and `INKLING2_GEMM_VARIANT=1`). On this GPU adjacent
threads need to read adjacent memory.

### Integer dot products

A micro-benchmark in the shape of the batched matrix kernel (`INKLING2_DOTBENCH=1`):

| Arithmetic | 1 row | 4 rows | 16 rows |
|---|---:|---:|---:|
| F32 multiply-add | 49 | 182 | 420 |
| 8-bit integer dot product (4 values per instruction) | 123 | 552 | 1,137 |

Billions of multiply-adds per second. The integer instruction gives about 2.7 times the float
rate at 16 rows. Using it means quantizing activations to 8 bits inside the expert products,
which is what llama.cpp's fast path does; the weights would stay exact (NVFP4 values are small
integers times a scale). That is not built; it is the next candidate for prompt speed and would
be judged by the quality harness.

## Not measured yet

- Output quality beyond that smoke test (no perplexity or logit comparison yet).
- Anything on GUFO: there is no Inkling runner yet.
- DSpark speculative decoding.
