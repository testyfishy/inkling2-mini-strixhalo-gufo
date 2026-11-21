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

### BF16 expert layers, and a second instruction

The four BF16 expert layers (20 to 23) got the same expert-major treatment. In a 64-row prompt
step they drop from 152 ms to 28 ms, with identical logits.

| | Before | After |
|---|---:|---:|
| 448-token prompt, 64 rows per step | 90 t/s | 109 t/s |
| 8 sessions on real text, total | 46.5 t/s | 49.0 t/s |

Where a 64-row prompt step goes now (about 590 ms, 9.2 ms per token):

| Kernel group | ms per step |
|---|---:|
| NVFP4 experts | 271 |
| Input projections | 143 |
| Attention (context up to 448) | 65 |
| Output projection | 49 |
| BF16 experts | 28 |
| Glue | about 30 |

The GPU's two-way BF16 dot-product instruction (`v_dot2_f32_bf16`, which reads two BF16
weights per instruction) was benchmarked for the projections: 564 billion multiply-adds per
second at 16 rows against 420 for the F32 kernel, about 1.3 times. That is not enough to
justify rounding activations to BF16, so it is not used. The 8-bit integer instruction
(2.7 times) remains the candidate, for the NVFP4 experts.

### 8-bit expert products

With the NVFP4 expert products on 8-bit integer dot products (on by default from here; see
[QUALITY.md](QUALITY.md) for the quality comparison):

| | Float products | 8-bit products |
|---|---:|---:|
| 448-token prompt, 64 rows per step | 109.5 t/s | 129.8 t/s |
| 8 sessions on real text, total | 49.0 t/s | 56.2 t/s |
| NVFP4 experts in a 64-row prompt step | 271 ms | 179 ms |

The gain is 1.5 times on the expert kernels, not the 2.7 times of the bare instruction: decoding
the 4-bit weights is now the larger part of that kernel.

### Where this stands against the target

The target is the speed GUFO gets from Qwen3.8 Flash-Next on this machine: about 1,400 to 1,650
tokens/s on prompts and 34 to 38 tokens/s generating (with MTP drafting).

| | This extension | Target | llama.cpp, same model |
|---|---:|---:|---:|
| Prompt | 130 | 1,400 to 1,650 | 346 to 520 |
| Generation, one session | 18.5 to 20.9 | 34 to 38 | 19.5 |

Generation is within reach of the target if speculative decoding with the model's drafter works
as it does for Qwen; that is not built. Prompt speed is an order of magnitude short. GUFO's
kernels for Qwen sustain several times the multiply-add rate of the kernels here (they keep
tiles of weights and inputs in fast local memory and use the integer and half-float
instructions throughout), so the remaining gap is kernel engineering of that kind, not a limit
of the machine.

## Context depth, 2026-11-21

The target use is sessions holding 100,000 tokens or more, so speed has to be measured there.

How the depth numbers were taken: a session is given a real 64-token prompt, then its
full-attention caches are filled to the stated depth by repeating the keys and values it has
(`INKLING2_FAKE_CONTEXT`). The tokens generated afterwards are meaningless, but the work of
attending over that much context is real. Correctness at depth is checked separately, against
llama.cpp on real text, and so far only to 4,096 tokens (see QUALITY.md).

### This extension

| Context depth | Generation, one session (t/s) | Prompt, 64 rows per step (t/s) |
|---:|---:|---:|
| short | 25.0 | 145 |
| 4,000 | 21.3 | 115 |
| 16,000 | 19.6 | 78 |
| 50,000 | 16.4 | |
| 100,000 | 13.3 | 27 |

Before this step, contexts past 1,024 positions used the original attention kernel and
generation at 100,000 tokens ran at 2.0 tokens/s. Two changes fixed that: a kernel that cuts the
context into chunks of 1,024 positions and reads each chunk's keys and values once for the three
query heads that share them, and a cache layout for the full-attention layers in which a block
of threads reads contiguous memory. Attention at 100,000 tokens now costs 36.7 ms per generated
token, which is about what reading that cache at this machine's memory bandwidth costs.

### Qwen3.8 Flash-Next on GUFO, same machine, same depths

Measured over HTTP against the production server, one request per depth, MTP drafting on:

| Prompt tokens | Prompt (t/s) | Generation (t/s) |
|---:|---:|---:|
| 4,002 | 1,662 | 33.0 |
| 15,837 | 1,706 | 35.8 |
| 98,883 | 1,549 | 40.4 |

Qwen's speed is flat with depth. Inkling 2 Mini's is not, and part of that is the model rather
than this code:

- Inkling 2 Mini has 8 full-attention layer visits per token, each over every earlier position
  with 8 key/value heads of 128 dimensions. With the cache in F32, generating one token at
  100,000 tokens of context reads about 6.4 GB of keys and values; at F16 it would be 3.2 GB.
  At this machine's bandwidth that is roughly 32 ms or 16 ms per token before anything else, so
  one-token-at-a-time generation at that depth cannot reach Qwen's 40 tokens/s on this hardware.
  Speculative decoding can share one cache read among several tokens and may recover part of it.
- Prompt processing at depth costs about 49,000 multiply-adds per earlier token for each new
  token (4.9 billion at 100,000). Rows of one prompt step can share the cache read, which this
  code does not do yet (each row reads it again, hence 27 tokens/s), but the arithmetic remains,
  so matching 1,549 tokens/s at that depth needs kernels several times faster than these.

Memory at depth: the full-attention caches take about 65 KB per token per session in F32
(6.6 GB at 100,000 tokens), next to about 21 GB of weights.

## Matrix-instruction kernels, 2026-11-21

The GPU has a 16x16x16 matrix-multiply instruction for FP16 inputs with FP32 sums. GUFO's own
kernels for other models are built on it; from this step the projections and the experts use it
in steps of several rows. A block copies a slice of the weights and of the inputs into shared
memory (converting NVFP4 and BF16 weights to FP16 on the way, which is exact) and its waves
multiply out of shared memory, so each value is read from main memory once per block.

One step of prompt processing, per batch size (512 tokens of real text, short context), in ms:

| Rows per step | 16 before | 16 | 64 before | 64 | 256 before | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Input projections | 36.1 | 12.2 | 139.8 | 12.9 | 3,831 | 33.0 | 60.7 |
| Output projection | 15.2 | 5.5 | 49.5 | 6.2 | 1,582 | 19.9 | 20.4 |
| Experts (NVFP4 + BF16) | 86.8 | 85.6 | 197.7 | 146.1 | 534.7 | 269.5 | 425.6 |
| Attention | 8.5 | 8.0 | 28.8 | 27.0 | 109.0 | 104.5 | 208.4 |
| Glue (mixes, norms, router, convolutions) | 12.4 | 11.5 | 30.3 | 28.0 | 107.5 | 99.5 | |
| Prompt tokens/s | 100 | 130 | 143 | 291 | 42 | 482 | 553 |

The previous projection kernel fell apart above 64 rows; that is why prompts used to be fed 16
rows at a time. What is left at 256 rows is the experts (about half, close to what reading
every expert's weights once costs), then attention and the glue kernels, which have not been
touched yet.

Prompt processing at depth (256 rows per step), tokens/s: 258 at 4K, 126 at 16K, 31 at 100K.
There the attention kernel is nearly all of the time: every row still reads the whole key and
value cache by itself.

Steps of 2 to 8 rows (verifying drafts, a few parallel sessions) gain from the projections
only: 8 sessions reach 63.4 tokens/s in total (was 56.2). The experts of such steps are limited
by reading weights: rows rarely share an expert, and each row's eight experts are about 2 GB
across the model.

## Attention for several rows at once, 2026-11-21

Until here every row read its whole key and value cache by itself: at 100,000 tokens of
context a step of 256 rows read about 1.7 TB. Now up to 32 consecutive rows of a session share
one read: a block owns 32 rows (96 query vectors of one key head) and 1,024 positions, walks
them 64 positions at a time through shared memory, and computes both the scores and the
weighted values with the matrix instruction from FP16 copies of the cache. The sliding-window
layers (40 of the 48 visits) use the same kernel on their rings.

Prompt processing, 256 rows per step, tokens/s (context faked beyond the prompt, as before):

| Context | Per-row attention | Full-attention layers batched | All layers batched |
|---|---:|---:|---:|
| short (512-token prompt) | 485 | 485 | 573 |
| short, 512 rows per step | 553 | | 680 |
| 4,000 | 258 | 326 | 523 |
| 16,000 | 126 | 301 | 470 |
| 100,000 | 31 | 206 | 273 |

Attention's share of a 256-row step, in ms: 102 -> 18 at short context, 377 -> 69 at 4,000
(the first number already has the full-attention layers batched), 835 -> 515 at 100,000
(before either it was not profiled; the whole step took about 8,300 ms there). The other phases are unchanged at about 420 ms.

What is left at 100,000 tokens is the full-attention kernel itself, 8 visits of about 60 ms.
By the count of multiplications it runs at roughly a quarter of the rate the projection
kernels reach, so there is room in it (it stops four times per 64 positions to hand data
between its waves, and its shared memory is nearly all of what a block may have).

Steps with one row per session (parallel generation) are slower through this kernel than
through the per-row one (4 sessions at 30,000 tokens: 22.4 against 27.0 tokens/s in total), so
they keep the per-row kernel: the batched one is used from 4 rows per session on average.

Steps that verify 7 given drafts (no drafter cost) at 4,000 tokens of real text give 54 to 67
tokens/s, against 21 for plain generation there. At 100,000 tokens of faked context, with only
the full-attention layers batched, such steps went from 17.8 to 32.9 tokens/s over 24 tokens;
with all layers batched there is no comparable number, because on that meaningless context the
run stops agreeing with plain generation after a few tokens (see DSPARK.md).

Generation on this build: 24.7 tokens/s for one session, 65.0 in total for 8 sessions.

Memory: the FP16 copies of the full-attention caches add about 33 KB per token per session to
the 65 KB of the F32 caches, which single-row steps still read.

## FP16 cache for generation, real text at depth, and how dense attention is, 2026-11-21

Generation for one session, tokens/s (context faked beyond the prompt):

| Context | F32 cache | FP16 cache |
|---|---:|---:|
| short | 24.7 | 25.0 |
| 4,000 | 21.3 | 23.7 |
| 16,000 | 19.6 | 22.8 |
| 100,000 | 13.3 | 15.4 |

(Only the 100,000 pair is from the same build and the same faked content; the other F32
numbers are from step 13.) At 100,000 tokens attention is 25 ms of a 65 ms step (with the F32
cache the step was 75 ms; that attention share was not profiled). It reads 3.3 GB per token
there, which the memory could deliver in about 17 ms, so the kernel is not at the limit yet.

Real text at depth, for the first time: a 102,400-token prompt of wikitext at 256 rows per
step takes 288 s (355 tokens/s on average from an empty context to 100K), and generation
after it runs at 15.8 tokens/s. Verifying 7 given drafts after 4,000 tokens of real text
still reproduces plain generation on this build (53.6 tokens/s with given drafts).

### What sets the speed at depth

Each generated token reads every earlier position's keys and values in the 8 full-attention
layer visits: 33 KB per token of context in FP16. That is 3.3 GB per generated token at
100,000 tokens and 33 GB at 1,000,000, on a machine that moves about 190 GB/s. No kernel
changes that; only reading less does.

GUFO's own numbers show the same for a dense model. Its published table for Qwen 27B on this
machine goes from 649 to 260 tokens/s for prompts and from 12.4 to 8.5 for generation between
an empty context and 131,072 tokens. The model that stays flat there, Flash-Next, was trained
with an index that picks 2,048 tokens to attend to; Inkling 2 Mini has nothing of the kind.

### Is the attention sparse enough to skip most of the cache? No

`INKLING2_ATTN_STATS` computes, for the last row of a step, the exact attention weights of
all 24 query heads in the 8 full-attention visits (192 heads) and asks how few positions hold
nearly all of the weight. On the wikitext prompt:

| Context | Positions holding all but 1e-3 of the weight | All but 1e-4 | Slices of 64 positions, all but 1e-3 | All but 1e-4 |
|---:|---:|---:|---:|---:|
| 16,384 | 45% (median 42%) | 65% (72%) | 74% (82%) | 89% (96%) |
| 32,768 | 35% (32%) | 54% (56%) | 60% (64%) | 76% (84%) |
| 65,536 | 44% (46%) | 62% (71%) | 66% (79%) | 82% (94%) |
| 98,304 | 42% (49%) | 61% (74%) | 65% (81%) | 81% (94%) |

The 2,048 heaviest positions (what Flash-Next's index would keep) hold 87% of the weight on
average at 16,384 tokens and 94% at 32,768 (median 98%): keeping only those would drop 6% to
13% of the attention on average.

The measurement was checked against the attention kernel itself: for the measured row the
host's and the kernel's per-chunk highest logit agree within 0.003 and the per-chunk sums
within 0.16%, over 6,144 chunks. That is also the first check of the batched kernel's scores
on real text beyond 4,096 tokens.

(Mean over the 192 heads, median in brackets.) On average 34% to 70% of the weight is on the
last 1,024 positions, and the rest is spread thinly over the whole context: a typical head
needs most of the 64-position slices to stay within one part in ten thousand. Skipping the
values of positions that get no weight, or screening keys coarsely first, would therefore save
a fifth of the reads at best while changing the result more than anything done so far. On
this evidence those two ideas are not pursued for now. It is encyclopedia prose and one row
per depth; code and agent logs may be more concentrated and should be measured the same way
before the question is closed.

### Another implementation of the same model on the same machine

The Loom project (github.com/ROCm/hrx-system) publishes a package for Inkling 2 Mini measured
on a Strix Halo on 2026-11-21 (huggingface.co/loom-fun/unsloth-Inkling-2-Mini-loom), from a
Q4_K_XL file with 8-bit projection weights, so not the same arithmetic as here. Its reported
numbers: prompts at 1,179 to 1,282 tokens/s from 1K to 16K and 1,102 at 32K, with steps of
2,048 tokens; generation at 42.7 tokens/s at short context, 39.6 at 2K and 29.6 at 32K with
its drafter on the GPU proposing one token per step (60% accepted, 1.6 tokens per pass, so
about 27 passes per second, close to the 25 plain steps per second here). The two things it
has that this extension does not are therefore large prompt steps with every kernel built for
them, and a drafter on the GPU. Its attention reads the whole FP16 cache like ours, in 16
splits per head with a running softmax, and it also slows with depth.

## Large prompt steps, 2026-11-21

Prompts are now fed up to 4,096 rows per step. What made that pay: expert tiles sized to the
step (a row picks 8 of 128 experts, so 2,048 rows give each expert about 128 rows to share one
read of its weights), and moving the router and the loop controllers off kernels that ran one
small block per row and output (they were over a third of a large step).

| Rows per step | 256 | 512 | 1,024 | 2,048 | 4,096 |
|---|---:|---:|---:|---:|---:|
| 4,096-token prompt, tokens/s | 564 | 715* | 971 | 1,082 | 1,146 |

(* before the router and controller change.) Real text: a 32,768-token prompt averages 836
tokens/s, a 102,400-token prompt 550 (186 s; 288 s before).

One 2,048-row step at short context takes about 1.9 s: experts 813 ms, attention 312 (the
sliding-window layers), input projections 236, loop controllers 180, glue 215, output
projection 78. Experts and projections run at about 7.5e12 multiply-adds per second, a third
of what this GPU is reported to sustain.

At 100,000 tokens of (faked) context the same step takes 5.6 s, 4.0 s of it in the 8
full-attention visits: 368 tokens/s. There attention costs 2 ms per row whatever the step
size: 4.9e9 multiply-adds per token, more than the rest of the model together, at a third of
the rate the other kernels reach. A cheaper exponential for the weights was tried and changed
nothing, so the exponentials are not what limits it. This kernel is the open item for prompts
at depth.

## What bounds generation for one session

One token reads these weights once (profile of a single-row step, 40.5 ms):

| Part | Bytes per token | Time at 190 GB/s | Measured |
|---|---:|---:|---:|
| Input projections (BF16, 48 visits) | 1.9 GB | 10.0 ms | 12.0 ms |
| Experts (8 of 128 per visit; NVFP4 on 44 visits, BF16 on 4) | 2.1 GB | 11.0 ms | 11.4 ms |
| Output head (BF16, 200K rows) | 0.9 GB | 4.8 ms | 5.1 ms |
| Output projection (BF16) | 0.7 GB | 3.6 ms | 5.3 ms |
| Everything else (about 15 small kernels per visit) | | | about 8 ms |

So plain generation with the weights as they are in the file cannot exceed about 34 tokens/s
on this machine, and it is at 24.7. The 8 ms of small kernels is mostly launch latency (726
launches per token); fusing them is worth perhaps 10%.

Steps that verify drafts: 2 rows take about 60 ms, 4 rows about 80, 8 rows about 113,
because every extra row reads its own 2.1 GB of experts. A drafter that gets 60% of single
drafts accepted (what Loom reports for this model) would therefore give 1.6 tokens per 60 ms
plus its own cost: no gain over 24.7. Drafts pay when many are accepted in a row: 8 tokens
per 113 ms is 70 tokens/s, which is the case for text copied from the context (code edits,
quoting), where the drafts cost nothing. Loom's 42.7 tokens/s comes with 8-bit projection
weights, which halve the largest read; that belongs to the later phase where weights change.

## Drafts copied from the context, 2026-11-21

`INKLING2_LOOKUP=15` in the development driver generates with drafts taken from the context:
the longest recent match of the last tokens (3 to 16) is found, and the tokens that followed
it there are proposed and verified by the model in one step of 1 + drafts rows. Nothing but
the model decides the output; a wrong draft only costs the larger step. The number of drafts
starts at 3, doubles while every draft is kept (up to 15) and falls back after a miss; after
a step that kept none, four steps go without drafts unless the match is 8 tokens or longer.

| Case (160 tokens generated) | Plain | With lookup | Drafts kept |
|---|---:|---:|---:|
| A 600-token passage given twice, then its first 100 tokens | 23.7 | 81.0 | 160 of 160 |
| The same after 30,000 tokens of other text | 20.9 | 75.3 | 160 of 160 |
| A 4,096-token article, continued | 23.0 | 30.0 | 65 of 87 |
| 2,000 tokens of other prose, continued | 23.6 | 23.1 | 9 of 35 |

With a fixed 15 drafts and no controller the copy case reaches 92.8 tokens/s and the prose
case drops to 18.9. This is the regime where drafting pays on this machine (see "What bounds
generation" above): long accepted runs, and drafts that cost nothing to make. It lives in the
driver for now; a serving runner would run the same loop.

## Things tried on the matrix-instruction kernels that did not help

Recorded so they are not tried again blindly. The batched attention kernel does its products
at about half the rate of the projection kernel, which itself is at a third of what this GPU
is reported to sustain. Timing the attention kernel with parts switched off (1,024 rows at
100,000 tokens, 1,930 ms): without the cache loads 1,850; without the score products 1,090;
without the value products 1,310; without loads and products 370. So the time is in the
matrix instructions themselves, not in memory. Then, none of these changed it: a cheaper
exponential for the weights; two interleaved accumulators per product, to shorten dependency
chains; padding the projection kernel's shared memory to the attention kernel's 60 KB (to
test whether the footprint limits how many blocks run at once: it does not); and a
projection kernel with 16 products per 8 operand reads instead of 8 per 6. What makes one
instruction slower in one kernel than in the other is not understood yet.

## Not measured yet

- Output quality beyond that smoke test (no perplexity or logit comparison yet).
- Anything on GUFO: there is no Inkling runner yet.
- DSpark speculative decoding.
