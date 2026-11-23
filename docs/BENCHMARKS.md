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
numbers: prompts at 1,179 to 1,282 tokens/s from 1K to 16K and 1,102 at 32K (its
configuration names 2,048-token prompt steps); generation at 42.7 tokens/s at short context, 39.6 at 2K and 29.6 at 32K with
its drafter on the GPU proposing one token per step (60% accepted, 1.6 tokens per pass, so
about 27 passes per second, close to the 25 plain steps per second here). The two things it
has that this extension does not are therefore large prompt steps with every kernel built for
them, and a drafter on the GPU. From a partial reading of its published program, its
attention reads the whole FP16 cache like ours, in 16 splits per head with a running softmax;
its own numbers slow with depth too.

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
on this machine if memory delivers 190 GB/s (what the single-row kernels here reach), or about
41 at the 228 GB/s that Loom's report gives for this machine; it is at 24.7. The 8 ms of small kernels is mostly launch latency (726
launches per token); fusing them is worth perhaps 10%.

Steps that verify drafts, at short context: 2 rows take about 60 ms, 4 rows about 80, 8 rows
about 113, because every extra row reads its own 2.1 GB of experts. A drafter that gets 60%
of single drafts accepted (what Loom reports for this model) would therefore give 1.6 tokens
per 60 ms plus its own cost: no gain over 24.7. At 100,000 tokens of (faked) context a plain
step is 63 ms, a 2-row step 97 ms and an 8-row step about 160 ms: one draft at 60% gives
about 4% there, eight kept tokens per step give 50 tokens/s against 16. So at any depth drafts
pay when several are accepted in a row, which is the case for text copied from the context
(code edits, quoting), where the drafts also cost nothing to make. Loom's 42.7 tokens/s comes
with 8-bit projection weights, which halve the largest read; that belongs to the later phase
where weights change.

The single-pass generation kernel (GUFO's layout for its dense models) was weighed with a
measurement instead of being built: at 100,000 tokens the per-row kernel spends about 9 to 11
ms on the keys, which is what reading them costs, and 14 to 16 ms on the values, about 5 ms
more than reading them costs. A rewrite could therefore gain at most 5 ms of a 63 ms step at
100,000 tokens and nothing at short context, and it needs the values laid out the way the
batched kernel cannot use. Not done.

## Saved sessions, 2026-11-21

`GpuEngine::Save` returns a session as bytes (its position, the sliding-window rings, the
used part of the full-attention caches, the convolution histories) and `Restore` puts it
back. A session of 32,768 tokens is 1.58 GB, saved in 1.09 s and restored in 0.06 s;
processing those tokens again takes 39 s. Generation after a restore gives the same tokens
as without it. About 0.5 GB of a snapshot is the rings, whatever the length. This is the
mechanism a prefix cache needs; choosing which prefixes to keep, and keeping them on disk, is
not built.

## Speed against context depth, 2026-11-21

`INKLING2_SWEEP=<token file>` in the development driver processes one long real text
(wikitext-2 train, 2,048 rows per prompt step) and stops at chosen depths. "Interval" is the
average prompt speed since the previous point, "at depth" the speed of the last full step
before the point. Generation is one step that produces a token, the median of 12; "8 drafts"
is one step that verifies 8 drafted tokens.

Baseline, engine as of step 25 (commit `e5b0104`):

| Depth | Prompt, interval | Prompt, at depth | Seconds from 0 | One token | Generation | 8 drafts |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 1,034 t/s | | 1 | 42.3 ms | 23.7 t/s | 128 ms |
| 4,096 | 1,020 | 1,068 | 4 | 43.4 | 23.1 | 124 |
| 16,384 | 920 | 847 | 17 | 44.7 | 22.4 | 136 |
| 32,768 | 744 | 678 | 39 | 48.3 | 20.7 | 148 |
| 65,536 | 561 | 486 | 98 | 55.7 | 17.9 | 171 |
| 100,000 | 419 | 378 | 180 | 63.3 | 15.8 | 182 |
| 131,072 | 331 | 310 | 274 | 70.2 | 14.2 | 200 |
| 196,608 | 234 | 227 | 554 | 84.6 | 11.8 | 234 |

The run was stopped at 196,608 (it was planned to 1,040,000 and would have taken about three
hours) to use the machine for the kernel work below. Both curves are straight lines in the
depth so far: a prompt token costs 0.94 ms plus 17.8 microseconds per 1,000 tokens of depth, a
generated token 42 ms plus 0.215 ms per 1,000. Extrapolated, not measured: about 51 prompt
tokens/s and 4 generated tokens/s at 1,000,000, and about three hours to process a
1,000,000-token prompt from nothing. The 131,072 and 196,608 intervals shared the GPU for
about a minute with the micro-benchmark below.

With the short path of step 26 (same run, to 100,000):

| Depth | Prompt, interval | Prompt, at depth | Seconds from 0 |
|---:|---:|---:|---:|
| 4,096 | 1,089 | 1,064 | 4 |
| 32,768 | 858 | 737 | 37 |
| 100,000 | 541 | 437 | 162 |

## What the matrix instruction can do, 2026-11-21

`INKLING2_WMMA_BENCH=1` runs a micro-benchmark of the FP16 matrix instruction
(`extension/src/hip/bench.hip`): blocks of 8 waves doing nothing but products, with operands
kept in registers or read at the rate a real kernel needs. Measured while the depth sweep was
also using the GPU, so the absolute figures are low; the ratios are the point.

| Pattern | Multiply-adds per second |
|---|---:|
| Operands in registers | 27e12 |
| One operand read per 3 or 8 products | 25 to 27e12 |
| 12 products per 7 operand reads (3 x 4 tiles) | 20e12 |
| 8 products per 6 reads (2 x 4 tiles) | 17.5e12 |
| 6 products per 5 reads (2 x 3 tiles) | 16.5e12 |
| 3 products per 4 reads (1 x 3 tiles: the attention kernel's score pattern) | 12e12 |

Reading the operands of the next products ahead of time did not help, and reading them from
device memory was as fast as from shared memory. So an operand read costs about as much as a
product, and a kernel is fast when each operand it reads is used by several products.

For comparison, what the engine's kernels reach on real work: experts and projections about
7.5e12, the batched attention kernel about 2.5e12 at 100,000 tokens of depth. The compiled
attention kernel showed why it was far below even its own pattern's 12e12: per slice of 64
positions a wave executed 48 products and well over a thousand other instructions (staging
keys and values through shared memory, a cross-wave maximum and sum for every query, a rescale
of every accumulator, masks and bias tests for every score).

Step 26 removes most of that for slices far before the rows (keys and values straight from the
cache into the instruction, the maximum only recomputed when a score exceeds it, sums kept per
lane, one barrier less): 16% faster at 100,000 tokens of depth (378 to 437 tokens/s), 29% over
the whole 100,000-token prompt (419 to 541 over the last interval). That is less than the
instruction counts suggested, so the kernel is not bound by them alone. A 2,048-row step at
100,000 now takes 4.7 s: 1.9 s is everything that does not depend on depth, and of the 2.8 s
of full attention about 1.1 s is reading the cache (each group of 32 rows reads all keys and
values of a head: 210 GB per step) and about 0.9 s the products at their pattern's rate.
What follows from that, not done yet: more rows per cache read, tiles that use each operand
for several products, and the same tiling in the expert and projection kernels, which is
where the 1.9 s is.

## Attention blocks that fit several to a compute unit, 2026-11-21

Step 26 left the attention kernel far below its pattern's rate, and the estimate of where the
time went (cache reads, instruction counts) was wrong: ordering the blocks so that concurrent
ones read the same cache region changed nothing. Switches that turn parts of the kernel off at
run time (`INKLING2_ATTN_SKIP` bits 4 to 64, so that the compiler cannot remove what depends
on them) gave the real split of the 3.2 s that full attention took in a 2,048-row step at
100,000 tokens: 2.3 s in the matrix products with their operand reads, nothing measurable in
the exponentials, 0.2 s in reading keys and values.

The products ran at a third of what the micro-benchmark gives for the same pattern, and the
micro-benchmark had also shown blocks with 60 KB of shared memory at about two thirds of
blocks with 16 KB. A compute unit has 64 KB, so a 60 KB block runs alone on it and its 8 waves
have nothing to overlap their waits with. Step 28 therefore splits the kernel by footprint:

- parts of the context wholly far before a group's rows: a variant with no queries, keys or
  values in shared memory, 16 KB;
- the rest (sliding-window layers, the part near the rows): queries read from device memory,
  32 KB.

| Depth | Prompt at depth, step 25 | Step 26 | Step 28 | Seconds from 0, step 25 | Step 28 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 1,068 t/s | 1,064 | 1,113 | 4 | 3.6 |
| 16,384 | 847 | | 981 | 17 | 15.5 |
| 32,768 | 678 | 737 | 867 | 39 | 33 |
| 100,000 | 378 | 437 | 594 | 180 | 129 |

Quality figures are the same in every variant of steps 26 and 28 (KLD 0.001525 at 4,096
tokens). A 2,048-row step at 100,000 tokens (context faked, 602 tokens/s) now spends 1.85 s in
attention, 0.78 s in the experts, 0.24 s in the input projections and 0.53 s in everything
else.

Tried in the same session and left off, each measured on a 4,096 to 16,384-token prompt or at
100,000 tokens:

| Change | Result |
|---|---|
| Projection kernel reading weights and inputs straight from device memory, no shared memory (`INKLING2_WMMA_DIRECT=1`) | 1,084 to 620 tokens/s |
| Expert kernel reading its inputs straight from device memory (`INKLING2_WMMA_DIRECT=2`) | 1,084 to 997 |
| Staging 2 slices of 16 columns instead of 4 in those kernels (half the shared memory) | 1,084 to 1,007 |
| Blocks ordered so that concurrent ones read the same keys and values (`INKLING2_ATTN_FAST` bit 2, still on) | no change |
| Asking for the next slice's keys and values one slice early | attention 1.85 to 1.93 s |

So what helped the attention kernel does not carry over to the projection and expert kernels:
their weights come from memory in volume, and staging them once per block in shared memory is
worth its footprint. In the far attention kernel, turning the score products off alone saves
nothing while turning the value products off saves 0.6 s and both 1.24 s; what the score phase
waits for is not identified yet.

## One-pass attention, and the sweep to 1,040,000 tokens, 2026-11-22

Step 29 replaces the batched attention kernels for prompt steps with `AttentionFlashKernel`,
which follows the prefill attention kernel of
[headbouyJB/inkling2-gufo](https://github.com/headbouyJB/inkling2-gufo) (MIT; see
`extension/THIRD_PARTY.md`). What it does differently from the kernels of steps 17 to 28:

- A wave owns 16 rows of one query head and keeps their queries, their running softmax and
  their output in registers for the whole context. The kernels before split a context into
  16 parts and combined them in a second kernel.
- The weights go from the first product to the second by an exchange between lane pairs,
  not through shared memory. Step 28's timing switches had just shown that reading the
  weights back from shared memory was most of the second product's cost (0.56 of 0.68 s).
- The rows of keys and values staged in shared memory are padded by 16 bytes. With a row
  stride that is a power of two, the 16 lanes of a wave read the same bank of shared memory
  at once: without the padding this kernel ran at 420 tokens/s at 100,000 tokens (slower than
  step 28), with it at 805. Paddings of 8, 16 and 24 bytes gave the same.

Steps of fewer than 64 rows (verifying drafts) keep the older kernels: one block walking a
whole context for a handful of rows was slower there (385 against 168 ms at 100,000 tokens).

Prompt speed at depth, same real text and method as the baseline above (2,048 rows per step):

| Depth | Step 25 | Step 28 | Step 29 | One token, step 29 | Generation |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 1,068 t/s | 1,113 | 1,189 | 43.4 ms | 23.1 t/s |
| 16,384 | 847 | 981 | 1,121 | 44.6 | 22.4 |
| 32,768 | 678 | 867 | 1,024 | 48.2 | 20.7 |
| 65,536 | 486 | | 891 | 55.8 | 17.9 |
| 100,000 | 378 | 594 | 787 | 63.3 | 15.8 |
| 131,072 | 310 | | 708 | 70.2 | 14.3 |
| 196,608 | 227 | | 584 | 84.6 | 11.8 |
| 262,144 | | | 498 | 98.8 | 10.1 |
| 393,216 | | | 384 | 127.5 | 7.8 |
| 524,288 | | | 312 | 156.1 | 6.4 |
| 655,360 | | | 263 | 184.7 | 5.4 |
| 786,432 | | | 227 | 213.5 | 4.7 |
| 917,504 | | | 199 | 242.1 | 4.1 |
| 1,040,000 | | | 180 | 268.6 | 3.7 |

Time to process a prompt from nothing: 4 s for 4,096 tokens, 30 s for 32,768, 106 s for
100,000 (step 25: 180 s), 251 s for 196,608 (was 554 s), 373 s for 262,144, 1,057 s for
524,288 and 3,323 s (55 minutes) for 1,040,000. This is the first run of the engine beyond
196,608 tokens; it completed, and the most likely next token was the text's own at 9 of the
15 depths, 1,040,000 included. Correctness against llama.cpp is still only checked to 4,096
tokens. Generation was not touched: one token costs 42 ms plus 0.22 ms per 1,000 tokens of
depth, which is reading the cache (33 KB per token of context).

Quality: KLD 0.001490 at 4,096 tokens with 256 rows per step (perplexity 7.4847 against
llama.cpp's 7.4828, same top token 97.70%) and 0.002124 at 512 tokens with 64 rows per step.

A 2,048-row step at short context is now 1.73 s (1,185 tokens/s): NVFP4 experts 0.72 s,
input projections 0.23, attention 0.16, loop controllers 0.19, the rest 0.43. At 100,000
tokens attention is 0.99 s of 2.5 s.

The expert kernel with parts switched off (`INKLING2_EXPERT_SKIP`): of its 0.71 s, 0.28 s is
converting the 4-bit weights to FP16, 0.30 s the products, 0.22 s everything else. Tried on it
and left out: padding its staged rows like the attention kernel's (0.74 s), doing the
conversion's multiplication in FP16 (0.72 s, same output).

Not built: using the CPU cores for part of the attention arithmetic. At 12e12 multiply-adds
per second on the GPU, 16 cores would add well under a tenth, for a large change.

## The drafter on the GPU, 2026-11-22

Step 30; the account and the tables are in [DSPARK.md](DSPARK.md), section 7. In short,
through the server with thinking on: code 24.0 to 31.4 tokens/s greedy and 22.3 to 33.2 at
the default temperature, technical answers about a tenth faster, prose and generation after
long prompts unchanged, prompt speed unchanged.

## Steps of a few rows over a long context, 2026-11-22

Step 31. A step that verifies drafts has 2 to 17 rows. Its full-attention visits used the
per-row kernel for 2 or 3 rows (the whole cache read once per row) and the batched kernel
from 4 rows (built for 32 rows per block). At 100,000 tokens that made a 2-row step cost 97
ms against 63 for one row, and a 4-row step 143.

Such steps now use the one-pass kernel of step 29 with the context cut into parts: one block
per (key head, part), up to 32 parts, each leaving its maximum, sum and weighted values per
row and head for the combine kernel. The cache is read once for all rows, by as many blocks
as there are parts.

One step, ms, context faked to the stated depth:

| Rows | Short | 32,000 before | after | 100,000 before | after |
|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 47 | 47 | 63 | 63 |
| 2 | 56 | 68 | 65 | 97 | 80 |
| 4 | 81 | 100 | 95 | 143 | 108 |
| 8 | 104 | 115 | 117 | 157 | 129 |

Short contexts and prompt steps of 16 or 32 rows are unchanged (a 4,096-token prompt at 32
rows per step: 208.9 tokens/s, before 208.0). Quality with every step going through it
(4,096-token context, 8 rows per step): KLD 0.001588, perplexity 7.4742 against llama.cpp's
7.4828, same top token 97.66%; the kernels it replaces give 0.001594 on the same run.

Tried first and dropped: the same arithmetic with each wave reading its 16 keys and its 16
value dimensions straight from the cache, no shared memory. It was slower than what it
replaced (119, 152 and 177 ms for 2, 4 and 8 rows at 100,000): sixteen separate streams per
wave do not read as fast as staged, contiguous loads. A single row is not faster through the
new kernel either, so plain generation at depth is where it was: 42 ms plus 0.22 ms per 1,000
tokens of context, which is reading the cache.

What it buys is drafting at depth. Through the server, drafter on, thinking on, 500 tokens
asked for after a prompt of C++ source (`decode_tps`, without and with the drafter):

| Context | Greedy | Default sampling |
|---|---:|---:|
| 39,963 tokens of source | 22.4 -> 36.4 | 18.6 -> 29.7 |
| 119,922 tokens of source | 16.8 -> 25.3 | 14.0 -> 19.1 |
| 100,024 tokens of prose, a summary asked | 15.5 -> 16.2 | |

## Sessions decoded together through the server, 2026-11-22

Step 32. The server now gives the engine one row per session that is generating, in one
step (GUFO's batched plans, up to 8 sessions; `INKLING2_SERVE_BATCH_WIDTH=1` turns it off).
Drafts copied from the context ride along per session. The model's drafter is left to
single sessions: beside other sessions a row costs a session little, and the drafter's pass
would cost each of them about as much.

Four requests sent at once to a server with 4 sessions (three of 700 tokens, one that ends
after about 340), drafter loaded:

| | Wall time | Tokens/s in all | Per request while all run |
|---|---:|---:|---:|
| One session at a time (each with the drafter) | 88.4 s | 27.6 | 24.6 to 32.5 when it has the GPU |
| Four in one step | 61.2 s | 39.8 | 11.6 each |
| Two requests, two sessions, in one step | 42.4 s | 33.0 | 16.7 each |

On the build of step 34 (which zeroes the positions after each step's rows and draws every
session's token from capped logits made on the device) the same four requests: 89.8 s one at
a time (27.4 tokens/s in all), 65.5 s in one step (37.2 in all, 10.8 each).

## Prompt steps: what the experts and the loop controllers cost, 2026-11-22

Step 33. A 2,048-row step at short context, by kernel group, before and after (ms):

| Group | Before | After |
|---|---:|---:|
| NVFP4 experts | 728 | 705 |
| Input projections | 236 | 235 |
| Attention | 124 | 123 |
| Glue A (mixing earlier blocks, messages) | 111 | 111 |
| BF16 experts | 83 | 70 |
| Output projection | 82 | 82 |
| Loop controllers (FP16 copies, down, norms, gain and forget, apply) | 33 + 41 + 20 + 60 + 46 | 27 + 20 + 19 + 17 + 47 |
| Glue B, C, D, norms, router | 44 + 42 + 12 + 29 + 19 | unchanged |
| Whole step | 1,710 | 1,600 |

Two changes:

- **Only the expert tiles that have rows are launched.** Measured on a real prompt
  (`INKLING2_EXPERT_STATS=1`): rows spread very unevenly over the 128 experts. Per layer visit
  9.4% of the experts get no row, 31% get more than 128 pairs (one tile), 14.8% more than
  256, 4.4% more than 512, and some get every row. The kernels launched a full grid of
  36,864 blocks of which nine in ten found nothing to do; the tiles with pairs are now
  listed on the host (one round trip per layer visit, for steps of 256 rows or more).
- **The loop controllers take their inputs as one FP16 value**, like every other
  projection. They had carried a second FP16 part for what rounding lost, which doubled their
  products. The router, which picks the experts, keeps both parts.
  (`INKLING2_CTRL_LOW_INPUTS=1` restores the old arithmetic.)

Prompt speed at depth, same text and method as before (2,048 rows per step):

| Depth | Step 29 | Step 33 | One token | A step with 8 drafts |
|---:|---:|---:|---:|---:|
| 4,096 | 1,189 t/s | 1,267 | 43.3 ms | 118 ms (was 120) |
| 16,384 | 1,121 | 1,189 | 44.6 | 124 (131) |
| 32,768 | 1,024 | 1,079 | 48.3 | 128 (138) |
| 65,536 | 891 | 928 | 55.8 | 137 (162) |
| 100,000 | 787 | 815 | 63.4 | 138 (174) |
| 131,072 | 708 | 728 | 70.2 | 147 (177) |
| 196,608 | 584 | 597 | 84.6 | 159 (215) |

A 100,000-token prompt takes 100.8 s (was 106.3), 196,608 tokens 241.9 s (was 251.1). With
4,096 rows per step a 4,096-token prompt runs at 1,385 tokens/s against 1,280; the server
keeps 2,048, because a session's sliding-window rings, and with them every cache checkpoint,
grow with the step size.

Quality, against llama.cpp on the same file: 4,096-token context at 256 rows per step, KLD
0.001469 (was 0.001490), perplexity 7.4823 against 7.4828, same top token 97.70%; 512-token
contexts at 64 rows per step, 8 chunks, KLD 0.002102 (was 0.002112); 4,096 tokens at 8 rows
per step 0.001647 (was 0.001588). All inside the band every change of kernel has moved
within.

Tried on these kernels and not kept, each measured on the same 2,048-row step:

| Tried | Result |
|---|---|
| Expert tiles of 256 pairs instead of 128 (fewer passes over an expert's weights: 1.14 per expert instead of 1.59) | 864 ms against 728: the larger shared-memory footprint costs more than the passes saved |
| Expert tiles of 64 and 32 pairs | 822 and 1,192 ms |
| Adding each expert's output straight into the row's sum with atomic adds, instead of eight contributions and a sum kernel | 733 ms against 726, and the result then depends on the order the blocks run in |
| The grouped-GEMM staging of headbouyJB/inkling2-gufo (`k_gemm4`: every thread owns half a weight row, two buffers, one barrier per 64 columns, a byte-permute table for the 4-bit codes) written for this engine's experts | 721 ms against 703 for the kernel here |
| The one-pass attention kernel with 128 rows per block instead of 64 (the cache read once for twice the rows) | 790 against 787 tokens/s at 100,000: at depth the kernel is bound by its products, not by reading the cache |
| Sums across a glue block by exchange between lanes and two barriers, instead of a tree of eleven | slower: generation 22.9 against 24.8 tokens/s |

So the expert, projection and attention kernels are where designs of this kind end on this
GPU: about 8e12 multiply-adds per second in the experts and projections and 10e12 in
attention, against 27e12 with every operand in registers. Each product here needs an operand
from shared memory, and that read costs about as much as the product.

## The same request gives the same text; smaller checkpoints, 2026-11-22

Step 34. With the drafter on, two identical temperature-0 requests could give different
text: one decoded from an empty session and one from a cache checkpoint parted ways after a
few hundred tokens. Without the drafter they never did. Three causes, found in this order:

1. **The controller read the clock.** Whether a step drafted depended on measured times. It
   now charges every drafted step a fixed cost in plain steps (the measurements of
   DSPARK.md), so a request drafts at the same steps every time.
2. **The drafter's shortlist carried over** from earlier requests, and was shared between
   sessions. It is now per session and rebuilt at the start of each request: the prompt's
   tokens in order, then the likely tokens of each decode step.
3. **A step of several rows depended, in its last bits, on what lay in the cache after its
   last row.** Found with a per-step trace: of an 8-row verification step, the first seven
   rows' logits were bit-identical between the two runs and the last row's were not. The
   multi-row attention kernels read keys and values in tiles of 64 positions; a tile reaches
   past the last row, and although those positions get no weight, the row's result moved in
   the seventh digit with their content (reproduced in the development driver:
   `INKLING2_STALE_TEST`, which runs a step once over leftovers and once over zeroed caches;
   it showed the effect when the last row's position was even, traced to the values of the
   one position after it in a sliding-window ring). The batched kernels were checked against
   the per-row F32 reference on such steps and are right to the usual 3e-4, so this was not
   a wrong result, only one that was not reproducible. Every step of several rows now zeroes
   the 64 positions after its last row in every cache first (one launch, about 13 MB of
   writes), and the rings hold 64 more positions for that.

After that: a cold request, the same request from a RAM checkpoint, from a disk checkpoint
after a server restart, and in a freshly started server give identical text, greedy (a
33,851-token prompt, and a 46-token prompt with 700 generated tokens) and sampled with a seed.

A side effect worth more than the repeatability: steps that verify drafts now reproduce plain
generation in the driver's test with given drafts. On the prose prompt, 1, 3 and 7 drafts per
step: every draft kept (48 of 48, 72 of 72, 84 of 84), output equal to plain generation, at
35.4, 48.3 and 74.1 tokens/s. Before, the same test lost agreement after a token or two (3 of
93 drafts kept with one draft per step), which had been put down to near-ties between
kernels.

**Checkpoints.** A checkpoint stored every sliding-window ring whole: 40 rings of 12.6 MB at
2,048 rows per step, 503 MB, although a later row can only read the 1,024 positions before the
frontier. The adapter now packs that window (a kernel, before a capture) and unpacks it
(after a restore): 168 MB. A checkpoint's fixed part, drafter state included, went from 577
MB to 233 MB, and no longer grows with the step size. Restoring 33,851 tokens takes 16 to 30
ms from RAM and about 300 ms from disk.

Also: the scratch for the convolution histories that a step's rows may leave was allocated
per row (7.9 GB at 2,048 rows per step) and is now per row that can end a run (1 GB).

**The soft cap on the device.** The model caps its logits with 30 tanh(x / 30). That ran on
the host for every row with logits, over 200,058 entries. As a kernel before the copy it
costs nothing measurable: plain generation through the server went from 24.0 to 25.0
tokens/s greedy and from 22.4 to 23.4 sampled. KLD against llama.cpp unchanged (0.001469).

The drafter through the server after all of this (thinking on, 700 tokens; plain -> with
the drafter):

| Request | Greedy | Default sampling |
|---|---:|---:|
| A Python module with tests | 25.0 -> 32.0 | 23.4 -> 27.4 |
| TCP against UDP, concise | 24.9 -> 26.8 | 23.5 -> 22.9 |
| How tides work, 300 words | 24.7 -> 25.2 | 23.4 -> 23.5 |
| Code asked after 39,963 tokens of C++ source (500 tokens) | 23.2 -> 35.3 | 19.3 -> 31.7 |
| Code asked after 119,922 tokens of source | 17.2 -> 24.9 | 14.4 -> 29.3 |
| A summary of 100,024 tokens of prose (300 tokens) | 15.9 -> 16.7 | |

Prompt speed through the server on those requests: 1,147 tokens/s for the 39,963-token
prompt and 988 for the 100,024-token one (step 25: 804 and 555).

## Against llama.cpp at 32,768 tokens, 2026-11-22

The same file, the same machine, one 32,768-token chunk of wikitext (QUALITY.md has the
quality side: perplexity 5.9089 against llama.cpp's 5.9063, KLD 0.0035).

| | This extension | llama.cpp (HIP) |
|---|---:|---:|
| A 32,768-token prompt | 27.9 s (1,175 tokens/s) | 181.5 s (180 tokens/s) |
| Prompt, 4,096 tokens (llama.cpp: llama-bench, synthetic text) | 1,267 tokens/s | 346 |
| Prompt, 512 tokens (this extension as measured at step 16, not since) | 553 | 520 |
| Generation, short context | 24.9 tokens/s plain; 32 on code with the drafter | 19.5 |

## Not measured yet

- Output quality beyond that smoke test (no perplexity or logit comparison yet).
- Anything on GUFO: there is no Inkling runner yet.
