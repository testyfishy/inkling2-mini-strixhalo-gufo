# Quality

Does running Inkling 2 Mini through this extension change the model's output? This is measured
the way GGUF quantizations are usually compared (Unsloth's model cards report the same three
numbers): perplexity, mean KL divergence against a reference distribution, and how often the top
token agrees.

`scripts/quality.sh` reproduces it. llama.cpp's `llama-perplexity --kl-divergence-base` writes its
distributions for wikitext-2 (test split); the extension runs the same token chunks through the GPU
engine and scores the second half of each chunk, exactly as llama.cpp does.

## Result, 2026-11-21

- Model file: native precision (NVFP4 experts + BF16), the same file for both.
- Reference: llama.cpp mainline with #32704 (commit `2a28df0f9`), HIP backend, all layers on GPU.
- 8 chunks of 512 tokens, 2,040 scored positions. Engine fed 16 rows per step.

| Measure | This engine | llama.cpp (reference) |
|---|---:|---:|
| Perplexity | 7.7930 | 7.7776 |
| Mean KLD from the reference | 0.0021 | 0 |
| Median / 99th percentile / max KLD | 0.0008 / 0.0203 / 0.3041 | |
| Same top token | 98.19% | 100% |

llama.cpp's own printed estimate for the same run is 7.7868; the 7.7776 above is recomputed from
the 16-bit log-probabilities it stores in the reference file.

For scale, Unsloth's card for this model reports mean KLD 0.0107 for UD-Q4_K_XL and 0.0109 for
UD-IQ4_XS against their 8-bit reference, and the community design doc cites 0.0030 for Q8_0. The
difference measured here, between two implementations running the same weights, is below all of
those.

## 8-bit expert products (on by default)

In steps of four or more rows (prompts and several sessions decoding together), the NVFP4 expert
products use the GPU's 8-bit integer dot product. The weights stay exact: NVFP4 values are small
integers times a scale. What changes is that the inputs to those products are rounded to 8 bits
per block of 16 values. llama.cpp's GPU path and GUFO's own kernels make the same kind of
approximation. Single-row steps, which is single-session generation, always use the float path.

Same reference file, 2,040 positions, 64 rows per step:

| Measure | Float products | 8-bit products |
|---|---:|---:|
| Perplexity (llama.cpp: 7.7776) | 7.7928 | 7.7994 |
| Mean KLD from llama.cpp | 0.00214 | 0.00199 |
| 99th percentile KLD | 0.0203 | 0.0180 |
| Same top token | 98.19% | 97.99% |

The two columns differ by far less than the sample's uncertainty (perplexity has a standard error
of about 0.44 here), so this measurement shows no quality cost. It is a small sample; a larger
one could show a difference this one cannot. `INKLING2_INT8_EXPERTS=0` (or `INT8_EXPERTS=0` for
`scripts/quality.sh`) selects the float products.

## FP16 matrix-instruction kernels (on by default for steps of several rows)

Steps of several rows round the inputs of the projections and experts to FP16 (weights stay
exact, sums are FP32; see PROGRESS.md step 16). Same 8 chunks of 512 tokens, against the same
llama.cpp reference:

| Kernels | Perplexity (llama.cpp 7.7776) | Mean KLD | Median / 99% KLD | Same top token |
|---|---:|---:|---:|---:|
| Before (float projections, 8-bit experts), 16 rows per step | 7.8172 | 0.001985 | 0.000772 / 0.0172 | 98.24% |
| FP16 projections, 8-bit experts, 16 rows per step | 7.7978 | 0.002122 | 0.000817 / 0.0198 | 97.35% |
| FP16 projections and experts, 16 rows per step | 7.7899 | 0.002158 | 0.000775 / 0.0214 | 98.14% |
| FP16 projections and experts, 64 rows per step | 7.7958 | 0.002137 | 0.000771 / 0.0214 | 97.99% |

The "before" row is not identical to the 8-bit result further up (7.799, 97.99%), which was
measured before the single-row and attention kernels changed (PROGRESS.md steps 11 to 13); it
was rerun here so that all four rows come from the same build apart from the switches.

With 4,096 tokens of context and 256 rows per step (one chunk, 4,095 scored positions), the
new kernels give perplexity 7.4767 against llama.cpp's 7.4828, mean KLD 0.001419, median / 99%
KLD 0.000628 / 0.0131, same top token 98.19% (the previous kernels at 16 rows per step: 7.496,
0.0017, 97.56%). That run crosses the sliding window and the 1,024-position cache chunks with
real text.

All four sit in the same band around llama.cpp (mean KLD 0.0020 to 0.0022). The differences
between them are within what this measurement can resolve: a percent of the 4,096 scored
positions is 41 tokens, and the top-token rate moves by that much between any two arithmetic
paths because of near-ties.

## FP16 attention (on by default for steps of 4 or more rows per session)

Attention in such steps reads FP16 copies of the queries, keys and values and stores the
attention weights as FP16 (PROGRESS.md steps 17 and 18). Against the same llama.cpp references:

| Kernels | Context, rows per step | Perplexity (llama.cpp) | Mean KLD | Median / 99% KLD | Same top token |
|---|---|---:|---:|---:|---:|
| Per-row attention (step 16) | 512, 64 | 7.7958 (7.7776) | 0.002137 | 0.000771 / 0.0214 | 97.99% |
| Batched FP16 attention | 512, 64 | 7.7984 (7.7776) | 0.002068 | 0.000762 / 0.0193 | 98.09% |
| Per-row attention, same build | 4,096, 256 | 7.4767 (7.4828) | 0.001419 | 0.000628 / 0.0131 | 98.19% |
| Batched FP16, full-attention layers only | 4,096, 256 | 7.4814 (7.4828) | 0.001580 | 0.000642 / 0.0167 | 97.90% |
| Batched FP16, all layers | 4,096, 256 | 7.4647 (7.4828) | 0.001573 | 0.000659 / 0.0150 | 98.09% |

At 512 tokens nothing moves. At 4,096 tokens the mean KLD is 0.0016 instead of 0.0014: a
difference of this size is at the edge of what one chunk of 2,047 scored positions resolves,
but it is in the direction one would expect from rounding more values, so it is recorded as a
small cost and not as noise. It is still below the 0.0017 of the kernels before step 16.

The attention output itself was compared with the per-row kernel on the same steps
(`INKLING2_ATTN_AB=1`): the root-mean-square difference is 2e-4 to 3e-4 of the output's size,
the largest single difference 0.006 where the largest value is 10, on 4,096 tokens of real
text (steps of 250 rows, so groups of 32 and of 26 rows), on steps of 40 rows at 100,000
tokens of faked context, and on three sessions in one step. Nothing was non-finite.

## FP16 key/value cache (the only cache from PROGRESS.md step 19)

Steps of few rows (generation) read the FP16 cache with F32 queries, weights and sums. To
score that kernel, the 4,096-token chunk was run with it on every step
(`INKLING2_WMMA_ATTN_MIN_ROWS=0`, 256 rows per step): perplexity 7.4837 against llama.cpp's
7.4828, mean KLD 0.001507, median / 99% 0.000627 / 0.0156, same top token 98.09% (F32 cache:
0.001419, 98.19%; batched FP16 kernel: 0.001573, 98.09%). Its attention output differs from
the F32 kernels' by 2e-4 of its size, and greedy generation of 96 tokens after a 512-token
prompt gives the same tokens with either cache.

## Longer context

One chunk of 4,096 tokens, 2,047 scored positions (2,048 to 4,095), 64 rows per step. This
exercises attention beyond 1,024 positions, the sliding-window ring wrapping, and the chunked
long-context kernel:

| Measure | This engine | llama.cpp |
|---|---:|---:|
| Perplexity | 7.4962 | 7.4828 |
| Mean KLD from llama.cpp | 0.0017 | |
| Same top token | 97.56% to 98.05% across two runs | |

Agreement is as close as at 512 tokens.

### 32,768 tokens, 2026-11-22

One chunk of 32,768 tokens of the same text, 16,383 scored positions (16,384 to 32,766),
against llama.cpp's logits for the same chunk (`llama-perplexity -c 32768
--kl-divergence-base`, a 6.6 GB file). Two runs of this engine at step 34, which take
different attention kernels: 256 rows per step (the one-pass kernel) and 8 rows per step (the
one-pass kernel in parts on full-attention layers, the batched kernel on sliding-window
layers).

| Measure | 256 rows per step | 8 rows per step | llama.cpp |
|---|---:|---:|---:|
| Perplexity | 5.9089 | 5.9125 | 5.9063 |
| Mean KLD from llama.cpp | 0.003457 | 0.003486 | |
| Median / 99% / largest KLD | 0.00157 / 0.0292 / 0.33 | 0.00156 / 0.0291 / 0.47 | |
| Same top token | 97.34% | 97.42% | |

Perplexity is within 0.1% of llama.cpp's. The divergence is about 2.3 times what it is at
4,096 tokens (0.0015), throughout its distribution, and the same for both kernel paths, so it
is not one kernel's doing. By position it is level: the mean over each eighth of the scored
positions, from 16,384 to 32,766, is 0.0041, 0.0034, 0.0035, 0.0032, 0.0030, 0.0032, 0.0036,
0.0037 (at 4,096 tokens the eighths run from 0.0012 to 0.0018). So it does not grow with
depth inside this range; it is a step somewhere between 4,096 and 16,384 tokens.

What is new in that range: from position 8,192 the model scales its queries on
full-attention layers by 1 + 0.1 ln(position / 8,192), which this run is the first to
exercise against a reference. An error in that factor would grow with the logarithm of the
position, which the eighths do not show; larger queries do make scores more sensitive to the
16-bit keys both engines keep. Which engine the step belongs to, or whether to both, this
comparison cannot say: it has no exact reference. Open.

llama.cpp took 181.5 s for this prompt (180 tokens/s); this engine takes 27.9 s (1,175
tokens/s on average from an empty context).

Nothing beyond 32,768 tokens has been checked against a reference.

## What the remaining difference is, and is not, known to be

- The two implementations use different arithmetic. llama.cpp's GPU path keeps its KV cache in
  F16 and computes its products at BF16 precision; this engine uses F32 throughout. A difference
  of this size is consistent with that, but that attribution has not been proven.
- A second reference was tried to settle it: llama.cpp on the CPU with F32 caches
  (`-ngl 0 -ctk f32 -ctv f32`). On the same text it gives perplexity 10.60, far from its own GPU
  path's 7.79, so llama.cpp's paths disagree with each other much more than this engine disagrees
  with its GPU path. That run cannot arbitrate and is not used. (Support for this model landed in
  llama.cpp the day before these measurements.)
- An independent ground truth would be the Hugging Face Transformers implementation of the
  original checkpoint. That comparison has not been run.
- Within this project, the GPU engine matches its own CPU reference to about 2.5e-4 on every layer
  visit, batched prompts give the same result as token-by-token feeding, and each session of a
  multi-session run gives the same tokens as that prompt run alone.

## Limits of this measurement

- 2,040 positions is a small sample; perplexity on it has a standard error of about 0.44.
- Contexts are checked to 4,096 tokens. The log-scaled positions past 8,192 tokens and anything
  near the depths of real use (100,000 tokens and more) are not exercised.
- Text only. Image input, sampling, tool calls and chat formatting are not covered.
