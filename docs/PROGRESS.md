# Progress log

One row per step, in order, so each change can be traced to its effect. All numbers are from the
same machine (Ryzen AI MAX+ 395, Radeon 8060S, 125 GiB) and the same native-precision file
(NVFP4 experts + BF16). Details and method are in [BENCHMARKS.md](BENCHMARKS.md) and
[QUALITY.md](QUALITY.md); the commit is where to look for the code.

Columns: generation for one session (tokens/s), prompt processing (tokens/s), total generation
for 8 sessions (tokens/s). Empty means not measured at that step. Until step 9 the prompts were
a short repeating test pattern; from step 9 they are real text (wikitext-2 chunks).

| # | Commit | Step | 1 session | Prompt | 8 sessions | Output check |
|---:|---|---|---:|---:|---:|---|
| 0 | | llama.cpp HIP baseline (reference, not this code) | 19.5 | 346 to 520 | | |
| 1 | `8710d47` | CPU reference forward pass | 2.2 | | | matches llama.cpp top-10 and 24 greedy tokens |
| 2 | `fb5452a` | HIP kernels for the matrix products, rest on CPU | 13.5 | | | equals CPU reference |
| 3 | `99a05a9` | Whole decode step on the GPU, one round trip per token | 17.0 | | | all 48 layer visits within 2.5e-4 of CPU reference |
| 4 | `21fe983` | Loop controllers and router spread across the GPU | 20.9 | | | same |
| 5 | `f4bc950` | Batched engine over (session, position) rows | 19.4 | 44 | 39 | batched equals unbatched |
| 6 | `5d45dd2` | Quality harness against llama.cpp | | | | PPL 7.793 vs 7.778, KLD 0.0021, top token 98.19% |
| 7 | `2910ff8` | Batched projections read inputs column-major | 19.4 | 49 | | identical logits |
| 8 | `7cf1feb` | Expert-major kernels (NVFP4): one decode per expert per step | 18.5 | 76 to 83 | 46.5 | identical logits; harness unchanged |
| 9 | `bf62ba6` | Expert-major kernels for the BF16 expert layers | 18.5 | 109 | 49.0 | identical logits |
| 10 | `7aa53c5` | 8-bit integer products for NVFP4 experts in batched steps | 18.5 | 130 | 56.2 | PPL 7.799, KLD 0.0020, top token 97.99% |
| 11 | `1144caa` | Single-row matrix kernel; integer NVFP4 experts with bitwise decode for small steps | 23.7 | 133 | | one row per step: PPL 6.592 vs 6.573, KLD 0.0021, 98.43% (510 positions) |
| 12 | `93172b8` | Attention kernel for up to 1,024 positions: a thread per position | 25.0 (short context), 23.3 (448 tokens of context; was 18.7) | 145 | | equals the previous kernel to 1e-5 on the float path |
| 13 | `f92929d` | Attention over long contexts: chunked kernel, full-attention cache laid out per head | 25.0 short; 21.3 / 19.6 / 16.4 / 13.3 at 4K / 16K / 50K / 100K (was 2.0 at 100K) | 145 short; 115 / 78 / 27 at 4K / 16K / 100K | | 4,096-token context vs llama.cpp: PPL 7.496 vs 7.483, KLD 0.0017, top token 97.56% |
| 14 | `da803a4` | Tentative steps: a run of rows can be computed, then only its first rows kept (groundwork for speculative decoding; no drafter yet) | 24.8 plain; with drafts supplied and no drafter cost: 44.0 (3 correct drafts per step), 66.6 (7), 36.7 (3, a third of blocks wrong) | | | verified output equals plain generation, including after rejected drafts (on the short test prompt of that step; step 15 found a prose prompt where the integer kernels of 8-row and single-row steps disagree) |
| 15 | `1aeb4a0` | DSpark drafter, host prototype: the engine exports the four hidden states it reads; drafts are verified in one tentative step | plain 24.5. Tokens per step with 7 drafts: 4.38 and 2.46 on two coding chat prompts (llama.cpp's own drafter: 4.39 and 2.58), 2.0 to 2.5 on encyclopedia prose (llama.cpp: 2.1 to 2.2). Not a speed result yet: the drafter runs on the CPU | | | drafted output equals plain generation on both coding prompts; on prose only with float experts or the per-row integer kernel (see DSPARK.md) |

Notes on reading this table:

- Step 5 made one session slightly slower because the kernels became general over the batch;
  step 11 took that back.
- The integer products (steps 10 and 11) are the only steps that change the arithmetic. The
  weights stay exact; the inputs to the NVFP4 expert products are rounded to 8 bits per block of
  16. `INKLING2_INT8_EXPERTS=0` restores float products everywhere. With them on, small
  differences elsewhere in the computation can move a logit by about 0.05, where float paths
  agree to about 0.00001.
- Context length matters and most rows above are at short context. Step 13 is the first with
  numbers at depth; see "Context depth" in BENCHMARKS.md, including the comparison with Qwen.
