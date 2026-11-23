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
| 16 | `8adc66c` | Matrix-instruction kernels (16x16x16 FP16, FP32 sums) for the projections in steps of 2+ rows and for the experts in steps of 16+ rows, staged through shared memory | 24.8 (single-row steps are unchanged) | 482 at 256 rows per step, 553 at 512 (was 145); at depth 258 / 126 / 31 at 4K / 16K / 100K (was 115 / 78 / 27) | 63.4 (was 56.2) | same greedy tokens; PPL 7.796 vs 7.778, KLD 0.0021, top token 97.99% (the previous kernels on the same run: 7.817, 0.0020, 98.24%); 4,096-token context at 256 rows per step: PPL 7.477 vs 7.483, KLD 0.0014, top token 98.19% |
| 17 | `d34635c` | Full attention for several rows of a session at once: one block per (32 rows, key head, 1,024 positions) reads FP16 keys and values once and multiplies with the matrix instruction | not re-measured (single-row steps keep their kernels) | 485 short; at depth 326 / 301 / 206 at 4K / 16K / 100K (was 258 / 126 / 31) | not re-measured (steps of one row per session keep the per-row kernel) | attention output against the per-row kernel on the same steps: relative difference 2e-4 to 3e-4 at 4K real text and at 100K faked; 4,096-token context: PPL 7.481 vs 7.483, KLD 0.0016, top token 97.90% |
| 18 | `72f32b9` | The sliding-window layers through the same kernel | 24.7 | 573 at 256 rows per step, 680 at 512 (was 485, 553); at depth 523 / 470 / 273 at 4K / 16K / 100K | 65.0 | relative difference 2e-4 on all 48 visits; 512-token context: PPL 7.798 vs 7.778, KLD 0.0021, top token 98.09%; 4,096-token context: PPL 7.465 vs 7.483, KLD 0.0016, top token 98.09% (per-row attention on the same build: 7.477, 0.0014, 98.19%) |
| 19 | `d30bca9` | The key/value cache is FP16 only; steps of few rows read it with a per-row kernel (F32 queries, weights and sums) | 25.0 short; 23.7 / 22.8 / 15.4 at 4K / 16K / 100K faked (was 21.3 / 19.6 / 13.3); 15.8 after 102,400 tokens of real text | unchanged at depth (275 at 100K faked); a real 102,400-token prompt averages 355 | 67.7 | attention output against the F32 kernels: relative difference 2e-4; same greedy tokens as the F32 kernels on the test prompt (96 tokens); 4,096-token context with the per-row kernel on every step: PPL 7.484 vs 7.483, KLD 0.0015, top token 98.09% |
| 20 | `5a4eebe` | Prompt steps of up to 4,096 rows: expert tiles of up to 128 pairs, router and loop controllers through the matrix-instruction kernel (values split into two FP16 parts), attention scratch independent of the context | 24.7 | 4,096-token prompt: 1,082 at 2,048 rows per step, 1,146 at 4,096 (was 573 at 256); real 32,768-token prompt 836 on average, real 102,400-token prompt 550 (was 355); at 100K faked 368 | 65.6 | 4,096-token context: PPL 7.485 vs 7.483, KLD 0.0015, top token 97.95%; 512: 7.793 vs 7.778, KLD 0.0021, 97.94% |
| 21 | `1348e16` | Output head of steps with fewer than 16 rows back on the F32 kernel (the matrix-instruction kernel took 21 ms there instead of 5 to 8) | 24.7 | unchanged | 70.6 (2 sessions: 33.1) | same tokens from both head kernels for 3 sessions |
| 22 | `e35ab36` | Drafts copied from the context (prompt lookup) in the development driver: when the last 3 or more tokens occur earlier, what followed them is proposed, up to 15 tokens, and verified in one step; the number of drafts grows while all are kept and backs off after a miss | plain 23.6; text the model copies: 81 (75 at 30K context); a 4,096-token article continued: 30.0 (65 of 87 drafts kept); other prose: 23.1 (9 of 35 kept) | unchanged | | output equal to plain generation in three of the four runs; in the fourth it differs from token 22, at a near-tie between single-row and multi-row steps |
| 23 | `1af86ef` | A session can be saved to bytes and restored (position, caches, histories) | unchanged | a 32,768-token session: saved in 1.1 s (1.6 GB), restored in 0.06 s, against 39 s to process it again | | generation after a restore gives the same tokens as without it (32,768 and 700 tokens of context, session overwritten in between) |
| 24 | `866d80f` | Tokenizer and chat template in the extension (see [SERVING.md](SERVING.md)) | unchanged | unchanged | | tokenizer identical to an independent Python oracle on 314,051 tokens of stress text, code and prose; chat template identical to the model's Jinja template on 7 conversations |
| 25 | `54bbafb` | Runner for `gufo serve`: API, sampling, reasoning and tool calls, the continuation cache, drafts copied from the context (see [SERVING.md](SERVING.md)) | through the server: 24.0 short, 22.6 after 4K, 15.5 after 100K; copied text 92.9 (23.4 without drafts), a code rewrite 85.7 (23.0), same text | through the server: 4K prompt 1,056, 32K 804, 100K 555; a repeated 100K request restores in 43 ms | | KLD 0.0015 at 4,096 tokens, unchanged; functional checks over HTTP listed in SERVING.md |
| 26 | `d67f2f7` | Batched attention: a short path for slices far before the rows (keys and values straight from the cache, maximum recomputed only when exceeded, sums per lane); a depth sweep mode and a micro-benchmark of the matrix instruction in the driver | unchanged | at 100,000 tokens of depth 437 (was 378); a 100,000-token prompt 162 s (was 180) | | old and new path in the same build: identical figures (KLD 0.001525, top token 97.80%); against step 25 the KLD moved from 0.001490, the size of difference any reordering of sums has produced |
| 27 | `fd0b270` | Cache reuse checked with room and with GUFO's disk tier; the adapter reports this model's costs to the cache; thinking as the template intends (response formats keep thinking) | unchanged | changing the end of a 100K prompt: 4.4 s to first token (98,304 tokens from cache); after a server restart a 100K request restores from disk in 0.9 s | | JSON output valid with thinking on, off and unset |
| 28 | `9db3e71` | Attention kernels with a small shared-memory footprint, so several blocks run on a compute unit: a 16 KB variant for the far part of a context, 32 KB for the rest | unchanged | at depth 4K / 16K / 32K / 100K: 1,113 / 981 / 867 / 594 (step 25: 1,068 / 847 / 678 / 378); a 100,000-token prompt in 129 s (was 180) | | KLD 0.001525 at 4,096 tokens in every variant |
| 29 | `8019c1c` | Prompt attention in one pass, with softmax and weights in registers (design of headbouyJB/inkling2-gufo's prefill kernel) and padded shared-memory rows | unchanged | at depth 4K / 32K / 100K / 196K / 524K / 1.04M: 1,189 / 1,024 / 787 / 584 / 312 / 180 (step 25: 1,068 / 678 / 378 / 227 / not run); a 100,000-token prompt in 106 s (was 180), 1,040,000 tokens in 55 min | | KLD 0.001490 at 4,096 tokens (PPL 7.4847 vs 7.4828), 0.002124 at 512 |
| 30 | `5edebc2` | The model's DSpark drafter on the GPU and in the server (`--speculative dspark`): context kept by every step, one chain of kernels per draft, head over a shortlist of 32,768 likely tokens, drafts gated by the drafter's own probability, drafting paused while the clock says it loses (see [DSPARK.md](DSPARK.md)) | through the server, thinking on: code 24.0 -> 31.4 greedy, 22.3 -> 33.2 at default sampling; technical answer 23.9 -> 26.8 and 22.4 -> 24.3; prose 23.7 -> 24.6 and 22.4 -> 22.9; after a 100K prompt 15.5 -> 15.6. One draft call 7 ms (first version 45) | unchanged with the drafter on (4,096 tokens: 3.41 s against 3.42 s) | | device proposals identical to the host prototype's in 104 of 104 steps |
| 31 | `e722b84` | Full attention of steps with 2 to 63 rows through the one-pass kernel with the context in parts (draft verification at depth) | plain unchanged; a step at 100,000 tokens: 2 rows 97 -> 80 ms, 4 rows 143 -> 108, 8 rows 157 -> 129. Through the server with the drafter: code after 40K tokens of source 22.4 -> 36.4 greedy and 18.6 -> 29.7 sampled, after 120K 16.8 -> 25.3 and 14.0 -> 19.1 | unchanged | | KLD 0.001588 at 4,096 tokens with 8 rows per step (the kernels it replaces: 0.001594) |
| 32 | `861583f` | The server decodes the sessions that are generating in one step (GUFO's batched plans) | four requests at once: 39.8 tokens/s in all against 27.6 one at a time (61 s against 88 s) | unchanged | | |
| 33 | `4c6b25b` | Expert kernels launched only for tiles that have rows (routing is very uneven); loop controllers' inputs as one FP16 value | unchanged | at depth 4K / 32K / 100K / 196K: 1,267 / 1,079 / 815 / 597 (step 29: 1,189 / 1,024 / 787 / 584); a 100,000-token prompt in 100.8 s (was 106.3); 1,385 at 4K with 4,096 rows per step | | KLD 0.001469 at 4,096 tokens (PPL 7.4823 vs 7.4828), 0.002102 at 512 |
| 34 | `32f0d01` | The same request gives the same text with the drafter on (controller without the clock, shortlist per session and rebuilt per request, steps of several rows zero the 64 cache positions after their last row); checkpoints store the live window of each ring, not the ring; the logit soft cap on the device | through the server, plain 24.0 -> 25.0 greedy, 22.4 -> 23.4 sampled; with the drafter, code 32.0 greedy and 27.4 sampled; steps with given drafts now keep every draft in the driver's test (74.1 tokens/s with 7 per step) | unchanged (1,287 at 4K) | | KLD 0.001469 at 4,096 tokens; cold, RAM-restored, disk-restored and fresh-server runs give identical text; a checkpoint's fixed part 577 -> 233 MB |
| 35 | `5a8f9ec` | Speculative sampling for sampled requests with the drafter (drafts drawn from the drafter's distribution, kept with probability min(1, p/q)); other implementations read for ideas; CPU headroom measured | at temperature 1.0, three seeds: code 28.4 -> 29.3, code after 40K tokens 27.1 -> 27.4, technical answer 23.0 -> 24.0, prose 22.9 (plain 23.4) | unchanged | | seeded requests repeat exactly from cold, RAM, disk and a new server |
| 36 | (this commit) | Single-row projections: a wave per weight row, 16-byte loads, no shared memory; launch cost measured (2 us) | plain 24.8 -> 29.8 | unchanged | | KLD at one row per step 0.00136 (0.00131 before) |
| 37 | (this commit) | One-token step: router products, loop controllers (FP16 weights, same values) and norms read 16 bytes per load and sum by exchange | plain 29.7 -> 31.0 | unchanged | | KLD at one row per step, 2,047 positions: 0.00153 (0.00170 at step 35) |
| 38 | (this commit) | Steps of 2 to 4 rows on the kernels of steps 36 and 37; drafter gate 0.75 / 0.65 (looser deep in a context) and its cost model on the new step times; drafter shortlist never read as BF16 | plain 31.0 (server 30.9); with the drafter: code 37.6 greedy / 34.2 sampled, code after 40K tokens 40.2, prose 30.3 | unchanged | | KLD at 2 to 4 rows per step 0.0017; repeatability test passes |
| 39 | (this commit) | Tokens drawn with sums the device makes per block of 1,024 logits (requests that only set a temperature); drafted chances and greedy verification from the same statistics; controller charges refused drafts; BF16 expert down projection on the one-row layout | sampled plain 28.5 -> 30.7; with the drafter, sampled: code 34.9, code after 40K tokens 29.7, prose 29.4; after 120K tokens greedy 26.1 (plain 19.1) | unchanged | | chances within 2.3e-12 of GUFO's sampler; repeatability test passes |

Work is paused after step 39 (2026-11-23). BENCHMARKS.md, step 39, ends with what is open.

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
- Step 16 changes the arithmetic of steps with several rows: the inputs of the projections and
  of the experts are rounded to FP16 (11 significant bits) instead of staying FP32 (projections)
  or being rounded to 8 bits (experts). The weights stay exact: BF16 and NVFP4 values are
  representable in FP16, except BF16 values below about 1e-5 in magnitude (227,565 of 1.1
  billion projection weights), which are rounded with an absolute error below 3e-8.
  `INKLING2_WMMA_MIN_BATCH=0` and `INKLING2_WMMA_EXPERT_MIN_BATCH=0` restore the previous
  kernels. Single-row steps (plain generation) still use the earlier kernels, so this step did
  not change generation speed for one session; a step that verifies 7 drafts went from about
  131 to 118 ms. The FP16 copies of the projection weights take about 2.2 GB more on the device.
- Steps 17 and 18 change the arithmetic of attention in steps with at least 4 rows per session
  on average: queries, keys and values are rounded to FP16 (the cache keeps F32 and FP16
  copies), the attention weights are stored as FP16 relative to a running maximum (11
  significant bits down to 2e-9 of it), and sums are FP32. `INKLING2_WMMA_ATTN_MIN_ROWS=0`
  restores the per-row kernels, `INKLING2_WMMA_ATTN_WINDOW=0` only for the sliding-window
  layers; `INKLING2_ATTN_AB=1` runs both kernels on every visit and reports the difference.
  The FP16 copies add about 33 KB per token of context per session (3.3 GB at 100,000 tokens).
  The threshold of 4 rows is not tuned: only 1 row per session (per-row kernel faster) and 8
  rows (batched kernel faster) were measured.
- The depth numbers still come from a faked context (the caches are filled by repetition). With
  real text the batched kernels are checked up to 4,096 tokens.
- Step 19 removes the F32 key/value caches: 33 KB per token of context per session instead of
  98 (3.3 GB instead of 9.8 at 100,000 tokens). `INKLING2_F32_CACHE=1` keeps them, which the
  F32 attention kernels and `INKLING2_ATTN_AB=1` need. From this step the faked context
  repeats the first 1,024 positions of both keys and values (before, values repeated the
  prompt), and the first run on real text at depth exists: 102,400 tokens of wikitext.
- Step 20 changes the arithmetic of the router and of the loop controllers in steps of 2 or
  more rows: their products are computed from FP16 halves, but with every input split into a
  rounded part and the rounding's remainder (and likewise the weights, which here are all
  exact in FP16), summed in FP32, so the result keeps about 22 significant bits.
  `INKLING2_SPLIT_GEMM_MIN_BATCH=0` restores the per-row F32 kernels. Plain generation after a
  prompt changed at one near-tie on the test prompt (token 8 of 96) because the prompt's
  cache contents changed in the last bits; the draft-checking test, which feeds a single-row
  run's tokens to multi-row steps, then stops agreeing at that token. That test is a
  consistency check between two arithmetic paths, not a quality measure; the quality harness
  is unchanged.
