# State and specification

Read this first. It says what the project is held to, where it stands, what has already been
tried (so it is not tried again), and what is open. The other documents hold the detail:
[PROGRESS.md](PROGRESS.md) is the step-by-step log, [BENCHMARKS.md](BENCHMARKS.md) the
measurements, [QUALITY.md](QUALITY.md) the checks against llama.cpp, [DSPARK.md](DSPARK.md) the
drafter, [SERVING.md](SERVING.md) the server.

**Keep it current:** every step that changes speed, arithmetic or the list of things tried
updates sections 2, 5, 6 or 7 here in the same commit. Last updated at step 39 (2026-11-23),
when work was paused.

## 1. What the project is held to

**Goal.** The fastest prompt processing and generation for Inkling 2 Mini on a Strix Halo
(Ryzen AI MAX+ 395, Radeon 8060S `gfx1151`, 128 GB) through a GUFO extension, without changing
the model.

**"Lossless" here means:**

- The weights are the released ones: NVFP4 experts, BF16 projections and output matrix. No
  requantization, no pruning, no skipped layers, no approximated attention.
- Arithmetic may differ from llama.cpp only in ways that the quality check cannot see: order of
  summation, FP16 copies of values that FP16 holds exactly, inputs of some products rounded
  (section 3 lists every such place and its switch).
- Sampling draws from the model's distribution exactly; drafts are always verified by the
  model, so a wrong draft costs time, never a token.
- A request running alone repeats: same text from an empty session, a RAM checkpoint, a disk
  checkpoint and a new server, greedy or seeded.

**The gate every step must pass before it is committed:**

| Check | Passes when |
|---|---|
| Mean KLD against llama.cpp, 4,096-token chunk, 2,047 scored positions, at the row counts the change touches | within the band below |
| Repeatability test (greedy and seeded sampled, drafter on and off) | every text identical |
| For a change in sampling | chances against GUFO's sampler agree (`INKLING2_FAST_DRAW_CHECK=1`) |

KLD band at step 39: 0.00147 at 256 rows per step, 0.00153 at 1 row, 0.00166 to 0.00170 at 2
to 4 rows. At 32,768 tokens: 0.0035 (open question, section 6). A change that moves a figure
by more than about 0.0003 needs an explanation.

**Not decided by the engine:** anything lossy (8-bit projections and the like) is a later
phase that the owner has declined for now. Nothing is deployed as a service from here.

## 2. Where it stands (step 39)

| | This extension | llama.cpp HIP, same file |
|---|---:|---:|
| Prompt, 4K tokens | 1,267 tokens/s (1,385 with 4,096-row steps) | 346 |
| Prompt, 32K / 100K / 196K / 1.04M | 1,079 / 815 / 597 / 180 | 180 at 32K |
| Generation, plain, greedy or sampled | 31.0 tokens/s | 19.5 |
| Generation with the drafter, code, greedy / sampled | 35.4 / 34.9 | |
| The same after 40K tokens of source, sampled | 29.7 (plain 24.9) | |
| The same after 120K tokens of source, greedy | 26.1 (plain 19.1) | |
| Prose and short answers with the drafter | 29.4 to 29.7 (plain 30.7 to 31.0): a loss of 4% | |
| Text copied from the context (lookup drafts) | 86 to 93 at step 25, not re-measured since | |
| Four requests at once | about 50 in all, 12 to 13 each | |

A same-GPU port of another engine (Strata `ink2`, HIP) reports 32.2 plain; its card reports
1,466 tokens/s for prompts. Both unverified here.

## 3. How a step is computed

Source map: `extension/src/decoder.cpp` (the engine: sessions, caches, the order of a step),
`extension/src/hip/decode.hip` (kernels), `hip/draft.hip` (drafter kernels), `serve.cpp`
(GUFO runner, cache adapter, drafting control, sampling), `reference.cpp` (CPU reference).

Which kernel runs depends on the rows in the step. "Lane layout" is the design of steps 36 to
38: a thread owns values next to each other, reads 16 bytes per load, a wave sums by exchange.

| Part | 1 row | 2 to 4 rows | 5 to 15 rows | 16 rows and more |
|---|---|---|---|---|
| Projections | lane layout, BF16 weights, F32 inputs | lane layout, weights read once for all rows | matrix instruction, straight from memory | matrix instruction, staged in shared memory, 32 to 128 rows wide |
| Output head | lane layout | lane layout | F32 kernel | matrix instruction |
| Experts, NVFP4 | per row, 8-bit integer products | per row, same | grouped by expert, 8-bit integer | matrix instruction per expert tile (tiles listed on the host from 256 rows) |
| Experts, BF16 (4 visits) | per row, F32 (down projection in lane layout) | same | grouped by expert | matrix instruction |
| Router, loop controllers, norms | lane layout, exact (controllers from FP16 weight parts) | same | matrix instruction on values split in two FP16 parts (controllers: one part for the inputs) | same |
| Full attention | per-row kernel over the FP16 cache | one-pass kernel, context in parts | same, to 63 rows | one-pass kernel from 64 rows |
| Sliding-window attention | per-row kernel | per-row kernel under 4 rows per session, batched kernel from 4 | batched kernel | batched kernel to 63 rows, one-pass kernel from 64 |
| Drawing a token | block sums of the softmax made on the device; GUFO's sampler only for requests that set more than a temperature | | | |

**Where values are rounded** (everything else is F32 or exact):

| Place | Rounding | Switch for the exact path |
|---|---|---|
| Inputs of NVFP4 expert products, steps under 16 rows | 8 bits per block of 16 | `INKLING2_INT8_EXPERTS=0` |
| Inputs of projections and experts in matrix-instruction kernels (5 rows and more; experts from 16) | FP16 | `INKLING2_WMMA_MIN_BATCH=0`, `INKLING2_WMMA_EXPERT_MIN_BATCH=0` |
| Keys and values in the cache; queries in batched attention | FP16 (as llama.cpp's cache) | `INKLING2_F32_CACHE=1` |
| Loop controller inputs, 5 rows and more | one FP16 part | `INKLING2_CTRL_LOW_INPUTS=1` |
| 227,565 of 1.1 billion BF16 projection weights under 1e-5, matrix-instruction kernels only | to FP16, error under 3e-8 | as the second row |

Steps of several rows zero the 64 cache positions after their last row (`ClearAhead`); without
it their last bits depended on stale cache contents and requests did not repeat.

## 4. Facts about this machine, measured

Use these before assuming; several earlier beliefs were wrong.

- **Memory:** the lane-layout projections read weights at about 220 GB/s. The expert kernels
  reach about 185. CPU: 16 threads read 102 GB/s alone and 64 GB/s beside a generating GPU,
  which then loses 4%; prompt steps are unaffected.
- **A kernel launch costs 2 us** (1 us to queue), a barrier 0.05 us, launch and wait 8 us, with
  64 bytes copied back 14 us. The 700 launches of a token cost 1.5 ms. Launch latency is not
  the problem; what a kernel does inside is.
- **Load width matters most for memory-bound kernels:** a wave instruction that loads 4 bytes
  per lane moves 128 bytes, one that loads 16 moves 512. That alone took the projections from
  12.0 to 8.7 ms.
- **`__launch_bounds__` is not free:** 1,024 instead of 256 cost a lane kernel 8% (fewer
  registers per thread).
- **A block runs on one compute unit.** A block with about 60 KB of shared memory runs alone
  on its unit (64 KB each) and loses about 3x; keep shared memory at 16 to 32 KB.
- **Matrix instruction (16x16x16 FP16, FP32 sums):** 27e12 multiply-adds/s with operands in
  registers; an operand read costs about one product. Staged designs end at about 8e12
  (experts, projections) and 10e12 (attention).
- **One token reads 5.6 GB of weights:** input projections 1.9, experts 2.1 (8 of 128 per
  visit, different per row), head 0.9, output projection 0.7. Every extra row in a step reads
  its own experts: 10 to 14 ms per row whatever the kernel.
- **Step times, short context:** 1 row 27.9 ms without the head (32.3 with), 2 rows 42.4, 3
  rows 52.4, 4 rows 64.6, 8 rows 90. Profile of one token: input projections 8.7, experts
  11.3, head 3.9, output projection 3.6, the rest about 6.
- **The drafter:** a pass is 6.5 ms (layers 5.0, shortlist product 1.0, slots 0.5); keeping it
  fed costs every step 1.3% even when it is never asked.
- **Routing is uneven:** in a large step 9% of experts get no row and 31% get more than 128.
- **Full attention is dense on real text:** cache reads cannot be skipped losslessly.

## 5. Already tried and not kept

Do not repeat these without a new reason. Each was measured; the section of BENCHMARKS.md
that holds it is in brackets.

**Kernels in general**

| Tried | Result |
|---|---|
| One thread per output value instead of a block or wave per value | 2 to 25x slower [expert-major kernels] |
| Projection kernel (matrix instruction) reading weights straight from memory in large steps | 1,084 to 620 tokens/s [attention blocks] |
| Expert kernel reading its inputs straight from memory | 1,084 to 997 |
| Staging 2 slices instead of 4 (half the shared memory) | 1,084 to 1,007 |
| Reading the next slice or operands early (prefetch) | no gain, twice |
| Fewer kernel launches as a goal in itself | launches cost 1.5 ms of a token [single-token steps] |
| Sums over a 1,024-thread glue block by exchange, 2 barriers instead of 11 | 31.05 against 31.01; an earlier form was slower (22.9 against 24.8) |

**Experts**

| Tried | Result |
|---|---|
| Tiles of 256 pairs, or 64, or 32, instead of 128 | 864, 822, 1,192 ms against 728 |
| Atomic adds into the row's sum instead of contributions and a sum kernel | no gain, and order-dependent |
| The grouped-GEMM staging of headbouyJB/inkling2-gufo (`k_gemm4`) | 721 ms against 703 |
| Padded staged rows; conversion multiply in FP16 | 0.74 and 0.72 s against 0.71 |
| Lane layout for the NVFP4 experts of one row (1 to 16 units per wave) | 31.1 to 31.3 against 31.05, identical output |
| Lane layout for the BF16 up projection of one row | slower, 30.5 against 31.0 (the down projection gained, 31.2, and is kept) |
| Grouping by expert from 4 rows instead of 5 | 67.8 ms against 64.6 at 4 rows |

**Attention**

| Tried | Result |
|---|---|
| A cheaper exponential | no change |
| Ordering blocks so concurrent ones share cache reads | no change |
| 128 rows per block instead of 64 in the one-pass kernel | 790 against 787 at 100K |
| Few-row kernel reading keys and values straight from the cache | slower than what it replaced |
| One-pass kernel over the whole context for a handful of rows | 385 against 168 ms at 100K |
| Single-pass generation kernel in GUFO's dense-model layout | not built: at most 5 of 63 ms at 100K, measured |
| Skipping far cache reads | not lossless: attention is dense |

**Projections of one row (lane layout variants)**

| Tried | Result |
|---|---|
| 2, 4, 8, 16 weight rows per wave instead of 1 | 28.6, 27.6, 26.7, 27.6 against 29.8 |
| 1 to 16 waves per block | no difference |
| The few-row lane kernel at 8 rows instead of the matrix instruction | 94 against 90 ms |

**Drafting**

| Tried | Result |
|---|---|
| One draft per step | does not pay: a second row costs 14 ms |
| Middle drafter layer seeing 4,096 or 2,048 positions instead of 8,192 | not better |
| A fixed cap of 3, 4 or 5 drafts | worse than the gate alone |
| Gates from 0.50 / 0.45 to 0.85 / 0.75 on sampled requests | code within noise of each other; 0.75 / 0.65 helps greedy code |
| Pauses of up to 128 steps instead of 32 | prose 29.8 from 29.4, code 33.4 from 34.9 |
| A controller that reads the clock | made requests unrepeatable |
| Speculative sampling against keeping exact matches | +1 to 4% (kept) |
| The few-row lane kernel for the drafter's own layers | 5.2 against 5.0 ms |

**Method traps that cost time**

- A sampled comparison on one seed (one showed +19%, three showed +3%). Use three.
- Greedy comparisons across builds: a change in the last bits gives a different text, so
  single cells move by a token per second. Compare means or step times.
- Benchmarking with a repeating prompt overstates expert sharing. Use real text.
- Removing code to see what it costs: the compiler removes what depended on it. Use the
  runtime skip switches.
- After the plain step gets faster, everything tuned against its old cost is stale (the
  drafter's gate and cost model were).
- A kernel that takes over "any small product" must check each caller's weight format (the
  drafter's shortlist is FP16 only).
- Check that the build printed "Installed" before testing; a failed build re-tests the old
  module.

## 6. Open, in the order of what is measured to be there

| # | Item | What is known | Possible gain |
|---:|---|---|---|
| 1 | Experts of a one-token step read at 185 GB/s, projections at 220 | Not load width (lane layout gave nothing). Not explained. Candidates not yet tested: the cost of the integer decode per block, the scatter of 8 experts across a 570 MB tensor | up to 1.9 ms of 32 |
| 2 | Exact search for the largest logit on greedy steps (a small copy of the output matrix with error bounds, exact scores for survivors) | Must also return every token within 7 of the largest, which the drafter's shortlist takes each step, or drafts and repeatability change. Greedy only | up to 3.9 ms of 32 |
| 3 | The three glue kernels (mixing over depth, convolutions, KV store) | About 3.2 ms profiled, 1,024-thread single blocks. Barriers are not the cost | 1 to 2 ms |
| 4 | The drafter's pass | Layers take 5.0 ms for about 500 MB of weights (2.5 ms at 220 GB/s). No sub-profile yet | 2 ms per pass |
| 5 | Empty drafter passes on prose | About 130 per 700 tokens. A cheaper first look (one slot before the full block) is not built | most of the 4% loss |
| 6 | A drafted tree instead of a chain; the next draft started before the host has read the step | Ideas from other engines, not built | unknown here |
| 7 | A share of each product on the CPU | 64 GB/s of CPU reading is free during generation; needs a hand-over per product | up to about 25% of memory traffic |
| 8 | Prompt speed | Experts 0.71 s and projections 0.23 s of a 2,048-row step at their plateau; attention bound by products at depth. 4,096-row steps give 1,385 but grow rings and checkpoints | not identified |
| 9 | Host copies of logit rows | Each kept row is still copied twice (about 0.3 ms) | 1% |
| 10 | KLD at 32,768 tokens is 0.0035 against 0.0015 at 4,096 | Level by position from 16K to 32K, the same on both attention paths. Which engine it belongs to is unknown without an exact reference | correctness, not speed |
| 11 | Loop controller inputs in steps of 5 rows and more: one FP16 part (+4.6% prompt speed, KLD unchanged) or two | Owner's call, not made | |
| 12 | Evaluation run (EVALS.md): adapters for LiveCodeBench, Aider polyglot and BFCL not written; nothing run | Waits for the owner | |

A ceiling for plain generation with these weights: 5.6 GB per token at 220 GB/s is 25.5 ms,
39 tokens/s, with nothing else in the step. It is at 31.0.

## 7. How to work on it

1. Find the gap by measurement first: `INKLING2_PROFILE=1` for a one-token step (each figure
   includes a wait of about 6 us per section and visit), `INKLING2_LAUNCH_BENCH=1` and
   `INKLING2_WMMA_BENCH=1` for layouts in isolation, `INKLING2_DRAFT_PROFILE=1` and
   `INKLING2_SERVE_DRAFT_LOG=1` for the drafter. Take the largest measured gap.
2. Put a new kernel behind a switch, with the old one reachable, and measure both in one run.
3. Development driver for speed: `INKLING2_GPU=2 INKLING2_TEXT=<reference file>
   INKLING2_PROMPT_LEN=256 INKLING2_BATCH=<rows per prompt step> INKLING2_REF_GEN=200`.
   Prompt speed at a given `INKLING2_BATCH` gives the time of a step of that many rows.
4. Quality: `INKLING2_KLD=<llama.cpp logits file> INKLING2_KLD_CHUNKS=1 INKLING2_BATCH=<rows>`
   (rows at most 256). The reference comes from `llama-perplexity --kl-divergence-base`
   (`scripts/quality.sh`).
5. Through the server (`scripts/serve-check.sh`, SERVING.md): plain and drafter, greedy and
   three seeds sampled, short context and 40K; the repeatability sequence; four requests at
   once when batching is touched.
6. Then section 1's gate, the docs (a row in PROGRESS.md, a section in BENCHMARKS.md, this
   file), and the commit.

Switches that select between kept implementations, for A/B runs:

| Switch | Default | Effect |
|---|---|---|
| `INKLING2_WMMA_DIRECT=4` | off | one-row projections on the kernel before step 36 |
| `INKLING2_GEMV_MAX_BATCH` | 4 | largest step on the lane projection kernel (1: only one row) |
| `INKLING2_SMALL_LANE` | 15 | bits: router, controller projection, controller apply, norms |
| `INKLING2_SMALL_MAX_BATCH` | 4 | largest step on those |
| `INKLING2_SPLIT_GEMM_MIN_BATCH` | 5 | smallest step with matrix-instruction controllers and router |
| `INKLING2_MAJOR_MIN_BATCH` | 5 | smallest step with experts grouped by expert |
| `INKLING2_WMMA_MIN_BATCH`, `INKLING2_WMMA_EXPERT_MIN_BATCH` | 2, 16 | smallest steps with the matrix instruction |
| `INKLING2_EXPERT_BF16_LANE` | 2 | bits: up, down projection of BF16 experts in lane layout |
| `INKLING2_ATTN_FAST`, `INKLING2_ATTN_FEW_MIN`, `INKLING2_ATTN_FEW_PARTS` | 63, 2, 32 | attention kernel choice |
| `INKLING2_FAST_DRAW`, `INKLING2_ROW_STATS` | 1, 1 | device sums for drawing; 0 gives GUFO's sampler on every row |
| `INKLING2_DRAFT_FIRST`, `INKLING2_DRAFT_CHAIN` | 0 (built-in 75, 65) | drafter gate in percent |
| `INKLING2_DRAFT_MAX`, `INKLING2_DRAFT_SPEC`, `INKLING2_DRAFT_NEVER` | 7, 1, 0 | drafts per step; speculative sampling; drafter fed but never asked |
| `INKLING2_DRAFT_WINDOW`, `INKLING2_DRAFT_VOCAB` | 8192, 32768 | drafter context and shortlist |
| `INKLING2_SERVE_BATCH_WIDTH` | 8 | sessions decoded in one step |
