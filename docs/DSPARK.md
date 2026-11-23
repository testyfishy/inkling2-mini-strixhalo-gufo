# The DSpark drafter (working notes)

Inkling 2 Mini ships a small drafter for speculative decoding (`mtp.safetensors`; as GGUF,
`dspark/dspark-Inkling-2-Mini-BF16.gguf` in unsloth/Inkling-2-Mini-GGUF, architecture `dflash`).
These notes are the specification this project implements. They were distilled from the
reference implementation in llama.cpp pull request ggml-org/llama.cpp#32218
(`src/models/dspark.cpp`, `src/models/dflash.cpp`, `common/speculative.cpp`). Items marked
UNVERIFIED in an earlier version have since been read from the source and confirmed by
measurement (see "Measured" at the end).

Sizes: width W = 2304, head dimension 128, 24 query heads, 8 key/value heads (query head h uses
key/value head h / 3), 3 layers, FFN 9216, block of 8 rows, Markov rank R = 256, eps 1e-6.
`RMS(x) = x / sqrt(mean(x^2) + eps)`. No biases except where named. Matrices are stored as
n_out rows of n_in values.

## 1. What it reads from the main model

For every position p the main model has processed, four hidden states:

| Name | What |
|---|---|
| f5 | the residual stream entering the 6th executed layer visit (output of visit 4) |
| f23 | the same entering executed visit 23 (second loop pass, physical layer 11) |
| f41 | the same entering executed visit 41 (third loop pass, physical layer 17) |
| f48 | the final hidden state: the output mix over blocks, then `RMS * output_norm` (before the LM head) |

Visits are counted in execution order (0 to 5 prelude, 6 to 17 first pass, 18 to 29 second,
30 to 41 third, 42 to 47 coda), not by cache slot. The first three are raw residuals, the last
is normalized.

Fusion, with `dspark_fuse` a 4 x 4 matrix whose row r is `flat[4r .. 4r+3]`:

```
w_r      = softmax(dspark_fuse[r])              r = 0, 1, 2 (one per drafter layer), 3 (head)
m_r[p]   = sum_k w_r[k] * f_k[p]                k over (f5, f23, f41, f48), features not normalized first
ctx_l[p] = RMS(m_l[p]) * blk.l.dflare_norm      l = 0, 1, 2
f_rnn[p] = RMS(m_3[p]) * dspark_norm
```

## 2. The block

Each draft step runs a block of exactly 8 rows at positions pos0 .. pos0+7, where pos0 is the
position the last accepted token (the anchor) will occupy; the main model has processed
positions up to pos0 - 1.

- Row 0: the anchor token's embedding, `RMS(target.tok_embd[token]) * target.tok_norm`.
- Rows 1 to 7: the same `mask_embd` vector, unnormalized.
- Row i (i >= 1) predicts the token at position pos0 + i. Row 0's output is unused.
- The block is always 8 rows even if fewer drafts are wanted; later rows influence earlier ones.

## 3. A drafter layer

```
r = x
a = RMS(x) * attn_norm
q = Wq a  (24 heads x 128)    k = Wk a  (8 x 128)    v = Wv a  (8 x 128)
q = RoPE(RMS_head(q) * attn_q_norm, pos)        RMS over each head's 128 values;
k = RoPE(RMS_head(k) * attn_k_norm, pos)        one 128-vector of weights shared by all heads
o = attention(q; keys and values = this layer's context cache plus the block's own k, v;
              scale 1/sqrt(128); one softmax over both; no sink, no bias)
r = r + Wo o
f = RMS(r) * ffn_norm
x = r + Wdown( silu(Wgate f) * (Wup f) )
```

After the third layer: `h = RMS(x) * output_norm`, and `base = target.output . h` (the main
model's LM head, no soft cap yet).

Context cache, per layer l and committed position p < pos0:

```
K_ctx[l][p] = RoPE(RMS_head(Wk_ctx ctx_l[p]) * attn_k_ctx_norm, p)
V_ctx[l][p] = Wv_ctx ctx_l[p]                   values are never normalized or rotated
```

Attention is not causal inside the block: every block row sees all 8 block rows. For the
context, layers 0 and 2 are windowed and layer 1 is global:

- Layers 0 and 2: all block rows see context positions pos0-1024 .. pos0-1 (measured from the
  anchor, the same for every row). The loader adds one to the 1,024 window so the anchor row
  reaches 1,024 positions back, and the block shares the anchor's window.
- Layer 1: every cached context position.

RoPE: absolute positions (p for context keys, pos0 + i for block rows), all 128 dimensions,
`theta_j = 1e6^(-2j/128)`, rotation `x0' = x0 cos - x1 sin`, `x1' = x0 sin + x1 cos`.
Pairs are (j, j+64): llama.cpp gives this drafter its NeoX rotation.

## 4. The recurrent head

For draft slot i = 1 .. 7, with `s_0 = 0`, `prev_1` = the anchor token and
`c = f_rnn[pos0 - 1]` (the last position the main model processed, constant over the block):

```
e      = markov_w1[prev_i]                        a row lookup, 256 values
u      = concat(s_{i-1}, e, h_i, c)               256 + 256 + 2304 + 2304 = 5120, in this order
j      = joint_proj . u + joint_proj.bias         768
gate   = sigmoid(j[0:256])   cand = tanh(j[256:512])   latent = tanh(j[512:768])
s_i    = gate * s_{i-1} + (1 - gate) * cand
corr   = dspark_hidden_out . latent + dspark_hidden_out.bias          W
logits = 30 * tanh( target.output . (h_i + corr) / 30 )               first 200,058 entries
prev_{i+1} = argmax(logits), lowest id on ties
```

## 5. Driver

- Up to 7 drafts per step (rows 1 to 7). There is no confidence head, so no early stop.
- Verification is the main model's job: feed the anchor and the drafts as consecutive rows and
  keep the longest prefix where each row's prediction equals the next draft.
- After verification the context caches gain the features of the rows that were kept. Features
  of rejected rows must not stay in the cache.
- Nothing else persists between steps: the block pass and the head are recomputed every step.

## 6. Measured

`extension/src/drafter.cpp` implements the above on the CPU (a prototype to settle correctness
and acceptance, not speed). `INKLING2_DSPARK=<drafter file>` makes the development driver
generate a second time with drafts and compare with plain greedy generation.
`scripts/dspark-oracle.sh` runs llama.cpp's implementation on the same prompt.

Same machine, same native-precision file, greedy, 7 drafts per step unless noted. "Tokens per
step" counts the token the model itself contributes plus the drafts it kept.

| Prompt | Engine | Drafts kept | Tokens per step | Output |
|---|---|---:|---:|---|
| Chat request A: a Python module (81 tokens), 160 generated | this extension | 125 of 259 (48.3%) | 4.38 | equals plain generation |
| | llama.cpp | 122 of 247 (49.4%) | 4.39 | |
| Chat request B: a short Python function (26 tokens), 64 generated | this extension | 38 of 182 (20.9%) | 2.46 | equals plain generation |
| | llama.cpp | 38 of 161 (23.6%) | 2.58 | |
| Encyclopedia prose (256 wikitext tokens), 48 generated | this extension, float experts | 29 of 154 (18.8%) | 2.32 | equals plain generation |
| | this extension, integer experts | 24 of 168 (14.3%) | 2.00 | differs from token 18 (see below) |
| | this extension, integer experts, per-row kernel | 30 of 140 (21.4%) | 2.50 | equals plain generation |
| | llama.cpp | 25 of 144 (17.4%) | 2.14 | |

With fewer drafts per step on chat request A: 3 drafts give 3.24 tokens per step (74.7%
kept), 5 give 3.86 (57.1%).

What this says:

- The drafter is reproduced closely: acceptance is within a few points of llama.cpp's on all
  three prompts, and llama.cpp's plain output on the prose prompt is token for token the same
  as ours. This is a statistical match. Draft blocks were not compared token for token, and
  llama.cpp's drafted runs do not produce exactly the plain output either.
- Acceptance depends heavily on the text, and one good prompt is not the rule: 4.4 tokens per
  step on request A, about 2.5 on request B and on prose. In llama.cpp the drafter turns
  19.1 tokens/s into 25.4 on request A, and into 15.1 and 12.4 on request B and on prose,
  which is slower than not drafting.
- Speed here is not yet a result. A verification step of 8 rows costs about 131 ms against
  41 ms for a plain step, so 4.38 tokens per step would be about 33 tokens/s if drafting cost
  nothing; the CPU prototype adds about 128 ms per step. What decides the final number is the
  cost of the extra rows in a verification step (the same cost that limits prompt processing)
  and a GPU drafter whose seven head evaluations do not each read the whole output matrix.
- The integer difference: steps of several rows use the expert-major 8-bit kernel, single-row
  steps the per-row 8-bit kernel, and the two do not give identical results. On the prose prompt an
  8-row verification step picked a different token than the single-row step at token 18 and
  the runs continued differently from there. Sending the 8-row steps through the per-row
  kernel (`INKLING2_MAJOR_MIN_BATCH=64`) or using float products makes the drafted run
  identical to the plain one. How close the two candidates were at that position was not
  measured. So with the default settings drafted generation is not guaranteed to reproduce
  plain generation exactly; it did whenever both ran the same kernels.
- All of this is at short context (at most about 300 tokens). At the 100K+ contexts this
  project is aimed at, each extra row of a verification step costs more and the drafter's
  middle layer attends over the whole history; none of that is measured yet.
- The table above was measured before the FP16 matrix-instruction kernels (PROGRESS.md step
  16), which changed the arithmetic of the verification steps again. With them, on the prose
  prompt, the default settings give a drafted run equal to the plain one (29 of 140 drafts
  kept, 2.45 tokens per step) and forcing the per-row integer expert kernel gives one that
  differs from token 9: the opposite of before. The conclusion stands and is the general one:
  whether a drafted run reproduces the plain run token for token depends on near-ties between
  kernels that round differently, and is not guaranteed. A verification step of 8 rows now
  costs about 118 ms.
- With attention batched for steps of several rows (PROGRESS.md steps 17 and 18), a
  verification step no longer reads the whole cache once per drafted row. At 4,000 tokens of
  real text 7 given drafts verify at 54 to 67 tokens/s against 21 for plain generation, and
  the run reproduces plain generation. At 1,500 tokens of real text it does not, with or
  without the new attention kernel. At 100,000 tokens of faked context it does not either
  once the sliding-window layers are batched (with only the full-attention layers batched it
  did for 24 tokens, at 32.9 tokens/s against 17.8, and not for 48). The attention outputs of
  those verification steps were compared with the per-row kernel directly and differ by 3e-4
  of their size, as everywhere else, so this is the same sensitivity to near-ties as before,
  on a context whose sliding windows are mostly empty; it is not a number to rely on in
  either direction.

## 7. On the device, 2026-11-22

The drafter now runs on the GPU inside the engine (`extension/src/hip/draft.hip`,
`GpuEngine::Draft`) and through `gufo serve llm --speculative dspark --dspark-model <file>`.

What the engine does:

- **Context.** Every step of the main model keeps what the drafter reads of its rows: the
  four hidden states are mixed and projected on the device and stored as FP16 keys and values
  in three rings per session. Layers 0 and 2 keep their 1,024-position window. The middle
  layer, which the file lets see the whole context, is given a window too (8,192 positions,
  `INKLING2_DRAFT_WINDOW`): only proposals depend on it, and it bounds the drafter's memory
  (32 MB per session) and cost at any depth. Prompt speed with this on is unchanged (a
  4,096-token prompt in 3.41 s with it, 3.42 s without).
- **A draft** is one chain of kernels without a host round trip: the block of 8 rows through
  the three layers, then the recurrent head slot by slot. Each slot's token and the drafter's
  own probability of it come back together.
- **The head's correction is precomputed.** `output . (h + hidden_out . latent + bias)` is
  linear, so `output . hidden_out` (200K x 256) is computed once at load and a slot reads that
  instead of the whole output matrix.
- **The head scores a shortlist, not the vocabulary.** 32,768 entries
  (`INKLING2_DRAFT_VOCAB`; 0 = everything): the tokens of the context and, after every step
  of the main model, the tokens within 12 of its best logit (at most 192 per row). A full
  list gives up entries that were not asked for again. Measured before building it, on 400
  generated tokens of three prompts: with the 256 most likely tokens of each earlier step
  listed, the next token was on the list in 99.5% to 99.7% of steps after the first 32, with
  6,000 to 19,000 tokens listed.
- **Verification** is unchanged: a tentative step of the main model, which draws its own
  token at every drafted position; a draft is kept only if it is the token drawn.

Checked against the host prototype of section 6 with the whole vocabulary scored: all seven
proposals identical in every one of 104 steps (two prompts).

Cost of one draft call, short context:

| Version | ms |
|---|---:|
| Products on the F32 kernels, head over the whole vocabulary | 45 |
| Matrix-instruction kernel, 32 batch rows wide, for the layers and the output matrix | 20 |
| Precomputed head product stored by column, as FP16 | 17 (layers 5.0, output matrix 7.4, slots 4.3) |
| Shortlist of 32,768 tokens | 7 (layers 5.3, output matrix 1.2, slots 0.4 to 0.9) |

(The matrix-instruction kernel that reads weights straight from memory, which the engine
uses for steps of up to 16 rows, was slower here than the F32 kernel: 23 ms for the output
matrix against 14.)

What a verification step costs, which is what limits all of this: 40 ms for one row, 55 for
2, 65 for 3, 81 for 4, 94 for 6, 105 for 8 at short context, because every extra row reads
its own experts (2.1 GB). At 100,000 tokens of context: 63, 97, 143 (4 rows) and 157 (8
rows).

### How many drafts, and when not to draft

Acceptance with seven drafts per step, greedy and at the model's default temperature of 1.0
(host prototype, tokens per step): 4.38 and 3.86 on chat request A (a Python module), 2.48
and 1.45 on request B, 2.42 and 1.47 on prose. At temperature 1.0 on ordinary text only one
draft in fifteen is kept, and drafting seven tokens every step would halve the speed.

The drafter's probability of a token predicts whether the model keeps it (request B, greedy,
first draft: 1 of 12 kept below 0.3, 18 of 21 kept above 0.9). So:

- **Gate.** A first draft is tried when its probability is at least 0.5 (0.6 when sampling);
  each further one while the product of the probabilities so far is at least 0.45 (0.5). The
  head stops at the first slot below the gate.
- **Controller.** Every drafted step is compared with what plain steps would have produced in
  the same time, from the clock. While drafting loses, it pauses (8 steps, then 16, then 32)
  and probes again. One step in 64 is plain so the cost of a plain step stays known.

Development driver, fixed gate of 0.6 on every slot, short context (plain: 24.8 tokens/s):

| Prompt | Greedy | Temperature 1.0 |
|---|---:|---:|
| Request A (Python module) | 43.1 | 35.8 |
| Request B (short function) | 31.6 | 20.4 |
| Encyclopedia prose | 31.9 | 20.3 |

Through the server, with gate and controller, thinking on, 700 tokens per request
(`decode_tps` of the server's log, without and with the drafter):

| Request | Greedy | Default sampling |
|---|---:|---:|
| A Python module with tests | 24.0 -> 31.4 | 22.3 -> 33.2 |
| TCP against UDP, concise | 23.9 -> 26.8 | 22.4 -> 24.3 |
| How tides work, 300 words | 23.7 -> 24.6 | 22.4 -> 22.9 |
| Three-paragraph summary of a 33,854-token text | 20.3 -> 20.1 | 19.2 -> 18.4 |
| The same of a 100,024-token text | 15.5 -> 15.6 | 14.9 -> 14.7 |

So: a third to a half faster on code, about a tenth on technical answers, nothing on prose,
and nothing at depth yet, where a verification step of a few rows reads the whole cache once
per row. Prompt speed through the server is unchanged (1,094 and 1,087 tokens/s on the
33,854-token prompt).

The drafter's state (rings and head feature, 32 MB) is one more private part of a cache
checkpoint, so a restored session drafts as it did before.
