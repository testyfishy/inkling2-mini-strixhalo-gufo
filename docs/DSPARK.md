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
