# The DSpark drafter (working notes)

Inkling 2 Mini ships a small drafter for speculative decoding (`mtp.safetensors`; as GGUF,
`dspark/dspark-Inkling-2-Mini-BF16.gguf` in unsloth/Inkling-2-Mini-GGUF, architecture `dflash`).
These notes are the specification this project implements. They were distilled from the
reference implementation in llama.cpp pull request ggml-org/llama.cpp#32218
(`src/models/dspark.cpp`, `src/models/dflash.cpp`, `common/speculative.cpp`). Items marked
UNVERIFIED could not be read from that source and have to be settled by experiment.

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
  anchor, the same for every row). UNVERIFIED in detail: inferred from comments.
- Layer 1: every cached context position.

RoPE: absolute positions (p for context keys, pos0 + i for block rows), all 128 dimensions,
`theta_j = 1e6^(-2j/128)`, rotation `x0' = x0 cos - x1 sin`, `x1' = x0 sin + x1 cos`.
UNVERIFIED: whether pairs are (2j, 2j+1) or (j, j+64).

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
