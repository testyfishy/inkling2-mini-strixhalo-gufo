# Correctness

The extension carries a plain CPU implementation of the forward pass
(`extension/src/reference.cpp`). It follows llama.cpp's graph operation by operation, reads the
NVFP4 experts directly, and runs one token at a time in F32. It exists to pin down the math before
any GPU code is written, and to be the reference the GPU runner is later compared with.

## Check against llama.cpp, 2026-11-21

- Oracle: llama.cpp mainline with #32704 (commit `2a28df0f9`), HIP backend, same GGUF file
  (native precision: NVFP4 experts + BF16).
- Input: five fixed token ids (`371, 151451, 149879, 151194, 149884`), no tokenizer involved.

### Next-token distribution

| Rank | Token id | llama.cpp logprob | CPU reference logprob |
|---:|---:|---:|---:|
| 1 | 152717 | -1.4921 | -1.4940 |
| 2 | 167815 | -2.7764 | -2.7714 |
| 3 | 149881 | -2.8699 | -2.8987 |
| 4 | 149876 | -3.0370 | -3.0184 |
| 5 | 151107 | -3.1929 | -3.1812 |
| 6 | 145038 | -3.2956 | -3.2818 |
| 7 | 149878 | -3.3133 | -3.3309 |
| 8 | 154706 | -3.3205 | -3.3628 |
| 9 | 25 | -3.4534 | -3.4695 |
| 10 | 151015 | -3.5957 | -3.5420 |

Same ten tokens in the same order; the largest log-probability difference is 0.054.

### Greedy continuation

Both produce the same 24 tokens:

`152717 13 68014 32 13 151663 198 33 13 152486 198 3256 5151 149897 13 151663 198 90048 25 152717 149884 149876 151451 149880`

### Per-layer values

Sums of the attention output, feed-forward output and layer output over the five tokens were
compared for all 48 layer visits against llama.cpp's tensor dump (`llama-eval-callback`). They
agree to within about 1% of the layer output throughout, including the second and third passes
through the loop core.

## What this does and does not show

- It shows the architecture is implemented correctly: loop passes, cross-pass messages, block
  residual mixing, short convolutions, relative-position bias with the attention sink, NVFP4
  decoding with per-expert scales, and the MoE routing.
- The remaining differences are consistent with precision: llama.cpp keeps the KV cache in F16 and
  runs the GPU products at BF16 precision, the reference uses F32 throughout. That attribution is
  an inference, not something measured.
- One short prompt is not a quality evaluation. Long contexts (past the 1,024-token sliding
  window and the 8,192-token log-scale floor), image input and sampling are untested.

One detail worth recording: the relative-position bias and the sink logit are added to the
attention scores unscaled; only the query-key product is multiplied by `1/sqrt(head_dim)`.

## Reproducing

With the extension installed (see the README), set `INKLING2_REF_TOKENS` to comma-separated token
ids and optionally `INKLING2_REF_GEN` to a number of greedy tokens:

```sh
INKLING2_REF_TOKENS=371,151451,149879,151194,149884 INKLING2_REF_GEN=24 \
  gufo prompt --model Inkling-2-Mini-UD-Q8_K_XL.gguf --prompt x
```

The reference runs at about 2 tokens per second on 16 CPU threads.
