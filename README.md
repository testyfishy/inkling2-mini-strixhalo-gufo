# Inkling 2 Mini on Strix Halo with GUFO

A [GUFO](https://github.com/gufo-org/gufo) extension (`gufo-extension`) that runs
Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on AMD
Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S `gfx1151`, 128 GB unified memory) at native weight
precision.

**Status: it runs and serves; work paused at step 39 (2026-11-23).** `gufo serve llm` answers on
GUFO's OpenAI-style API with this extension, with the weights exactly as released.
Start with [docs/STATE.md](docs/STATE.md): what the project is held to, where it stands, what has
been tried and what is open. Then [docs/SERVING.md](docs/SERVING.md) and
[docs/PROGRESS.md](docs/PROGRESS.md).

## Summary

Same machine, same file (NVFP4 experts + BF16, 21.3 GiB), one session, tokens per second unless
a time is given.

**Against llama.cpp (HIP backend)**

| | llama.cpp | This extension | |
|---|---:|---:|---:|
| Prompt, 4,096 tokens | 346 | 1,267 | 3.7x |
| Prompt, 32,768 tokens | 181.5 s | 27.9 s | 6.5x |
| Prompt, 100,000 tokens | not run | 100.8 s | |
| Generation | 19.5 | 31.0 | 1.6x |
| Generation of code, with the model's drafter | not measured | about 35 | |
| Output agreement with llama.cpp | | KLD 0.0015 at 4K tokens, 0.0035 at 32K; perplexity within 0.1% | |

For scale, llama.cpp with the smaller UD-Q4_K_XL quantization generates at 24.3: the released
weights run faster here than a 4-bit requantization does there.

**Before and after, inside this project**

| Stage | Step | Generation | Prompt, short / 4K / 100K deep |
|---|---:|---:|---:|
| llama.cpp baseline | 0 | 19.5 | 520 / 346 / |
| CPU reference, correct against llama.cpp | 1 | 2.2 | |
| Whole decode step on the GPU | 4 | 20.9 | |
| Batched engine, first prompt steps | 5 | 19.4 | 44 |
| Expert-major kernels, 8-bit expert products | 10 | 18.5 | 130 |
| Single-row kernels, attention for long contexts | 13 | 25.0 | 145 / 115 / 27 |
| FP16 matrix-instruction kernels, batched attention | 18 | 24.7 | 573 / 523 / 273 |
| Prompt steps of up to 4,096 rows | 20 | 24.7 | / 1,082 / 368 |
| One-pass attention, expert tiles listed ahead | 33 | 24.9 | / 1,267 / 815 |
| One-token and few-row kernels rebuilt, tokens drawn with the device's help | 39 | 31.0 | / 1,267 / 815 |

| With the model's drafter, through the server | Step 30 | Step 39 | Plain at step 39 |
|---|---:|---:|---:|
| Code, greedy | 31.4 | 35.4 | 31.0 |
| Code, sampled (three seeds) | about 28 | 34.9 | 30.7 |
| Code after 120K tokens of source, greedy | 25.3 | 26.1 | 19.1 |
| Prose | about plain | 29.4 to 29.7 | 30.7 to 31.0 |

Also built on the way: the tokenizer and chat template, tool calls and reasoning, a continuation
cache with RAM and disk checkpoints (a repeated 100K request restores in 43 ms), drafts copied
from the context (86 to 93 tokens/s on text the model copies), several sessions decoded in one
step (four requests: about 50 tokens/s in all), and requests that repeat exactly.

What is left is in [docs/STATE.md](docs/STATE.md), section 6. With these weights one token reads
5.6 GB, which this machine's memory delivers in about 25 ms: a ceiling near 39 tokens/s for plain
generation, against 31.0 now.

## Goal

- Run Inkling 2 Mini inside GUFO, so it gets GUFO's serving, scheduling and prompt caching.
- Keep the weights at the precision they were released in: the routed experts stay NVFP4, everything
  else stays BF16. No requantization.
- Match the reference implementation (llama.cpp) on output before making any speed claim.

## Why this needs an extension

Inkling 2 Mini is a looped mixture-of-experts transformer (GGUF architecture `inkling2_loop`):
24 layers, with layers 6-17 run three times per token, so each token makes 48 layer visits. GUFO has
no provider for that architecture. A GUFO extension owns loading, tokenization, chat template and
all GPU execution for its model, so this is a new runner, not a configuration change.

## Baseline (llama.cpp on the same machine)

Measured 2026-11-20/21 with `llama-bench`, all layers on the GPU, no speculative drafter.
Details and caveats in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Weights | Backend | Prompt 512 (t/s) | Prompt 4096 (t/s) | Generation (t/s) |
|---|---|---:|---:|---:|
| Native precision (NVFP4 experts + BF16), 21.3 GiB | HIP / ROCm 10.1 | 520 | 346 | 19.5 |
| Native precision (NVFP4 experts + BF16), 21.3 GiB | Vulkan / RADV | 344 | 296 | 15.3 |
| UD-Q4_K_XL, 16.4 GiB | HIP / ROCm 10.1 | 509 | 343 | 24.3 |
| UD-Q4_K_XL, 16.4 GiB | Vulkan / RADV | 421 | 351 | 21.7 |

These numbers are the floor the extension has to beat, not a prediction of GUFO performance.

## Milestones

- [x] Baseline: llama.cpp Vulkan and HIP builds and benchmarks on the target machine
- [x] Native-precision GGUF identified (mainline `inkling2_loop` layout, NVFP4 experts kept as-is)
- [x] Extension build loop proven: `gufo extension install <dir>` builds and loads a module in seconds
- [x] M1: probe and loader — GUFO routes Inkling 2 Mini `inkling2_loop` files to this extension,
      which validates the hyperparameters and every tensor shape (`gufo prompt` reports the
      inventory and then says the runner is missing)
- [x] M2: CPU reference forward pass (`extension/src/reference.cpp`), matching llama.cpp on the
      native-precision file: same top-10 next tokens, log-probabilities within 0.05, identical
      24-token greedy continuation, all 48 layer visits within about 1% (see
      [docs/CORRECTNESS.md](docs/CORRECTNESS.md))
- [x] M3: first GPU decode. The heavy matrix products (BF16 projections, NVFP4 and BF16
      experts, output head) run in HIP kernels; everything else still runs on the CPU. Output
      matches the CPU reference (logits to five digits, identical greedy tokens). Speed is
      13.5 tokens/s, **below** the llama.cpp baseline of 19.5: the time goes to three
      CPU-GPU round trips per layer visit, not to the kernels
- [x] M3b: whole decode step on the GPU with one round trip per token (mixing, loop messages,
      norms, convolutions, attention over a device KV cache, routing). All 48 layer visits match
      the CPU reference to about 2.5e-4 and the greedy tokens are identical. **20.9 tokens/s**
      single stream at native precision, against 19.5 for llama.cpp HIP on the same file
- [x] M4a: batched engine over (session, position) rows: several sessions decode together and
      prompts are fed several tokens per step, with per-session KV caches on the device.
      Verified equal to unbatched runs. 19.4 tokens/s for one session, 44.4 total for 16;
      prompts at 49 tokens/s
- [x] Quality harness: perplexity 7.793 vs 7.778, mean KLD 0.0021 and 98.19% top-token
      agreement against llama.cpp on wikitext-2 (see [docs/QUALITY.md](docs/QUALITY.md))
- [x] M4b: expert-major kernels: each expert's weights are decoded once per step for all the
      rows that selected it. On real text, 8 sessions reach 49.0 tokens/s total (was 37.9) and
      prompts 109 tokens/s (was 47 to 49), with identical output
- [x] M4b2: prompt processing at llama.cpp's rate for a 512-token prompt: 482 tokens/s at 256
      rows per step, 553 at 512 (llama.cpp: 520 at 512 tokens, 346 at 4,096; ours at 4K depth is
      258), with the GPU's FP16 matrix instruction for
      projections and experts. Quality unchanged in the harness (KLD 0.0021 at 512
      tokens of context, 0.0014 at 4,096)
- [x] M4b3: attention for several rows of a session at once, with the same matrix instruction
      and FP16 copies of the cache. Prompt processing at 256 rows per step: 573 tokens/s at
      short context, 523 / 470 / 273 at 4K / 16K / 100K tokens of context (was 258 / 126 / 31).
      Quality in the harness: KLD 0.0021 at 512 tokens, 0.0016 at 4,096 (0.0014 before)
- [x] M4b4: the key/value cache is FP16 only (33 KB per token of context instead of 98) and
      generation reads it: 23.7 / 22.8 / 15.4 tokens/s at 4K / 16K / 100K (was 21.3 / 19.6 /
      13.3). First run on real text at depth: a 102,400-token prompt at 355 tokens/s on
      average, generation after it at 15.8
- [x] M4b5: prompt steps of up to 4,096 rows: 1,082 tokens/s at 2,048 rows per step on a
      4,096-token prompt (was 573), 550 on average over a real 102,400-token prompt (was 355).
      Quality unchanged (KLD 0.0015 at 4,096 tokens)
- [x] M4c2: drafts copied from the context (prompt lookup), in the development driver: 81
      tokens/s on text the model copies (75 at 30K context), 30 on an article continuation,
      23 where nothing repeats (plain: 23.6)
- [x] M4d: a session can be saved and restored (32,768 tokens: 0.06 s to restore instead of
      39 s to process again). A prefix cache on top of it is not built
- [ ] M4b6: what is left for speed with the weights as they are: the batched attention kernel
      at depth (368 tokens/s for prompts at 100K; its matrix instructions run at half the rate
      of the projection kernel's, cause unknown), fewer small kernels per token (about 8 of
      40 ms), a prefix cache, and a serving runner (tokenizer, chat template, API) so that all
      of this is usable outside the development driver
- [x] M4c: speculative decoding with the model's DSpark drafter, on the GPU and in the server
      (`--speculative dspark`): through the server with thinking on, code goes from 24.0 to
      31.4 tokens/s greedy and from 22.3 to 33.2 at default sampling, technical answers gain
      about a tenth, prose and generation after long prompts nothing yet
      ([docs/DSPARK.md](docs/DSPARK.md), section 7)
- [ ] M3c: tokenizer and chat template, checked against the reference
- [ ] M4: prefill and decode through GUFO's text-runner interface (`gufo prompt`, `gufo bench`)
- [x] M5a: the model's tokenizer and chat template in the extension, checked against
      independent references ([docs/SERVING.md](docs/SERVING.md))
- [x] M5: a runner for `gufo serve`: OpenAI-style API with streaming, reasoning and tool
      calls, GUFO's continuation cache (a repeated 100K-token request restores in 43 ms) and
      drafts copied from the context (86 to 93 tokens/s greedy on rewrites and copies, with
      the same text as without drafts). Measured
      through the server: 1,056 tokens/s on a 4K prompt, 555 on a 100K prompt, 24.0 tokens/s
      generation ([docs/SERVING.md](docs/SERVING.md))
- [ ] M7: a standard benchmark panel comparing this model at released precision with a larger
      4-bit model on the same machine: prepared, not run ([docs/EVALS.md](docs/EVALS.md))
- [x] M5c: sessions that generate at the same time share a step (four requests at once: about
      50 tokens/s in all against 31)
- [ ] M5b: what serving still lacks: smaller checkpoints,
      `tool_choice: required` in the model's own framing, the disk cache and GUFO's
      conformance suites
- [ ] M6: kernel work on the hot path, each change checked against M3's logit comparison
- [ ] Later: vision (mmproj)

## Repository layout

- `docs/` — benchmarks and notes
- `scripts/` — reproducible builds of the llama.cpp baselines and the benchmark run
- `extension/` — the GUFO extension (probe and loader so far; no GPU code)

## Building the extension

GUFO's runtime image carries the extension SDK, so the extension builds as a loadable module:

```sh
podman volume create inkling-gufo-ext
podman run --rm --userns=keep-id:uid=1000,gid=1000 --security-opt label=disable \
  -v inkling-gufo-ext:/home/gufo/.local/share/gufo/extensions \
  -v "$PWD/extension":/src/extension:ro \
  ghcr.io/gufo-org/toolboxes/gufo-runtime:edge \
  gufo extension install /src/extension --yes
```

`--yes` skips GUFO's prompt about building code its maintainers have not reviewed; read the
source first.

## Related work

- [hrisheekeshr/inkling2-mini-gufo-extension](https://github.com/hrisheekeshr/inkling2-mini-gufo-extension)
  — a community scaffold and design doc for the same model, targeting requantized GGUFs
  ([gufo-org/gufo#1306](https://github.com/gufo-org/gufo/issues/1306)). This project differs in
  keeping the native NVFP4 experts.
- llama.cpp support: [ggml-org/llama.cpp#32694](https://github.com/ggml-org/llama.cpp/pull/32694)
  plus the loop fix [#32704](https://github.com/ggml-org/llama.cpp/pull/32704).
