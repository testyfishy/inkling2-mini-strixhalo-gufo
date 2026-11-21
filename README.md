# Inkling 2 Mini on Strix Halo with GUFO

Work in progress: a [GUFO](https://github.com/gufo-org/gufo) extension (`gufo-extension`) that runs
Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on AMD
Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S `gfx1151`, 128 GB unified memory) at native weight
precision.

**Status: no GUFO runner exists yet.** What exists today is the baseline, the tooling and the plan.

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
- [ ] M4c: speculative decoding with the model's DSpark drafter. Done so far: tentative steps
      in the engine, and a CPU prototype of the drafter whose acceptance matches llama.cpp's
      (2.0 to 4.4 tokens per step depending on the text; [docs/DSPARK.md](docs/DSPARK.md)).
      Still to do: the drafter on the GPU, and cheaper extra rows in a verification step
- [ ] M3c: tokenizer and chat template, checked against the reference
- [ ] M4: prefill and decode through GUFO's text-runner interface (`gufo prompt`, `gufo bench`)
- [x] M5a: the model's tokenizer and chat template in the extension, checked against
      independent references ([docs/SERVING.md](docs/SERVING.md))
- [ ] M5: a runner for `gufo serve` (API, streaming, sampling) with the continuation cache, so
      conversations resume without reprocessing
- [ ] M6: kernel work on the hot path, each change checked against M3's logit comparison
- [ ] Later: vision (mmproj), DSpark speculative decoding, tool calls, multi-session serving

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
