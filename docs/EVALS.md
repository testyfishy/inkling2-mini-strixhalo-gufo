# Evaluation plan: is the lossless model the smarter choice?

The question this answers is a practical one: on one Strix Halo machine, is Inkling 2 Mini at
its released weight precision more capable than the larger model that fits only after 4-bit
quantization (Qwen 3.8 Flash-Next, as served by GUFO)? Both are tested as they would be used,
through their OpenAI-style endpoints, with thinking on and their own recommended sampling. Speed
is not part of this; it is in [BENCHMARKS.md](BENCHMARKS.md).

Status: prepared, not run. Runs wait until the engine work is done.

## Suites

Standard public benchmarks, so that the scores can be set beside published ones.

| Area | Suite | Questions | Scoring | State |
|---|---|---:|---|---|
| Knowledge and reasoning | MMLU-Pro (5-shot chain of thought, per category) | 1,000 stratified (of 12,032) | letter match | runner ready, dry-run passes |
| Graduate-level science | GPQA Diamond (zero-shot) | 198 | letter match | runner ready, 2-question check against a live server passes |
| Long context | LongBench v2, "short" and "medium" length classes | 395 (documents of 12K to 110K tokens) | letter match | runner ready, dry-run passes |
| Code generation | LiveCodeBench, code generation, latest release | about 200 | hidden tests | harness and data downloaded; adapter for a local endpoint to write |
| Code editing | Aider polyglot | 225 | the exercises' own tests | harness downloaded; its container to build, endpoint to configure |
| Tool calling | Berkeley Function Calling Leaderboard | about 400 | call match | harness downloaded; endpoint to configure |

The first three run through `evals/run_mcq.py`, one request loop that uses each suite's
official dataset, prompt and answer-extraction rule. The last three use the benchmarks' own
harnesses, because their scoring (running code, matching calls) is the benchmark.

## Rules, the same for both models

- **Thinking on, answer scored.** Both models think before answering. The token limit is
  32,768 per question so that thinking is not cut off, and only the answer text is scored,
  never the reasoning. In the plumbing check one GPQA question used all of an 8,192 limit on
  thinking alone, which is why the limit is this high.
- **Each model's own sampling defaults**, since that is how it is used. One seeded subset per
  suite, the same questions for both models.
- **Noise first.** 200 MMLU-Pro questions are run twice per model; differences between the
  models smaller than the difference between two runs of one model are not differences.
- **Reported per suite**: accuracy with a 95% interval, questions with no extractable answer,
  questions stopped by the token limit, tokens and time. Raw answers are kept.
- **What differs from the official scripts**, stated in `run_mcq.py`: the token limit, the
  sampling defaults, and that a long document is cut in the middle by characters, not tokens.
  Published scores are therefore a reference, not an exact match of method.
- **The Inkling file tested** is the one this project runs. Scores of other files of the model
  may differ.

## How to run

On the test machine, once: `evals/prepare.sh` (clones the harnesses, downloads the datasets,
builds the container `localhost/inkling-evals:1`; about 5.5 GB).

Per model, with its server up (only one model fits at a time):

```sh
RUN="podman run --rm --network=host --security-opt label=disable -e HF_HUB_OFFLINE=1 \
  -v $HOME/inkling-bench/evals:/evals -v /path/to/repo/evals:/code:ro \
  localhost/inkling-evals:1 python /code/run_mcq.py"
$RUN mmlu-pro  --base-url http://127.0.0.1:PORT/v1 --model NAME --out /evals/results/NAME --limit 1000 --parallel S
$RUN gpqa      --base-url http://127.0.0.1:PORT/v1 --model NAME --out /evals/results/NAME --parallel S
$RUN longbench --base-url http://127.0.0.1:PORT/v1 --model NAME --out /evals/results/NAME --parallel 1
```

`S` is the server's session count. A run can be stopped and started again; it continues after
the last answered question. `--dry-run` builds the prompts without contacting a model.

Inkling is served for this with
`gufo serve llm --model <file> --context 131072 --sessions 1 --cache-ram-bytes 8589934592`.

## Expected cost

Rough, from the speeds measured so far (Inkling: about 24 tokens/s generation, one session;
prompt speed as in BENCHMARKS.md) and from thinking lengths seen in two questions, so the
generation part is uncertain by a factor of two or more.

| Suite | Inkling | Note |
|---|---|---|
| MMLU-Pro, 1,000 | 6 to 12 h | about 2,500 prompt tokens and several hundred to a few thousand generated per question |
| GPQA Diamond, 198 | 3 to 8 h | long thinking |
| LongBench v2, 395 | 9 to 12 h | dominated by reading documents: about 60K tokens on average |
| LiveCodeBench, 200 | 3 to 6 h | |
| Aider polyglot, 225 | 8 to 15 h | several turns per exercise |
| BFCL, 400 | 1 to 2 h | |

So one model takes roughly one and a half to two and a half days of machine time for the full
panel, during which the other model is not available. A first pass of MMLU-Pro, GPQA and
LiveCodeBench is about half a day to a day per model.

## Still to do before the run

- Adapters for the three harnesses that run code or match calls (LiveCodeBench, Aider
  polyglot, BFCL): each needs the local endpoint registered and a short check.
- The code suites execute model-written code. They will run inside the evaluation container
  without access to the network or the host's files.
- Decide whether LongBench's "long" class (documents up to several million tokens) is worth
  running in part; it needs contexts beyond what either server is configured for today.
