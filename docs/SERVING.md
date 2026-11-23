# Serving: tokenizer, chat template, runner

What is needed to use the model from text instead of token ids, and how each part was checked.
GUFO owns the HTTP server, the OpenAI-style API, scheduling and the prefix cache; the extension
brings the tokenizer, the chat template and a runner that GUFO drives.

## Tokenizer

`extension/src/tokenizer.cpp` reads the vocabulary and merges from the GGUF file and follows
the model's `tokenizer.json`:

- no normalization, and no token added at either end;
- the text is cut into pieces by the tokenizer's splitting rule (the GPT-4o rule: words with an
  optional leading character and an English contraction, digits in groups of up to three,
  punctuation runs, whitespace). The rule is written out by hand with the backtracking a
  regular-expression engine does; the Unicode classes come from `unicode_tables.inc`, generated
  by `scripts/gen-unicode-tables.py` (Unicode 16);
- a piece that is itself a vocabulary entry is one token (`ignore_merges`); otherwise its bytes
  are merged by rank;
- control tokens (`<|message_user|>` and the like) are never produced from plain text. Message
  content that spells one stays ordinary text.

Checked against two references that share no code with it:

| Text | Tokens | Python oracle | llama.cpp |
|---|---:|---|---|
| Stress text: 3,000 random lines over 21 Unicode blocks, contractions, digits, long runs, CRLF | 94,599 | identical | identical |
| C++ source, 205 KB | 72,021 | identical | identical |
| Encyclopedia text, 620 KB | 147,431 | identical | differs in 11 places |
| The stress text with control-token spellings parsed | 94,582 | | identical |

The Python oracle (`scripts/tokenizer-oracle.py`) takes the splitting rule from
`tokenizer.json`, with the Unicode classes written out from Python's tables, and merges the
tiktoken way from the model's tiktoken vocabulary. The 11 differences from llama.cpp are all
one case: a word such as " colleagues" is a vocabulary entry, which the model's tokenizer
(and this one, and the oracle) emits as one token, while llama.cpp builds it from merges and
gets " ", "colleague", "s". llama.cpp does not apply `ignore_merges` for this tokenizer.

Speed: 6 to 16 million tokens per second on one core, so a 100,000-token prompt is tokenized
in about 15 ms.

Not checked: the vendor's own tokenizer library was not available on the test machine, so the
oracle stands in for it. The case-insensitive contractions are matched for ASCII letters (and
U+017F) only.

## Chat template

`extension/src/chat.cpp` is the file's `tokenizer.chat_template` written out in C++: an
optional tool declaration, a system message carrying the thinking effort before the first
message that is not a system message, then each message as role token, content-type token,
content and `<|end_message|>`, with reasoning, tool calls and tool results in the template's
own layout. The template has no generation prompt: the model writes its own
`<|message_model|>` header.

`tests/chat-cases.json` holds seven conversations (plain, system only, reasoning kept in the
history, empty contents, tools with calls and results, content that spells control tokens, a
named assistant). `scripts/chat-oracle.py` renders them with the template itself under Jinja2,
set up as Hugging Face's `apply_chat_template` does. All seven give the same text, character
for character, and the tokens equal tokenizing that text.

Two deliberate differences from running the template and tokenizing its output: control-token
spellings inside message content stay text, and numbers in tool arguments keep the spelling
they arrived with instead of being reformatted.

## Runner

`extension/src/serve.cpp` is the runner GUFO's server drives, so the model is served with

```sh
gufo serve llm --model /models/<file>.gguf --context 131072 --sessions 1 --host 127.0.0.1 --port 8080
```

and answers on GUFO's OpenAI-style API (`/v1/chat/completions`, streaming included). `--context`
defaults to 262,144 tokens; a session's caches take 33 KB per token of context, allocated at
start, so 131,072 tokens is 4.3 GB and the model's full 1,048,576 would be 35 GB.

What the runner does:

- **Prompts** run in passes of up to 2,048 rows (`INKLING2_SERVE_BATCH`).
- **Sampling** is GUFO's sampler on the soft-capped logits of the 200,058 real vocabulary
  entries. The default is the file's own: temperature 1.0, no truncation.
- **Reasoning.** The model frames its output with control tokens
  (`<|message_model|><|content_thinking|>...<|end_message|>`, then
  `<|message_model|><|content_text|>...`). The runner shows the server `<think>...</think>` and
  plain text, so reasoning arrives as `reasoning_content`. `reasoning_effort` selects the
  template's effort level (minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh and max 0.99; the
  template's default is 0.9).
- **Thinking off** (`reasoning_effort: "none"`) does what the template does and no more: the
  prompt says "Thinking effort level: 0". The weights file tested here still opened a thinking
  block at that level (5 of 5 such requests in the trace of one test run), so a client that
  asks for no thinking gets a shorter one. `INKLING2_SERVE_ENFORCE_NO_THINKING=1` makes the
  runner enforce it: without tools the prompt opens the answer itself; with tools only the
  token that opens a thinking block is never sampled, so the model can still start with a
  call. (Blocking that token without tools made the model write its deliberation into the
  answer.) Off by default, because the model answers better when it thinks.
- **Response formats** (`response_format`) keep thinking: a grammar reads the text the server
  sees, so the model thinks first and only the answer is constrained. (With the framing tokens
  read as empty text, as in the first version, the constrained output came out malformed.) The
  one exception is a request that asks for both a response format and no thinking: there the
  prompt opens the answer.
- **Tool calls.** The model writes
  `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>`;
  the server sees `<tool_call>{"name":"NAME","args":{...}}</tool_call>`, a syntax registered as
  GUFO tool dialect `inkling2` with its scanner (`tools.cpp`).
  The dialect tells GUFO's grammar that a call ends with `}</think>`: the model ends a call
  with `<|end_message|>`, which a grammar reads as `</think>` (next point). With the visible
  `</tool_call>` there instead, a second call in one turn derailed into junk text once
  thinking was on.
- **The model's drafter.** `--speculative dspark --dspark-model <drafter file>` loads the
  DSpark drafter that ships with the model; `INKLING2_DRAFTER=<file>` does the same for
  commands without those options. Drafts copied from the context are tried first; when there
  is none, the drafter proposes. How many drafts a step carries and when drafting pauses is
  decided by the runner ([DSPARK.md](DSPARK.md), section 7); the server's draft-length options
  are not used. `INKLING2_SERVE_DRAFT_LOG=1` prints the controller's counts.
- **Several sessions.** With `--sessions N`, requests that generate at the same time share
  one step of the engine (up to 8; `INKLING2_SERVE_BATCH_WIDTH`). Four requests at once ran
  at about 50 tokens/s in all (12 to 13 each) against 31 one at a time.
- **Repeatable.** A request running alone at temperature 0, or sampled with a seed, gives
  the same text whether it starts from an empty session, a RAM checkpoint, a disk checkpoint
  after a restart or a new server, with drafts on or off (checked; BENCHMARKS.md, step 34).
  Drafts on and drafts off are two different, each repeatable, outputs: a step of several
  rows rounds differently from a step of one. For the same reason a request decoded beside
  others is not covered: which sessions share a step depends on when requests arrive.
- **Continuation cache.** GUFO keeps checkpoints of a session and restores them when a later
  request shares a prefix. The adapter describes the engine's state to it: per full-attention
  layer visit and head, keys as one row per position and values as one row per 1,024 positions,
  both shared between checkpoints; and, copied whole into every checkpoint, the block of values
  still being filled, the sliding-window rings, the convolution histories and the next token's
  logits. That private part is 525 MB per checkpoint, mostly the rings.
- **Drafts copied from the context.** A decode step may verify several drafted tokens at once.
  GUFO's prompt lookup proposes them when the last 12 generated tokens (24 when sampling)
  repeat an earlier passage; the first copy carries 4 drafts and the number doubles, up to 16,
  while every draft is kept. At each drafted position the token is drawn from the model's own
  distribution as always and the draft is kept only if it is the token drawn; a draft never
  decides a token. Off with `INKLING2_SERVE_LOOKUP=0`. One qualification: a step of several
  rows takes a different arithmetic path than a step of one row (see the notes in
  [PROGRESS.md](PROGRESS.md)), so logits differ in their last bits and a near-tie can fall the
  other way. In the three comparisons below the text was identical. The same holds for a
  prompt restored from the cache against one processed again.

### Measured through the server, 2026-11-21

One session, greedy unless noted, thinking off unless noted, wikitext prose as long prompts.
Speeds are the server's own figures.

| Request | Prompt tokens processed | Prompt speed | Time to first token | Generation |
|---|---:|---:|---:|---:|
| Short question | 25 | | 0.2 s | 24.0 tokens/s |
| 4K prompt | 3,838 | 1,051 tokens/s | 3.7 s | 22.7 |
| The same request again | 0 (3,838 from cache) | | 0.05 s | 22.8 |
| 4K prompt, other question at the end | 1,797 (2,048 from cache) | 987 | 1.9 s | 22.8 |
| 32K prompt | 31,805 (2,048 from cache) | 804 | 39.6 s | 20.2 |
| 32K prompt, other question at the end | 9,284 (24,576 from cache) | 688 | 13.6 s | 20.0 |
| 100K prompt | 100,021 | 555 | 180 s | 15.5 |
| The same request again | 0 (100,021 from cache, restored in 43 ms) | | 0.13 s | 15.5 |
| 100K prompt, other question at the end | 26,300 (73,728 from cache) | 406 | 65 s | 15.5 |
| Follow-up turn of a short conversation | 18 (25 from cache) | | 0.14 s | 23.9 |

Generation with drafts copied from the context:

| Output | Tokens | Without drafts | With drafts | Text |
|---|---:|---:|---:|---|
| A 556-token passage repeated verbatim | 556 | 23.4 | 92.9 | identical |
| The same, sampled at temperature 1.0 with a fixed seed (one seed) | 556 | 22.1 | 76.2 | identical |
| 6 KB of C++ rewritten with one identifier changed | 1,500 | 23.0 | 85.7 | identical |
| A new story (nothing to copy) | 300 | | 23.9 | no draft was proposed |
| An article continued | 300 | | 22.6 | no draft was proposed |

Lowering the 12-token threshold to 6 or 3 (`INKLING2_SERVE_LOOKUP_MIN`) moved these by a few
percent in both directions (copy 97 and 99, rewrite 86 and 90, article 22.4 and 22.2), so
GUFO's value stays.

Functional checks over HTTP: plain and streamed chat, reasoning at default and low effort,
thinking off, a tool call as the first action, two parallel calls, a streamed call, an answer
from a tool result, a JSON response format (with thinking requested and not), cache reuse
within and across requests, two requests at once on two sessions, and one request each on
`/v1/messages` and `/v1/responses`. The quality harness after the engine changes: KLD 0.0015
at 4,096 tokens, unchanged.

`scripts/serve-requests.py` writes these requests and `scripts/serve-check.sh` posts them.
The script collects what was run here in several separate sessions; it has not been run from
start to end in this form.

### Cache reuse with a large RAM budget and the disk tier, 2026-11-21

The cache is GUFO's; nothing in it is fixed by this extension except what a checkpoint costs:
525 MB of private state, plus 33 KB per token of rows that checkpoints of one conversation
share. GUFO's own policy is dynamic: the RAM budget defaults to half of the free memory (at
most 32 GiB), a request keeps up to four checkpoints on a 2,048-token grid with the last one
within 2,048 tokens of the prompt's end, a point where two prompts diverge is learned and kept
once a second prompt shows it, and under pressure the least useful checkpoints go first.

With `--cache-ram-bytes 34359738368 --cache-disk DIR --cache-disk-bytes 42949672960`:

| Request | From cache | Processed | Time to first token |
|---|---:|---:|---:|
| 32K prompt, cold | 0 | 33,851 | 39.5 s |
| 32K prompt, other question at the end | 32,768 | 1,090 | 2.0 s |
| 100K prompt (shares the first 32K) | 32,768 | 67,253 | 126 s |
| 100K prompt, other question at the end | 98,304 | 1,724 | 4.4 s |
| After a server restart: the 32K request | 33,858 from disk in 0.43 s | 0 | 0.52 s |
| After a server restart: the 100K requests | 100,021 and 100,028 from disk in 0.86 and 0.98 s | 0 | 1.1 s |

So the earlier figures in the table above (2,048 of 3,845, 24,576 of 33,860 and 73,728 of
100,028 tokens from cache) were not the policy's limit: those runs had an 8 to 20 GB budget
that earlier test requests had filled, and the newest grid checkpoints were the ones dropped.
With room, changing the end of a long prompt costs at most one pass of 2,048 tokens. The disk
directory held 19 GB after these requests.

The adapter also gives GUFO this model's measured costs (`Cost()`): 0.94 ms per prompt token
plus 1.35 ms per 100,000 tokens of depth, 90 GB/s to restore from RAM and 4 GB/s from disk.
GUFO documents that estimate as diagnostic for now.

What decides whether reuse pays here: restoring from RAM took 5 to 42 ms and from disk 0.4 to
1.0 s, while reprocessing costs 1 ms per token near the start and 2.3 ms at 100,000 tokens. A
RAM checkpoint therefore pays from a few dozen tokens on, a disk one from about a thousand,
and the deeper the context the more a checkpoint is worth. The 2,048-token grid is GUFO's and
equals this engine's pass size; one pass costs 2 s near the start and 4.7 s at 100,000 tokens.

### Not done, or known to be weak

- `gufo prompt` and `gufo chat` reach a runner without drivers only in GUFO revisions newer than
  the one in the container used here, which answers "unsupported model architecture".
- `tool_choice: "required"` gave a direct call in 1.8 s once the grammar read the framing tokens
  as the text the server sees (one request tested; before that the model wrote several
  sentences first). Strict tool schemas were not tested.
- Each checkpoint's 525 MB of private state limits how many fit: about 60 in the default
  32 GiB. With a small budget the newest checkpoints inside a prompt are dropped first, and
  changing the end of a long prompt then reprocesses from an earlier one (26,300 tokens in the
  first 100K test).
- With several sessions, requests take turns step by step: two at once both finished
  correctly, each slower than alone (4.4 s against 2.4 s for the short one). The engine's
  batched decode across sessions, which would run them in one step, is not wired to the
  server yet.
- The 525 MB of private state per checkpoint could shrink to about a third by saving only the
  live part of each ring.
- Not tested: log-probabilities, reasoning budgets, cancellation inside a
  step, GUFO's functional conformance suites, image input (not implemented).
- Drafts when sampling were measured with one seed on one text; how often they are kept at
  temperature 1.0 on real work was not studied.
