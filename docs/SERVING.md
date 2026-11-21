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
- **Thinking off** (`reasoning_effort: "none"`). The template writes effort 0 for it, and this
  model opens a thinking block anyway. The runner therefore enforces it, in two ways, each for
  the case where it was seen to work: without tools the prompt opens the answer itself; with
  tools the prompt is left alone, so the model can start with a call, and only the token that
  opens a thinking block is never sampled. (Blocking that token without tools made the model
  write its deliberation into the answer.)
- **Tool calls.** The model writes
  `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>`;
  the server sees `<tool_call>{"name":"NAME","args":{...}}</tool_call>`, a syntax registered as
  GUFO tool dialect `inkling2` with its scanner (`tools.cpp`).
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
  distribution as always and the draft is kept only if it is the token drawn, so the output is
  distributed as without drafts. Off with `INKLING2_SERVE_LOOKUP=0`.

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

| Output | Tokens | Without drafts | With drafts |
|---|---:|---:|---:|
| A 556-token passage repeated verbatim | 556 | about 24 | 92.6 |
| The same, sampled at temperature 1.0 (one run) | 556 | about 24 | 76.7 |
| 6 KB of C++ rewritten with one identifier changed | 1,500 | about 24 | 85.5 |
| A new story (nothing to copy) | 300 | 23.9 | 23.9 |
| An article continued | 300 | 22.6 | 22.6 |

Lowering the 12-token threshold to 6 or 3 (`INKLING2_SERVE_LOOKUP_MIN`) moved these by a few
percent in both directions (copy 97 and 99, rewrite 86 and 90, article 22.4 and 22.2), so
GUFO's value stays.

Functional checks over HTTP: plain and streamed chat, reasoning at default and low effort,
thinking off, a tool call as the first action, two parallel calls, a streamed call, an answer
from a tool result, a JSON response format, and cache reuse within and across requests. The
quality harness after these changes: KLD 0.0015 at 4,096 tokens, unchanged.

### Not done, or known to be weak

- `gufo prompt` and `gufo chat` reach a runner without drivers only in GUFO revisions newer than
  the one in the container used here, which answers "unsupported model architecture".
- `tool_choice: "required"` and strict tools go through GUFO's grammar, which the model's
  framing does not fit: it wrote several sentences before the call (8.5 s for one call).
- GUFO keeps checkpoints at the end of a prompt and at a few pass boundaries inside it. A
  conversation that only grows resumes from its end; changing the end of a long prompt
  reprocesses from the last kept boundary (26,300 tokens in the 100K test).
- Several sessions are served one after another: the engine's batched decode across sessions
  is not wired to the server yet.
- The 525 MB of private state per checkpoint could shrink to about a third by saving only the
  live part of each ring.
- Not tested: the disk cache, log-probabilities, reasoning budgets, cancellation inside a
  step, GUFO's functional conformance suites, image input (not implemented).
- The sampled copy figure is one run; acceptance of drafts when sampling was not studied.
