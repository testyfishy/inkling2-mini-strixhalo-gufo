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
