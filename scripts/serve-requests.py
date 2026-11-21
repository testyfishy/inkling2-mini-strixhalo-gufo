#!/usr/bin/env python3
"""Writes the request bodies scripts/serve-check.sh posts, and compares answers.

    serve-requests.py make PROSE.txt CODE.txt OUT_DIR
    serve-requests.py compare OUT_DIR NAME...   (o-NAME-off.json against o-NAME-on.json)

PROSE.txt is any long plain text (the measurements in docs/SERVING.md used the
wikitext-2 test split, 620,000 characters); CODE.txt any source file of 8 KB
or more.
"""
import json
import os
import sys

MODEL = "inkling-2-mini"


def body(messages, max_tokens, effort="none", **extra):
    b = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens, "messages": messages}
    if effort is not None:
        b["reasoning_effort"] = effort
    b.update(extra)
    return b


def make(prose_path, code_path, out):
    text = open(prose_path, encoding="utf-8").read()
    code = open(code_path, encoding="utf-8").read()
    os.makedirs(out, exist_ok=True)

    def save(name, b):
        json.dump(b, open(os.path.join(out, name + ".json"), "w"))

    user = lambda content: [{"role": "user", "content": content}]
    q_sum = "\n\nSummarize the text above in one sentence."
    q_who = "\n\nWhich person is mentioned first in the text above? Answer with the name only."
    save("q1", body(user("What is the capital of France? Answer in one sentence."), 48))
    save("q2", body(user("What is 17 * 23?"), 300, effort="low"))
    for name, chars in (("p4k", 17300), ("p32k", 138000), ("p100k", 420000)):
        if len(text) >= chars:
            save(name, body(user(text[:chars] + q_sum), 40))
            save(name + "b", body(user(text[:chars] + q_who), 24))
    passage = text[200000:202600] if len(text) >= 202600 else text[:2600]
    copy = user("Repeat the following text exactly, with no changes and nothing else:\n\n" + passage)
    save("copy", body(copy, 700))
    sampled = body(copy, 700, seed=11)
    del sampled["temperature"]
    save("copys", sampled)
    save("prose", body(user("Write a 200-word story about a lighthouse keeper who finds a message "
                            "in a bottle."), 300))
    save("edit", body(user("Here is a C++ file:\n\n```cpp\n" + code[2000:8000] + "\n```\n\nOutput the "
                           "same code again, unchanged except that every `std::string` becomes "
                           "`String`. Output only the code."), 1500))
    tools = [
        {"type": "function", "function": {
            "name": "get_weather", "description": "Get the current weather for a city",
            "parameters": {"type": "object", "required": ["city"], "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["c", "f"], "description": "Temperature units"}}}}},
        {"type": "function", "function": {
            "name": "read_file", "description": "Read a text file",
            "parameters": {"type": "object", "required": ["path"], "properties": {
                "path": {"type": "string"}, "max_lines": {"type": "integer"}}}}},
    ]
    ask = user("What is the weather in Paris right now, in celsius?")
    save("t1", body(ask, 200, tools=tools))
    save("t2", body(ask, 400, effort="low", tools=tools))
    save("t3", body(ask + [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\",\"units\":\"c\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21, \"sky\": \"clear\"}"}],
        200, tools=tools))
    save("t4", body(ask, 200, tools=tools, tool_choice="required"))
    save("t5", body(ask, 200, tools=tools, stream=True))
    save("t6", body(user("Read the first 20 lines of /etc/hosts and of /etc/fstab."), 300, tools=tools))
    question = user("Give the city and country of the Eiffel Tower as a JSON object with keys city "
                    "and country.")
    save("j1", body(question, 100, response_format={"type": "json_object"}))
    save("j2", body(question, 400, effort=None, response_format={"type": "json_object"}))
    print("wrote", len(os.listdir(out)), "files to", out)


def compare(out, names):
    for name in names:
        a = json.load(open(os.path.join(out, "o-%s-off.json" % name)))
        b = json.load(open(os.path.join(out, "o-%s-on.json" % name)))
        ta = a["choices"][0]["message"].get("content") or ""
        tb = b["choices"][0]["message"].get("content") or ""
        ua, ub = a["usage"], b["usage"]
        k = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), min(len(ta), len(tb)))
        print("%s: %s; %d and %d tokens, %.1f and %.1f tokens/s" % (
            name, "identical text" if ta == tb else "DIFFERENT text from character %d" % k,
            ua["completion_tokens"], ub["completion_tokens"],
            ua["completion_tokens_per_second"], ub["completion_tokens_per_second"]))


if sys.argv[1] == "make":
    make(sys.argv[2], sys.argv[3], sys.argv[4])
else:
    compare(sys.argv[2], sys.argv[3:])
