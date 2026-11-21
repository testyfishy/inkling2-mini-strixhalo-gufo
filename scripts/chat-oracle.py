#!/usr/bin/env python3
"""Renders test conversations with the model's own chat template (Jinja2), to
compare with the extension's C++ rendering.

    chat-oracle.py TEMPLATE.jinja CASES.json OUT_DIR

Writes OUT_DIR/case-N.ref.txt, and reports whether OUT_DIR/case-N.txt (written
by the extension with INKLING2_CHAT=CASES.json INKLING2_CHAT_OUT=OUT_DIR) is
the same text.

The environment is the one Hugging Face's apply_chat_template builds:
trim_blocks, lstrip_blocks, a `tojson` that does not escape for HTML, and
`raise_exception`. The template's {% generation %} tags only mark which part
is the assistant's and are removed.
"""
import json
import os
import re
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def main():
    source = open(sys.argv[1], encoding="utf-8").read()
    source = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", source)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(source)
    cases = json.load(open(sys.argv[2], encoding="utf-8"))
    out_dir = sys.argv[3]
    bad = 0
    for n, case in enumerate(cases):
        args = {"messages": case["messages"]}
        if "tools" in case:
            args["tools"] = case["tools"]
        if "reasoning_effort" in case:
            args["reasoning_effort"] = case["reasoning_effort"]
        ref = template.render(**args)
        open(os.path.join(out_dir, "case-%d.ref.txt" % n), "w", encoding="utf-8",
             newline="").write(ref)
        try:
            got = open(os.path.join(out_dir, "case-%d.txt" % n), encoding="utf-8",
                       newline="").read()
        except OSError:
            got = None
        if got == ref:
            print("case %d: same text (%d characters)" % (n, len(ref)))
            continue
        bad += 1
        print("case %d: DIFFERENT" % n)
        if got is not None:
            k = next((i for i in range(min(len(got), len(ref))) if got[i] != ref[i]),
                     min(len(got), len(ref)))
            print("  template: %r" % ref[max(0, k - 60):k + 80])
            print("  ours    : %r" % got[max(0, k - 60):k + 80])
    print("%d of %d cases differ" % (bad, len(cases)))


main()
