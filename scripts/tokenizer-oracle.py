#!/usr/bin/env python3
"""An independent tokenizer for checking the extension's, standard library only.

It takes the splitting rule from the model's tokenizer.json (the Unicode
classes written out from Python's tables, since `re` has no \\p{..}) and does
the byte-pair merges the tiktoken way from the model's tiktoken vocabulary:
merge the adjacent pair whose joined bytes have the lowest rank.

    tokenizer-oracle.py encode MODEL_DIR TEXT OUT.i32
    tokenizer-oracle.py compare MODEL_DIR TEXT A.i32 B.i32
    tokenizer-oracle.py stress OUT.txt        (writes a hard test text)
    tokenizer-oracle.py list2bin LIST.txt OUT.i32   ("[1, 2, 3]" to integers)
"""
import base64
import json
import os
import random
import re
import struct
import sys
import unicodedata


def class_body(pred):
    out, start = [], None
    for cp in range(0x110000 + 1):
        ok = cp < 0x110000 and not (0xD800 <= cp <= 0xDFFF) and pred(cp)
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append("\\U%08x-\\U%08x" % (start, cp - 1))
            start = None
    return "".join(out)


WHITE = [(0x09, 0x0D), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680),
         (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F),
         (0x3000, 0x3000)]


def build_rule(model_dir):
    rule = json.load(open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8"))
    rule = rule["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    cat = lambda cp: unicodedata.category(chr(cp))
    bodies = {
        "L": class_body(lambda cp: cat(cp)[0] == "L"),
        "N": class_body(lambda cp: cat(cp)[0] == "N"),
        "M": class_body(lambda cp: cat(cp)[0] == "M"),
        "Lu": class_body(lambda cp: cat(cp) == "Lu"),
        "Ll": class_body(lambda cp: cat(cp) == "Ll"),
        "Lt": class_body(lambda cp: cat(cp) == "Lt"),
        "Lm": class_body(lambda cp: cat(cp) == "Lm"),
        "Lo": class_body(lambda cp: cat(cp) == "Lo"),
    }
    white = "".join("\\U%08x-\\U%08x" % r for r in WHITE)
    rule = rule.replace("|\\p{N}{1,3}|", "|[\\p{N}]{1,3}|")
    for name in sorted(bodies, key=len, reverse=True):
        rule = rule.replace("\\p{%s}" % name, bodies[name])
    assert "\\p" not in rule, rule[:200]
    rule = rule.replace("(?!\\S)", "(?![^%s])" % white)
    rule = rule.replace("[^\\s", "[^" + white).replace("\\s", "[%s]" % white)
    return re.compile(rule)


def load_ranks(model_dir):
    ranks = {}
    with open(os.path.join(model_dir, "tiktoken", "tokenizer.model"), "rb") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                ranks[base64.b64decode(parts[0])] = int(parts[1])
    return ranks


def bpe(ranks, piece):
    if piece in ranks:
        return [ranks[piece]]
    parts = [piece[i:i + 1] for i in range(len(piece))]
    while len(parts) > 1:
        best, at = None, -1
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best is None or r < best):
                best, at = r, i
        if best is None:
            break
        parts[at:at + 2] = [parts[at] + parts[at + 1]]
    return [ranks[p] for p in parts]


def encode(model_dir, text):
    rule, ranks = build_rule(model_dir), load_ranks(model_dir)
    ids, cache, end = [], {}, 0
    for m in rule.finditer(text):
        assert m.start() == end, ("gap", end, m.start())
        end = m.end()
        piece = m.group(0).encode("utf-8")
        got = cache.get(piece)
        if got is None:
            got = cache[piece] = bpe(ranks, piece)
        ids.extend(got)
    assert end == len(text), ("unmatched tail", end, len(text))
    return ids


def read_ids(path):
    data = open(path, "rb").read()
    return list(struct.unpack("<%dI" % (len(data) // 4), data))


def stress(path):
    rng = random.Random(20261121)
    blocks = [(0x20, 0x7E), (0xA0, 0x24F), (0x300, 0x36F), (0x370, 0x3FF), (0x400, 0x4FF),
              (0x590, 0x6FF), (0x900, 0x97F), (0xE00, 0xE7F), (0x1100, 0x11FF),
              (0x1E00, 0x1FFF), (0x2000, 0x206F), (0x2070, 0x209F), (0x2150, 0x218F),
              (0x2190, 0x21FF), (0x3000, 0x30FF), (0x4E00, 0x4FFF), (0xAC00, 0xACFF),
              (0xFF00, 0xFFEF), (0x1F300, 0x1F64F), (0x1D400, 0x1D4FF), (0x1F1E6, 0x1F1FF)]
    out = []
    fixed = [
        "Hello, world! It's John's; they'RE we'Ve I'M he'LL she'D don't DON'T can'T\n",
        "aÉ Éa ÉCOLE École écoLE ǅungla ǆ ǅ ǄA ʰabc aʰB 中A A中 中文English混合テスト한국어\n",
        "x = 12345678901234567890; y=1,234.56e-10 0x1F 3.14159 ½ ² ٣٤٥ १२३ ⅷ\n",
        "tabs\t\tand   spaces    \t \n\n\n  \r\n\r\n \r \n trailing   ",
        "\n", "   \n   \n", "a b c　d\u0085e f g​h﻿i\n",
        "é ǟ ́x ́̂ Å กิน कि\n",
        "path/to/file.cpp // comment /* x */ a/b/c\n///\n--- === *** ### $$$ %%% ^^^ &&& |||\n",
        "<|end_message|> <|message_user|><|content_text|> <think> </s> [INST] {{x}} {%y%}\n",
        "snake_case camelCase PascalCase SCREAMING_SNAKE kebab-case __init__ a1b2C3\n",
        "\U0001F600\U0001F1FA\U0001F1F8\U0001F468‍\U0001F469‍\U0001F467 ❤️ © ™\n",
        "'s 't 're 've 'm 'll 'd 'S 'T 'RE 'VE 'M 'LL 'D 'ſ 'x '' ' s\n",
        " " * 300 + "x" + "\t" * 70 + "\n" * 40 + "=" * 500 + "-" * 129 + "\n",
        "ÀÉÎÕÜ àéîõü Straße STRASSE ß ẞ İstanbul ıi Ǆ ǆ ΑΒΓ αβγ ΆΈ άέ АБВ абв Ёё\n",
    ]
    out.extend(fixed)
    for _ in range(3000):
        line = []
        for _ in range(rng.randint(1, 12)):
            kind = rng.random()
            if kind < 0.25:
                word = "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
                               for _ in range(rng.randint(1, 9)))
            elif kind < 0.35:
                word = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 8)))
            elif kind < 0.5:
                word = "".join(rng.choice(" \t\n\r.,;:!?'\"()[]{}<>/\\|-_=+*&^%$#@~`")
                               for _ in range(rng.randint(1, 5)))
            else:
                lo, hi = rng.choice(blocks)
                word = ""
                for _ in range(rng.randint(1, 6)):
                    cp = rng.randint(lo, hi)
                    if unicodedata.category(chr(cp)) not in ("Cn", "Cs", "Co"):
                        word += chr(cp)
            line.append(word)
            line.append(rng.choice(["", " ", " ", "  ", "'s", "\n"]))
        out.append("".join(line) + "\n")
    open(path, "w", encoding="utf-8", newline="").write("".join(out))


def main():
    mode = sys.argv[1]
    if mode == "stress":
        stress(sys.argv[2])
    elif mode == "list2bin":
        text = open(sys.argv[2], encoding="utf-8", errors="replace").read()
        ids = [int(x) for x in re.findall(r"\d+", text[text.rindex("["):])]
        open(sys.argv[3], "wb").write(struct.pack("<%dI" % len(ids), *ids))
        print(len(ids), "ids")
    elif mode == "encode":
        text = open(sys.argv[3], encoding="utf-8", newline="").read()
        ids = encode(sys.argv[2], text)
        open(sys.argv[4], "wb").write(struct.pack("<%dI" % len(ids), *ids))
        print(len(ids), "tokens")
    elif mode == "compare":
        ranks = load_ranks(sys.argv[2])
        back = {v: k for k, v in ranks.items()}
        a, b = read_ids(sys.argv[4]), read_ids(sys.argv[5])
        show = lambda ids: [back.get(i, b"<%d>" % i).decode("utf-8", "backslashreplace")
                            for i in ids]
        if a == b:
            print("identical:", len(a), "tokens")
            return
        # Count the places where the two differ, resynchronizing on bytes.
        i = j = diffs = 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                i += 1
                j += 1
                continue
            i0, j0 = i, j
            la, lb = len(back.get(a[i], b"")), len(back.get(b[j], b""))
            i += 1
            j += 1
            while la != lb and i < len(a) and j < len(b):
                if la < lb:
                    la += len(back.get(a[i], b""))
                    i += 1
                else:
                    lb += len(back.get(b[j], b""))
                    j += 1
            diffs += 1
            if diffs <= 12:
                print("differ at", i0, ":", show(a[max(0, i0 - 2):i]), "vs",
                      show(b[max(0, j0 - 2):j]))
        print("DIFFERENT:", len(a), "vs", len(b), "tokens,", diffs, "places")


main()
