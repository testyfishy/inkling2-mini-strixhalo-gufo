#!/usr/bin/env python3
"""Multiple-choice benchmark suites against an OpenAI-style chat endpoint.

One request loop for three suites, each with its official dataset, prompt
and answer-extraction rule, so that two models are asked exactly the same
thing and scored the same way:

  mmlu-pro   TIGER-Lab/MMLU-Pro, 5-shot chain of thought per category
             (prompt and extraction of evaluate_from_api.py in the benchmark's
             repository)
  gpqa       GPQA Diamond, zero-shot (prompt, option shuffling with seed 0 and
             extraction of OpenAI's simple-evals gpqa_eval.py)
  longbench  LongBench v2, zero-shot template and extraction of the benchmark's
             pred.py; by default the "short" and "medium" length classes

Where this differs from the official scripts, on purpose and for both models
alike: the model's own sampling defaults are used unless --temperature is
given; the token limit is generous (--max-tokens, default 32768) because both
models think before they answer, and only the answer text is scored, not the
reasoning; a long document is cut in the middle by characters (--max-chars)
instead of by a model-specific tokenizer.

    run_mcq.py SUITE --base-url http://127.0.0.1:8080/v1 --model NAME \
        --out results/NAME [--limit N] [--parallel 2] [--dry-run]

Results: OUT/SUITE.jsonl (one line per question; a rerun continues where the
file ends) and OUT/SUITE.summary.json.
"""
import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import sys
import threading
import time

SRC = os.environ.get("EVALS_SRC", "/evals/src")
DATA = os.environ.get("EVALS_DATA", "/evals/data")


def snapshot(repo):
    """The directory of a dataset that prepare.sh downloaded (no network)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, repo_type="dataset", local_files_only=True)


# ---- MMLU-Pro ---------------------------------------------------------------

def mmlu_pro_items(limit, seed):
    import pandas
    root = snapshot("TIGER-Lab/MMLU-Pro")
    data = {split: pandas.read_parquet(
                os.path.join(root, "data", split + "-00000-of-00001.parquet")
            ).to_dict("records")
            for split in ("test", "validation")}
    head = open(os.path.join(SRC, "mmlu-pro/cot_prompt_lib/initial_prompt.txt"),
                encoding="utf-8").read()

    def clean(rows):
        by_category = {}
        for row in rows:
            row = dict(row)
            row["options"] = [o for o in list(row["options"]) if o != "N/A"]
            by_category.setdefault(row["category"], []).append(row)
        return by_category

    def example(question, options, cot=""):
        if cot == "":
            cot = "Let's think step by step."
        if cot.startswith("A: "):
            cot = cot[3:]
        text = "Question: {}\nOptions: ".format(question)
        for i, option in enumerate(options):
            text += "{}. {}\n".format("ABCDEFGHIJ"[i], option)
        return text + "Answer: " + cot + "\n\n"

    test, shots = clean(data["test"]), clean(data["validation"])
    rng = random.Random(seed)
    items = []
    for category in sorted(test):
        rows = test[category]
        if limit:  # the same number from every category
            per = max(1, limit // len(test))
            rows = rng.sample(rows, min(per, len(rows)))
        prefix = head.replace("{$}", category) + "\n"
        for shot in shots[category]:
            prefix += example(shot["question"], shot["options"], shot["cot_content"])
        for row in rows:
            items.append({
                "id": "mmlu-pro-%s" % row["question_id"],
                "group": category,
                "prompt": prefix + example(row["question"], row["options"]),
                "gold": row["answer"],
            })
    return items


def mmlu_pro_extract(text):
    match = re.search(r"answer is \(?([A-J])\)?", text)
    if match:
        return match.group(1)
    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else None


# ---- GPQA Diamond -----------------------------------------------------------

GPQA_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


def gpqa_items(limit, seed):
    import pandas
    rows = pandas.read_csv(os.path.join(DATA, "gpqa_diamond.csv")).to_dict("records")
    rng = random.Random(0)  # as simple-evals
    rows = [row | {"permutation": rng.sample(range(4), 4)} for row in rows]
    if limit:
        rows = random.Random(seed).sample(rows, min(limit, len(rows)))
    items = []
    for n, row in enumerate(rows):
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        choices = [choices[i] for i in row["permutation"]]
        items.append({
            "id": "gpqa-%s" % row.get("Record ID", n),
            "group": row.get("High-level domain", "all"),
            "prompt": GPQA_TEMPLATE.format(Question=row["Question"], A=choices[0],
                                           B=choices[1], C=choices[2], D=choices[3]),
            "gold": "ABCD"[choices.index(row["Correct Answer"])],
        })
    return items


def gpqa_extract(text):
    match = re.search(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?", text)
    return match.group(1).upper() if match else None


# ---- LongBench v2 -----------------------------------------------------------

def longbench_items(limit, seed, lengths, max_chars):
    data = json.load(open(os.path.join(snapshot("THUDM/LongBench-v2"), "data.json"),
                          encoding="utf-8"))
    template = open(os.path.join(SRC, "longbench/prompts/0shot.txt"),
                    encoding="utf-8").read()
    rows = [row for row in data if row["length"] in lengths]
    if limit:
        rows = random.Random(seed).sample(rows, min(limit, len(rows)))
    items = []
    for row in rows:
        context = row["context"].strip()
        cut = len(context) > max_chars
        if cut:  # keep both ends, as the official script does by tokens
            context = context[:max_chars // 2] + context[-(max_chars // 2):]
        prompt = (template.replace("$DOC$", context)
                  .replace("$Q$", row["question"].strip())
                  .replace("$C_A$", row["choice_A"].strip())
                  .replace("$C_B$", row["choice_B"].strip())
                  .replace("$C_C$", row["choice_C"].strip())
                  .replace("$C_D$", row["choice_D"].strip()))
        items.append({"id": "longbench-%s" % row["_id"],
                      "group": "%s/%s" % (row["length"], row["difficulty"]),
                      "prompt": prompt, "gold": row["answer"], "cut": cut})
    items.sort(key=lambda item: len(item["prompt"]))  # short ones first
    return items


def longbench_extract(text):
    text = text.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", text)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", text)
    return match.group(1) if match else None


EXTRACT = {"mmlu-pro": mmlu_pro_extract, "gpqa": gpqa_extract,
           "longbench": longbench_extract}


# ---- running ----------------------------------------------------------------

def wilson(correct, total):
    if total == 0:
        return [0.0, 0.0]
    z, p = 1.96, correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [round(centre - half, 4), round(centre + half, 4)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=sorted(EXTRACT))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="questions (0: all)")
    parser.add_argument("--seed", type=int, default=20261122, help="for --limit")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--lengths", default="short,medium", help="longbench classes")
    parser.add_argument("--max-chars", type=int, default=440000,
                        help="longbench: longest document, in characters")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.suite == "mmlu-pro":
        items = mmlu_pro_items(args.limit, args.seed)
    elif args.suite == "gpqa":
        items = gpqa_items(args.limit, args.seed)
    else:
        items = longbench_items(args.limit, args.seed, set(args.lengths.split(",")),
                                args.max_chars)
    groups = {}
    for item in items:
        groups[item["group"]] = groups.get(item["group"], 0) + 1
    print("%s: %d questions in %d groups; prompts of %d to %d characters" % (
        args.suite, len(items), len(groups),
        min(len(i["prompt"]) for i in items), max(len(i["prompt"]) for i in items)))
    if args.dry_run:
        print(json.dumps(groups, indent=1, sort_keys=True)[:1500])
        print("--- first prompt, last 600 characters, and its answer")
        print(items[0]["prompt"][-600:])
        print("gold:", items[0]["gold"])
        return

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "none"),
                    timeout=args.timeout, max_retries=2)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.suite + ".jsonl")
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            record = json.loads(line)
            done[record["id"]] = record
    todo = [item for item in items if item["id"] not in done]
    print("%d already answered, %d to go" % (len(items) - len(todo), len(todo)))
    lock = threading.Lock()
    out = open(path, "a", encoding="utf-8")

    def ask(item):
        request = dict(model=args.model, max_tokens=args.max_tokens,
                       messages=[{"role": "user", "content": item["prompt"]}])
        if args.temperature is not None:
            request["temperature"] = args.temperature
        started = time.time()
        record = {"id": item["id"], "group": item["group"], "gold": item["gold"]}
        try:
            reply = client.chat.completions.create(**request)
            choice = reply.choices[0]
            text = choice.message.content or ""
            record.update(pred=EXTRACT[args.suite](text), finish=choice.finish_reason,
                          answer=text[-2000:],
                          usage=reply.usage.model_dump() if reply.usage else None)
        except Exception as error:  # recorded, and asked again on a rerun
            record.update(error=repr(error)[:500])
        record["seconds"] = round(time.time() - started, 2)
        record["correct"] = record.get("pred") == item["gold"]
        with lock:
            if "error" not in record:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                done[item["id"]] = record
            n = len(done)
            right = sum(1 for r in done.values() if r["correct"])
            print("%d/%d  %.1f%% so far  %s  %s -> %s (%s)  %.0f s" % (
                n, len(items), 100.0 * right / max(n, 1), item["id"], item["gold"],
                record.get("pred"), record.get("finish", record.get("error", ""))[:40],
                record["seconds"]), flush=True)

    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(ask, todo))
    out.close()

    records = [done[item["id"]] for item in items if item["id"] in done]
    by_group = {}
    for record in records:
        g = by_group.setdefault(record["group"], [0, 0])
        g[0] += record["correct"]
        g[1] += 1
    right = sum(r["correct"] for r in records)
    summary = {
        "suite": args.suite, "model": args.model, "questions": len(items),
        "answered": len(records), "correct": right,
        "accuracy": round(right / max(len(records), 1), 4),
        "interval_95": wilson(right, len(records)),
        "no_answer_found": sum(1 for r in records if r.get("pred") is None),
        "stopped_by_token_limit": sum(1 for r in records if r.get("finish") == "length"),
        "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in records),
        "prompt_tokens": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in records),
        "seconds": round(sum(r["seconds"] for r in records), 1),
        "groups": {g: {"correct": c, "questions": n, "accuracy": round(c / n, 4)}
                   for g, (c, n) in sorted(by_group.items())},
        "settings": {"max_tokens": args.max_tokens, "temperature": args.temperature,
                     "limit": args.limit, "seed": args.seed},
    }
    json.dump(summary, open(os.path.join(args.out, args.suite + ".summary.json"), "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("suite", "model", "answered", "correct", "accuracy",
                                              "interval_95", "no_answer_found",
                                              "stopped_by_token_limit")}))


if __name__ == "__main__":
    sys.exit(main())
