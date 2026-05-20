#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import textwrap
import time
import urllib.error
from pathlib import Path

from coding_eval import TASKS as BASE_TASKS
from coding_eval import extract_code, post_chat, run_tests


EXTRA_TASKS = [
    {
        "id": "group_anagrams",
        "prompt": """Implement this Python function:

def group_anagrams(words):
    \"\"\"Group strings that are anagrams of each other.

    Return a list of groups. Words inside each group must preserve their input
    order. The groups must be sorted by the first word that appears in each
    group. Treat uppercase and lowercase as different characters.
    \"\"\"

Return only a single Python code block defining group_anagrams.""",
        "tests": r"""
fn = ns["group_anagrams"]
assert fn(["eat", "tea", "tan", "ate", "nat", "bat"]) == [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
assert fn(["", "b", "", "B"]) == [["", ""], ["b"], ["B"]]
assert fn(["ab", "ba", "abc", "cab", "bac"]) == [["ab", "ba"], ["abc", "cab", "bac"]]
""",
    },
    {
        "id": "flatten_dict",
        "prompt": """Implement this Python function:

def flatten_dict(data, sep="."):
    \"\"\"Flatten nested dictionaries.

    Example: {"a": {"b": 2}, "c": 3} becomes {"a.b": 2, "c": 3}.
    Empty dictionaries are kept as values. Keys must be strings. Raise
    ValueError if sep is empty or if a key is not a string.
    The input must not be mutated.
    \"\"\"

Return only a single Python code block defining flatten_dict.""",
        "tests": r"""
fn = ns["flatten_dict"]
src = {"a": {"b": 2, "c": {"d": 4}}, "e": 5}
orig = {"a": {"b": 2, "c": {"d": 4}}, "e": 5}
assert fn(src) == {"a.b": 2, "a.c.d": 4, "e": 5}
assert src == orig
assert fn({"a": {}, "b": {"c": {}}}) == {"a": {}, "b.c": {}}
assert fn({"x": {"y": 1}}, sep="/") == {"x/y": 1}
for bad in [("", {"a": 1}), (".", {1: "x"})]:
    try:
        fn(bad[1], sep=bad[0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
""",
    },
    {
        "id": "sliding_window",
        "prompt": """Implement this Python function:

def sliding_window(items, size):
    \"\"\"Return consecutive windows from items as tuples.

    If items=[1,2,3,4] and size=3, return [(1,2,3), (2,3,4)].
    If size is larger than the input length, return [].
    Raise ValueError when size <= 0. The input may be any iterable.
    \"\"\"

Return only a single Python code block defining sliding_window.""",
        "tests": r"""
fn = ns["sliding_window"]
assert fn([1, 2, 3, 4], 3) == [(1, 2, 3), (2, 3, 4)]
assert fn(iter([1, 2, 3]), 2) == [(1, 2), (2, 3)]
assert fn("abcd", 2) == [("a", "b"), ("b", "c"), ("c", "d")]
assert fn([1, 2], 3) == []
try:
    fn([1, 2], 0)
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError")
""",
    },
    {
        "id": "valid_brackets",
        "prompt": """Implement this Python function:

def valid_brackets(text):
    \"\"\"Return True if (), [], and {} brackets are balanced.

    Ignore all non-bracket characters. Brackets must be properly nested.
    \"\"\"

Return only a single Python code block defining valid_brackets.""",
        "tests": r"""
fn = ns["valid_brackets"]
assert fn("a(b[c]{d})") is True
assert fn("([{}])") is True
assert fn("([)]") is False
assert fn("(()") is False
assert fn("no brackets") is True
assert fn("}") is False
""",
    },
    {
        "id": "parse_csv_line",
        "prompt": """Implement this Python function:

def parse_csv_line(line):
    \"\"\"Parse one RFC-4180-like CSV record into fields.

    Commas separate fields. Double quotes may wrap a field. Inside quoted
    fields, two consecutive double quotes represent one literal quote. Spaces
    are ordinary characters and must not be stripped. Raise ValueError for an
    unterminated quote or any non-comma character after a closing quote.
    \"\"\"

Return only a single Python code block defining parse_csv_line.""",
        "tests": r"""
fn = ns["parse_csv_line"]
assert fn("a,b,c") == ["a", "b", "c"]
assert fn('"a,b",c') == ["a,b", "c"]
assert fn('"a""b",x') == ['a"b', "x"]
assert fn(",") == ["", ""]
assert fn('"",tail') == ["", "tail"]
assert fn(' a , "b" ') == [" a ", ' "b" ']
for bad in ['"abc', '"a"x', 'a,"b"c']:
    try:
        fn(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"expected ValueError for {bad!r}")
""",
    },
    {
        "id": "topological_sort",
        "prompt": """Implement this Python function:

def topological_sort(nodes, edges):
    \"\"\"Return a deterministic topological ordering.

    nodes is an iterable of hashable node names. edges is an iterable of
    (before, after) pairs. Include nodes that only appear in edges. When more
    than one node is available, choose lexicographically smallest first. Raise
    ValueError if the graph contains a cycle.
    \"\"\"

Return only a single Python code block defining topological_sort.""",
        "tests": r"""
fn = ns["topological_sort"]
assert fn(["a", "b", "c"], [("a", "c"), ("b", "c")]) == ["a", "b", "c"]
assert fn([], [("build", "test"), ("lint", "test")]) == ["build", "lint", "test"]
assert fn(["z", "a"], []) == ["a", "z"]
order = fn(["a", "b", "c", "d"], [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
assert order == ["a", "b", "c", "d"]
try:
    fn(["a", "b"], [("a", "b"), ("b", "a")])
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError")
""",
    },
    {
        "id": "edit_distance",
        "prompt": """Implement this Python function:

def edit_distance(a, b):
    \"\"\"Return Levenshtein distance between strings a and b.

    Insertions, deletions, and substitutions all cost 1.
    \"\"\"

Return only a single Python code block defining edit_distance.""",
        "tests": r"""
fn = ns["edit_distance"]
assert fn("", "") == 0
assert fn("kitten", "sitting") == 3
assert fn("flaw", "lawn") == 2
assert fn("abc", "abc") == 0
assert fn("abc", "") == 3
assert fn("", "abc") == 3
""",
    },
    {
        "id": "binary_search_bounds",
        "prompt": """Implement this Python function:

def binary_search_bounds(nums, target):
    \"\"\"Return (first_index, last_index) for target in sorted nums.

    Return (-1, -1) if target is absent. Do not use list.index or scan the
    whole list linearly.
    \"\"\"

Return only a single Python code block defining binary_search_bounds.""",
        "tests": r"""
fn = ns["binary_search_bounds"]
assert fn([1, 2, 2, 2, 3], 2) == (1, 3)
assert fn([1, 1, 1], 1) == (0, 2)
assert fn([1, 2, 3], 4) == (-1, -1)
assert fn([], 1) == (-1, -1)
assert fn([5], 5) == (0, 0)
""",
    },
]


TASKS = BASE_TASKS + EXTRA_TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8106")
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--timeout", type=float, default=8)
    ap.add_argument("--out-dir", default="tuning/coding_eval_results")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Run the selected task set this many times.")
    ap.add_argument("--only", action="append", default=[],
                    help="Comma-separated task ids to run. Can be passed more than once.")
    args = ap.parse_args()
    if args.repeat < 1:
        ap.error("--repeat must be >= 1")

    only = []
    for value in args.only:
        only.extend(part.strip() for part in value.split(",") if part.strip())
    if only:
        wanted = set(only)
        tasks = [task for task in TASKS if task["id"] in wanted]
        missing = sorted(wanted - {task["id"] for task in tasks})
        if missing:
            ap.error("unknown task id(s): " + ", ".join(missing))
    else:
        tasks = TASKS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.time()
    for run in range(1, args.repeat + 1):
        for task in tasks:
            print(f"== {args.label}:r{run}:{task['id']} ==", flush=True)
            try:
                text, usage, elapsed = post_chat(args.base_url, task["prompt"], args.max_tokens)
                code = extract_code(text)
                ok, err = run_tests(code, task["tests"], args.timeout)
            except (urllib.error.URLError, subprocess.TimeoutExpired, Exception) as e:
                text, usage, elapsed, code, ok, err = "", {}, 0.0, "", False, repr(e)
            result = {
                "id": task["id"],
                "run": run,
                "ok": ok,
                "elapsed_s": elapsed,
                "usage": usage,
                "error": err,
                "text": text,
                "code": code,
            }
            results.append(result)
            status = "PASS" if ok else "FAIL"
            completion = usage.get("completion_tokens")
            tps = completion / elapsed if completion and elapsed > 0 else None
            speed = f" tokens={completion} tps={tps:.2f}" if tps else ""
            print(f"{status}{speed}", flush=True)
            if err:
                print(textwrap.shorten(err.replace("\n", " "), width=260), flush=True)

    passed = sum(1 for r in results if r["ok"])
    by_task = {}
    for result in results:
        item = by_task.setdefault(result["id"], {"passed": 0, "total": 0})
        item["total"] += 1
        if result["ok"]:
            item["passed"] += 1
    summary = {
        "label": args.label,
        "passed": passed,
        "total": len(results),
        "task_count": len(tasks),
        "repeat": args.repeat,
        "by_task": by_task,
        "elapsed_s": time.time() - started,
        "results": results,
    }
    out_path = out_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"SUMMARY {args.label}: {passed}/{len(results)} passed")
    print(f"WROTE {out_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
