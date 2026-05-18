#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path


TASKS = [
    {
        "id": "merge_intervals",
        "prompt": """Implement this Python function:

def merge_intervals(intervals):
    \"\"\"Return merged closed intervals sorted by start.

    intervals is an iterable of [start, end] pairs. Intervals that overlap or
    touch must be merged, so [1, 3] and [3, 5] become [1, 5]. The input must not
    be mutated. Raise ValueError if an interval has start > end.
    \"\"\"

Return only a single Python code block defining merge_intervals.""",
        "tests": r"""
fn = ns["merge_intervals"]
def norm(x):
    return [list(p) for p in x]
src = [[5, 7], [1, 3], [2, 6], [8, 10], [10, 12]]
orig = [p[:] for p in src]
assert norm(fn(src)) == [[1, 7], [8, 12]]
assert src == orig
assert norm(fn([])) == []
assert norm(fn([(1, 1), (2, 3), (3, 3), (10, 11)])) == [[1, 1], [2, 3], [10, 11]]
assert norm(fn([[0, 100], [20, 30], [-5, -1]])) == [[-5, -1], [0, 100]]
try:
    fn([[3, 2]])
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError")
""",
    },
    {
        "id": "top_k_frequent",
        "prompt": """Implement this Python function:

def top_k_frequent(items, k):
    \"\"\"Return the k most frequent strings.

    Sort by descending frequency; break ties lexicographically ascending.
    If k is larger than the number of distinct items, return every distinct
    item. If k <= 0, return [].
    \"\"\"

Return only a single Python code block defining top_k_frequent.""",
        "tests": r"""
fn = ns["top_k_frequent"]
assert fn(["b", "a", "b", "c", "a", "b"], 2) == ["b", "a"]
assert fn(["z", "x", "y", "x", "y", "z"], 3) == ["x", "y", "z"]
assert fn(["aa", "b", "aa", "c", "b"], 10) == ["aa", "b", "c"]
assert fn(["a"], 0) == []
assert fn([], 3) == []
""",
    },
    {
        "id": "lru_cache",
        "prompt": """Implement this Python class:

class LRUCache:
    def __init__(self, capacity): ...
    def get(self, key): ...
    def put(self, key, value): ...

Rules:
- capacity must be positive, otherwise raise ValueError.
- get(key) returns the value or -1 if absent.
- get and put mark a key as recently used.
- when capacity is exceeded, evict the least recently used key.

Return only a single Python code block defining LRUCache.""",
        "tests": r"""
LRUCache = ns["LRUCache"]
try:
    LRUCache(0)
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError")
c = LRUCache(2)
c.put("a", 1)
c.put("b", 2)
assert c.get("a") == 1
c.put("c", 3)
assert c.get("b") == -1
assert c.get("c") == 3
c.put("a", 10)
assert c.get("a") == 10
c.put("d", 4)
assert c.get("c") == -1
assert c.get("d") == 4
""",
    },
    {
        "id": "parse_duration",
        "prompt": """Implement this Python function:

def parse_duration(text):
    \"\"\"Parse duration strings into total seconds.

    Valid chunks are positive integer + unit, with unit one of d, h, m, s.
    Chunks may have optional spaces between them: "1h30m", "2d 3h 4m 5s".
    Units may appear at most once. Reject empty strings, repeated units,
    negative numbers, decimals, unknown units, or trailing junk by raising
    ValueError.
    \"\"\"

Return only a single Python code block defining parse_duration.""",
        "tests": r"""
fn = ns["parse_duration"]
assert fn("45s") == 45
assert fn("1h30m") == 5400
assert fn("2d 3h 4m 5s") == 183845
assert fn("10m 5s") == 605
for bad in ["", "  ", "1h 2h", "-1s", "1.5h", "1x", "1h nope", "h", "0s"]:
    try:
        fn(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"expected ValueError for {bad!r}")
""",
    },
]


SAFE_DRIVER = r'''
import builtins
import collections
import dataclasses
import functools
import heapq
import math
import re
import typing

allowed_modules = {
    "bisect", "collections", "dataclasses", "functools", "heapq", "math",
    "re", "typing",
}

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root not in allowed_modules:
        raise ImportError(f"blocked import: {name}")
    return __import__(name, globals, locals, fromlist, level)

safe_builtins = {
    "__build_class__": builtins.__build_class__,
    "__import__": guarded_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "iter": iter,
    "KeyError": KeyError,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "print": print,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "super": super,
    "tuple": tuple,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "zip": zip,
}

ns = {"__builtins__": safe_builtins, "__name__": "__generated__"}
code = CODE_UNDER_TEST
exec(code, ns)
TEST_CODE
'''


def post_chat(base_url, prompt, max_tokens):
    body = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior Python engineer. Produce minimal, correct, "
                    "runnable Python. Do not include explanations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    choice = payload["choices"][0]
    msg = choice.get("message", {})
    text = msg.get("content")
    if text is None:
        text = choice.get("text", "")
    usage = payload.get("usage", {})
    return text, usage, elapsed


def extract_code(text):
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.S | re.I)
    if match:
        return match.group(1).strip()
    return text.strip()


def run_tests(code, test_code, timeout):
    banned = [
        "open(",
        "exec(",
        "eval(",
        "__import__",
        "subprocess",
        "socket",
        "pathlib",
        "shutil",
        "os.",
        "sys.",
    ]
    for needle in banned:
        if needle in code:
            return False, f"blocked unsafe token: {needle}"
    driver = SAFE_DRIVER.replace("CODE_UNDER_TEST", repr(code)).replace("TEST_CODE", test_code)
    proc = subprocess.run(
        [sys.executable, "-I", "-c", driver],
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8106")
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--timeout", type=float, default=8)
    ap.add_argument("--out-dir", default="tuning/coding_eval_results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for task in TASKS:
        print(f"== {args.label}:{task['id']} ==", flush=True)
        try:
            text, usage, elapsed = post_chat(args.base_url, task["prompt"], args.max_tokens)
            code = extract_code(text)
            ok, err = run_tests(code, task["tests"], args.timeout)
        except (urllib.error.URLError, subprocess.TimeoutExpired, Exception) as e:
            text, usage, elapsed, code, ok, err = "", {}, 0.0, "", False, repr(e)
        result = {
            "id": task["id"],
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
            print(textwrap.shorten(err.replace("\n", " "), width=220), flush=True)

    passed = sum(1 for r in results if r["ok"])
    summary = {
        "label": args.label,
        "passed": passed,
        "total": len(results),
        "results": results,
    }
    out_path = out_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"SUMMARY {args.label}: {passed}/{len(results)} passed")
    print(f"WROTE {out_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
