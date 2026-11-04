#!/usr/bin/env python3
"""Validate the host-side long-context recall result without touching the run."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

EXPECTED = {
    "Bob": 34, "Alice": 52, "Clara": 71, "Diego": 93,
    "Elena": 16, "Felix": 88, "Greta": 47, "Hugo": 29,
    "Iris": 64, "Jonas": 12, "Kira": 81, "Leo": 39,
    "Marta": 76, "Nadia": 23, "Owen": 58, "Priya": 97,
}
LINE = re.compile(r"([A-Za-z]+)=([0-9]+)\n\Z")
FIXTURE = Path(__file__).resolve().parent / "fixtures/recall-256k.txt"
EXPECTED_FIXTURE_SHA256 = "28c641b634fd1156823582d0eca7aa276a8db032c137124f79da9804182e2e00"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(run_dir: Path) -> tuple[bool, str, str]:
    summary = run_dir / "summary.json"
    stdout = run_dir / "stdout.log"
    if not summary.is_file() or not stdout.is_file():
        return False, "missing summary.json or stdout.log", ""
    try:
        state = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False, "invalid summary.json", digest(stdout)
    if not isinstance(state, dict):
        return False, "summary.json must contain an object", digest(stdout)
    if state.get("returncode") != 0:
        return False, "run returncode is not zero", digest(stdout)
    if state.get("stop_reason") is not None:
        return False, "run has stop_reason", digest(stdout)
    try:
        text = stdout.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False, "cannot read stdout.log", ""
    lines = text.splitlines(keepends=True)
    if len(lines) != len(EXPECTED) or any(not line.endswith("\n") for line in lines):
        return False, "stdout must contain exactly 16 newline-terminated lines", digest(stdout)
    found: dict[str, int] = {}
    for line in lines:
        match = LINE.fullmatch(line)
        if not match:
            return False, "stdout contains extra prose or malformed line", digest(stdout)
        name, value = match.groups()
        if name in found:
            return False, f"duplicate name: {name}", digest(stdout)
        found[name] = int(value)
    if found != EXPECTED:
        return False, "recall names or values differ", digest(stdout)
    return True, "PASS", digest(stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print(f"refusing to overwrite report: {args.output}")
        return 1
    try:
        source_hash = digest(FIXTURE)
    except (OSError, ValueError):
        print(f"cannot read source fixture: {FIXTURE}")
        return 1
    if source_hash != EXPECTED_FIXTURE_SHA256:
        print("source fixture SHA-256 does not match pinned recall fixture")
        return 1
    ok, error, stdout_hash = check(args.run_dir.resolve())
    report = {
        "status": "PASS" if ok else "FAIL",
        "error": None if ok else error,
        "source": str(FIXTURE),
        "source_sha256": source_hash,
        "run": str(args.run_dir.resolve()),
        "stdout_sha256": stdout_hash,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(report["status"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
