#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path


DEFAULT_QUESTION = 'mi fai la lista delle province italiane che cominciano per la lettera "B" ?'
TOKEN_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) token index=(\d+) ")


def parse_env(items):
    env = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        env[k] = v
    return env


def safe_name(name):
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def read_proc_rss_kib(pid):
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        return None
    return None


def read_free_bytes():
    try:
        out = subprocess.check_output(["free", "-b"], text=True)
    except Exception:
        return {}
    lines = out.strip().splitlines()
    if len(lines) < 2:
        return {}
    parts = lines[1].split()
    if len(parts) < 7 or parts[0] != "Mem:":
        return {}
    return {
        "mem_total_bytes": int(parts[1]),
        "mem_used_bytes": int(parts[2]),
        "mem_free_bytes": int(parts[3]),
        "mem_shared_bytes": int(parts[4]),
        "mem_buff_cache_bytes": int(parts[5]),
        "mem_available_bytes": int(parts[6]),
    }


def read_cuda_used_mib(pid):
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return None
    for line in out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) >= 2 and cols[0] == str(pid):
            try:
                return int(cols[1])
            except ValueError:
                return None
    return None


def parse_trace(path):
    generated = []
    saw_prefill_suffix = False
    in_generated = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "tokens label=prefill_suffix" in line:
                saw_prefill_suffix = True
                in_generated = False
                continue
            m = TOKEN_RE.match(line)
            if not m:
                continue
            idx = int(m.group(2))
            if saw_prefill_suffix and not in_generated:
                if idx == 1:
                    in_generated = True
                else:
                    continue
            if in_generated:
                ts = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
                generated.append(ts)

    if not generated:
        return {
            "generated_tokens": 0,
            "trace_span_s": 0.0,
            "trace_tps": 0.0,
            "trace_interval_tps": 0.0,
        }
    span = (generated[-1] - generated[0]).total_seconds()
    return {
        "generated_tokens": len(generated),
        "trace_span_s": span,
        "trace_tps": (len(generated) / span) if span > 0 else 0.0,
        "trace_interval_tps": ((len(generated) - 1) / span) if span > 0 and len(generated) > 1 else 0.0,
        "first_token_ts": generated[0].isoformat(sep=" "),
        "last_token_ts": generated[-1].isoformat(sep=" "),
    }


def gib(v):
    return None if v is None else v / 1073741824.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--repo", default="/home/alessandro/projects/ds4-stepwise")
    ap.add_argument("--model", default="/home/alessandro/projects/ds4/ds4flash.gguf")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--out-root", default="tuning/agent_stepwise_runs/results")
    args = ap.parse_args()

    repo = Path(args.repo)
    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = repo / args.out_root / f"{run_stamp}_{safe_name(args.name)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.log"
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    mem_path = run_dir / "memory.csv"
    summary_path = run_dir / "summary.json"
    root_summary = repo / args.out_root / "summary.csv"

    extra_env = parse_env(args.env)
    env = os.environ.copy()
    env.update(extra_env)

    cmd = [
        str(repo / "ds4-agent"),
        "--cuda",
        "--ctx",
        str(args.ctx),
        "--chdir",
        str(repo),
        "-m",
        args.model,
        "--non-interactive",
        "-p",
        args.question,
        "-n",
        str(args.tokens),
        "--trace",
        str(trace_path),
    ]

    start = time.time()
    samples = []
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=out, stderr=err, start_new_session=True)
        deadline = start + args.timeout
        last_cuda_sample = 0.0
        while True:
            now = time.time()
            rss_kib = read_proc_rss_kib(proc.pid)
            free_info = read_free_bytes()
            cuda_mib = None
            if now - last_cuda_sample >= 1.0:
                cuda_mib = read_cuda_used_mib(proc.pid)
                last_cuda_sample = now
            samples.append({
                "t_s": now - start,
                "rss_kib": rss_kib,
                "cuda_used_mib": cuda_mib,
                **free_info,
            })
            if proc.poll() is not None:
                break
            if now >= deadline:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(2)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        rc = proc.wait()
    end = time.time()

    with open(mem_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "t_s",
            "rss_kib",
            "cuda_used_mib",
            "mem_total_bytes",
            "mem_used_bytes",
            "mem_free_bytes",
            "mem_shared_bytes",
            "mem_buff_cache_bytes",
            "mem_available_bytes",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in samples:
            w.writerow({k: s.get(k) for k in fields})

    trace_stats = parse_trace(trace_path) if trace_path.exists() else {}
    peak_rss_kib = max((s.get("rss_kib") or 0 for s in samples), default=0)
    peak_cuda_mib = max((s.get("cuda_used_mib") or 0 for s in samples), default=0)
    peak_mem_used = max((s.get("mem_used_bytes") or 0 for s in samples), default=0)
    peak_mem_total_minus_available = max(
        ((s.get("mem_total_bytes") or 0) - (s.get("mem_available_bytes") or 0) for s in samples),
        default=0,
    )

    summary = {
        "name": args.name,
        "returncode": rc,
        "elapsed_s": end - start,
        "env": extra_env,
        "command": cmd,
        "run_dir": str(run_dir),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "trace": str(trace_path),
        "memory": str(mem_path),
        "peak_rss_gib": gib(peak_rss_kib * 1024),
        "peak_cuda_gib": peak_cuda_mib / 1024.0,
        "peak_mem_used_gib": gib(peak_mem_used),
        "peak_mem_total_minus_available_gib": gib(peak_mem_total_minus_available),
        **trace_stats,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    root_summary.parent.mkdir(parents=True, exist_ok=True)
    root_exists = root_summary.exists()
    with open(root_summary, "a", newline="", encoding="utf-8") as f:
        fields = [
            "name",
            "returncode",
            "generated_tokens",
            "trace_tps",
            "trace_interval_tps",
            "trace_span_s",
            "elapsed_s",
            "peak_rss_gib",
            "peak_cuda_gib",
            "peak_mem_used_gib",
            "peak_mem_total_minus_available_gib",
            "run_dir",
            "env",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        if not root_exists:
            w.writeheader()
        w.writerow({k: json.dumps(summary[k]) if k == "env" else summary.get(k) for k in fields})

    print(json.dumps(summary, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
