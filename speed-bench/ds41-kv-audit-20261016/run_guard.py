#!/usr/bin/env python3
"""Run one isolated DS41 experiment with durable provenance and resource samples.

Arguments after -- are executed directly; {run} expands to the new run directory.
Raw results are never overwritten. Inference jobs share an exclusive host lock.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
RAW = Path(__file__).resolve().parent / "raw"


def text_command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def quota_status():
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('check_quota.py')), '--threshold', '50'],
                                capture_output=True, text=True, timeout=10)
        report = json.loads(result.stdout)
        expected = {0: 'PROCEED', 1: 'STOP', 2: 'UNKNOWN_STOP'}.get(result.returncode)
        if expected is None or report.get('decision') != expected:
            raise ValueError('inconsistent quota checker result')
        return report
    except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired) as exc:
        return {'decision': 'UNKNOWN_STOP', 'reason': str(exc)}


def counters(path):
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, value = line.replace(":", " ", 1).split(None, 1)
            try:
                out[key] = int(value.split()[0])
            except ValueError:
                pass
    except (OSError, ValueError):
        pass
    return out


def sample(pid):
    mem = counters("/proc/meminfo")
    proc = counters(f"/proc/{pid}/status")
    vm = counters("/proc/vmstat")
    io = counters(f"/proc/{pid}/io")
    return {"time": time.time(), "available_kib": mem.get("MemAvailable"),
            "swap_used_kib": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0),
            "rss_kib": proc.get("VmRSS"), "process_swap_kib": proc.get("VmSwap"),
            "pswpin": vm.get("pswpin"), "pswpout": vm.get("pswpout"), **io}


def model_processes():
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
            known = {name[:15] for name in (
                "payload-roundtrip", "ds4", "ds4-agent", "ds4-server", "ds4-bench", "ds4-eval", "score_official",
                "replay", "test_q8_cache_x", "test_deepseek41_prefill",
                "test_deepseek41_cuda", "test_cuda_q8_rows", "test_cuda_ssd_cache", "test_cuda_ssd_batch")}
            if comm in known:
                found.append({"pid": int(entry.name), "name": comm})
        except OSError:
            pass
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--min-available-gib", type=float, default=2)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("provide a simple run name and a command after --")
    if args.timeout <= 0 or args.min_available_gib < 0:
        parser.error("invalid timeout or memory floor")
    RAW.mkdir(exist_ok=True)
    with open("/tmp/ds4-ds41-benchmark.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another DS41 experiment holds the host lock")
        competing = model_processes()
        if competing:
            parser.error(f"existing model processes: {competing}")
        quota_before = quota_status()
        if quota_before['decision'] != 'PROCEED':
            print(json.dumps({'run': args.name, 'not_started': True, 'quota': quota_before}))
            return 2
        disk = os.statvfs(RAW)
        if disk.f_bavail * disk.f_frsize < 5 * 1024**3:
            parser.error("less than 5 GiB free for artifacts")
        run = RAW / args.name
        run.mkdir()  # Refuse overwrite, including incomplete earlier runs.
        command = [arg.replace("{run}", str(run)) for arg in command]
        env = {k: v for k, v in os.environ.items() if not k.startswith("DS4_")}
        env["DS4_CUDA_DECODE_GRAPHS"] = "0"
        env["DS4_BENCH_FORCE_SNAPSHOT"] = "1"
        for value in args.env:
            key, sep, val = value.partition("=")
            if not sep or not key:
                parser.error("--env needs KEY=VALUE")
            env[key] = val
        binary = Path(command[0])
        if not binary.is_absolute():
            binary = ROOT / binary
        metadata = {"command": command, "cwd": str(ROOT), "quota_before": quota_before,
                    "git_commit": text_command("git", "rev-parse", "HEAD"),
                    "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(
                        ["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
                    "environment": {k: v for k, v in env.items() if k.startswith(("DS4_", "CUDA_"))},
                    "start": time.time(), "timeout_seconds": args.timeout,
                    "minimum_available_gib": args.min_available_gib,
                    "cache_state": "fresh engine; OS page cache uncontrolled, not a cold-disk claim"}
        if binary.is_file():
            with binary.open("rb") as f:
                metadata["binary_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        metadata['harness_sha256'] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.iterdir())
            if p.is_file() and p.suffix in {'.py', '.c', '.mk'}}
        (run / "command.json").write_text(json.dumps(metadata, indent=2) + "\n")
        samples = []
        reason = None
        low_memory = 0
        child = None
        try:
            with (run / "stdout.log").open("w") as out, (run / "stderr.log").open("w") as err, \
                    (run / "resources.jsonl").open("w") as resources:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out,
                                         stderr=err, start_new_session=True)
                (run / "pid").write_text(str(child.pid) + "\n")
                while child.poll() is None:
                    current = sample(child.pid)
                    samples.append(current)
                    resources.write(json.dumps(current) + "\n")
                    resources.flush()
                    low_memory = low_memory + 1 if (current["available_kib"] is not None and
                        current["available_kib"] < args.min_available_gib * 1024**2) else 0
                    if low_memory >= 3:
                        reason = "memory-floor"
                        break
                    if time.time() - metadata["start"] > args.timeout:
                        reason = "timeout"
                        break
                    time.sleep(2)
                if reason and child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        except BaseException as exc:
            reason = repr(exc)
            if child and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        available = [s["available_kib"] for s in samples if s["available_kib"] is not None]
        summary = {"run": args.name, "returncode": child.returncode if child else None,
                   "stop_reason": reason, "wall_seconds": time.time() - metadata["start"],
                   "minimum_available_kib": min(available) if available else None,
                   "peak_rss_kib": max((s.get("rss_kib") or 0 for s in samples), default=0),
                   "peak_process_swap_kib": max((s.get("process_swap_kib") or 0 for s in samples), default=0),
                   "last_sample": samples[-1] if samples else None}
        (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        quota_after = quota_status()
        (run / 'quota-after.json').write_text(json.dumps(quota_after, indent=2) + '\n')
        summary['quota_after'] = quota_after
        print(json.dumps(summary))
        if not child or child.returncode != 0 or reason is not None:
            return 1
        return 0 if quota_after['decision'] == 'PROCEED' else 2


if __name__ == "__main__":
    sys.exit(main())
