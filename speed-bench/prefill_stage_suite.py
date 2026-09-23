#!/usr/bin/env python3
"""Exact-gated, counterbalanced V4.1 prefill opt-in experiment.

Uses an explicitly supplied quota/resource runner; never rebuilds, deploys,
flushes filesystem caches or stops another service. Engram pool8 stays fixed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys

from engram_pool_suite import digest
from prefill_timing_summary import collect


def jobs(pairs):
    if pairs < 1:
        raise ValueError('at least one timing pair is required')
    result = [('gate-off', 0, True), ('gate-on', 1, True)]
    for pair in range(pairs):
        # ON/OFF, then OFF/ON; repeated ABBA reduces simple order bias.
        for enabled in ([1, 0] if pair % 2 == 0 else [0, 1]):
            result.append((f'timing-{pair + 1}-{ "on" if enabled else "off"}', enabled, False))
    return result


def identical(left, right, vectors):
    expected = vectors * 129280 * 4
    return (left.stat().st_size == right.stat().st_size == expected and
            digest(left) == digest(right))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['runner', 'model', 'engram-model']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--pairs', type=int, default=2)
    parser.add_argument('--context', type=int, default=65536)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--toggle', choices=['DS4_CUDA_V41_PREFILL_STAGE_SYNC',
                                            'DS4_CUDA_V41_TOPK_BATCH'],
                        default='DS4_CUDA_V41_PREFILL_STAGE_SYNC')
    parser.add_argument('--reference', type=Path,
                        help='optional pre-change full-logit file for the OFF gate')
    args = parser.parse_args()
    if Path(args.prefix).name != args.prefix or args.prefix in ('.', '..'):
        parser.error('prefix must be a simple name')
    if not 1024 <= args.context <= 261888 or args.pairs < 1 or args.timeout < 1:
        parser.error('invalid context, pairs or timeout')
    root = Path(__file__).resolve().parents[1]
    binary = root / 'ds4-bench'
    raw = args.runner.resolve().parent / 'raw'
    plan = jobs(args.pairs)
    if any((raw / (args.prefix + '-' + label)).exists() for label, _, _ in plan):
        parser.error('one or more output runs already exist; use a new prefix')
    sha = digest(binary)
    patch = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root)
    provenance = dict(source_root=str(root), source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        source_diff_sha256=hashlib.sha256(patch).hexdigest(), binary_sha256=sha,
        suite_sha256=digest(Path(__file__)), populated_tokens=args.context,
        allocated_context=262144, decode_tokens=256, cache_argument='72GB',
        plan=plan, toggle=args.toggle,
        reference=str(args.reference) if args.reference else None)
    timings = {0: [], 1: []}
    for label, enabled, dump in plan:
        if digest(binary) != sha:
            raise RuntimeError('binary changed during suite')
        name = args.prefix + '-' + label
        directory = raw / name
        command = [sys.executable, str(args.runner.resolve()), '--timeout', str(args.timeout),
                   '--min-available-gib', '8', '--env', 'DS4_CUDA_ENGRAM_READERS=8',
                   '--env', f'{args.toggle}={enabled}']
        if dump:
            command += ['--env', f'DS4_BENCH_DECODE_LOGITS_FILE={directory}/decode.f32']
        command += [name, '--', str(binary), '--model', str(args.model.resolve()),
                    '--engram-model', str(args.engram_model.resolve()), '--cuda',
                    '--ssd-streaming', '--ssd-streaming-cache-experts', '72GB',
                    '--prompt-file', str(root / 'speed-bench/promessi_sposi.txt'),
                    '--ctx-start', str(args.context), '--ctx-max', str(args.context),
                    '--ctx-alloc', '262144', '--gen-tokens', '256',
                    '--teacher-forced-decode', '--csv', '{run}/bench.csv']
        print(json.dumps(dict(starting=name, toggle=args.toggle, enabled=enabled, dump=dump)), flush=True)
        child = subprocess.Popen(command)
        try:
            while child.poll() is None:
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    print(json.dumps(dict(running=name)), flush=True)
        except BaseException:
            child.send_signal(signal.SIGINT)
            child.wait()
            raise
        if directory.exists():
            (directory / 'source-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
            (directory / 'source.patch').write_bytes(patch)
        if child.returncode:
            raise RuntimeError(f'{name} failed/stopped: {child.returncode}')
        if label == 'gate-off' and args.reference:
            if not identical(args.reference, directory / 'decode.f32', 256):
                raise RuntimeError('OFF no longer matches pre-change reference')
        if label == 'gate-on':
            off = raw / (args.prefix + '-gate-off') / 'decode.f32'
            on = directory / 'decode.f32'
            exact = identical(off, on, 256)
            comparison = dict(exact=exact, vectors=256, off_sha256=digest(off), on_sha256=digest(on))
            (directory / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
            print(json.dumps(comparison), flush=True)
            if not exact:
                raise RuntimeError('full logits differ; no timing runs started')
        if not dump:
            with (directory / 'bench.csv').open() as stream:
                row = next(csv.DictReader(stream))
            timings[enabled].append({key: float(row[key]) for key in ['prefill_tps', 'gen_tps']})
    means = {str(enabled): {key: statistics.mean(row[key] for row in rows)
                           for key in ['prefill_tps', 'gen_tps']}
             for enabled, rows in timings.items()}
    summary = dict(prefix=args.prefix, unfiltered_means=means, timings=timings,
                   **collect(raw, args.prefix))
    (directory / 'suite-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
