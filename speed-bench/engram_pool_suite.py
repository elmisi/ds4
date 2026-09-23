#!/usr/bin/env python3
"""Bounded Engram A/B using the existing quota/resource-guarded local runner.

The runner path is explicit: it is an operational dependency, not vendored here.
No build, service stop, cache flush, or deployment is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--engram-model', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    binary = root / 'ds4-bench'
    raw = args.runner.resolve().parent / 'raw'
    if Path(args.prefix).name != args.prefix or args.prefix in ('.', '..'):
        parser.error('prefix must be a simple name')
    sha = digest(binary)
    diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root)
    provenance = {
        'source_root': str(root),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'source_diff_sha256': hashlib.sha256(diff).hexdigest(),
        'binary_sha256': sha,
        'suite_sha256': digest(Path(__file__)),
        'populated_tokens': 65536, 'allocated_context': 262144,
        'decode_tokens': 256, 'cache_argument': '72GB',
    }
    jobs = [('gate-off', 0, True), ('gate-on8', 8, True),
            ('timing-on8', 8, False), ('timing-off', 0, False)]
    for label, readers, dump in jobs:
        if digest(binary) != sha:
            raise RuntimeError('binary changed during suite')
        name = args.prefix + '-' + label
        directory = raw / name
        command = [sys.executable, str(args.runner.resolve()), '--timeout', '1500',
                   '--min-available-gib', '8', '--env', f'DS4_CUDA_ENGRAM_READERS={readers}']
        if dump:
            command += ['--env', f'DS4_BENCH_DECODE_LOGITS_FILE={directory}/decode.f32']
        command += [name, '--', str(binary), '--model', str(args.model.resolve()),
                    '--engram-model', str(args.engram_model.resolve()), '--cuda',
                    '--ssd-streaming', '--ssd-streaming-cache-experts', '72GB',
                    '--prompt-file', str(root / 'speed-bench/promessi_sposi.txt'),
                    '--ctx-start', '65536', '--ctx-max', '65536', '--ctx-alloc', '262144',
                    '--gen-tokens', '256', '--teacher-forced-decode', '--csv', '{run}/bench.csv']
        print(json.dumps({'starting': name, 'readers': readers, 'dump': dump}), flush=True)
        child = subprocess.Popen(command)
        try:
            while child.poll() is None:
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    sample = directory / 'resources.jsonl'
                    last = None
                    if sample.exists():
                        with sample.open('rb') as stream:
                            stream.seek(max(0, sample.stat().st_size - 4096))
                            lines = stream.read().splitlines()
                        for line in reversed(lines):
                            try:
                                last = json.loads(line)
                                break
                            except (ValueError, UnicodeDecodeError):
                                pass
                    print(json.dumps({'running': name, 'resources': last}), flush=True)
        except BaseException:
            child.send_signal(signal.SIGINT)  # run.py catches KeyboardInterrupt and drains its child group.
            child.wait()
            raise
        if child.returncode:
            raise RuntimeError(f'{name} failed/stopped: {child.returncode}')
        (directory / 'source-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        (directory / 'source.patch').write_bytes(diff)
        if label == 'gate-on8':
            off = raw / (args.prefix + '-gate-off') / 'decode.f32'
            on = directory / 'decode.f32'
            # V4.1 vocabulary has 129280 entries: 256 full F32 vectors.
            expected = 256 * 129280 * 4
            exact = off.stat().st_size == on.stat().st_size == expected and digest(off) == digest(on)
            result = {'exact': exact, 'vectors': 256, 'bytes_per_file': expected,
                      'off_sha256': digest(off), 'on_sha256': digest(on)}
            (directory / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
            if not exact:
                raise RuntimeError('full decode logits differ; timing runs not started')
    print(json.dumps({'suite_complete': args.prefix, 'binary_sha256': sha}), flush=True)


if __name__ == '__main__':
    main()
