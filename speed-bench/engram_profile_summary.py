#!/usr/bin/env python3
"""Summarize inclusive V4.1 profiling intervals; stdout is a small JSON report."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics


def fields(line):
    return {k: float(v) for k, v in re.findall(r'(\w+)=([-+\d.eE]+)', line)}


def summarize(log, frontier):
    caches, tokens, layers, prefill, prefetch = [], [], [], [], []
    stages = defaultdict(float)
    for line in log.splitlines():
        if line.startswith('ds4: CUDA SSD cache '):
            caches.append(dict(fields(line), pos=None))
        elif line.startswith('ds4: V4.1 decode layer '):
            row = fields(line)
            if row['pos'] >= frontier:
                layers.append(row)
            # The synchronous cache load precedes this layer's completion log.
            if caches and caches[-1]['pos'] is None and caches[-1]['layer'] == row['layer']:
                caches[-1]['pos'] = row['pos']
        elif line.startswith('ds4: V4.1 decode token '):
            row = fields(line)
            if row['pos'] >= frontier:
                tokens.append(row)
        elif line.startswith('ds4: V4.1 prefill layer='):
            prefill.append(fields(line))
        elif line.startswith('ds4: CUDA SSD prefetch '):
            prefetch.append(fields(line))
        elif line.startswith('ds4: V4.1 stage '):
            match = re.search(r'rows=\d+ (.*?)=([-+\d.eE]+) ms', line)
            if match:
                stages[match[1]] += float(match[2])
    decode_cache = [r for r in caches if r['pos'] is not None and r['pos'] >= frontier]
    n = len(tokens)
    sums = lambda rows, key: sum(r.get(key, 0) for r in rows)
    unique = sums(decode_cache, 'unique')
    by_layer = defaultdict(float)
    for row in layers:
        by_layer[int(row['layer'])] += row['wall_ms']
    result = {
        'frontier': frontier, 'decode_tokens': n, 'decode_layer_records': len(layers),
        'decode_cache_records': len(decode_cache),
        'decode_all_ok': bool(tokens) and all(r['ok'] == 1 for r in tokens),
        'decode_cache_miss_fraction': sums(decode_cache, 'misses') / unique if unique else None,
        'decode_logical_expert_read_GiB': sums(decode_cache, 'bytes') / 2**30,
        'prefill_stage_seconds': {k: v / 1000 for k, v in stages.items()},
        'prefill_inclusive_seconds': {k: sums(prefill, k) / 1000 for k in ('map','engram','encode','drain','seed')},
        'prefetch_foreground_wait_seconds': sums(prefetch, 'wait') / 1000,
        'prefetch_background_elapsed_seconds': sums(prefetch, 'read') / 1000,
        'prefetch_logical_GiB': sums(prefetch, 'bytes') / 2**30,
        'interval_warning': 'Stage/cache/prefetch timings are nested or overlapping; do not add them all. load_ms includes victim selection, disk reads, uploads and upload completion. Profiling is not a throughput benchmark.',
    }
    if n:
        result['decode_mean_ms'] = {k: sums(tokens, k) / n for k in ('total_ms','submit_or_serial_ms','wait0_ms','wait1_ms')}
        result['decode_cache_ms_per_token'] = {k: sums(decode_cache, k) / n for k in ('prefetch_wait_ms','reuse_sync_ms','prepare_ms','load_ms','finish_ms','total_ms')}
        result['decode_top_layers_ms_per_token'] = sorted(
            [{'layer': k, 'ms': v / n} for k, v in by_layer.items()], key=lambda r: -r['ms'])[:10]
        result['decode_median_ms'] = statistics.median(r['total_ms'] for r in tokens)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--frontier', type=int, default=65536)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log.read_text(), args.frontier), indent=2))


if __name__ == '__main__':
    main()
