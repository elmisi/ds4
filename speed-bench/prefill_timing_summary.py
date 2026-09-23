#!/usr/bin/env python3
"""Report all timing samples, excluding an entire pair if either side swapped."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import statistics


def paired_summary(rows):
    pairs = defaultdict(dict)
    for row in rows:
        if row['mode'] in pairs[row['pair']]:
            raise ValueError('duplicate timing sample')
        pairs[row['pair']][row['mode']] = row
    clean, excluded = [], []
    for number, sides in sorted(pairs.items()):
        reasons = []
        if set(sides) != {'on', 'off'}:
            reasons.append('incomplete pair')
        for mode, row in sides.items():
            if not row['success']:
                reasons.append(mode + ': failed/stopped run')
            if row['swap_kib'] is None or row['swap_kib'] != 0:
                reasons.append(mode + ': swap present/unknown')
        if reasons:
            excluded.append(dict(pair=number, reasons=reasons))
        else:
            clean.append(sides)
    means = {mode: statistics.mean(pair[mode]['prefill_tps'] for pair in clean)
             for mode in ['off', 'on']} if clean else None
    return dict(all_samples=rows, clean_pair_count=len(clean), excluded_pairs=excluded,
                clean_prefill_means=means,
                clean_prefill_gain_percent=(means['on'] / means['off'] - 1) * 100 if means else None,
                warning='Small sample; this is not a confidence interval. Disk telemetry is host-wide.')


def collect(raw, prefix, telemetry=None):
    samples = [json.loads(line) for line in telemetry.read_text().splitlines()] if telemetry else []
    rows = []
    for directory in sorted(raw.glob(prefix + '-timing-*')):
        match = re.fullmatch(re.escape(prefix) + r'-timing-(\d+)-(on|off)', directory.name)
        if not match or not (directory / 'summary.json').exists():
            continue
        summary = json.loads((directory / 'summary.json').read_text())
        with (directory / 'bench.csv').open() as stream:
            bench = next(csv.DictReader(stream))
        row = dict(run=directory.name, pair=int(match[1]), mode=match[2],
                   success=summary['returncode'] == 0 and summary['stop_reason'] is None,
                   prefill_tps=float(bench['prefill_tps']), decode_tps=float(bench['gen_tps']),
                   wall_seconds=summary['wall_seconds'], swap_kib=summary.get('peak_process_swap_kib'))
        start = json.loads((directory / 'command.json').read_text())['start']
        window = [s for s in samples if start <= s['time'] <= start + row['wall_seconds']]
        if len(window) > 1:
            temperatures = [value for s in window for name, value in s['temperatures'].items()
                            if name.endswith('/nvme/Composite')]
            if temperatures:
                row['nvme_celsius_range'] = [min(temperatures), max(temperatures)]
            row['telemetry_window_seconds'] = window[-1]['time'] - window[0]['time']
            row['disk_read_GiB_in_window'] = {name: (values['read_sectors'] -
                window[0]['disks'][name]['read_sectors']) * 512 / 2**30
                for name, values in window[-1]['disks'].items() if name in window[0]['disks']}
        rows.append(row)
    return paired_summary(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--telemetry', type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.raw, args.prefix, args.telemetry), indent=2))
