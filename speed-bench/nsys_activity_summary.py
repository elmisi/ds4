#!/usr/bin/env python3
"""Read-only whole-trace GPU activity union; gaps are not CPU/disk attribution."""
import argparse
import json
from pathlib import Path
import sqlite3


def intervals_summary(intervals):
    first = end = previous_start = None
    busy = count = largest_gap = 0
    for start, stop in intervals:
        if stop < start or (previous_start is not None and start < previous_start):
            raise ValueError('invalid interval')
        previous_start = start
        count += 1
        if first is None:
            first, end = start, stop
            busy = stop - start
        elif start > end:
            largest_gap = max(largest_gap, start - end)
            busy += stop - start
            end = stop
        elif stop > end:
            busy += stop - end
            end = stop
    span = end - first if first is not None else 0
    return dict(events=count, first_ns=first, last_ns=end,
                span_seconds=span / 1e9, busy_union_seconds=busy / 1e9,
                gap_seconds=(span - busy) / 1e9, largest_gap_seconds=largest_gap / 1e9)


def collect(path):
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        available = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        tables = ['CUPTI_ACTIVITY_KIND_' + kind for kind in ['KERNEL', 'MEMCPY', 'MEMSET']]
        tables = [t for t in tables if t in available]
        if not tables:
            raise ValueError('no GPU activity tables')
        query = ' UNION ALL '.join('SELECT start,end FROM ' + t for t in tables) + ' ORDER BY start,end'
        return dict(scope='whole trace, not an isolated prefill range',
                    warning='Gaps are unobserved GPU activity, not proof of disk/CPU causation.',
                    **intervals_summary(db.execute(query)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.sqlite), indent=2))
