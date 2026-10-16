#!/usr/bin/env python3
"""Read Codex session quota telemetry. Exit 0=proceed, 1=stop, 2=unknown.

This is a checkpoint guard, not a reservation or a live account API.
It never reads credentials or launches inference requests.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path


def evaluate(event, threshold, max_age, now):
    timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
    age = (now - timestamp).total_seconds()
    if not 0 <= age <= max_age:
        raise ValueError('quota telemetry is stale or future-dated')
    limits = event['payload']['rate_limits']
    if not isinstance(limits, dict) or limits.get('limit_id') != 'codex':
        raise ValueError('missing or unrecognized account quota')
    windows = []
    for name in ('primary', 'secondary'):
        value = limits.get(name)
        if value is None:
            continue
        used = value['used_percent']
        if isinstance(used, bool) or not isinstance(used, (int, float)) or not math.isfinite(used) or not 0 <= used <= 100:
            raise ValueError('invalid quota percentage')
        reset = datetime.fromtimestamp(value['resets_at'], timezone.utc)
        if reset <= now:
            raise ValueError('quota window expired; fresh telemetry required')
        windows.append({'window': name, 'remaining_percent': 100 - used,
                        'window_minutes': value['window_minutes'],
                        'resets_at': reset.isoformat()})
    if not windows:
        raise ValueError('no quota windows reported')
    stop = (any(w['remaining_percent'] < threshold for w in windows)
            or bool(limits.get('spend_control_reached'))
            or bool(limits.get('rate_limit_reached_type')))
    return {'decision': 'STOP' if stop else 'PROCEED',
            'observed_at': timestamp.isoformat(), 'age_seconds': round(age, 1),
            'threshold_remaining_percent': threshold, 'windows': windows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-file', type=Path)
    parser.add_argument('--threshold', type=float, default=60)
    parser.add_argument('--max-age', type=float, default=600)
    args = parser.parse_args()
    try:
        if not 0 <= args.threshold <= 100 or not math.isfinite(args.max_age) or args.max_age <= 0:
            raise ValueError('invalid threshold or max-age')
        path = args.session_file
        if path is None:
            thread = os.environ.get('CODEX_THREAD_ID', '')
            if not thread or any(c not in '0123456789abcdef-' for c in thread):
                raise ValueError('missing/invalid CODEX_THREAD_ID; supply --session-file')
            base = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex')))
            matches = list((base / 'sessions').glob(f'**/*{thread}.jsonl'))
            if len(matches) != 1:
                raise ValueError('cannot uniquely locate current session')
            path = matches[0]
        latest = None
        with path.open() as stream:
            for line in stream:
                if 'token_count' not in line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue  # The writer may still be appending the last line.
                if item.get('type') == 'event_msg' and item.get('payload', {}).get('type') == 'token_count':
                    latest = item
        if latest is None:
            raise ValueError('no quota telemetry found')
        result = evaluate(latest, args.threshold, args.max_age, datetime.now(timezone.utc))
        print(json.dumps(result))
        return 1 if result['decision'] == 'STOP' else 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(json.dumps({'decision': 'UNKNOWN_STOP', 'reason': str(exc)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
