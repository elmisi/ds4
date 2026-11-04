#!/usr/bin/env python3
"""Read-only sysfs temperature/block-I/O sampling during a bounded experiment.

No SMART commands, GPU instrumentation or privileged settings are used.
Temperatures are Celsius; block-stat sector counts use 512-byte sectors.
"""
import argparse
import json
from pathlib import Path
import time


def sample(process_name=None):
    result = {'time': time.time(), 'temperatures': {}, 'disks': {}}
    for hwmon in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            name = (hwmon / 'name').read_text().strip()
            for sensor in hwmon.glob('temp*_input'):
                label_path = sensor.with_name(sensor.name.replace('_input', '_label'))
                label = label_path.read_text().strip() if label_path.exists() else sensor.stem
                result['temperatures'][f'{hwmon.name}/{name}/{label}'] = int(sensor.read_text()) / 1000
        except (OSError, ValueError):
            continue
    for name in ['nvme0n1', 'sda']:
        try:
            values = list(map(int, (Path('/sys/class/block') / name / 'stat').read_text().split()))
            result['disks'][name] = dict(reads=values[0], read_sectors=values[2],
                read_ms=values[3], write_sectors=values[6], inflight=values[8],
                io_ms=values[9], weighted_io_ms=values[10])
        except (OSError, ValueError, IndexError):
            continue
    try:
        result['memory_pressure'] = Path('/proc/pressure/memory').read_text().strip()
    except OSError:
        pass
    if process_name:
        result['processes'] = {}
        for proc in Path('/proc').iterdir():
            if not proc.name.isdecimal():
                continue
            try:
                if (proc / 'comm').read_text().strip() != process_name:
                    continue
                fields = {}
                for line in (proc / 'status').read_text().splitlines():
                    key, _, value = line.partition(':')
                    if key in ('VmRSS', 'VmHWM', 'VmSwap'):
                        fields[key + '_kib'] = int(value.split()[0])
                result['processes'][proc.name] = fields
            except (OSError, ValueError, IndexError):
                continue
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=1200)
    parser.add_argument('--process', help='optional exact /proc comm name, e.g. ds4-bench')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    deadline = time.monotonic() + args.seconds
    with args.output.open('x') as stream:
        while time.monotonic() < deadline:
            stream.write(json.dumps(sample(args.process)) + '\n')
            stream.flush()
            time.sleep(min(2, max(0, deadline - time.monotonic())))
