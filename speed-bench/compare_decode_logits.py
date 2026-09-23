#!/usr/bin/env python3
"""Compare full native float32 vectors; diagnostics, not a quality acceptance gate."""
import argparse
from array import array
import json
import math
from pathlib import Path


def compare(left, right, vocab):
    size = left.stat().st_size
    if size != right.stat().st_size or not size or size % (4 * vocab):
        raise ValueError('incompatible or incomplete float32 vector files')
    count = size // (4 * vocab)
    exact = top1 = 0
    squared = absolute = maximum = kl_sum = kl_max = 0.0
    with left.open('rb') as a, right.open('rb') as b:
        for _ in range(count):
            x, y = array('f'), array('f')
            x.fromfile(a, vocab)
            y.fromfile(b, vocab)
            if not all(math.isfinite(v) for v in x) or not all(math.isfinite(v) for v in y):
                raise ValueError('non-finite logit')
            exact += x.tobytes() == y.tobytes()
            ix = max(range(vocab), key=x.__getitem__)
            iy = max(range(vocab), key=y.__getitem__)
            top1 += ix == iy
            mx, my = x[ix], y[iy]
            zx = sum(math.exp(v - mx) for v in x)
            zy = sum(math.exp(v - my) for v in y)
            # KL(p_left || p_right), with a stable log normalization difference.
            norm_diff = my - mx + math.log(zy) - math.log(zx)
            weighted = 0.0
            for u, v in zip(x, y):
                d = u - v
                maximum = max(maximum, abs(d))
                absolute += abs(d)
                squared += d * d
                weighted += math.exp(u - mx) * d
            kl = max(0.0, weighted / zx + norm_diff)
            kl_sum += kl
            kl_max = max(kl_max, kl)
    return dict(vectors=count, vocabulary=vocab, exact_vectors=exact,
                top1_matches=top1, max_abs=maximum,
                mean_abs=absolute / (count * vocab),
                rms=math.sqrt(squared / (count * vocab)),
                mean_kl_left_right=kl_sum / count, max_kl_left_right=kl_max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--vocab', type=int, default=129280)
    args = parser.parse_args()
    if args.vocab <= 0:
        parser.error('--vocab must be positive')
    print(json.dumps(compare(args.left, args.right, args.vocab), indent=2))
