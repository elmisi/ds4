#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from pathlib import Path


HOT_SIZES = (4, 8, 16, 32, 64, 96, 128)
N_LAYER = 43
N_EXPERT = 256
N_USED = 6


def load_csv(path):
    by_seq_layer = defaultdict(lambda: [0] * N_EXPERT)
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq_s, layer_s, expert_s, count_s = line.split(",")
            by_seq_layer[(int(seq_s), int(layer_s))][int(expert_s)] += int(count_s)
    return by_seq_layer


def top_sum(counts, hot_n):
    return sum(sorted(counts, reverse=True)[:hot_n])


def summarize_csv(by_seq_layer):
    seqs = sorted({seq for seq, _ in by_seq_layer})
    by_layer = defaultdict(lambda: [0] * N_EXPERT)
    for (_seq, layer), counts in by_seq_layer.items():
        dst = by_layer[layer]
        for expert, count in enumerate(counts):
            dst[expert] += count

    total = sum(sum(counts) for counts in by_layer.values())
    print(f"csv_requests {len(seqs)}")
    print(f"csv_selected {total}")
    print(f"csv_decode_tokens_est {total / (N_LAYER * N_USED):.1f}")

    for hot_n in HOT_SIZES:
        covered = sum(top_sum(by_layer[layer], hot_n) for layer in range(N_LAYER))
        print(f"aggregate_layer_hot_top{hot_n} {covered * 100.0 / total:.2f}%")

    for hot_n in (16, 32, 64):
        vals = []
        for seq in seqs:
            selected = sum(sum(by_seq_layer[(seq, layer)]) for layer in range(N_LAYER))
            covered = sum(top_sum(by_seq_layer[(seq, layer)], hot_n) for layer in range(N_LAYER))
            vals.append(covered * 100.0 / selected)
        print(
            f"request_oracle_top{hot_n} "
            f"mean={sum(vals) / len(vals):.2f}% min={min(vals):.2f}% max={max(vals):.2f}%"
        )

    for hot_n in (16, 32, 64):
        worst = []
        for layer in range(N_LAYER):
            counts = by_layer[layer]
            selected = sum(counts)
            covered = top_sum(counts, hot_n)
            worst.append((covered * 100.0 / selected, layer))
        print(
            f"worst_top{hot_n} "
            + " ".join(f"L{layer:02d}:{cov:.1f}%" for cov, layer in sorted(worst)[:8])
        )


def summarize_log(path):
    text = Path(path).read_text()
    selected = []
    cover = []
    overlaps = []
    hists = []
    for line in text.splitlines():
        m = re.search(
            r"selected=(\d+) decode_tokens~[0-9.]+ .*"
            r"top4=([0-9.]+)% top8=([0-9.]+)% top16=([0-9.]+)% "
            r"top32=([0-9.]+)% top64=([0-9.]+)%",
            line,
        )
        if m:
            selected.append(int(m.group(1)))
            cover.append(tuple(float(m.group(i)) for i in range(2, 7)))
            continue

        m = re.search(r"overlap avg=([0-9.]+)/6 hit=([0-9.]+)% hist\[0\.\.6\]=(.*)", line)
        if m:
            overlaps.append((float(m.group(1)), float(m.group(2))))
            hist = {}
            for part in m.group(3).split():
                bucket_s, pct_s = part.split(":")
                hist[int(bucket_s)] = float(pct_s.rstrip("%"))
            hists.append(hist)

    if not selected:
        return
    total = sum(selected)
    print(f"log_requests {len(selected)}")
    print(f"log_selected {total}")
    print(f"log_decode_tokens_est {total / (N_LAYER * N_USED):.1f}")
    for idx, hot_n in enumerate((4, 8, 16, 32, 64)):
        pct = sum(sel * cov[idx] for sel, cov in zip(selected, cover)) / total
        print(f"log_request_hot_top{hot_n} {pct:.2f}%")
    if overlaps:
        avg = sum(sel * ov[0] for sel, ov in zip(selected, overlaps)) / total
        hit = sum(sel * ov[1] for sel, ov in zip(selected, overlaps)) / total
        print(f"log_temporal_overlap {avg:.2f}/6")
        print(f"log_temporal_hit {hit:.2f}%")
        for bucket in range(7):
            pct = sum(sel * hist.get(bucket, 0.0) for sel, hist in zip(selected, hists)) / total
            print(f"log_temporal_hist_{bucket} {pct:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV written by DS4_MOE_EXPERT_STATS_DUMP")
    ap.add_argument("--log", help="server stderr log containing moe expert stats lines")
    args = ap.parse_args()
    if not args.csv and not args.log:
        ap.error("provide --csv, --log, or both")
    if args.csv:
        summarize_csv(load_csv(args.csv))
    if args.log:
        summarize_log(args.log)


if __name__ == "__main__":
    main()
