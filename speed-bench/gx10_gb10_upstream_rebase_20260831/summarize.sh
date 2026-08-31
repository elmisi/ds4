#!/usr/bin/env bash
set -euo pipefail

artifact_dir=$(cd "$(dirname "$0")" && pwd)
root_dir=$(cd "$artifact_dir/../.." && pwd)

printf '%s\n' 'ctx_tokens,baseline_prefill_tps,candidate_prefill_tps,prefill_delta_percent,baseline_gen_tps,candidate_gen_tps,gen_delta_percent,kvcache_match' \
    >"$artifact_dir/comparison.csv"

awk -F, '
    NR == FNR {
        if (FNR > 1) {
            bp[$1] = $3
            bg[$1] = $5
            bk[$1] = $9
        }
        next
    }
    FNR > 1 {
        printf "%s,%.2f,%.2f,%.6f,%.2f,%.2f,%.6f,%s\n",
            $1, bp[$1], $3, (($3 / bp[$1]) - 1) * 100,
            bg[$1], $5, (($5 / bg[$1]) - 1) * 100,
            (bk[$1] == $9 ? "yes" : "no")
    }
' "$artifact_dir/baseline.csv" "$artifact_dir/candidate.csv" \
    >>"$artifact_dir/comparison.csv"

printf '%s\n' 'metric,baseline,candidate,delta_percent,samples' \
    >"$artifact_dir/summary.csv"

awk -F, '
    NR == FNR {
        if (FNR > 1) {
            bp += $3
            bg += $5
            n += 1
        }
        next
    }
    FNR > 1 {
        cp += $3
        cg += $5
    }
    END {
        printf "sweep_prefill_mean_tps,%.6f,%.6f,%.6f,%d\n", bp/n, cp/n, ((cp/bp)-1)*100, n
        printf "sweep_generation_mean_tps,%.6f,%.6f,%.6f,%d\n", bg/n, cg/n, ((cg/bg)-1)*100, n
    }
' "$artifact_dir/baseline.csv" "$artifact_dir/candidate.csv" \
    >>"$artifact_dir/summary.csv"

awk -F, '
    FNR > 1 {
        if (FILENAME ~ /baseline/) {
            bp += $3
            bg += $5
            bn += 1
        } else {
            cp += $3
            cg += $5
            cn += 1
        }
    }
    END {
        printf "repeat_32k_prefill_mean_tps,%.6f,%.6f,%.6f,%d\n", bp/bn, cp/cn, ((cp/cn)/(bp/bn)-1)*100, bn
        printf "repeat_32k_generation_mean_tps,%.6f,%.6f,%.6f,%d\n", bg/bn, cg/cn, ((cg/cn)/(bg/bn)-1)*100, bn
    }
' "$artifact_dir"/decode_baseline_{1,2,3}.csv \
  "$artifact_dir"/decode_candidate_{1,2,3}.csv \
    >>"$artifact_dir/summary.csv"

python3 "$root_dir/speed-bench/plot_speed.py" "$artifact_dir/baseline.csv" \
    --title 'GB10 pre-rebase baseline' -o "$artifact_dir/baseline_ts.svg"
python3 "$root_dir/speed-bench/plot_speed.py" "$artifact_dir/candidate.csv" \
    --title 'GB10 post-rebase candidate' -o "$artifact_dir/candidate_ts.svg"
