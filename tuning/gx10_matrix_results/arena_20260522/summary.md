# Arena Placement Probe - 2026-05-22

Command shape:

```sh
DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=<chunk> \
DS4_CUDA_GRAPH_DECODE=1 \
DS4_CUDA_Q8_SOA_CACHE=1 \
./ds4-bench --cuda -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 \
  --gen-tokens 128
```

| Chunk | Artifact | Gen t/s | Decision |
| --- | --- | ---: | --- |
| default | `default_exact_fast_bench.csv` | 16.06 | keep |
| 1024 MiB | none | n/a | failed startup cache at tensor span 116 with OOM |
| 4096 MiB | `4096_exact_fast_bench.csv` | 15.80 | reject |
| 8192 MiB | `8192_exact_fast_bench.csv` | 15.88 | reject |

Conclusion: default arena placement remains the best tested option.
