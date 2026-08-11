# GB10 `dgx-performance` current safe snapshot — 2026-08-11

Fresh NVIDIA GB10 / DGX Spark measurement of `dgx-performance` after removing
the unsafe GB10 decode-fusion bundle. The sweep uses the same 0731 Flash GGUF,
standard *Promessi sposi* prompt, 2,048-token incremental prefill, and 128
greedy generation tokens as the `DGX_PERFORMANCE.md` speed table.

## Command

```sh
DS4_BENCH_FORCE_SNAPSHOT=1 DS4_CUDA_DECODE_GRAPHS=0 \
  ./ds4-bench --cuda -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 \
  --gen-tokens 128 --csv dgx_performance.csv
```

`dgx_performance.csv` contains all 32 measured frontiers. The final row's
`kvcache_bytes=0` is the established `ds4-bench` end-frontier representation;
the preceding 63,488-token row records the allocated cache size.

## Table rows

| Context | Prefill | Generation |
| ---: | ---: | ---: |
| 2048 | 830.90 t/s | 21.62 t/s |
| 16384 | 869.54 t/s | 17.53 t/s |
| 32768 | 850.89 t/s | 16.66 t/s |
| 65536 | 821.55 t/s | 15.86 t/s |
