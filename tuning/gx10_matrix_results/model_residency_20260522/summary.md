# Full Model Copy Probe - 2026-05-22

Command shape:

```sh
DS4_CUDA_COPY_MODEL=1 \
DS4_CUDA_GRAPH_DECODE=1 \
DS4_CUDA_Q8_SOA_CACHE=1 \
./ds4-bench --cuda -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 \
  --gen-tokens 128
```

Startup copied the full 80.76 GiB model image to device memory and completed in
457.539s.

| Artifact | Prefill t/s | Gen t/s | Decision |
| --- | ---: | ---: | --- |
| `copy_model_exact_fast_bench.csv` | 223.08 | 15.75 | reject |

Adjacent/default exact-fast controls in this bench protocol were around
16.05-16.06 t/s, so the device-owned full-image placement is slower at steady
decode and much worse operationally.
