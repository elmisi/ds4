# GB10 tile-8 MoE down-projection A/B

This is the acceptance benchmark for making the tile-8 row-span MoE down
projection the default on GB10/sm_121. It measures the DeepSeek V4 Flash 0731
GGUF used by the DGX agent, with a 262,144-token allocated context.

## Reproduction

Run each variant from a fresh process, alternating variants when collecting
multiple samples:

```sh
./ds4-bench --cuda \
  -m /home/alessandro/projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 2048 --ctx-max 2048 --ctx-alloc 262144 \
  --gen-tokens 128 --prefill-chunk 2048 --gpu-vram auto

DS4_CUDA_NO_MOE_DOWN_TILE8_ROWSPAN=1 ./ds4-bench --cuda \
  -m /home/alessandro/projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 2048 --ctx-max 2048 --ctx-alloc 262144 \
  --gen-tokens 128 --prefill-chunk 2048 --gpu-vram auto
```

The first command is tile8 on GB10; the second is the former path. Both runs
planned 85.31 GiB for the resident model, KV cache, and buffers.

## Result

| Variant | Prefill t/s (mean) | Generation t/s (mean) | Steady generation t/s (mean) |
| --- | ---: | ---: | ---: |
| Former path | 369.18 | 18.30 | 18.46 |
| Tile8 default | 389.98 | 18.34 | 18.50 |
| Delta | **+5.6%** | +0.2% | +0.2% |

Tile8 is a material prefill improvement and generation-neutral within the two
sample A/B. The raw results are in [`tile8_ab.csv`](tile8_ab.csv).
