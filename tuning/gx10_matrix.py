#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "ds4flash.gguf"
DEFAULT_MTP_MODEL = "gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"


BASE_UNSET_NAMES = {
    "DS4_CUDA_GRAPH_DECODE",
    "DS4_CUDA_GRAPH_DECODE_NO_SYNC",
    "DS4_CUDA_GRAPH_VERIFY",
    "DS4_CUDA_DIRECT_MODEL",
    "DS4_CUDA_OUTPUT_TOP1",
    "DS4_CUDA_FORCE_ORDERED_F16_MATMUL",
    "DS4_CUDA_NO_ORDERED_F16_MATMUL",
    "DS4_CUDA_Q8_CUBLAS_DECODE",
    "DS4_CUDA_Q8_BATCH1_CACHE_X",
    "DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN",
    "DS4_CUDA_ATTENTION_OUTPUT_B_CUBLAS_MIN",
    "DS4_CUDA_ATTENTION_OUTPUT_A_HWARP16",
    "DS4_CUDA_ATTENTION_OUTPUT_A_SHAPE8192",
    "DS4_CUDA_ATTENTION_OUTPUT_A_CACHE_X16",
    "DS4_CUDA_ATTN_Q_B_CUBLAS_DECODE",
    "DS4_CUDA_ATTN_Q_B_HWARP16",
    "DS4_CUDA_ATTN_Q_B_B32_SPECIAL",
    "DS4_CUDA_Q8_F16_ALL",
    "DS4_CUDA_Q8_F32_ALL",
    "DS4_CUDA_Q8_F16_PRELOAD",
    "DS4_CUDA_Q8_F32_PRELOAD",
    "DS4_CUDA_HC_EXPAND_NHC4_SPECIAL",
    "DS4_CUDA_HC_EXPAND_NO_BLOCK_OUT",
    "DS4_CUDA_Q8_SOA_CACHE",
    "DS4_CUDA_Q8_SOA_BATCH2",
    "DS4_CUDA_Q8_SOA_BATCH2_ATTN_OUTPUT_B",
    "DS4_CUDA_Q8_SOA_QB",
    "DS4_CUDA_Q8_SOA_QKV",
    "DS4_CUDA_Q8_SOA_SHARED",
    "DS4_CUDA_Q8_SOA_HC_EXPAND",
    "DS4_CUDA_Q8_SOA_ALL",
    "DS4_CUDA_Q8_SOA_CACHE_X",
    "DS4_CUDA_Q8_SOA_ATTN_OUTPUT_B_DECODE",
    "DS4_CUDA_Q8_SOA_NO_ATTN_OUTPUT_A",
    "DS4_CUDA_Q8_SOA_NO_ATTN_OUTPUT_B",
    "DS4_CUDA_SHARED_GATE_UP_NOAUX",
    "DS4_CUDA_SHARED_GATE_UP_SHAPE2048",
    "DS4_CUDA_FFN_PARALLEL_SHARED",
    "DS4_CUDA_FFN_SHARED_FIRST",
    "DS4_CUDA_MOE_ACTIVE_EXPERTS",
    "DS4_CUDA_MOE_DECODE_GATE_H16",
    "DS4_CUDA_MOE_DECODE_GATE_NOAUX",
    "DS4_CUDA_MOE_DECODE_GATE_PAIR2",
    "DS4_CUDA_MOE_DECODE_FUSED_MIDQ",
    "DS4_CUDA_MOE_DECODE_GATE_WEIGHT_CACHE",
    "DS4_CUDA_MOE_DECODE_GATE_SPAN128_TEMPLATE",
    "DS4_CUDA_MOE_DECODE_GATE_GLOBAL_LUT",
    "DS4_CUDA_MOE_DECODE_GATE_MAXR48",
    "DS4_CUDA_MOE_DECODE_GATE_LDG",
    "DS4_CUDA_MOE_DECODE_GATE_SHAPE2048",
    "DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_CONSTSTRIDE",
    "DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_CONSTCLAMP",
    "DS4_CUDA_MOE_GATE_PREFER_L1",
    "DS4_CUDA_MOE_DECODE_GATE_SPAN",
    "DS4_CUDA_TOPK_CHUNK8192",
    "DS4_CUDA_WEIGHT_TENSOR_ALIGN_MB",
    "DS4_CUDA_MOE_DOWN_SUM6_PARALLEL",
    "DS4_CUDA_MOE_DOWN_SUM6_ROW4",
    "DS4_CUDA_MOE_DOWN_SUM6_META_CACHE",
    "DS4_CUDA_MOE_DOWN_SUM6_LDG",
    "DS4_CUDA_MOE_DOWN_SUM6_SHAPE4096",
    "DS4_CUDA_MOE_K2_DIRECT_GATE",
    "DS4_CUDA_MOE_PROFILE",
    "DS4_CUDA_WEIGHT_CACHE_VERBOSE",
    "DS4_SAMPLE_CACHE_PROBS",
    "DS4_METAL_DECODE_STAGE_PROFILE",
    "DS4_METAL_GRAPH_TOKEN_PROFILE",
    "DS4_MOE_ACTIVE_EXPERTS",
    "DS4_MOE_ACTIVE_EXPERTS_RENORM",
    "DS4_MOE_ACTIVE_EXPERTS_LAYERS",
}

BASE_UNSET_PREFIXES = ("DS4_MTP_",)


def env(**kwargs):
    return {key: str(value) for key, value in kwargs.items()}


EXACT_FAST = env(
    DS4_CUDA_GRAPH_DECODE=1,
    DS4_CUDA_Q8_SOA_CACHE=1,
)


MTP_QUALITY = {
    **EXACT_FAST,
    **env(
        DS4_MTP_STRICT=1,
        DS4_MTP_BATCH_VERIFY=1,
        DS4_MTP_UNSAFE_BATCH_VERIFY=1,
        DS4_MTP_BATCH_MARGIN_GUARD=0.25,
        DS4_MTP_BATCH_HC_PRE_EXACT=1,
        DS4_MTP_BATCH_ATTENTION_EXACT=1,
        DS4_MTP_BATCH_COMPRESS_PROJ_EXACT=1,
        DS4_MTP_BATCH_ROUTER_EXACT=1,
        DS4_MTP_CAPTURE_PREFIX1=1,
        DS4_MTP_CAPTURE_PREFIX1_MIN_MARGIN=2.0,
        DS4_MTP_DRAFT2_SKIP_MIN_MARGIN=2.0,
    ),
}


MATRIX = {
    "plain": {
        "category": "baseline",
        "env": {},
        "status": "current branch with sanitized experiment env",
    },
    "graph": {
        "category": "exact-safe",
        "env": env(DS4_CUDA_GRAPH_DECODE=1),
        "status": "isolates CUDA graph decode",
    },
    "soa": {
        "category": "exact-safe",
        "env": env(DS4_CUDA_Q8_SOA_CACHE=1),
        "status": "isolates default Q8 SoA cache",
    },
    "exact_fast": {
        "category": "production",
        "env": EXACT_FAST,
        "status": "recommended exact-safe row",
    },
    "soa_b_forced": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_Q8_SOA_ATTN_OUTPUT_B_DECODE=1)},
        "status": "quality passed, slower than default SoA",
    },
    "soa_qb": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_Q8_SOA_QB=1)},
        "status": "micro-positive, full decode neutral",
    },
    "attn_qb_hwarp16": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_ATTN_Q_B_HWARP16=1)},
        "status": "half-warp attn_q_b kernel, reduction-order probe",
    },
    "attn_qb_soa_hwarp16": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_Q8_SOA_QB=1, DS4_CUDA_ATTN_Q_B_HWARP16=1)},
        "status": "SoA half-warp attn_q_b kernel, reduction-order probe",
    },
    "attn_qb_b32_special": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_ATTN_Q_B_B32_SPECIAL=1)},
        "status": "exact-order attn_q_b blocks=32 specialized kernel",
    },
    "soa_qkv": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_Q8_SOA_QKV=1)},
        "status": "q/kv micro-positive, fused route regressed",
    },
    "soa_shared": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_Q8_SOA_SHARED=1)},
        "status": "memory-safe, neutral, not byte-identical in stored logprobs",
    },
    "soa_hc_expand": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_Q8_SOA_HC_EXPAND=1)},
        "status": "target only the remaining HC-expand Q8 path",
    },
    "hc_expand_nhc4_special": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_HC_EXPAND_NHC4_SPECIAL=1)},
        "status": "exact-order n_hc=4 HC-expand specialization",
    },
    "hc_expand_no_block_out": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_HC_EXPAND_NO_BLOCK_OUT=1)},
        "status": "HC-expand fused path skips auxiliary block_out store",
    },
    "shared_gate_up_noaux": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_SHARED_GATE_UP_NOAUX=1)},
        "status": "shared expert gate/up writes only mid output",
    },
    "shared_gate_up_shape2048": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_SHARED_GATE_UP_SHAPE2048=1)},
        "status": "shape-specialized DS4 shared expert gate/up Q8 swiglu kernel",
    },
    "soa_cache_x": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_Q8_SOA_CACHE_X=1)},
        "status": "noisy/unstable and slower",
    },
    "output_top1": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_OUTPUT_TOP1=1)},
        "status": "exact but slower in A/B",
    },
    "attn_b_cublas_min1": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_ATTENTION_OUTPUT_B_CUBLAS_MIN=1)},
        "status": "did not improve refreshed profile",
    },
    "attn_a_hwarp16": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_ATTENTION_OUTPUT_A_HWARP16=1)},
        "status": "attention-output-A SoA half-warp reduction probe",
    },
    "attn_a_shape8192": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_ATTENTION_OUTPUT_A_SHAPE8192=1)},
        "status": "exact-order DS4-shape attention-output-A SoA kernel",
    },
    "attn_a_cache_x16": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_ATTENTION_OUTPUT_A_CACHE_X16=1)},
        "status": "exact-order attention-output-A SoA kernel, 16 rows per CTA with shared x",
    },
    "moe_h16": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DECODE_GATE_H16=1)},
        "status": "negative",
    },
    "moe_noaux": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DECODE_GATE_NOAUX=1)},
        "status": "neutral/negative",
    },
    "moe_pair2": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DECODE_GATE_PAIR2=1)},
        "status": "byte-identical but slower",
    },
    "moe_fused_midq": {
        "category": "diagnostic",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DECODE_FUSED_MIDQ=1)},
        "status": "byte-identical but slower",
    },
    "moe_down_meta_cache": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DOWN_SUM6_META_CACHE=1)},
        "status": "exact down-sum6 selected-expert metadata hoist",
    },
    "moe_gate_weight_cache": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DECODE_GATE_WEIGHT_CACHE=1)},
        "status": "exact gate/up route-weight shared cache",
    },
    "moe_span128_template": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DECODE_GATE_SPAN128_TEMPLATE=1)},
        "status": "explicit span<128> decode gate template",
    },
    "moe_global_lut": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DECODE_GATE_GLOBAL_LUT=1)},
        "status": "global IQ2 LUT instead of per-CTA shared LUT copy",
    },
    "moe_gate_maxr48": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DECODE_GATE_MAXR48=1)},
        "status": "max 48 registers routed MoE gate/up occupancy probe",
    },
    "moe_gate_ldg": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DECODE_GATE_LDG=1)},
        "status": "read-only-cache loads for decode routed MoE gate/up weights",
    },
    "moe_down_ldg": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DOWN_SUM6_LDG=1)},
        "status": "read-only-cache loads for decode routed MoE down weights",
    },
    "moe_ldg_weights": {
        "category": "prototype",
        "env": {
            **EXACT_FAST,
            **env(DS4_CUDA_MOE_DECODE_GATE_LDG=1, DS4_CUDA_MOE_DOWN_SUM6_LDG=1),
        },
        "status": "read-only-cache loads for both hot decode routed MoE weight paths",
    },
    "moe_gate_shape2048": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DECODE_GATE_SHAPE2048=1)},
        "status": "shape-specialized DS4 decode routed MoE gate/up kernel",
    },
    "moe_gate_shape2048_conststride": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_CONSTSTRIDE=1)},
        "status": "shape2048 routed MoE gate/up with DS4 constant strides",
    },
    "moe_gate_shape2048_constclamp": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_CONSTCLAMP=1)},
        "status": "shape2048 routed MoE gate/up with DS4 constant strides and clamp",
    },
    "moe_gate_prefer_l1": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_GATE_PREFER_L1=1)},
        "status": "routed MoE gate/up kernel cache config prefers L1",
    },
    "moe_gate_shape2048_l1": {
        "category": "prototype",
        "env": {
            **EXACT_FAST,
            **env(DS4_CUDA_MOE_DECODE_GATE_SHAPE2048=1, DS4_CUDA_MOE_GATE_PREFER_L1=1),
        },
        "status": "shape-specialized routed MoE gate/up plus prefer-L1 cache config",
    },
    "moe_down_shape4096": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_MOE_DOWN_SUM6_SHAPE4096=1)},
        "status": "shape-specialized DS4 decode routed MoE down-sum6 kernel",
    },
    "moe_shape_special": {
        "category": "prototype",
        "env": {
            **EXACT_FAST,
            **env(
                DS4_CUDA_MOE_DECODE_GATE_SHAPE2048=1,
                DS4_CUDA_MOE_DOWN_SUM6_SHAPE4096=1,
            ),
        },
        "status": "shape-specialized DS4 decode routed MoE gate/up and down kernels",
    },
    "shape_gate_shared": {
        "category": "prototype",
        "env": {
            **EXACT_FAST,
            **env(
                DS4_CUDA_MOE_DECODE_GATE_SHAPE2048=1,
                DS4_CUDA_SHARED_GATE_UP_SHAPE2048=1,
            ),
        },
        "status": "shape-specialized routed gate/up plus shared gate/up kernels",
    },
    "shape_gate_attn_a": {
        "category": "prototype",
        "env": {
            **EXACT_FAST,
            **env(
                DS4_CUDA_MOE_DECODE_GATE_SHAPE2048=1,
                DS4_CUDA_ATTENTION_OUTPUT_A_SHAPE8192=1,
            ),
        },
        "status": "shape-specialized routed gate/up plus attention-output-A SoA kernels",
    },
    "indexer_topk_chunk8192": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_TOPK_CHUNK8192=1)},
        "status": "long-context exact top-k chunking 8192 vs default 4096",
    },
    "graph_no_presync": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_GRAPH_DECODE_NO_SYNC=1)},
        "status": "normal decode graph capture without the pre-capture device synchronize",
    },
    "weight_tensor_align2m": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_WEIGHT_TENSOR_ALIGN_MB=2)},
        "status": "2 MiB device-base alignment for cached model tensors",
    },
    "q8_batch1_cache_x": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_Q8_BATCH1_CACHE_X=1)},
        "status": "use cached-x warp8 kernel for n_tok=1 Q8 projections with <=32 blocks",
    },
    "sample_cache_probs": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_SAMPLE_CACHE_PROBS=1)},
        "status": "cache default sampler probabilities to avoid duplicate full-vocab expf pass",
    },
    "ffn_parallel_shared": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_FFN_PARALLEL_SHARED=1)},
        "status": "parallel shared-expert gate/up with router+routed MoE via a second CUDA stream",
    },
    "ffn_shared_first": {
        "category": "prototype",
        "env": {**EXACT_FAST, **env(DS4_CUDA_FFN_SHARED_FIRST=1)},
        "status": "run shared-expert gate/up immediately after ffn_norm before router+routed MoE",
    },
    "moe_meta_cache": {
        "category": "prototype",
        "env": {
            **EXACT_FAST,
            **env(
                DS4_CUDA_MOE_DECODE_GATE_WEIGHT_CACHE=1,
                DS4_CUDA_MOE_DOWN_SUM6_META_CACHE=1,
            ),
        },
        "status": "combined exact MoE metadata caches",
    },
    "k3_renorm": {
        "category": "quality-tradeoff",
        "env": {
            **EXACT_FAST,
            **env(DS4_MOE_ACTIVE_EXPERTS=3, DS4_MOE_ACTIVE_EXPERTS_RENORM=1),
        },
        "status": "crosses 20 t/s in server logs, not exact full-K",
    },
    "k2_renorm": {
        "category": "quality-tradeoff",
        "env": {
            **EXACT_FAST,
            **env(DS4_MOE_ACTIVE_EXPERTS=2, DS4_MOE_ACTIVE_EXPERTS_RENORM=1),
        },
        "status": "fastest reduced-K row, coding unsafe in smoke",
    },
    "k6_0_2_k3_renorm": {
        "category": "quality-tradeoff",
        "env": {
            **EXACT_FAST,
            **env(
                DS4_MOE_ACTIVE_EXPERTS=3,
                DS4_MOE_ACTIVE_EXPERTS_RENORM=1,
                DS4_MOE_ACTIVE_EXPERTS_LAYERS="0-2:6",
            ),
        },
        "status": "hit target in smoke, 12-task eval regressed",
    },
    "mtp_quality": {
        "category": "mtp-research",
        "env": MTP_QUALITY,
        "needs_mtp": True,
        "status": "quality-first MTP candidate, slower than no-MTP exact-fast",
    },
    "mtp_attn_b_soa_batch2": {
        "category": "mtp-research",
        "env": {**MTP_QUALITY, **env(DS4_CUDA_Q8_SOA_BATCH2_ATTN_OUTPUT_B=1)},
        "needs_mtp": True,
        "status": "narrow MTP output-projection diagnostic",
    },
}


CANARY_TASKS = "merge_intervals,parse_duration,flatten_dict,valid_brackets"

ROW_GROUPS = {
    "core": ["plain", "graph", "soa", "exact_fast"],
    "exact": [
        "plain",
        "graph",
        "soa",
        "exact_fast",
        "soa_b_forced",
        "soa_qb",
        "attn_qb_hwarp16",
        "attn_qb_soa_hwarp16",
        "attn_qb_b32_special",
        "soa_qkv",
        "soa_shared",
        "soa_hc_expand",
        "hc_expand_nhc4_special",
        "hc_expand_no_block_out",
        "shared_gate_up_noaux",
        "shared_gate_up_shape2048",
        "soa_cache_x",
        "output_top1",
        "attn_b_cublas_min1",
        "attn_a_hwarp16",
        "moe_h16",
        "moe_noaux",
        "moe_pair2",
        "moe_fused_midq",
        "moe_down_meta_cache",
        "moe_gate_weight_cache",
        "moe_span128_template",
        "moe_global_lut",
        "moe_gate_maxr48",
        "moe_gate_ldg",
        "moe_down_ldg",
        "moe_ldg_weights",
        "moe_gate_shape2048",
        "moe_gate_shape2048_conststride",
        "moe_gate_shape2048_constclamp",
        "moe_down_shape4096",
        "moe_shape_special",
        "shape_gate_shared",
        "indexer_topk_chunk8192",
        "graph_no_presync",
        "weight_tensor_align2m",
        "q8_batch1_cache_x",
        "sample_cache_probs",
        "ffn_parallel_shared",
        "ffn_shared_first",
        "moe_meta_cache",
    ],
    "tradeoff": ["k3_renorm", "k2_renorm", "k6_0_2_k3_renorm"],
    "mtp": ["mtp_quality", "mtp_attn_b_soa_batch2"],
}
ROW_GROUPS["all"] = ROW_GROUPS["exact"] + ROW_GROUPS["tradeoff"] + ROW_GROUPS["mtp"]


def clean_env(extra):
    merged = dict(os.environ)
    for key in list(merged):
        if key in BASE_UNSET_NAMES or any(key.startswith(prefix) for prefix in BASE_UNSET_PREFIXES):
            merged.pop(key, None)
    merged.update(extra)
    return merged


def variant(name):
    try:
        return MATRIX[name]
    except KeyError:
        raise SystemExit(f"unknown matrix row: {name}")


def expand_rows(names, default_group="core"):
    if not names:
        names = [default_group]
    rows = []
    for name in names:
        if name in ROW_GROUPS:
            rows.extend(ROW_GROUPS[name])
        else:
            variant(name)
            rows.append(name)
    out = []
    seen = set()
    for name in rows:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def describe_env(extra):
    if not extra:
        return "(none)"
    return " ".join(f"{key}={value}" for key, value in sorted(extra.items()))


def print_command(cmd, env_delta):
    print("env " + describe_env(env_delta))
    print(" ".join(str(part) for part in cmd))


def server_cmd(row, args):
    cmd = [
        "./ds4-server",
        "--cuda",
        "-m",
        args.model,
        "--ctx",
        str(args.ctx),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tokens",
        str(args.tokens),
    ]
    if row.get("needs_mtp"):
        cmd += ["--mtp", args.mtp_model, "--mtp-draft", str(args.mtp_draft)]
    cmd += flatten_extra(args.extra)
    return cmd


def bench_cmd(row_name, row, args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{row_name}_bench.csv"
    cmd = [
        "./ds4-bench",
        "--cuda",
        "-m",
        args.model,
        "--prompt-file",
        args.prompt_file,
        "--ctx-start",
        str(args.ctx_start),
        "--ctx-max",
        str(args.ctx_max),
        "--ctx-alloc",
        str(args.ctx_alloc),
        "--gen-tokens",
        str(args.gen_tokens),
        "--csv",
        str(csv_path),
    ]
    if row.get("needs_mtp"):
        cmd += ["--mtp", args.mtp_model, "--mtp-draft", str(args.mtp_draft)]
    cmd += flatten_extra(args.extra)
    return cmd


def bench_output_path(row_name, out_dir):
    return Path(out_dir) / f"{row_name}_bench.csv"


def flatten_extra(values):
    out = []
    for value in values:
        out.extend(shlex.split(value))
    return out


def wait_for_server(base_url, proc, timeout):
    deadline = time.time() + timeout
    url = base_url.rstrip("/") + "/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    return False


def stop_proc(proc):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def cmd_list(_args):
    width = max(len(name) for name in MATRIX)
    for name, row in MATRIX.items():
        print(f"{name:<{width}}  {row['category']:<16}  {row['status']}")


def cmd_env(args):
    row = variant(args.name)
    for key, value in sorted(row["env"].items()):
        print(f"export {key}={value!r}")


def cmd_server(args):
    row = variant(args.name)
    cmd = server_cmd(row, args)
    if args.dry_run:
        print_command(cmd, row["env"])
        return 0
    os.execvpe(cmd[0], cmd, clean_env(row["env"]))


def cmd_bench(args):
    row = variant(args.name)
    cmd = bench_cmd(args.name, row, args)
    if args.dry_run:
        print_command(cmd, row["env"])
        return 0
    return subprocess.run(cmd, cwd=ROOT, env=clean_env(row["env"])).returncode


def cmd_bench_suite(args):
    rows = expand_rows(args.rows, default_group=args.group)
    failures = []
    for row_name in rows:
        row = variant(row_name)
        cmd = bench_cmd(row_name, row, args)
        print(f"== bench {row_name} ==", flush=True)
        if args.dry_run:
            print_command(cmd, row["env"])
            continue
        rc = subprocess.run(cmd, cwd=ROOT, env=clean_env(row["env"])).returncode
        if rc != 0:
            failures.append((row_name, rc))
            if args.stop_on_fail:
                break
    if args.dry_run:
        return 0
    cmd_summary(argparse.Namespace(out_dir=args.out_dir, output=args.summary, markdown=args.markdown))
    if failures:
        for row_name, rc in failures:
            print(f"FAILED {row_name}: exit {rc}", file=sys.stderr)
        return failures[0][1]
    return 0


def cmd_run(args):
    row = variant(args.name)
    if args.dry_run:
        print_command(args.command, row["env"])
        return 0
    return subprocess.run(args.command, cwd=ROOT, env=clean_env(row["env"])).returncode


def cmd_eval(args):
    row = variant(args.name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or args.name
    server_log = out_dir / f"{label}_server.log"
    base_url = f"http://{args.host}:{args.port}"

    ns = argparse.Namespace(
        model=args.model,
        mtp_model=args.mtp_model,
        mtp_draft=args.mtp_draft,
        ctx=args.ctx,
        host=args.host,
        port=args.port,
        tokens=args.tokens,
        extra=args.server_extra,
    )
    server = server_cmd(row, ns)
    eval_cmd = [
        sys.executable,
        "tuning/coding_eval_extended.py",
        "--base-url",
        base_url,
        "--label",
        label,
        "--max-tokens",
        str(args.max_tokens),
        "--repeat",
        str(args.repeat),
        "--out-dir",
        str(out_dir),
    ]
    if args.canary:
        eval_cmd += ["--only", CANARY_TASKS]
    if args.only:
        for item in args.only:
            eval_cmd += ["--only", item]

    if args.dry_run:
        print_command(server, row["env"])
        print("then")
        print(" ".join(eval_cmd))
        print(f"server log: {server_log}")
        return 0

    with server_log.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            server,
            cwd=ROOT,
            env=clean_env(row["env"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            if not wait_for_server(base_url, proc, args.startup_timeout):
                print(f"server did not become ready; see {server_log}", file=sys.stderr)
                return proc.returncode if proc.poll() is not None else 2
            return subprocess.run(eval_cmd, cwd=ROOT).returncode
        finally:
            stop_proc(proc)


def cmd_eval_suite(args):
    rows = expand_rows(args.rows, default_group=args.group)
    failures = []
    port = args.port
    for row_name in rows:
        print(f"== eval {row_name} ==", flush=True)
        ns = argparse.Namespace(**vars(args))
        ns.name = row_name
        ns.label = args.label_prefix + row_name
        ns.port = port
        rc = cmd_eval(ns)
        if rc != 0:
            failures.append((row_name, rc))
            if args.stop_on_fail:
                break
        port += 1
    if failures:
        for row_name, rc in failures:
            print(f"FAILED {row_name}: exit {rc}", file=sys.stderr)
        return failures[0][1]
    return 0


def ds_eval_cmd(row_name, row, args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or row_name
    trace = args.trace or out_dir / f"{label}_ds4_eval.txt"
    cmd = [
        "./ds4-eval",
        "--cuda",
        "-m",
        args.model,
        "--tokens",
        str(args.tokens),
        "--seed",
        str(args.seed),
        "--trace",
        str(trace),
        "--plain",
    ]
    if args.questions:
        cmd += ["--questions", str(args.questions)]
    if args.quality:
        cmd += ["--quality"]
    if args.warm_weights:
        cmd += ["--warm-weights"]
    if args.think:
        cmd += ["--think"]
    if args.nothink:
        cmd += ["--nothink"]
    if row.get("needs_mtp"):
        cmd += ["--mtp", args.mtp_model]
    cmd += flatten_extra(args.extra)
    return cmd, trace


def cmd_ds_eval(args):
    if args.think and args.nothink:
        print("use only one of --think or --nothink", file=sys.stderr)
        return 2
    row = variant(args.name)
    cmd, trace = ds_eval_cmd(args.name, row, args)
    if args.dry_run:
        print_command(cmd, row["env"])
        print(f"trace: {trace}")
        return 0
    try:
        return subprocess.run(
            cmd,
            cwd=ROOT,
            env=clean_env(row["env"]),
            timeout=args.timeout_sec if args.timeout_sec > 0 else None,
        ).returncode
    except subprocess.TimeoutExpired:
        print(f"ds4-eval timed out after {args.timeout_sec}s; trace: {trace}", file=sys.stderr)
        return 124


def cmd_ds_eval_suite(args):
    rows = expand_rows(args.rows, default_group=args.group)
    failures = []
    for row_name in rows:
        print(f"== ds-eval {row_name} ==", flush=True)
        ns = argparse.Namespace(**vars(args))
        ns.name = row_name
        ns.label = args.label_prefix + row_name
        ns.trace = None
        rc = cmd_ds_eval(ns)
        if rc != 0:
            failures.append((row_name, rc))
            if args.stop_on_fail:
                break
    if failures:
        for row_name, rc in failures:
            print(f"FAILED {row_name}: exit {rc}", file=sys.stderr)
        return failures[0][1]
    return 0


def summarize_csv(path):
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        return None
    gen = [float(row["gen_tps"]) for row in rows if row.get("gen_tps")]
    prefill = [float(row["prefill_tps"]) for row in rows if row.get("prefill_tps")]
    steady_key = "gen_tps_ss" if "gen_tps_ss" in rows[0] else "gen_tps"
    steady = [float(row[steady_key]) for row in rows if row.get(steady_key)]
    by_ctx = {int(row["ctx_tokens"]): row for row in rows if row.get("ctx_tokens")}
    first = rows[0]
    last = rows[-1]
    def value_at(ctx, key):
        row = by_ctx.get(ctx)
        return float(row[key]) if row and row.get(key) else None
    return {
        "row": path.name.removesuffix("_bench.csv"),
        "rows": len(rows),
        "ctx_first": int(first["ctx_tokens"]),
        "ctx_last": int(last["ctx_tokens"]),
        "gen_first": float(first["gen_tps"]),
        "gen_8192": value_at(8192, "gen_tps"),
        "gen_last": float(last["gen_tps"]),
        "gen_mean": sum(gen) / len(gen) if gen else None,
        "gen_max": max(gen) if gen else None,
        "gen_ss_mean": sum(steady) / len(steady) if steady else None,
        "prefill_mean": sum(prefill) / len(prefill) if prefill else None,
        "prefill_max": max(prefill) if prefill else None,
    }


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}"


def write_summary_markdown(items, path):
    lines = [
        "# GX10 Matrix Summary",
        "",
        "| Row | Rows | Ctx first | Ctx last | Gen first | Gen @8192 | Gen last | Gen mean | Gen max | Prefill mean | Prefill max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in items:
        lines.append(
            "| {row} | {rows} | {ctx_first} | {ctx_last} | {gen_first} | {gen_8192} | "
            "{gen_last} | {gen_mean} | {gen_max} | {prefill_mean} | {prefill_max} |".format(
                row=item["row"],
                rows=item["rows"],
                ctx_first=item["ctx_first"],
                ctx_last=item["ctx_last"],
                gen_first=fmt(item["gen_first"]),
                gen_8192=fmt(item["gen_8192"]),
                gen_last=fmt(item["gen_last"]),
                gen_mean=fmt(item["gen_mean"]),
                gen_max=fmt(item["gen_max"]),
                prefill_mean=fmt(item["prefill_mean"]),
                prefill_max=fmt(item["prefill_max"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_summary(args):
    out_dir = Path(args.out_dir)
    paths = sorted(out_dir.glob("*_bench.csv"))
    items = [item for path in paths if (item := summarize_csv(path))]
    if not items:
        print(f"no bench CSV files found in {out_dir}", file=sys.stderr)
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = list(items[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)
    markdown = Path(args.markdown)
    write_summary_markdown(items, markdown)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {markdown}")
    print(f"wrote {json_path}")
    return 0


def add_common_runtime(ap):
    ap.add_argument("--model", default=os.environ.get("DS4_MODEL", DEFAULT_MODEL))
    ap.add_argument("--mtp-model", default=os.environ.get("DS4_MTP_MODEL", DEFAULT_MTP_MODEL))
    ap.add_argument("--mtp-draft", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")


def main():
    parser = argparse.ArgumentParser(description="Run the GX10 tuning matrix with sanitized env.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="List matrix rows.")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("env", help="Print exports for a matrix row.")
    p.add_argument("name")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("server", help="Exec ds4-server for a matrix row.")
    p.add_argument("name")
    add_common_runtime(p)
    p.add_argument("--ctx", type=int, default=100000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8106)
    p.add_argument("--tokens", type=int, default=900)
    p.add_argument("--extra", action="append", default=[], metavar="ARG")
    p.set_defaults(func=cmd_server)

    p = sub.add_parser("bench", help="Run ds4-bench for a matrix row.")
    p.add_argument("name")
    add_common_runtime(p)
    p.add_argument("--prompt-file", default="speed-bench/promessi_sposi.txt")
    p.add_argument("--ctx-start", type=int, default=8192)
    p.add_argument("--ctx-max", type=int, default=8192)
    p.add_argument("--ctx-alloc", type=int, default=100000)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--out-dir", default="tuning/gx10_matrix_results")
    p.add_argument("--extra", action="append", default=[], metavar="ARG")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("bench-suite", help="Run ds4-bench for a row group or explicit rows.")
    add_common_runtime(p)
    p.add_argument("rows", nargs="*", help="Rows or groups: core, exact, tradeoff, mtp, all.")
    p.add_argument("--group", default="core", choices=sorted(ROW_GROUPS))
    p.add_argument("--prompt-file", default="speed-bench/promessi_sposi.txt")
    p.add_argument("--ctx-start", type=int, default=8192)
    p.add_argument("--ctx-max", type=int, default=8192)
    p.add_argument("--ctx-alloc", type=int, default=100000)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--out-dir", default="tuning/gx10_matrix_results")
    p.add_argument("--summary", default="tuning/gx10_matrix_results/summary.csv")
    p.add_argument("--markdown", default="tuning/gx10_matrix_results/summary.md")
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument("--extra", action="append", default=[], metavar="ARG")
    p.set_defaults(func=cmd_bench_suite)

    p = sub.add_parser("summary", help="Summarize matrix bench CSV files.")
    p.add_argument("--out-dir", default="tuning/gx10_matrix_results")
    p.add_argument("--output", default="tuning/gx10_matrix_results/summary.csv")
    p.add_argument("--markdown", default="tuning/gx10_matrix_results/summary.md")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("eval", help="Start ds4-server for a row and run coding eval.")
    p.add_argument("name")
    add_common_runtime(p)
    p.add_argument("--ctx", type=int, default=100000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8106)
    p.add_argument("--tokens", type=int, default=900)
    p.add_argument("--max-tokens", type=int, default=900)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--label")
    p.add_argument("--out-dir", default="tuning/gx10_matrix_results")
    p.add_argument("--canary", action="store_true")
    p.add_argument("--only", action="append", default=[])
    p.add_argument("--startup-timeout", type=float, default=180.0)
    p.add_argument("--server-extra", action="append", default=[], metavar="ARG")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("eval-suite", help="Run coding eval over a row group or explicit rows.")
    add_common_runtime(p)
    p.add_argument("rows", nargs="*", help="Rows or groups: core, exact, tradeoff, mtp, all.")
    p.add_argument("--group", default="core", choices=sorted(ROW_GROUPS))
    p.add_argument("--ctx", type=int, default=100000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8106)
    p.add_argument("--tokens", type=int, default=900)
    p.add_argument("--max-tokens", type=int, default=900)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--label-prefix", default="")
    p.add_argument("--out-dir", default="tuning/gx10_matrix_results")
    p.add_argument("--canary", action="store_true")
    p.add_argument("--only", action="append", default=[])
    p.add_argument("--startup-timeout", type=float, default=180.0)
    p.add_argument("--server-extra", action="append", default=[], metavar="ARG")
    p.add_argument("--stop-on-fail", action="store_true")
    p.set_defaults(func=cmd_eval_suite)

    p = sub.add_parser("ds-eval", help="Run ds4-eval for a matrix row.")
    p.add_argument("name")
    add_common_runtime(p)
    p.add_argument("--questions", type=int, default=0, help="0 means all embedded questions.")
    p.add_argument("--tokens", type=int, default=16000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--label")
    p.add_argument("--trace", type=Path)
    p.add_argument("--out-dir", default="tuning/gx10_matrix_results")
    p.add_argument("--quality", action="store_true")
    p.add_argument("--warm-weights", action="store_true")
    p.add_argument("--think", action="store_true")
    p.add_argument("--nothink", action="store_true")
    p.add_argument("--timeout-sec", type=float, default=0.0)
    p.add_argument("--extra", action="append", default=[], metavar="ARG")
    p.set_defaults(func=cmd_ds_eval)

    p = sub.add_parser("ds-eval-suite", help="Run ds4-eval over a row group or explicit rows.")
    add_common_runtime(p)
    p.add_argument("rows", nargs="*", help="Rows or groups: core, exact, tradeoff, mtp, all.")
    p.add_argument("--group", default="core", choices=sorted(ROW_GROUPS))
    p.add_argument("--questions", type=int, default=0, help="0 means all embedded questions.")
    p.add_argument("--tokens", type=int, default=16000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--label-prefix", default="")
    p.add_argument("--out-dir", default="tuning/gx10_matrix_results")
    p.add_argument("--quality", action="store_true")
    p.add_argument("--warm-weights", action="store_true")
    p.add_argument("--think", action="store_true")
    p.add_argument("--nothink", action="store_true")
    p.add_argument("--timeout-sec", type=float, default=0.0)
    p.add_argument("--extra", action="append", default=[], metavar="ARG")
    p.add_argument("--stop-on-fail", action="store_true")
    p.set_defaults(func=cmd_ds_eval_suite)

    p = sub.add_parser("run", help="Run an arbitrary command under a row env.")
    p.add_argument("name")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
