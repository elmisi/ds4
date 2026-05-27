#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO = Path("/home/alessandro/projects/ds4-stepwise")
RUNNER = REPO / "tuning/agent_stepwise_runs/run_agent_variant.py"


CURRENT_VARIANTS = [
    ("0ee3c82_indexer_no_graph", {
        "DS4_CUDA_INDEXER_SCORE_TOPK_FUSED": "1",
    }),
    ("0ee3c82_soa_no_graph", {
        "DS4_CUDA_Q8_SOA_CACHE": "1",
    }),
    ("0ee3c82_soa_indexer_no_graph", {
        "DS4_CUDA_Q8_SOA_CACHE": "1",
        "DS4_CUDA_INDEXER_SCORE_TOPK_FUSED": "1",
    }),
    ("0ee3c82_graph_indexer", {
        "DS4_CUDA_GRAPH_DECODE": "1",
        "DS4_CUDA_INDEXER_SCORE_TOPK_FUSED": "1",
    }),
    ("0ee3c82_graph_soa_indexer", {
        "DS4_CUDA_GRAPH_DECODE": "1",
        "DS4_CUDA_Q8_SOA_CACHE": "1",
        "DS4_CUDA_INDEXER_SCORE_TOPK_FUSED": "1",
    }),
    ("0ee3c82_graph_indexer_no_hcpre", {
        "DS4_CUDA_GRAPH_DECODE": "1",
        "DS4_CUDA_INDEXER_SCORE_TOPK_FUSED": "1",
        "DS4_CUDA_NO_FUSED_HC_PRE": "1",
    }),
    ("0ee3c82_graph_indexer_no_qnorm", {
        "DS4_CUDA_GRAPH_DECODE": "1",
        "DS4_CUDA_INDEXER_SCORE_TOPK_FUSED": "1",
        "DS4_CUDA_NO_Q_NORM_ROPE_FUSED": "1",
    }),
    ("0ee3c82_graph_indexer_no_kvrope", {
        "DS4_CUDA_GRAPH_DECODE": "1",
        "DS4_CUDA_INDEXER_SCORE_TOPK_FUSED": "1",
        "DS4_CUDA_NO_KV_ROPE_STORE_FUSED": "1",
    }),
    ("0ee3c82_graph_indexer_no_attnrope", {
        "DS4_CUDA_GRAPH_DECODE": "1",
        "DS4_CUDA_INDEXER_SCORE_TOPK_FUSED": "1",
        "DS4_CUDA_NO_ATTN_OUTPUT_ROPE_LOW_FUSED": "1",
    }),
    ("0ee3c82_graph_indexer_down_tile8_rowspan", {
        "DS4_CUDA_GRAPH_DECODE": "1",
        "DS4_CUDA_INDEXER_SCORE_TOPK_FUSED": "1",
        "DS4_CUDA_MOE_DOWN_TILE8_ROWSPAN": "1",
    }),
    ("0ee3c82_graph_no_q8_cache_x", {
        "DS4_CUDA_GRAPH_DECODE": "1",
        "DS4_CUDA_NO_Q8_CACHE_X": "1",
    }),
]


HISTORICAL_VARIANTS = [
    ("origin/main", "origin_main_no_env", {}),
    ("01662bc", "01662bc_tune_f16_decode_no_env", {}),
    ("5ee0970", "5ee0970_q8_cache_x_no_env", {}),
    ("81a1a3b", "81a1a3b_fused_hc_rope_no_env", {}),
    ("81a1a3b", "81a1a3b_fused_hc_rope_no_hcpre", {
        "DS4_CUDA_NO_FUSED_HC_PRE": "1",
    }),
    ("81a1a3b", "81a1a3b_fused_hc_rope_no_qnorm", {
        "DS4_CUDA_NO_Q_NORM_ROPE_FUSED": "1",
    }),
    ("81a1a3b", "81a1a3b_fused_hc_rope_no_kvrope", {
        "DS4_CUDA_NO_KV_ROPE_STORE_FUSED": "1",
    }),
    ("81a1a3b", "81a1a3b_fused_hc_rope_no_attnrope", {
        "DS4_CUDA_NO_ATTN_OUTPUT_ROPE_LOW_FUSED": "1",
    }),
    ("bcc390e", "bcc390e_skip_ordered_no_env", {}),
    ("dbb6cd7", "dbb6cd7_moe_tile8_no_env", {}),
    ("dbb6cd7", "dbb6cd7_moe_tile8_enabled", {
        "DS4_CUDA_MOE_DOWN_TILE8_ROWSPAN": "1",
    }),
    ("cfd44c3", "cfd44c3_graph_update_no_env", {}),
    ("cfd44c3", "cfd44c3_graph_update_graph", {
        "DS4_CUDA_GRAPH_DECODE": "1",
    }),
    ("origin/gx10-graph-soa-indexer-topk", "old_pr_no_env", {}),
    ("origin/gx10-graph-soa-indexer-topk", "old_pr_graph", {
        "DS4_CUDA_GRAPH_DECODE": "1",
    }),
    ("origin/gx10-graph-soa-indexer-topk", "old_pr_graph_soa_indexer", {
        "DS4_CUDA_GRAPH_DECODE": "1",
        "DS4_CUDA_Q8_SOA_CACHE": "1",
        "DS4_CUDA_INDEXER_SCORE_TOPK_FUSED": "1",
    }),
]


def run(cmd, check=False):
    print("+ " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=REPO)
    if check and rc != 0:
        raise SystemExit(rc)
    return rc


def run_variant(name, env):
    cmd = [sys.executable, str(RUNNER), "--name", name]
    for key, value in env.items():
        cmd += ["--env", f"{key}={value}"]
    return run(cmd)


def build_current():
    run(["rm", "-f", "ds4.o", "ds4_cuda.o", "ds4-agent"], check=True)
    return run(["make", "-j", str(os.cpu_count() or 1), "ds4-agent", "CUDA_ARCH="])


def run_current():
    for name, env in CURRENT_VARIANTS:
        run_variant(name, env)


def run_historical():
    start_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True
    ).strip()
    start_head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
    ).strip()
    last_ref = None
    try:
        for ref, name, env in HISTORICAL_VARIANTS:
            if ref != last_ref:
                run(["git", "checkout", "--detach", ref], check=True)
                if build_current() != 0:
                    print(f"build failed for {ref}, skipping its variants", flush=True)
                    last_ref = None
                    continue
                last_ref = ref
            run_variant(name, env)
    finally:
        target = start_branch or start_head
        run(["git", "checkout", target], check=True)
        if build_current() != 0:
            raise SystemExit("failed to rebuild starting checkout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=["current", "historical", "all"])
    args = ap.parse_args()

    if args.set in ("current", "all"):
        run_current()
    if args.set in ("historical", "all"):
        run_historical()


if __name__ == "__main__":
    main()
