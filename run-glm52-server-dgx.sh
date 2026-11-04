#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${DS4_MODEL:-gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf}"
CTX="${DS4_CTX:-256000}"
EXPERT_CACHE="${DS4_EXPERT_CACHE:-12GB}"
TOKENS="${DS4_TOKENS:-2048}"
POWER="${DS4_POWER:-100}"
MEMORY_HIGH="${DS4_MEMORY_HIGH:-72G}"
MEMORY_MAX="${DS4_MEMORY_MAX:-80G}"
COORDINATOR="${DS4_COORDINATOR:-192.168.0.20}"
DIST_PORT="${DS4_DIST_PORT:-9000}"
API_HOST="${DS4_API_HOST:-127.0.0.1}"
API_PORT="${DS4_API_PORT:-8000}"
LOG_DIR="${DS4_LOG_DIR:-logs}"

if [[ "$(hostname -s)" == "amarcord" ]]; then
  ROLE="${DS4_ROLE:-worker}"
  # Layer 38 is an indexed-attention anchor.  Keep it with the worker so
  # layer 39 can consume its local indexer selection.
  LAYERS="${DS4_LAYERS:-38:output}"
else
  ROLE="${DS4_ROLE:-coordinator}"
  LAYERS="${DS4_LAYERS:-0:37}"
fi

BIN="${DS4_BIN:-./ds4-server}"
if [[ ! -x "$BIN" ]]; then
  echo "Executable not found or not runnable: $BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  exit 1
fi

if pgrep -ax ds4-server >/dev/null 2>&1; then
  echo "A ds4-server process is already running on this host." >&2
  pgrep -ax ds4-server >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
LOG="${DS4_LOG:-$LOG_DIR/glm52-server-${ROLE}-$(date +%Y%m%d-%H%M%S).log}"

cmd=(
  "$BIN"
  --cuda
  -m "$MODEL"
  --ssd-streaming
  --ssd-streaming-cache-experts "$EXPERT_CACHE"
  -c "$CTX"
  -n "$TOKENS"
  --power "$POWER"
  --role "$ROLE"
  --layers "$LAYERS"
)
if [[ "$ROLE" == "worker" ]]; then
  cmd+=(--coordinator "$COORDINATOR" "$DIST_PORT")
else
  cmd+=(--listen 0.0.0.0 "$DIST_PORT" --host "$API_HOST" --port "$API_PORT")
fi
cmd+=("$@")

env_args=(
  "DS4_CUDA_NO_MODEL_PREFETCH=${DS4_CUDA_NO_MODEL_PREFETCH:-1}"
  "DS4_CUDA_NO_Q8_F16_CACHE=${DS4_CUDA_NO_Q8_F16_CACHE:-1}"
  "DS4_CUDA_NO_Q8_F32_CACHE=${DS4_CUDA_NO_Q8_F32_CACHE:-1}"
  "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-32}"
  "DS4_CUDA_WEIGHT_CACHE_EVICT=${DS4_CUDA_WEIGHT_CACHE_EVICT:-1}"
  "DS4_CUDA_WEIGHT_CACHE_EVICT_RESERVE_GB=${DS4_CUDA_WEIGHT_CACHE_EVICT_RESERVE_GB:-12}"
  "DS4_CUDA_STRICT_WEIGHT_CACHE=${DS4_CUDA_STRICT_WEIGHT_CACHE:-1}"
  "DS4_GLM_MEMORY_GUARD_FRACTION=${DS4_MEMORY_GUARD_FRACTION:-0.75}"
  "DS4_GLM_MEMORY_GUARD_RESERVE_GB=${DS4_MEMORY_GUARD_RESERVE_GB:-24}"
  "DS4_GLM_MEMORY_GUARD_REPORT=${DS4_GLM_MEMORY_GUARD_REPORT:-1}"
)
if [[ "${DS4_DEBUG:-0}" == "1" ]]; then
  env_args+=(
    CUDA_LAUNCH_BLOCKING=1
    DS4_GLM_SYNC_TRACE=1
    DS4_GLM_FULL_PREFILL_TRACE=1
    DS4_GLM_INDEXED_PREFILL_TRACE=1
    DS4_GLM_INDEXED_PREFILL_TRACE_ALL=1
    DS4_CUDA_STREAM_SELECTED_DEBUG=1
    DS4_CUDA_WEIGHT_CACHE_VERBOSE=1
  )
fi

scope=()
if [[ "${DS4_NO_CGROUP:-0}" != "1" ]] && command -v systemd-run >/dev/null 2>&1; then
  scope=(
    systemd-run --user --scope --quiet
    -p "MemoryHigh=$MEMORY_HIGH"
    -p "MemoryMax=$MEMORY_MAX"
    --
  )
fi

echo "Launching: ${cmd[*]}"
echo "Log: $LOG"

# A transient service survives the launching shell.  Use this for background
# server processes; an interactive scope remains convenient for foreground use.
if [[ "${DS4_SERVICE:-0}" == "1" ]]; then
  UNIT="${DS4_UNIT:-ds4-glm52-${ROLE}}"
  if systemctl --user is-active --quiet "$UNIT"; then
    echo "Systemd unit is already active: $UNIT" >&2
    exit 1
  fi
  service_props=(
    -p "WorkingDirectory=$PWD"
    -p "StandardOutput=append:$PWD/$LOG"
    -p "StandardError=append:$PWD/$LOG"
  )
  if ((${#scope[@]})); then
    service_props+=(
      -p "MemoryHigh=$MEMORY_HIGH"
      -p "MemoryMax=$MEMORY_MAX"
    )
  fi
  exec systemd-run --user --unit "$UNIT" --collect --quiet \
    "${service_props[@]}" -- env "${env_args[@]}" "${cmd[@]}"
fi

exec "${scope[@]}" env "${env_args[@]}" "${cmd[@]}" >"$LOG" 2>&1
