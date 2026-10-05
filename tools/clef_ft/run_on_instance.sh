#!/usr/bin/env bash
# Clef fine-tuning on a rented vast.ai GPU instance (1..N x RTX PRO 6000 96GB / H100 SXM 80GB, pytorch image).
#
# GPUs: CLEF_FT_NPROC (default: the visible GPU count, CUDA_VISIBLE_DEVICES or nvidia-smi). Above 1 every train.py
# runs under torchrun, one rank per GPU with the whole model on each (data parallel, see ftdist.py); 1 is the plain
# single-process path. setup checks an NCCL all_reduce across them (falls back to NCCL_P2P_DISABLE=1, kept in
# /workspace/.nccl_env); smoke runs the N-rank path first. Rows/s in throughput.json and the ETAs are global.
#
# Layout (fixed; CLEF_FT_WORKSPACE overrides /workspace for tests):
#   /workspace/clef_ft/            this directory (scp -r tools/clef_ft)
#   /workspace/clef_ft_data.tar    from tools/clef_ft/build_data.py (extracted to /workspace/data/clef_ft_data)
#   /workspace/models/<model>/     HF snapshots (clef-flash, clef)
#   /workspace/cache/              hidden-state cache (large; not pulled)
#   /workspace/out/                everything to pull: <model>_<arm>/, smoke_*/, logs/, steps.jsonl,
#                                  DONE | FAILED (written by pipeline); the puller touches PULLED when done
#
# Stages:
#   setup                    pip install pinned stack (+ torch 2.9.1+cu128 if the image has another), unpack data,
#                            NCCL check when N > 1
#   fetch <model>            download clef-flash | clef
#   smoke                    clef-flash on N ranks: lora --max-steps 10N (4N rows/split) + zero on 4N rows
#   run <model> <arm>        background train.py run, log in out/logs/<model>_<arm>.log
#   status                   runs, log tails, GPU, markers
#   pipeline <deadline_h>    background: setup -> fetch clef-flash -> (fetch clef in parallel) smoke -> clef zero
#                            -> clef head -> clef lora -> clef-flash head/lora if the time left allows -> DONE.
#                            Any failure writes out/FAILED and stops. Starts the self-destroy watchdog.
#                            CLEF_FT_ARMS="clef:zero clef:lora" runs only those <model>:<arm> after the smoke test.
#                            CLEF_FT_TRAIN_ARGS is appended to every train.py call (e.g. "--false-yes-penalty 1.0").
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
CODE="$(dirname "$SELF")"
W="${CLEF_FT_WORKSPACE:-/workspace}"
TAR="$W/clef_ft_data.tar"
DATA="$W/data/clef_ft_data"
MODELS="$W/models"
CACHE="$W/cache"
OUT="$W/out"
LOGS="$OUT/logs"
PY="${PYTHON:-python}"
VASTAI="${CLEF_FT_VASTAI:-vastai}"
export PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 CLEF_FT_WORKSPACE="$W"

TORCH_VERSION="2.9.1+cu128"
PIP_PINS=(transformers==5.18.0 huggingface_hub==1.33.0 tokenizers==0.23.2 safetensors==0.8.0 accelerate==1.15.0
          peft==0.21.2 hf_xet hf_transfer pillow vastai)
declare -A HF_REPOS=([clef-flash]=Cloudflare/clef-flash [clef]=Cloudflare/clef)

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }
now() { date +%s.%N; }

gpu_count() {  # GPUs this script may use: CUDA_VISIBLE_DEVICES when set, else nvidia-smi; 0 without any
  if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES//[[:space:]]/}" | grep -c . || true
  elif command -v nvidia-smi >/dev/null; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true
  else
    echo 0
  fi
}
NPROC="${CLEF_FT_NPROC:-$(gpu_count)}"
(( NPROC >= 1 )) || NPROC=1
# 80 GB GPUs (H100) hold the 27B LoRA peak (64.6 GB) with ~15 GB to spare: expandable segments keep allocator
# fragmentation out of it, and NVLink SHARP (NVLS) buffers are not worth their memory for one ~1 GB gradient
# sum per optimizer step. Both only when unset.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
(( NPROC > 1 )) && export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
[[ -f "$W/.nccl_env" ]] && source "$W/.nccl_env"

stage_setup() {
  local have
  have="$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)"
  log "image torch: $have"
  if [[ "$have" != "$TORCH_VERSION" ]]; then
    "$PY" -m pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
  fi
  # The pytorch *runtime* image has no C compiler; triton (fla kernels) needs one to build its CUDA launcher.
  if ! command -v gcc >/dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc libc6-dev
  fi
  "$PY" -m pip install --no-cache-dir "${PIP_PINS[@]}"
  "$PY" -m pip install --no-cache-dir flash-linear-attention==0.5.2 \
    || log "WARNING: flash-linear-attention did not install; gated delta layers fall back to the slow torch path"
  if [[ ! -f "$DATA/dataset.jsonl" ]]; then
    [[ -f "$TAR" ]] || die "missing $TAR (scp clef_ft_data.tar to $W/)"
    mkdir -p "$W/data"
    tar -xf "$TAR" -C "$W/data"
  fi
  local rows images
  rows="$(wc -l < "$DATA/dataset.jsonl")"
  images="$(find "$DATA/images" -name '*.jpg' | wc -l)"
  log "data: $rows rows, $images images"
  # v1 has 846 rows; later tars (v2+) add domains / near frames -- every row needs its image
  [[ "$rows" -gt 0 && "$rows" -eq "$images" ]] || die "expected one image per row (rows $rows, images $images)"
  "$PY" - <<'PYEOF'
import importlib.util, torch, transformers, peft
from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "peft", peft.__version__)
print("fla", importlib.util.find_spec("fla") is not None, "causal_conv1d", importlib.util.find_spec("causal_conv1d") is not None)
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    x = torch.ones(256, 256, device=f"cuda:{i}", dtype=torch.bfloat16)  # a bf16 kernel runs on this arch
    assert (x @ x)[0, 0].item() == 256, f"gpu {i}: bf16 matmul gave a wrong result"
    print("gpu", i, props.name, f"sm_{props.major}{props.minor}", round(props.total_memory / 1e9, 1), "GB", "bf16 ok")
print("torch arch list", torch.cuda.get_arch_list())
PYEOF
  log "cpu: $(nproc) threads, RAM $(awk '/MemTotal/ { printf "%.0f GB", $2 / 1e6 }' /proc/meminfo); ranks: $NPROC"
  nccl_check
  touch "$W/.setup_done"
}

nccl_check() {  # an all_reduce across the NPROC GPUs; retried without peer-to-peer (kept in .nccl_env for train)
  (( NPROC > 1 )) || { log "one GPU: no NCCL check"; return 0; }
  local run=("$PY" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" "$CODE/nccl_check.py")
  if timeout 300 "${run[@]}"; then rm -f "$W/.nccl_env"; return 0; fi
  log "WARNING: NCCL check failed on $NPROC GPUs; retrying with NCCL_P2P_DISABLE=1"
  if NCCL_P2P_DISABLE=1 timeout 300 "${run[@]}"; then
    echo "export NCCL_P2P_DISABLE=1" > "$W/.nccl_env"
    log "NCCL works without P2P; train runs export NCCL_P2P_DISABLE=1 ($W/.nccl_env)"
    return 0
  fi
  die "NCCL all_reduce failed across $NPROC GPUs (CLEF_FT_NPROC=1 trains on one GPU)"
}

stage_fetch() {
  local model="${1:?fetch <clef-flash|clef>}" repo="${HF_REPOS[${1:-}]:-}" attempt
  [[ -n "$repo" ]] || die "unknown model $model"
  mkdir -p "$MODELS"
  for attempt in 1 2 3; do
    if "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('$repo', local_dir='$MODELS/$model')"; then
      [[ -f "$MODELS/$model/joint_head.safetensors" ]] || die "$model: no joint_head.safetensors after download"
      log "$model: $(du -sh "$MODELS/$model" | cut -f1) in $MODELS/$model"
      return 0
    fi
    log "download attempt $attempt failed"; sleep 10
  done
  die "could not download $repo"
}

train() {  # train <model> <arm> <out> [extra args]: one process, or NPROC torchrun ranks
  local model="$1" arm="$2" out="$3" launch=("$PY"); shift 3
  if (( NPROC > 1 )); then
    launch=("$PY" -m torch.distributed.run --standalone --nproc_per_node "$NPROC")
    # torchrun's default is 1 thread per rank; image preprocessing runs on the CPU
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(nproc) / NPROC > 0 ? $(nproc) / NPROC : 1 ))}"
  fi
  "${launch[@]}" "$CODE/train.py" --model "$model" --arm "$arm" --out "$out" --data "$DATA" --models "$MODELS" \
    --cache "$CACHE" "$@" ${CLEF_FT_TRAIN_ARGS:-}
}

estimate() {  # estimate <arm>: seconds for a full run at the smoke's global rates, this data, CLEF_FT_TRAIN_ARGS epochs
  local arm="$1" extra=() epochs
  if [[ -f "$DATA/dataset.jsonl" ]]; then
    extra=("$DATA/dataset.jsonl")
    epochs="$(sed -n 's/.*--epochs[ =]\([0-9][0-9]*\).*/\1/p' <<< "${CLEF_FT_TRAIN_ARGS:-}")"
    if [[ -n "$epochs" ]]; then extra+=("$epochs"); fi
  fi
  "$PY" "$CODE/ftutil.py" estimate "$OUT/smoke_flash_lora/throughput.json" "$arm" "${extra[@]}"
}

stage_smoke() {
  # Each rank gets about what the one-GPU smoke had (4 rows per split, 10 training rows).
  train clef-flash lora "$OUT/smoke_flash_lora" --max-steps $(( 10 * NPROC )) --limit $(( 4 * NPROC ))
  [[ -f "$OUT/smoke_flash_lora/throughput.json" ]] || die "smoke lora wrote no throughput.json"
  grep -q "\"world_size\": $NPROC\b" "$OUT/smoke_flash_lora/throughput.json" \
    || die "smoke lora did not run on $NPROC ranks: $(cat "$OUT/smoke_flash_lora/throughput.json")"
  train clef-flash zero "$OUT/smoke_flash_zero" --limit $(( 4 * NPROC ))
  cat "$OUT/smoke_flash_lora/throughput.json"
  log "estimated full clef-flash runs on $NPROC GPU(s) (s): head $(estimate head), lora $(estimate lora)"
}

stage_run() {
  local model="${1:?run <model> <arm>}" arm="${2:?run <model> <arm>}"
  mkdir -p "$LOGS"
  nohup bash "$SELF" _train "$model" "$arm" > "$LOGS/${model}_${arm}.log" 2>&1 &
  log "started $model $arm on $NPROC GPU(s) pid $! -> $LOGS/${model}_${arm}.log"
}

stage_status() {
  local marker d
  log "ranks per run: $NPROC"
  for marker in DONE FAILED PULLED DESTROYING; do [[ -e "$OUT/$marker" ]] && log "marker: $marker"; done
  [[ -f "$OUT/pipeline.pid" ]] && kill -0 "$(cat "$OUT/pipeline.pid")" 2>/dev/null && log "pipeline running (pid $(cat "$OUT/pipeline.pid"))"
  [[ -f "$OUT/watchdog.pid" ]] && kill -0 "$(cat "$OUT/watchdog.pid")" 2>/dev/null && log "watchdog running (pid $(cat "$OUT/watchdog.pid"))"
  [[ -f "$OUT/steps.jsonl" ]] && cat "$OUT/steps.jsonl"
  for d in "$OUT"/*/; do
    [[ -d "$d" && "$(basename "$d")" != logs ]] || continue
    if [[ -f "$d/result.json" ]]; then log "$(basename "$d"): finished"; else log "$(basename "$d"): running or incomplete"; fi
  done
  for d in "$LOGS"/*.log; do [[ -f "$d" ]] && { echo "== $d"; tail -n 3 "$d"; }; done
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv || true
  df -h "$W" | tail -1
}

# ---------------------------------------------------------------------------------------------- pipeline

pipeline_step() {  # pipeline_step <name> <stage args...>: run a stage in a child, record it, stop on failure
  local name="$1"; shift
  local logf="$LOGS/$name.log" t0 t1
  t0="$(now)"
  log "step $name: start"
  if bash "$SELF" "$@" > "$logf" 2>&1; then
    t1="$(now)"; "$PY" "$CODE/ftutil.py" step "$OUT" "$name" ok "$t0" "$t1" "$logf"
    log "step $name: ok"
  else
    t1="$(now)"; "$PY" "$CODE/ftutil.py" step "$OUT" "$name" failed "$t0" "$t1" "$logf" || true
    "$PY" "$CODE/ftutil.py" failed "$OUT" "$name" "$logf" || echo "{\"step\": \"$name\"}" > "$OUT/FAILED"
    log "step $name: FAILED (see $logf)"
    exit 1
  fi
}

pipeline_train() {  # pipeline_train <model> <arm>, skipped when a previous run finished it
  local model="$1" arm="$2"
  if [[ -f "$OUT/${model}_${arm}/result.json" ]]; then
    "$PY" "$CODE/ftutil.py" step "$OUT" "${model}_${arm}" ok "$(now)" "$(now)" "already finished"
    return 0
  fi
  pipeline_step "${model}_${arm}" _train "$model" "$arm"
}

pipeline_optional() {  # pipeline_optional <model> <arm> <deadline_s> <start>
  local model="$1" arm="$2" deadline_s="$3" start="$4" est left
  est="$(estimate "$arm")"
  left="$("$PY" -c "import time; print(int($deadline_s - (time.time() - $start)))")"
  # Run only with 1.5x the estimate plus 30 min for the pull to spare.
  if (( left > est * 3 / 2 + 1800 )); then
    pipeline_train "$model" "$arm"
  else
    log "skip $model $arm: ${left}s left, estimate ${est}s"
    "$PY" "$CODE/ftutil.py" step "$OUT" "${model}_${arm}" skipped "$(now)" "$(now)" "left ${left}s, estimate ${est}s"
  fi
}

run_pipeline() {
  local deadline_s="$1" start fetch_pid rc
  start="$(cat "$OUT/pipeline_start")"
  # Any exit without DONE (an unexpected error in this script itself) still leaves FAILED for the puller.
  trap '[[ -e "$OUT/DONE" || -e "$OUT/FAILED" ]] || "$PY" "$CODE/ftutil.py" failed "$OUT" pipeline "$LOGS/pipeline.log" || touch "$OUT/FAILED"' EXIT
  if [[ -f "$W/.setup_done" ]]; then log "setup already done"; else pipeline_step setup setup; fi
  pipeline_step fetch_clef-flash fetch clef-flash
  # The 27B download runs while the smoke test uses the GPU.
  ( bash "$SELF" fetch clef > "$LOGS/fetch_clef.log" 2>&1; echo $? > "$OUT/.fetch_clef_rc" ) &
  fetch_pid=$!
  pipeline_step smoke smoke
  wait "$fetch_pid" || true
  rc="$(cat "$OUT/.fetch_clef_rc" 2>/dev/null || echo 1)"
  if [[ "$rc" != 0 ]]; then
    "$PY" "$CODE/ftutil.py" step "$OUT" fetch_clef failed "$start" "$(now)" "$LOGS/fetch_clef.log"
    "$PY" "$CODE/ftutil.py" failed "$OUT" fetch_clef "$LOGS/fetch_clef.log"
    exit 1
  fi
  "$PY" "$CODE/ftutil.py" step "$OUT" fetch_clef ok "$start" "$(now)" "$LOGS/fetch_clef.log"
  if [[ -n "${CLEF_FT_ARMS:-}" ]]; then  # e.g. "clef:zero clef:lora" -- only these, no optional runs
    local spec
    for spec in $CLEF_FT_ARMS; do pipeline_train "${spec%%:*}" "${spec#*:}"; done
  else
    pipeline_train clef zero
    pipeline_train clef head
    pipeline_train clef lora
    pipeline_optional clef-flash head "$deadline_s" "$start"
    pipeline_optional clef-flash lora "$deadline_s" "$start"
  fi
  "$PY" "$CODE/ftutil.py" done "$OUT"
  log "pipeline DONE"
}

stage_pipeline() {
  local hours="${1:?pipeline <deadline_hours>}" deadline_s
  deadline_s="$(awk -v h="$hours" 'BEGIN { printf "%d", h * 3600 }')"
  (( deadline_s > 0 )) || die "deadline must be positive hours"
  mkdir -p "$LOGS"
  if [[ -f "$OUT/pipeline.pid" ]] && kill -0 "$(cat "$OUT/pipeline.pid")" 2>/dev/null; then
    die "pipeline already running (pid $(cat "$OUT/pipeline.pid"))"
  fi
  for marker in DONE FAILED; do
    [[ -e "$OUT/$marker" ]] && die "$OUT/$marker exists from an earlier run; move it away first"
  done
  date +%s > "$OUT/pipeline_start"
  nohup bash "$SELF" _watchdog "$deadline_s" >> "$LOGS/watchdog.log" 2>&1 &
  echo $! > "$OUT/watchdog.pid"
  nohup bash "$SELF" _pipeline "$deadline_s" >> "$LOGS/pipeline.log" 2>&1 &
  echo $! > "$OUT/pipeline.pid"
  log "pipeline pid $(cat "$OUT/pipeline.pid"), watchdog pid $(cat "$OUT/watchdog.pid"), deadline ${hours}h"
  log "logs: $LOGS/pipeline.log, $LOGS/watchdog.log; progress: bash $SELF status"
}

# ---------------------------------------------------------------------------------------------- watchdog

container_var() {  # the value of a vast-provided variable: own env, else PID 1's env
  local name="$1"
  if [[ -n "${!name:-}" ]]; then echo "${!name}"; return; fi
  tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$name=//p" | head -n 1 || true
}

destroy_self() {
  local reason="$1" id key attempt
  log "destroying this instance: $reason"
  echo "{\"reason\": \"$reason\", \"time\": $(date +%s)}" > "$OUT/DESTROYING"
  id="$(container_var CONTAINER_ID)"
  key="$(container_var CONTAINER_API_KEY)"
  if [[ -z "$id" || -z "$key" ]]; then
    log "WARNING: CONTAINER_ID or CONTAINER_API_KEY not set; cannot self-destroy. Destroy the instance by hand."
    return 0
  fi
  for attempt in 1 2 3 4 5; do
    if "$VASTAI" destroy instance "$id" -y --api-key "$key"; then log "destroy requested"; return 0; fi
    log "destroy attempt $attempt failed"; sleep "${CLEF_FT_DESTROY_RETRY_S:-30}"
  done
  log "WARNING: self-destroy failed; destroy the instance by hand"
}

run_watchdog() {
  # Destroy when: the deadline passed and nothing was pulled; or DONE/FAILED is older than the pull grace and
  # nothing was pulled; or (whatever PULLED says) the deadline plus the hard extra passed. PULLED after
  # DONE/FAILED ends the watchdog (the puller destroys the instance).
  local deadline_s="$1" poll="${CLEF_FT_WATCHDOG_POLL_S:-60}" grace="${CLEF_FT_PULL_GRACE_S:-5400}"
  local hard="${CLEF_FT_HARD_EXTRA_S:-7200}" start finished now_s elapsed age marker reason
  start="$(cat "$OUT/pipeline_start")"
  log "watchdog: deadline ${deadline_s}s, pull grace ${grace}s, hard cap +${hard}s, poll ${poll}s"
  while true; do
    now_s="$(date +%s)"; elapsed=$(( now_s - start )); reason=""
    finished=""
    for marker in DONE FAILED; do [[ -e "$OUT/$marker" ]] && finished="$OUT/$marker"; done
    if [[ -e "$OUT/PULLED" && -n "$finished" ]]; then log "watchdog: pulled after $(basename "$finished"); exiting"; return 0; fi
    if (( elapsed >= deadline_s )) && [[ ! -e "$OUT/PULLED" ]]; then reason="deadline ${deadline_s}s passed, not pulled"; fi
    if [[ -n "$finished" && ! -e "$OUT/PULLED" ]]; then
      age=$(( now_s - $(stat -c %Y "$finished") ))
      (( age >= grace )) && reason="$(basename "$finished") ${age}s ago, not pulled"
    fi
    (( elapsed >= deadline_s + hard )) && reason="hard cap: deadline + ${hard}s passed"
    if [[ -n "$reason" ]]; then destroy_self "$reason"; return 0; fi
    sleep "$poll"
  done
}

# ---------------------------------------------------------------------------------------------- dispatch

stage="${1:-}"; shift || true
case "$stage" in
  setup) stage_setup ;;
  fetch) stage_fetch "$@" ;;
  smoke) stage_smoke ;;
  run) stage_run "$@" ;;
  status) stage_status ;;
  pipeline) stage_pipeline "$@" ;;
  _train) train "$1" "$2" "$OUT/${1}_${2}" ;;
  _pipeline) run_pipeline "$@" ;;
  _watchdog) run_watchdog "$@" ;;
  *) sed -n '2,28p' "$SELF"; exit 2 ;;
esac
