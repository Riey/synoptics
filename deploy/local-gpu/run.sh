#!/usr/bin/env bash
# Run one local service in the foreground, or check the services that are already running.
#
#   bash deploy/local-gpu/run.sh tracker   tracker service      127.0.0.1:8090  .venv-tracker  (GPU)
#   bash deploy/local-gpu/run.sh clef      Clef decision server 127.0.0.1:8085  .venv-clef     (GPU)
#   bash deploy/local-gpu/run.sh backend   reference backend    127.0.0.1:8040  .venv
#   bash deploy/local-gpu/run.sh live      Live server          127.0.0.1:8104  live-api/.venv
#   bash deploy/local-gpu/run.sh check     probe the four health routes; exit 0 only when all are up and ready
#
# One service per terminal: every command is foreground, binds 127.0.0.1 only, and stops with Ctrl-C.
# This script never starts, stops, restarts or supervises another service, and never runs model inference
# itself. Configuration comes from <repo>/.env.local-gpu (written by deploy/local-gpu/setup.sh); the model
# paths below it are derived from MODEL_ROOT unless the environment file sets them explicitly.
#
# `check` fails (non-zero) when a service does not answer, when the tracker or Clef model service is not
# ready, or when the backend / Live self-report ready=false — each with the not-ready reason. It probes
# 127.0.0.1 directly, ignoring ambient http_proxy/https_proxy variables. A CPU-only stack (backend + Live,
# no GPU services) therefore reports failure, as it should.
#
# Model prerequisites are checked before launch and reported as errors instead of being downloaded:
# model weights, the c adapter and the gated grounder come from the operator's MODEL_ROOT (MODELS.md).
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/$(basename "${BASH_SOURCE[0]:-$0}")"
DEPLOY_DIR="$(dirname "$SELF")"
ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"
ENV_FILE="$ROOT/.env.local-gpu"
VENV_CLEF="$ROOT/.venv-clef"
VENV_TRACKER="$ROOT/.venv-tracker"
VENV_APP="$ROOT/.venv"
VENV_LIVE="$ROOT/live-api/.venv"
BACKEND_PORT=8040
LIVE_PORT=8104
LOG_PREFIX="[$(date '+%F %T')]"

log() { printf '%s %s\n' "$LOG_PREFIX" "$*"; }
die() { printf '오류: %s\n' "$*" >&2; exit 1; }

CMD="${1:-}"
case $CMD in
  tracker|clef|backend|live|check) ;;
  -h|--help) sed -n '2,16p' "$SELF"; exit 0 ;;
  "") sed -n '2,16p' "$SELF" >&2; exit 2 ;;
  *) die "알 수 없는 명령입니다: $CMD (tracker | clef | backend | live | check)" ;;
esac
shift || true
(( $# == 0 )) || die "명령 뒤에 인자를 받지 않습니다: $*"

[[ -f $ENV_FILE ]] || die "환경 파일이 없습니다: $ENV_FILE — 먼저 bash deploy/local-gpu/setup.sh 를 실행하세요"
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------ derived model paths (explicit wins)
need_model_root() {
  [[ -n ${MODEL_ROOT:-} ]] || die "MODEL_ROOT가 비어 있습니다 — $ENV_FILE 에 절대 경로를 지정하세요"
  [[ $MODEL_ROOT == /* ]] || die "MODEL_ROOT는 절대 경로여야 합니다: '$MODEL_ROOT'"
}

cuda_devices() { # cuda_devices <label> <CUDA_VISIBLE_DEVICES value> [required]
  local label=$1 value=${2:-} required=${3:-0}
  if [[ -z $value ]]; then
    if (( required )); then
      die "$label: 사용할 GPU가 지정되지 않았습니다(예: 0 또는 0,1)"
    fi
    return 0
  fi
  [[ $value =~ ^[0-9]+(,[0-9]+)*$ ]] || return 0   # UUID/MIG 표기는 그대로 둔다
  local count
  count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
  [[ $count =~ ^[0-9]+$ ]] || return 0
  # 드라이버가 없거나 컨테이너에서 GPU를 셀 수 없으면 검사하지 않는다(잘못된 실패를 만들지 않는다).
  (( count > 0 )) || return 0
  local index
  for index in ${value//,/ }; do
    (( index < count )) || die "$label: GPU $index 를 지정했지만 이 호스트에는 $count개만 보입니다($value)"
  done
}

service_check() { # service_check <venv> <label>
  [[ -x "$1/bin/python" ]] || die "$2 환경이 없습니다($1) — 먼저 bash deploy/local-gpu/setup.sh 를 실행하세요"
}

# 이 앱들은 루프백의 트래커·Clef를 호출한다. 프록시 환경변수가 있으면 그 호출이 프록시로 새므로
# 127.0.0.1/localhost를 NO_PROXY에 덧붙인다(운영자의 기존 목록은 지우지 않는다).
loopback_no_proxy() {
  NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
  no_proxy="$NO_PROXY"
  export NO_PROXY no_proxy
}

# ------------------------------------------------------------------ tracker
cmd_tracker() {
  need_model_root
  service_check "$VENV_TRACKER" "트래커"
  SAM2_SRC="${TRACKER_SAM2_SRC:-$MODEL_ROOT/sam2-src}"
  SAM2_CKPT="${TRACKER_SAM2_CKPT:-$MODEL_ROOT/sam2.1_hiera_base_plus.pt}"
  # 가중치 캐시: .env.local-gpu나 셸에 HF_HOME이 이미 있으면 그대로 쓰고, 없으면 TRACKER_HF_HOME/MODEL_ROOT에서 유도
  HF_HOME="${HF_HOME:-${TRACKER_HF_HOME:-$MODEL_ROOT/hf}}"
  [[ -f $SAM2_SRC/sam2/build_sam.py ]] \
    || die "SAM 2 소스가 없습니다: $SAM2_SRC/sam2/build_sam.py — setup.sh를 실행하거나 TRACKER_SAM2_SRC를 지정하세요"
  [[ -f $SAM2_CKPT ]] \
    || die "SAM 2.1 체크포인트가 없습니다: $SAM2_CKPT — MODELS.md 절차로 준비하세요"
  case "${TRACKER_GROUNDER:-gdino}" in
    sam3) GROUNDER=sam3; GROUNDER_CACHE="models--facebook--sam3" ;;
    gdino) GROUNDER=gdino; GROUNDER_CACHE="models--IDEA-Research--grounding-dino-tiny" ;;
    *) die "TRACKER_GROUNDER는 gdino 또는 sam3여야 합니다: '${TRACKER_GROUNDER:-}'" ;;
  esac
  [[ -d "$HF_HOME/hub/$GROUNDER_CACHE" ]] \
    || die "그라운더 가중치가 HF 캐시에 없습니다: $HF_HOME/hub/$GROUNDER_CACHE — MODELS.md 절차로 준비하세요(이 러너는 내려받지 않습니다)"
  TRACKER_PYDEPS="${TRACKER_PYDEPS:-$("$VENV_TRACKER/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')}"
  cuda_devices "TRACKER_CUDA_VISIBLE_DEVICES" "${TRACKER_CUDA_VISIBLE_DEVICES:-0}" 1

  export CUDA_VISIBLE_DEVICES="${TRACKER_CUDA_VISIBLE_DEVICES:-0}"
  export TRACKER_HOST=127.0.0.1
  export TRACKER_PORT="${TRACKER_PORT:-8090}"
  export TRACKER_SAM2_SRC="$SAM2_SRC"
  export TRACKER_SAM2_CKPT="$SAM2_CKPT"
  export TRACKER_HF_HOME="$HF_HOME"
  export HF_HOME="$HF_HOME"
  export TRACKER_PYDEPS
  export TRACKER_GROUNDER="$GROUNDER"
  export TRACKER_GROUNDER_DEVICE="${TRACKER_GROUNDER_DEVICE:-cuda:0}"
  export TRACKER_SAM2_DEVICE="${TRACKER_SAM2_DEVICE:-cuda:0}"
  export PYTHONPATH="$TRACKER_PYDEPS:$SAM2_SRC:$ROOT/tools/tracker_service${PYTHONPATH:+:$PYTHONPATH}"

  log "트래커 서비스 시작: 127.0.0.1:$TRACKER_PORT (그라운더 $GROUNDER $TRACKER_GROUNDER_DEVICE, SAM 2.1 $TRACKER_SAM2_DEVICE, GPU [$CUDA_VISIBLE_DEVICES])"
  cd "$ROOT"
  exec "$VENV_TRACKER/bin/python" "$ROOT/tools/tracker_service/service.py"
}

# ------------------------------------------------------------------ clef
cmd_clef() {
  need_model_root
  service_check "$VENV_CLEF" "Clef"
  CLEF_MODEL="${CLEF_MODEL:-$MODEL_ROOT/clef}"
  CLEF_ADAPTER="${CLEF_ADAPTER-$MODEL_ROOT/clef-ft-20261004c}"
  [[ -d $CLEF_MODEL ]] \
    || die "Clef 릴리스 스냅샷이 없습니다: $CLEF_MODEL — MODELS.md 절차로 준비하세요"
  if [[ -n $CLEF_ADAPTER ]]; then
    [[ -d $CLEF_ADAPTER ]] \
      || die "Clef c 어댑터가 없습니다: $CLEF_ADAPTER — MODELS.md 절차로 준비하거나, 어댑터 없이 실행하려면 CLEF_ADAPTER= 로 두세요"
  fi
  if [[ -n ${CLEF_QUANT:-} ]]; then
    "$VENV_CLEF/bin/python" -c 'import torchao' 2>/dev/null \
      || die "CLEF_QUANT=${CLEF_QUANT}에 torchao가 필요합니다: uv pip install --python $VENV_CLEF/bin/python torchao==0.14.1"
  fi
  cuda_devices "CLEF_CUDA_VISIBLE_DEVICES" "${CLEF_CUDA_VISIBLE_DEVICES:-0}" 1

  export CUDA_VISIBLE_DEVICES="${CLEF_CUDA_VISIBLE_DEVICES:-0}"
  export CLEF_MODEL CLEF_ADAPTER
  export CLEF_DEVICE="${CLEF_DEVICE:-split}"
  export CLEF_PORT="${CLEF_PORT:-8085}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export PYTHONPATH="$ROOT/tools/clef_ft/vendor${PYTHONPATH:+:$PYTHONPATH}"
  [[ ${CLEF_DEVICE} == "split" ]] \
    || log "경고: CLEF_DEVICE=$CLEF_DEVICE — 배포된 설정은 split(가시 GPU 전체에 분산)입니다"

  log "Clef 결정 서버 시작: 127.0.0.1:$CLEF_PORT (device $CLEF_DEVICE, GPU [$CUDA_VISIBLE_DEVICES], 어댑터 ${CLEF_ADAPTER:-없음}, 양자화 ${CLEF_QUANT:-bf16})"
  cd "$ROOT"
  exec "$VENV_CLEF/bin/python" "$ROOT/tools/clef_ft/serve_clef.py"
}

# ------------------------------------------------------------------ backend
cmd_backend() {
  service_check "$VENV_APP" "백엔드"
  [[ -d $ROOT/web/dist ]] \
    || log "경고: $ROOT/web/dist 가 없습니다 — API만 뜨고 참조 앱 화면은 제공되지 않습니다(setup.sh가 빌드합니다)"
  loopback_no_proxy
  log "백엔드 서버 시작: 127.0.0.1:$BACKEND_PORT (참조 앱은 http://127.0.0.1:$BACKEND_PORT/)"
  cd "$ROOT"
  exec "$VENV_APP/bin/python" -m uvicorn backend.app.main:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" --no-proxy-headers
}

# ------------------------------------------------------------------ live
cmd_live() {
  service_check "$VENV_LIVE" "Live"
  loopback_no_proxy
  log "Live 서버 시작: 127.0.0.1:$LIVE_PORT (진단 클라이언트 http://127.0.0.1:$LIVE_PORT/demo/)"
  cd "$ROOT/live-api"
  exec "$VENV_LIVE/bin/python" -m uvicorn synoptics_live.app:app --host 127.0.0.1 --port "$LIVE_PORT"
}

# ------------------------------------------------------------------ check
cmd_check() {
  command -v python3 >/dev/null 2>&1 || die "python3가 필요합니다"
  local tracker_port="${TRACKER_PORT:-8090}" clef_port="${CLEF_PORT:-8085}"
  python3 - "$BACKEND_PORT" "$LIVE_PORT" "$tracker_port" "$clef_port" <<'PY'
import json
import os
import sys
import urllib.request

backend_port, live_port, tracker_port, clef_port = (int(value) for value in sys.argv[1:5])

# 루프백 확인은 프록시를 우회한다: 환경의 http_proxy/https_proxy가 127.0.0.1 요청을 가로채면
# "서비스가 없다"는 거짓 실패(또는 거짓 성공)가 만들어진다.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROXY_ENV = ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
AMBIENT_PROXIES = [name for name in PROXY_ENV if os.environ.get(name)]
if AMBIENT_PROXIES:
    print("[참고] 프록시 환경변수(%s)가 있습니다 — 이 확인은 127.0.0.1에 직접 연결합니다"
          % ", ".join(AMBIENT_PROXIES))

# (이름, URL, 그 서비스의 /health 본문에 반드시 있어야 하는 키) — 키가 없으면 다른 프로세스가 그 포트를
# 잡고 있는 것이므로 실패다.
TARGETS = (
    ("tracker", "http://127.0.0.1:%d/health" % tracker_port, ("ready", "device")),
    ("clef", "http://127.0.0.1:%d/health" % clef_port, ("ok", "load_s")),
    ("backend", "http://127.0.0.1:%d/api/health" % backend_port, ("tracker_ready", "guide_ready")),
    ("live", "http://127.0.0.1:%d/v1/health" % live_port, ("engine", "tts_ready")),
)
reach: list[str] = []      # /health를 읽지 못한 서비스 — 줄마다 즉시 출력
findings: list[str] = []   # 준비 상태·레인 불일치 실패 — 마지막에 사유와 함께 출력
bodies: dict[str, dict] = {}


def probe(name: str, url: str, keys: tuple) -> None:
    try:
        with OPENER.open(url, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
    except Exception as exc:  # 연결 실패, 비 2xx, JSON 아님 — 모두 이 서비스의 실패다
        reach.append(name)
        print("[실패] %-7s %s  %s: %s" % (name, url, type(exc).__name__, exc))
        return
    if not isinstance(body, dict) or any(key not in body for key in keys):
        reach.append(name)
        print("[실패] %-7s %s  %s 서비스의 /health 형식이 아닙니다 (기대 키 %s)"
              % (name, url, name, ", ".join(keys)))
        return
    bodies[name] = body
    print("[%s] %-7s %s  %s"
          % ("정상" if ready(name, body) else "미준비", name, url, describe(name, body)))


def describe(name: str, body: dict) -> str:
    if name == "tracker":
        return f"ready={body.get('ready')} device={body.get('device')}"
    if name == "clef":
        return ("ok={ok} device={device} 추론 GPU={gpu} 로드={load_s}s".format(
            ok=body.get("ok"), device=body.get("device"),
            gpu=sorted((body.get("gpu_gib") or {}).keys()), load_s=body.get("load_s")))
    if name == "backend":
        return ("ready=%s access_mode=%s tracker_ready=%s follow_provider=%s follow_ready=%s"
                % (body.get("ready"), body.get("access_mode"), body.get("tracker_ready"),
                   body.get("follow_provider"), body.get("follow_ready")))
    return ("ready=%s engine=%s tracker_ready=%s follow_ready=%s tts_ready=%s"
            % (body.get("ready"), body.get("engine"), body.get("tracker_ready"),
               body.get("follow_ready"), body.get("tts_ready")))


def ready(name: str, body: dict) -> bool:
    """이 서비스가 스스로 보고한 준비 상태. Clef는 ok, 나머지는 ready다."""
    return body.get("ok" if name == "clef" else "ready") is True


for name, url, keys in TARGETS:
    probe(name, url, keys)

# 준비 플래그 → 사람이 읽는 사유. ready=false는 실패이며, 어느 레인이 꺼졌는지 이름으로 말한다.
REASONS = {
    "tracker_ready": "트래커 레인 미준비(AISW_TRACKER_URL 또는 트래커 서비스)",
    "guide_ready": "기본 안내 레인 미준비(DeepSeek 상위 계획 키)",
    "plan_ready": "상위 계획 프로필 미준비(유료 계획 키)",
    "follow_ready": "단계 판정 레인 미준비(AISW_FOLLOW_CLEF_URL 또는 Clef)",
}
READY_FLAGS = ("guide_ready", "plan_ready", "follow_ready", "tracker_ready")
PROXY_HINT = (" — 프록시 환경변수(%s)로 루프백이 우회되는지도 확인하세요(NO_PROXY에 127.0.0.1)"
              % ", ".join(AMBIENT_PROXIES)) if AMBIENT_PROXIES else ""


def not_ready_reason(body: dict) -> str:
    flags = [flag for flag in READY_FLAGS if body.get(flag) is not True]
    if not flags:
        return "레인 플래그는 모두 true입니다 — 상류(demo backend)의 /health를 확인하세요"
    return ", ".join("%s=false(%s)" % (flag, REASONS[flag]) for flag in flags)


# 모델 서비스는 스스로의 준비 상태를 보고해야 한다(모델 로드 실패는 /health에 그대로 드러난다).
tracker = bodies.get("tracker")
if tracker is not None and tracker.get("ready") is not True:
    findings.append("tracker: ready=false error=%r" % (tracker.get("error"),))
clef = bodies.get("clef")
if clef is not None and clef.get("ok") is not True:
    findings.append("clef: ok=false error=%r" % (clef.get("error"),))

# backend/Live의 ready=false도 실패다(사유를 함께 말한다). ready=true인데 레인이 꺼져 있으면 불일치다.
for name in ("backend", "live"):
    body = bodies.get(name)
    if body is None:
        continue
    if body.get("ready") is not True:
        findings.append("%s: ready=false — %s%s" % (name, not_ready_reason(body), PROXY_HINT))
        continue
    if tracker is not None and tracker.get("ready") is True and body.get("tracker_ready") is not True:
        findings.append("%s: ready=true인데 tracker_ready=false — AISW_TRACKER_URL이 이 트래커를 보지 못합니다" % name)
    if clef is not None and clef.get("ok") is True and body.get("follow_provider") == "clef" \
            and body.get("follow_ready") is not True:
        findings.append("%s: ready=true인데 follow_ready=false — AISW_FOLLOW_CLEF_URL이 이 Clef를 보지 못합니다" % name)

for item in findings:
    print("[실패] " + item)
print("확인 완료: /health 응답 %d개, 연결 실패 %d개, 준비·불일치 실패 %d개"
      % (len(bodies), len(reach), len(findings)))
sys.exit(1 if (reach or findings) else 0)
PY
}

case $CMD in
  tracker) cmd_tracker ;;
  clef)    cmd_clef ;;
  backend) cmd_backend ;;
  live)    cmd_live ;;
  check)   cmd_check ;;
esac
