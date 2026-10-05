#!/usr/bin/env bash
# Run the standalone synoptics tracker service on the GPU host (loopback only).
#
#   bash deploy/tracker/run.sh                 # start (two-GPU: grounder cuda:0, SAM 2.1 cuda:1)
#   bash deploy/tracker/run.sh --logs          # follow logs
#   bash deploy/tracker/run.sh --stop          # stop and remove the OWNED service container
#   bash deploy/tracker/run.sh --push          # copy the service sources into the volume first
#   bash deploy/tracker/run.sh --migrate <id>  # ONE-TIME: retire the known unlabelled legacy container
#
# Options (before the action; both repeatable/combinable):
#   --port <n>         service port on the GPU host (same as TRACKER_PORT; the default name is 8090-only)
#   --env KEY=VALUE    extra TRACKER_* env for the service container (e.g. TRACKER_OCCLUDED_MAX_FRAMES=30)
# A second, side-by-side instance (measurement spike) needs its own name and source dir, e.g.
#   TRACKER_CONTAINER=synoptics-tracker-spike TRACKER_SERVICE_DIR=/v/tracker/service-spike \
#     bash deploy/tracker/run.sh --port 8093 --push && ... --port 8093 --env TRACKER_OCCLUDED_MAX_FRAMES=30
#
# Safety contract (the reason this script is written the long way):
#   * Every container this script creates carries ownership labels `synoptics.owner` / `synoptics.role`.
#     A container is OURS only when both labels match. Anything else (foreign labels, partially
#     matching labels, or no labels at all) is refused, never removed.
#   * The identity and labels of an existing same-name container are inspected FIRST, and any removal is
#     by the verified immutable 64-hex container ID, never by name. A non-"not found" inspect error
#     (daemon down, permission denied, …) aborts without removing anything.
#   * Unexpected Docker failures are never suppressed: no `|| true` on any mutating call and no blanket
#     `>/dev/null`. The single tolerated non-fatal call is the best-effort log dump, which reports itself
#     when it cannot read logs instead of hiding the failure it was there to explain.
#   * `--migrate` is the single, explicit, one-time path for the pre-existing unlabelled container. It
#     requires the full 64-hex ID recorded when that container was created and re-validates ID + name +
#     image + command + volume binding before removing exactly that ID. It never whitelists unlabelled
#     containers by name.
#   * Env/args are validated against strict character classes and single-quoted before they are
#     interpolated into a remote command, so a hostile value cannot become shell syntax.
#   * The service sources are streamed to the GPU host over SSH stdin; nothing is base64-encoded onto a
#     command line.
#
# Why sources are mounted from the volume instead of baked into an image: the pinned SAM 2 source tree,
# both pinned checkpoints and the Python dependency target already live on the synoptics-gpu-repro volume
# (/v, Docker partition) by requirement, so an image would add a build step and a second copy of the code
# with no isolation benefit. This runner is the single tested deploy path.
#
# Both GPUs are exposed to the service (NVIDIA_VISIBLE_DEVICES=0,1): the grounder is placed on cuda:0 and
# the tracker on cuda:1. The llama server on the host keeps using GPU memory too, which is fine — it is a
# co-tenant this script must never touch, and its container name is reserved below.
#
# The service binds 127.0.0.1:${TRACKER_PORT} ON THE GPU HOST. The application reaches it through an SSH
# local forward, so the app-side URL stays a loopback URL (the app validates that):
#     ssh -N -L 8090:127.0.0.1:8090 riey@192.168.219.250
# (Direct IP by default: the `vast-gpu-node` alias hangs from the dev host. Override with TRACKER_SSH_HOST.)
set -euo pipefail

# ---------------------------------------------------------------- configuration
HOST=${TRACKER_SSH_HOST:-riey@192.168.219.250}
NAME=${TRACKER_CONTAINER:-synoptics-tracker}
PORT=${TRACKER_PORT:-8090}
IMAGE=${TRACKER_IMAGE:-pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime}
VOLUME=${TRACKER_VOLUME:-synoptics-gpu-repro}
GPU=${TRACKER_GPU:-0,1}
OWNER=${TRACKER_OWNER:-aisw-hybrid}
GROUNDER_DEVICE=${TRACKER_GROUNDER_DEVICE:-cuda:0}
SAM2_DEVICE=${TRACKER_SAM2_DEVICE:-cuda:1}
HEALTH_WAIT=${TRACKER_HEALTH_WAIT:-90}
ROLE_SERVICE=tracker-service
ROLE_HELPER=tracker-helper
DEFAULT_NAME=synoptics-tracker
DEFAULT_PORT=8090
# Where the service sources live on the volume. The production instance runs from the default dir; a
# side-by-side instance gets its own dir so pushing its sources never rewrites production's code.
SERVICE_DIR=${TRACKER_SERVICE_DIR:-/v/tracker/service}
# The legacy-migration evidence always refers to the original production path, whatever SERVICE_DIR is.
SERVICE_SRC_MARK=/v/tracker/service/service.py
SRC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../tools/tracker_service" && pwd)

# Containers this script must never remove or replace, whatever the env says.
RESERVED_NAMES="synoptics-llama-b11146 synoptics-llama"

die() { printf '오류: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- validation
# Values interpolate into remote shell text, so they are restricted to character classes that carry no
# shell syntax. A single quote is impossible in every class below.
RE_HOST='^[A-Za-z0-9_][A-Za-z0-9_.@:-]*$'
RE_NAME='^[A-Za-z0-9][A-Za-z0-9_.-]*$'
RE_PORT='^[0-9]{1,5}$'
RE_VOLUME='^[A-Za-z0-9][A-Za-z0-9_.-]*$'
RE_IMAGE='^[A-Za-z0-9][A-Za-z0-9._/:-]*$'
RE_GPU='^[0-9]+(,[0-9]+)*$'
RE_DEVICE='^[a-z]+(:[0-9]+)?$'
RE_DIGITS='^[0-9]{1,5}$'
RE_SERVICE_DIR='^/v/tracker/[A-Za-z0-9_-]+$'
RE_ENV_KEY='^TRACKER_[A-Z0-9_]{1,60}$'
RE_ENV_VALUE='^[A-Za-z0-9_.,:/+-]{1,256}$'
# Keys the script itself sets from validated config; --env may not override them.
RESERVED_ENV_KEYS="TRACKER_HOST TRACKER_PORT TRACKER_GROUNDER_DEVICE TRACKER_SAM2_DEVICE"

# Parsed from the command line (see dispatch). EXTRA_ENV holds validated KEY=VALUE pairs.
ACTION='' ACTION_ARG='' EXTRA_ENV=()

need() { # need <label> <value> <regex>
  [[ $2 =~ $3 ]] || die "$1 값이 허용된 형식이 아닙니다: '$2' (기대 형식 $3)"
}

validate_config() {
  need HOST "$HOST" "$RE_HOST"
  need NAME "$NAME" "$RE_NAME"
  need PORT "$PORT" "$RE_PORT"
  need VOLUME "$VOLUME" "$RE_VOLUME"
  need IMAGE "$IMAGE" "$RE_IMAGE"
  need GPU "$GPU" "$RE_GPU"
  need OWNER "$OWNER" "$RE_NAME"
  need GROUNDER_DEVICE "$GROUNDER_DEVICE" "$RE_DEVICE"
  need SAM2_DEVICE "$SAM2_DEVICE" "$RE_DEVICE"
  need HEALTH_WAIT "$HEALTH_WAIT" "$RE_DIGITS"
  need SERVICE_DIR "$SERVICE_DIR" "$RE_SERVICE_DIR"
  (( 10#$PORT >= 1 && 10#$PORT <= 65535 )) || die "PORT 범위를 벗어났습니다: $PORT"
  local r kv key
  for r in $RESERVED_NAMES; do
    [[ $NAME == "$r" ]] && die "예약된 컨테이너 이름입니다(이 스크립트가 다루지 않습니다): $NAME"
  done
  # The default name is the production service on the default port. Starting it on another port (or from
  # another source dir) would REPLACE production — a side-by-side instance must carry its own name.
  if [[ $NAME == "$DEFAULT_NAME" ]]; then
    [[ $PORT == "$DEFAULT_PORT" ]] \
      || die "기본 컨테이너 이름($DEFAULT_NAME)은 포트 $DEFAULT_PORT 전용입니다 — 다른 포트는 TRACKER_CONTAINER로 별도 이름을 주세요"
    [[ $SERVICE_DIR == /v/tracker/service ]] \
      || die "기본 컨테이너 이름($DEFAULT_NAME)은 /v/tracker/service 전용입니다 — 다른 소스 디렉터리는 별도 이름으로만"
  fi
  for kv in ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}; do
    [[ $kv == *=* ]] || die "--env 는 KEY=VALUE 형식이어야 합니다: '$kv'"
    key=${kv%%=*}
    need "--env 키" "$key" "$RE_ENV_KEY"
    need "--env 값($key)" "${kv#*=}" "$RE_ENV_VALUE"
    for r in $RESERVED_ENV_KEYS; do
      [[ $key == "$r" ]] && die "--env 로 덮어쓸 수 없는 키입니다(스크립트가 직접 설정): $key"
    done
  done
  return 0
}

# ---------------------------------------------------------------- remote plumbing
remote() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$@"; }
q() { printf "'%s'" "$1"; }                     # single-quote a validated value

# show_logs <id>: best-effort diagnostics. This is the only non-fatal remote call in this script, and it
# reports itself when it fails instead of swallowing the failure it exists to explain.
show_logs() {
  local out
  if ! out=$(remote "docker logs --tail 40 $(q "$1")" 2>&1); then
    printf '    (컨테이너 로그를 읽지 못했습니다 — 위 오류 메시지가 우선입니다)\n' >&2
    return 0
  fi
  printf '%s\n' "$out" | sed 's/^/    /'
}

# Inspect one line: id|name|image|owner-label|role-label|mounts|cmd
inspect_probe() {
  remote "docker inspect -f '{{.Id}}|{{.Name}}|{{.Config.Image}}|{{index .Config.Labels \"synoptics.owner\"}}|{{index .Config.Labels \"synoptics.role\"}}|{{range .Mounts}}{{.Name}}:{{.Destination}},{{end}}|{{range .Config.Cmd}}{{.}} {{end}}' $(q "$1")" 2>&1
}
inspect_running() { remote "docker inspect -f '{{.State.Running}}' $(q "$1")" 2>&1; }

NOT_FOUND_RE='No such (object|container|image)'

# probe_container <name-or-id>
#   EXIST=1 -> CID CNAME CIMAGE COWNER CROLE CMOUNTS CCMD are set
#   EXIST=0 -> container absent (the only tolerated non-zero inspect result)
#   anything else -> abort WITHOUT removing anything
EXIST=0 CID='' CNAME='' CIMAGE='' COWNER='' CROLE='' CMOUNTS='' CCMD=''
probe_container() {
  local out rc
  if out=$(inspect_probe "$1"); then rc=0; else rc=$?; fi
  if (( rc == 0 )); then
    IFS='|' read -r CID CNAME CIMAGE COWNER CROLE CMOUNTS CCMD <<<"$out"
    [[ $CID =~ ^[0-9a-f]{64}$ ]] || die "inspect 결과의 컨테이너 ID가 예상 형식이 아닙니다 — 중단합니다: '$CID'"
    # docker inspect prints `<no value>` for a template that reads a missing map key — treat it as absent.
    [[ $COWNER == '<no value>' ]] && COWNER=''
    [[ $CROLE == '<no value>' ]] && CROLE=''
    EXIST=1
    return 0
  fi
  if [[ $out =~ $NOT_FOUND_RE ]]; then
    EXIST=0
    return 0
  fi
  printf '%s\n' "$out" >&2
  die "예상치 못한 docker inspect 오류 — 아무것도 제거하지 않았습니다 (대상 '$1', rc=$rc)"
}

describe_probe() {
  printf '  id=%s\n  name=%s\n  image=%s\n  label synoptics.owner=%s\n  label synoptics.role=%s\n  mounts=%s\n  cmd=%s\n' \
    "$CID" "$CNAME" "$CIMAGE" "${COWNER:-<없음>}" "${CROLE:-<없음>}" "$CMOUNTS" "$CCMD"
}

is_owned() { [[ $COWNER == "$OWNER" && $CROLE == "$ROLE_SERVICE" ]]; }

# remove_container_by_id <verified-64hex-id>
remove_container_by_id() {
  [[ $1 =~ ^[0-9a-f]{64}$ ]] || die "제거 거부: 검증된 전체 컨테이너 ID가 아닙니다: '$1'"
  if ! out=$(remote "docker rm -f $(q "$1")" 2>&1); then
    printf '%s\n' "$out" >&2
    die "컨테이너 제거에 실패했습니다 (id=$1)"
  fi
  # Re-check the end state before claiming anything.
  if out=$(inspect_probe "$1"); then
    describe_probe
    die "제거를 보고받았지만 컨테이너가 아직 존재합니다 (id=$1)"
  fi
  if [[ ! $out =~ $NOT_FOUND_RE ]]; then
    printf '%s\n' "$out" >&2
    die "제거 후 상태 확인이 실패했습니다 (id=$1) — 직접 확인하세요"
  fi
  printf '제거 완료: %s\n' "$1"
}

# refuse_existing <reason>
refuse_existing() {
  printf '오류: %s\n' "$1" >&2
  describe_probe >&2
  printf '  이 스크립트는 소유권이 확인되지 않은 컨테이너를 제거하지 않습니다.\n' >&2
  exit 1
}

# Ensure no same-name container blocks a start, removing it only when it is verifiably ours.
clear_owned_or_refuse() {
  probe_container "$NAME"
  (( EXIST == 1 )) || return 0
  if is_owned; then
    printf '같은 이름의 소유 컨테이너를 발견했습니다(교체합니다):\n'
    describe_probe
    remove_container_by_id "$CID"
    return 0
  fi
  if [[ -z $COWNER && -z $CROLE ]]; then
    printf '  무라벨 레거시 1회 이관(전체 ID로 증거 재검증): bash deploy/tracker/run.sh --migrate %s\n' "$CID" >&2
    refuse_existing "같은 이름의 '무라벨' 컨테이너가 있습니다 — 자동 교체하지 않습니다."
  fi
  refuse_existing "같은 이름의 컨테이너가 있으나 소유권 라벨이 우리 것이 아닙니다."
}

# ---------------------------------------------------------------- subcommands
push_sources() {
  [[ -f $SRC_DIR/service.py && -f $SRC_DIR/models.py ]] \
    || die "서비스 소스를 찾을 수 없습니다: $SRC_DIR (service.py/models.py)"
  echo "== pushing $SRC_DIR -> volume:$VOLUME:$SERVICE_DIR (SSH stdin) =="
  # SERVICE_DIR is validated against RE_SERVICE_DIR (no quote, no space, no '..'), so it is safe inside
  # the single-quoted bash -c text below.
  if ! tar -C "$SRC_DIR" -czf - . | remote "docker run --rm -i \
      --label synoptics.owner=$(q "$OWNER") --label synoptics.role=$(q "$ROLE_HELPER") \
      --log-driver local -v $(q "$VOLUME:/v") python:3.12-slim \
      bash -c 'mkdir -p $SERVICE_DIR && tar -C $SERVICE_DIR -xzf - && ls -l $SERVICE_DIR'"; then
    die "소스 전송에 실패했습니다 (아무 서비스 컨테이너도 건드리지 않았습니다)"
  fi
}

start_service() {
  local cid out waited=0 h kv extra=''
  for kv in ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}; do
    extra+=" -e $(q "$kv")"
  done
  echo "== starting $NAME on $HOST (host loopback 127.0.0.1:$PORT, sources $SERVICE_DIR) =="
  if ! cid=$(remote "docker run -d --name $(q "$NAME") \
      --label synoptics.owner=$(q "$OWNER") --label synoptics.role=$(q "$ROLE_SERVICE") \
      --network=host --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$(q "$GPU") \
      --log-driver local \
      -v $(q "$VOLUME:/v") \
      -e PYTHONPATH=$(q "/v/tracker/pydeps:/v/tracker/src:$SERVICE_DIR") \
      -e TRACKER_HOST=127.0.0.1 -e TRACKER_PORT=$(q "$PORT") \
      -e HF_HOME=/v/tracker/hf \
      -e TRACKER_GROUNDER_DEVICE=$(q "$GROUNDER_DEVICE") \
      -e TRACKER_SAM2_DEVICE=$(q "$SAM2_DEVICE")$extra \
      $(q "$IMAGE") python3 $(q "$SERVICE_DIR/service.py")" 2>&1); then
    printf '%s\n' "$cid" >&2
    die "컨테이너 시작에 실패했습니다"
  fi
  [[ $cid =~ ^[0-9a-f]{12,64}$ ]] || die "docker run -d가 컨테이너 ID를 반환하지 않았습니다: '$cid'"
  if [[ $(inspect_running "$cid") != "true" ]]; then
    show_logs "$cid"
    die "컨테이너가 실행 중이 아닙니다 (id=$cid)"
  fi
  printf 'started %s (id=%s, 127.0.0.1:%s)\n' "$NAME" "$cid" "$PORT"

  while (( waited < HEALTH_WAIT )); do
    if h=$(remote "docker exec $(q "$cid") python3 -c \"import urllib.request,sys; sys.stdout.write(urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5).read().decode())\"" 2>/dev/null); then
      printf 'health: %s\n' "$h"
      return 0
    fi
    sleep 2
    waited=$(( waited + 2 ))
  done
  show_logs "$cid"
  die "컨테이너는 실행 중이지만 ${HEALTH_WAIT}s 안에 /health가 응답하지 않았습니다 — 로그를 확인하세요 (id=$cid)"
}

migrate_legacy() { # $1 = full 64-hex id recorded when the legacy container was created
  local id=$1
  [[ $id =~ ^[0-9a-f]{64}$ ]] || die "--migrate에는 기록해 둔 전체 컨테이너 ID(64 hex)가 필요합니다"
  probe_container "$id"
  (( EXIST == 1 )) || die "그 ID의 컨테이너가 없습니다 — 아무것도 제거하지 않았습니다 (id=$id)"
  [[ $CID == "$id" ]] || die "ID 재검증 실패 — 제거하지 않았습니다 (요청 '$id', 실제 '$CID')"
  [[ $CNAME == "/$NAME" ]] || die "레거시 검증 실패: 이름 불일치 (기대 '/$NAME', 실제 '$CNAME') — 제거하지 않았습니다"
  [[ $CIMAGE == "$IMAGE" ]] || die "레거시 검증 실패: 이미지 불일치 (기대 '$IMAGE', 실제 '$CIMAGE') — 제거하지 않았습니다"
  [[ $CMOUNTS == *"$VOLUME:/v,"* ]] || die "레거시 검증 실패: 볼륨 바인딩 불일치 (기대 '$VOLUME:/v') — 제거하지 않았습니다"
  [[ $CCMD == *"$SERVICE_SRC_MARK"* ]] || die "레거시 검증 실패: 실행 명령 불일치 (기대 '$SERVICE_SRC_MARK') — 제거하지 않았습니다"
  [[ -z $COWNER && -z $CROLE ]] || die "이미 소유권 라벨이 있는 컨테이너입니다 — --migrate는 무라벨 레거시 1회용입니다(일반 start/stop을 쓰세요)"
  echo "== legacy 컨테이너 재검증 통과 (1회 이관) =="
  describe_probe
  remove_container_by_id "$CID"
  start_service
}

stop_service() {
  probe_container "$NAME"
  if (( EXIST == 0 )); then
    echo "$NAME 컨테이너가 없습니다 (정지할 것 없음)"
    return 0
  fi
  if is_owned; then
    remove_container_by_id "$CID"
    return 0
  fi
  if [[ -z $COWNER && -z $CROLE ]]; then
    printf '  무라벨 레거시 1회 이관(전체 ID로 증거 재검증): bash deploy/tracker/run.sh --migrate %s\n' "$CID" >&2
    refuse_existing "같은 이름의 '무라벨' 컨테이너가 있습니다 — 제거하지 않았습니다."
  fi
  refuse_existing "같은 이름의 컨테이너가 있으나 소유권 라벨이 우리 것이 아닙니다 — 제거하지 않았습니다."
}

# ---------------------------------------------------------------- dispatch
set_action() {
  [[ -z $ACTION ]] || die "동작은 하나만 지정하세요: $ACTION 와 $1"
  ACTION=$1
}
while (( $# > 0 )); do
  case $1 in
    --port)
      [[ -n ${2:-} ]] || die "--port <번호> 형식으로 실행하세요"
      PORT=$2; shift 2 ;;
    --env)
      [[ -n ${2:-} ]] || die "--env KEY=VALUE 형식으로 실행하세요"
      EXTRA_ENV+=("$2"); shift 2 ;;
    --stop|--logs|--push) set_action "$1"; shift ;;
    --migrate)
      set_action "$1"
      [[ -n ${2:-} ]] || die "--migrate <전체 컨테이너 ID(64 hex)> 형식으로 실행하세요"
      ACTION_ARG=$2; shift 2 ;;
    -h|--help)
      sed -n '2,15p' "${BASH_SOURCE[0]:-$0}"
      exit 0
      ;;
    '') shift ;;
    *) die "알 수 없는 옵션입니다: $1 (--port, --env, --logs, --stop, --push, --migrate)" ;;
  esac
done

validate_config

case "$ACTION" in
  --stop)    stop_service; exit 0 ;;
  --logs)    remote "docker logs --tail 120 -f $(q "$NAME")"; exit 0 ;;
  --push)    push_sources; exit 0 ;;
  --migrate) migrate_legacy "$ACTION_ARG"; exit 0 ;;
  '') : ;;
esac

clear_owned_or_refuse
start_service
