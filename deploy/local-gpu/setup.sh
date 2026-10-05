#!/usr/bin/env bash
# Install the local runtimes for this source on a standalone Linux + NVIDIA server.
#
#   bash deploy/local-gpu/setup.sh              CPU runtimes + GPU runtimes + SAM 2 source
#   bash deploy/local-gpu/setup.sh --cpu-only   CPU runtimes only (no GPU libraries, no SAM 2 source)
#
# CPU half  : <repo>/.venv (backend), <repo>/live-api/.venv (Live), npm dependencies, web/dist build.
# GPU half  : <repo>/.venv-clef (Clef decision server) and <repo>/.venv-tracker (tracker service), both
#             Python 3.12, plus the pinned SAM 2 source checkout under $MODEL_ROOT.
# It also writes <repo>/.env.local-gpu from deploy/local-gpu/.env.example when that file is absent.
#
# This script installs dependencies only. It does NOT download or verify model weights, accept model
# licences, install system packages (gcc comes from your distribution), start any service, or touch any
# existing deployment. Prepare the weights yourself as described in MODELS.md, then set MODEL_ROOT in
# <repo>/.env.local-gpu and run the services with deploy/local-gpu/run.sh.
#
# Requirements: Linux x86_64, uv, git, Node.js 24+/npm, network access to PyPI/npm/GitHub.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/$(basename "${BASH_SOURCE[0]:-$0}")"
DEPLOY_DIR="$(dirname "$SELF")"
ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"
ENV_TEMPLATE="$DEPLOY_DIR/.env.example"
ENV_FILE="$ROOT/.env.local-gpu"
REQ_CLEF="$DEPLOY_DIR/requirements-clef.txt"
REQ_TRACKER="$DEPLOY_DIR/requirements-tracker.txt"
VENV_CLEF="$ROOT/.venv-clef"
VENV_TRACKER="$ROOT/.venv-tracker"
PYTHON_PIN="3.12"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_PIN=(torch==2.9.1 torchvision==0.24.1)
SAM2_REPO_URL="https://github.com/facebookresearch/sam2.git"
# SAM 2.1 source revision (facebookresearch/sam2), also declared by the tracker runtime.
# This commit contains sam2/build_sam.py and sam2/configs/sam2.1/sam2.1_hiera_b+.yaml.
SAM2_REVISION="2b90b9f5ceec907a1c18123530e92e794ad901a4"
FLA_PIN="flash-linear-attention==0.5.2"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { printf '오류: %s\n' "$*" >&2; exit 1; }

CPU_ONLY=0
while (( $# > 0 )); do
  case $1 in
    --cpu-only) CPU_ONLY=1; shift ;;
    -h|--help) sed -n '2,17p' "$SELF"; exit 0 ;;
    *) die "알 수 없는 옵션입니다: $1 (--cpu-only)" ;;
  esac
done

# ------------------------------------------------------------------ preflight
command -v uv >/dev/null 2>&1 || die "uv가 필요합니다 (https://docs.astral.sh/uv/ 설치 후 다시 실행하세요)"
command -v npm >/dev/null 2>&1 || die "npm(Node.js 24+)이 필요합니다"
command -v node >/dev/null 2>&1 || die "node(Node.js 24+)가 필요합니다"
node_major="$(node -p 'process.versions.node.split(".")[0]')"
[[ $node_major =~ ^[0-9]+$ && $node_major -ge 24 ]] \
  || die "Node.js 24+가 필요합니다 (현재 $(node --version))"
[[ -f "$ENV_TEMPLATE" ]] || die "환경 예시 파일이 없습니다: $ENV_TEMPLATE"
[[ -f "$REQ_CLEF" && -f "$REQ_TRACKER" ]] || die "요구사항 파일이 없습니다: $DEPLOY_DIR/requirements-*.txt"

log "소스 루트: $ROOT"
command -v nvidia-smi >/dev/null 2>&1 \
  || log "경고: nvidia-smi가 없습니다 — 런타임은 설치되지만 GPU 서비스는 실행되지 않습니다"

# ------------------------------------------------------------------ CPU runtimes
log "== 백엔드 런타임: $ROOT/.venv (Python $PYTHON_PIN) =="
( cd "$ROOT" && uv sync --frozen --python "$PYTHON_PIN" )

log "== Live 런타임: $ROOT/live-api/.venv (Python $PYTHON_PIN) =="
( cd "$ROOT/live-api" && uv sync --frozen --python "$PYTHON_PIN" )

log "== 참조 프런트엔드: npm 의존성 + web/dist =="
( cd "$ROOT" && npm ci --no-audit --no-fund && npm run build )
[[ -d "$ROOT/web/dist" ]] || die "프런트엔드 빌드 산출물이 없습니다: $ROOT/web/dist"

# ------------------------------------------------------------------ environment file
if [[ -f "$ENV_FILE" ]]; then
  log "환경 파일을 유지합니다: $ENV_FILE"
else
  cp "$ENV_TEMPLATE" "$ENV_FILE"
  log "환경 파일을 만들었습니다: $ENV_FILE — MODEL_ROOT를 실제 경로로 채우세요"
fi

if (( CPU_ONLY == 1 )); then
  log "== --cpu-only: GPU 런타임(.venv-clef/.venv-tracker)과 SAM 2 소스는 설치하지 않았습니다 =="
  log "완료. GPU 실행 준비는 배포 문서의 전체 setup을 사용하세요: bash deploy/local-gpu/setup.sh"
  exit 0
fi

# ------------------------------------------------------------------ GPU runtimes
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

MODEL_ROOT="${MODEL_ROOT:-}"
[[ -n $MODEL_ROOT ]] || die "MODEL_ROOT가 비어 있습니다 — $ENV_FILE 에 절대 경로를 지정한 뒤 다시 실행하세요"
[[ $MODEL_ROOT == /* ]] || die "MODEL_ROOT는 절대 경로여야 합니다: '$MODEL_ROOT'"
[[ $MODEL_ROOT != *".."* ]] || die "MODEL_ROOT에 '..'를 쓸 수 없습니다: '$MODEL_ROOT'"
command -v git >/dev/null 2>&1 || die "git이 필요합니다 (SAM 2 소스 체크아웃)"
mkdir -p "$MODEL_ROOT" || die "MODEL_ROOT를 만들 수 없습니다: $MODEL_ROOT"
[[ -w $MODEL_ROOT ]] || die "MODEL_ROOT에 쓸 수 없습니다: $MODEL_ROOT"
case "$MODEL_ROOT/" in
  "$ROOT"/*)
    if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1 \
       && git -C "$ROOT" check-ignore -q --no-index "$MODEL_ROOT/probe" 2>/dev/null; then
      log "참고: MODEL_ROOT가 저장소 안에 있지만 .gitignore로 제외됩니다: $MODEL_ROOT"
    else
      log "경고: MODEL_ROOT가 저장소 안에 있고 .gitignore에도 없습니다($MODEL_ROOT) — 모델 파일이 버전 관리에 들어가지 않게 하세요"
    fi ;;
esac
command -v gcc >/dev/null 2>&1 \
  || log "경고: gcc가 없습니다 — TRACKER_SAM2_COMPILE=image_encoder와 flash-linear-attention 빌드가 실패할 수 있습니다 (예: sudo apt-get install -y gcc libc6-dev)"
log "MODEL_ROOT: $MODEL_ROOT"

make_venv() { # make_venv <path> <label>
  local venv=$1 label=$2 version
  if [[ -d $venv ]]; then
    [[ -x $venv/bin/python ]] || die "$label 환경이 손상되었습니다($venv) — 지운 뒤 다시 실행하세요: rm -rf $venv"
    version="$("$venv/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    [[ $version == "$PYTHON_PIN" ]] \
      || die "$label 환경이 Python $version입니다(기대 $PYTHON_PIN) — 지운 뒤 다시 실행하세요: rm -rf $venv"
    log "$label 환경을 재사용합니다: $venv (Python $version)"
    return 0
  fi
  log "$label 환경을 만듭니다: $venv (Python $PYTHON_PIN)"
  uv venv --seed --python "$PYTHON_PIN" "$venv"
}

install_torch() { # install_torch <venv>
  log "PyTorch CUDA 12.8 휠 설치: ${TORCH_PIN[*]} -> $1"
  uv pip install --python "$1/bin/python" "${TORCH_PIN[@]}" --index-url "$TORCH_INDEX"
}

install_requirements() { # install_requirements <venv> <requirements file> <label>
  log "$3 의존성 설치: $(basename "$2")"
  uv pip install --python "$1/bin/python" -r "$2"
}

make_venv "$VENV_CLEF" "Clef"
make_venv "$VENV_TRACKER" "트래커"
install_torch "$VENV_CLEF"
install_torch "$VENV_TRACKER"
install_requirements "$VENV_CLEF" "$REQ_CLEF" "Clef"
install_requirements "$VENV_TRACKER" "$REQ_TRACKER" "트래커"

# Optional accelerator for the gated-delta layers: a failure here only means the slower (identical) path.
if uv pip install --python "$VENV_CLEF/bin/python" "$FLA_PIN"; then
  log "선택 설치 성공: $FLA_PIN"
else
  log "경고: $FLA_PIN 설치 실패 — transformers가 순수 torch 경로로 동작합니다(느리지만 같은 결과)"
fi

# ------------------------------------------------------------------ SAM 2 source (pinned)
SAM2_SRC="${TRACKER_SAM2_SRC:-$MODEL_ROOT/sam2-src}"
sam2_checkout() { # fetch the pinned revision into an empty directory
  git init -q "$1"
  git -C "$1" remote add origin "$SAM2_REPO_URL"
  git -C "$1" fetch -q --depth 1 origin "$SAM2_REVISION"
  git -C "$1" checkout -q --detach FETCH_HEAD
}

if [[ -d $SAM2_SRC ]]; then
  [[ -f $SAM2_SRC/sam2/build_sam.py && -f "$SAM2_SRC/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml" ]] \
    || die "SAM 2 소스 트리가 아닙니다(sam2/build_sam.py, sam2/configs/sam2.1 없음): $SAM2_SRC — 지우고 다시 실행하거나 TRACKER_SAM2_SRC를 바꾸세요"
  if git -C "$SAM2_SRC" rev-parse --git-dir >/dev/null 2>&1; then
    sam2_head="$(git -C "$SAM2_SRC" rev-parse HEAD)"
    [[ $sam2_head == "$SAM2_REVISION" ]] \
      || die "SAM 2 소스 리비전 불일치: $SAM2_SRC HEAD=$sam2_head (기대 $SAM2_REVISION) — 지우고 다시 실행하거나 TRACKER_SAM2_SRC를 바꾸세요"
    log "SAM 2 소스를 재사용합니다: $SAM2_SRC @ $sam2_head"
  else
    die "SAM 2 소스의 고정 리비전을 확인할 수 없습니다: $SAM2_SRC — 별도 경로에 setup.sh로 준비하거나 올바른 Git 체크아웃을 지정하세요"
  fi
else
  log "SAM 2 소스를 받습니다: $SAM2_SRC (commit $SAM2_REVISION)"
  sam2_tmp="$SAM2_SRC.setup-tmp.$$"
  rm -rf "$sam2_tmp"
  if ! sam2_checkout "$sam2_tmp"; then
    rm -rf "$sam2_tmp"
    die "SAM 2 소스를 받지 못했습니다(네트워크/프록시 확인): $SAM2_REPO_URL"
  fi
  mv "$sam2_tmp" "$SAM2_SRC"
  log "SAM 2 소스 준비 완료: $SAM2_SRC"
fi

# ------------------------------------------------------------------ model prerequisites (informational)
missing=()
check_model() { # check_model <label> <path>
  if [[ -e $2 ]]; then
    log "  [있음] $1: $2"
  else
    log "  [없음] $1: $2"
    missing+=("$1: $2")
  fi
}

log "== 모델 사전요건 (이 스크립트는 내려받지 않습니다 — MODELS.md의 절차로 준비하세요) =="
check_model "Clef 릴리스 스냅샷" "${CLEF_MODEL:-$MODEL_ROOT/clef}"
check_model "Clef c 어댑터" "${CLEF_ADAPTER:-$MODEL_ROOT/clef-ft-20261004c}"
check_model "SAM 2.1 체크포인트" "${TRACKER_SAM2_CKPT:-$MODEL_ROOT/sam2.1_hiera_base_plus.pt}"
check_model "HF 캐시(그라운더 가중치)" "${TRACKER_HF_HOME:-$MODEL_ROOT/hf}"

if (( ${#missing[@]} > 0 )); then
  log "아직 없는 모델 ${#missing[@]}건 — 준비하기 전까지 해당 서비스는 실행 시 명확히 실패합니다(대체 mock 없음)"
fi

cat <<EOF
완료. 다음 순서:
  1) MODELS.md에 따라 위 모델 가중치·어댑터를 준비합니다(라이선스·gated 접근은 운영자 본인 처리).
     SAM 2.1 체크포인트(공식 배포 URL, 약 308.6 MiB, sha256 a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5
     — tools/tracker_service/models.py가 시작할 때 이 해시를 검사합니다):
       curl -fL --retry 3 -o "$MODEL_ROOT/sam2.1_hiera_base_plus.pt" \\
         https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
       sha256sum "$MODEL_ROOT/sam2.1_hiera_base_plus.pt"
  2) .env.local-gpu의 MODEL_ROOT와 GPU 선택(${CLEF_CUDA_VISIBLE_DEVICES:-0})을 확인합니다.
  3) 터미널마다 하나씩 실행합니다(모두 포그라운드, 루프백 전용):
       bash deploy/local-gpu/run.sh tracker
       bash deploy/local-gpu/run.sh clef
       bash deploy/local-gpu/run.sh backend
       bash deploy/local-gpu/run.sh live
  4) 상태 확인: bash deploy/local-gpu/run.sh check
EOF
