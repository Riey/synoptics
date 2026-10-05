# Synoptics — 영상 기반 작업 안내 AI Agent

팀 Idiopathic · 제4회 경남 AI·SW 경진대회 · 소스 스냅샷 2026-10-05 / 로컬 GPU 실행 구성 2026-10-06

매뉴얼·작업 영상 비교 기반 조립·포장 교육 Agent입니다. 목표·매뉴얼 입력 → 절차 계획·질문 → 검토·승인 → 카메라 관찰 → 다음 행동 안내 → 결과 확인·피드백 흐름으로 동작합니다.

이 문서는 **독립 실행형 Linux + NVIDIA GPU 서버 한 대**에서 이 스냅숏을 `clone → setup → models → start → open` 순서로 재현하는 절차입니다. 팀 내부 개발 호스트·원격 볼륨·원격 배포 도구 없이 동작하도록 작성했습니다.

## 1. 이 저장소는 무엇인가

개발 저장소와 분리한 **독립 Git 저장소의 단일 초기 커밋**입니다. 원래 Git 이력·remote·worktree 연결을 포함하지 않습니다. 공개 저장소 [github.com/Riey/synoptics](https://github.com/Riey/synoptics) 로 제공합니다.
- `source-manifest.json`은 구성요소 출처와 파일별 SHA-256을 기록한 원장이며 Git 개발 이력 자체가 아닙니다.
- 구성:
  - `backend/`, `web/`, `packages/visual-tools/`: 계획·매뉴얼·질문·그래프 계약과 **참조 React 앱**.
  - `live-api/`: 실제 REST/WebSocket 서버와 서버 내장 진단 클라이언트(`/demo/`).
  - `tools/tracker_service/`: SAM 3 대상 획득 + SAM 2.1 연속 추적 서비스.
  - `tools/clef_ft/`: 공개 Cloudflare/Clef + 팀 c 어댑터의 실행·병합·학습 도구(어댑터 가중치는 별도 ZIP).
  - `api-spec/live-api.md`: 함께 제공한 Live 소스와 일치하는 API 계약.
  - `deploy/local-gpu/`: 이 문서가 쓰는 로컬 설치·실행 자산 — `setup.sh`, `run.sh`, `.env.example`, 그리고 setup.sh가 설치하는 검토 가능한 의존성 집합 `requirements-clef.txt`·`requirements-tracker.txt`.
  - `verification/`: 스냅숏 조립 시점의 고정 응답 연동 스모크와 회귀·빌드 기록.
- **무엇이 아닌지**: 이 저장소는 운영 최신 배포가 아니며 인식 정확도·교육 효과를 주장하지 않습니다. `web/`은 참조 React 앱이고, 협업용 외부 PWA(운영 클라이언트)의 소스는 이 묶음에 없습니다. `live-api`의 `/demo/`는 서버에 내장된 진단 클라이언트입니다. 서버 변경이 외부 클라이언트에 자동 반영됐다고 주장하지 않습니다.

## 2. 요구 사항

| 항목 | 요구 |
|---|---|
| OS | Linux x86_64 |
| GPU | NVIDIA + CUDA 12.8 호환 드라이버(3절의 메모리 조건 포함) |
| Python | 3.12 + [uv](https://docs.astral.sh/uv/) |
| Node | Node.js 24+ / npm |
| 기타 | `git`, `gcc`(`torch.compile`·flash-linear-attention 빌드용). `ffmpeg`는 서버 음성 경로를 켤 때만 필요하며 기본 TTS에서는 불필요합니다 |

## 3. GPU 요구 — 현실적인 크기

**"아무 GPU나 된다"는 주장이 아닙니다.** 두 모델 갈래의 요구가 다릅니다.

- **Clef 단계 판정 서버(BF16 27B)**: 가중치만 약 **52 GiB**이고 여기에 활성값·KV 캐시가 더해집니다. 기본값 `CLEF_DEVICE=split`은 보이는 GPU 전체에 나눠 적재합니다.
  - 기본 목표: **80 GiB GPU 한 장**(`CLEF_MAX_MEMORY=72GiB`) 또는 메모리가 충분한 여러 장(예: `CLEF_CUDA_VISIBLE_DEVICES=0,1`, `CLEF_MAX_MEMORY=28GiB,28GiB`).
  - 작은 단일 GPU(예: 24 GiB)로는 **BF16을 적재할 수 없습니다.** `CLEF_QUANT=fp8dyn`(약 28.5 GiB)은 별도·선택 경로이며 torchao가 필요합니다(4·5절).
- **추적기(SAM 3 획득 + SAM 2.1 `hiera_base_plus`)**: 24 GiB 카드 한 장에 두 모델이 함께 들어갑니다. 기본값은 서비스가 보이는 장치를 0부터 재번호한 뒤 두 모델 모두 `cuda:0`에 둡니다.
- 두 서비스는 같은 GPU를 공유할 수 있지만, 메모리가 빠듯하면 Clef와 추적기를 다른 장에 나눠 배치하십시오(`CLEF_CUDA_VISIBLE_DEVICES` / `TRACKER_CUDA_VISIBLE_DEVICES`).
- 모델 파일은 저장소 밖 `MODEL_ROOT`에 두며 **수십 GB의 디스크**가 필요합니다.
- 상위 계획·확인용 **외부 API(DeepSeek 등)와 로컬 GPU는 별개**입니다. GPU가 있어도 클라우드 API가 오프라인이 되지 않습니다(8절).

## 4. 설치 — `deploy/local-gpu/setup.sh`

```bash
git clone https://github.com/Riey/synoptics.git synoptics && cd synoptics

# CPU 절반만 (GPU 라이브러리·SAM 2 소스 없음)
bash deploy/local-gpu/setup.sh --cpu-only

# 전체 (권장): CPU 절반 + GPU 런타임 + SAM 2 소스 체크아웃
bash deploy/local-gpu/setup.sh
```

- 전체 설치 내용: 루트 `.venv`(backend), `live-api/.venv`(Live), npm 의존성 + `web/dist` 빌드, `.venv-clef`·`.venv-tracker`(Python 3.12), `$MODEL_ROOT/sam2-src`에 고정 커밋의 SAM 2 소스 체크아웃.
- `--cpu-only`: 위 중 앞의 CPU 부분만 설치하며 GPU 환경·SAM 2 소스를 만들지 않습니다.
- 두 스크립트 모두 **어느 cwd에서든** 실행할 수 있습니다(스크립트 위치로 저장소 루트를 찾습니다).
- setup.sh는 `deploy/local-gpu/.env.example`을 저장소 루트 `.env.local-gpu`(git-ignored)로 **복사만** 하며 기존 파일을 덮지 않습니다. 값을 편집할 때는 템플릿이 아니라 이 복사본을 고치십시오.
- **setup.sh는 모델 가중치를 내려받지 않고, 라이선스에 동의하지 않고, gated 모델을 받지 않습니다.** PyPI/npm/GitHub 소스만 받습니다. 가중치는 5절로 직접 준비합니다.
- 비 `--cpu-only` 실행은 `.env.local-gpu`의 `MODEL_ROOT`가 비어 있으면 실패합니다. 먼저 경로를 채우십시오.

## 5. 모델 준비 (요약 — 전체는 `MODELS.md`)

`MODEL_ROOT` 레이아웃:

아래 모델 준비 명령은 저장소 루트에서 실행합니다. 설정 파일 편집만으로는 현재 셸에 변수가 생기지 않으므로 먼저 불러옵니다.

```bash
set -a
. ./.env.local-gpu
set +a
mkdir -p "$MODEL_ROOT"
```

```
$MODEL_ROOT/clef/                      공개 Cloudflare/Clef 릴리스 스냅숏(설정·토크나이저·joint_head·backbone)
$MODEL_ROOT/clef-ft-20261004c/         팀 c 어댑터(adapter/, head.safetensors, joint_head_config.json)
$MODEL_ROOT/sam2-src/                  facebookresearch/sam2 @ 2b90b9f5… 체크아웃(setup.sh가 생성)
$MODEL_ROOT/sam2.1_hiera_base_plus.pt  SAM 2.1 체크포인트(sha256 a2345aed…)
$MODEL_ROOT/hf/                        그라운더 가중치 HF 캐시(SAM 3 또는 GroundingDINO)
```
- **공개 백본**: `HF_HUB_OFFLINE=0 .venv-clef/bin/hf download Cloudflare/clef --local-dir "$MODEL_ROOT/clef"`
- **팀 c 어댑터**: 팀 제출 폴더의 `clef-ft-20261004c.zip`(sha256 `f2f7e4fb…`)을 받아 `sha256sum -c`로 검증한 뒤 `$MODEL_ROOT/clef-ft-20261004c/`로 해제합니다.
- **SAM3**: Hugging Face에서 gated(수동 승인)입니다. 사용자가 직접 약관에 동의하고 제공자의 접근 승인을 받아야 하며 이 저장소가 대신 동의하지 않습니다. 승인 계정으로 `HF_HOME="$MODEL_ROOT/hf" HF_HUB_OFFLINE=0 .venv-tracker/bin/hf auth login`한 뒤, `HF_HOME="$MODEL_ROOT/hf" HF_HUB_OFFLINE=0 .venv-tracker/bin/hf download facebook/sam3 --revision 3c879f39826c281e95690f02c7821c4de09afae7`을 실행합니다.
- SAM 2.1 체크포인트·소스, CUDA 의존성, API 키는 저장소에 없습니다. **가중치 파일을 저장소에 커밋하지 마십시오.**

## 6. 실행 — `deploy/local-gpu/run.sh`

터미널마다 하나씩, 전부 **포그라운드**로 실행합니다(Ctrl-C로 종료). 순서: 로컬 GPU 서비스 → 앱.

```bash
# 1) 추적기          (GPU, 127.0.0.1:8090)
bash deploy/local-gpu/run.sh tracker
# 2) Clef 판정 서버  (GPU, 127.0.0.1:8085)
bash deploy/local-gpu/run.sh clef
# 3) 참조 백엔드     (CPU, 127.0.0.1:8040)
bash deploy/local-gpu/run.sh backend
# 4) Live 서버       (CPU, 127.0.0.1:8104)
bash deploy/local-gpu/run.sh live
# 준비 상태 확인
bash deploy/local-gpu/run.sh check
```

`run.sh`는 `.env.local-gpu`를 소싱하고 모델 경로를 `MODEL_ROOT`에서 파생합니다(환경 파일에 명시하면 그 값이 우선). `CLEF_CUDA_VISIBLE_DEVICES`/`TRACKER_CUDA_VISIBLE_DEVICES`는 **그 프로세스에만** `CUDA_VISIBLE_DEVICES`로 전달됩니다.

**준비 상태·실패 신호** — `run.sh check`가 4개 라우트를 확인합니다:

| 서비스 | 라우트 | 기준 |
|---|---|---|
| tracker | `http://127.0.0.1:8090/health` | `ready=true`. `ready=false`면 `error`가 그대로 노출됩니다 |
| clef | `http://127.0.0.1:8085/health` | `ok=true` |
| backend | `http://127.0.0.1:8040/api/health` | `ready=true`. 키가 없는 기본 예제는 미준비로 표시하고 실패 종료합니다 |
| live | `http://127.0.0.1:8104/v1/health` | `ready=true`, `engine=real`. 브라우저 TTS이므로 `tts_ready=false`는 정상입니다 |

`check`는 네 서비스가 자기 형식의 health 응답을 내고 모두 준비된 경우에만 종료 코드 0을 반환합니다. 연결 실패·다른 JSON·미준비 응답은 종료 코드 1이며, 키·추적기·Clef 연결 중 빠진 항목을 표시합니다. GPU 모델과 키가 없는 CPU-only 구성에서는 실패하는 것이 정상입니다. 루프백 검사는 환경 프록시를 우회하며 **모델 추론이나 유료 API 호출은 하지 않습니다**.

**실행 단계의 명확한 실패**(다운로드 대신 오류로 알려줍니다):

- `.env.local-gpu` 없음 → 먼저 `setup.sh`.
- `MODEL_ROOT`가 비었거나 절대 경로가 아님 → `.env.local-gpu`에서 채움.
- SAM 2 소스 또는 체크포인트 없음 → 5절.
- `.env.local-gpu`의 `TRACKER_GROUNDER`(기본 `gdino`, 파일에서는 `sam3`)에 해당하는 그라운더 HF 캐시 디렉터리가 없음(`$MODEL_ROOT/hf/hub/models--facebook--sam3` 또는 `models--IDEA-Research--grounding-dino-tiny`) → 5절. `HF_HUB_OFFLINE=1`이면 gated SAM 3는 다운로드를 시도하지 않고 명확히 실패합니다.
- `CLEF_QUANT=fp8dyn`인데 torchao 없음 → `uv pip install --python .venv-clef/bin/python torchao==0.14.1`(setup.sh는 torchao를 설치하지 않습니다).
- `.venv-clef`/`.venv-tracker` 없음 → `setup.sh`. 다른 Python 판본으로 만들어진 잘못된 환경도 재설치 대신 거부합니다.

현행 구성을 재현하려면 **`TRACKER_GROUNDER=sam3`를 명시**해야 합니다(미지정 시 소스 기본값은 `gdino`). `.env.local-gpu`가 이미 `sam3`으로 설정합니다.

## 7. 접속

- **참조 React 앱**: 백엔드가 `web/dist`를 서빙 → <http://127.0.0.1:8040/>
- **Live 진단 클라이언트**: <http://127.0.0.1:8104/demo/>

**카메라**는 브라우저가 보안 컨텍스트로 취급하는 origin에서만 열립니다. `http://127.0.0.1`·`http://localhost`는 보안 컨텍스트이므로 로컬 브라우저에서는 그대로 동작하며, 카메라 권한을 허용해야 프레임이 흐릅니다.

**원격 서버 화면을 내 PC 브라우저에서** 보려면 SSH 터널로 루프백을 그대로 끌어옵니다. 직접 터널은 포워딩 헤더가 없어 `AISW_LOCAL_ONLY=1`에서도 그대로 서빙됩니다:

```bash
ssh -N -L 8040:127.0.0.1:8040 -L 8104:127.0.0.1:8104 user@gpu-server
# 내 PC 브라우저에서 http://127.0.0.1:8040/ , http://127.0.0.1:8104/demo/
```

터널 없이 원격 IP/프록시로 직접 노출하면 `AISW_LOCAL_ONLY=1`이 비루프백 트래픽을 거부합니다. 외부에 공개해야 한다면 HTTPS·접근 코드(`DEMO_ACCESS_CODE`, `LIVE_ACCESS_CODE`)·CORS origin·접근 제어를 갖춘 뒤 로컬 전용 모드를 해제하십시오. 예제 설정은 그런 공개 구성이 아니며, 코드 없는 인스턴스를 공개 릴레이 뒤에 두지 마십시오.

## 8. API 키·유료 호출

- 키는 **파일로** 지정합니다: `DEEPSEEK_API_KEY_FILE`, `OPENAI_API_KEY_FILE`, `GEMINI_API_KEY_FILE`. 예제는 `/dev/null`이라 계획용 키가 없는 상태가 정상이고, 이때 백엔드 `ready=false`가 나옵니다.
- 상위 계획·완료 확인·재계획은 **외부 API**로 나가며 **사용자 키와 과금**이 필요합니다. 이 호출은 사용자의 동의 아래에서만 발생합니다.
- **로컬 GPU 서비스(추적기·Clef)는 키 없이 동작합니다.** GPU가 있다고 해서 클라우드 계획이 오프라인이 되지는 않습니다.
- 예제 설정을 소싱하기 전에 셸에 남아 있는 `*_API_KEY` 환경변수가 있는지 확인하십시오.

## 9. 음성 (TTS)

- **제출 기본은 브라우저·기기 TTS(Web Speech API)** 입니다. `AISW_TTS_URL`과 `LIVE_TTS_URL`을 **비워 둡니다.** 그러면 서버 음성 경로가 꺼지고 브라우저가 읽습니다.
- 내부 **OmniVoice 서버·가중치(CC-BY-NC)는 이 제출물에 포함하지 않습니다.** 서버 음성은 별도 구현을 직접 연결한 경우에만 동작합니다.

## 10. 확인·검증의 범위

- **`run.sh check`** 는 위 4개 라우트의 도달·준비 상태만 확인하며 추론을 하지 않습니다.
- `verification/` 아래 기록(고정 응답 연동 스모크, 회귀·빌드 로그)은 **이 스냅숏을 조립한 시점(2026-10-05)의 산출물 기록**입니다. 저장소를 패키징하는 시점에 소유자가 최종 검증을 한 번 수행하며, 그 결과가 최종 상태입니다. 이 기록을 실모델 정확도 시험이나 최신 배포 검증으로 해석하지 마십시오.
- 회귀·스모크 통과와 **실제 모델 연동 준비는 서로 다른 조건**입니다. 실제 인식 품질은 라벨·영상에 따라 별도로 측정해야 하며, 이 문서는 어떤 정확도도 주장하지 않습니다.

## 11. 포함하지 않는 것

비공개 원본 영상·학습 데이터·API 키·내부 배포 설정·모델 가중치는 저장소에 없습니다. `tools/clef_ft/`에 학습 코드가 있어도 비공개 데이터 없이 동일 가중치가 재현되지는 않습니다. 생성 모델의 라벨은 사람 정답과 구분합니다.

## 12. 출처·라이선스

모델·런타임·외부 API의 출처·판본·라이선스·활용 범위는 `MODELS.md`와 제출 문서 「04 출처·AI 활용 신고서」를 확인하십시오. 자체 소스에 임의 라이선스를 부여하지 않았으며, 포함한 제3자 저작물(예: `tools/clef_ft/vendor/LICENSE`)에는 각 원래 조건이 적용됩니다.
