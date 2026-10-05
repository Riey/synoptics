# 모델·가중치 출처와 준비 절차

2026-10-05 기준. 자체 학습 가중치는 공개 배포 대신 팀 제출 폴더를 통해 제공합니다. 공개 사전학습 모델과 외부 API 모델은 출처·판본·라이선스·활용 범위를 신고합니다. 이 문서는 **독립 실행형 Linux + NVIDIA 서버**에서 `deploy/local-gpu/setup.sh`로 설치한 뒤 가중치를 어디에 어떻게 준비하는지를 설명하며, 임의 환경에서의 실행 성공·성능을 보증하지 않습니다. 이 문서의 정리를 위해 GPU 모델을 다시 적재하거나 추론을 재실행하지 않았습니다.

모든 경로는 저장소 밖 `MODEL_ROOT`(절대 경로, `.env.local-gpu`에서 지정) 아래를 기준으로 합니다. `deploy/local-gpu/run.sh`가 이 경로들을 `MODEL_ROOT`에서 파생하므로, 준비만 이 레이아웃대로 하면 별도 지정이 필요 없습니다. 파일에 직접 값을 적으면 파생값보다 우선합니다.

명령은 저장소 루트에서 실행합니다. 먼저 `set -a; . ./.env.local-gpu; set +a`로 `MODEL_ROOT`를 불러오고 `mkdir -p "$MODEL_ROOT"`로 외부 모델 폴더를 만드십시오. 다운로드할 때만 `HF_HUB_OFFLINE=0`을 명시하며 서비스 실행은 오프라인 캐시를 사용합니다.

```
$MODEL_ROOT/clef/                      공개 Cloudflare/Clef 릴리스 스냅숏(config·tokenizer·processor,
                                       joint_head.safetensors, joint_head_config.json, backbone *.safetensors)
$MODEL_ROOT/clef-ft-20261004c/         팀 c 어댑터(adapter/, head.safetensors, joint_head_config.json)
$MODEL_ROOT/sam2-src/                  facebookresearch/sam2 체크아웃 커밋 2b90b9f5…(저장소 루트)
$MODEL_ROOT/sam2.1_hiera_base_plus.pt  SAM 2.1 체크포인트, sha256 a2345aed…
$MODEL_ROOT/hf/                        그라운더 가중치를 담은 HF 캐시(SAM 3 또는 GroundingDINO)
```

`setup.sh`는 의존성만 설치하며 **가중치를 내려받지 않고 라이선스에 동의하지 않습니다**(`--cpu-only`면 GPU 런타임과 SAM 2 소스도 만들지 않습니다). `setup.sh`가 만드는 것은 `$MODEL_ROOT/sam2-src`뿐이고, 나머지 가중치는 아래 절차로 직접 준비합니다.

## 1. 자체 미세조정분 — clef-ft-20261004c

**공개 Clef 모델 전체를 자체 개발한 것이 아닙니다.** 공개 [Cloudflare/Clef](https://huggingface.co/Cloudflare/clef)(27B, `Qwen/Qwen3.8-27B` 기반)에 팀의 작업 영상·라벨을 이용해 LoRA와 joint decision head를 추가 미세조정했습니다. 공개 모델의 학습 이력과 팀의 추가 학습 범위는 별개입니다.

| 항목 | 제공 내용 |
|---|---|
| 검증 파일 | `clef-ft-20261004c.zip` + `.zip.sha256` |
| 접근 위치 | [팀 최종제출물 폴더](https://drive.google.com/drive/folders/1n6IEsJiVzBetiIEtY4einrg8_cleyn5a)에서 같은 이름의 파일 다운로드 |
| 무결성 | ZIP의 SHA-256은 `.zip.sha256`(값 `f2f7e4fba330f286098811f08f6ef6f3cf7835b546fb03c525d52732999a5769`), 원본 파일별 크기·SHA-256은 압축 내부 `manifest.json`에 기록 |
| 포함 | `adapter/adapter_config.json`, `adapter/adapter_model.safetensors`, `head.safetensors`, `joint_head_config.json`, `LICENSE`, `NOTICE.txt`, `manifest.json` |
| 제외 | 공개 백본 전체, 원본 카메라 영상, 학습 데이터, API 키, 다른 모델 가중치 |
| 원본 식별 | 현재 어댑터 `clef-ft-20261004c`; 운영 설정은 c 병합 체크포인트·FP8 동적 적재. 제공 어댑터의 BF16 재현과 양자화 출력이 수치적으로 동일하다는 뜻은 아님 |
| 권한 | 팀 공유폴더이며 공개 링크가 아님. 심사위원이 다운로드할 수 있도록 제출 시 해당 계정에 별도 권한 제공 필요 |

`manifest.json`에는 각 원본 파일의 크기·SHA-256이 있습니다. 공개 Clef 배포물의 Apache-2.0 `LICENSE`를 동봉했고 팀 수정 범위는 `NOTICE.txt`에 기록했습니다. 전체 백본 대신 이 어댑터와 판정 head를 제공하므로, **공개 백본은 별도로 준비**해야 합니다.

## 2. Clef 실행 경로

GPU·CUDA 및 큰 모델 실행 환경이 필요한 경로입니다. 앱만 실행하는 Python 환경과 분리하십시오(`deploy/local-gpu/setup.sh`가 `.venv-clef`를 따로 만듭니다). 다운로드·의존성 판본의 참고는 `tools/clef_ft/run_on_instance.sh`의 `setup`/`fetch` 정의입니다(이 스크립트에는 원격 학습 인스턴스 관리도 있으므로 내용을 확인하지 않고 전체 파이프라인을 실행하지 마십시오).

```bash
# 1) 어댑터 ZIP과 .zip.sha256을 MODEL_ROOT에 받은 후 검증·해제
(cd "$MODEL_ROOT" && sha256sum -c clef-ft-20261004c.zip.sha256)
unzip -q "$MODEL_ROOT/clef-ft-20261004c.zip" -d "$MODEL_ROOT"
#   -> $MODEL_ROOT/clef-ft-20261004c/adapter/... , head.safetensors, joint_head_config.json

# 2) 공개 백본은 제공자 원본에서 준비
HF_HUB_OFFLINE=0 .venv-clef/bin/hf download Cloudflare/clef --local-dir "$MODEL_ROOT/clef"

# 3) .env.local-gpu에 MODEL_ROOT(절대 경로)만 채우면 run.sh가 나머지 경로를 파생한다
#    $MODEL_ROOT/clef , $MODEL_ROOT/clef-ft-20261004c
bash deploy/local-gpu/run.sh clef      # 127.0.0.1:8085, 포그라운드
```

- 기본값은 `CLEF_DEVICE=split`(보이는 GPU 전체에 분산 적재)이고 `.env.local-gpu`가 그 값을 설정합니다. **BF16 27B는 가중치만 약 52 GiB**이므로 작은 단일 GPU로는 적재할 수 없습니다. `CLEF_CUDA_VISIBLE_DEVICES`와 `CLEF_MAX_MEMORY`(예: 80 GiB 한 장 `72GiB`, 또는 `0,1` + `28GiB,28GiB`)로 장수·상한을 맞추십시오. 실행 시 가시 장치 수·인덱스는 `run.sh`가 검사합니다.
- `CLEF_ADAPTER`를 비우면 c 어댑터 없이 릴리스 head만으로 서빙합니다(`CLEF_ADAPTER= bash deploy/local-gpu/run.sh clef`).
- 운영 중인 다른 서비스와 같은 장치에서 무단으로 실행하지 마십시오.

**선택적 병합·FP8 레시피(운영 적재 방식)**: `tools/clef_ft/merge_ckpt.py`로 어댑터를 백본에 미리 병합한 뒤 로드 시 양자화할 수 있습니다.

```bash
# .venv-clef의 Python으로 실행합니다(어댑터 로드에 torch 필요)
.venv-clef/bin/python tools/clef_ft/merge_ckpt.py --base "$MODEL_ROOT/clef" \
  --adapter "$MODEL_ROOT/clef-ft-20261004c" --out "$MODEL_ROOT/clef-merged-20261004c"
# .env.local-gpu에서 (주석 해제):
#   CLEF_WEIGHTS=$MODEL_ROOT/clef-merged-20261004c
#   CLEF_QUANT=fp8dyn        # torchao==0.14.1을 .venv-clef에 설치해야 함
```

`merge_ckpt.py`는 샤드 단위로 LoRA를 병합해 출력 디렉터리가 그대로 `CLEF_WEIGHTS`가 되게 하며, 어댑터가 head를 계속 공급합니다. `CLEF_QUANT`를 비워두면 BF16이고, `fp8dyn`은 언어모델 Linear를 float8로 바꿉니다(약 28.5 GiB). 이는 **적재 방식 선택지**이며 이 문서는 그 성능·속도를 주장하지 않습니다.

앱 프로세스는 다음을 설정합니다(기본 로컬 추종 경로가 자동으로 Clef를 선택하는 것은 아닙니다). `.env.local-gpu`가 이미 설정합니다.

```bash
export AISW_FOLLOW_PROVIDER=clef
export AISW_FOLLOW_CLEF_URL=http://127.0.0.1:8085/v1/systemone
export AISW_FOLLOW_CLEF_MODE=choice
export AISW_FOLLOW_CLEF_YES_MIN=0.7
```

서비스의 `GET /health`와 앱의 `GET /api/health`로 연결 상태를 확인하십시오(`bash deploy/local-gpu/run.sh check`가 이 둘을 포함해 확인합니다). API 키는 파일/환경변수로 직접 공급하며 제출물에는 포함하지 않습니다. 학습 재현 코드는 `tools/clef_ft/`에 있지만 학습 데이터가 별도이므로 **소스만으로 동일 가중치를 다시 학습할 수 있다는 의미는 아닙니다**.

## 3. 추적기 — SAM 3 획득 + SAM 2.1 추적

위치는 추적기가, 의미는 상위/로컬 모델이 담당합니다. 추적기는 별도 로컬 GPU 서비스(`tools/tracker_service/`)이고, 앱은 프레임을 전달만 하며 모델을 직접 돌리지 않습니다.

**역할 구분**: **SAM 3는 대상 획득/재획득(acquisition)에만** 쓰고, 프레임별 추적은 **SAM 2.1 `hiera_base_plus`**가 맡습니다. 소스 기본값은 GroundingDINO 획득이므로, 현행 구성을 재현하려면 **`TRACKER_GROUNDER=sam3`를 명시**해야 합니다(미지정 시 `gdino`). `.env.local-gpu`가 이미 `sam3`으로 설정합니다.

`tools/tracker_service/models.py`가 고정하는 값(소스에 박힌 상수):

| 항목 | 값 |
|---|---|
| 획득 모델 | `facebook/sam3`, revision `3c879f39826c281e95690f02c7821c4de09afae7`, 체크포인트 SHA-256 `6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a` |
| 추적 모델 | SAM 2.1 `hiera_base_plus`, 소스 커밋 `2b90b9f5ceec907a1c18123530e92e794ad901a4`, 설정 `configs/sam2.1/sam2.1_hiera_b+.yaml`, 체크포인트 SHA-256 `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5` |
| 대체/기본 획득 | GroundingDINO-tiny `IDEA-Research/grounding-dino-tiny`, revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, 체크포인트 SHA-256 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3` |
| 선택 | `TRACKER_GROUNDER`(기본 `gdino`, 대안 `sam3`), `TRACKER_GROUNDER_DEVICE`, `TRACKER_SAM2_DEVICE` |

**가중치 준비 조건**

- **SAM 3는 Hugging Face에서 gated(수동 승인)** 입니다. 사용자가 직접 약관에 동의하고 제공자의 접근 승인을 받아야 합니다. 이 제출 작업은 약관에 대신 동의하거나 연락처를 제출하지 않았습니다.
- 승인된 계정으로 `HF_HOME="$MODEL_ROOT/hf" HF_HUB_OFFLINE=0 .venv-tracker/bin/hf auth login`을 먼저 실행합니다. 이후 **HF 캐시 레이아웃**으로 미리 받습니다: `HF_HOME="$MODEL_ROOT/hf" HF_HUB_OFFLINE=0 .venv-tracker/bin/hf download facebook/sam3 --revision 3c879f39826c281e95690f02c7821c4de09afae7`. 서비스는 `$HF_HOME/hub/models--facebook--sam3` 캐시를 사용합니다.
- `TRACKER_GROUNDER=gdino`라면 같은 HF_HOME으로 `HF_HOME="$MODEL_ROOT/hf" HF_HUB_OFFLINE=0 .venv-tracker/bin/hf download IDEA-Research/grounding-dino-tiny --revision a2bb814dd30d776dcf7e30523b00659f4f141c71`을 실행합니다. `HF_HUB_OFFLINE=1`인 요청 시점에는 오프라인 캐시만 사용하며, 캐시에 없으면 다운로드하지 않고 실패합니다.
  `run.sh tracker`는 이미 설정된 `HF_HOME`을 그대로 쓰고, 없으면 `TRACKER_HF_HOME`, 그것도 없으면 `$MODEL_ROOT/hf`을 사용하므로 위 `HF_HOME=…` 다운로드 위치와 정확히 일치합니다.
- SAM 2.1 **소스**는 `setup.sh`가 커밋 `2b90b9f5…`로 `$MODEL_ROOT/sam2-src`에 체크아웃합니다(이 커밋에 `sam2/build_sam.py`와 `sam2/configs/sam2.1/sam2.1_hiera_b+.yaml`이 있습니다). 체크포인트 `sam2.1_hiera_base_plus.pt`는 위 해시로 `$MODEL_ROOT`에 준비합니다. SAM 3 경로를 쓸 때 GroundingDINO 가중치는 필요하지 않습니다.
- 공개 벤치마크나 성능 수치는 이 문서에서 주장하지 않으며, 리플레이 하네스(`tools/tracker_service/replay_clip.py`, `REPLAY.md`)는 측정 도구이고 데모 경로가 아닙니다.

**SAM 2.1 체크포인트 다운로드** — 저장소 루트에서 설정을 불러온 상태로 실행합니다. 아래 파일은 공개 SAM 2.1 체크포인트이며, SAM 3 접근 승인을 대신하지 않습니다.

```bash
curl -fL --retry 3 -o "$MODEL_ROOT/sam2.1_hiera_base_plus.pt" \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
printf '%s  %s\n' \
  a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5 \
  "$MODEL_ROOT/sam2.1_hiera_base_plus.pt" | sha256sum -c -
```

실행(`.venv-tracker`와 SAM 2 소스 의존성이 설치된 환경, `run.sh`가 경로·PYTHONPATH·`CUDA_VISIBLE_DEVICES`를 설정):

```bash
# .env.local-gpu에 MODEL_ROOT만 채우면 아래 경로가 파생된다
#   TRACKER_SAM2_SRC=$MODEL_ROOT/sam2-src
#   TRACKER_SAM2_CKPT=$MODEL_ROOT/sam2.1_hiera_base_plus.pt
#   HF_HOME=$MODEL_ROOT/hf  (그라운더 가중치 캐시)
bash deploy/local-gpu/run.sh tracker      # 127.0.0.1:8090, 포그라운드
# 앱 쪽: AISW_TRACKER_URL=http://127.0.0.1:8090 (예제에 설정됨)
```

기본 배치는 두 모델 모두 `cuda:0`(서비스가 보이는 장치를 0부터 재번호)이고 24 GiB 카드 한 장에 들어갑니다. `TRACKER_GROUNDER_DEVICE`/`TRACKER_SAM2_DEVICE`로 나눌 수 있습니다. `deploy/tracker/run.sh`는 팀의 원격 Docker 볼륨(`/v`)이 이미 준비된 환경용 **원격 관리 도구**이므로 로컬 실행법으로 사용하지 마십시오. 로컬 독립 실행은 위 `deploy/local-gpu/run.sh tracker`입니다.

## 4. 공개 모델 · 런타임

| 모델·런타임 | 버전/출처 | 라이선스 | 범위 |
|---|---|---|---|
| Cloudflare/Clef | [모델 카드](https://huggingface.co/Cloudflare/clef), 27B, `Qwen/Qwen3.8-27B`에서 추가 학습된 공개 결정 모델 | Apache-2.0 | 팀 미세조정의 기반; 백본·기본 판정 head·처리 코드 |
| SAM 3 | `facebook/sam3`, gated | Meta SAM License(비독점·무상, 상업적 이용 허용, 배포 시 동의문 동봉·Meta 출처 표시, 특정 용도 금지) | 대상 획득(acquisition) |
| SAM 2.1 | `facebook/sam2`, `hiera_base_plus` | Apache-2.0 | 대상 추적 |
| GroundingDINO-tiny | `IDEA-Research/grounding-dino-tiny` | Apache-2.0 | 소스 기본·대체 획득 경로 |
| Qwen3.6-35B-A3B · Qwen3.5-4B | Qwen, UD-Q4_K_M / Q4_K_M | Apache-2.0 | **코드 대안 추종 경로(현재 시스템의 추종 모델로 주장하지 않음)** |
| llama.cpp | ggml-org (`server-cuda-b11146`) | MIT | 위 코드 대안 경로의 로컬 추론 서버. Clef의 `/v1/systemone` 서버와는 별도 구현 |

공개 Clef의 모델 카드와 LICENSE: https://huggingface.co/Cloudflare/clef/blob/main/README.md · https://huggingface.co/Cloudflare/clef/blob/main/LICENSE . 백본 이름(Qwen3.8)과 로딩 아키텍처 클래스(`Qwen3_5ForConditionalGeneration`)는 다른 종류의 식별자입니다.

## 5. 외부 API

| 모델 | 역할 | 조건 |
|---|---|---|
| DeepSeek `deepseek-flash` | 기본 계획·완료 확인·재계획 | 제공자 API 약관·과금, 사용자 키 필요 |
| OpenAI `gpt-6-astra` | 선택적 상위 계획·확인, Responses API/high | 제공자 API 약관·과금, 사용자 키 필요 |
| Google `gemini-3.8-flash` | 대체 프로바이더 | 제공자 API 약관·과금, 사용자 키 필요 |

외부 API의 가중치는 배포하지 않습니다. 키는 파일로 공급합니다(`DEEPSEEK_API_KEY_FILE`, `OPENAI_API_KEY_FILE`, `GEMINI_API_KEY_FILE`; 예제는 `/dev/null`). 개발 도구·오픈소스·데이터 출처는 별도 「04 출처·AI 활용 신고서」와 함께 확인하십시오.

## 6. 제출 범위

- 소스 ZIP과 미세조정 가중치 ZIP은 별개 파일입니다. 두 파일과 각 `.sha256`을 함께 확보하십시오. 어댑터 ZIP의 SHA-256은 `f2f7e4fb…`입니다.
- 공개 백본·추적기 체크포인트·CUDA 환경·API 키는 **소스 ZIP에 없습니다.** 회귀 테스트 통과와 실제 모델 연동 준비는 서로 다른 조건입니다. `deploy/local-gpu/setup.sh`는 의존성만 설치하고 가중치·라이선스는 다루지 않습니다.
- 음성 출력은 **브라우저·기기 TTS(Web Speech API)**를 기준으로 합니다. 서버 음성 경로는 `AISW_TTS_URL`이 설정된 경우에만 동작하며 기본 비활성입니다. 내부 OmniVoice 서버·가중치(CC-BY-NC)는 제출하지 않습니다.
- 현재 어댑터(`clef-ft-20261004c`)는 별도 ZIP, SAM 3 추적기·최신 계획/Live 코드는 소스 ZIP에 제공합니다. 두 ZIP을 같은 제출 묶음으로 제공하며, 소스 식별은 「02」의 파일별 SHA-256 manifest를 따릅니다.
