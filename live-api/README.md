# synoptics Live API 서버 (모의 엔진 / 실제 엔진)

`synoptics-live/0.1-draft` 프로토콜(스펙: `~/Projects/aisw/api-spec/live-api.md`)을 그대로 구현한 독립 서버입니다.
REST·WebSocket·바이너리 봉투·세션·크레딧·재연결·TTS 연결은 엔진과 상관없이 실제로 동작합니다.
가이드 판단(계획·추적·진행 판정·대화)은 엔진이 맡으며, `LIVE_ENGINE`으로 고릅니다.

- `mock`(기본값): 각본대로 움직이는 모의 엔진입니다. 웹 클라이언트를 프로토콜에 맞춰 먼저 만들 때 씁니다.
- `real`: 기존 데모 백엔드(aisw-hybrid-talk, DeepSeek 계획·판정 + 추적기 + 로컬 추종 모델)를 상위 서버로 호출하는 실제 엔진입니다. 아래 "실제 엔진" 절을 보세요.

**Plan 모드**(스펙 §15, 2026-10-03 추가)는 두 엔진 모두에 들어 있습니다. 계획을 먼저 만들고 `phase:"reviewing"`으로
검토를 받은 뒤에야 `plan_approve`로 실행합니다. 승인 전에는 추적도 판정도 시작하지 않습니다. 필수 검사
(`GuideStep.required`)는 건너뛸 수 없고, `check:"user"`·`"measure"` 단계는 `step_ack`으로만 끝나며, 실행 중
재계획은 계획을 갈아치우지 않고 `state.proposal`로 올라와 `proposal_accept`를 기다립니다. 규칙은
`plan_policy.py` 한 곳에 있고, 두 엔진이 그것을 씁니다.

### 그래프 코어 연구 후보 (운영 미배포)

`start.core_mode:"graph"`를 선택하면 단계 번호는 화면의 제안 위치일 뿐 완료 기록이 아닙니다.
`steps_done`은 상위 확인 또는 명시적 사용자 확인, `steps_ready`는 선행 안전 절차를 포함한 행동 가능 노드,
`step_statuses`는 최신 `yes/no/unsure` 관측입니다. 전체 16개 노드를 관찰하며 독립 노드는 어떤 순서로도 완료할 수 있습니다.
`condition_kind:"state"`는 명확한 반증으로 완료를 취소하고 의존 노드를 재확인합니다. `event`는 실제 과거 발생만 표현합니다.
부품 삽입 결과가 유지돼야 한다면 `state`이며, 가려짐을 이유로 `event`로 바꾸지 않습니다.

확정 근거는 내부 `graph_facts`에 최신 관측과 별도로 보관합니다. 느린 상위 확인이 도착해도 더 최신인
`unsure`를 `yes`로 덮지 않습니다. 실제 응답의 provider/model·승인 경로와 원래 계획/단계/프레임을 유지하며,
Clef의 단일 사진 `visible` 의미나 검수용 `temporal-state-v2` 입력을 바꾸지 않습니다.
승인된 재계획과 변경안 거절 후 복원은 같은 대상·목표 아래 **조건/검사 권한/논리 대상/직접 의존성**이
정확히 일치하고 대응이 유일한 노드만 이월합니다. 번호나 안내 문장 변경만으로 확정 사실을 잊지 않지만,
중복 부품의 애매한 대응·바뀐 조건·바뀐 의존 노드는 다시 확인합니다. 이월은 새 프레임 관측이 아닙니다.
명시적 대상 재선택에는 동일 물체 증명이 없으므로 사용자 확인을 포함한 현재 적용을 해제합니다.
최초 자동 추적 획득은 재선택이 아닙니다. 활성 확정 근거는 관측 수와 무관하게 보존하며,
과거 관측·확인 로그는 실행 메모리의 최근 1024개까지입니다(영구 세션 기록 아님).

`goal_required:true`는 물리적 목표 달성에 필수인 조건입니다. `goal_required:false, required:true`는
안전·절차 선행조건으로 행동을 계속 제한하지만, 이미 달성된 목표 자체를 거짓으로 만들지는 않습니다.
`false/false`는 선택 조건입니다. `pending_required_checks`는 목표 확인·최종 완료 후에도 남는 미확인 절차를 표시합니다.
`classic/sequential`은 대조용 기존 진행 정책을 유지합니다. 내장 `/demo/`에서 그래프 선택과 두 필드 편집을 지원합니다.

동작 계약에는 `align/connect/disconnect/bend`가 추가됐으며 방향은 `none`만 허용합니다.
내장 Canvas 진단 클라이언트는 기존 방식대로 문자 라벨을 표시합니다. 애니메이션은 별도 React 렌더러의
`/motion-preview.html`에서 확인하며, 실제 핀 좌표·극성·통전 안전을 뜻하지 않습니다.

## 실행

```bash
cd ~/Projects/aisw/live-api
uv sync
uv run pytest -q          # 네트워크·포트 없이 TestClient로만 검증
uv run uvicorn synoptics_live.app:app --host 127.0.0.1 --port 8104 \
  --proxy-headers --forwarded-allow-ips 127.0.0.1 --ws-ping-interval 15
```

- 포트는 `nodectl lease`로 빌린 뒤 띄웁니다 (`127.0.0.1:8104`, nginx `location /synoptics/api/`가 접두사를 떼고 넘김).
- `--ws-ping-interval 15`: 스펙 §4.1의 15초 ping은 uvicorn이 보냅니다. 앱 코드에는 ping이 없습니다.
- 메시지 크기 상한(512 KB → close 4009)은 앱이 검사합니다. uvicorn `--ws-max-size`는 기본값(16 MB)으로 두세요. 더 작게 잡으면 uvicorn이 먼저 1009로 끊어 4009가 나가지 않습니다.
- 참조 사진을 실을 수 있는 제어 메시지(`start`, `plan_answer`)만 10 MB 봉투(`max_reference_message_bytes`)를 씁니다. 장당 상한(≤1.5MB JPEG, 최대 12장)은 그 안에서 그대로 강제합니다.
- 참고용 테스트 클라이언트: `GET /demo/` (nginx 뒤에서는 `https://<호스트>/synoptics/api/demo/`). 상대 URL만 쓰므로 접두사 아래에서도 동작합니다.
- 프로세스는 메모리에만 세션을 둡니다. 재시작하면 세션은 모두 사라집니다(종료 시 연결 중인 소켓에는 `bye{server_restart}` + 1012).

## 환경 변수

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `LIVE_ACCESS_CODE` | (없음) | 설정하면 `access_mode:"code"`, 세션 생성에 코드 필요. 없으면 `"open"` |
| `LIVE_ENGINE` | `mock` | `mock` 또는 `real`. 그 밖의 값(또는 `real`인데 상위 URL이 없을 때)은 health `ready:false`, 세션 생성 503 `service_unavailable` |
| `LIVE_UPSTREAM_URL` | `http://127.0.0.1:8045` | `real` 엔진이 호출하는 데모 백엔드(개발 호스트 터널 → GPU 노드 app 8040). 요청의 `Origin`은 이 URL의 `scheme://host[:port]`로 붙입니다 |
| `LIVE_UPSTREAM_ACCESS_CODE` | (없음) | 상위 health가 `access_code_required:true`일 때 상위 세션을 만들며 보낼 접근 코드 |
| `LIVE_CORS_ORIGINS` | `*` | 쉼표 목록. REST CORS와 WebSocket `Origin` 검사에 같이 씁니다(`*`이면 WS 검사 안 함). 자격 증명 없음(Bearer) |
| `LIVE_PATH_PREFIXES` | (없음) | `호스트=/접두사,…`. `X-Forwarded-Prefix`를 보내지 않고 마운트 경로를 떼는 프록시(Tailscale Serve `--set-path`) 뒤에서 `live_url`에 붙일 접두사. 호스트에 포트가 붙으면(`:8450`) 다른 키다. 저장소 밖 프록시를 쓸 때만 설정하며, 로컬 루프백 실행에는 필요하지 않다 |
| `LIVE_MOCK_SPEED` | `1.0` | 모의 엔진 시간 배율. 2.0이면 모든 지연이 절반 |
| `LIVE_MOCK_STEP_S` | `8` | 추적 중인 시간 기준으로 단계가 넘어가는 간격(초, 최소 1.5) |
| `LIVE_TTS_URL` | (없음) | 제출 기본은 서버 음성 비활성(`tts_ready:false`), 브라우저·기기 TTS 사용. 음성 모델·서비스는 포함하지 않음 |
| `LIVE_TTS_VOICE` | `default` | 별도 서버 음성을 명시적으로 구성할 때만 쓰는 일반 라벨. 제출 기본에서는 `tts_voice:null` |
| `LIVE_FFMPEG` | `/usr/bin/ffmpeg` | WAV → Ogg Opus(32k) / AAC(mp4) 변환 |
| `LIVE_HELLO_TIMEOUT_S` | `5` | `hello` 대기 시간 |
| `LIVE_RESUME_GRACE_S` | `30` | 끊긴 뒤 재연결 유예. `ready.limits.resume_grace_ms`에도 반영 |
| `LIVE_BUILD_COMMIT`, `LIVE_BUILD_TIME` | `dev`, 시작 시각 | health `build` |

고정 값: 동시 세션 32, 세션 생성 IP당 분당 5회(실패한 시도도 셈), 유휴 15분, 크레딧 위반 20회 → 4008.
클라이언트 IP는 `CF-Connecting-IP` → `X-Real-IP` → 소켓 주소 순으로 봅니다. 이 헤더들은 그대로 믿으므로 서버는 반드시 프록시 뒤(127.0.0.1)에만 둡니다.

## 무엇이 진짜이고 무엇이 모의인가

| 영역 | 상태 |
|---|---|
| REST(health, 세션 생성·삭제), 접근 코드, 레이트 리밋, 상한, 유휴 만료 | 실제 |
| WS 인증(hello 5초, 4001/4002/4003/4008/4009), 바이너리 봉투, JPEG 검증(FFD8 + Pillow 헤더, w/h 일치), 크기 상한, 크레딧 1개 | 실제 |
| `state` 스냅샷·`rev`, 재연결(`resume_rev`, 30초 유예 동안 엔진 일시정지), `say`/`capture_hi` 전달 | 실제 |
| 브라우저·기기 TTS | 제출 기본. 서버는 읽을 문장을 전달하고 클라이언트가 Web Speech API로 읽음. 서버 음성 모델·가중치 미포함 |
| 물체 추적 | **모의**: 화면 가운데 근처를 천천히 도는 박스. 이미지를 보지 않습니다. 시작 후 5프레임은 `acquiring` |
| 계획 | **모의**: 목표와 무관한 일반 3단계(잡기 / 시계 방향 돌리기 / 라벨+놓기), 대상 이름 "목표 물체" |
| 단계 진행·완료 판정 | **모의**: 추적 중인 시간이 `LIVE_MOCK_STEP_S`만큼 쌓이면 다음 단계, 마지막 단계 뒤 `checking` → 1.5초 → `confirmed` |
| 대화 | **모의**: 1초 뒤 "(모의 응답)" 고정 문장. "다시"가 들어 있으면 재계획, "했어"/"완료"면 현재 단계를 사용자 완료로 표시하고 넘어감 |
| 지금 다시 보기 / 재계획 | **모의**: 0.8초 뒤 `notice{rechecked}` / 즉시 revision+1 (가이드당 3회, 15초 간격) |
| Plan 모드(§15) 초안 내용 | **모의**: 목표와 무관한 고정 6단계(필수 검사 1개·사용자 확인 1개·선행 조건 1개) |
| Plan 모드 규칙(승인 경계·필수 검사 건너뛰기 금지·선행 조건·완료 조건·변경안 승인) | **실제**: `plan_policy.py`가 강제합니다. 모의 엔진도 실제 엔진도 같은 함수를 씁니다 |
| Plan 프로필 선택(`start.plan_model`, health `plan_models`) | **실제**: 닫힌 두 값(`deepseek:high` 기본 / `astra:high`)이고, 실제 엔진은 그 값을 계획 요청에 실어 보냅니다. 모의 엔진은 값을 저장하고 두 프로필 모두 준비된 것으로 보고합니다 |
| 질문 왕복(`state.clarification` + `clarification_id` → `plan_answer`) | **실제**: 검토 모드에서 상위가 계획 대신 질문을 돌려주면 `phase:"clarifying"`으로 멈추고, 사용자가 그 질문 id를 에코해 답하면 같은 상위 세션으로 계획을 이어 씁니다(목표·자료·사진은 리셋하지 않음). 런당 답변 상한 12회. 모의 엔진은 목표에 `?`가 있을 때 한 번 묻습니다 |
| 참조 사진(`start`/`plan_answer`의 `reference_images`) | **실제**: 현재 장면이 아닌 별도 증거(최대 12장, 장당 JPEG ≤1.5MB). 계획 요청마다 함께 실어 상위가 계속 참고하게 합니다. 모의 엔진은 검증 후 보관만 합니다 |
| 리서치 출처(`Plan.research_sources` / `state.research_sources`) | **실제**: 계획 프로필이 `astra:high`이고 검토 모드일 때만 상위에 `research:true`로 요청하고, 돌아온 공개 HTTP(S) 출처만 와이어로 냅니다(최대 6개). 모의 엔진은 만들지 않습니다 |
| 단계 상세 문구(`GuideStep.details`) | **실제**: 음성 `say` 아래에 보여 줄 읽기용 지침. 초안·수정·재계획·승인을 거쳐도 보존됩니다 |
| 호출 예산(`budget_notice`, `talk_budget`), `needs_reselect`, `basis_age_s`, `phase:"error"` | 필드만 있고 모의 엔진은 만들지 않습니다 |

위 표는 모의 엔진 기준입니다. 실제 엔진은 아래 절을 보세요. 엔진은 `engine.py`의 `Engine` 프로토콜(`on_start` … `on_frame` → `track`, `close`)을 구현하고, 소켓을 직접 만지지 않고 `EngineSink`(`state` / `say` / `capture_hi`)로만 내보냅니다.

## 실제 엔진 (`LIVE_ENGINE=real`)

브라우저 데모(aisw-hybrid-talk `web/src`, 커밋 73ec3a8)가 하던 가이드 조율을 서버로 옮긴 것입니다. 데모 백엔드는 **바꾸지 않고** HTTP 상위 서버로 그대로 씁니다.

```
웹 클라이언트 ──WS(프레임·제어)──▶ Live API (이 서버, RealEngine) ──HTTP──▶ 데모 백엔드 (LIVE_UPSTREAM_URL)
                                     │  가이드 루프(트리거·게이트·수락 규칙·말하기)          ├─ /api/guide/plan (SSE) · follow · confirm · talk
                                     │  live 세션 1개 = 상위 세션 1개 (쿠키 + Origin)        ├─ /api/track/control · frame (추적기)
                                     └─ 상위 health 10초 캐시 → /v1/health                  └─ /api/health
```

| 브라우저가 하던 일 | 이 서버에서 |
|---|---|
| 카메라 캡처 | 클라이언트 스트림 프레임. 현재 추적 run이 있으면 그 프레임을 `/api/track/frame`에 넘기고 같은 정규화 박스를 `track`으로 돌려줍니다. run이 없으면 상위 호출 없이 `track{idle}` |
| `liveTrackStore` 스냅샷 | 가장 최근 추적 응답 + **그 응답이 계산된 JPEG** 한 쌍 |
| follow/confirm/talk 장면 | 그 한 쌍의 JPEG과, 같은 응답으로 만든 AnchorRef(박스는 tracking일 때만). 박스와 장면이 항상 같은 캡처입니다(`anchorProject.ts`). 나중에 고화질 장면으로 바꾸려면 `real_engine.py`의 `_capture_scene` 한 곳만 고치면 됩니다 |
| 계획 장면 | `start` 때 `capture_hi`를 보내고 2초 안에 온 고화질 프레임, 없으면 최신 스트림 프레임 |
| 트리거·디스패치 게이트·예산 | `guide_policy.py`에 TS 그대로 이식 (acquired / target_moved / anchor_lost 1초 / target_changed(박스 안 외형) / heartbeat 로컬 4초·DeepSeek 8초, 단계별 in-flight 1 + pending 1 병합, DeepSeek 간격 1.2초·바닥 2.2초·2분 30회·분당 20회) |
| 단계 진행 | follow 체크리스트로 advance / skip(두 번 연속 보일 때) / pendingSkip, step_done confirm으로 재확인 후 `no`면 되돌림, goal_seen → goal_check → `checking` → 1.5초 뒤 새 프레임으로 재확인 → `confirmed` |
| 재계획 | unsure 2회, target_changed 3회 정체, "계획 다시 짜기", 대화. 가이드당 3회, 15초 간격 |
| Plan 모드(§15) 승인·되돌리기 | 승인하면 검토한 계획을 `POST /api/guide/plan/approve`로 상위에 올립니다. 그래서 사용자가 고친 초안과 상위가 아는 판본이 갈라지지 않습니다. 변경안을 거절하면 `POST /api/guide/plan/revert`로 상위 계획을 되돌리고 실행을 이어 갑니다. 상위가 되돌리기를 거절할 때만 `plan_changed`로 끝납니다(예외 경로) |
| 대화 | 한 번에 하나, DeepSeek 예산 공유, 바닥 시간만큼 기다리는 동안 다른 유료 호출을 막음, 적용 순서 계획 문장 → 위치 → 대상 |
| 오류 | 401 → `phase:error`(상위 세션 무효화, 다음 `start`에 새 상위 세션), 409 stale_plan(대화·재계획에 추월된 것)은 조용히 버림·그 밖의 409는 오류, 429/503 → `Retry-After`(기본 1초) 뒤 같은 호출 재시도, 연속 3회 실패 → 오류 |
| 화면 문구·음성 | `speechPolicy.ts`처럼 직전 상태와 비교해 `say`를 만듭니다(단계 문장·대화 답은 `replace`, 안내·오류·완료는 `append`). notice는 단계가 바뀌거나 8초 뒤 지워집니다 |

### 실제 엔진의 알려진 한계

- **상위 서버가 전제입니다.** 추적기·추종 모델이 내려가 있으면 시작 후 추적이 실패할 수 있습니다. 최초 계획에는 이미 선택한 프로필(`start.plan_model`, 기본 `deepseek:high`) 호출 1회를 쓴 뒤입니다.
- **Plan 모드는 상위가 승인·되돌리기와 계획 프로필(`plan_model`)을 지원해야 합니다.** 실제 엔진은 검토 여부와 무관하게 `plan_model`을 상위 `/api/guide/plan`에 보내고, 상위는 그 선택을 계획에 저장해 확인·재계획·대화가 다른 프로필로 새지 않게 합니다(검토 경계는 Live의 `start.plan_mode`가 정하며 상위로는 전달하지 않습니다). 두 프로필 모두 높은 추론 강도와 180초 기한을 쓰므로 최초 계획·확인·대화 상위 요청은 190초까지 기다립니다(반복 follower 기한은 그대로). 인증 실패는 다른 모델로 자동 대체하지 않습니다. 변경된 상위와 Live를 함께 배포해야 하며 이 소스 변경만으로 실행 중 프로세스를 재시작하지 않습니다.
- **추적이 `lost`/`unavailable`이 되면 그 run은 끝입니다**(브라우저와 같은 규칙). `lost` 1초 뒤 완료 확인(goal_check)을 한 번 하고, 그 뒤 다시 잡으려면 클라이언트가 `select_box`로 직접 지정해야 합니다. 서버가 저절로 다시 찾지는 않습니다.
- `select_box`는 상위에서는 새 추적 run(시드 박스, 대상 `user-selected object`)이지만, 와이어에서는 같은 `run_id`에 `generation`+1, 새 `track_id`로 보입니다. 계획 중(`planning`)에는 무시합니다.
- 장면은 스트림 해상도(640px)입니다. 데모는 최대 1920px로 판단했으므로 판단 정확도는 아직 비교하지 않았습니다. 상위는 320×240보다 작은 프레임을 거절합니다.
- 연결이 끊기면(재연결 유예 중) 추적 run을 놓고 판단 호출도 멈춥니다. 다시 붙으면 계획의 대상으로 새 run을 시작합니다(처음 몇 프레임은 `acquiring`).
- `confirmed`가 되어도 `phase`는 `confirm_done`까지 `running`입니다(스펙 §11). 그동안 판단 루프는 멈춰 있고 `talk`·`follow_now`·`replan_now`는 `not_running`으로 거절합니다.
- 대화 실패 문구는 별도 필드가 없어 `notice{code:"talk_failed"}`로 보냅니다.
- 상위 세션은 상위 서버의 15분 유휴 만료를 따릅니다. 만료되면 진행 중인 가이드는 `session_expired` 오류로 끝나고, 다음 `start`가 새 상위 세션을 만듭니다.

### 시험

```bash
uv run pytest -q                       # 가짜 상위(httpx.MockTransport) + 주입 시계, 포트·네트워크 없음
# 실제 상위(8045)로 하는 스모크: 앱을 프로세스 안에서 띄움(TestClient, 포트 바인딩 없음), TTS 끔, DeepSeek 몇 회 사용
ffmpeg -i ~/Projects/aisw/glass.mov -vf fps=10,scale=640:-2 -q:v 5 $DIR/frames/%04d.jpg
ffmpeg -i ~/Projects/aisw/glass.mov -vf fps=10,scale=1024:-2 -q:v 4 $DIR/frames_hi/%04d.jpg
uv run python scripts/smoke_real.py --frames $DIR/frames --frames-hi $DIR/frames_hi [--hold-end] [--talk "…"]
# Plan 모드(§15) 끝단 스모크: 네트워크 너머의 서버에 실제로 붙는다(카메라 없이 합성 JPEG 프레임, mock 엔진 기준)
uv run python scripts/smoke_plan_mode.py --base http://127.0.0.1:8104 --max-seconds 120
```

## 파일

| 파일 | 내용 |
|---|---|
| `src/synoptics_live/contracts.py` | 모든 REST 본문과 WS 메시지의 pydantic 모델, `Limits` |
| `src/synoptics_live/plan_policy.py` | Plan 모드 규칙(스펙 §15.4): 승인 경계, 필수 검사, 선행 조건, 수정 적용, 변경안 설치. 순수 함수만 |
| `scripts/smoke_plan_mode.py` | 네트워크 너머의 서버에 붙어 Plan 모드 흐름(초안→편집→승인→실행→확인)을 끝까지 밀어 보는 스모크 |
| `src/synoptics_live/envelope.py` | 바이너리 봉투 pack/unpack |
| `src/synoptics_live/sessions.py` | `SessionStore`(생성 정책·만료), `LiveSession`(엔진 sink, rev, TTS 전달) |
| `src/synoptics_live/ws.py` | live 소켓: hello, 프레임 검증·크레딧, 상태 기반 거절(gating), 디스패치 |
| `src/synoptics_live/engine.py` / `mock_engine.py` | 엔진 계약 / 모의 엔진 |
| `src/synoptics_live/real_engine.py` | 실제 엔진: 가이드 루프(`useIntentLoop.ts`)와 추적 run(`useObjectTrack.ts`)의 서버 이식 |
| `src/synoptics_live/guide_policy.py` | TS 순수 규칙 이식(트리거·게이트·단계·확인·재계획·대화·수락·외형 변화·음성). 규칙마다 TS 파일:줄 표시 |
| `src/synoptics_live/upstream.py` | 데모 백엔드 HTTP 클라이언트(쿠키·Origin, 계획 SSE 파서, health 캐시) |
| `scripts/smoke_real.py` | 실제 상위로 하는 스모크(위 "시험") |
| `src/synoptics_live/tts.py` | TTS 브리지 |
| `src/synoptics_live/app.py` | FastAPI 앱, `live_url` 생성, CORS, `/demo/` |
| `src/synoptics_live/static/index.html` | 참고 테스트 클라이언트 (바닐라 JS) |
