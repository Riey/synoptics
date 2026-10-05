# synoptics Live API · 클라이언트 통합 안내

- 프로토콜: `synoptics-live/0.1-draft`
- 작성: 2026-10-03 · 통합 갱신: 2026-10-04
- 상태: 제출 스냅샷의 draft 프로토콜입니다. 로컬 GPU 실행 설정은 `health.engine == "real"`을 사용하며 실제 준비 여부는 `/v1/health`로 확인합니다. 고정 응답 계약 검증과 실제 모델 품질 측정은 구분합니다.
- 테스트 클라이언트: `http://127.0.0.1:8104/demo/` (로컬 GPU 설정의 내장 진단 클라이언트. 카메라·프레임 전송·박스·오버레이 표시·브라우저 TTS를 제공하며 서버 음성은 기본 비활성)
- 바뀔 수 있는 값은 문서에 숫자를 박지 않고 `ready.limits`로 내려보냅니다. 클라이언트는 하드코딩하지 말고 그 값을 쓰세요.
- 이 문서 하나에 **API 명세**(§1–15)와 **클라이언트 통합 인계서**(§16)를 함께 담았습니다. 별도 PWA 수정·인수는 클라이언트 담당자의 작업이며, 서버 반영만으로 끝난 것이 아닙니다.

[API 계약](#api-reference) · [통합 인계서](#client-handoff) · [인수 체크리스트](#client-acceptance)

---

<a id="api-reference"></a>

## 1. 역할 분담

synoptics는 카메라 화면을 보고 사용자가 손으로 하는 작업(예: 안경 벗기, 에어팟 넣기)을 단계별로 안내하는 서비스입니다. 진행 판단은 **모두 서버**가 합니다. 웹 클라이언트는 입출력 장치 역할만 맡습니다.

| 웹 클라이언트가 하는 일 | 서버가 하는 일 |
|---|---|
| 카메라 캡처, 좌우 반전 처리, JPEG 인코딩, 프레임 전송 | 물체 추적(박스), 대상 선택 |
| 서버가 준 박스와 오버레이 명령으로 화면 그리기 (rAF) | 계획 수립, 단계 진행 판정, 완료 확인, 재계획 |
| 음성 인식(Web Speech) → 텍스트 전송 | 사용자 발화 해석과 응답 |
| 브라우저·기기 TTS(Web Speech API)로 문장 읽기 | 읽을 문장 결정. 제출 기본은 `tts_ready:false`이며 서버 음성 모델 미포함 |
| 버튼(시작, 중지, 다시 보기, 계획 다시 짜기, 완료 확인) | 호출 예산, 재시도, 오래된 응답 걸러내기 |

클라이언트는 상태 기계를 갖지 않습니다. 서버가 보내는 `state` 스냅샷을 그대로 화면에 그리면 됩니다.

---

## 2. 연결 개요

```
GET    /v1/health                 준비 상태 (인증 없음)
POST   /v1/sessions               세션 발급
DELETE /v1/sessions/{sid}         세션 종료
WS     /v1/sessions/{sid}/live    나머지 전부
```

- 로컬 GPU 설정의 Base URL: **`http://127.0.0.1:8104`** (예: `GET http://127.0.0.1:8104/v1/health`, WebSocket은 `ws://127.0.0.1:8104/v1/sessions/{sid}/live`). 세션 발급 응답의 `live_url`을 그대로 쓰면 됩니다. 별도 HTTPS 프록시의 주소·경로는 운영자가 설정합니다.
- 실행 및 원격 브라우저 접속: 저장소 루트의 `README.md`를 따릅니다.
- REST는 CORS 허용 목록 방식입니다. 웹앱 출처(origin)를 서버 운영자에게 알려 주세요. WebSocket도 같은 목록으로 `Origin`을 검사합니다.
- 모든 JSON 필드는 `snake_case`, 시각은 **epoch 밀리초**(정수), 박스 좌표는 **0..1 정규화**입니다.

---

## 3. REST

### 3.1 `GET /v1/health`

```json
{
  "ready": true,
  "protocol": "synoptics-live/0.1-draft",
  "access_mode": "code",
  "access_code_required": true,
  "guide_ready": true,
  "plan_ready": true,
  "plan_models": { "deepseek:high": true, "astra:high": true },
  "follow_ready": true,
  "tracker_ready": true,
  "tts_ready": true,
  "tts_voice": "operator-ko",
  "build": { "commit": "73ec3a8", "time": "2026-10-03T10:00:00+09:00" }
}
```

- `access_mode`: `code` | `local` | `open`. `access_code_required`가 `false`이면 접근 코드 입력 UI를 보이지 마세요. 클라이언트가 자기 호스트명으로 추측하면 안 됩니다.
- `tts_ready`가 `false`이면 처음부터 브라우저 TTS를 쓰세요 (§7).
- `plan_models`는 `start.plan_model`이 받는 닫힌 집합의 프로필별 준비 여부입니다(키는 `deepseek:high`, `astra:high` 둘뿐). `plan_ready`는 그 집계(하나라도 `true`)이고, `guide_ready`는 종전의 기본 provider 준비 상태입니다. 이 값은 **구성 여부**일 뿐 프로젝트 권한·잔액·할당량을 보증하지 않습니다. 시작 화면에서 프로필을 고르게 할 때 이 맵으로 선택 가능 여부를 표시하세요.
- `ready`가 `false`이면 시작 버튼을 막고 안내만 보여 주세요.

### 3.2 `POST /v1/sessions`

요청:
```json
{ "access_code": "…" }
```
`access_code_required`가 `false`이면 필드를 아예 빼고 보냅니다.

응답 `201`:
```json
{ "session_id": "6f1c…", "token": "…", "expires_at": 1759480000000, "live_url": "wss://sangye.kr/synoptics/api/v1/sessions/6f1c…/live" }
```

| 상태 | 코드 | 의미 |
|---|---|---|
| 401 | `invalid_access_code` | 코드 누락이나 불일치 |
| 429 | `rate_limited` | 같은 IP에서 분당 5회 초과. `Retry-After` 헤더를 보세요 |
| 503 | `capacity` | 동시 세션 상한(현재 32) 도달 |
| 503 | `service_unavailable` | 서버 설정 문제 |

- 세션은 **15분 동안 아무 활동이 없으면** 만료됩니다. live 연결이 열려 있으면 활동 중으로 칩니다.
- 토큰은 메모리에만 두세요. localStorage에 저장하지 마세요.

### 3.3 `DELETE /v1/sessions/{sid}`

헤더 `Authorization: Bearer <token>`을 붙입니다. 응답은 `{ "ended": true }`입니다. AI 동의를 철회하거나 페이지를 떠날 때 부르세요(`navigator.sendBeacon`은 헤더를 붙일 수 없으므로 `fetch(..., {keepalive: true})`를 쓰세요).

---

## 4. Live WebSocket

### 4.1 연결과 인증

1. `new WebSocket(live_url)`로 엽니다. 브라우저 WebSocket은 헤더를 붙일 수 없으므로 토큰은 첫 메시지로 보냅니다.
2. 열리면 **바로** `hello`를 보냅니다. 5초 안에 `hello`가 없으면 서버가 닫습니다.
3. 서버가 `ready`를 보내면 그때부터 다른 메시지를 보낼 수 있습니다.

```json
{ "type": "hello", "token": "…", "protocol": "synoptics-live/0.1-draft", "resume_rev": null, "audio_accept": ["audio/ogg"] }
```

```json
{
  "type": "ready",
  "rev": 0,
  "server_time": 1759479000000,
  "limits": {
    "stream_long_side": 640,
    "stream_jpeg_quality": 0.7,
    "stream_max_bytes": 120000,
    "hi_long_side": 1024,
    "hi_max_bytes": 500000,
    "max_fps": 15,
    "box_max_age_ms": 500,
    "tts_wait_ms": 1500,
    "resume_grace_ms": 30000,
    "goal_max_chars": 300,
    "context_max_chars": 1000,
    "utterance_max_chars": 200,
    "plan_steps_max": 12,
    "materials_max": 4,
    "material_text_max_chars": 4000
  }
}
```

`ready` 직후에 서버는 현재 `state`를 한 번 보냅니다.

- 세션당 live 연결은 **하나**입니다. 같은 세션으로 새 연결이 붙으면 이전 연결은 close code `4003`으로 닫힙니다.
- 서버는 15초마다 WebSocket ping을 보냅니다. 브라우저가 알아서 응답하므로 클라이언트가 할 일은 없습니다.

### 4.2 메시지 형식

- **텍스트 메시지**: JSON 객체 하나이고, `type` 필드로 구분합니다.
- **바이너리 메시지**: 프레임(클라이언트 → 서버)과 오디오(서버 → 클라이언트)입니다. 형식은 아래와 같습니다.

```
┌──────────────────────┬──────────────────────────┬─────────────┐
│ header_len (uint32 BE)│ header (UTF-8 JSON)       │ payload     │
└──────────────────────┴──────────────────────────┴─────────────┘
```

```js
function packBinary(header, payload /* ArrayBuffer */) {
  const h = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(4 + h.length + payload.byteLength);
  new DataView(out.buffer).setUint32(0, h.length, false);
  out.set(h, 4);
  out.set(new Uint8Array(payload), 4 + h.length);
  return out.buffer;
}

function unpackBinary(buf /* ArrayBuffer */) {
  const len = new DataView(buf).getUint32(0, false);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, len)));
  return { header, payload: buf.slice(4 + len) };
}
```

`ws.binaryType = "arraybuffer"`로 설정하세요.

---

## 5. 클라이언트 → 서버

### 5.1 프레임 (바이너리)

```json
{ "t": "frame", "seq": 128, "captured_at": 1759479001234, "w": 640, "h": 360 }
```
payload는 JPEG 바이트입니다.

- **좌우 반전**: 화면을 거울처럼 보여 준다면, 캡처할 때 **픽셀에 반전을 적용한 상태로** 인코딩해서 보내세요. 서버가 돌려주는 박스 좌표는 받은 이미지 기준이므로, 화면에 보이는 그대로의 좌표가 됩니다.
- **해상도**: 긴 변을 `limits.stream_long_side`로 맞추고, 품질은 `stream_jpeg_quality`로 인코딩합니다. 결과가 `stream_max_bytes`를 넘으면 품질을 0.1씩 낮추세요.
- `seq`는 연결마다 0부터 시작해서 1씩 증가합니다. 재연결하면 다시 0부터 시작합니다.
- `captured_at`은 캔버스에 그린 순간의 `Date.now()`입니다.

**흐름 제어 (크레딧 1개).** 응답을 기다리지 않은 프레임은 한 번에 **하나만** 있을 수 있습니다.

1. 프레임 `seq=n`을 보냅니다.
2. 서버가 `track{seq:n}`을 보낼 때까지 다음 프레임을 보내지 않습니다.
3. `track{seq:n}`을 받으면, 카메라에 새 프레임이 있을 때 바로 `n+1`을 캡처해서 보냅니다(`requestVideoFrameCallback`을 쓰고, 없으면 rAF). 다만 `1000 / max_fps` ms보다 빨리 보내지는 않습니다.
4. 2초가 지나도 `track`이 오지 않으면 다음 프레임을 보내도 됩니다. 서버는 오래된 프레임을 버립니다.

카메라가 켜져 있고 live 연결이 살아 있는 동안에는 가이드를 시작하지 않았더라도 계속 보냅니다. 서버가 시작 순간의 화면을 바로 쓸 수 있게 하기 위해서입니다. 가이드가 없을 때 서버는 `track{state:"idle"}`로 답합니다.

**고화질 프레임.** 서버가 `capture_hi{req_id}`를 보내면, 다음 캡처를 긴 변 `limits.hi_long_side`로 인코딩해서 아래 헤더로 **한 장** 보냅니다. 고화질 프레임은 크레딧과 무관하게 바로 보내도 됩니다.
```json
{ "t": "frame", "seq": 129, "captured_at": 1759479001301, "w": 1024, "h": 576, "hi_req": "h7" }
```

**사용자가 직접 대상을 지정할 때.** 사용자가 정지 화면 위에 박스를 그렸다면, **그 정지 화면 자체**를 프레임으로 보내면서 `select_box`를 붙입니다.
```json
{ "t": "frame", "seq": 140, "captured_at": 1759479005000, "w": 640, "h": 360,
  "select_box": { "x": 0.31, "y": 0.22, "width": 0.18, "height": 0.25 } }
```
- 박스는 보낸 이미지 기준의 정규화 좌표입니다(반전이 적용된 상태). `x + width ≤ 1`, `y + height ≤ 1`을 지켜야 합니다.
- 화면에서 12px보다 작게 끌어 그린 것은 클라이언트에서 거부하세요.
- 서버는 이 박스로 추적을 새로 시작합니다. 가이드는 멈추지 않고 그대로 이어집니다.

### 5.2 제어 메시지 (텍스트)

| `type` | 필드 | 언제 |
|---|---|---|
| `hello` | `token`, `protocol`, `resume_rev` | 연결 직후 (§4.1, §9) |
| `start` | `goal`(1–300자), `context`(≤1000자, 생략 가능), `consent_ai:true`, `plan_mode`(bool, 기본 false), `plan_model`(`"deepseek:high"`\|`"astra:high"`), `core_mode`(`"classic"`\|`"sequential"`\|`"graph"`, 기본 `"classic"`), `materials`(≤4개), `reference_images`(≤12개) | 기존 가이드가 있으면 새 실행으로 교체합니다. 검토·상위 프로필·진행 코어는 독립이며 실행 중 고정입니다. §15.13–14 |
| `plan_answer` | `clarification_id`(현재 질문 id), `answer`(1–1000자), `reference_images`(≤12개, 생략 시 보존) | `clarifying`의 확인 질문에 답하기. 새 `start`가 아님. 상세 §15.14 |
| `stop` | — | 중지 버튼. 가이드와 추적을 모두 멈춥니다 |
| `talk` | `utterance`(1–200자) | 음성 인식이 최종 결과를 냈거나, 텍스트로 질문을 보냈을 때 |
| `follow_now` | — | "지금 다시 보기" 버튼 |
| `replan_now` | — | "계획 다시 짜기" 버튼 |
| `confirm_done` | — | `completion == "confirmed"`일 때 사용자가 "완료 확인"을 누름 |
| `prefs` | `voice_out: bool`, `tts: "server"\|"browser"`, `visuals: bool` | 설정이 바뀔 때마다. 연결 직후에도 한 번 보내세요 |
| `played` | `line_id`, `via: "server"\|"browser"\|"skipped"` | 한 문장을 다 읽었거나 건너뛰었을 때 (서버 측 측정용) |
| `plan_edit` | `edits`, `add`, `remove`, `order`, `goal_when` (모두 생략 가능) | §15. 검토 중인 초안 고치기 |
| `plan_approve` | — | §15. 검토한 계획을 승인하고 실행 시작 |
| `plan_discard` | — | §15. 초안 버리고 `idle`로 |
| `run_pause` | — | §15. 실행 일시정지 |
| `run_resume` | — | §15. 실행 재개 |
| `step_ack` | `step_id` | §15. `check:"user"` 단계를 사용자가 확인 |
| `proposal_accept` / `proposal_reject` | — | §15. 실행 중 변경안을 적용 / 버리기 |

예:
```json
{ "type": "start", "goal": "안경을 벗어서 케이스에 넣기", "context": "", "consent_ai": true }
{ "type": "talk", "utterance": "케이스 뚜껑이 안 열려요" }
```

- `consent_ai`가 `true`가 아니면 `start`와 `talk`는 `error{code:"consent_required"}`로 거절됩니다. 동의를 철회하면 `stop`을 보내고 `DELETE /v1/sessions/{sid}`를 호출하세요.
- 목표나 맥락 문장을 바꾸는 것은 새 작업입니다. 다시 `start`를 보내면 됩니다.

---

## 6. 서버 → 클라이언트

### 6.1 `track` — 프레임마다

```json
{
  "type": "track",
  "seq": 128,
  "captured_at": 1759479001234,
  "state": "tracking",
  "run_id": "r-9a1",
  "track_id": "t3",
  "generation": 2,
  "target": "안경",
  "box": { "x": 0.40, "y": 0.31, "width": 0.22, "height": 0.14 }
}
```

- `state`: `idle`(추적하지 않음) | `acquiring` | `tracking` | `occluded` | `lost` | `unavailable`.
- `box`는 `state == "tracking"`일 때**만** 있습니다.
- `captured_at`은 클라이언트가 보낸 값을 그대로 돌려준 것입니다. 박스의 나이는 `Date.now() - captured_at`입니다.
- `lost`가 되면 서버가 알아서 다시 대상을 잡거나 완료 확인으로 넘어갑니다. 클라이언트는 다음 `state`를 기다리면 됩니다.

### 6.2 `state` — 화면 스냅샷

무언가 바뀔 때마다 **전체를** 보냅니다. `rev`는 세션 안에서 계속 증가합니다. 클라이언트는 자기가 가진 것보다 `rev`가 큰 것만 반영하세요.

```json
{
  "type": "state",
  "rev": 42,
  "phase": "running",
  "core_mode": "classic",
  "plan": {
    "plan_id": "…",
    "revision": 1,
    "target": "안경",
    "goal_when": "안경이 닫힌 케이스 안에 있다",
    "steps": [
      { "id": "s1", "say": "양손으로 안경다리를 잡으세요", "done_when": "두 손이 안경다리를 잡고 있다",
        "commands": [ { "kind": "focus", "anchor": "target", "pad": 0.15 },
                      { "kind": "action", "anchor": "target", "action": "grasp", "direction": "none" } ] }
    ]
  },
  "partial": null,
  "clarification": null,
  "step_index": 0,
  "steps_skipped": [],
  "steps_user_done": [],
  "pending": { "stage": "follow", "trigger": "heartbeat", "since": 1759479001500 },
  "basis_age_s": null,
  "notice": null,
  "budget_notice": null,
  "completion": "none",
  "needs_reselect": false,
  "replan_blocked_reason": null,
  "talk": null,
  "talk_pending": false,
  "talk_blocked_reason": null,
  "overlay": {
    "binding": { "anchor_id": "a1", "run_id": "r-9a1", "track_id": "t3", "generation": 2 },
    "commands": [ { "kind": "focus", "pad": 0.15 }, { "kind": "action", "action": "grasp", "direction": "none" } ],
    "warn": null,
    "motion_key": "r-9a1|1|s1|a1|t3|2"
  },
  "error": null
}
```

| 필드 | 뜻 / 화면 처리 |
|---|---|
| `phase` | `idle` → `planning` ↔ `clarifying` → (`reviewing` →) `running` → `completed`, 또는 `error`. 질문/검토 중에는 추적·판정이 시작되지 않습니다 |
| `core_mode` | 이 실행의 코어 `"classic"` / `"sequential"` / `"graph"`. 시작 선택과 재연결 표시를 구분해 서버 값을 따릅니다(§15.13) |
| `plan` | 계획 전체. `status`, `approved_revision`, `materials`, `research_sources`와 아래 단계 필드를 포함합니다(§15). 최대 `limits.plan_steps_max`개(이 후보 16), id `s1`~`s16`, say/done_when ≤60자, details ≤600자. draft는 아직 승인되지 않은 초안입니다 |
| `partial` | `planning` 중에 먼저 도착한 일부 결과 `{target?, first_say?}`. 계획이 확정되면 `null`이 됩니다. 화면에 먼저 띄워 주면 체감 지연이 줄어듭니다 |
| `clarification` | 서버가 화면만으로는 판단이 안 된다고 할 때 사용자에게 보여 줄 질문 문장 |
| `clarification_id` | `clarifying`에서 현재 질문과 함께 제공. `plan_answer`에 그대로 반환; 이전 질문의 답은 거절 |
| `research_sources` | 조사한 출처 `{url,title,summary}` 최대 6개. 참고 링크이며 요약 내용의 정확성이나 작업 완료를 보증하지 않음 |
| `step_index` | 지금 화면에 보여 줄 단계 (0부터 시작) |
| `steps_skipped` / `steps_user_done` | 체크리스트 표시용. 건너뛴 단계 / 사용자가 말로 "했다"고 한 단계 |
| `steps_done` / `steps_ready` | graph의 수락된 완료 사실 / 지금 행동이 허용된 단계 id 목록. 서로 같은 의미가 아니며 단계 번호로 추론하지 않습니다 |
| `step_statuses` | graph의 최신 관측: 단계 id → `yes`/`no`/`unsure`. 수락된 과거 사실과 분리됩니다 |
| `pending_required_checks` | 미확인 절차 검사. 목표 달성 뒤에도 표시하며 완료나 안전 승인을 위조하지 않습니다 |
| `pending` | 서버가 지금 판단 중인 호출. "확인 중…" 같은 표시에만 쓰세요 |
| `basis_age_s` | 값이 있으면 "N초 전 화면 기준"이라는 보조 문구를 붙입니다 (판단에 쓴 화면이 지금과 다를 수 있다는 뜻) |
| `notice` | `{code, text}`. 짧은 안내. 다음 단계로 넘어가거나 8초가 지나면 서버가 지웁니다 |
| `budget_notice` | `{text, until}`. 호출 예산을 다 써서 잠시 기다리는 중이라는 안내 |
| `completion` | `none` → `checking`(완료로 보여서 한 번 더 확인 중) → `confirmed`(완료 버튼 보이기) → `user_confirmed` |
| `needs_reselect` | `true`이면 "대상이 맞지 않아 보임" 안내와 함께 직접 지정 UI(§5.1)를 권하세요 |
| `replan_blocked_reason` | 재계획 한도에 걸렸을 때의 이유 문장 |
| `talk` | 마지막 대화 `{utterance, reply, spoken, at}`. `reply`(최대 400자)는 화면에, `spoken`(최대 60자)은 음성용입니다 |
| `talk_pending` | `true`이면 입력창과 마이크를 잠그세요. 대화는 한 번에 하나만 보낼 수 있습니다 |
| `talk_blocked_reason` | 대화 요청이 거절된 이유 (예산 등) |
| `overlay` | §8 |
| `paused` | 실행이 사용자에 의해 멈춰 있음(`run_pause`). 이때는 판정·단계 진행이 없습니다 |
| `proposal` | 실행 중 서버가 제안한 변경안(§15.6). `null`이 아니면 화면에 보여 주고 `proposal_accept`/`proposal_reject`를 받으세요. 현재 계획은 그대로 유효합니다 |
| `blocked` | 지금 단계로 갈 수 없는 이유(§15.4). `{step_id, requires, reason}`. 필수 선행 단계가 끝나지 않았거나 필수 검사를 건너뛸 수 없을 때 채워집니다 |
| `error` | `{code, message, retryable}`. `phase == "error"`일 때만 있습니다 |

### 6.3 `say` — 읽을 문장

```json
{ "type": "say", "line_id": "L17", "text": "2단계. 케이스 뚜껑을 여세요", "mode": "replace", "audio": "pending" }
```
- `mode: "replace"`: 지금 읽고 있는 것을 끊고 이 문장을 읽습니다. `"append"`: 지금 것이 끝난 뒤 이어서 읽습니다.
- `audio`: `"pending"`(서버 오디오가 뒤따라 옴) | `"none"`(서버 오디오 없음, 바로 브라우저 TTS로 읽기).
- 어떤 문장을 언제 읽을지는 서버가 정합니다. 클라이언트는 화면 변화를 보고 직접 문장을 만들지 마세요.

### 6.4 오디오 (바이너리)

```json
{ "t": "audio", "line_id": "L17", "mime": "audio/ogg; codecs=opus", "duration_ms": 2380 }
```
payload는 오디오 파일 전체입니다(Opus를 넣은 Ogg). Safari에서 Ogg Opus 재생이 안 되면 `hello`에 `"audio_accept": ["audio/mp4"]`를 넣어 보내세요. 서버가 AAC로 보냅니다.

### 6.5 `capture_hi`

```json
{ "type": "capture_hi", "req_id": "h7" }
```
§5.1의 고화질 프레임으로 답합니다. 2초 안에 오지 않으면 서버는 일반 프레임을 대신 씁니다.

### 6.6 `error`

```json
{ "type": "error", "code": "talk_busy", "message": "이전 질문에 답하는 중입니다", "retryable": true, "retry_after_ms": null, "fatal": false }
```
`fatal: true`이면 서버가 곧 연결을 닫습니다. 코드 목록은 §10에 있습니다.

### 6.7 `bye`

닫기 직전에 이유를 알려 줍니다. `{ "type": "bye", "reason": "session_expired" }`

---

## 7. 음성 출력 규칙 (서버 TTS 우선, 브라우저 TTS 대체)

1. `say`를 받으면 `line_id`별로 대기열에 넣습니다. `mode`에 따라 지금 읽는 것을 끊거나 뒤에 붙입니다.
2. 그 문장을 읽을 차례가 되면:
   - 같은 `line_id`의 오디오가 이미 와 있으면 그것을 재생합니다.
   - `audio == "none"`이거나, `prefs.tts == "browser"`이거나, `health.tts_ready == false`이면 바로 `speechSynthesis`(ko-KR)로 읽습니다.
   - 그 밖의 경우에는 `say`를 받은 시점부터 `limits.tts_wait_ms`까지 오디오를 기다립니다. 그때까지 오지 않으면 `speechSynthesis`로 읽고, 그 `line_id`의 오디오가 나중에 오면 **버립니다**. 같은 문장을 두 번 읽으면 안 됩니다.
3. 마이크가 열려 있는 동안에는 출력을 멈추고, 닫히면 다시 이어서 읽습니다.
4. `voice_out`이 꺼져 있어도 `say`는 계속 옵니다. 읽지 말고 버리세요. 다시 켰을 때 밀린 문장을 몰아서 읽지 않기 위해서입니다.
5. 다 읽거나 건너뛸 때마다 `played`를 보냅니다.

---

## 8. 오버레이 그리기

박스 위에 그리는 안내는 `state.overlay`와 매 프레임의 `track`으로 클라이언트가 직접 그립니다. 서버는 좌표를 만들어 내지 않습니다. 모든 그림은 **추적 박스에 상대적**입니다.

**그릴 조건** (rAF마다 확인하고, 하나라도 어긋나면 박스 위 오버레이를 숨깁니다):
- 가장 최근에 받은 `track.state == "tracking"`
- `track.run_id`, `track.track_id`, `track.generation`이 `overlay.binding`과 같음
- `Date.now() - track.captured_at ≤ limits.box_max_age_ms`
- `prefs.visuals == true`

**명령** (`overlay.commands`, 최대 2개, `action`은 최대 1개):

| `kind` | 필드 | 그리는 것 |
|---|---|---|
| `focus` | `pad` (0..0.5, 박스 크기에 대한 비율) | 박스를 둘러싼 강조 고리 |
| `label` | `text` (≤40자) | 박스 위에 붙는 말풍선 |
| `action` | `action`, `direction` | 박스 기준의 동작 애니메이션, 그리고 그 동작의 짧은 한국어 라벨 말풍선. `focus`가 없어도 기본 pad 0.15로 고리를 같이 그립니다 |

`action` 값과 허용되는 `direction`:

| action | direction |
|---|---|
| `move`, `pull`, `push` | `up` `down` `left` `right` (화면에 보이는 방향 기준) |
| `rotate` | `clockwise` `counterclockwise` |
| `screw` | `tighten` `loosen` |
| `press` `grasp` `open` `close` `insert` `remove` `attach` `fold` `place` `fit` `hold` `flip` | `none` |
| `align` `connect` `disconnect` `bend` | `none` — 정렬·연결·분리·구부리기 도식. 실제 핀 위치·극성·전압을 뜻하지 않으며 “동작 예시 · 실제 핀 위치 아님”을 표시합니다 |

- `motion_key`가 바뀔 때만 애니메이션을 처음부터 다시 시작하세요. 같은 키라면 박스를 따라 움직이기만 하면 됩니다.
- `overlay.warn == "needs_reselect"`이면 박스를 주황색 점선으로 그리고 "대상이 맞지 않아 보임"을 표시합니다.
- 오버레이를 끈 상태(`visuals == false`)에서는 현재 단계의 `say`를 큰 글씨로 보여 주세요.
- 참고 구현: 현재 데모의 `web/src/intent/actionMotion.ts`, `web/src/components/AnchoredGuidanceOverlay.tsx`(저장소 `aisw-hybrid-talk`). 동작별 모션 디자인은 디자인 캔버스에 따로 있습니다.

추적 박스 자체(`track.box`)는 디버그용으로 얇게 그려도 됩니다. 사용자에게 보여 줄 필요는 없습니다.

---

## 9. 끊김과 재연결

- 연결이 끊겨도 세션과 가이드는 `limits.resume_grace_ms`(30초) 동안 **일시정지 상태로** 유지됩니다.
- 재연결할 때는 `hello`에 마지막으로 반영한 `rev`를 `resume_rev`로 넣어 보냅니다. 서버는 `ready` 다음에 최신 `state`를 보냅니다.
- 재연결 후 프레임 `seq`는 0부터 다시 시작합니다. 추적은 서버가 계획의 대상으로 다시 잡으므로, 처음 몇 프레임 동안은 `acquiring`일 수 있습니다.
- 대기 중이던 `say`와 오디오는 다시 보내지 않습니다.
- 재연결 간격은 1초, 2초, 4초…로 늘리고 최대 8초로 합니다. 유예 시간이 지나면 서버는 `bye{reason:"session_expired"}`를 보내거나, 연결 단계에서 `4002`로 거절합니다. 그때는 새 세션을 만드세요.

| close code | 뜻 | 클라이언트 동작 |
|---|---|---|
| 1000 | 정상 종료 | — |
| 1012 | 서버 재시작 | 재연결. 실패하면 새 세션 |
| 4001 | `hello` 인증 실패 | 새 세션 |
| 4002 | 세션 만료 | 새 세션 |
| 4003 | 같은 세션에 다른 연결이 붙음 | 재연결하지 말 것 (다른 탭) |
| 4008 | 프로토콜 위반 (`hello` 누락, 잘못된 JSON, 크레딧 위반 반복) | 버그. 로그를 남기고 새 세션 |
| 4009 | 메시지가 너무 큼 | 프레임 크기 확인 |

---

## 10. 오류 코드

| code | 어디서 | 뜻 | 클라이언트 처리 |
|---|---|---|---|
| `invalid_access_code` | REST | 코드 불일치 | 다시 입력받기 |
| `rate_limited` | REST, WS | 너무 잦은 요청 | `retry_after_ms` 후 재시도 |
| `capacity` | REST | 동시 세션 상한 | 잠시 후 다시 시도하라고 안내 |
| `consent_required` | WS | `consent_ai`가 true가 아님 | 동의 UI 보이기 |
| `not_running` | WS | 가이드가 없는데 `talk`나 `follow_now` 등을 보냄 | 버튼 상태 점검 |
| `plan_in_flight` | WS | 계획을 세우는 중에 다른 요청을 보냄 | 무시 |
| `talk_busy` | WS | 이전 대화에 아직 답하는 중 | 입력을 잠가 두었다면 생기지 않음 |
| `talk_budget` | WS | 대화 호출 예산 소진 | `talk_blocked_reason`을 표시 |
| `frame_invalid` | WS | JPEG 해석 실패, 헤더 오류, 박스 범위 오류 | 인코더 점검 |
| `frame_too_large` | WS | `*_max_bytes` 초과 | 품질 낮추기 |
| `tracker_unavailable` | WS | 추적 서버 문제 | `state.error`를 따름 |
| `provider_error` | WS | 계획/판정 모델 호출 실패 (서버가 이미 재시도한 뒤) | `state.error`를 따름 |
| `not_reviewing` | WS | 검토 중이 아닌데 `plan_edit`/`plan_approve`/`plan_discard`를 보냄 | 버튼 상태 점검 |
| `plan_locked` | WS | 승인된 뒤에 계획을 고치려 함 | 새로 시작하거나 초안부터 다시 |
| `invalid_edit` | WS | 수정 내용이 규칙에 어긋남(단계 수 상한, 알 수 없는 `step_id`, 선행 조건 순환, `step_ack`을 쓸 수 없는 단계 등) | `message`를 보고 고치기 |
| `required_check` | WS | 아직 완료 자격이 없는 `confirm_done` 또는 필수 절차를 넘는 행동·확인 | 서버 `blocked`/`pending_required_checks`를 표시. graph의 목표 증거와 절차 이행은 §15.13처럼 분리 |
| `no_proposal` | WS | 변경안이 없는데 `proposal_accept`/`proposal_reject`를 보냄 | 무시 |
| `plan_changed` | WS/state | 변경안을 거절했는데 상위가 이미 계획을 바꿔 되돌릴 수 없음(§15.6). 실행을 끝냅니다 | 새로 시작해 검토부터 |
| `unsupported_protocol` | WS | `hello.protocol`이 맞지 않음 | 클라이언트 업데이트 |

진행 중에 생긴 일시적인 오류(모델 지연, 재시도 등)는 서버가 처리하고 `notice`로만 알립니다. 클라이언트가 직접 재시도할 일은 없습니다.

---

## 11. 한 번의 가이드 흐름 (예)

```
C  POST /v1/sessions                       → {session_id, token, live_url}
C  WS open, hello                          ← ready, state{phase:idle}
C  prefs{voice_out:true, tts:"server", visuals:true}
C  frame seq=0,1,2…                        ← track{state:"idle"} 각각
C  start{goal:"안경을 벗어서 케이스에 넣기"}
                                           ← state{phase:"planning"}
                                           ← say{"계획을 세우는 중입니다."}
                                           ← capture_hi{h1}
C  frame{hi_req:"h1"} (1024px)
                                           ← state{partial:{target:"안경"}}         (~1.2초)
                                           ← state{partial:{first_say:"…"}}        (~1.5초)
                                           ← state{phase:"running", plan, step_index:0, overlay}
                                           ← say{"1단계. …", audio:"pending"} → 오디오(바이너리)
C  frame…                                  ← track{state:"acquiring"} → track{state:"tracking", box}
   (서버가 진행 판정, 단계 전환, 완료 확인을 수행)
                                           ← state{step_index:1, overlay…}, say{"2단계. …"}
C  talk{"뚜껑이 안 열려요"}                 ← state{talk_pending:true}
                                           ← state{talk:{reply, spoken}}, say{spoken}
                                           ← state{completion:"checking"} → state{completion:"confirmed"}
C  confirm_done                            ← state{completion:"user_confirmed", phase:"completed"}
C  DELETE /v1/sessions/{sid}
```

괄호 안의 시간은 현재 데모를 테일넷에서 잰 값입니다(계획 스트리밍: 대상 1.12초, 첫 문장 1.54초, 전체 2.39초). 외부 망에서는 더 늘어날 수 있습니다.

---

## 12. 서버 쪽 한도 (참고)

클라이언트가 지킬 필요는 없습니다. 서버가 알아서 지키고, 걸리면 `budget_notice`나 `notice`로 알려 줍니다.

- 유료 모델 호출: 세션당 분당 20회, 가이드 한 번당 120초 동안 30회, 호출 사이 최소 간격 1.2초
- 재계획: 가이드 한 번에 최대 3회, 15초 간격
- 대화: 한 번에 하나, 대기열 없음
- 동시 세션 32개, 세션 생성은 IP당 분당 5회, 유휴 15분 후 만료
- Plan 모드(§15): 이 후보는 계획 16단계, 단계당 직접 선행 조건 15개·논리적 대상 3개, 자료 4개(자료 본문 각 4000자)

---

## 13. 미정이거나 검증이 필요한 것

| 항목 | 현재 상태 |
|---|---|
| 공개 입구 | `https://sangye.kr/synoptics/api` 확정 (Cloudflare → nginx). 공개되는 것은 `/v1/` API뿐이고, 스펙 문서와 테스트 클라이언트는 테일넷 전용 |
| 640px 스트림에서의 추적 정확도 | **아직 측정하지 않음.** 현재 데모는 최대 1920px로 보냄. 측정 결과에 따라 `stream_long_side`가 바뀔 수 있음 |
| `box_max_age_ms` | 현재 데모는 300ms. 외부 망 왕복을 고려해 500ms로 잡았으며, 실측 후 조정 |
| 동시 사용자 수 | 추적기는 GPU 하나에서 프레임당 약 20ms. 사용자당 초당 10프레임이면 대략 5명이 한계로 추정 |
| TTS 구성 | 제출 기본은 브라우저·기기 TTS. 서버 음성 모델·가중치 미포함 |
| 서버 구현 | 최신 계획·그래프 결합 소스 제공. 운영 배포와 별개이며 현재 서비스 상태는 각 `/v1/health`를 따름 |
| Plan 모드 검증 | 제출 소스의 Live 246개 회귀 및 실제 REST/WebSocket→백엔드 통제 입력 스모크 통과. 모델 품질·외부 PWA 인수와 구분 |
| TTS 상태 | 제출 기본 `health.tts_ready:false`, `health.tts_voice:null`. 브라우저 음성 사용 |

---

## 14. 구현 노트 (현재 서버 기준, 위 본문을 보충)

| 항목 | 동작 |
|---|---|
| `health.engine` | `"mock"`(각본) 또는 `"real"`(실제 상위 엔진). 현재 응답을 기준으로 표시합니다 |
| REST 오류 본문 | `{"error": {"code": "...", "message": "..."}}`. 본문 형식 오류는 400 `invalid_request` |
| `DELETE /v1/sessions/{sid}` 오류 | 401 `invalid_token`, 404 `session_not_found`(없거나 이미 종료) |
| 세션 생성 한도 | 틀린 접근 코드 시도도 IP당 분당 5회에 포함. `Retry-After`는 초 단위 |
| 알 수 없는 `type`·잘못된 필드 | `error{code:"invalid_message"}`(비치명). JSON이 아니거나 `type`이 없으면 close 4008 |
| 클라이언트가 보낸 모르는 필드 | 무시합니다(하위 호환) |
| 크레딧 위반 (응답 전에 일반 프레임을 또 보냄) | 그 프레임은 버리고 `error{code:"rate_limited"}`(비치명). 20회 누적이면 4008 |
| 거절된 프레임 | `track` 응답이 없습니다. `error`에는 `seq` 필드가 없으므로, 2초 규칙(§5.1-4)대로 다음 프레임을 보내세요 |
| 헤더 `w`/`h`와 실제 JPEG 크기 불일치 | `frame_invalid` |
| 고화질 프레임(`hi_req`) | `track` 응답 없음, 크레딧과 무관. 모르는 `hi_req`나 재사용은 `frame_invalid` |
| `track`의 필드 | 키는 항상 있고, 해당 없으면 `null`(`run_id`, `track_id`, `generation`, `target`, `box`) |
| `select_box` | 같은 `run_id`, `generation`+1, 새 `track_id`. 가이드가 없을 때(`idle`)는 무시 |
| `state.overlay` | 처음 `tracking`이 되기 전, 그리고 `phase`가 `running`이 아닐 때는 `null` |
| `pending.stage` / `trigger` | stage는 `plan`·`follow`·`confirm`·`talk`, trigger는 자유 문자열(표시용) |
| `prefs` | 부분 갱신. 빠진 필드는 이전 값 유지 |
| `talk`의 동의 | 마지막 `start`의 `consent_ai`를 따릅니다. `talk`에 `consent_ai:false`를 넣으면 거절 |
| `confirm_done`을 `confirmed`가 아닐 때 보냄 | `not_running` |
| `bye.reason` | `session_expired`, `session_ended`, `replaced`, `auth_failed`, `protocol_violation`, `unsupported_protocol`, `message_too_large`, `server_restart` |
| 세션 삭제 중 연결이 열려 있음 | `bye{session_ended}` 후 close 1000. 삭제·만료된 sid로 다시 붙으면 4002, 모르는 sid는 4001 |
| WebSocket Origin 거절 | 핸드셰이크가 HTTP 403 |

---

## 15. Plan 모드 — 검토한 절차로 실행 (2026-10-03 추가)

지금까지의 흐름(§11)은 `start` 직후 곧바로 실행합니다. **Plan 모드**는 계획을 먼저 만들고 **사용자가 검토·승인한 뒤에 실행**합니다. 매뉴얼·작업 지침을 자료로 넣을 수 있고, 단계마다 완료 기준·확인 방법·근거·필수 여부를 가집니다.

- 시작: `start{goal, context?, consent_ai:true, plan_mode:true, plan_model?, core_mode?, materials?:[…], reference_images?:[…]}`. `plan_mode:false`는 초기 검토만 생략합니다. 기본 모델은 `deepseek:high`, 코어는 `classic`입니다. 순차·그래프 코어의 실행 중 proposal 경계는 초기 검토를 생략해도 유지됩니다.
- 계획 프로필: `start.plan_model`이 이 run의 상위 계획 프로필을 고릅니다 — `deepseek:high`(기본값) 또는 `astra:high`. 닫힌 집합이고 그 밖의 값은 `invalid_message`로 거절됩니다. **검토 경계(`plan_mode`)와 독립**이며, 실행 중에는 바꿀 수 없습니다(바꾸려면 새 `start`). 실제 엔진은 검토 여부와 무관하게 이 값을 상위 `/api/guide/plan`에 그대로 보내고, 상위는 그 선택을 계획에 저장해 확인·재계획·대화가 다른 프로필로 새지 않게 합니다. 두 프로필 모두 높은 추론 강도와 180초 기한을 쓰므로 최초 계획·확인·대화 상위 요청은 190초까지 기다립니다(반복 follower 기한은 그대로). 준비 여부는 health의 `plan_models`(프로필별)·`plan_ready`(집계)로 알립니다. 클라이언트는 모델명·추론 강도·API 키를 지정하지 않습니다. 상위 `plan_model` 지원 판본과 함께 배포해야 하며, 인증·응답 실패 시 다른 모델이나 Codex 로그인으로 자동 대체하지 않습니다.
- 실행 전 경계: `phase:"reviewing"` 동안 서버는 **추적을 시작하지 않고**(모든 프레임에 `track{state:"idle"}`) 판정 호출도 하지 않습니다. `overlay`는 `null`입니다.
- 승인은 **그 `revision`을 고정**합니다. 실제 엔진은 승인할 때 그 계획을 상위에 올려(`POST /api/guide/plan/approve`) 상위 판본도 함께 움직입니다. 그래서 초안을 고쳐도 클라이언트가 보는 판본과 상위 판본이 갈라지지 않습니다. 승인 뒤에는 계획을 고칠 수 없고, 고치려면 `plan_edit`이 `plan_locked`로 거절됩니다.
- 판단은 여전히 전부 서버가 합니다. 클라이언트는 `state`를 그리고, 검토 화면에서 사용자의 입력(수정·승인·일시정지·확인)만 전달합니다.

### 15.1 흐름

```
start{plan_mode:true} → planning → reviewing ──plan_edit──▶ reviewing (revision+1)
                                        │
                                        ├──plan_approve──▶ running (승인한 revision 고정)
                                        └──plan_discard──▶ idle

running ──run_pause──▶ running+paused:true ──run_resume──▶ running
running ──(정체·재계획 필요)──▶ running+proposal ──proposal_accept──▶ running (revision+1, 승인)
                                                └──proposal_reject──▶ running (그대로)
running ──필수 단계 모두 완료──▶ completion:"checking" → "confirmed" ──confirm_done──▶ completed
```

`plan_mode`는 위 흐름의 **검토 경계만** 정합니다. 어느 상위 프로필로 계획·확인·재계획·대화를 할지는 별도의 `start.plan_model`(기본 `deepseek:high`)이 정하고, 그 선택은 그 run에 고정됩니다. 검토를 건너뛰는 즉시 안내 흐름도 같은 `plan_model`을 씁니다.

### 15.2 계획의 구조

`state.plan`은 계획 전체입니다. Plan 모드에서 `status`는 `draft`(검토 중) 또는 `approved`(승인됨)이고, `approved_revision`이 승인한 판본입니다. 즉시 안내 흐름의 계획은 `status:"approved"`, `approved_revision:null`입니다.

```json
{
  "plan_id": "8f3c…",
  "revision": 2,
  "status": "draft",
  "approved_revision": null,
  "target": "책",
  "goal_when": "책이 완충재와 함께 상자에 담겨 봉함되어 있다",
  "materials": [
    { "id": "m1", "title": "포장 지침", "version": "v3", "chars": 1840, "added_at": 1759479000000 }
  ],
  "steps": [
    { "id": "s1", "say": "완충재를 상자 바닥에 까세요", "done_when": "상자 바닥이 완충재로 덮여 있다",
      "commands": [ { "kind": "focus", "anchor": "target", "pad": 0.15 } ],
      "check": "visual", "required": false, "requires": [], "targets": ["상자", "완충재"],
      "evidence": { "material_id": "m1", "version": "v3", "locator": "3.1", "quote": "바닥면은 완충재로 먼저 덮는다" } },
    { "id": "s4", "say": "봉함 전에 상자 안을 화면에 보여 주세요", "done_when": "지침에 맞는 배치가 확인됨",
      "commands": [ { "kind": "label", "anchor": "target", "text": "내부 확인" } ],
      "check": "user", "required": true, "requires": ["s3"], "targets": ["상자 내부"],
      "evidence": { "material_id": "m1", "version": "v3", "locator": "4.2" } }
  ]
}
```

단계(`GuideStep`)의 새 필드 — 나머지(`id`, `say`, `done_when`, `commands`)는 §6.2와 같습니다:

| 필드 | 값 | 뜻 |
|---|---|---|
| `check` | `"visual"` \| `"user"` \| `"measure"` | 완료를 무엇으로 판단하는지. 기본 `"visual"`(화면) |
| `required` | bool | **필수 검사 지점**. 건너뛸 수 없습니다. 기본 `false` |
| `details` | str ≤600 또는 null | 읽기 쉬운 상세 안내. 생략한 수정은 보존, 명시적 null은 삭제 |
| `condition_kind` | `"state"` \| `"event"` | 기본 state: 현재 유지되는 조건. event는 과거 발생 자체가 목표인 경우이며 가려진 부품을 event로 바꾸지 않습니다 |
| `goal_required` | bool | 기본 true: 원래 목표의 조건. false인 안전·준비 조건도 required=true이면 행동·승인을 계속 차단합니다 |
| `requires` | `[step_id]` (≤15) | 실제 직접 선행 조건. 표시 순서가 아니며 독립 단계끼리 불필요한 연결을 만들지 않습니다 |
| `targets` | `[str]` (≤3, 각 ≤40자) | 이 단계가 다루는 **논리적 대상**(부품·상자·도구). 화면에서 추적하는 `plan.target`과 별개이며 좌표가 아닙니다 |
| `evidence` | `{material_id, version?, locator?, quote?}` \| null | 이 단계의 근거가 된 자료와 위치. 살아남는 조건은 아래 「근거가 살아남는 조건」 |

- 이 후보의 단계 수 상한은 `limits.plan_steps_max`(16), id는 `s1`~`s16`입니다. 운영 서버 적용 여부는 별도로 확인합니다.
- `requires`는 같은 계획 안의 단계 id만 가리킬 수 있고, 자기 자신·순환은 거절됩니다.
- 계획에는 **논리적 대상만** 저장합니다. 화면 좌표·박스는 저장하지 않고, 실행을 시작하거나 재개할 때 최신 화면에서 다시 잡습니다.

### 15.3 자료 입력 (`materials`)

```json
{ "type": "start", "goal": "책을 상자에 포장하기", "consent_ai": true, "plan_mode": true,
  "plan_model": "astra:high",
  "materials": [ { "title": "포장 지침", "version": "v3", "text": "3.1 바닥면은 완충재로 먼저 덮는다. …" } ] }
```

- 최대 `limits.materials_max`(4)개, 자료 본문은 `limits.material_text_max_chars`(4000)자까지입니다. 문서 추출(PDF·캡처 텍스트화)은 클라이언트가 하고, 서버는 **추출된 텍스트**를 받습니다.
- 서버는 자료를 id(`m1`…)와 함께 세션에 보관하고, `state.plan.materials`에 되돌려 줍니다(본문 제외, `chars`만). 단계의 `evidence.material_id`가 이 id를 가리킵니다.
- **근거가 살아남는 조건(상위가 강제, 2026-10-04 실측).** 초안을 만드는 쪽이 쓴 근거는 다음을 모두 만족할 때만 계획에 남고, 아니면 **근거만 조용히 지워집니다**(계획은 설치되고, `evidence`는 null이 됩니다):
  - `material_id`가 그 요청의 자료 id(`m1`…`mN`) 중 하나일 것,
  - `version`을 쓴 경우 자료의 `version`과 같을 것(자료에 version이 없으면 근거에 쓸 수 없음),
  - `quote`를 쓴 경우 공백 정규화 후 자료 본문의 **실제 부분 문자열**이고 120자 이하일 것(길다고 잘라 주지 않고 버립니다),
  - `locator`가 80자 이하일 것.
  그래서 "그럴듯한 인용"은 통과하지 못합니다 — 자료에 실제로 있는 문장만 남습니다. 클라이언트가 `plan_edit`/승인으로 보내는 단계의 근거도 같은 규칙으로 검사됩니다(어긋나면 요청 자체가 거부됩니다).
- 긴 지침을 `context`에 억지로 넣지 마세요. `context`는 작업 조건(장소·재료 등)이고, 자료는 `materials`입니다.

### 15.4 단계 진행 규칙 (서버가 강제)

1. **승인 전에는 아무것도 진행되지 않습니다.** `reviewing` 중 `follow_now`·`replan_now`·`talk`은 `not_running`입니다.
2. **필수 검사는 건너뛸 수 없습니다.** 어떤 근거로도 `required:true` 단계를 `skipped`로 넘기지 않습니다. 건너뛰려 하면 `state.blocked`에 이유를 쓰고 `notice{code:"required_check"}`를 보낸 뒤 **그 단계에 머무릅니다**.
3. **선행 조건은 행동 권한을 제한합니다.** classic/sequential에서는 단계 진행을 막습니다. graph에서는 `step_index`가 관찰 초점일 뿐이며 `steps_ready`에 없는 단계의 동작·상세 지시를 실행하지 않습니다.
4. **`check:"user"`·`"measure"`는 화면 판정만으로 완료되지 않습니다.** 사용자의 `step_ack{step_id}`가 필요합니다. graph는 `required:true`인 visual 검사도 명시적 확인을 받습니다. 나머지 visual ack는 `invalid_edit`입니다.
5. **완료 판정**: classic/sequential은 필수 검사 완료를 요구합니다. graph는 원래 목표(`goal_required`)의 증거와 안전 절차 이행을 분리합니다. 이미 달성된 물리적 목표를 미확인 준비 상태 때문에 거짓으로 만들지 않되, `pending_required_checks`를 계속 표시합니다. `skipped`/`pending`은 done이 아니며 목표 확인이 전기 안전 승인을 뜻하지 않습니다.
6. 머무는 이유는 `state.blocked`와 함께 `notice`로도 한 번 알립니다. 코드는 `required_check`(필수 검사),
   `user_check`(사용자 확인 대기), `prerequisite`(선행 조건)입니다.
7. **안 보임은 없음이 아닙니다.** 확인할 수 없으면 머무릅니다. 없는 것으로 단정하거나 자동으로 넘어가지 않습니다.
8. 진행 상태(`steps_skipped`, `steps_user_done`, `step_index`, `blocked`)는 항상 서버가 만들고, 끊겼다 다시 붙어도 그대로입니다.

### 15.5 상태 전이

| `phase` | `plan.status` | 뜻 | 허용되는 클라이언트 요청 |
|---|---|---|---|
| `idle` | — | 가이드 없음 | `start` |
| `planning` | — | 계획 생성 중 | `stop` |
| `clarifying` | — | 필수 정보 한 가지를 사용자에게 질문 중 (추적·판정 없음) | `plan_answer`, `start`, `stop` |
| `reviewing` | `draft` | 초안 검토 중 (추적·판정 없음) | `plan_edit`, `plan_approve`, `plan_discard`, `stop` |
| `running` | `approved` | 실행 중 | `talk`, `follow_now`, `replan_now`, `step_ack`, `run_pause`, `stop`, (`proposal_*`) |
| `running` + `paused:true` | `approved` | 일시정지 | `run_resume`, `stop` |
| `completed` | `approved` | 완료(사용자 확인까지) | `start`, `stop` |
| `error` | — | 오류 | `start`, `stop` |

### 15.6 변경안 (`state.proposal`)

실행 중 계획을 바꿔야 하면(정체, `replan_now`, 대화) 서버는 계획을 **곧바로 바꾸지 않고** 변경안으로 올립니다.

```json
{ "plan_id": "8f3c…", "revision": 3, "reason": "완충재 위치를 확인할 수 없어 단계를 나눴어요",
  "goal_when": "…", "steps": [ … ] }
```

- `state.proposal`이 채워져 있는 동안 **현재 계획은 그대로 유효**합니다. 화면은 무엇이 바뀌는지 보여 주고 두 버튼을 띄우세요.
- `proposal_accept` → 제안을 설치하고 `revision`+1, `approved_revision`도 함께 갱신합니다.
- **변경안이 떠 있는 동안 서버는 판정·진행을 멈춥니다.** 승인되지 않은 절차로 실행하지 않기 위해서입니다(frame은 계속 받고 추적 상자도 보여 줄 수 있지만, 단계 진행·판정 호출은 없습니다). `paused`와 다른 상태이므로 화면에는 `proposal`을 띄우고 두 버튼을 보여 주세요.
- `proposal_reject` → 제안을 버립니다. 실제 엔진은 상위의 `POST /api/guide/plan/revert`로 **직전 계획을 되돌려** 실행을 그대로 이어 갑니다(되돌린 계획은 새 판본을 받습니다). 상위가 되돌리기를 거절하면(기록 없음·판본 불일치) 그때만 `error{code:"plan_changed"}`로 실행을 끝냅니다 — 이 코드는 **예외 경로**이고 정상 경로가 아닙니다. mock 엔진은 각본 안에서 되돌립니다.
- 변경안이 없는데 `proposal_accept`/`proposal_reject`를 보내면 `no_proposal`입니다.

### 15.7 일시정지·재개

- `run_pause` → `state.paused:true`. 판정·단계 진행이 멈추고 추적도 멈춥니다(`track{state:"idle"}`).
- `run_resume` → 재개. 재개하면 최신 화면에서 대상을 다시 잡습니다(처음 몇 프레임은 `acquiring`).
- 일시정지 중에는 `follow_now`·`talk`·`replan_now`가 `not_running`입니다.
- 연결이 끊겼다 다시 붙어도(§9) **사용자가 건 일시정지는 풀리지 않습니다**(연결 끊김 때문에 생긴 멈춤만 자동으로 풀립니다).

### 15.8 화면 요구 (최소)

- 검토 화면은 빈 채팅창이 아니라 **작업표**입니다: 단계 목록(순서대로, 필수·선행·확인 방법 표시)과 선택한 단계의 상세(행동·대상·완료 기준·확인 방법·근거), 그리고 `승인하고 실행`·`초안 버리기` 버튼.
- 승인 전에는 카메라 안내 오버레이를 그리지 마세요(`overlay`가 `null`입니다).
- 실행 중에는 현재 단계와 함께 `blocked`(왜 못 넘어가는지)와 `proposal`(무엇이 바뀌는지)을 보여 주세요.
- `plan.status == "draft"`면 화면 어딘가에 "아직 승인되지 않음"을 표시하세요.

### 15.9 예 (mock 엔진)

```
C  start{goal:"책을 상자에 포장하기", plan_mode:true, plan_model:"astra:high",
         materials:[{title:"포장 지침", version:"v3", text:"…"}]}
                                           ← state{phase:"planning"}, say{"계획을 세우는 중입니다."}
                                           ← capture_hi{h1}
C  frame{hi_req:"h1"}
                                           ← state{phase:"reviewing", plan:{status:"draft", revision:1, materials:[m1], steps:[s1…s7]}}
                                           ← say{"계획 초안을 만들었어요. 검토하고 실행을 눌러 주세요."}
C  plan_edit{edits:[{step_id:"s4", required:true, check:"user"}]}
                                           ← state{plan:{revision:2, …}}  say{"계획을 고쳤어요."}
C  plan_approve                            ← state{phase:"running", plan:{status:"approved", approved_revision:2}}
                                           ← track{state:"acquiring"} → track{state:"tracking", box}
   …
                                           ← state{blocked:{step_id:"s4", requires:[], reason:"봉함 전 내부 확인이 필요해요"}}
C  step_ack{step_id:"s4"}                  ← state{steps_user_done:["s4"], step_index:…}
                                           ← state{completion:"checking"} → state{completion:"confirmed"}
C  confirm_done                            ← state{phase:"completed", completion:"user_confirmed"}
C  DELETE /v1/sessions/{sid}
```

### 15.10 아직 정하지 않은 것

| 항목 | 현재 |
|---|---|
| 문서 추출(PDF 등) | 클라이언트가 텍스트로 만들어 `materials.text`로 보냅니다. 서버는 파일을 받지 않습니다 |
| 자료의 영속 저장 | 지금은 세션 메모리에만 있습니다(서버 재시작 시 사라짐). 여러 세션에서 재사용하는 계획 보관함은 다음 단계입니다 |
| 계획 판본 이력 | 서버는 최신 revision만 유지합니다(이력 조회 API 없음) |
| 실제 엔진의 필수 검사·근거 채우기 | 실제 엔진은 상위 계획 호출 결과를 Plan 모드 모양으로 옮깁니다. 2026-10-04 상위 계약이 넓어진 뒤로는 `materials`를 계획 호출에 실어 보내고 `check`·`required`·`requires`·`targets`·`evidence`를 그대로 옮깁니다. 상위가 만들지 않은 필드는 기본값(`visual`·`false`·`[]`·null)으로 두고 **지어내지 않습니다**. 상위가 자료에 없는 `evidence`를 버리는 것도 확인했습니다(교차 검증: 실제 상위 앱 + 실제 클라이언트로 19/19) |
| 상위 승인·되돌리기 엔드포인트 | 상위(데모 백엔드)에 `POST /api/guide/plan/approve`(검토한 계획을 그대로 인정하고 판본을 올림)와 `POST /api/guide/plan/revert`(직전 계획으로 되돌림)를 두었습니다. 실제 엔진은 이 둘로 §15.6을 지킵니다. 상위가 이 엔드포인트를 가진 판본으로 배포돼 있어야 하며, 그 전에는 되돌리기가 실패해 `plan_changed`로 끝납니다 |
| 계획 프로필 선택 | `start.plan_model`(닫힌 두 값, 기본 `deepseek:high`)을 실제 엔진이 계획 요청마다 그대로 보내고, 상위가 계획에 저장해 확인·재계획·대화에 재사용합니다. 상위의 옛 `plan_mode` provider 선택자는 없어졌고, 검토 경계는 Live의 `plan_mode`가 정합니다. 준비 여부는 health `plan_models`(프로필별)·`plan_ready`(집계)로 알리며, 이는 구성 여부일 뿐 프로젝트 권한·잔액이 아닙니다 |
| 상위 계획 단계 상한 | 이 후보는 상위·Live 모두 16단계, id `s1`~`s16`, 직접 선행 조건 최대 15개입니다 |
| 상위 추종 체크리스트 여유 | 상위 follow와 로컬 생성 상한은 800입니다. 16단계 생성 성공률은 별도 실모델 측정 대상이며 잘림은 실패입니다. Clef 선택 출력은 생성 토큰 상한과 별개입니다 |
| 실제 엔진의 자료 활용 | `materials` 원문은 상위 계획 요청의 별도 필드로 보냅니다(`context`에 합치지 않음). 상태의 자료 목록은 제목·판본·글자 수 등 메타데이터만 반환합니다. 상위는 세션 안에 원문을 보존해 재계획에 재사용합니다 |
| 근거(`evidence`) | 초안과 재계획에서 자료 ID·판본·원문 인용을 검증합니다. 근거가 없으면 지어내지 않고, 모델의 잘못된 인용은 버립니다. 원문 일치가 절차의 안전성·의미적 적합성을 보증하지는 않습니다. 클라이언트가 `plan_edit`으로 근거를 작성할 수는 없습니다 |

### 15.11 웹 클라이언트 구현 체크리스트

계획 모델·코어 선택, Plan 자료/승인, 실행 중 변경안과 재연결을 합친 현행 체크리스트는 [§16.7](#client-acceptance)에서 관리합니다.

### 15.12 기존 PWA 이관 안내

화면의 검토/실행 전환만 바꾸는 것이 아니라 서버의 `reviewing` → `plan_approve` → `running` 경계를 따라야 합니다.
자료를 `context`에 합치지 말고 `materials`로 보내세요. 모델·코어 선택과 나머지 UI 변경, 요청 예시, 담당자 인수 범위는 [통합 인계서 §16](#client-handoff)로 합쳤습니다.

### 15.13 실행 코어 선택 (2026-10-04)

이 미배포 후보의 `start.core_mode`는 `classic`(기본), `sequential`, `graph`입니다. 다른 값은 `invalid_message`입니다.
상위 계획 모델·초기 검토 여부와 독립이므로 2모델 × 3코어 × 검토 여부의 12조합입니다. 아래 종전 8조합 실측 기록은 새 12조합 전체 검증이 아닙니다.
값은 run에 고정되며 바꾸려면 새 실행을 시작합니다. 서버는 `state.core_mode`를 항상 보냅니다.
State에는 `plan_model`이 없으므로 실행 모델 표시는 해당 run에 전송한 값을 보존하고, 복구하지 못하면 알 수 없다고 표시합니다.

| 코어 | 진행 규칙 |
|---|---|
| `classic` | 남은 단계 체크리스트를 판정하고 먼저 진행한 뒤 상위 확인에서 되돌릴 수 있습니다. 기존 동작입니다 |
| `sequential` | 현재 단계만 판정합니다. 추종 yes는 후보이며 같은 프레임의 상위 `step_done` yes 후 정확히 한 단계 진행합니다. 미래 단계 자동 스킵은 없습니다 |
| `graph` | 이전·완료 단계를 포함한 전체 계획을 매번 관측합니다. 독립 단계는 순서와 무관하게 같은 노드에 대한 상위 확인 후 수락합니다. 새로운 no는 state와 그 의존 승인을 철회하고 무관한 사실은 보존합니다 |

순차 확인 대기는 `notice.code:"step_candidate"` 및 `pending.stage:"confirm"` / `pending.trigger:"step_done"`로 표시합니다.
클라이언트는 `state.step_index`를 그대로 그리고 추종 응답·전송 성공·애니메이션 종료를 완료로 바꾸지 않습니다.
서버는 run·판본·단계 활성화·프레임에 묶인 후보/확인 근거를 관리합니다. 오래된 응답·동일 프레임 재사용은 새 단계 완료 근거가 아닙니다.
user/measure 사용자 확인, required/requires, 최종 `confirm_done` 경계는 유지합니다.
순차·그래프 코어는 `plan_mode:false`여도 실행 중 재계획이 proposal 수락/거절을 거칩니다. 기존 코어+검토 꺼짐은 종전 즉시 재계획 동작입니다.
모델 가중치·추종 모드·임계값은 바꾸지 않았으며 정확도/속도 우위는 미측정입니다. 이 후보의 네 도식 action은 위 §8에 별도로 정의합니다.

graph의 최신 관측과 수락된 사실은 별개입니다. 일반 가림의 unsure는 이미 수락한 사실을 지우지 않으며, 늦은 upper yes가 더 최신 unsure를 yes로 덮지 않습니다. 사실에는 원래 plan/frame/step과 실제 provider/model/승인 경로를 보존합니다. 재계획은 대상·조건·종류·확인 방식·의존 관계가 유일하게 일치하는 사실만 이관하고, 재선택은 이력을 보존하되 현재 적용 가능성을 무효화합니다. 동결된 offline `temporal-state-v2` 교사 데이터로 자동 변환하지 않습니다.

### 15.14 초보자용 확인 질문·사진 (배포 전 변경)

`pendings/plan-usability`의 변경이며 운영 반영은 별도입니다. Plan 검토 모드는 공개 정보를 우선 조사하고, 사용자만 답할 수 있는 필수 정보가 부족하면 `clarifying`에 머뭅니다. 지금은 Astra native 검색을 지원하며 DeepSeek 검색/문서 읽기는 실험 도구에서만 검증했습니다.

- `start.reference_images`는 선택적인 별도 참고 사진 전체 목록(최대 12개)입니다. 원소 `{frame_id,image_base64,label?}`: JPEG ≤1,500,000 bytes, 최소 320×240, 각 변 ≤2048, 총 ≤2,073,600 pixels. 업로드 전 EXIF 방향 적용 및 긴 변 1600px 준비를 권장합니다. 현재 카메라 프레임과 혼동하지 마세요.
- `plan_answer`의 사진 목록은 생략하면 보존, 제공하면 전체 교체, `[]`면 제거입니다. start/plan_answer만 텍스트 봉투 상한 10 MiB이고 나머지는 512 KiB입니다. Live 클라이언트는 상위 요청의 카메라·자료·답변 공간을 남겨 참고 사진 base64 합계를 7.5 MiB로 제한합니다. 한도 초과 시 기존 사진을 자동으로 버리지 않습니다. 사진/답변 원문을 로그나 영구 저장소에 남기지 않습니다.
- `clarification` 한 문장과 입력·사진 추가·“모르겠어요”를 제공합니다. 답변 시 `clarification_id`를 그대로 보내며 목표/자료를 다시 시작하지 않습니다. 최대 12개의 답변을 이어서 보존합니다.
- 참고 사진 변환 중에는 시작/답변 전송을 잠그고 준비 상태를 표시합니다. 목표 변경·중지·세션 종료로 취소된 변환이 뒤늦게 끝나도 새 작업에 사진을 붙이지 않습니다.
- 실패 시 `notice.code:"plan_answer_failed"`와 새 질문 id가 돌아옵니다. 입력을 지우지 말고 사용자가 다시 보낼 수 있게 하세요. 목표·자료·사진·출처와 마지막 답변은 유지하며 재전송은 마지막 실패 답변을 교체합니다. 자동 provider 재시도/새 start는 금지합니다.
- planning 동안에도 입력을 보존하고, 새 질문이나 reviewing이 확인된 뒤에만 완료된 입력을 비웁니다. reviewing 이후에도 `plan_approve` 전 실행은 금지합니다.
- 단계 `details`(선택, ≤600자)는 say와 함께 표시하는 구체적 설명입니다. `plan_edit.edits`/`add`에서도 지원합니다. `state.research_sources`와 `plan.research_sources`는 최대 6개이며 HTTP(S) 링크만 안전하게 표시합니다. URL 확인은 요약의 사실 검증이 아닙니다.

Live 내장 진단 화면은 사진·질문 왕복과 전송 실패 보존을 구현했습니다. 별도 PWA는 별도 연동이 필요하며, 실제 사용자 작업의 정확성/편의성 검증을 진단 화면 테스트로 대신하지 않습니다.

---

<a id="client-handoff"></a>

## 16. 클라이언트 연동 변경 통합 인계서

갱신: 2026-10-04 · 대상: **Live API를 사용하는 별도 웹/PWA 클라이언트**

계획 모델·실행 코어 선택과 Plan 검토·자료·승인을 한 번에 연동하는 작업 목록입니다. API 계약은 이 문서 §1–15를 따릅니다.
**서버 구현·배포와 외부 클라이언트 수정은 별개입니다.** React 참조 화면과 Live 진단 페이지를 수정했다고 별도 PWA가 자동으로 바뀌지 않습니다. 아래 인수 항목은 외부 앱에서 확인해야 합니다. 모델 키 준비 상태는 잔액·품질·추론 성공을 보증하지 않습니다.

[우선순위](#client-section-1) · [모델·코어·검토 선택](#client-section-2) · [전송 예시](#client-section-3) · [자료·편집](#client-section-4) · [실행 화면](#client-section-5) · [프레임·오류](#client-section-6) · [인수 체크리스트](#client-acceptance) · [참조 구현](#client-section-8)

<a id="client-section-1"></a>

### 16.1. 우선 반영할 변경

| 우선순위 | 클라이언트 변경 | 서버 계약 / 완료 기준 |
|---|---|---|
| P0 | 시작 전 **계획 모델** 선택 | `start.plan_model`: `deepseek:high` / `astra:high`. 선택한 모델의 준비 여부로 시작을 제어; 자동 대체 금지 |
| P0 | 시작 전 **코어 로직** 선택 | `start.core_mode`: `classic` / `sequential` / `graph`. 실행 중 고정; `state.core_mode`로 표시 |
| P0 | Plan **검토**와 실행 분리 | `start.plan_mode:true` → `phase:reviewing`에서 검토 → `plan_approve` → 서버의 `running`을 받은 뒤 실행 화면 |
| P0 | 자료와 작업 맥락 분리 | 문서 본문은 `materials`, 작업 조건은 `context`. 조용한 잘라내기 금지 |
| P0 | 서버 상태를 그대로 반영 | 클라이언트가 단계 번호·완료·스킵을 추론하지 않음. 특히 순차 코어는 상위 확인 전에 다음 단계로 넘기지 않음 |
| P1 | 단계 메타데이터·출처·차단 사유 표시 | `details`, `check`, `required`, `goal_required`, `condition_kind`, `requires`, `targets`, `evidence`, `blocked` 및 graph 원장 |
| P1 | 실행 중 변경안 승인/거절 | `state.proposal` → `proposal_accept` / `proposal_reject`; 응답 전 기존 계획 유지 |
| P1 | 일시정지·재연결 상태 복구 | `run_pause` / `run_resume`, 최신 `state.paused`, 새 `start` 자동 발송 금지 |
| P1 | 사용자 확인과 작업 완료 구분 | `step_ack`은 user/measure 또는 graph의 required 검사; `confirm_done`은 `completion:confirmed`에서만 |
| P1 | 오류·실행 기록·접근성 | 서버 오류를 분류해 표시; 선택 코어/모델을 실행 기록에 남김; 키보드·모바일 인수 |

<a id="client-section-2"></a>

### 16.2. 세 가지 독립 선택축

| 필드 | 허용 값 | 생략 기본값 | 의미 |
|---|---|---|---|
| `plan_model` | `"deepseek:high"`, `"astra:high"` | `"deepseek:high"` | 계획·완료 확인·재계획·대화에 사용할 상위 모델 프로필 |
| `core_mode` | `"classic"`, `"sequential"`, `"graph"` | `"classic"` | 추종 판정 범위와 단계 진행 규칙 |
| `plan_mode` | `true`, `false` | `false` | 실행 전에 사용자의 계획 검토·승인을 받을지 여부 |

세 축은 독립이며 후보 계약은 **2모델 × 3코어 × 검토 여부 2가지 = 12조합**이다. 종전 8조합 검증과 구분한다. `plan_mode`를 모델 선택자로 쓰거나 `core_mode`를 검토 스위치로 쓰지 않는다. 다른 값은 `invalid_message`로 거절된다.

권장 시작 화면:

```text
계획 모델  [DeepSeek · high] [Astra · high]
코어 로직  [기존 코어]       [순차 코어]
계획 검토  [실행 전에 검토하기]
목표 / 작업 맥락 / 참고자료
[AI 전송 동의] [시작]
```

- 기존 동작을 유지하려면 기본 선택은 DeepSeek + 기존 코어 + 검토 꺼짐이다.
- `planning`, `reviewing`, `running` 및 일시정지 중에는 모델·코어·검토 선택을 잠근다. 바꾸려면 기존 실행을 종료하고 새 `start`를 보낸다.
- 시작 버튼을 눌렀다는 이유만으로 선택 UI를 풀지 않는다. 종료·완료·오류의 서버 상태를 기준으로 한다.
- 선택 상태와 실행 상태를 분리한다. 재연결 때 실행 코어는 `state.core_mode`를 따른다. 현재 State 계약에는 `plan_model` 필드가 없으므로 실행 모델 표시는 해당 run에 전송한 값을 보존하고, 알 수 없으면 임의로 기본 모델이라고 표시하지 않는다.

#### 16.2.1 계획 모델 준비 상태

`GET /v1/health`의 예시 일부:

```json
{
  "ready": true,
  "plan_ready": true,
  "plan_models": {"deepseek:high": true, "astra:high": false},
  "follow_ready": true,
  "tracker_ready": true
}
```

- `plan_models`의 선택한 항목이 정확히 `true`일 때 해당 프로필이 구성된 것이다. 누락·잘못된 값·조회 중은 준비 완료로 가정하지 않는다.
- `plan_ready`는 둘 중 하나라도 구성되면 true인 집계값이다. **선택한 프로필의 준비 여부 대신 사용하면 안 된다.**
- 선택한 프로필이 준비되지 않았으면 시작을 막고 이유를 표시한다. 다른 프로필로 조용히 보내지 않는다.
- 세 코어는 같은 모델 서비스·가중치·임계값을 쓴다. 별도 `core_modes` health 맵은 없다. 코어를 바꿔도 누락된 추종/추적 서비스가 생기지 않는다.
- 모델명·추론 강도·API 키를 브라우저가 자유 문자열로 지정하지 않는다. 키는 서버에서만 관리한다.
- 화면·자료 전송 동의는 사용자가 직접 한다. 모델 변경에 맞춰 전송 대상 안내도 바꾸되 동의를 자동 체크하지 않는다.

#### 16.2.2 두 코어의 사용자 설명

| 기존 코어 (`classic`) | 순차 코어 (`sequential`) |
|---|---|
| 현재부터 남은 단계들을 함께 판정 | 현재 단계만 판정 |
| 먼저 진행한 뒤 상위 모델이 재확인; 틀리면 복귀 | 추종자의 yes는 후보일 뿐; 상위 step_done yes 후 한 단계 진행 |
| 기존 미래 단계 관측에 따른 위치 이동 정책 유지 | 미래 단계 yes로 자동 건너뛰기 금지 |
| 기존 응답 동작 유지 | 상위 확인을 기다리는 시간이 사용자에게 보임 |

순차 코어에서 화면은 서버 `step_index`를 그대로 유지한다. `pending.stage:"confirm"` / `pending.trigger:"step_done"` 및 `notice`를 이용해 “현재 단계 확인 중”을 표시한다. 후보 알림은 `notice.code:"step_candidate"`다. **추종 yes, 요청 전송 성공, 애니메이션 종료를 단계 완료로 바꾸지 않는다.**

완료 근거는 서버가 run·계획 revision·단계 활성화·프레임에 묶어 관리한다. 늦게 도착한 응답, 다른 단계의 응답, 같은 프레임 재사용은 새 단계 완료의 근거가 아니다. 클라이언트가 이 원장을 재구현하거나 자체 `step_done` 메시지를 보내지 않는다. 마지막 단계는 제자리에서 확인할 수 있으며 전체 완료는 여전히 별도의 목표 확인이다.

순차 코어가 더 정확하거나 빠르다는 품질 보증은 없다. 현재 단계만 보내는 입력은 기존 학습 분포와 다르며, 이번 변경은 재학습·모델 교체가 아니다.

<a id="client-section-3"></a>

### 16.3. 전송 예시와 접속 경로

브라우저는 **Live API**의 세션·WebSocket을 사용한다. 내부 `/api/guide/*`는 Live→백엔드 호출이며 브라우저가 직접 연결할 API가 아니다.

- 기본 API: `https://sangye.kr/synoptics/api`
- 세션 발급: `POST /v1/sessions`; 응답의 `live_url` 그대로 사용
- 연결 후 `hello` → `ready`를 받은 다음 제어 메시지 전송
- 현재 프로토콜 문자열: `synoptics-live/0.1-draft`

자료 없이 바로 실행하는 예시:

```json
{
  "type": "start",
  "goal": "컵을 오른쪽으로 옮기기",
  "consent_ai": true,
  "plan_model": "deepseek:high",
  "core_mode": "sequential",
  "plan_mode": false
}
```

자료를 검토한 뒤 실행하는 예시:

```json
{
  "type": "start",
  "goal": "컵을 작업서에 따라 포장하기",
  "context": "책상 위에서 작업",
  "consent_ai": true,
  "plan_model": "astra:high",
  "core_mode": "sequential",
  "plan_mode": true,
  "materials": [
    {"title": "컵 포장 작업서", "version": "v2", "text": "컵을 완충재로 감싼다. 감싼 컵을 상자에 넣는다."}
  ]
}
```

`consent_ai:true`는 동의를 받은 뒤 전송하는 값이지 예시를 복사해 자동 동의시키라는 뜻이 아니다. `mode`, `manual`은 현재 계약의 대체 필드가 아니다. Live의 `plan_mode`는 검토용 bool이며 상위 백엔드의 구형 provider 선택자는 제거됐다.

계획·재계획·대화는 느린 상위 호출일 수 있다. 서버는 최대 180초 공급자 기한 및 Live 190초 상위 요청 기한을 사용한다. 클라이언트의 짧은 임의 timeout으로 같은 `start`를 재전송하지 않는다. 중지 선택은 제공하되 화면에 기다리는 상태를 표시한다. 코어 선택이 이 기한·호출 비용을 줄인다고 표시하지 않는다.

<a id="client-section-4"></a>

### 16.4. 자료 입력과 계획 편집

#### 자료

- `context`: 장소·준비물 등 작업 맥락. 문서 본문을 이어 붙이지 않는다.
- `materials`: `{title, version?, text}` 배열. 자료는 없어도 된다.
- `ready.limits`의 `materials_max`, `material_text_max_chars`, `plan_steps_max`, `goal_max_chars`, `context_max_chars`를 읽는다. 이 후보는 자료 4개·각 4000자·계획 16단계지만 UI 상수로 고정하지 않는다.
- 제목은 1–80자, 판본은 최대 32자다. 초과하면 편집을 안내하며 `slice()`로 조용히 자르거나 자동 분할하지 않는다.
- 파일을 목록에 붙였다는 사실은 모델에 본문을 전송했다는 뜻이 아니다. TXT/PDF 등의 추출 성공·실패를 구분하고 전송할 텍스트를 사용자가 검토할 수 있게 한다.
- React 참조 앱의 `/api/plan/manual`은 그 앱의 쿠키·Origin 인증용이다. 별도 PWA의 Bearer API로 오해해 직접 호출하지 않는다. 별도 클라이언트의 추출은 실제 지원하는 방법으로 구현하고, 미지원 형식은 텍스트 붙여넣기를 안내한다.

#### 검토 단계

```text
start(plan_mode:true)
  → planning
  → reviewing + plan.status:draft
  → 사용자가 편집 / 승인 / 버리기
  → plan_approve
  → 서버 running + plan.status:approved
```

- `planning`: 생성 중. 부분 문구 `partial`은 확정 계획이 아니다.
- `reviewing`: 아직 실행되지 않는다. 추적·판정·오버레이를 실행 화면처럼 표시하지 않는다.
- 승인 버튼은 `{"type":"plan_approve"}`를 보내고 대기한다. 단순한 화면 이동이나 WebSocket `send()` 성공을 승인으로 간주하지 않는다.
- 버리기는 `{"type":"plan_discard"}`이며 서버 `idle`을 기다린다.
- 승인 후 `plan_edit`은 `plan_locked`다.

편집 요청 예시:

```json
{
  "type": "plan_edit",
  "edits": [{"step_id": "s2", "check": "user", "required": true}],
  "order": ["s1", "s2", "s3"]
}
```

`order`를 보내면 해당 계획의 전체 ID를 한 번씩 보낸다. 추가는 `add`, 삭제는 `remove`이며 서버가 새 ID를 부여한다. 편집 후 서버의 새 `state.plan.revision`을 받아 표시한다. 클라이언트가 변경을 확정하거나 서버 revision을 직접 증가시키지 않는다.

단계 상세에서 표시할 값:

| 값 | 의미 / UI |
|---|---|
| `say` | 현재 행동 문장; 수정 가능 |
| `done_when` | 결과 상태; 수정 가능 |
| `commands` | 그림 명령; 렌더링 계약 그대로 사용 |
| `check` | `visual` / `user` / `measure`; 확인 방식 |
| `required` | 건너뛰기 금지; 색만이 아닌 텍스트 표시 |
| `requires` | 선행 완료 단계 ID |
| `targets` | 관련 대상 이름 목록; 임의 좌표나 새 추적 상자를 만들지 않음 |
| `evidence` | 자료 ID·판본·절·인용을 읽기 전용 표시; 사용자가 허위 근거를 입력하는 필드가 아님 |

`evidence`는 `{material_id,version?,locator?,quote?}` 또는 null이다. `source_quote`는 폐기된 필드다. null은 근거 없음이며 “원문 검증 완료”로 표시하지 않는다. 인용이 존재한다고 절차의 안전성·완전성이 보증되는 것도 아니다.

<a id="client-section-5"></a>

### 16.5. 실행 화면은 서버 상태의 투영

| 서버 상태 | 표시 / 허용 행동 |
|---|---|
| `idle` | 목표·자료·모델·코어 선택, 시작 |
| `planning` | 생성 진행, 중지; 선택 변경 잠금 |
| `reviewing` + `draft` | 전체 단계·상세 편집, 승인, 버리기 |
| `running` + `approved` | 현재 `step_index`와 서버 안내, 대화, 다시 보기, 정지 |
| `running` + `paused:true` | 일시정지, 재개 또는 중지 |
| `running` + `proposal` | 현재 계획/변경안 비교, 수락 또는 거절 |
| `completion:checking` | 최종 목표 재확인 중; 완료로 미리 전환하지 않음 |
| `completion:confirmed` | 사용자의 완료 확인 버튼 활성화 |
| `completed` | 최종 결과·기록, 새 시작 |
| `error` | 오류·재시도 가능 여부; 새 시작 전 기존 상태 정리 |

#### 필수·사용자 확인

- `blocked.step_id`, `blocked.reason`, `blocked.requires`를 표시한다. 서버가 막은 단계를 클라이언트가 넘어가지 않는다.
- 사용자가 직접 확인한 user/measure 또는 graph의 required 검사에 `{"type":"step_ack","step_id":"s2"}`를 보낸다. 서버가 전체 선행 조건을 검사한다.
- 그 외 visual ack는 `invalid_edit`다. 사진 한 장이 사용자 검사를 대신하지 않는다.
- `steps_skipped`와 `steps_user_done`은 다르다. skipped를 done으로 저장하거나 필수 확인을 임의로 채우지 않는다.
- `step_index`는 0부터 시작한다. UI 순번만 +1 해 표시한다.
- 남은 비필수 단계가 있어도 원래 사용자 목표는 이미 달성될 수 있다. 클라이언트가 자체 “모든 단계 완료” 조건으로 서버의 목표 완료를 막지 않는다.

#### 변경안과 일시정지
- 실행 중 proposal 경계는 `plan_mode:true` 또는 `core_mode:"sequential"/"graph"`에 적용된다. 초기 검토를 꺼도 변경안을 수락/거절하며, `classic` + 검토 꺼짐의 즉시 재계획과 구분한다.

- `state.proposal`이 있으면 현재 계획과 변경안의 차이 및 `reason`을 보여 준다.
- 수락: `{"type":"proposal_accept"}`; 거절: `{"type":"proposal_reject"}`. 새 서버 상태가 오기 전 로컬 계획을 덮어쓰지 않는다.
- 정상적인 거절은 오류가 아니다. 서버의 복원 실패 `plan_changed`는 일반 거절과 구분한다.
- 일시정지: `{"type":"run_pause"}`; 재개: `{"type":"run_resume"}`. 화면/버튼만 멈추는 로컬 pause로 대체하지 않는다.
- 재연결 후 최신 `paused`, `blocked`, `proposal`, `plan.status`, `core_mode`를 다시 그린다. 자동 새 `start`나 사용자 pause 해제는 금지한다.

#### 목표 완료와 기록

- `completion:"confirmed"`는 서버가 시각적으로 확인한 단계이고 사용자 확인 전이다.
- 사용자가 완료 버튼을 누르면 `{"type":"confirm_done"}`를 보낸다. 서버 `phase:"completed"`를 받은 뒤 최종 기록을 저장한다. 일반 `stop`을 완료로 기록하지 않는다.
- 실행 기록에는 최소 run의 목표·선택 모델·코어·계획 ID/revision·최종 상태를 구분한다. 문서 원문·영상 자동 저장은 별도의 사용자 동의/보존 정책 없이 추가하지 않는다.

<a id="client-section-6"></a>

### 16.6. 프레임·오버레이·오류: 기존 계약도 유지

선택 UI 추가 과정에서 다음을 퇴행시키지 않는다.

- WebSocket `ready` 전 제어 메시지/프레임을 보내지 않는다. token은 로그·URL에 넣지 않는다.
- 한 번에 미응답 일반 프레임은 하나. `track.seq`로 크레딧을 반환하고 `limits.max_fps`를 지킨다. 재연결 시 프레임 seq는 다시 0부터다.
- `capture_hi.req_id`에 대응하는 실제 고해상도 프레임을 전송한다. 카메라 반전은 캡처 픽셀과 화면 좌표를 일치시킨다.
- 오버레이는 최신 tracking의 run/track/generation과 `overlay.binding`이 같고, 박스 나이가 `limits.box_max_age_ms` 이내일 때만 그린다. `reviewing`, `paused`, 추적 상실에서는 오래된 그림을 남기지 않는다.
- `motion_key` 변경 때만 모션을 재시작한다. 이번 코어 선택은 새 action enum, 여러 대상의 좌표, timer 확인 계약을 추가하지 않는다.
- `notice`·`budget_notice`·`pending`은 표시용이다. 이를 읽어 자체 단계 진행 결정을 만들지 않는다.
- `invalid_message`, `invalid_edit`, `plan_locked`, `not_reviewing`, `not_running`, `required_check`, `no_proposal`, `plan_changed`를 상황에 맞게 표시한다. 모델·코어 실패를 다른 값으로 자동 재전송하지 않는다.
- 선택 그룹은 native radio/select와 연결된 레이블을 사용한다. 진행/정지/오류를 색만으로 구분하지 않는다. 주요 버튼은 44px 이상을 확보하고 키보드·모바일에서 확인한다.
- 기존 음성 출력/서버 TTS 계약은 변경하지 않는다. 이번 문서의 작업을 이유로 TTS 설정·모델·서비스를 교체하지 않는다.

<a id="client-acceptance"></a>

### 16.7. 외부 클라이언트 인수 체크리스트

아래는 **클라이언트 담당자가 외부 앱에서 확인할 항목**이며, 체크박스가 비어 있는 상태를 이미 통과한 것으로 해석하지 않는다.

- [ ] 후보 계약의 모델 2 × 코어 3 × 검토 여부 2의 12조합에서 실제 `start`와 서버 `state.core_mode`가 일치한다.
- [ ] 생략 기본값은 DeepSeek/classic/검토 꺼짐이고, 잘못된 선택은 서버 거절을 표시한다.
- [ ] 선택 모델이 준비되지 않으면 시작이 막히며 다른 모델로 전송하지 않는다.
- [ ] planning/reviewing/running/paused 동안 선택이 고정되고, 새 실행에서만 바뀐다.
- [ ] 순차 코어에서 상위 확인이 지연돼도 현재 단계에 머무르며 대기를 보여 준다. no/unsure/오류에서 미리 진행하지 않는다.
- [ ] 서버의 단계 변경·완료만 반영한다. 미래 단계의 모습이나 로컬 애니메이션으로 자동 건너뛰지 않는다.
- [ ] 자료 본문을 `materials`로 보내고 초과·추출 실패를 숨기지 않는다. context에 합치거나 자르지 않는다.
- [ ] Plan은 실제 reviewing에서 검토하며, approve 요청 후 running을 받은 뒤에만 실행 화면으로 간다.
- [ ] 편집 결과와 revision을 서버에서 받아 표시하고, evidence를 허위 생성/편집하지 않는다.
- [ ] user/measure 확인, visual ack 거절, required/requires 차단 사유가 올바르게 표시된다.
- [ ] proposal 수락/거절 시 현재/변경 계획과 paused 상태가 일치한다.
- [ ] pause → 연결 끊김 → 재연결 후 자동 재개·새 start·선택 변경이 없다.
- [ ] confirmed → 사용자 confirm_done → completed 순서를 지키고 stop을 완료로 기록하지 않는다.
- [ ] 프레임 크레딧·고화질 응답·좌우 반전·박스 신선도·overlay binding이 유지된다.
- [ ] 390px 모바일과 데스크톱에서 선택 그룹, 긴 안내, 오류, 대기 상태가 잘리지 않고 키보드로 조작된다.
- [ ] 외부 PWA 실제 화면과 WebSocket 송수신 증거를 남긴다. 참조 앱 검증을 외부 앱 인수로 대신하지 않는다.

<a id="client-section-8"></a>

### 16.8. 참조 구현과 범위

| 대상 | 위치 | 역할 |
|---|---|---|
| React 모델 선택 | `web/src/planMode/PlanModelSelector.tsx` | 모델 준비 상태와 선택 UI 참고 |
| React 코어 선택 | `web/src/planMode/CoreModeSelector.tsx` | native 선택·실행 중 잠금 UI 참고 |
| React 실행 | `web/src/intent/useIntentLoop.ts`, `sequentialCore.ts` | 내부 HTTP 참조 앱용; **Live PWA로 복사할 상태 머신이 아님** |
| Live 구현 보존 | `deploy/live-api/plan-astra.patch` | 비-Git Live 소스와 명세에 대한 누적 변경; 이미 적용된 서버에 중복 적용 금지 |
| HTTP 계약 | `backend/app/guide_contracts.py`, `web/src/generated/api.generated.ts` | Live→백엔드 내부 계약 |
| Live 계약 | `live-api/src/synoptics_live/contracts.py` | 별도 배포 디렉터리의 Start/State/Plan/편집 계약 |
| 검증 기록 | `docs/e2e/core-choice-verification.json` | 실제 실행한 범위·미측정 항목·배포 식별자 |

외부 클라이언트는 서버 상태를 표현하는 입출력 계층이다. 모델 판정·단계 후보·확인 근거 원장·재시도/예산 제어를 클라이언트에 새로 복제하지 않는다. 코드 병합과 서버 배포 후에도 외부 PWA의 위 인수는 별도로 필요하다.
