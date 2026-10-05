# 클립 리플레이 하네스 (`replay_clip.py`)

추적기 서비스의 비공개 `/runs/*` 와이어를 앱(`backend/app/tracking.py`)과 같은 모양으로 구동해, 동영상
클립을 프레임 단위로 재생하고 서비스가 돌려준 상태를 기록한다. 서비스 코드는 바꾸지 않는다.
표준 라이브러리와 `ffmpeg` CLI만 쓴다. 측정 도구이지 데모 경로가 아니다.

## 흐름

1. `ffmpeg`로 클립을 `--fps`로 뽑고 **긴 변**을 `--long-side`(기본 1280)로 맞춘다(종횡비 유지).
   미리 뽑은 프레임을 쓰는 모드는 아래 "미리 뽑은 프레임으로 재생"을 본다.
2. `GET /health`로 준비 상태와 `policy.occluded_max_frames`(가림 창)를 읽는다.
3. `POST /runs/start {run_id, target, fresh:true}` → 프레임마다 `POST /runs/frame {run_id, frame_seq, frame_b64, seed_box?}`
   (`--seed-box`는 0번 프레임에만; 없으면 `target` 명사로 GroundingDINO 획득) → `POST /runs/stop`.
4. 기본은 폐루프(앞 응답이 와야 다음 프레임). `--realtime`이면 클립 시계에 맞춰 보낸다.

가림 창은 **처리된 연속 비관측 프레임 수**다(카메라 초가 아님). 그래서 같은 창 값이라도 추출 fps가
다르면 덮는 시간이 다르다: 창 15는 10fps에서 1.5초, 30fps에서 0.5초.

## 실행

```
ssh -N -L 8090:127.0.0.1:8090 user@gpu-server  # 다른 PC에서 연결할 때만; 자신의 SSH 접속 정보로 지정
python3 tools/tracker_service/replay_clip.py \
  --clip ./sample.mp4 --target glasses \
  --base-url http://127.0.0.1:8090 --fps 10 --long-side 1280 \
  --out-dir /tmp/replay/glass-10
```

`sample.mp4`는 실행자가 준비한 영상입니다. 이 저장소에는 원본 촬영 영상을 포함하지 않습니다.

`--base-url`은 루프백 주소만 받는다. 픽스처 원본은 읽기만 하고, 추출 프레임은 `--out-dir`(또는 `--frames-dir`) 아래에 쓴다 — 저장소에 넣지 않는다.

기본 동작에서는 `--frames-dir`에 이미 있던 `f_*.jpg`를 지우고 다시 추출한다.

### 미리 뽑은 프레임으로 재생 (ffmpeg 없는 호스트, 예: 데모 Mac)

ffmpeg가 없는 곳에서는 다른 호스트에서 뽑은 `f_*.jpg` 디렉터리를 옮겨 와 그대로 재생한다.

```
python3 tools/tracker_service/replay_clip.py --use-existing-frames \
  --frames-dir /path/glass-30fps-1280 --fps 30 --target glasses \
  --base-url http://127.0.0.1:8090 --out-dir /tmp/replay/glass-30
```

- 위 예시처럼 운영 추적기(8090, Mac에서는 터널)로 보낼 때는 라이브 세션이 없을 때만 돌린다 — 같은 GPU를 쓴다.
- `--use-existing-frames`: `--frames-dir`의 `f_*.jpg`를 이름순으로 그대로 보낸다. ffmpeg를 부르지 않고,
  프레임을 **지우거나 고치지 않는다**. `--clip`은 필요 없다(주면 요약에 이름만 남는다). `--frames-dir`이 없거나
  `f_*.jpg`가 하나도 없으면 시작하지 않고 실패한다.
- 플래그가 없어도, `--frames-dir`에 `f_*.jpg`가 있고 **PATH에 ffmpeg가 없으면** 같은 모드로 동작한다(표준오류에
  `replay: using N existing frames ...` 한 줄). ffmpeg가 있으면 기본 동작(재추출)은 그대로다.
- `--fps`는 그 프레임을 **뽑을 때의 fps**여야 한다. 여기서는 `video_t_s`와 `--realtime` 시계에만 쓰이고, 하네스는
  이 값을 프레임에서 확인할 수 없다. `--long-side`·`--jpeg-quality`는 무시된다.
- `summary.json`에 `frames_source`(`extracted`/`existing`), `frames_dir`, `frame_size_first`(첫 프레임 JPEG
  헤더에서 읽은 `[폭, 높이]`)가 붙는다. 기존 프레임 모드에서는 크기를 하네스가 정하지 않았으므로 `long_side`는 `null`이다.

## 출력

- `frames.jsonl` — 프레임당 `{frame_seq, video_t_s, state, transition, box, confidence, source, generation, compute_ms, latency_ms, http_status}`.
  `compute_ms`는 서비스가 잰 단계 시간, `latency_ms`는 하네스가 잰 HTTP 왕복(터널 포함).
- `summary.json` — 획득 프레임, 상태별 프레임 수, 전이 횟수, 가림 에피소드(연속 `occluded` 구간과 그 끝:
  `tracking` 복귀 / `lost` / 클립 끝), `lost` 이벤트와 그 프레임, 커버리지(박스 있는 프레임 / 전체 프레임,
  획득 이후 기준도 함께), `compute_ms`·`latency_ms`의 p50/p95(최근접 순위 — 측정된 값 그대로), 정상 상태
  SAM 단계만의 p50/p95, 첫 프레임 `compute_ms`, `/health`의 작업 카운터 전후.
- 표준출력에 한 줄 요약 JSON. HTTP 오류나 알 수 없는 상태가 오면 그 자리에서 멈추고 `error`를 남긴 뒤 종료 코드 1.

`lost` 뒤에는 서비스 정책상 조용한 재획득이 없으므로, 남은 프레임은 모델 작업 없이 `lost`로 답한다.
커버리지가 그만큼 떨어지는 것이 정상이다.

## 가림 창 바꾸기 (나란히 띄운 측정용 인스턴스)

운영 컨테이너(`synoptics-tracker`, 8090)는 건드리지 않고, 이름·포트·소스 디렉터리가 다른 두 번째
인스턴스를 띄운다. 기본 이름은 8090과 `/v/tracker/service` 전용으로 잠겨 있다.

```
export TRACKER_CONTAINER=synoptics-tracker-spike TRACKER_SERVICE_DIR=/v/tracker/service-spike
bash deploy/tracker/run.sh --port 8093 --push
bash deploy/tracker/run.sh --port 8093 --env TRACKER_OCCLUDED_MAX_FRAMES=30   # 같은 이름의 소유 컨테이너를 ID로 교체
bash deploy/tracker/run.sh --port 8093 --stop                                 # 끝나면 반드시 제거
```

`TRACKER_OCCLUDED_MAX_FRAMES`는 정수 1..600만 받는다(그 밖이면 서비스가 시작 단계에서 실패한다 — 기본값으로
조용히 되돌리지 않는다). 적용된 값과 출처(`env`/`default`)는 `/health`의 `policy.occluded_max_frames`,
`policy.occluded_max_frames_source`에 나온다. 두 인스턴스는 GPU를 함께 쓰므로, 운영 추적기가 라이브 세션을
처리하는 중에는 측정하지 않는다(`docker logs --tail 5 synoptics-tracker`에 `/health` 줄만 있어야 한다).

## 지연 해석 주의

`latency_ms`는 하네스에서 잰 왕복이라 `ssh -L` 포워드를 포함한다. 2026-10-01 측정에서 이 호스트 → 포워드 →
8093 경로는 POST마다 약 40ms 바닥이 있었다(작은 `/runs/start` 40.8ms, 모델 작업 전에 422로 거절된 프레임
42ms; keep-alive+`TCP_NODELAY` 클라이언트와 `urllib` 모두 같음). 같은 프레임을 GPU 노드 안에서 보내면
`compute_ms` 대비 오버헤드 p50 0.95ms(n=39)였다. 서비스 자체의 단계 시간은 `compute_ms`로 읽는다. 포워드의
40ms 바닥 원인(지연 ACK·Nagle 추정)은 확인하지 않았다.
