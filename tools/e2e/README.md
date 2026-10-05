# tools/e2e — 헤드리스 끝단 벤치마크 (합성 카메라 + 실제 DeepSeek + 실제 추적기 + 로컬 4B)

이전 세션의 스크래치 하네스(phase 3~5)를 저장소로 옮긴 것이다. 2026-10-01 체크리스트 추종 벤치마크
(`docs/e2e/2026-10-01-checklist-benchmark.md`)가 이 디렉토리로 측정됐다.

## 준비
- 클립: `bash tools/e2e/make_clips.sh` → `tools/e2e/clips/*.y4m` (gitignore). 원본은 `~/Projects/aisw/{glass,put-airpod,increase_temp}.mov`.
- 터널(개발 호스트 → GPU 노드 LAN; Tailscale IP는 22번이 닫혀 있음):
  `ssh -tt -L 127.0.0.1:18090:127.0.0.1:8090 -L 127.0.0.1:18081:127.0.0.1:8081 -L 127.0.0.1:18084:127.0.0.1:8084 riey@192.168.219.250 'exec sleep 86400'`
  (`-tt`가 없으면 Nagle+지연 ACK로 RTT 40ms 바닥이 생긴다.)
- 서버(저장소 루트, web/dist 빌드 뒤):
  ```
  PROVIDER=deepseek DEEPSEEK_MODEL=deepseek-flash DEEPSEEK_API_KEY_FILE=$HOME/.config/synoptics/API_KEY.txt \
  AISW_LOCAL_ONLY=1 AISW_TRACKER_URL=http://127.0.0.1:18090 AISW_FOLLOW_PROVIDER=local \
  AISW_FOLLOW_LOCAL_URL=http://127.0.0.1:18084/v1/chat/completions AISW_FOLLOW_LOCAL_MODEL=Qwen3.5-9B-Q8_0 \
  AISW_FOLLOW_LOCAL_MODE=json_schema .venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8043
  ```
- 환경변수: `APP_PORT`(기본 8043), `E2E_PUPPETEER`(puppeteer-core 경로), `E2E_CHROMIUM`(브라우저 실행 파일),
  `E2E_OUT`(runs/·budget.json 위치, 기본 이 디렉토리), `FOLLOW_PAID=1`이면 follow도 유료로 센다. `E2E_SHOTS_MS=<ms>`이면 가이드 시작부터 클립 끝까지 `.camera-stage-media`를 <ms>마다 `<OUT>/shots/t<clipMs>.png`로 찍고, 그 순간의 오버레이(`data-guide-visible`·`-reason`·`data-motion`·`-phase`)와 콜아웃(`data-avoid`·visibility)을 timeline의 `stageShots`에 남긴다(미설정이면 기존과 동일). 유료 호출 예산 40회(`budget.json`).

## 실행
```
cd tools/e2e
node run.cjs run1/glass clips/glass.y4m 9.633 "안경 벗기"
node run.cjs run1/put-airpod clips/put-airpod.y4m 17.367 "에어팟 케이스 열기"
node run.cjs run1/increase_temp clips/increase_temp.y4m 8.167 "온도를 28도로 올리기"
python3 summarize.py runs/run1/glass/timeline.json     # 호출 목록(step_checks·옛 step_status 둘 다) + UI 단계 전환
python3 analyze.py runs/run1/glass/timeline.json       # summary.json (지연 분위수·비용)
```
클립은 반복 재생되며 두 번째 반복의 0번 프레임에 맞춰 "가이드 시작"을 누른다. clip 시각 오차 ±100ms(`align.py`).
클립이 끝나면 새 guide 호출을 막고 진행 중 호출만 기다린다(`BLOCKED after_clip_end`는 하네스 차단이지 앱 오류가 아니다).
정답 라벨은 `truth/*-tile.png`(2fps 시트)를 눈으로 읽는다.

## 발화 스크립트 (`/api/guide/talk`, 2026-10-01)

`E2E_TALK=<json>`이면 클립 시각 `clipMs`(가이드 시작 기준)가 지난 뒤 TalkBar 입력이 열리는 즉시 `text`를 입력하고 Enter를
누른다. 입력이 아직 없거나 잠겨 있으면(계획 전, 앞 발화의 답 대기 중) 열릴 때까지 기다리고, 클립이 끝나면 보내지 않은 발화는 버린다.
talk는 유료 호출로 센다(예산 40회 공유). 파일: `talk/glass-target.json`·`glass-done.json`·`glass-say.json`(발화 1개씩),
`talk/glass-target-done.json`(3.0 s "어디를 말하는지 모르겠어" → 6.5 s "이미 했어"), `talk/glass-all.json`(3.0 s say → 5.5 s done →
7.8 s target; 마지막 발화는 confirm 하한 대기로 9.6 s 클립 안에 못 나갈 수 있다). 결과: `docs/e2e/2026-10-02-guide-talk.md`.
```
cd tools/e2e
E2E_TALK=talk/glass-target-done.json node run.cjs talk1/glass-target-done clips/glass.y4m 9.633 "안경 벗기"
E2E_TALK=talk/glass-say.json         node run.cjs talk1/glass-say         clips/glass.y4m 9.633 "안경 벗기"
python3 talk_summary.py runs/talk1/glass-target-done/timeline.json   # 행동·왕복 지연·plan_revision 흐름
```
기대 행동: target → `target`(추적 재시작, 새 run), done → `step_mark: done`(다음 단계, PlanCard `사용자 확인 ✓`),
say → `step_say`(같은 단계, plan_revision +1). 다른 행동이 나오면 그대로 기록한다(적중률은 측정값).

## follower_bench — 추종자 모델 단독 비교
`follower_bench.py`는 브라우저·DeepSeek 없이 추종자 하나만 잰다: 앱의 `LocalFollower`(실제 follow 프롬프트, 요청별 strict json_schema,
앵커 상자 그리기)로 세 클립의 2fps 프레임을 보내고 라우트와 같은 규칙으로 검증한 뒤, 고정 계획·눈 판독 정답(`PLANS`·`TRUTH`)과 비교한다.
준비: 클립마다 `ffmpeg -i ~/Projects/aisw/<clip>.mov -vf "scale=1280:-2,crop=1280:720,fps=10" -q:v 3 F/<clip>/f_%05d.jpg`, 그 프레임을
`python3 tools/tracker_service/replay_clip.py --use-existing-frames --frames-dir F/<clip> --fps 10 --target <명사> --base-url http://127.0.0.1:18090 --out-dir T/<clip>`
로 재생해 프레임별 상자를 얻는다(put-airpod는 `"earbud case"`; `"airpods case"`는 11s까지 획득 못 함). 실행·채점:
```
.venv/bin/python tools/e2e/follower_bench.py run --url http://127.0.0.1:18084/v1/chat/completions --model Qwen3.5-9B-Q8_0 \
    --frames-root F --tracks-root T --out out/9b-q8.jsonl          # --clips glass --sample-every 1 --t-min 5.5 --t-max 7.5: 촘촘 창
.venv/bin/python tools/e2e/follower_bench.py score out/*.jsonl    # 모델별 정확도·거짓 yes·glass s2·p50/p95 표 + 프레임별 판정 줄
```
결과: `docs/e2e/2026-10-01-follower-model-spike.md`.
