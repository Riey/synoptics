#!/usr/bin/env bash
# Stub verification for the ownership/destructive-command guards in deploy/tracker/run.sh.
#
# The script cannot be exercised against the real GPU host from here, so this harness puts a stateful
# fake `docker` and a fake `ssh` on PATH and runs the REAL script against them. No network, no daemon,
# no container, no sudo. Every case asserts on the exit code AND on the message text, and on whether a
# removal happened at all — a false "removed"/"stopped" claim must fail the harness.
#
#   bash deploy/tracker/test_run_guard.sh
#
# Scenarios covered: absent container, foreign labels, unlabelled legacy, partially-matching labels,
# inspect permission failure, daemon-down error, malformed ID, owned container (targeted removal by
# full ID), removal that did not take effect, reserved container name, hostile env value, unverified
# `--migrate` evidence, the push helper (stdin streaming + ownership labels), and the side-by-side
# instance options (--port, --env pass-through, TRACKER_SERVICE_DIR, default-name/port lock, direct-IP host).
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
SCRIPT=$HERE/run.sh
[[ -f $SCRIPT ]] || { echo "run.sh not found next to this harness" >&2; exit 2; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
SHIM=$W/shim
mkdir -p "$SHIM"

OWNED_ID=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
LEGACY_ID=39d964ec4bdf6341b610c7435f45edcbb5b35aa10312114d7cf49da3a5bf0397
NEW_ID=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb

# ---------------------------------------------------------------- fake binaries
cat >"$SHIM/ssh" <<'EOS'
#!/usr/bin/env bash
# Drop ssh options and the host, then run the remaining single command string locally (the stub
# `docker` is first on PATH, so this emulates the GPU host's daemon faithfully enough for the guards).
printf '%s\n' "$*" >>"$ST/ssh.log"
while (( $# > 0 )); do
  case $1 in
    -o) shift 2 ;;
    -*) shift ;;
    *)  break ;;
  esac
done
shift
exec bash -c "$*"
EOS

cat >"$SHIM/docker" <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
mode=${FAKE_MODE:-notfound}
sub=$1; shift
fmt=''
case $sub in
  inspect)
    if [[ ${1:-} == -f ]]; then fmt=${2:-}; shift 2; fi
    target=${!#}
    if [[ $fmt == *'State.Running'* ]]; then echo "true"; exit 0; fi
    if [[ -f $ST/removed && ${FAKE_RM_INEFFECTIVE:-0} != 1 ]]; then
      echo "Error: No such object: $target" >&2; exit 1
    fi
    case $mode in
      notfound) echo "Error: No such object: $target" >&2; exit 1 ;;
      perm)     echo "permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get \"http://%2Fvar%2Frun%2Fdocker.sock/v1.45/containers/$target/json\": dial unix: permission denied" >&2; exit 1 ;;
      daemon)   echo "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?" >&2; exit 1 ;;
      malformed) echo "not-a-container-id|/x|img|||v:/v,|cmd "; exit 0 ;;
    esac
    case $mode in
      unlabelled|malformed) id=${FAKE_LEGACY_ID:-unknown-legacy-id} ;;
      *)                    id=${FAKE_OWNED_ID:-unknown-owned-id} ;;
    esac
    if [[ $target =~ ^[0-9a-f]{64}$ ]]; then id=$target; fi
    case $mode in
      owned)          printf '%s|/synoptics-tracker|%s|aisw-hybrid|tracker-service|%s:/v,|%s \n' "$id" "${FAKE_IMAGE:-img}" "${FAKE_VOL:-synoptics-gpu-repro}" "${FAKE_CMD:-python3 /v/tracker/service/service.py}" ;;
      owned_otherrole) printf '%s|/synoptics-tracker|img|aisw-hybrid|db|v:/v,|cmd \n' "$id" ;;
      half_labelled)   printf '%s|/synoptics-tracker|img|aisw-hybrid|<no value>|v:/v,|cmd \n' "$id" ;;
      unlabelled)     printf '%s|/synoptics-tracker|%s|<no value>|<no value>|%s:/v,|%s \n' "$id" "${FAKE_LEGACY_IMAGE:-pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime}" "${FAKE_LEGACY_VOL:-synoptics-gpu-repro}" "${FAKE_LEGACY_CMD:-python3 /v/tracker/service/service.py}" ;;
      foreign)        printf '%s|/synoptics-tracker|postgres:16|somebody|db|data:/var/lib/postgresql/data,|postgres \n' "$id" ;;
    esac
    exit 0 ;;
  rm)
    target=${!#}
    printf '%s\n' "$target" >>"$ST/rm.log"
    [[ $target =~ ^[0-9a-f]{64}$ ]] || printf '%s\n' "$target" >>"$ST/rm_bad.log"
    [[ ${FAKE_RM_INEFFECTIVE:-0} == 1 ]] || : >"$ST/removed"
    exit 0 ;;
  logs) [[ ${FAKE_LOGS:-up} == up ]] || { echo "Error: fake logs failure" >&2; exit 1; }; echo "fake service log line"; exit 0 ;;
  exec)
    [[ ${FAKE_HEALTH:-up} == up ]] || exit 1
    printf '{"ready": true, "runs": 0}\n'; exit 0 ;;
  run)
    argv="$*"
    printf '%s\n' "$argv" >>"$ST/run.log"
    if [[ ${1:-} == -d ]]; then
      [[ ${FAKE_RUN_FAIL:-0} == 1 ]] && { echo "docker: Error response from daemon: fake run failure" >&2; exit 125; }
      printf '%s\n' "${FAKE_NEW_ID:-unknown-new-id}"; exit 0
    fi
    cat >"$ST/pushed.tar.gz"; printf 'fake pushed\n'; exit 0 ;;
esac
echo "fake docker: unsupported invocation: $sub" >&2
exit 2
EOS
chmod +x "$SHIM/ssh" "$SHIM/docker"

# ---------------------------------------------------------------- case runner
pass=0 fail=0
chk() { # chk <desc> <expected> <actual>
  if [[ $2 == "$3" ]]; then pass=$((pass+1)); printf 'ok   %s\n' "$1"
  else fail=$((fail+1)); printf 'FAIL %s\n  expected: %s\n  actual:   %s\n' "$1" "$2" "$3"; fi
}
chk_has() { # chk_has <desc> <needle> <file>
  if grep -qF -- "$2" "$3" 2>/dev/null; then pass=$((pass+1)); printf 'ok   %s\n' "$1"
  else fail=$((fail+1)); printf 'FAIL %s\n  needle not found: %s\n  in: %s\n' "$1" "$2" "$(cat "$3" 2>/dev/null | head -40)"; fi
}
chk_empty() { # chk_empty <desc> <file>
  if [[ ! -s $2 ]]; then pass=$((pass+1)); printf 'ok   %s\n' "$1"
  else fail=$((fail+1)); printf 'FAIL %s\n  expected empty, got:\n%s\n' "$1" "$(head -20 "$2")"; fi
}

RC=0 OUT=''
run_case() { # run_case <extra-env-string> [args...]
  local envs=$1; shift
  ST=$W/state; rm -rf "$ST"; mkdir -p "$ST"; : >"$ST/rm.log"; : >"$ST/ssh.log"; : >"$ST/run.log"
  export ST PATH="$SHIM:$PATH"
  export TRACKER_SSH_HOST=fakehost TRACKER_CONTAINER=synoptics-tracker TRACKER_PORT=8090 \
         TRACKER_VOLUME=synoptics-gpu-repro TRACKER_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime \
         TRACKER_HEALTH_WAIT=4 FAKE_LEGACY_ID=$LEGACY_ID FAKE_OWNED_ID=$OWNED_ID FAKE_NEW_ID=$NEW_ID
  unset FAKE_MODE FAKE_RM_INEFFECTIVE FAKE_HEALTH FAKE_RUN_FAIL FAKE_LOGS FAKE_LEGACY_IMAGE FAKE_LEGACY_CMD \
        FAKE_LEGACY_VOL FAKE_IMAGE FAKE_VOL FAKE_CMD TRACKER_OWNER FAKE_MODE_MALFORMED 2>/dev/null || true
  # shellcheck disable=SC2086
  env $envs bash "$SCRIPT" "$@" >"$W/out.log" 2>&1; RC=$?
  OUT=$(cat "$W/out.log")
}

echo "== guard verification (stubbed docker/ssh; the real script, no daemon) =="

# 1. absent container: nothing to stop, no removal, success
run_case "FAKE_MODE=notfound" --stop
chk "absent --stop exits 0" 0 "$RC"
chk_has "absent --stop says nothing to stop" "컨테이너가 없습니다" "$W/out.log"
chk_empty "absent --stop removed nothing" "$ST/rm.log"

# 2. foreign labels: refuse, no removal
run_case "FAKE_MODE=foreign" --stop
chk "foreign --stop exits 1" 1 "$RC"
chk_has "foreign --stop names the ownership refusal" "소유권 라벨이 우리 것이 아닙니다" "$W/out.log"
chk_has "foreign --stop prints the foreign owner" "somebody" "$W/out.log"
chk_empty "foreign --stop removed nothing" "$ST/rm.log"

# 3. partially matching labels (our owner, different role): still not ours
run_case "FAKE_MODE=owned_otherrole" --stop
chk "other-role --stop exits 1" 1 "$RC"
chk_has "other-role --stop refuses" "소유권 라벨이 우리 것이 아닙니다" "$W/out.log"
chk_empty "other-role --stop removed nothing" "$ST/rm.log"

# 3b. our owner label present but the role key is missing (docker prints `<no value>`) => not ours
run_case "FAKE_MODE=half_labelled" --stop
chk "half-labelled --stop exits 1" 1 "$RC"
chk_has "half-labelled --stop refuses" "소유권 라벨이 우리 것이 아닙니다" "$W/out.log"
chk_empty "half-labelled --stop removed nothing" "$ST/rm.log"

# 4. unlabelled legacy container: refuse automatic replacement, offer the explicit migration
run_case "FAKE_MODE=unlabelled" --stop
chk "unlabelled --stop exits 1" 1 "$RC"
chk_has "unlabelled --stop refuses" "무라벨" "$W/out.log"
chk_has "unlabelled --stop offers --migrate" "--migrate $LEGACY_ID" "$W/out.log"
chk_empty "unlabelled --stop removed nothing" "$ST/rm.log"

# 5. inspect permission failure: abort, no removal
run_case "FAKE_MODE=perm" --stop
chk "perm --stop exits 1" 1 "$RC"
chk_has "perm --stop reports the unexpected inspect error" "예상치 못한 docker inspect 오류" "$W/out.log"
chk_has "perm --stop echoes the daemon message" "permission denied" "$W/out.log"
chk_empty "perm --stop removed nothing" "$ST/rm.log"

# 6. daemon down: abort, no removal
run_case "FAKE_MODE=daemon" --stop
chk "daemon-down --stop exits 1" 1 "$RC"
chk_has "daemon-down --stop reports the unexpected inspect error" "예상치 못한 docker inspect 오류" "$W/out.log"
chk_empty "daemon-down --stop removed nothing" "$ST/rm.log"

# 7. malformed ID from inspect: refused as unverified
run_case "FAKE_MODE=malformed" --stop
chk "malformed-id --stop exits 1" 1 "$RC"
chk_has "malformed-id --stop refuses" "예상 형식이 아닙니다" "$W/out.log"
chk_empty "malformed-id --stop removed nothing" "$ST/rm.log"

# 8. owned container: targeted removal by full immutable ID, verified gone
run_case "FAKE_MODE=owned" --stop
chk "owned --stop exits 0" 0 "$RC"
chk "owned --stop removed exactly once" 1 "$(wc -l <"$ST/rm.log" | tr -d ' ')"
chk "owned --stop removed the full ID" "$OWNED_ID" "$(cat "$ST/rm.log")"
chk_empty "owned --stop never passed a name to docker rm" "$ST/rm_bad.log"
chk_has "owned --stop verified the end state" "제거 완료: $OWNED_ID" "$W/out.log"

# 9. removal reported success but the container is still there: must not claim success
run_case "FAKE_MODE=owned FAKE_RM_INEFFECTIVE=1" --stop
chk "ineffective-rm --stop exits 1" 1 "$RC"
chk_has "ineffective-rm --stop reports the surviving container" "아직 존재합니다" "$W/out.log"
chk "ineffective-rm --stop attempted one removal" 1 "$(wc -l <"$ST/rm.log" | tr -d ' ')"

# 10. reserved name (the llama co-tenant): refused before any remote call
run_case "FAKE_MODE=owned TRACKER_CONTAINER=synoptics-llama-b11146" --stop
chk "reserved-name exits 1" 1 "$RC"
chk_has "reserved-name refuses" "예약된 컨테이너 이름" "$W/out.log"
chk_empty "reserved-name made no remote call" "$ST/ssh.log"

# 11. hostile env value: rejected before any remote call
run_case "FAKE_MODE=owned TRACKER_PORT=8090;touch$W/pwned" --stop
chk "hostile-port exits 1" 1 "$RC"
chk_has "hostile-port rejects the value" "허용된 형식이 아닙니다" "$W/out.log"
chk_empty "hostile-port made no remote call" "$ST/ssh.log"
chk "hostile-port created no file" 0 "$([[ -e $W/pwned ]] && echo 1 || echo 0)"

# 12. start when a foreign container holds the name: refuse, no run, no removal
run_case "FAKE_MODE=foreign" ''
chk "start/foreign exits 1" 1 "$RC"
chk_has "start/foreign refuses" "소유권 라벨이 우리 것이 아닙니다" "$W/out.log"
chk_empty "start/foreign removed nothing" "$ST/rm.log"
chk_empty "start/foreign started nothing" "$ST/run.log"

# 13. start with an owned container present: replace by ID, then start with the ownership labels
run_case "FAKE_MODE=owned" ''
chk "start/owned exits 0" 0 "$RC"
chk "start/owned removed the old container by ID" "$OWNED_ID" "$(cat "$ST/rm.log")"
chk_has "start/owned labels the service owner" "--label synoptics.owner=aisw-hybrid" "$ST/run.log"
chk_has "start/owned labels the service role" "--label synoptics.role=tracker-service" "$ST/run.log"
chk_has "start/owned keeps the local log driver" "--log-driver local" "$ST/run.log"
chk_has "start/owned binds the Docker-partition volume" "-v synoptics-gpu-repro:/v" "$ST/run.log"
chk_has "start/owned reports the health payload" '"runs": 0' "$W/out.log"

# 14. start where the service never answers /health: reported, not silently declared healthy
run_case "FAKE_MODE=owned FAKE_HEALTH=down" ''
chk "start/unhealthy exits 1" 1 "$RC"
chk_has "start/unhealthy says so honestly" "health가 응답하지 않았습니다" "$W/out.log"

# 14b. the health-timeout path still reports honestly when even the log dump fails
run_case "FAKE_MODE=owned FAKE_HEALTH=down FAKE_LOGS=down" ''
chk "start/unhealthy/no-logs exits 1" 1 "$RC"
chk_has "start/unhealthy/no-logs still reports the timeout" "health가 응답하지 않았습니다" "$W/out.log"
chk_has "start/unhealthy/no-logs says the log dump failed" "로그를 읽지 못했습니다" "$W/out.log"

# 15. --migrate rejects a name (only a recorded full ID is accepted), with no remote call
run_case "FAKE_MODE=unlabelled" --migrate synoptics-tracker
chk "migrate/name exits 1" 1 "$RC"
chk_has "migrate/name demands the full ID" "64 hex" "$W/out.log"
chk_empty "migrate/name made no remote call" "$ST/ssh.log"

# 16. --migrate with mismatched evidence (wrong image): refuse, no removal
run_case "FAKE_MODE=unlabelled FAKE_LEGACY_IMAGE=ubuntu:22.04" --migrate "$LEGACY_ID"
chk "migrate/bad-image exits 1" 1 "$RC"
chk_has "migrate/bad-image reports the mismatch" "이미지 불일치" "$W/out.log"
chk_empty "migrate/bad-image removed nothing" "$ST/rm.log"

# 17. --migrate with mismatched volume binding: refuse, no removal
run_case "FAKE_MODE=unlabelled FAKE_LEGACY_VOL=some-other-volume" --migrate "$LEGACY_ID"
chk "migrate/bad-volume exits 1" 1 "$RC"
chk_has "migrate/bad-volume reports the mismatch" "볼륨 바인딩 불일치" "$W/out.log"
chk_empty "migrate/bad-volume removed nothing" "$ST/rm.log"

# 18. --migrate with a mismatched command: refuse, no removal
run_case "FAKE_MODE=unlabelled FAKE_LEGACY_CMD=/v/tracker/service/other.py" --migrate "$LEGACY_ID"
chk "migrate/bad-cmd exits 1" 1 "$RC"
chk_has "migrate/bad-cmd reports the mismatch" "실행 명령 불일치" "$W/out.log"
chk_empty "migrate/bad-cmd removed nothing" "$ST/rm.log"

# 19. --migrate on an already-labelled container: one-time legacy path only, no removal
run_case "FAKE_MODE=owned FAKE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime" --migrate "$OWNED_ID"
chk "migrate/owned exits 1" 1 "$RC"
chk_has "migrate/owned refuses" "--migrate는 무라벨 레거시" "$W/out.log"
chk_empty "migrate/owned removed nothing" "$ST/rm.log"

# 20. --migrate with fully re-validated evidence: remove exactly that ID, then start labelled
run_case "FAKE_MODE=unlabelled" --migrate "$LEGACY_ID"
chk "migrate/ok exits 0" 0 "$RC"
chk "migrate/ok removed exactly the confirmed ID" "$LEGACY_ID" "$(cat "$ST/rm.log")"
chk_has "migrate/ok printed the re-validated evidence" "컨테이너 재검증 통과" "$W/out.log"
chk_has "migrate/ok started the labelled service" "--label synoptics.role=tracker-service" "$ST/run.log"

# 21. --push: sources over SSH stdin, labelled helper, no base64 on any command line
run_case "FAKE_MODE=owned" --push
chk "push exits 0" 0 "$RC"
chk "push streamed the tar to the remote stdin" 1 "$([[ -s $ST/pushed.tar.gz ]] && echo 1 || echo 0)"
chk_has "push uses the labelled helper (owner)" "--label synoptics.owner=aisw-hybrid" "$ST/run.log"
chk_has "push uses the labelled helper (role)" "--label synoptics.role=tracker-helper" "$ST/run.log"
chk_has "push helper reads stdin" "--rm -i" "$ST/run.log"
chk "push never base64s the sources onto a command line" 0 "$(grep -c base64 "$ST/ssh.log" || true)"

# 22. side-by-side instance: own name, own port, own source dir, extra env passed through verbatim
SPIKE_ENV="FAKE_MODE=notfound TRACKER_CONTAINER=synoptics-tracker-spike TRACKER_SERVICE_DIR=/v/tracker/service-spike"
run_case "$SPIKE_ENV" --port 8093 --env TRACKER_OCCLUDED_MAX_FRAMES=30
chk "spike/start exits 0" 0 "$RC"
chk_empty "spike/start removed nothing" "$ST/rm.log"
chk_has "spike/start probed only its own name" "synoptics-tracker-spike" "$ST/ssh.log"
chk "spike/start never probed the production name" 0 "$(grep -cE "inspect .*'synoptics-tracker'" "$ST/ssh.log" || true)"
chk_has "spike/start binds the requested port" "-e TRACKER_PORT=8093" "$ST/run.log"
chk_has "spike/start passes the extra env" "-e TRACKER_OCCLUDED_MAX_FRAMES=30" "$ST/run.log"
chk_has "spike/start runs its own sources" "python3 /v/tracker/service-spike/service.py" "$ST/run.log"
chk_has "spike/start imports its own sources" "PYTHONPATH=/v/tracker/pydeps:/v/tracker/src:/v/tracker/service-spike" "$ST/run.log"
chk_has "spike/start still labels the service role" "--label synoptics.role=tracker-service" "$ST/run.log"

# 23. --push for the side-by-side instance writes only its own source dir
run_case "$SPIKE_ENV" --port 8093 --push
chk "spike/push exits 0" 0 "$RC"
chk_has "spike/push targets its own dir" "tar -C /v/tracker/service-spike -xzf -" "$ST/ssh.log"
chk "spike/push never writes the production dir" 0 "$(grep -c 'tar -C /v/tracker/service ' "$ST/ssh.log" || true)"

# 24. the default (production) name on another port would replace production: refused, no remote call
run_case "FAKE_MODE=owned" --port 8093
chk "default-name/other-port exits 1" 1 "$RC"
chk_has "default-name/other-port refuses" "포트 8090 전용" "$W/out.log"
chk_empty "default-name/other-port made no remote call" "$ST/ssh.log"
chk_empty "default-name/other-port removed nothing" "$ST/rm.log"

# 24b. the default name from another source dir: refused, no remote call
run_case "FAKE_MODE=owned TRACKER_SERVICE_DIR=/v/tracker/service-spike" ''
chk "default-name/other-dir exits 1" 1 "$RC"
chk_empty "default-name/other-dir made no remote call" "$ST/ssh.log"

# 25. --env guards: non-TRACKER key, reserved key, hostile value, malformed pair -> refused before any remote call
run_case "$SPIKE_ENV" --port 8093 --env LD_PRELOAD=/tmp/x.so
chk "env/foreign-key exits 1" 1 "$RC"
chk_empty "env/foreign-key made no remote call" "$ST/ssh.log"
run_case "$SPIKE_ENV" --port 8093 --env TRACKER_PORT=8090
chk "env/reserved-key exits 1" 1 "$RC"
chk_has "env/reserved-key refuses" "덮어쓸 수 없는 키" "$W/out.log"
chk_empty "env/reserved-key made no remote call" "$ST/ssh.log"
run_case "$SPIKE_ENV" --port 8093 --env "TRACKER_X=1;touch$W/pwned2"
chk "env/hostile-value exits 1" 1 "$RC"
chk_empty "env/hostile-value made no remote call" "$ST/ssh.log"
chk "env/hostile-value created no file" 0 "$([[ -e $W/pwned2 ]] && echo 1 || echo 0)"
run_case "$SPIKE_ENV" --port 8093 --env TRACKER_OCCLUDED_MAX_FRAMES
chk "env/no-equals exits 1" 1 "$RC"
chk_empty "env/no-equals made no remote call" "$ST/ssh.log"

# 26. hostile source dir: refused before any remote call
run_case "FAKE_MODE=notfound TRACKER_CONTAINER=synoptics-tracker-spike TRACKER_SERVICE_DIR=/v/tracker/../x" --port 8093
chk "service-dir/traversal exits 1" 1 "$RC"
chk_empty "service-dir/traversal made no remote call" "$ST/ssh.log"

# 27. two actions at once are refused
run_case "FAKE_MODE=owned" --stop --push
chk "two-actions exits 1" 1 "$RC"
chk_empty "two-actions made no remote call" "$ST/ssh.log"

# 28. the default SSH host is the direct IP (the vast-gpu-node alias hangs from the dev host)
run_case "-u TRACKER_SSH_HOST FAKE_MODE=notfound" --stop
chk "default-host --stop exits 0" 0 "$RC"
chk_has "default-host uses the direct IP" "riey@192.168.219.250" "$ST/ssh.log"

printf '\n== %d passed, %d failed ==\n' "$pass" "$fail"
(( fail == 0 )) || exit 1
