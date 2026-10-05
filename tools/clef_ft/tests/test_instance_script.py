"""``run_on_instance.sh`` pipeline and watchdog, with a stub python (downloads and train.py runs faked) and a
stub ``vastai`` that records its arguments. Real python still runs ``ftutil.py``."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_on_instance.sh"

FAKE_PYTHON = r"""#!/usr/bin/env bash
REAL="{real}"
world=1
if [[ "${{1:-}}" == -m && "${{2:-}}" == torch.distributed.run ]]; then
  shift 2
  while [[ $# -gt 0 && "$1" != */train.py ]]; do [[ "$1" == --nproc_per_node ]] && {{ world="$2"; shift; }}; shift; done
  world="${{FAKE_WORLD_OVERRIDE:-$world}}"
fi
if [[ "${{1:-}}" == */train.py ]]; then
  echo "world=$world $*" >> "{launches}"
  shift; model=""; arm=""; out=""
  while [[ $# -gt 0 ]]; do
    case "$1" in --model) model="$2"; shift 2;; --arm) arm="$2"; shift 2;; --out) out="$2"; shift 2;; *) shift;; esac
  done
  echo "fake train $model $arm"
  [[ "${{model}}_${{arm}}" == "${{FAIL_RUN:-}}" ]] && {{ echo "boom: CUDA out of memory"; exit 3; }}
  mkdir -p "$out"
  if [[ "$arm" == lora ]]; then echo "{{\"train_rows_per_s\": 1.0, \"eval_rows_per_s\": 2.0, \"world_size\": $world}}" > "$out/throughput.json"; fi
  echo "{{\"model\": \"$model\", \"arm\": \"$arm\", \"seconds\": 1, \"max_gpu_memory_gb\": 1.5, \"selected\": {{\"epoch\": 1}}}}" > "$out/result.json"
  exit 0
fi
if [[ "${{1:-}}" == -c && "${{2:-}}" == *snapshot_download* ]]; then
  dir="$(sed -n "s/.*local_dir='\([^']*\)'.*/\1/p" <<< "$2")"
  [[ "$dir" == *"/${{FAIL_FETCH:-none}}" ]] && exit 4
  mkdir -p "$dir" && touch "$dir/joint_head.safetensors"; exit 0
fi
exec "$REAL" "$@"
"""

FAKE_VASTAI = """#!/usr/bin/env bash
echo "$@" >> "{record}"
"""


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, object]:
    ws = tmp_path / "ws"
    (ws / "out").mkdir(parents=True)
    (ws / ".setup_done").touch()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_py = bindir / "python"
    launches = tmp_path / "launches"
    fake_py.write_text(FAKE_PYTHON.format(real=sys.executable, launches=launches))
    record = tmp_path / "vastai_calls"
    fake_vast = bindir / "vastai"
    fake_vast.write_text(FAKE_VASTAI.format(record=record))
    for f in (fake_py, fake_vast):
        f.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CONTAINER_")}
    env.update({"CLEF_FT_WORKSPACE": str(ws), "PYTHON": str(fake_py), "CLEF_FT_VASTAI": str(fake_vast), "CLEF_FT_NPROC": "1",
                "CLEF_FT_WATCHDOG_POLL_S": "1", "CLEF_FT_DESTROY_RETRY_S": "0"})
    yield {"ws": ws, "out": ws / "out", "env": env, "record": record, "launches": launches}
    for name in ("pipeline.pid", "watchdog.pid"):
        pid_file = ws / "out" / name
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass


def sh(env: dict[str, str], *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60,
                          check=check)


def wait_for(*paths: Path, timeout: float = 40.0) -> Path:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for path in paths:
            if path.exists():
                time.sleep(0.3)
                return path
        time.sleep(0.2)
    raise AssertionError(f"none of {paths} appeared")


def steps(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]


def test_pipeline_runs_every_step_and_writes_done(workspace) -> None:
    out, env = workspace["out"], workspace["env"]
    sh(env, "pipeline", "0.02")  # 72 s: the optional clef-flash runs do not fit
    done = wait_for(out / "DONE", out / "FAILED")
    assert done.name == "DONE", (out / "FAILED").read_text() if (out / "FAILED").exists() else ""
    summary = json.loads(done.read_text())
    names = [s["step"] for s in summary["steps"]]
    assert names == ["fetch_clef-flash", "smoke", "fetch_clef", "clef_zero", "clef_head", "clef_lora",
                     "clef-flash_head", "clef-flash_lora"]
    status = {s["step"]: s["status"] for s in summary["steps"]}
    assert status["clef_lora"] == "ok" and status["clef-flash_head"] == "skipped"
    assert summary["ok"] is True
    assert summary["runs"]["clef_lora"]["max_gpu_memory_gb"] == 1.5
    assert (workspace["ws"] / "models" / "clef" / "joint_head.safetensors").exists()
    assert "marker: DONE" in sh(env, "status").stdout


def test_pipeline_runs_optional_steps_when_time_allows(workspace) -> None:
    out, env = workspace["out"], workspace["env"]
    sh(env, "pipeline", "10")
    assert wait_for(out / "DONE", out / "FAILED").name == "DONE"
    status = {s["step"]: s["status"] for s in steps(out)}
    assert status["clef-flash_head"] == "ok" and status["clef-flash_lora"] == "ok"


def test_pipeline_stops_at_the_first_failure(workspace) -> None:
    out, env = workspace["out"], workspace["env"]
    sh({**env, "FAIL_RUN": "clef_head"}, "pipeline", "10")
    assert wait_for(out / "DONE", out / "FAILED").name == "FAILED"
    failed = json.loads((out / "FAILED").read_text())
    assert failed["step"] == "clef_head" and "CUDA out of memory" in failed["log_tail"]
    assert [s["step"] for s in steps(out)][-1] == "clef_head"
    assert not (out / "clef_lora").exists() and not (out / "DONE").exists()
    # A second pipeline refuses to start over the FAILED marker.
    again = sh(env, "pipeline", "1", check=False)
    assert again.returncode != 0 and "FAILED exists" in again.stderr


def test_pipeline_fails_when_the_parallel_27b_download_fails(workspace) -> None:
    out, env = workspace["out"], workspace["env"]
    sh({**env, "FAIL_FETCH": "clef"}, "pipeline", "10")
    assert wait_for(out / "DONE", out / "FAILED").name == "FAILED"
    assert json.loads((out / "FAILED").read_text())["step"] == "fetch_clef"
    assert not (out / "clef_zero").exists()


def test_pipeline_on_two_gpus_launches_torchrun(workspace) -> None:
    out, env = workspace["out"], workspace["env"]
    sh({**env, "CLEF_FT_NPROC": "2"}, "pipeline", "0.02")
    assert wait_for(out / "DONE", out / "FAILED").name == "DONE"
    launches = workspace["launches"].read_text().splitlines()
    assert launches and all(line.startswith("world=2 ") for line in launches)
    smoke = next(line for line in launches if "smoke_flash_lora" in line)
    assert "--max-steps 20 --limit 8" in smoke
    assert "ranks per run: 2" in sh({**env, "CLEF_FT_NPROC": "2"}, "status").stdout


def test_one_gpu_keeps_the_plain_launch(workspace) -> None:
    out, env = workspace["out"], workspace["env"]
    sh(env, "pipeline", "0.02")
    assert wait_for(out / "DONE", out / "FAILED").name == "DONE"
    launches = workspace["launches"].read_text().splitlines()
    assert all(line.startswith("world=1 ") for line in launches)
    assert "--max-steps 10 --limit 4" in next(line for line in launches if "smoke_flash_lora" in line)


def test_smoke_fails_when_the_ranks_do_not_match(workspace) -> None:
    out, env = workspace["out"], workspace["env"]
    sh({**env, "CLEF_FT_NPROC": "2", "FAKE_WORLD_OVERRIDE": "1"}, "pipeline", "10")
    assert wait_for(out / "DONE", out / "FAILED").name == "FAILED"
    failed = json.loads((out / "FAILED").read_text())
    assert failed["step"] == "smoke" and "did not run on 2 ranks" in failed["log_tail"]


def watchdog(workspace, deadline_s: int, **env_extra: str) -> subprocess.CompletedProcess:
    out = workspace["out"]
    (out / "pipeline_start").write_text(str(int(time.time())))
    return subprocess.run(["bash", str(SCRIPT), "_watchdog", str(deadline_s)],
                          env={**workspace["env"], **env_extra}, capture_output=True, text=True, timeout=30)


def test_watchdog_destroys_at_the_deadline_when_nothing_was_pulled(workspace) -> None:
    result = watchdog(workspace, 2, CONTAINER_ID="4242", CONTAINER_API_KEY="k-self")
    assert workspace["record"].read_text().split() == ["destroy", "instance", "4242", "-y", "--api-key", "k-self"]
    assert "deadline" in json.loads((workspace["out"] / "DESTROYING").read_text())["reason"]
    assert "destroy requested" in result.stdout


def test_watchdog_destroys_when_done_is_not_pulled_within_the_grace(workspace) -> None:
    (workspace["out"] / "DONE").write_text("{}")
    watchdog(workspace, 3600, CONTAINER_ID="7", CONTAINER_API_KEY="k", CLEF_FT_PULL_GRACE_S="1")
    assert "not pulled" in json.loads((workspace["out"] / "DESTROYING").read_text())["reason"]
    assert workspace["record"].read_text().split()[:3] == ["destroy", "instance", "7"]


def test_watchdog_exits_quietly_once_pulled(workspace) -> None:
    (workspace["out"] / "FAILED").write_text("{}")
    (workspace["out"] / "PULLED").touch()
    result = watchdog(workspace, 1, CONTAINER_ID="7", CONTAINER_API_KEY="k", CLEF_FT_PULL_GRACE_S="0")
    assert "pulled after FAILED" in result.stdout
    assert not workspace["record"].exists() and not (workspace["out"] / "DESTROYING").exists()


def test_watchdog_hard_cap_ignores_pulled_without_done(workspace) -> None:
    (workspace["out"] / "PULLED").touch()
    watchdog(workspace, 1, CONTAINER_ID="9", CONTAINER_API_KEY="k", CLEF_FT_HARD_EXTRA_S="1")
    assert "hard cap" in json.loads((workspace["out"] / "DESTROYING").read_text())["reason"]


def test_watchdog_without_the_vast_key_only_warns(workspace) -> None:
    result = watchdog(workspace, 1)
    assert "cannot self-destroy" in result.stdout
    assert not workspace["record"].exists()
