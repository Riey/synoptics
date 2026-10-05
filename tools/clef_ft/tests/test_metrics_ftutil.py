from __future__ import annotations

import json
from pathlib import Path

import pytest

import ftutil
import metrics as M


def row(sid: str, domain: str, current: str, labels: dict[str, str], pred: dict[str, str]) -> dict:
    return {"sample_id": sid, "split": "val", "domain": domain, "current_step": current, "labels": labels, "pred": pred}


ROWS = [
    # current s1: label yes, pred yes; s2 label no pred yes (false yes); goal no/no
    row("r1", "cooking", "s1", {"s1": "yes", "s2": "no", "goal": "no"}, {"s1": "yes", "s2": "yes", "goal": "no"}),
    # current s2: label unsure pred unsure; goal unsure pred yes (false yes on goal)
    row("r2", "cooking", "s2", {"s2": "unsure", "goal": "unsure"}, {"s2": "unsure", "goal": "yes"}),
    # current s1: label no pred unsure; goal no pred no
    row("r3", "assembly", "s1", {"s1": "no", "goal": "no"}, {"s1": "unsure", "goal": "no"}),
    # current s3: label unsure pred no (missed unsure); all else right
    row("r4", "assembly", "s3", {"s3": "unsure", "goal": "no"}, {"s3": "no", "goal": "no"}),
]


def test_metrics_by_hand() -> None:
    m = M.compute(ROWS)
    assert m["rows"] == 4
    assert (m["key_acc"]["k"], m["key_acc"]["n"]) == (5, 9)
    assert (m["row_exact"]["k"], m["row_exact"]["n"]) == (0, 4)
    # non-yes labels: r1 s2,goal; r2 s2,goal; r3 s1,goal; r4 s3,goal = 8; predicted yes: r1 s2, r2 goal
    assert (m["false_yes"]["k"], m["false_yes"]["n"]) == (2, 8)
    assert (m["current_false_yes"]["k"], m["current_false_yes"]["n"]) == (0, 3)
    assert (m["goal_false_yes"]["k"], m["goal_false_yes"]["n"]) == (1, 4)
    assert (m["current_acc"]["k"], m["current_acc"]["n"]) == (2, 4)
    cm = m["current_confusion"]
    assert cm["yes"] == {"yes": 1, "no": 0, "unsure": 0}
    assert cm["no"] == {"yes": 0, "no": 0, "unsure": 1}
    assert cm["unsure"] == {"yes": 0, "no": 1, "unsure": 1}
    assert (m["current_unsure_precision"]["k"], m["current_unsure_precision"]["n"]) == (1, 2)
    assert (m["current_unsure_recall"]["k"], m["current_unsure_recall"]["n"]) == (1, 2)
    assert m["by_domain"]["cooking"]["key_acc"]["k"] == 3 and m["by_domain"]["cooking"]["key_acc"]["n"] == 5
    assert m["by_domain"]["assembly"]["current_acc"]["k"] == 0


def test_metrics_missing_prediction_fails() -> None:
    with pytest.raises(ValueError, match="no prediction"):
        M.compute([row("x", "d", "s1", {"s1": "yes", "goal": "no"}, {"s1": "yes"})])


def test_metrics_report_files(tmp_path: Path) -> None:
    M.write_report(ROWS, tmp_path / "val_metrics")
    assert json.loads((tmp_path / "val_metrics.json").read_text())["rows"] == 4
    md = (tmp_path / "val_metrics.md").read_text()
    assert "| false_yes | 2/8 (25.0%) |" in md and "| unsure | 0 | 1 | 1 |" in md


def epoch(n: int, acc: float, fy: float) -> dict:
    return {"epoch": n, "val": {"current_acc": {"rate": acc}, "false_yes": {"rate": fy}}}


def test_select_checkpoint_respects_the_zero_shot_false_yes_cap() -> None:
    epochs = [epoch(1, 0.70, 0.05), epoch(2, 0.80, 0.12), epoch(3, 0.75, 0.10)]
    chosen = ftutil.select_checkpoint(epochs, zero_false_yes=0.10)
    assert chosen["epoch"] == 3 and chosen["constraint_met"] is True
    assert ftutil.select_checkpoint(epochs, zero_false_yes=None)["epoch"] == 2
    none_ok = ftutil.select_checkpoint(epochs, zero_false_yes=0.01)
    assert none_ok["epoch"] == 2 and none_ok["constraint_met"] is False
    tie = ftutil.select_checkpoint([epoch(1, 0.8, 0.0), epoch(2, 0.8, 0.0)], 0.1)
    assert tie["epoch"] == 1


def test_estimate_seconds() -> None:
    tp = {"train_rows_per_s": 1.0, "eval_rows_per_s": 2.0}
    assert ftutil.estimate_seconds(tp, "lora") == pytest.approx(600 + 3 * 646 + 4 * 81 / 2 + 119 / 2)
    assert ftutil.estimate_seconds(tp, "zero") == pytest.approx(600 + 200 / 2)
    assert ftutil.estimate_seconds(tp, "head") > ftutil.estimate_seconds(tp, "zero")


def test_estimate_seconds_on_other_splits_and_epochs(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    tp = tmp_path / "throughput.json"
    tp.write_text(json.dumps({"train_rows_per_s": 4.0, "eval_rows_per_s": 8.0}))
    data = tmp_path / "dataset.jsonl"
    data.write_text("".join(json.dumps({"split": s}) + "\n" for s in ["train"] * 40 + ["val"] * 8 + ["test"] * 16))
    assert ftutil.split_rows(data) == {"train": 40, "val": 8, "test": 16}
    ftutil.main(["estimate", str(tp), "lora", str(data), "2"])
    assert int(capsys.readouterr().out) == 600 + 2 * 40 / 4 + 3 * 8 / 8 + 16 / 8


def test_is_near() -> None:
    assert ftutil.is_near({"sample_id": "near_aid_x_r1_t1"})
    assert not ftutil.is_near({"sample_id": "aid_x_t1"})
    assert ftutil.is_near({"sample_id": "x", "sample_kind": "near_s2"})
    assert not ftutil.is_near({"sample_id": "near_x", "sample_kind": "mid_s1"})  # the field wins when present


def test_done_and_failed_summaries(tmp_path: Path) -> None:
    ftutil.append_step(tmp_path, "setup", "ok", 0.0, 5.0)
    ftutil.append_step(tmp_path, "flash_head", "skipped", 5.0, 5.0)
    (tmp_path / "clef_lora").mkdir()
    (tmp_path / "clef_lora" / "result.json").write_text(json.dumps(
        {"model": "clef", "arm": "lora", "seconds": 10, "max_gpu_memory_gb": 70.1, "selected": {"epoch": 2}}))
    ftutil.main(["done", str(tmp_path)])
    done = json.loads((tmp_path / "DONE").read_text())
    assert done["ok"] is True and [s["step"] for s in done["steps"]] == ["setup", "flash_head"]
    assert done["runs"]["clef_lora"]["selected"] == {"epoch": 2}
    log = tmp_path / "x.log"
    log.write_text("\n".join(f"line {i}" for i in range(100)))
    ftutil.main(["failed", str(tmp_path), "smoke", str(log)])
    failed = json.loads((tmp_path / "FAILED").read_text())
    assert failed["step"] == "smoke" and failed["log_tail"].splitlines()[-1] == "line 99"
    assert len(failed["log_tail"].splitlines()) == ftutil.TAIL_LINES
