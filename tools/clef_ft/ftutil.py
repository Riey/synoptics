"""Stdlib helpers for the Clef fine-tuning runs (checkpoint selection, time estimates, pipeline summaries).

CLI (used by ``run_on_instance.sh``):
  ftutil.py estimate THROUGHPUT_JSON ARM [DATASET_JSONL [EPOCHS]]
                                                seconds a full ARM run needs at the measured (global) rates;
                                                split sizes from DATASET_JSONL (default: the v1 sizes)
  ftutil.py step OUT NAME STATUS START END [LOG] append one step line to OUT/steps.jsonl
  ftutil.py done OUT                            write OUT/DONE (summary JSON) from steps.jsonl + run results
  ftutil.py failed OUT STEP LOG                 write OUT/FAILED (step, time, log tail)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

SPLIT_ROWS = {"train": 646, "val": 81, "test": 119}
EPOCHS = {"head": 10, "lora": 3}
#: Head-only steps on cached hidden states are cheap next to a backbone forward; a generous per-row guess.
HEAD_ROW_S = 0.05
LOAD_S = 600.0
TAIL_LINES = 60


def is_near(row: dict[str, Any]) -> bool:
    """A near / mined hard-negative row: ``sample_kind`` ``near_*`` when the dataset carries it, else the
    ``near_`` sample id prefix (build_data v4b rows have no ``sample_kind``)."""
    kind = row.get("sample_kind")
    if kind is not None:
        return str(kind).startswith("near")
    return str(row["sample_id"]).startswith("near_")


def split_rows(data: Path) -> dict[str, int]:
    """Rows per split of a ``dataset.jsonl``."""
    counts = dict.fromkeys(SPLIT_ROWS, 0)
    with data.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                split = json.loads(line)["split"]
                counts[split] = counts.get(split, 0) + 1
    return counts


def select_checkpoint(epochs: list[dict[str, Any]], zero_false_yes: float | None) -> dict[str, Any]:
    """Pick the epoch with the best val current-step key accuracy among those whose val false ``yes`` rate (all
    keys) is at most the zero-shot val rate. Without a zero-shot rate, or when no epoch meets it, the best
    accuracy overall (``constraint_met`` says which). Ties go to the earlier epoch.

    ``epochs``: ``[{"epoch": n, "val": metrics.compute(...)}, ...]``.
    """
    if not epochs:
        raise ValueError("no epochs to select from")

    def acc(entry: dict[str, Any]) -> float:
        return entry["val"]["current_acc"]["rate"] or 0.0

    def fy(entry: dict[str, Any]) -> float:
        return entry["val"]["false_yes"]["rate"] or 0.0

    eligible = epochs if zero_false_yes is None else [e for e in epochs if fy(e) <= zero_false_yes + 1e-12]
    pool = eligible or epochs
    best = max(pool, key=lambda e: (acc(e), -e["epoch"]))
    return {
        "epoch": best["epoch"],
        "val_current_acc": best["val"]["current_acc"],
        "val_false_yes": best["val"]["false_yes"],
        "zero_val_false_yes": zero_false_yes,
        "constraint_met": zero_false_yes is None or bool(eligible),
        "rule": "max val current-step key accuracy s.t. val false-yes rate <= zero-shot val rate",
    }


def estimate_seconds(throughput: dict[str, Any], arm: str, splits: dict[str, int] | None = None,
                     epochs: int | None = None) -> float:
    """Full-run seconds for ``arm`` from a smoke run's ``throughput.json`` (``train_rows_per_s``,
    ``eval_rows_per_s``: rows/s over all ranks) on the same model, for ``splits`` rows (default the v1 sizes)
    and ``epochs`` (default ``EPOCHS[arm]``)."""
    train_rps = float(throughput["train_rows_per_s"])
    eval_rps = float(throughput["eval_rows_per_s"])
    splits = splits or SPLIT_ROWS
    eval_rows = splits["val"] + splits["test"]
    if arm == "zero":
        return LOAD_S + eval_rows / eval_rps
    if arm == "head":
        rows = sum(splits.values())
        return LOAD_S + rows / eval_rps + (epochs or EPOCHS["head"]) * (splits["train"] + splits["val"]) * HEAD_ROW_S
    if arm == "lora":
        n = epochs or EPOCHS["lora"]
        return LOAD_S + n * splits["train"] / train_rps + (n + 1) * splits["val"] / eval_rps + splits["test"] / eval_rps
    raise ValueError(f"unknown arm {arm!r}")


def append_step(out: Path, name: str, status: str, start: float, end: float, log: str | None = None) -> None:
    entry = {"step": name, "status": status, "start": start, "end": end, "seconds": round(end - start, 1)}
    if log:
        entry["log"] = log
    with (out / "steps.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def summarize(out: Path) -> dict[str, Any]:
    steps = []
    if (out / "steps.jsonl").exists():
        steps = [json.loads(line) for line in (out / "steps.jsonl").read_text(encoding="utf-8").splitlines() if line]
    runs = {}
    for result in sorted(out.glob("*/result.json")):
        data = _read_json(result) or {}
        runs[result.parent.name] = {
            key: data.get(key) for key in ("model", "arm", "seconds", "max_gpu_memory_gb", "selected",
                                           "throughput", "val_metrics", "test_metrics")
        }
    return {"finished_at": time.time(), "ok": all(s["status"] in ("ok", "skipped") for s in steps),
            "steps": steps, "runs": runs}


def log_tail(path: Path, lines: int = TAIL_LINES) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError as exc:
        return f"(no log: {exc})"


def main(argv: list[str]) -> None:
    command, args = argv[0], argv[1:]
    if command == "estimate":
        splits = split_rows(Path(args[2])) if len(args) > 2 else None
        epochs = int(args[3]) if len(args) > 3 else None
        print(round(estimate_seconds(json.loads(Path(args[0]).read_text()), args[1], splits, epochs)))
    elif command == "step":
        append_step(Path(args[0]), args[1], args[2], float(args[3]), float(args[4]), args[5] if len(args) > 5 else None)
    elif command == "done":
        out = Path(args[0])
        (out / "DONE").write_text(json.dumps(summarize(out), ensure_ascii=False, indent=1) + "\n")
    elif command == "failed":
        out, step, log = Path(args[0]), args[1], Path(args[2])
        failed = {"step": step, "time": time.time(), "log": str(log), "log_tail": log_tail(log),
                  "summary": summarize(out)}
        (out / "FAILED").write_text(json.dumps(failed, ensure_ascii=False, indent=1) + "\n")
    else:
        raise SystemExit(f"unknown command {command!r}\n{__doc__}")


if __name__ == "__main__":
    main(sys.argv[1:])
