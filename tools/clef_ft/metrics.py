"""Follower metrics for Clef fine-tuning predictions (CPU, stdlib only).

A prediction row (``train.py`` writes these as JSONL): ``sample_id``, ``split``, ``domain``, ``current_step``,
``labels`` {key: yes/no/unsure}, ``pred`` {key: yes/no/unsure} and optionally ``probs``. The goal key is
``goal`` (``clef_follow.GOAL_KEY``; ``g`` in the label files).

Metrics: key accuracy, row exact match, false ``yes`` rate (predicted ``yes`` where the label is not ``yes``)
over all keys / the current step key / the goal key, current step key 3-way accuracy and confusion matrix,
``unsure`` precision and recall on the current step key (``unsure`` there calls the big model), and the same
headline numbers per domain.

Usage: python metrics.py PREDICTIONS.jsonl [--out PREFIX]   (writes PREFIX.json and PREFIX.md)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GOAL_KEY = "goal"
CLASSES = ("yes", "no", "unsure")


def ratio(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None}


def _pairs(rows: list[dict[str, Any]], which: str) -> list[tuple[str, str]]:
    out = []
    for row in rows:
        for key, label in row["labels"].items():
            if which == "all" or (which == "current" and key == row["current_step"]) or (
                    which == "goal" and key == GOAL_KEY):
                out.append((label, row["pred"][key]))
    return out


def false_yes(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    negatives = [pred for label, pred in pairs if label != "yes"]
    return ratio(sum(pred == "yes" for pred in negatives), len(negatives))


def accuracy(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    return ratio(sum(label == pred for label, pred in pairs), len(pairs))


def confusion(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """``matrix[label][pred]``."""
    matrix = {label: dict.fromkeys(CLASSES, 0) for label in CLASSES}
    for label, pred in pairs:
        matrix[label][pred] += 1
    return matrix


def headline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_pairs, current = _pairs(rows, "all"), _pairs(rows, "current")
    return {
        "rows": len(rows),
        "key_acc": accuracy(all_pairs),
        "row_exact": ratio(sum(all(row["pred"][k] == v for k, v in row["labels"].items()) for row in rows), len(rows)),
        "false_yes": false_yes(all_pairs),
        "current_acc": accuracy(current),
        "current_false_yes": false_yes(current),
    }


def compute(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        missing = set(row["labels"]) - set(row["pred"])
        if missing:
            raise ValueError(f"{row.get('sample_id')}: no prediction for {sorted(missing)}")
    current = _pairs(rows, "current")
    matrix = confusion(current)
    predicted_unsure = sum(matrix[label]["unsure"] for label in CLASSES)
    report = headline(rows)
    report.update({
        "goal_false_yes": false_yes(_pairs(rows, "goal")),
        "current_confusion": matrix,
        "current_unsure_precision": ratio(matrix["unsure"]["unsure"], predicted_unsure),
        "current_unsure_recall": ratio(matrix["unsure"]["unsure"], sum(matrix["unsure"].values())),
        "by_domain": {domain: headline([row for row in rows if row["domain"] == domain])
                      for domain in sorted({row["domain"] for row in rows})},
    })
    return report


def _fmt(value: dict[str, Any]) -> str:
    if value["rate"] is None:
        return f"{value['k']}/{value['n']}"
    return f"{value['k']}/{value['n']} ({value['rate'] * 100:.1f}%)"


def markdown(report: dict[str, Any], title: str = "Clef follower metrics") -> str:
    lines = [f"# {title}", "", f"rows {report['rows']}", "", "| metric | value |", "|---|---|"]
    for name in ("key_acc", "row_exact", "false_yes", "current_false_yes", "goal_false_yes", "current_acc",
                 "current_unsure_precision", "current_unsure_recall"):
        lines.append(f"| {name} | {_fmt(report[name])} |")
    lines += ["", "Current step key confusion (rows = label, columns = prediction):", "",
              "| label \\ pred | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for label in CLASSES:
        lines.append(f"| {label} | " + " | ".join(str(report["current_confusion"][label][p]) for p in CLASSES) + " |")
    lines += ["", "| domain | rows | key_acc | row_exact | false_yes | current_acc | current_false_yes |",
              "|---|---:|---|---|---|---|---|"]
    for domain, m in report["by_domain"].items():
        lines.append(f"| {domain} | {m['rows']} | {_fmt(m['key_acc'])} | {_fmt(m['row_exact'])} | "
                     f"{_fmt(m['false_yes'])} | {_fmt(m['current_acc'])} | {_fmt(m['current_false_yes'])} |")
    return "\n".join(lines) + "\n"


def read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_report(rows: list[dict[str, Any]], prefix: Path, title: str = "Clef follower metrics") -> dict[str, Any]:
    report = compute(rows)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    Path(f"{prefix}.md").write_text(markdown(report, title), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clef follower metrics from a predictions JSONL")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--out", type=Path, help="output prefix (default: predictions path without .jsonl)")
    args = parser.parse_args()
    prefix = args.out or Path(f"{args.predictions.with_suffix('')}_metrics")
    report = write_report(read_predictions(args.predictions), prefix, args.predictions.name)
    print(markdown(report, args.predictions.name))


if __name__ == "__main__":
    main()
