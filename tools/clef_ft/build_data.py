"""Build the Clef fine-tuning set: ``dataset.jsonl`` + prepared images, packed as ``clef_ft_data.tar``.

Sources (``--src``, default ``~/.local/share/synoptics-astra-distill/multidomain``):

* ``examples.jsonl`` (826 rows: plan, current step, prepared image) + ``final_labels.jsonl`` (astra final labels,
  principles v2, same ``sample_id``);
* ``training_exclusions.jsonl``: sample ids left out (plans with a major issue);
* ``plan_fix_rows.jsonl`` + ``plan_fix_labels.jsonl``: replacement rows on the fixed plans;
* ``--extra NAME`` (repeatable): ``NAME_examples.jsonl`` + ``NAME_final_labels.jsonl``, a domain added later in
  its own files (e.g. ``soldering``, astra-only labels in the same final-label schema).

Each output row carries the Clef record (``state`` + ``questions``) built by ``backend.app.clef_follow.clef_record``
in ``choice`` mode with no anchor (the prepared frames have no tracker box) -- the same function the app's Clef
follower calls, so the training and serving wording cannot drift -- plus the labels, key weights (current step
key 2.0, others 1.0) and the image path inside the tar. Image bytes are checked against the recorded sha256.

An example that carries ``temporal_context`` is the distinct CONTEXTUAL state task (``backend.app.temporal_follow``,
context ``temporal-state-v2``): its ``record`` is the contextual student input, ``observation_record`` keeps the
frozen single-frame record, and the label must declare matching ``temporal_context_sha256``/``student_record_sha256``
-- the record hash preserves question order, so it binds the exact field order the native Clef encoder packs. An
old ``temporal-state-v1`` context is refused, never accepted as v2. Examples without it keep the legacy
observation row byte-for-byte.

Usage: python tools/clef_ft/build_data.py [--src DIR] [--out DIR] [--extra NAME ... --expected train=N,val=N,test=N]
       [--name NAME]   (writes DIR/NAME/ and DIR/NAME.tar; NAME defaults to clef_ft_data. Inside the tar the
       directory is always clef_ft_data/, which is what run_on_instance.sh extracts.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.app.clef_follow import GOAL_KEY, MODE_CHOICE, clef_record  # noqa: E402
from backend.app.temporal_follow import (  # noqa: E402
    TemporalContextError,
    contextual_clef_record,
    student_record_sha256,
    temporal_context_sha256,
)

DEFAULT_SRC = Path.home() / ".local/share/synoptics-astra-distill/multidomain"
DEFAULT_OUT = Path.home() / ".local/share/synoptics-astra-distill/clef-ft"
DATA_DIR = "clef_ft_data"
LABEL_GOAL = "g"  # the goal key in the label files
VERDICTS = ("yes", "no", "unsure")
CURRENT_WEIGHT, OTHER_WEIGHT = 2.0, 1.0
EXPECTED_SPLITS = {"train": 646, "val": 81, "test": 119}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_key(label_key: str) -> str:
    return GOAL_KEY if label_key == LABEL_GOAL else label_key


def merge_sources(src: Path, extra: tuple[str, ...] = ()) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(example, label) pairs: final labels minus the exclusions, plus the plan-fix rows and every ``extra``
    domain's rows. Sorted by sample id."""
    examples = {row["sample_id"]: row for row in read_jsonl(src / "examples.jsonl")}
    labels = read_jsonl(src / "final_labels.jsonl")
    excluded = {row["sample_id"] for row in read_jsonl(src / "training_exclusions.jsonl")}
    unknown = excluded - {row["sample_id"] for row in labels}
    if unknown:
        raise ValueError(f"exclusions name unknown sample ids: {sorted(unknown)[:5]}")
    pairs = [(examples[row["sample_id"]], row) for row in labels if row["sample_id"] not in excluded]

    fix_rows = {row["sample_id"]: row for row in read_jsonl(src / "plan_fix_rows.jsonl")}
    fix_labels = read_jsonl(src / "plan_fix_labels.jsonl")
    if set(fix_rows) != {row["sample_id"] for row in fix_labels}:
        raise ValueError("plan_fix_rows and plan_fix_labels name different sample ids")
    pairs += [(fix_rows[row["sample_id"]], row) for row in fix_labels]

    for name in extra:
        extra_rows = {row["sample_id"]: row for row in read_jsonl(src / f"{name}_examples.jsonl")}
        extra_labels = read_jsonl(src / f"{name}_final_labels.jsonl")
        if set(extra_rows) != {row["sample_id"] for row in extra_labels}:
            raise ValueError(f"{name}_examples and {name}_final_labels name different sample ids")
        pairs += [(extra_rows[row["sample_id"]], row) for row in extra_labels]

    ids = [example["sample_id"] for example, _ in pairs]
    duplicates = [sid for sid, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate sample ids after merging: {duplicates[:5]}")
    return sorted(pairs, key=lambda pair: pair[0]["sample_id"])


def dataset_row(example: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    """One output row. Raises ``ValueError`` when the example and its label disagree.

    Without ``example["temporal_context"]`` this is the legacy single-frame observation task: the row is exactly
    the Clef follower's own ``choice`` record (unchanged, ``record``). WITH ``temporal_context`` (v2) it is the
    contextual state task: ``record`` is the contextual ``{state, questions}`` built by
    ``backend.app.temporal_follow.contextual_clef_record`` (the student input that row trains against), the
    frozen single-frame record is kept as ``observation_record``, and the label MUST declare the matching
    ``temporal_context_sha256`` and ``student_record_sha256`` (the latter order-sensitive, binding the packed
    field order) — so a single-frame label can never be silently exported as a contextual target, an old v1
    context is refused, and a context can never be paired with a record it does not describe.
    """
    sid = example["sample_id"]
    if label["sample_id"] != sid or label["current_step"] != example["current_step"] or label["split"] != example["split"]:
        raise ValueError(f"{sid}: label row does not match the example (id, current step or split)")
    plan = example["plan"]
    observation_record = clef_record(
        user_goal=plan["user_goal"],
        steps=[{"id": step["id"], "say": step["say"], "done_when": step["done_when"]} for step in plan["steps"]],
        goal_when=plan["goal_when"],
        current_step=example["current_step"],
        anchor=None,
        mode=MODE_CHOICE,
    )
    temporal_context = example.get("temporal_context")
    contextual = temporal_context is not None
    declared = (label.get("judgment") == "contextual_state" or "temporal_context_sha256" in label
                or "student_record_sha256" in label)
    if declared and not contextual:
        raise ValueError(f"{sid}: label targets a contextual record but the example has no temporal_context")
    if contextual and not declared:
        raise ValueError(f"{sid}: example carries a temporal_context but the label declares no contextual hashes")
    student_record: dict[str, Any] | None = None
    context_digest = record_digest = None
    if contextual:
        try:
            student_record = contextual_clef_record(
                user_goal=plan["user_goal"], steps=plan["steps"], goal_when=plan["goal_when"],
                current_step=example["current_step"], temporal_context=temporal_context)
        except TemporalContextError as exc:
            raise ValueError(f"{sid}: unusable temporal context: {exc}") from exc
        context_digest = temporal_context_sha256(temporal_context)
        record_digest = student_record_sha256(student_record)
        if label.get("temporal_context_sha256") != context_digest:
            raise ValueError(f"{sid}: label temporal_context_sha256 does not match the example temporal_context")
        if label.get("student_record_sha256") != record_digest:
            raise ValueError(f"{sid}: label student_record_sha256 does not match the contextual record")
    record = student_record if student_record is not None else observation_record
    final = label["final"]
    if set(final) != set(label["keys"]):
        raise ValueError(f"{sid}: final labels do not cover the label keys")
    labels = {record_key(key): final[key]["verdict"] for key in label["keys"]}
    basis = {record_key(key): final[key]["basis"] for key in label["keys"]}
    if list(labels) != list(record["questions"]):
        raise ValueError(f"{sid}: label keys {list(labels)} != question keys {list(record['questions'])}")
    bad = {key: value for key, value in labels.items() if value not in VERDICTS}
    if bad:
        raise ValueError(f"{sid}: verdicts outside yes/no/unsure: {bad}")
    current = example["current_step"]
    row = {
        "sample_id": sid,
        "split": example["split"],
        "domain": example["domain"],
        "current_step": current,
        "plan_id": example.get("plan_id"),
        "image": f"images/{sid}.jpg",
        "image_sha256": example["prepared_image_sha256"],
        "record": record,
        "labels": labels,
        "basis": basis,
        "weights": {key: CURRENT_WEIGHT if key == current else OTHER_WEIGHT for key in labels},
        "label_source": label.get("label_source_before"),
        "principles_version": label.get("principles_version"),
    }
    if student_record is not None:
        row.update({
            "judgment": "contextual_state",
            "temporal_context": temporal_context,
            "temporal_context_sha256": context_digest,
            "observation_record": observation_record,
            "student_record": student_record,
            "student_record_sha256": record_digest,
        })
    return row


def split_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        sel = [row for row in rows if row["split"] == split]
        verdicts = Counter(v for row in sel for v in row["labels"].values())
        current = Counter(row["labels"][row["current_step"]] for row in sel)
        summary[split] = {
            "rows": len(sel),
            "keys": sum(len(row["labels"]) for row in sel),
            "verdicts": dict(sorted(verdicts.items())),
            "current_step_verdicts": dict(sorted(current.items())),
            "domains": dict(sorted(Counter(row["domain"] for row in sel).items())),
        }
    return summary


def build(src: Path, out: Path, *, expected: dict[str, int] | None = EXPECTED_SPLITS, extra: tuple[str, ...] = (),
          name: str = DATA_DIR) -> dict[str, Any]:
    pairs = merge_sources(src, extra)
    rows = [dataset_row(example, label) for example, label in pairs]
    summary = split_summary(rows)
    if expected is not None:
        got = {split: summary[split]["rows"] for split in expected}
        if got != expected:
            raise ValueError(f"split rows {got} != expected {expected}")

    data = out / name
    if data.exists():
        shutil.rmtree(data)
    (data / "images").mkdir(parents=True)
    for (example, _), row in zip(pairs, rows):
        source = Path(example["prepared_image_path"])
        if sha256_file(source) != row["image_sha256"]:
            raise ValueError(f"{row['sample_id']}: image sha256 mismatch for {source}")
        shutil.copyfile(source, data / row["image"])
    with (data / "dataset.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"rows": len(rows), "splits": summary, "source": str(src), "extra": list(extra),
                "weights": {"current_step": CURRENT_WEIGHT, "other": OTHER_WEIGHT}}
    (data / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    tar_path = out / f"{name}.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(data, arcname=DATA_DIR)
    manifest["tar"] = str(tar_path)
    manifest["tar_bytes"] = tar_path.stat().st_size
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--extra", action="append", default=[], help="extra domain NAME (NAME_examples.jsonl ...)")
    parser.add_argument("--expected", help="split row guard train=N,val=N,test=N or 'none'; required with --extra")
    parser.add_argument("--name", default=DATA_DIR, help="staging dir and tar file name under --out")
    args = parser.parse_args()
    if args.expected is None:
        if args.extra:
            parser.error("--expected is required with --extra")
        expected: dict[str, int] | None = EXPECTED_SPLITS
    elif args.expected == "none":
        expected = None
    else:
        expected = {k: int(v) for k, v in (item.split("=") for item in args.expected.split(","))}
    manifest = build(args.src, args.out, expected=expected, extra=tuple(args.extra), name=args.name)
    for split, info in manifest["splits"].items():
        print(f"{split:5s} rows {info['rows']:4d} keys {info['keys']:5d} verdicts {info['verdicts']} "
              f"current {info['current_step_verdicts']}")
    print(f"tar {manifest['tar']} ({manifest['tar_bytes'] / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
