"""Shared fixtures: a small synthetic multidomain source (same file shapes as the real one)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parents[1]
for path in (str(REPO), str(PKG), str(PKG / "vendor")):
    if path not in sys.path:
        sys.path.insert(0, path)

REAL_SRC = Path.home() / ".local/share/synoptics-astra-distill/multidomain"

PLAN = {
    "user_goal": "컵에 물을 따르고 뚜껑을 닫는다.",
    "goal_when": "뚜껑이 닫힌 컵이 보인다.",
    "steps": [
        {"id": "s1", "say": "컵에 물을 따르세요.", "done_when": "컵에 물이 차 있다."},
        {"id": "s2", "say": "뚜껑을 닫으세요.", "done_when": "컵 위에 뚜껑이 닫혀 있다."},
        {"id": "s3", "say": "컵을 내려놓으세요.", "done_when": "뚜껑 닫힌 컵이 탁자 위에 있다."},
    ],
}


def _image(path: Path, seed: int) -> str:
    from PIL import Image

    image = Image.new("RGB", (128, 96), ((seed * 40) % 256, (seed * 90) % 256, (seed * 17) % 256))
    image.save(path, "JPEG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verdict(v: str) -> dict[str, str]:
    basis = {"yes": "visible_met", "no": "visible_counter_evidence", "unsure": "nothing_decides"}[v]
    return {"verdict": v, "basis": basis, "evidence": "..."}


def make_source(root: Path) -> dict[str, Any]:
    """Seven final-label rows (one excluded) + two plan-fix rows; splits after merging: train 4, val 2, test 2.

    Every split has at least two rows and every verdict appears on a current step key.
    """
    (root / "img").mkdir(parents=True)
    spec = [  # sample_id, split, current_step, verdicts by label key
        ("a_train_1", "train", "s1", {"s1": "yes", "s2": "no", "s3": "no", "g": "no"}),
        ("a_train_2", "train", "s2", {"s2": "unsure", "s3": "no", "g": "unsure"}),
        ("a_train_3", "train", "s3", {"s3": "no", "g": "no"}),
        ("a_excluded", "train", "s1", {"s1": "no", "s2": "no", "s3": "no", "g": "no"}),
        ("b_val_1", "val", "s1", {"s1": "no", "s2": "no", "s3": "no", "g": "no"}),
        ("b_val_2", "val", "s2", {"s2": "yes", "s3": "unsure", "g": "unsure"}),
        ("c_test_1", "test", "s2", {"s2": "unsure", "s3": "no", "g": "no"}),
    ]
    fix_spec = [
        ("a_fix_1_r2", "train", "s1", {"s1": "yes", "s2": "yes", "s3": "no", "g": "no"}),
        ("c_fix_2_r2", "test", "s3", {"s3": "yes", "g": "yes"}),
    ]

    def rows(entries, offset: int):
        examples, labels = [], []
        for index, (sid, split, current, verdicts) in enumerate(entries):
            image = root / "img" / f"{sid}_prep.jpg"
            sha = _image(image, index + offset)
            keys = list(verdicts)
            examples.append({"sample_id": sid, "split": split, "domain": "cooking" if index % 2 else "assembly",
                             "current_step": current, "checklist_ids": keys[:-1], "expected_keys": keys,
                             "plan": PLAN, "plan_id": "p1", "prepared_image_path": str(image),
                             "prepared_image_sha256": sha, "label_source": "x"})
            labels.append({"sample_id": sid, "domain": examples[-1]["domain"], "split": split, "current_step": current,
                           "keys": keys, "principles_version": "2026-10-03.v2",
                           "final": {k: _verdict(v) for k, v in verdicts.items()},
                           "label_source_before": "astra_opus_agree"})
        return examples, labels

    examples, labels = rows(spec, 0)
    fix_examples, fix_labels = rows(fix_spec, 50)

    def dump(name: str, data: list[dict[str, Any]]) -> None:
        (root / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")

    dump("examples.jsonl", examples)
    dump("final_labels.jsonl", labels)
    dump("training_exclusions.jsonl", [{"sample_id": "a_excluded", "reason": "plan_major_issue"}])
    dump("plan_fix_rows.jsonl", fix_examples)
    dump("plan_fix_labels.jsonl", fix_labels)
    return {"splits": {"train": 4, "val": 2, "test": 2}}


@pytest.fixture
def source(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    make_source(src)
    return src
