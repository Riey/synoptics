from __future__ import annotations

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import build_data
from backend.app.clef_follow import ClefFollower
from backend.app.temporal_follow import (
    TEMPORAL_VERSION,
    contextual_clef_record,
    student_record_sha256,
    temporal_context_sha256,
)
from conftest import PLAN, REAL_SRC, _image, _verdict


def built(source: Path, tmp_path: Path) -> tuple[dict, list[dict]]:
    manifest = build_data.build(source, tmp_path / "out", expected={"train": 4, "val": 2, "test": 2})
    data = tmp_path / "out" / build_data.DATA_DIR
    rows = [json.loads(line) for line in (data / "dataset.jsonl").read_text().splitlines()]
    return manifest, rows


def test_merge_drops_exclusions_and_adds_plan_fix_rows(source: Path, tmp_path: Path) -> None:
    manifest, rows = built(source, tmp_path)
    ids = [row["sample_id"] for row in rows]
    assert "a_excluded" not in ids
    assert {"a_fix_1_r2", "c_fix_2_r2"} <= set(ids)
    assert ids == sorted(ids) and len(ids) == 8
    assert {split: info["rows"] for split, info in manifest["splits"].items()} == {"train": 4, "val": 2, "test": 2}
    assert manifest["splits"]["train"]["keys"] == 4 + 3 + 2 + 4


def test_rows_map_goal_key_and_weight_the_current_step(source: Path, tmp_path: Path) -> None:
    _, rows = built(source, tmp_path)
    row = next(r for r in rows if r["sample_id"] == "a_train_2")
    assert row["labels"] == {"s2": "unsure", "s3": "no", "goal": "unsure"}
    assert row["weights"] == {"s2": 2.0, "s3": 1.0, "goal": 1.0}
    assert row["basis"]["s2"] == "nothing_decides"
    assert list(row["record"]["questions"]) == ["s2", "s3", "goal"]


def test_tar_holds_dataset_manifest_and_every_image(source: Path, tmp_path: Path) -> None:
    manifest, rows = built(source, tmp_path)
    with tarfile.open(manifest["tar"]) as tar:
        names = set(tar.getnames())
    assert {"clef_ft_data/dataset.jsonl", "clef_ft_data/manifest.json"} <= names
    assert {f"clef_ft_data/{row['image']}" for row in rows} <= names


def test_image_sha_mismatch_fails(source: Path, tmp_path: Path) -> None:
    (source / "img" / "b_val_1_prep.jpg").write_bytes(b"not the image")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        build_data.build(source, tmp_path / "out", expected=None)


def test_split_count_guard(source: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split rows"):
        build_data.build(source, tmp_path / "out")  # the real 646/81/119


def test_unknown_exclusion_fails(source: Path, tmp_path: Path) -> None:
    (source / "training_exclusions.jsonl").write_text('{"sample_id": "nope"}\n')
    with pytest.raises(ValueError, match="unknown sample ids"):
        build_data.build(source, tmp_path / "out", expected=None)


def add_extra(source: Path, name: str = "soldering") -> None:
    """Two rows of an extra domain in its own files (one train, one test)."""
    examples, labels = [], []
    for index, (sid, split, verdicts) in enumerate([("sol_a", "train", {"s1": "yes", "s2": "no", "s3": "no", "g": "no"}),
                                                    ("sol_b", "test", {"s3": "unsure", "g": "no"})]):
        image = source / "img" / f"{sid}_prep.jpg"
        sha = _image(image, 90 + index)
        keys = list(verdicts)
        current = keys[0]
        examples.append({"sample_id": sid, "split": split, "domain": name, "current_step": current,
                         "checklist_ids": keys[:-1], "expected_keys": keys, "plan": PLAN, "plan_id": "ps",
                         "prepared_image_path": str(image), "prepared_image_sha256": sha})
        labels.append({"sample_id": sid, "domain": name, "split": split, "current_step": current, "keys": keys,
                       "principles_version": "2026-10-03.v2", "final": {k: _verdict(v) for k, v in verdicts.items()},
                       "label_source_before": "astra_final_soldering"})
    for suffix, data in (("examples", examples), ("final_labels", labels)):
        (source / f"{name}_{suffix}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data), encoding="utf-8")


def test_extra_domain_rows_join_under_a_new_tar_name(source: Path, tmp_path: Path) -> None:
    add_extra(source)
    out = tmp_path / "out"
    (out / build_data.DATA_DIR).mkdir(parents=True)
    (out / build_data.DATA_DIR / "keep.txt").write_text("v1 staging must survive")
    manifest = build_data.build(source, out, expected={"train": 5, "val": 2, "test": 3}, extra=("soldering",),
                                name="clef_ft_data_v2")
    assert manifest["tar"] == str(out / "clef_ft_data_v2.tar") and manifest["extra"] == ["soldering"]
    assert manifest["splits"]["train"]["domains"]["soldering"] == 1
    assert (out / build_data.DATA_DIR / "keep.txt").exists()
    with tarfile.open(manifest["tar"]) as tar:
        names = set(tar.getnames())
    assert {"clef_ft_data/dataset.jsonl", "clef_ft_data/images/sol_b.jpg"} <= names
    rows = [json.loads(line) for line in (out / "clef_ft_data_v2" / "dataset.jsonl").read_text().splitlines()]
    sol = next(r for r in rows if r["sample_id"] == "sol_b")
    assert sol["labels"] == {"s3": "unsure", "goal": "no"} and sol["label_source"] == "astra_final_soldering"


def test_extra_domain_id_mismatch_fails(source: Path, tmp_path: Path) -> None:
    add_extra(source)
    labels = (source / "soldering_final_labels.jsonl").read_text().splitlines()
    (source / "soldering_final_labels.jsonl").write_text(labels[0] + "\n")
    with pytest.raises(ValueError, match="different sample ids"):
        build_data.build(source, tmp_path / "out", expected=None, extra=("soldering",))


@pytest.mark.parametrize("current_step", ["s1", "s2", "s3"])
def test_training_record_is_the_served_choice_record(current_step: str) -> None:
    """The record a training row carries equals what the app's Clef follower sends in choice mode (no anchor)."""
    example = {"sample_id": "x", "split": "train", "domain": "cooking", "current_step": current_step, "plan": PLAN,
               "plan_id": "p", "prepared_image_sha256": "0" * 64}
    keys = [s["id"] for s in PLAN["steps"]][[s["id"] for s in PLAN["steps"]].index(current_step):] + ["g"]
    label = {"sample_id": "x", "split": "train", "current_step": current_step, "keys": keys,
             "final": {k: {"verdict": "no", "basis": "visible_counter_evidence"} for k in keys}}
    row = build_data.dataset_row(example, label)

    follower = ClefFollower(None, url="http://127.0.0.1:8085/v1/systemone", mode="choice")  # type: ignore[arg-type]
    plan = SimpleNamespace(user_goal=PLAN["user_goal"], goal_when=PLAN["goal_when"], core_mode="classic",
                           steps=[SimpleNamespace(**step) for step in PLAN["steps"]])
    request = SimpleNamespace(current_step=current_step, anchors=[])
    sent = follower.payload(request, plan, "AAAA")  # type: ignore[arg-type]
    assert row["record"] == {"state": sent["state"], "questions": sent["questions"]}
    # Same JSON text, so the same tokens: key order matters to encode_record.
    assert json.dumps(row["record"], ensure_ascii=False) == json.dumps(
        {"state": sent["state"], "questions": sent["questions"]}, ensure_ascii=False)


@pytest.mark.skipif(not (REAL_SRC / "final_labels.jsonl").exists(), reason="real multidomain source not here")
def test_real_source_split_counts(tmp_path: Path) -> None:
    manifest = build_data.build(REAL_SRC, tmp_path)
    splits = manifest["splits"]
    assert {s: splits[s]["rows"] for s in splits} == {"train": 646, "val": 81, "test": 119}
    assert {s: splits[s]["keys"] for s in splits} == {"train": 1841, "val": 230, "test": 358}


def _row_keys(current_step: str) -> list[str]:
    ids = [step["id"] for step in PLAN["steps"]]
    return ids[ids.index(current_step):] + ["g"]


def _observation_example(current_step: str = "s2") -> dict:
    return {"sample_id": "x", "split": "train", "domain": "cooking", "current_step": current_step, "plan": PLAN,
            "plan_id": "p", "prepared_image_sha256": "0" * 64}


def _label(current_step: str = "s2", **over) -> dict:
    keys = _row_keys(current_step)
    label = {"sample_id": "x", "split": "train", "current_step": current_step, "keys": keys,
             "final": {key: {"verdict": "no", "basis": "visible_counter_evidence"} for key in keys}}
    label.update(over)
    return label


def _context(**over) -> dict:
    context = {"version": TEMPORAL_VERSION, "recording_id": "elk6q6as", "plan_revision": "led-plan-v3-atomic-1",
               "current_timestamp_s": 10.0, "observed_facts": [], "continuity": []}
    context.update(over)
    return context


def _observed_fact(**over) -> dict:
    fact = {"id": "f1", "key": "s2", "predicate": "컵 위에 뚜껑이 닫혀 있다.", "verdict": "yes",
            "observed_at_s": 4.0, "source": "astra_generated", "source_ref": "video:00:04#frame",
            "evidence": "4초 프레임에서 뚜껑이 닫혀 보인다."}
    fact.update(over)
    return fact


def _contextual_pair(current_step: str = "s2", **context_over) -> tuple[dict, dict]:
    """An example and its label: the label declares the hashes of exactly this context and record."""
    context = _context(**context_over)
    example = {**_observation_example(current_step), "judgment": "contextual_state", "temporal_context": context,
               "temporal_context_sha256": temporal_context_sha256(context)}
    record = build_data.contextual_clef_record(
        user_goal=PLAN["user_goal"], steps=PLAN["steps"], goal_when=PLAN["goal_when"], current_step=current_step,
        temporal_context=context)
    label = _label(current_step, judgment="contextual_state",
                   temporal_context_sha256=temporal_context_sha256(context),
                   student_record_sha256=student_record_sha256(record))
    return example, label




def test_contextual_label_hashes_must_match() -> None:
    example, label = _contextual_pair()
    label["student_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="student_record_sha256"):
        build_data.dataset_row(example, label)
    _, other = _contextual_pair()
    other["temporal_context_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="temporal_context_sha256"):
        build_data.dataset_row(example, other)


def test_label_bound_to_a_different_context_is_refused() -> None:
    example, _ = _contextual_pair(observed_facts=[_observed_fact()])
    _, other = _contextual_pair(observed_facts=[_observed_fact(verdict="no")])
    with pytest.raises(ValueError, match="temporal_context_sha256"):
        build_data.dataset_row(example, other)


def test_the_export_hash_binds_the_actual_question_order() -> None:
    example, label = _contextual_pair(observed_facts=[_observed_fact()])
    record = build_data.contextual_clef_record(
        user_goal=PLAN["user_goal"], steps=PLAN["steps"], goal_when=PLAN["goal_when"], current_step="s2",
        temporal_context=example["temporal_context"])
    reordered = {**record, "questions": {key: record["questions"][key]
                                         for key in reversed(list(record["questions"]))}}
    assert list(reordered["questions"]) != list(record["questions"])
    label["student_record_sha256"] = student_record_sha256(reordered)
    with pytest.raises(ValueError, match="student_record_sha256"):
        build_data.dataset_row(example, label)


def test_a_v1_context_cannot_be_exported_as_v2() -> None:
    context = _context(version="temporal-state-v1")
    example = {**_observation_example(), "temporal_context": context,
               "temporal_context_sha256": temporal_context_sha256(context)}
    label = _label(judgment="contextual_state", temporal_context_sha256=temporal_context_sha256(context),
                   student_record_sha256="0" * 64)
    with pytest.raises(ValueError, match="unusable temporal context"):
        build_data.dataset_row(example, label)


def test_single_frame_label_cannot_become_a_contextual_target() -> None:
    with pytest.raises(ValueError, match="no temporal_context"):
        build_data.dataset_row(_observation_example(), _label(judgment="contextual_state"))
    with pytest.raises(ValueError, match="no temporal_context"):
        build_data.dataset_row(_observation_example(), _label(temporal_context_sha256="0" * 64))


def test_contextual_example_requires_the_label_hashes() -> None:
    example, _ = _contextual_pair()
    with pytest.raises(ValueError, match="no contextual hashes"):
        build_data.dataset_row(example, _label())


def test_unusable_temporal_context_fails_the_row() -> None:
    # observed_at_s is not strictly before current_timestamp_s: a future fact must fail closed.
    context = _context(observed_facts=[_observed_fact(observed_at_s=10.0)])
    example = {**_observation_example(), "temporal_context": context,
               "temporal_context_sha256": temporal_context_sha256(context)}
    label = _label(judgment="contextual_state", temporal_context_sha256=temporal_context_sha256(context),
                   student_record_sha256="0" * 64)
    with pytest.raises(ValueError, match="unusable temporal context"):
        build_data.dataset_row(example, label)
