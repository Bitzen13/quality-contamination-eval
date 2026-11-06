"""Loader guardrails, shuffle and swap."""

from __future__ import annotations

import hashlib
import json
import logging
import random

import pytest
from conftest import DATA, make_entry, make_question, write_jsonl

from quality_probe.dataset import (
    LETTERS,
    build_samples,
    load_questions,
    normalize_option,
    select_questions,
    shuffle_permutation,
    swap_options,
)
from quality_probe.task import quality


@pytest.fixture(scope="module")
def data():
    return load_questions(DATA)


# --- Loader ---------------------------------------------------------------------------


def test_counts_match_the_published_split(data):
    assert data.entries == 230
    assert len({q.article_id for q in data.questions}) == 115
    assert len(data.questions) == 2086
    assert sum(q.difficult for q in data.questions) == 1065
    assert data.skipped == []


def test_sha256_is_recorded_in_task_metadata(data):
    expected = hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert data.sha256 == expected
    assert quality(data=str(DATA)).metadata["data_sha256"] == expected


def test_samples_are_ordered_by_article(data):
    article_ids = [q.article_id for q in data.questions]
    assert article_ids == sorted(article_ids)


def test_duplicate_ids_abort(tmp_path):
    path = write_jsonl(tmp_path / "d", [make_entry("1", [make_question("q"), make_question("q")])])
    with pytest.raises(ValueError, match="Duplicate"):
        load_questions(path)


def test_test_split_aborts_before_any_model_call(tmp_path):
    path = write_jsonl(tmp_path / "t", [make_entry("1", [make_question("q", gold=None)])])
    with pytest.raises(ValueError, match="no gold_label"):
        load_questions(path)
    with pytest.raises(ValueError, match="no gold_label"):
        quality(data=str(path))


@pytest.mark.parametrize("bad", [0, 5, -1, "2", 2.0, True, None])
def test_invalid_gold_is_skipped_and_logged(tmp_path, caplog, bad):
    entry = make_entry("1", [make_question("good", gold=2), make_question("bad", gold=bad)])
    entry["questions"][1]["gold_label"] = bad
    with caplog.at_level(logging.WARNING):
        loaded = load_questions(write_jsonl(tmp_path / "d", [entry]))
    assert [q.id for q in loaded.questions] == ["good"]
    assert loaded.skipped == ["bad"]
    assert "bad" in caplog.text


@pytest.mark.parametrize("value,expected", [(1, 1), (0, 0), (2, 0), (None, 0), ("1", 0)])
def test_only_difficult_equal_one_is_hard(tmp_path, value, expected):
    question = make_question("q", difficult=value)
    loaded = load_questions(write_jsonl(tmp_path / "d", [make_entry("1", [question])]))
    assert loaded.questions[0].difficult == expected


def test_no_gold_writer_or_validation_field_reaches_a_sample(tmp_path):
    marker = "LEAKMARKER"
    questions = [
        make_question(
            f"q{i}",
            gold=(i % 4) + 1,
            writer_label=marker,
            validation=marker,
            speed_validation=marker,
            untimed_annotator=marker,
        )
        for i in range(8)
    ]
    path = write_jsonl(tmp_path / "d", [make_entry("1", questions)])
    for mode in ("original", "shuffled", "swapped"):
        for sample in build_samples(load_questions(path), options=mode):
            dumped = sample.model_dump_json()
            assert marker not in dumped
            assert "gold_label" not in dumped
            assert "writer" not in dumped
            assert "validation" not in dumped
            assert set(sample.metadata) <= {"article_id", "difficult", "story", "perm", "borrowed"}


def test_story_is_absent_from_no_story_samples(data):
    assert all("story" not in s.metadata for s in build_samples(data, story=False))
    assert all("story" in s.metadata for s in build_samples(data, story=True)[:10])


def test_abstain_appends_a_final_option(data):
    for mode in ("original", "shuffled", "swapped"):
        for sample in build_samples(data, options=mode, abstain=True)[:50]:
            assert len(sample.choices) == 5
            assert sample.choices[-1] == "I don't know"
            assert sample.target in LETTERS


# --- Shuffle --------------------------------------------------------------------------


def test_permutation_is_independent_of_order_and_limit(data):
    full = {s.id: s for s in build_samples(data, options="shuffled", seed=3)}

    reversed_data = type(data)(
        data.path, data.sha256, data.entries, list(reversed(data.questions)), []
    )
    backwards = {s.id: s for s in build_samples(reversed_data, options="shuffled", seed=3)}
    assert all(backwards[k].metadata["perm"] == full[k].metadata["perm"] for k in full)

    limited = quality(options="shuffled", seed=3, data=str(DATA)).dataset[:20]
    for sample in limited:
        assert sample.metadata["perm"] == full[sample.id].metadata["perm"]


def test_permutation_is_seeded_and_not_constant(data):
    assert shuffle_permutation("x", 0) == shuffle_permutation("x", 0)
    assert any(shuffle_permutation(q.id, 0) != shuffle_permutation(q.id, 1) for q in data.questions)
    assert len({tuple(shuffle_permutation(q.id, 0)) for q in data.questions}) == 24


def test_target_follows_the_gold_and_perm_is_stored(data):
    shuffled = build_samples(data, options="shuffled")
    for question, sample in zip(data.questions, shuffled):
        perm = sample.metadata["perm"]
        assert sorted(perm) == [0, 1, 2, 3]
        assert list(sample.choices) == [question.options[i] for i in perm]
        gold_text = question.options[question.gold_index]
        assert sample.choices[LETTERS.index(sample.target)] == gold_text


# --- Swap -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def swapped(data):
    by_id = {q.id: q for q in data.questions}
    return by_id, build_samples(data, options="swapped")


def test_gold_is_kept_once_at_its_letter(swapped):
    by_id, samples = swapped
    for sample in samples:
        question = by_id[sample.id]
        gold = question.options[question.gold_index]
        assert sample.target == LETTERS[question.gold_index]
        assert sample.choices[question.gold_index] == gold
        others = [c for i, c in enumerate(sample.choices) if i != question.gold_index]
        assert normalize_option(gold) not in {normalize_option(c) for c in others}


def test_borrowed_options_come_from_same_article_and_another_question(swapped):
    by_id, samples = swapped
    for sample in samples:
        for item in sample.metadata["borrowed"]:
            source = by_id[item["source_question_id"]]
            assert source.id != sample.id
            assert source.article_id == sample.metadata["article_id"]
            wrong = [t for i, t in enumerate(source.options) if i != source.gold_index]
            assert item["text"] in wrong


def test_no_normalized_duplicates(swapped):
    _, samples = swapped
    for sample in samples:
        keys = [normalize_option(c) for c in sample.choices]
        assert len(set(keys)) == len(keys)


def test_every_borrow_is_recorded(swapped):
    by_id, samples = swapped
    for sample in samples:
        question = by_id[sample.id]
        recorded = sample.metadata["borrowed"]
        assert len(recorded) == 3
        for item in recorded:
            assert sample.choices[LETTERS.index(item["letter"])] == item["text"]
        assert {i["letter"] for i in recorded} == set(LETTERS) - {LETTERS[question.gold_index]}


def test_swap_skips_normalized_duplicates_of_the_gold():
    from quality_probe.dataset import Question

    gold = Question("a", "1", "s", "q?", ("The Gold!", "x", "y", "z"), 0, 0)
    other = Question("b", "1", "s", "q?", ("right", "the  gold", "THE-GOLD", "w1"), 0, 0)
    third = Question("c", "1", "s", "q?", ("right", "w2", "w3", "w4"), 0, 0)
    options, borrowed = swap_options(gold, [gold, other, third], seed=0)
    assert options[0] == "The Gold!"
    assert {normalize_option(o) for o in options[1:]}.isdisjoint({"the gold"})
    assert len(borrowed) == 3


def test_swap_never_borrows_a_gold_answer():
    from quality_probe.dataset import Question

    me = Question("a", "1", "s", "q?", ("g", "x", "y", "z"), 0, 0)
    other = Question("b", "1", "s", "q?", ("other gold", "w1", "w2", "w3"), 0, 0)
    for seed in range(50):
        _, borrowed = swap_options(me, [me, other], seed)
        assert "other gold" not in {item["text"] for item in borrowed}


def test_swap_aborts_when_an_article_has_too_few_options():
    from quality_probe.dataset import Question

    me = Question("a", "1", "s", "q?", ("g", "x", "y", "z"), 0, 0)
    with pytest.raises(ValueError, match="not enough"):
        swap_options(me, [me], seed=0)


def test_swap_is_independent_of_pool_order(data):
    question = data.questions[0]
    pool = [q for q in data.questions if q.article_id == question.article_id]
    expected = swap_options(question, pool, 0)
    random.Random(1).shuffle(pool)
    assert swap_options(question, pool, 0) == expected


def test_samples_are_json_serializable(data):
    for mode in ("shuffled", "swapped"):
        json.dumps(build_samples(data, story=False, options=mode)[0].metadata)


# --- Story subset ---------------------------------------------------------------------


def test_selection_takes_the_requested_shape_and_repeats(data):
    chosen = select_questions(data, stories=30, per_story=5, seed=0)
    by_id = {q.id: q for q in data.questions}
    per_story: dict[str, int] = {}
    for qid in chosen:
        per_story[by_id[qid].article_id] = per_story.get(by_id[qid].article_id, 0) + 1
    assert len(chosen) == len(set(chosen)) == 150
    assert len(per_story) == 30
    assert set(per_story.values()) == {5}
    assert chosen == select_questions(data, stories=30, per_story=5, seed=0)
    assert chosen != select_questions(data, stories=30, per_story=5, seed=1)


def test_selection_is_not_just_the_first_stories(data):
    chosen = set(select_questions(data, stories=30, per_story=5, seed=0))
    first_stories = sorted({q.article_id for q in data.questions})[:30]
    stories = {q.article_id for q in data.questions if q.id in chosen}
    assert stories != set(first_stories)


def test_selection_rejects_impossible_requests(data):
    with pytest.raises(ValueError, match="stories"):
        select_questions(data, stories=116, per_story=5)
    with pytest.raises(ValueError, match="only"):
        select_questions(data, stories=1, per_story=500)


# --- Hidden options -------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["letters_only", "letters_reordered"])
def test_hidden_modes_contain_no_option_text_and_keep_the_real_target(data, mode):
    for question, sample in zip(data.questions, build_samples(data, story=False, options=mode)):
        assert sample.target == LETTERS[question.gold_index]
        assert not sample.choices
        dumped = sample.model_dump_json()
        for option in question.options:
            if len(option) > 25:
                assert json.dumps(option)[1:-1] not in dumped


def test_reordered_letters_are_seeded_per_question_and_vary(data):
    samples = build_samples(data, options="letters_reordered", seed=2)
    assert all(sorted(s.metadata["letter_order"]) == list(LETTERS) for s in samples)
    assert len({tuple(s.metadata["letter_order"]) for s in samples}) == 24
    again = build_samples(data, options="letters_reordered", seed=2)
    assert [s.metadata["letter_order"] for s in samples] == [
        s.metadata["letter_order"] for s in again
    ]


def test_letters_only_has_no_letter_order(data):
    assert all(
        "letter_order" not in s.metadata for s in build_samples(data, options="letters_only")
    )


def test_hidden_modes_reject_abstain(data):
    with pytest.raises(ValueError, match="abstain"):
        build_samples(data, options="letters_only", abstain=True)
