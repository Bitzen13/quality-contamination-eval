"""Options only, letter recall, verbatim recall, paraphrasing and the control comparison."""

from __future__ import annotations

import asyncio
import json
import shutil

import pandas as pd
import pytest
from conftest import DATA
from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

import analysis
import compare
import paraphrase
import study
from analysis import summarize_recall
from quality_probe.dataset import (
    LETTERS,
    QUESTION_WITHHELD,
    build_samples,
    load_questions,
    normalize_option,
    paraphrase_path,
    recall_samples,
)
from quality_probe.scorer import parse_option, text_similarity, verbatim_scorer
from test_end_to_end import logs_by_condition, read_eval_log

MODEL = "mockllm/model"
SUBSET = ("--stories", "4", "--per-story", "5")


@pytest.fixture(scope="module")
def data():
    return load_questions(DATA)


def run(log_dir, fake="oracle", *extra):
    return study.main(["--models", MODEL, "--fake", fake, "--log-dir", str(log_dir), *SUBSET, *extra])


def summary_of(log_dir):
    assert analysis.main([str(log_dir)]) == 0
    return json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))[MODEL]


# --- Options only ----------------------------------------------------------------------


def test_options_only_hides_the_question_and_keeps_the_answer(data):
    samples = build_samples(data, story=False, options="options_only")
    for question, sample in zip(data.questions, samples):
        assert sample.input == QUESTION_WITHHELD
        assert question.question not in sample.model_dump_json()
        assert list(sample.choices) == list(question.options)
        assert sample.target == LETTERS[question.gold_index]
        assert "story" not in sample.metadata


def test_options_only_refuses_a_story(data):
    with pytest.raises(ValueError, match="without the story"):
        build_samples(data, story=True, options="options_only")


# --- Verbatim recall samples -----------------------------------------------------------


@pytest.fixture(scope="module")
def recall(data):
    return recall_samples(data)


def test_recall_hides_the_longest_wrong_option_and_never_the_gold(data, recall):
    for question, sample in zip(data.questions, recall):
        gold = question.options[question.gold_index]
        letter = sample.metadata["hidden_letter"]
        assert letter != LETTERS[question.gold_index]
        assert sample.target == question.options[LETTERS.index(letter)]
        wrong = [t for i, t in enumerate(question.options) if i != question.gold_index]
        assert len(sample.target) == max(len(t) for t in wrong)
        assert f"{letter}) [removed]" in sample.input
        assert f") {gold}" in sample.input


def test_recall_prompt_never_contains_the_hidden_option(data, recall):
    for question, sample in zip(data.questions, recall):
        others = [t for t in question.options if t != sample.target]
        if not any(sample.target in other for other in others) and sample.target not in question.question:
            assert sample.target not in sample.input
        assert question.story not in sample.input
        assert "story" not in sample.metadata


def test_recall_decoy_comes_from_the_same_story_and_a_similar_length(data, recall):
    by_id = {q.id: q for q in data.questions}
    for question, sample in zip(data.questions, recall):
        source = by_id[sample.metadata["decoy_source_question_id"]]
        assert source.id != question.id
        assert source.article_id == question.article_id
        decoy = sample.metadata["decoy"]
        assert decoy in [t for i, t in enumerate(source.options) if i != source.gold_index]
        assert normalize_option(decoy) not in {normalize_option(t) for t in question.options}
    lengths = [abs(s.metadata["hidden_words"] - len(s.metadata["decoy"].split())) for s in recall]
    assert sum(lengths) / len(lengths) < 2


def test_recall_is_seeded(data):
    first = [s.metadata["decoy"] for s in recall_samples(data, seed=0)]
    assert first == [s.metadata["decoy"] for s in recall_samples(data, seed=0)]
    assert first != [s.metadata["decoy"] for s in recall_samples(data, seed=1)]


# --- Verbatim recall scoring -----------------------------------------------------------


@pytest.mark.parametrize(
    "completion,expected",
    [
        ("OPTION: He went home", "He went home"),
        ("option: he went home.", "he went home."),
        ("**OPTION:** He went home", "He went home"),
        ("Thinking.\nOPTION: First\nOPTION: Second", "Second"),
        ("OPTION:", None),
        ("OPTION:\nHe went home", None),
        ("The option: nope", None),
        ("He went home", None),
        ("", None),
    ],
)
def test_parse_option(completion, expected):
    assert parse_option(completion) == expected


def test_similarity_ignores_case_and_punctuation():
    assert text_similarity("He went HOME!", "he went home") == 1.0
    assert text_similarity("completely different words", "zzz") < 0.3


def score_recall(completion, target="He went home to see his mother", stop_reason="stop"):
    output = ModelOutput.from_content(MODEL, completion, stop_reason=stop_reason)
    state = TaskState(
        model=ModelName(MODEL),
        sample_id="q",
        epoch=1,
        input="q",
        messages=[],
        target=Target(target),
        output=output,
        metadata={"decoy": "The robots took over the whole city"},
    )
    return asyncio.run(verbatim_scorer()(state, Target(target)))


def test_exact_match_scores_correct():
    score = score_recall("OPTION: he went home to see his mother.")
    assert score.value == "C"
    assert score.metadata["exact"] is True
    assert score.metadata["similarity"] == 1.0
    assert score.metadata["decoy_exact"] is False


def test_near_match_is_not_exact_but_has_high_similarity():
    score = score_recall("OPTION: He went home to visit his mother")
    assert score.value == "I"
    assert score.metadata["similarity"] > 0.8


def test_matching_the_decoy_is_recorded_as_the_decoy():
    score = score_recall("OPTION: The robots took over the whole city")
    assert score.value == "I"
    assert score.metadata["decoy_exact"] is True
    assert score.metadata["decoy_similarity"] == 1.0


@pytest.mark.parametrize(
    "completion,stop_reason,status",
    [
        ("I don't remember this.", "stop", "parse_fail"),
        ("", "max_tokens", "truncated_output"),
        ("OPTION: He went home to see his mother", "content_filter", "refused"),
    ],
)
def test_recall_non_answers(completion, stop_reason, status):
    score = score_recall(completion, stop_reason=stop_reason)
    assert score.value == "I"
    assert score.metadata["status"] == status
    assert score.metadata["similarity"] == 0.0


def _recall_frame(similarity, decoy, words=None, exact=None) -> pd.DataFrame:
    n = len(similarity)
    return pd.DataFrame(
        {
            "model": "m",
            "question_id": [f"q{i}" for i in range(n)],
            "article_id": [f"a{i % 4}" for i in range(n)],
            "hidden_words": words or [8] * n,
            "target": "t",
            "status": "ok",
            "recalled": "r",
            "exact": exact or [s == 1.0 for s in similarity],
            "decoy_exact": [d == 1.0 for d in decoy],
            "similarity": similarity,
            "decoy_similarity": decoy,
        }
    )


def test_recall_summary_is_exact():
    frame = _recall_frame([1.0, 1.0, 0.85, 0.1] * 2, [0.1, 0.1, 0.1, 0.9] * 2)
    summary = summarize_recall(frame, samples=200)
    assert summary["exact"]["value"] == 0.5
    assert summary["decoy_exact"]["value"] == 0
    assert summary["near"]["value"] == 0.75
    assert summary["decoy_near"]["value"] == 0.25
    assert summary["near_above_decoy"]["value"] == 0.5


def test_recall_summary_ignores_short_options():
    frame = _recall_frame([1.0, 0.1, 0.1, 0.1], [0.1] * 4, words=[1, 8, 8, 8])
    summary = summarize_recall(frame, samples=200)
    assert summary["long_options"] == 3
    assert summary["exact"]["value"] == 0


# --- Probes end to end -----------------------------------------------------------------


def test_probes_end_to_end_with_the_oracle(tmp_path):
    assert run(tmp_path, "oracle", "--probes") == 0
    logs = logs_by_condition(tmp_path)
    assert set(logs) == {name for name, _, _ in study.PROBE_CONDITIONS} | {
        "verbatim_recall",
        "context_check",
    }
    summary = summary_of(tmp_path)
    assert summary["conditions"]["no_story_options_only"]["accuracy"] == 1.0
    assert summary["conditions"]["no_story_letters_recall"]["accuracy"] == 1.0
    assert summary["conditions"]["no_story_letters_recall"]["letter_habit"]["excess"] > 0.5
    assert summary["verbatim_recall"]["exact"]["value"] == 1.0
    assert summary["verbatim_recall"]["questions"] == 20
    assert summary["verbatim_recall"]["near_above_decoy"]["value"] > 0.9

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "### Verbatim recall" in report
    assert "**Verbatim recall.**" in report
    assert "**Options only.**" in report
    assert (tmp_path / "recall_results.csv").exists()

    prompts = read_eval_log(str(logs["no_story_letters_recall"])).samples[0].messages[1].text
    assert "From memory" in prompts
    verbatim = read_eval_log(str(logs["verbatim_recall"])).samples[0]
    assert "[removed]" in verbatim.messages[0].text
    assert verbatim.target not in verbatim.messages[0].text


def test_answering_with_the_decoy_counts_as_the_decoy(tmp_path):
    # The always_a fake answers with the decoy, so it never reproduces the real option.
    assert run(tmp_path, "always_a", "--probes") == 0
    recall = summary_of(tmp_path)["verbatim_recall"]
    assert recall["exact"]["value"] == 0
    assert recall["decoy_exact"]["value"] == 1.0
    assert recall["near_above_decoy"]["value"] < 0


def test_probes_add_to_a_main_run_and_compare_with_it(tmp_path):
    assert run(tmp_path) == 0
    assert run(tmp_path, "oracle", "--probes") == 0
    summary = summary_of(tmp_path)
    assert summary["comparisons"]["question_clues"]["value"] == 0
    assert "verbatim_recall" in summary


# --- Paraphrasing ----------------------------------------------------------------------

ORIGINALS = ["He went home quickly", "She stayed at work", "Flight", "They all left town"]
GOOD = ["He returned home fast", "She remained at her job", "Flight", "Everyone departed town"]


@pytest.mark.parametrize(
    "completion,accepted",
    [
        (json.dumps(GOOD), True),
        (f"```json\n{json.dumps(GOOD)}\n```", True),
        (json.dumps(GOOD[:3]), False),
        ("no json here", False),
        ('["unterminated', False),
        (json.dumps([*GOOD[:3], ""]), False),
        (json.dumps([*GOOD[:3], 7]), False),
        (json.dumps([ORIGINALS[0], *GOOD[1:]]), False),  # long option left unchanged
        (json.dumps([GOOD[0], GOOD[0], GOOD[2], GOOD[3]]), False),  # two options merged
    ],
)
def test_parse_paraphrases(completion, accepted):
    assert (paraphrase.parse_paraphrases(completion, ORIGINALS) is not None) == accepted


@pytest.fixture
def data_copy(tmp_path):
    """Paraphrases are written beside the dataset, so use a copy, not the real folder."""
    path = tmp_path / "quality.dev"
    shutil.copy(DATA, path)
    return path


def make_paraphrases(data_copy, tmp_path):
    return paraphrase.main(
        ["--model", MODEL, "--fake", "--data", str(data_copy), *SUBSET, "--log-dir", str(tmp_path / "gen")]
    )


def test_paraphrase_generation_and_conditions_end_to_end(data_copy, tmp_path, monkeypatch):
    assert make_paraphrases(data_copy, tmp_path) == 0
    rows = paraphrase_path(data_copy).read_text(encoding="utf-8").strip().split("\n")
    assert len(rows) == 20

    log_dir = tmp_path / "logs"
    args = ("--data", str(data_copy))
    assert run(log_dir, "oracle", *args) == 0
    assert run(log_dir, "oracle", *args, "--paraphrased") == 0
    summary = summary_of(log_dir)
    assert summary["conditions"]["no_story_paraphrased"]["accuracy"] == 1.0
    assert summary["comparisons"]["wording_probe"]["value"] == 0
    assert summary["comparisons"]["paraphrase_control"]["value"] == 0

    logs = logs_by_condition(log_dir)
    sample = read_eval_log(str(logs["no_story_paraphrased"])).samples[0]
    assert "Reworded: " in sample.messages[1].text
    assert "paraphrases_sha256" in read_eval_log(
        str(logs["no_story_paraphrased"]), header_only=True
    ).eval.metadata
    assert "**Paraphrased options.**" in (log_dir / "report.md").read_text(encoding="utf-8")


def test_paraphrase_generation_only_fills_gaps(data_copy, tmp_path, monkeypatch):
    from quality_probe import fakes

    assert make_paraphrases(data_copy, tmp_path) == 0
    calls: list[str] = []
    monkeypatch.setattr(fakes, "on_call", calls.append)
    assert make_paraphrases(data_copy, tmp_path) == 0
    assert calls == []


def test_paraphrased_needs_the_generated_file(data_copy, tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, "oracle", "--data", str(data_copy), "--paraphrased")


def test_paraphrased_refuses_questions_with_no_paraphrase(data_copy, tmp_path):
    assert make_paraphrases(data_copy, tmp_path) == 0
    with pytest.raises(SystemExit):
        study.main(
            ["--models", MODEL, "--fake", "oracle", "--stories", "5", "--per-story", "6",
             "--data", str(data_copy), "--paraphrased", "--log-dir", str(tmp_path / "x")]
        )  # fmt: skip


def test_extra_condition_flags_are_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, "oracle", "--probes", "--letters-only")


# --- Control comparison ----------------------------------------------------------------


def _results(accuracy: dict[str, float], stories=("a1", "a2", "a3", "a4")) -> pd.DataFrame:
    rows = []
    for condition, value in accuracy.items():
        for story in stories:
            for i in range(10):
                rows.append(
                    {
                        "model": "m",
                        "condition": condition,
                        "article_id": story,
                        "correct": i < round(value * 10),
                    }
                )
    return pd.DataFrame(rows)


def test_comparison_gives_the_exact_difference_of_differences():
    benchmark = _results({"full": 0.9, "no_story": 0.5})
    control = _results({"full": 0.8, "no_story": 0.3})
    entry = compare.compare_runs(benchmark, control)["m"]
    assert entry["full_difference"]["value"] == pytest.approx(0.1)
    assert entry["no_story_difference"]["value"] == pytest.approx(0.2)
    assert entry["difference_of_differences"]["value"] == pytest.approx(0.1)
    assert entry["shared_stories"] == 4


def test_comparison_uses_only_shared_stories():
    benchmark = _results({"full": 0.9, "no_story": 0.5}, stories=("a1", "a2", "a3"))
    control = _results({"full": 0.8, "no_story": 0.3}, stories=("a2", "a3", "a4"))
    assert compare.compare_runs(benchmark, control)["m"]["shared_stories"] == 2


@pytest.mark.parametrize(
    "benchmark,control,message",
    [
        (_results({"full": 0.9}), _results({"full": 0.8, "no_story": 0.3}), "missing"),
        (
            _results({"full": 0.9, "no_story": 0.5}, stories=("a1",)),
            _results({"full": 0.8, "no_story": 0.3}, stories=("a2",)),
            "share no stories",
        ),
    ],
)
def test_comparison_refuses_runs_it_cannot_compare(benchmark, control, message):
    with pytest.raises(analysis.AnalysisError, match=message):
        compare.compare_runs(benchmark, control)


def test_compare_cli_end_to_end(tmp_path):
    oracle, always_a = tmp_path / "oracle", tmp_path / "always_a"
    assert run(oracle, "oracle") == 0
    assert run(always_a, "always_a") == 0
    assert analysis.main([str(oracle)]) == 0
    assert compare.main([str(oracle), str(always_a), "--out", str(tmp_path / "c.md")]) == 1
    assert analysis.main([str(always_a)]) == 0
    assert compare.main([str(oracle), str(always_a), "--out", str(tmp_path / "c.md")]) == 0
    text = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert "Difference of differences" in text
    assert "shared stories" in text
