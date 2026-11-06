"""Metrics, the bootstrap and every check, on synthetic results and small fake logs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from inspect_ai.log import read_eval_log, write_eval_log

import analysis
import study
from analysis import AnalysisError, analyze, bootstrap_ci, check_pairing, summarize_model

SMALL_B = 500


def synthetic(condition: str, correct: list[bool], status=None, difficult=None) -> pd.DataFrame:
    n = len(correct)
    return pd.DataFrame(
        {
            "model": "m",
            "condition": condition,
            "question_id": [f"q{i}" for i in range(n)],
            "article_id": [f"a{i % 4}" for i in range(n)],
            "difficult": difficult if difficult is not None else [i % 2 for i in range(n)],
            "correct": correct,
            "status": status or ["ok"] * n,
            "prompt_tokens": [10] * n,
            "chosen_letter": ["A"] * n,
            "landed_on_gold_original_letter": [None] * n,
        }
    )


def test_known_accuracies_give_exact_metrics():
    full = synthetic("full", [True] * 6 + [False] * 2)  # 0.75
    no_story = synthetic("no_story", [True] * 2 + [False] * 6)  # 0.25
    summary = summarize_model(pd.concat([full, no_story]), samples=SMALL_B)

    assert summary["questions"] == 8
    assert summary["stories"] == 4
    assert summary["conditions"]["full"]["accuracy"] == 0.75
    assert summary["conditions"]["no_story"]["accuracy"] == 0.25
    assert summary["comparisons"]["gap"]["value"] == 0.5
    assert summary["conditions"]["full"]["prompt_tokens"] == 80


def test_hard_accuracy_uses_only_hard_questions():
    frame = synthetic("full", [True, False, True, False], difficult=[1, 1, 0, 0])
    assert summarize_model(frame, samples=SMALL_B)["conditions"]["full"]["hard_accuracy"] == 0.5
    frame = synthetic("full", [True, True, False, False], difficult=[1, 1, 0, 0])
    assert summarize_model(frame, samples=SMALL_B)["conditions"]["full"]["hard_accuracy"] == 1.0


def test_status_rates_and_answered_only_accuracy():
    statuses = ["ok", "ok", "ok", "ok", "parse_fail", "refused", "truncated_output", "ok"]
    correct = [True, True, True, False, False, False, False, False]
    entry = summarize_model(synthetic("full", correct, statuses), samples=SMALL_B)["conditions"]["full"]
    assert entry["accuracy"] == 3 / 8
    assert entry["answered_accuracy"] == 3 / 5
    assert entry["non_answer_rate"] == 3 / 8
    assert entry["status_rates"]["parse_fail"] == 1 / 8
    assert entry["status_rates"]["ok"] == 5 / 8
    assert "sat_score" not in entry


def test_abstain_is_not_a_non_answer_and_sat_appears_only_with_abstention():
    statuses = ["ok", "ok", "abstain", "abstain", "ok", "ok", "ok", "parse_fail"]
    correct = [True, True, False, False, True, False, False, False]
    entry = summarize_model(synthetic("full_abstain", correct, statuses), samples=SMALL_B)[
        "conditions"
    ]["full_abstain"]
    assert entry["non_answer_rate"] == 1 / 8
    # 3 correct, 2 abstained, 3 wrong or non-answers
    assert entry["sat_score"] == pytest.approx((3 - 3 / 3) / 8)


def test_letter_share_and_distribution():
    frame = synthetic("no_story_shuffled", [False] * 4)
    frame["landed_on_gold_original_letter"] = [True, True, True, False]
    frame["chosen_letter"] = ["A", "A", "B", "C"]
    entry = summarize_model(frame, samples=SMALL_B)["conditions"]["no_story_shuffled"]
    assert entry["gold_original_letter_share"] == 0.75
    assert entry["letter_distribution"] == {"A": 0.5, "B": 0.25, "C": 0.25}


def test_interval_contains_the_point_and_differences_are_paired():
    rng = np.random.default_rng(0)
    frames = []
    for condition in ("full", "no_story"):
        n = 400
        frame = synthetic(condition, list(rng.random(n) < 0.6))
        frame["article_id"] = [f"a{i % 40}" for i in range(n)]
        frames.append(frame)
    # Identical answers in both conditions: a paired difference must be exactly zero.
    frames[1]["correct"] = frames[0]["correct"].values
    summary = summarize_model(pd.concat(frames), samples=SMALL_B)
    low, high = summary["conditions"]["full"]["accuracy_ci"]
    assert low < summary["conditions"]["full"]["accuracy"] < high
    assert summary["comparisons"]["gap"]["value"] == 0
    assert summary["comparisons"]["gap"]["ci"] == [0.0, 0.0]


def test_bootstrap_is_reproducible():
    num = {"m": np.array([1.0, 2.0, 3.0, 0.0])}
    den = {"m": np.array([4.0, 4.0, 4.0, 4.0])}
    assert bootstrap_ci(num, den, {}, samples=200) == bootstrap_ci(num, den, {}, samples=200)


def test_bootstrap_coverage_is_near_95_percent_under_story_correlation():
    """Stories with correlated questions: the true mean is known, intervals should cover it."""
    rng = np.random.default_rng(7)
    stories, per_story, trials = 115, 18, 400
    alpha, beta = 3.0, 2.0
    truth = alpha / (alpha + beta)
    covered = 0
    for trial in range(trials):
        story_rates = rng.beta(alpha, beta, size=stories)
        correct = rng.binomial(per_story, story_rates).astype(float)
        count = np.full(stories, float(per_story))
        low, high = bootstrap_ci({"m": correct}, {"m": count}, {}, samples=1000, seed=trial)["m"]
        covered += low <= truth <= high
    assert 0.91 <= covered / trials <= 0.98


def test_question_level_resampling_would_understate_that_uncertainty():
    """Why the story bootstrap exists: story intervals are wider on correlated data."""
    rng = np.random.default_rng(1)
    story_rates = rng.beta(3.0, 2.0, size=115)
    correct = rng.binomial(18, story_rates).astype(float)
    count = np.full(115, 18.0)
    low, high = bootstrap_ci({"m": correct}, {"m": count}, {}, samples=2000)["m"]
    p = correct.sum() / count.sum()
    naive_width = 2 * 1.96 * np.sqrt(p * (1 - p) / count.sum())
    assert high - low > 1.5 * naive_width


def _letters_frame(chosen: list[str], target: list[str]) -> pd.DataFrame:
    frame = synthetic("full_letters_only", [c == t for c, t in zip(chosen, target)])
    frame["chosen_letter"] = chosen
    frame["target"] = target
    return frame


def _habit(frame: pd.DataFrame) -> dict:
    return summarize_model(frame, samples=SMALL_B)["conditions"]["full_letters_only"]["letter_habit"]


def test_letter_habit_baseline_is_exact():
    # Model: 3/4 A, 1/4 B. Gold: 1/2 A, 1/2 B. Baseline = .75 * .5 + .25 * .5 = 0.5.
    frame = _letters_frame(list("AAABAAAB"), list("ABABABAB"))
    habit = _habit(frame)
    assert habit["baseline"] == 0.5
    assert habit["excess"] == pytest.approx(frame["correct"].mean() - 0.5)
    assert habit["excess_ci"][0] <= habit["excess_ci"][1]


def test_always_picking_one_letter_earns_exactly_the_baseline():
    habit = _habit(_letters_frame(["A"] * 8, list("AABBCCDD")))
    assert habit["baseline"] == 0.25
    assert habit["excess"] == 0
    assert habit["excess_ci"] == pytest.approx([0.0, 0.0])


def test_non_answers_cannot_earn_baseline_credit():
    frame = _letters_frame(["A", "A", "A", "A"], list("AAAA"))
    frame["chosen_letter"] = ["A", "A", None, None]
    frame["correct"] = [True, True, False, False]
    frame["status"] = ["ok", "ok", "parse_fail", "parse_fail"]
    habit = _habit(frame)
    assert habit["baseline"] == 0.5
    assert habit["excess"] == 0


def test_mismatched_id_sets_stop_the_report():
    full = synthetic("full", [True] * 8)
    no_story = synthetic("no_story", [True] * 7)
    with pytest.raises(AnalysisError, match="different question IDs"):
        check_pairing(pd.concat([full, no_story]))


def test_conditions_not_being_compared_may_differ():
    frame = pd.concat([synthetic("full", [True] * 8), synthetic("no_story_swapped", [True] * 7)])
    check_pairing(frame)


# --- Checks against real logs ---------------------------------------------------------


@pytest.fixture(scope="module")
def fake_logs(tmp_path_factory):
    log_dir = tmp_path_factory.mktemp("logs")
    assert study.main(
        ["--models", "mockllm/model", "--fake", "oracle", "--limit", "20", "--log-dir", str(log_dir)]
    ) == 0
    return log_dir


@pytest.fixture
def logs(fake_logs, tmp_path):
    """A private copy, so tests can damage it."""
    import shutil

    target = tmp_path / "logs"
    shutil.copytree(fake_logs, target)
    return target


def find_log(log_dir, condition):
    for path in sorted(log_dir.glob("*.eval")):
        header = read_eval_log(str(path), header_only=True)
        if header.eval.metadata.get("condition") == condition:
            return path
    raise AssertionError(condition)


def test_good_logs_give_a_partial_report(logs, tmp_path):
    summaries = analyze(logs, tmp_path / "out")
    assert summaries["mockllm/model"]["conditions"]["full"]["accuracy"] == 1.0
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "[partial]" in report
    assert "[fake:oracle]" in report
    assert "What each number does and does not show" in report


def test_errored_log_status_stops_the_report(logs):
    path = find_log(logs, "no_story")
    log = read_eval_log(str(path))
    log.status = "error"
    write_eval_log(log, str(path))
    with pytest.raises(AnalysisError, match="status 'error'"):
        analyze(logs)


def test_errored_sample_stops_the_report(logs):
    from inspect_ai._util.error import EvalError

    path = find_log(logs, "no_story")
    log = read_eval_log(str(path))
    log.samples[3].error = EvalError(message="boom", traceback="", traceback_ansi="")
    write_eval_log(log, str(path))
    with pytest.raises(AnalysisError, match="errored sample"):
        analyze(logs)


def test_missing_context_check_stops_the_report(logs):
    find_log(logs, "context_check").unlink()
    with pytest.raises(AnalysisError, match="no passing context_check"):
        analyze(logs)


def test_failed_context_check_stops_the_report(logs):
    path = find_log(logs, "context_check")
    log = read_eval_log(str(path))
    log.results.scores[0].metrics["accuracy"].value = 0.0
    write_eval_log(log, str(path))
    with pytest.raises(AnalysisError, match="no passing context_check"):
        analyze(logs)


def test_mismatched_ids_in_logs_stop_the_report(logs):
    path = find_log(logs, "no_story")
    log = read_eval_log(str(path))
    log.samples = log.samples[:-1]
    write_eval_log(log, str(path))
    with pytest.raises(AnalysisError, match="different question IDs"):
        analyze(logs)


def test_refusal_removes_an_earlier_report(logs):
    analyze(logs)
    assert (logs / "report.md").exists()
    find_log(logs, "context_check").unlink()
    assert analysis.main([str(logs)]) == 1
    assert not (logs / "report.md").exists()


def test_outputs_in_the_log_dir_are_not_mistaken_for_logs(logs):
    analyze(logs)
    analyze(logs)
    assert study.main(
        ["--models", "mockllm/model", "--fake", "oracle", "--limit", "20", "--log-dir", str(logs)]
    ) == 0
