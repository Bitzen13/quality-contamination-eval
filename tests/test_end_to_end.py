"""All five conditions over the full dev set under mockllm, through study.py and analysis.py."""

from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from conftest import DATA, ROOT
from inspect_ai.log import read_eval_log, write_eval_log

import analysis
import study
from quality_probe import fakes
from quality_probe.dataset import LETTERS, build_samples, load_questions
from quality_probe.prompts import NO_STORY, SENTINEL_END, SENTINEL_START

CONDITIONS = [name for name, _, _ in study.CONDITIONS]
MODEL = "mockllm/model"
INSPECT = str(Path(sys.executable).parent / "inspect")


def run_fake(log_dir: Path, fake: str, *extra: str) -> int:
    return study.main(["--models", MODEL, "--fake", fake, "--log-dir", str(log_dir), *extra])


@pytest.fixture(scope="module")
def studies(tmp_path_factory):
    """One full study per fake, analysed from its logs."""
    results = {}
    for fake in ("oracle", "always_a", "random"):
        log_dir = tmp_path_factory.mktemp(fake)
        assert run_fake(log_dir, fake) == 0
        assert analysis.main([str(log_dir)]) == 0
        summary = json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))
        results[fake] = (log_dir, summary[MODEL])
    return results


def logs_by_condition(log_dir: Path) -> dict[str, Path]:
    found = {}
    for path in log_dir.glob("*.eval"):
        condition = read_eval_log(str(path), header_only=True).eval.metadata["condition"]
        found[condition] = path
    return found


def test_all_five_conditions_cover_the_full_dev_set(studies):
    for _, summary in studies.values():
        assert summary["questions"] == 2086
        assert summary["stories"] == 115
        assert set(summary["conditions"]) == set(CONDITIONS)
        assert all(summary["conditions"][c]["questions"] == 2086 for c in CONDITIONS)


def test_oracle_scores_100_percent_everywhere(studies):
    _, summary = studies["oracle"]
    for condition in CONDITIONS:
        entry = summary["conditions"][condition]
        assert entry["accuracy"] == 1.0
        assert entry["hard_accuracy"] == 1.0
        assert entry["status_rates"]["ok"] == 1.0
    assert all(result["value"] == 0 for result in summary["comparisons"].values())


def test_always_a_matches_each_conditions_share_of_gold_a(studies):
    _, summary = studies["always_a"]
    data = load_questions(DATA)
    for name, story, options in study.CONDITIONS:
        samples = build_samples(data, story=story, options=options)
        share = sum(s.target == "A" for s in samples) / len(samples)
        assert summary["conditions"][name]["accuracy"] == pytest.approx(share)


def test_random_lands_near_25_percent(studies):
    _, summary = studies["random"]
    for condition in CONDITIONS:
        assert summary["conditions"][condition]["accuracy"] == pytest.approx(0.25, abs=0.035)


def test_report_and_outputs_exist(studies):
    log_dir, _ = studies["oracle"]
    report = (log_dir / "report.md").read_text(encoding="utf-8")
    assert "model · QuALITY dev · 2,086 q · 115 stories" in report
    assert "[partial]" not in report
    assert "[fake:oracle]" in report
    for label in ("gap (full − no_story)", "letter probe", "option probe", "shuffle control"):
        assert label in report

    with open(log_dir / "results.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5 * 2086
    assert {"model", "condition", "question_id", "article_id", "difficult", "correct", "status"} <= set(rows[0])


def test_borrowed_options_are_exported(studies):
    log_dir, _ = studies["oracle"]
    with open(log_dir / "borrowed_options.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 2086
    assert all(row["source_question_id"] != row["question_id"] for row in rows)


def test_prompts_are_shaped_as_specified(studies):
    log_dir, _ = studies["oracle"]
    logs = logs_by_condition(log_dir)

    with_story = read_eval_log(str(logs["full"])).samples[0]
    data = load_questions(DATA)
    first = next(q for q in data.questions if q.id == with_story.id)
    system, user = with_story.messages[0], with_story.messages[1]
    assert system.role == "system" and user.role == "user"
    assert system.text.startswith(
        "You are answering a reading comprehension question about the story below."
    )
    assert f"<story>\n{first.story}\n</story>" in system.text
    assert first.question in user.text
    assert first.story not in user.text
    assert "ANSWER: $LETTER" in user.text
    assert "A) " + first.options[0] in user.text

    no_story = read_eval_log(str(logs["no_story"])).samples[0]
    assert no_story.messages[0].text == NO_STORY
    assert first.story not in " ".join(m.text for m in no_story.messages)


def test_every_with_story_prompt_for_an_article_shares_its_prefix(studies):
    log_dir, _ = studies["oracle"]
    samples = read_eval_log(str(logs_by_condition(log_dir)["full"])).samples
    systems: dict[str, set[str]] = {}
    for sample in samples:
        systems.setdefault(sample.metadata["article_id"], set()).add(sample.messages[0].text)
    assert len(systems) == 115
    assert all(len(texts) == 1 for texts in systems.values())


def test_fixed_generation_settings_are_recorded(studies):
    log_dir, _ = studies["oracle"]
    log = read_eval_log(str(logs_by_condition(log_dir)["full"]), header_only=True)
    assert log.plan.config.temperature == 0
    assert log.plan.config.max_tokens == 512
    assert log.eval.metadata["data_sha256"]


def test_in_log_metrics_exist_for_inspect_view(studies):
    log_dir, _ = studies["oracle"]
    log = read_eval_log(str(logs_by_condition(log_dir)["full"]), header_only=True)
    metrics = log.results.scores[0].metrics
    assert {"accuracy", "stderr", "0", "1", "all"} <= set(metrics)


# --- The commands in the README -------------------------------------------------------


def test_inspect_eval_cli_works_with_the_default_data_path(tmp_path):
    result = subprocess.run(
        [
            INSPECT, "eval", "quality_probe/task.py@quality",
            "-T", "story=false", "-T", "fake=oracle",
            "--model", MODEL, "--limit", "5",
            "--log-dir", str(tmp_path), "--display", "none",
        ],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    log = read_eval_log(str(next(tmp_path.glob("*.eval"))))
    assert log.status == "success"
    assert log.results.scores[0].metrics["accuracy"].value == 1.0


# --- Context check --------------------------------------------------------------------


def test_longest_story_and_sentinels_are_in_the_context_prompt(studies):
    log_dir, _ = studies["oracle"]
    sample = read_eval_log(str(logs_by_condition(log_dir)["context_check"])).samples[0]
    data = load_questions(DATA)
    longest = max(len(q.story) for q in data.questions)
    system, user = sample.messages[0].text, sample.messages[1].text

    assert longest < len(system) < longest + 400
    assert f"<story>\n{SENTINEL_START}\n\n" in system
    assert f"\n\n{SENTINEL_END}\n</story>" in system
    # Naming them in the instruction would let a truncated prompt pass.
    assert SENTINEL_START not in user
    assert SENTINEL_END not in user


def test_undersized_server_stops_the_study_and_analysis_refuses(tmp_path, capsys):
    assert run_fake(tmp_path, "undersized") == 1
    assert "Context check FAILED" in capsys.readouterr().err
    assert set(logs_by_condition(tmp_path)) == {"context_check"}
    # Only a failed check exists: nothing to report on.
    assert analysis.main([str(tmp_path)]) == 1
    assert not (tmp_path / "report.md").exists()


def test_model_that_ignores_the_prompt_fails_the_check(tmp_path):
    # Plain mockllm with no fake returns canned text with no sentinels.
    assert study.main(["--models", MODEL, "--log-dir", str(tmp_path)]) == 1
    assert set(logs_by_condition(tmp_path)) == {"context_check"}


def test_a_fixed_server_is_checked_again(tmp_path):
    assert run_fake(tmp_path, "undersized", "--limit", "5") == 1
    assert run_fake(tmp_path, "oracle", "--limit", "5") == 0
    assert set(logs_by_condition(tmp_path)) == set(CONDITIONS) | {"context_check"}


def test_fakes_refuse_real_models():
    with pytest.raises(RuntimeError, match="only run on mockllm"):
        import asyncio

        from inspect_ai.model import ModelName
        from inspect_ai.solver import TaskState

        state = TaskState(
            model=ModelName("anthropic/claude-haiku-5-5"),
            sample_id="q",
            epoch=1,
            input="q",
            messages=[],
            choices=["a", "b", "c", "d"],
        )
        asyncio.run(fakes._fake_generate(lambda s: "x")(state))


# --- Resume ---------------------------------------------------------------------------


def test_rerun_regenerates_nothing(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(fakes, "on_call", calls.append)
    assert run_fake(tmp_path, "oracle", "--limit", "10") == 0
    assert len(calls) == 5 * 10 + 1
    before = {p.name: p.stat().st_mtime for p in tmp_path.glob("*.eval")}

    calls.clear()
    assert run_fake(tmp_path, "oracle", "--limit", "10") == 0
    assert calls == []
    assert {p.name: p.stat().st_mtime for p in tmp_path.glob("*.eval")} == before


def test_retry_after_a_failure_reuses_completed_samples(tmp_path, monkeypatch):
    """A sample keeps failing until the task fails; eval_set's retry must reuse the rest."""
    samples = [s.id for s in build_samples(load_questions(DATA), story=False)[:40]]
    bad = samples[-1]
    attempts: Counter = Counter()
    completed: list[str] = []

    def hook(sample_id: str) -> None:
        attempts[sample_id] += 1
        # retry_on_error=3 means 4 tries per sample per run, which fails the first run
        if sample_id == bad and attempts[sample_id] <= 4:
            raise RuntimeError("simulated provider failure")
        completed.append(sample_id)

    monkeypatch.setattr(fakes, "on_call", hook)
    success = study.run_study(
        [MODEL], str(tmp_path), limit=40, fake="oracle", retry_wait=0.1
    )
    assert success
    assert attempts[bad] > 4

    # 40 samples in each of 5 conditions, plus the context check. Only the failing
    # condition's retried run may have repeated work, and never a completed sample of it
    # that was already flushed: every condition must finish with all 40 scored.
    logs = logs_by_condition(tmp_path)
    assert set(logs) == set(CONDITIONS) | {"context_check"}
    for condition in CONDITIONS:
        log = read_eval_log(str(logs[condition]))
        assert log.status == "success"
        assert len(log.samples) == 40
        assert all(sample.error is None for sample in log.samples)

    assert analysis.main([str(tmp_path)]) == 0


def _leave_started_copy(log_dir: Path, condition: str) -> Path:
    """Make the `started` log a hard interrupt leaves, older than the task's real log."""
    original = logs_by_condition(log_dir)[condition]
    log = read_eval_log(str(original))
    log.status = "started"
    log.results = None
    stale = log_dir / f"2020-01-01T00-00-00-00-00_quality_stale{condition}.eval"
    write_eval_log(log, str(stale))
    os.utime(stale, (1_600_000_000, 1_600_000_000))
    return stale


def test_rerun_clears_logs_an_interrupted_run_left_behind(tmp_path, monkeypatch):
    assert run_fake(tmp_path, "oracle", "--limit", "5") == 0
    stale = _leave_started_copy(tmp_path, "no_story")
    assert analysis.main([str(tmp_path)]) == 1  # refused while the stale log is there

    calls: list[str] = []
    monkeypatch.setattr(fakes, "on_call", calls.append)
    assert run_fake(tmp_path, "oracle", "--limit", "5") == 0
    assert not stale.exists()
    assert calls == []
    assert analysis.main([str(tmp_path)]) == 0


def test_started_log_with_no_successful_run_is_kept(tmp_path):
    assert run_fake(tmp_path, "oracle", "--limit", "5") == 0
    original = logs_by_condition(tmp_path)["no_story"]
    log = read_eval_log(str(original))
    log.status = "started"
    log.results = None
    write_eval_log(log, str(original))

    study.clear_superseded_logs(str(tmp_path))
    assert original.exists()
    assert analysis.main([str(tmp_path)]) == 1


def test_story_subset_runs_every_condition_and_is_marked_partial(tmp_path, monkeypatch):
    extra = ("--stories", "3", "--per-story", "2")
    calls: list[str] = []
    monkeypatch.setattr(fakes, "on_call", calls.append)
    assert run_fake(tmp_path, "oracle", *extra) == 0
    assert len(calls) == 5 * 6 + 1

    assert analysis.main([str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))[MODEL]
    assert summary["questions"] == 6
    assert summary["stories"] == 3
    assert "[partial]" in (tmp_path / "report.md").read_text(encoding="utf-8")

    calls.clear()
    assert run_fake(tmp_path, "oracle", *extra) == 0
    assert calls == []


# --- Letters only ---------------------------------------------------------------------

SUBSET = ("--stories", "4", "--per-story", "5")
LETTER_NAMES = [name for name, _, _ in study.LETTER_CONDITIONS]


def _letter_summary(log_dir: Path) -> dict:
    assert analysis.main([str(log_dir)]) == 0
    return json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))[MODEL]


def test_letters_only_prompts_hide_the_options(tmp_path):
    assert run_fake(tmp_path, "oracle", *SUBSET, "--letters-only") == 0
    logs = logs_by_condition(tmp_path)
    assert set(logs) == set(LETTER_NAMES) | {"context_check"}
    questions = {q.id: q for q in load_questions(DATA).questions}

    orders = set()
    for condition in LETTER_NAMES:
        for sample in read_eval_log(str(logs[condition])).samples:
            question = questions[sample.id]
            system, user = sample.messages[0].text, sample.messages[1].text
            assert question.question in user
            assert "withheld on purpose" in user
            for option in question.options:
                assert option not in user.replace(question.question, "")
            assert (question.story in system) == condition.startswith("full")
            order = sample.metadata.get("letter_order", list("ABCD"))
            assert user.split("\n\n")[-1].split("\n") == [f"{letter})" for letter in order]
            if condition == "full_letters_reordered":
                orders.add(tuple(order))
    assert len(orders) > 1


def test_letters_only_scores_under_each_fake(tmp_path):
    oracle = tmp_path / "oracle"
    assert run_fake(oracle, "oracle", *SUBSET, "--letters-only") == 0
    summary = _letter_summary(oracle)
    for condition in LETTER_NAMES:
        entry = summary["conditions"][condition]
        assert entry["accuracy"] == 1.0
        assert entry["letter_habit"]["excess"] > 0.5
    assert summary["comparisons"]["letters_story_effect"]["value"] == 0
    assert summary["comparisons"]["letters_reorder_effect"]["value"] == 0

    always_a = tmp_path / "always_a"
    assert run_fake(always_a, "always_a", *SUBSET, "--letters-only") == 0
    for condition, entry in _letter_summary(always_a)["conditions"].items():
        # Always picking one letter earns exactly its share, so nothing above the baseline.
        assert entry["letter_habit"]["excess"] == pytest.approx(0, abs=1e-12), condition


def test_letters_only_adds_to_an_existing_run_without_regenerating(tmp_path, monkeypatch):
    assert run_fake(tmp_path, "oracle", *SUBSET) == 0
    before = {p.name for p in tmp_path.glob("*.eval")}

    calls: list[str] = []
    monkeypatch.setattr(fakes, "on_call", calls.append)
    assert run_fake(tmp_path, "oracle", *SUBSET, "--letters-only") == 0
    assert len(calls) == 3 * 20
    assert before <= {p.name for p in tmp_path.glob("*.eval")}

    summary = _letter_summary(tmp_path)
    assert set(summary["conditions"]) == set(CONDITIONS) | set(LETTER_NAMES)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Letters-only conditions" in report
    assert "expected from letter habits alone" in report
    assert "letters only: listed − reordered" in report


def test_main_matrix_report_omits_the_letters_caveat(tmp_path):
    assert run_fake(tmp_path, "oracle", *SUBSET) == 0
    assert analysis.main([str(tmp_path)]) == 0
    assert "Letters-only conditions" not in (tmp_path / "report.md").read_text(encoding="utf-8")


# --- Re-scoring -----------------------------------------------------------------------


ALT_SCORER = '''
from inspect_ai.scorer import INCORRECT, Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState

@scorer(metrics=[accuracy()])
def quality_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=INCORRECT, metadata={"status": "parse_fail"})
    return score
'''


def test_rescoring_updates_the_report_with_no_new_model_calls(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(fakes, "on_call", calls.append)
    log_dir = tmp_path / "logs"
    assert run_fake(log_dir, "oracle", "--limit", "10") == 0
    assert analysis.main([str(log_dir)]) == 0
    summary = json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))[MODEL]
    assert summary["conditions"]["no_story"]["accuracy"] == 1.0
    calls.clear()

    target = logs_by_condition(log_dir)["no_story"]
    completions = [s.output.completion for s in read_eval_log(str(target)).samples]

    def rescore(scorer: str) -> None:
        result = subprocess.run(
            [INSPECT, "score", str(target), "--action", "overwrite", "--overwrite", "--scorer", scorer],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        )  # fmt: skip
        assert result.returncode == 0, result.stdout + result.stderr

    # The README's command, with the repo's parser: the result is unchanged.
    rescore("quality_probe/scorer.py@quality_scorer")
    assert analysis.main([str(log_dir)]) == 0
    summary = json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))[MODEL]
    assert summary["conditions"]["no_story"]["accuracy"] == 1.0

    # A changed parser: every answer now fails to parse.
    scorer_file = tmp_path / "changed_parser.py"
    scorer_file.write_text(ALT_SCORER, encoding="utf-8")
    rescore(f"{scorer_file}@quality_scorer")

    assert analysis.main([str(log_dir)]) == 0
    summary = json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))[MODEL]
    assert summary["conditions"]["no_story"]["accuracy"] == 0.0
    assert summary["conditions"]["no_story"]["status_rates"]["parse_fail"] == 1.0
    assert summary["conditions"]["full"]["accuracy"] == 1.0
    assert calls == []
    # In place: no second log for the condition, and the completions untouched.
    assert len(list(log_dir.glob("*.eval"))) == 6
    assert [s.output.completion for s in read_eval_log(str(target)).samples] == completions


# --- No network -----------------------------------------------------------------------


def test_analysis_runs_with_networking_disabled(studies, tmp_path, monkeypatch):
    log_dir, _ = studies["oracle"]
    real_connect = socket.socket.connect

    def guarded(self, address):
        # asyncio on Windows makes a loopback socketpair for its own event loop
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"Network call to {address}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: pytest.fail(f"DNS lookup for {a[0]}")
    )
    assert analysis.main([str(log_dir), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "report.md").exists()


def test_letter_constant_matches_the_template():
    assert LETTERS == "ABCD"
