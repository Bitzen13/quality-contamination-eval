"""Checks, metrics, paired story bootstrap and report, computed from Inspect logs alone."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.analysis import SampleColumn, samples_df
from inspect_ai.log import list_eval_logs, read_eval_log

from quality_probe.scorer import NEAR_MATCH, ABSTAIN, OK, PARSE_FAIL, REFUSED, STATUSES, TRUNCATED_OUTPUT
from study import check_passed

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20240229
NON_ANSWER_FLAG = 0.02
NON_ANSWERS = (PARSE_FAIL, REFUSED, TRUNCATED_OUTPUT)
OUTPUT_FILES = (
    "results.csv",
    "summary.json",
    "report.md",
    "borrowed_options.csv",
    "recall_results.csv",
)
NOT_IN_MATRIX = ("context_check", "verbatim_recall")
# Short options can match by luck, so recall is judged on options at least this long.
LONG_OPTION_WORDS = 5

# (key, label, left condition, right condition)
COMPARISONS = [
    ("gap", "gap (full − no_story)", "full", "no_story"),
    ("letter_probe", "letter probe (no_story − shuffled)", "no_story", "no_story_shuffled"),
    ("option_probe", "option probe (no_story − swapped)", "no_story", "no_story_swapped"),
    ("shuffle_control", "shuffle control (full − shuffled)", "full", "full_shuffled"),
    (
        "letters_story_effect",
        "letters only: story − no story",
        "full_letters_only",
        "no_story_letters_only",
    ),
    (
        "letters_reorder_effect",
        "letters only: listed − reordered",
        "full_letters_only",
        "full_letters_reordered",
    ),
    (
        "letters_recall_effect",
        "letters only: recall prompt − guess prompt",
        "no_story_letters_recall",
        "no_story_letters_only",
    ),
    (
        "question_clues",
        "question clues (no_story − options only)",
        "no_story",
        "no_story_options_only",
    ),
    (
        "wording_probe",
        "wording probe (no_story − paraphrased)",
        "no_story",
        "no_story_paraphrased",
    ),
    (
        "paraphrase_control",
        "paraphrase control (full − paraphrased)",
        "full",
        "full_paraphrased",
    ),
]
CONDITION_ORDER = [
    "full",
    "no_story",
    "no_story_shuffled",
    "no_story_swapped",
    "full_shuffled",
    "full_letters_only",
    "no_story_letters_only",
    "full_letters_reordered",
    "no_story_letters_recall",
    "full_letters_recall",
    "no_story_options_only",
    "no_story_paraphrased",
    "full_paraphrased",
]
LETTERS = "ABCD"

LETTERS_ONLY_CAVEAT = """\
- **Letters-only conditions.** With no option text there is nothing to eliminate, so \
accuracy above the letter-habit baseline is evidence of a memorized answer key. The baseline \
is what the model's letter preferences would score by luck given how often each letter is \
correct. A score at the baseline only rules out effects the sample is large enough to detect, \
and says nothing about memorized answer text. A drop when the letters are reordered means \
the model leaned on position in the list rather than the letter itself.
"""

OPTIONS_ONLY_CAVEAT = """\
- **Options only.** With no story and no question, a score above chance can only come from \
the options themselves: how the right answer is written, or memory of the option text. \
`no_story` minus this shows what the question adds, whether as clues or as a memory cue.
"""

PARAPHRASE_CAVEAT = """\
- **Paraphrased options.** A drop from `no_story` suggests reliance on the exact wording, \
while test-taking skill should survive rewording. Read it next to the paraphrase control: \
if `full` also drops, the rewording changed the difficulty and the probe can't be trusted. \
Model-written rewording should be reviewed, since it can make options easier to tell apart.
"""

RECALL_CAVEAT = """\
- **Verbatim recall.** Being above the decoy is not evidence of memory. The removed option \
shares its topic and much of its wording with the question and the options still shown, so \
a model can often work it out, while the decoy answers a different question. Compare with \
the same task on a control set, and read the closest matches in `recall_results.csv`. A \
result near the control proves little: a model may retain meaning without wording, or may \
not try. Recall also doesn't show how much exposure inflates a with-story score.
"""

CAVEATS = """\
## What each number does and does not show

- **Gap.** Shows how much of the score needs the story. A small gap is consistent with \
contamination but isn't proof, because above-chance no-story accuracy can also come from \
eliminating implausible options.
- **Letter probe.** A drop under shuffling means no-story success leaned on memorized \
letters. Read it next to the shuffle control.
- **Option probe.** A drop under swapping means no-story success leaned on the original \
options, either memorized items or cues among them. Borrowed options can be partly true \
for the new question, so review the export before trusting a drop. They are also often \
off-topic, which makes the right answer stand out, so a rise is expected and says nothing \
about memory, and it can hide a real drop.
- **Limit.** None of these detect memorization of the story text itself.
"""

_COLUMNS = [
    SampleColumn("question_id", path="id", required=True),
    SampleColumn("article_id", path="metadata.article_id"),
    SampleColumn("difficult", path="metadata.difficult"),
    SampleColumn("target", path="target"),
    SampleColumn("error", path="error"),
    SampleColumn("score_value", path="scores.quality_scorer.value"),
    SampleColumn("status", path="scores.quality_scorer.metadata.status"),
    SampleColumn("chosen_letter", path="scores.quality_scorer.metadata.chosen_letter"),
    SampleColumn(
        "landed_on_gold_original_letter",
        path="scores.quality_scorer.metadata.landed_on_gold_original_letter",
    ),
    SampleColumn("prompt_tokens", path="scores.quality_scorer.metadata.prompt_tokens"),
]


class AnalysisError(Exception):
    """A check failed, so no report may be produced."""


@dataclass(frozen=True)
class RunLog:
    path: str
    status: str
    model: str
    condition: str | None
    seed: int | None
    data_sha256: str | None
    fake: str | None
    partial: bool
    context_passed: bool


def read_run_logs(log_dir: str | Path) -> list[RunLog]:
    runs = []
    for info in list_eval_logs(str(log_dir)):
        header = read_eval_log(info, header_only=True)
        metadata = header.eval.metadata or {}
        config = header.eval.config
        runs.append(
            RunLog(
                path=info.name,
                status=header.status,
                model=header.eval.model,
                condition=metadata.get("condition"),
                seed=metadata.get("seed"),
                data_sha256=metadata.get("data_sha256"),
                fake=metadata.get("fake"),
                partial=config.limit is not None or config.sample_id is not None,
                context_passed=check_passed(header),
            )
        )
    return runs


def check_logs(runs: list[RunLog]) -> None:
    """Check 1 (log status), plus the study's shape. Raises AnalysisError."""
    if not runs:
        raise AnalysisError("No Inspect logs found.")
    for run in runs:
        if run.status != "success":
            raise AnalysisError(f"Log has status {run.status!r}, not success: {run.path}")
        if run.condition is None:
            raise AnalysisError(f"Log has no condition, so it isn't from this study: {run.path}")

    seen: dict[tuple[str, str], str] = {}
    for run in runs:
        if run.condition == "context_check":
            continue
        key = (run.model, run.condition)
        if key in seen:
            raise AnalysisError(
                f"Two logs for {run.model} / {run.condition}: {seen[key]} and {run.path}"
            )
        seen[key] = run.path

    study = [run for run in runs if run.condition != "context_check"]
    if len({run.seed for run in study}) > 1:
        raise AnalysisError("Logs use different seeds, so models saw different conditions.")
    if len({run.data_sha256 for run in study}) > 1:
        raise AnalysisError("Logs were built from different dataset files.")


def check_context(runs: list[RunLog]) -> None:
    """Check 3: every model with study logs has a passing context_check log."""
    passed = {run.model for run in runs if run.condition == "context_check" and run.context_passed}
    for model in sorted({run.model for run in runs if run.condition != "context_check"}):
        if model not in passed:
            raise AnalysisError(f"{model} has no passing context_check log in this directory.")


RESULT_COLUMNS = [
    "model",
    "condition",
    "question_id",
    "article_id",
    "difficult",
    "target",
    "correct",
    "status",
    "prompt_tokens",
    "chosen_letter",
    "landed_on_gold_original_letter",
]


def load_results(runs: list[RunLog]) -> pd.DataFrame:
    """One long table across every study log. Raises if any sample errored."""
    frames = []
    for run in runs:
        if run.condition in NOT_IN_MATRIX:
            continue
        frame = samples_df(run.path, columns=_COLUMNS, quiet=True)
        errored = frame[frame["error"].notna()]
        if len(errored):
            raise AnalysisError(
                f"Log contains {len(errored)} errored sample(s): {run.path} "
                f"(first: {errored['question_id'].iloc[0]})"
            )
        if frame["status"].isna().any():
            raise AnalysisError(f"Log contains unscored samples: {run.path}")
        frames.append(frame.assign(model=run.model, condition=run.condition))
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    results = pd.concat(frames, ignore_index=True)
    results["correct"] = results["score_value"].eq("C")
    results["difficult"] = results["difficult"].astype(int)
    return results[RESULT_COLUMNS].sort_values(["model", "condition", "question_id"]).reset_index(drop=True)


_RECALL_COLUMNS = [
    SampleColumn("question_id", path="id", required=True),
    SampleColumn("article_id", path="metadata.article_id"),
    SampleColumn("hidden_words", path="metadata.hidden_words"),
    SampleColumn("target", path="target"),
    SampleColumn("error", path="error"),
    SampleColumn("status", path="scores.verbatim_scorer.metadata.status"),
    SampleColumn("recalled", path="scores.verbatim_scorer.metadata.recalled"),
    SampleColumn("exact", path="scores.verbatim_scorer.metadata.exact"),
    SampleColumn("decoy_exact", path="scores.verbatim_scorer.metadata.decoy_exact"),
    SampleColumn("similarity", path="scores.verbatim_scorer.metadata.similarity"),
    SampleColumn("decoy_similarity", path="scores.verbatim_scorer.metadata.decoy_similarity"),
]


def load_recall(runs: list[RunLog]) -> pd.DataFrame:
    """One row per question across every verbatim recall log."""
    frames = []
    for run in runs:
        if run.condition != "verbatim_recall":
            continue
        frame = samples_df(run.path, columns=_RECALL_COLUMNS, quiet=True)
        if frame["error"].notna().any():
            raise AnalysisError(f"Log contains errored samples: {run.path}")
        if frame["status"].isna().any():
            raise AnalysisError(f"Log contains unscored samples: {run.path}")
        frames.append(frame.assign(model=run.model))
    if not frames:
        return pd.DataFrame()
    columns = [column.name for column in _RECALL_COLUMNS if column.name != "error"]
    return pd.concat(frames, ignore_index=True)[["model", *columns]]


def summarize_recall(
    frame: pd.DataFrame, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict:
    """Recall of the hidden option against the decoy, on options long enough to judge."""
    stories = pd.Index(sorted(frame["article_id"].unique()))
    long = frame["hidden_words"].astype(int) >= LONG_OPTION_WORDS

    def per_story(values: pd.Series) -> np.ndarray:
        summed = values.astype(float).groupby(frame["article_id"]).sum()
        return summed.reindex(stories, fill_value=0).to_numpy(dtype=float)

    near = frame["similarity"].astype(float) >= NEAR_MATCH
    decoy_near = frame["decoy_similarity"].astype(float) >= NEAR_MATCH
    numerators = {
        "exact": per_story(frame["exact"].astype(bool) & long),
        "decoy_exact": per_story(frame["decoy_exact"].astype(bool) & long),
        "near": per_story(near & long),
        "decoy_near": per_story(decoy_near & long),
        "similarity": per_story(frame["similarity"].astype(float).where(long, 0.0)),
        "decoy_similarity": per_story(frame["decoy_similarity"].astype(float).where(long, 0.0)),
    }
    long_count = per_story(long)
    denominators = {name: long_count for name in numerators}
    differences = {
        "exact_above_decoy": ("exact", "decoy_exact"),
        "near_above_decoy": ("near", "decoy_near"),
        "similarity_above_decoy": ("similarity", "decoy_similarity"),
    }
    intervals = bootstrap_ci(numerators, denominators, differences, samples=samples, seed=seed)

    total = long_count.sum()
    values = {name: float(num.sum() / total) if total else None for name, num in numerators.items()}
    summary: dict = {
        "questions": len(frame),
        "stories": len(stories),
        "long_options": int(total),
        "non_answer_rate": float(frame["status"].isin(NON_ANSWERS).mean()),
    }
    for name, value in values.items():
        summary[name] = {"value": value, "ci": list(intervals[name])}
    for name, (left, right) in differences.items():
        difference = None if not total else values[left] - values[right]
        summary[name] = {"value": difference, "ci": list(intervals[name])}
    return summary


def _families(conditions) -> list[str]:
    return sorted({"_abstain" if c.endswith("_abstain") else "" for c in conditions})


def check_pairing(results: pd.DataFrame) -> None:
    """Check 2: conditions being compared have identical question ID sets."""
    for model, frame in results.groupby("model"):
        ids = {c: set(g["question_id"]) for c, g in frame.groupby("condition")}
        for family in _families(ids):
            for _, _, left, right in COMPARISONS:
                a, b = ids.get(left + family), ids.get(right + family)
                if a is not None and b is not None and a != b:
                    raise AnalysisError(
                        f"{model}: {left + family} and {right + family} have different "
                        f"question IDs ({len(a ^ b)} differ)."
                    )


def bootstrap_ci(
    numerators: dict[str, np.ndarray],
    denominators: dict[str, np.ndarray],
    differences: dict[str, tuple[str, str]],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, tuple[float, float]]:
    """Story-level paired bootstrap: all metrics and differences use the same resamples.

    Each array holds one value per story. A metric is sum(numerator) / sum(denominator).
    """
    stories = len(next(iter(numerators.values())))
    weights = np.random.default_rng(seed).multinomial(
        stories, np.full(stories, 1 / stories), size=samples
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = {
            name: (weights @ numerators[name]) / (weights @ denominators[name])
            for name in numerators
        }
    for name, (left, right) in differences.items():
        draws[name] = draws[left] - draws[right]
    return {
        name: tuple(float(v) for v in np.nanpercentile(values, [2.5, 97.5]))
        for name, values in draws.items()
    }


def letter_habit_baseline(
    rows: pd.DataFrame,
    stories: pd.Index,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """What the model's letter preferences would score by luck, and accuracy above that.

    Baseline = sum over letters of P(model picks letter) * P(letter is correct). Non-answers
    can't score, so they count in the denominator only. The bootstrap draws the same
    resamples as bootstrap_ci (same seed and story count).
    """
    index = stories.get_indexer(rows["article_id"])
    chosen = np.zeros((len(stories), len(LETTERS)))
    gold = np.zeros((len(stories), len(LETTERS)))
    for position, letter in enumerate(LETTERS):
        np.add.at(chosen[:, position], index[rows["chosen_letter"].eq(letter).to_numpy()], 1)
        np.add.at(gold[:, position], index[rows["target"].eq(letter).to_numpy()], 1)
    count = np.bincount(index, minlength=len(stories)).astype(float)
    correct = np.bincount(index, weights=rows["correct"].astype(float), minlength=len(stories))

    weights = np.random.default_rng(seed).multinomial(
        len(stories), np.full(len(stories), 1 / len(stories)), size=samples
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        totals = (weights @ count)[:, None]
        draws = ((weights @ chosen / totals) * (weights @ gold / totals)).sum(axis=1)
        excess = (weights @ correct) / totals[:, 0] - draws

    baseline = float(((chosen.sum(0) / count.sum()) * (gold.sum(0) / count.sum())).sum())
    return {
        "baseline": baseline,
        "baseline_ci": [float(v) for v in np.nanpercentile(draws, [2.5, 97.5])],
        "excess": float(correct.sum() / count.sum() - baseline),
        "excess_ci": [float(v) for v in np.nanpercentile(excess, [2.5, 97.5])],
    }


def _metric_arrays(frame: pd.DataFrame, condition: str, stories: pd.Index):
    """Per-story numerators and denominators for one condition's metrics."""
    rows = frame[frame["condition"] == condition]

    def per_story(series: pd.Series) -> np.ndarray:
        summed = series.groupby(rows["article_id"]).sum()
        return summed.reindex(stories, fill_value=0).to_numpy(dtype=float)

    count = per_story(pd.Series(1, index=rows.index))
    hard = rows["difficult"].eq(1)
    ok = rows["status"].eq(OK)
    non_answer = rows["status"].isin(NON_ANSWERS)
    landed = rows["landed_on_gold_original_letter"].eq(True)
    arrays = {
        "accuracy": (per_story(rows["correct"].astype(int)), count),
        "hard_accuracy": (per_story((rows["correct"] & hard).astype(int)), per_story(hard.astype(int))),
        "non_answer_rate": (per_story(non_answer.astype(int)), count),
        "answered_accuracy": (per_story((rows["correct"] & ok).astype(int)), per_story(ok.astype(int))),
    }
    if condition.startswith("no_story_shuffled"):
        arrays["gold_original_letter_share"] = (per_story(landed.astype(int)), per_story(ok.astype(int)))
    return arrays


def summarize_model(
    frame: pd.DataFrame, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict:
    """Metrics, status rates, paired differences and intervals for one model."""
    stories = pd.Index(sorted(frame["article_id"].unique()))
    numerators: dict[str, np.ndarray] = {}
    denominators: dict[str, np.ndarray] = {}
    conditions: dict[str, dict] = {}

    for condition, rows in frame.groupby("condition"):
        entry: dict = {
            "questions": len(rows),
            "stories": rows["article_id"].nunique(),
            "status_rates": {s: float(rows["status"].eq(s).mean()) for s in STATUSES},
        }
        for metric, (num, den) in _metric_arrays(frame, condition, stories).items():
            numerators[f"{condition}.{metric}"] = num
            denominators[f"{condition}.{metric}"] = den
            entry[metric] = float(num.sum() / den.sum()) if den.sum() else None

        prompt_tokens = pd.to_numeric(rows["prompt_tokens"], errors="coerce")
        entry["prompt_tokens"] = int(prompt_tokens.sum()) if prompt_tokens.notna().any() else None
        hidden_options = "_letters_" in condition
        if hidden_options:
            entry["letter_habit"] = letter_habit_baseline(rows, stories, samples, seed)
        if condition.startswith("no_story_shuffled") or hidden_options:
            letters = rows["chosen_letter"].dropna()
            entry["letter_distribution"] = {
                letter: float(share)
                for letter, share in letters.value_counts(normalize=True).sort_index().items()
            }
        if condition.endswith("_abstain"):
            # SAT-style: +1 correct, 0 abstain, -1/3 any other answer or non-answer
            wrong = (~rows["correct"]) & rows["status"].ne(ABSTAIN)
            entry["sat_score"] = float((rows["correct"].sum() - wrong.sum() / 3) / len(rows))
        conditions[condition] = entry

    differences = {}
    for family in _families(conditions):
        for key, _, left, right in COMPARISONS:
            if left + family in conditions and right + family in conditions:
                differences[key + family] = (f"{left + family}.accuracy", f"{right + family}.accuracy")

    intervals = bootstrap_ci(numerators, denominators, differences, samples=samples, seed=seed)
    for name, interval in intervals.items():
        if "." in name:
            condition, metric = name.split(".")
            conditions[condition][f"{metric}_ci"] = list(interval)

    comparisons = {
        key: {
            "value": conditions[left.split(".")[0]]["accuracy"]
            - conditions[right.split(".")[0]]["accuracy"],
            "ci": list(intervals[key]),
        }
        for key, (left, right) in differences.items()
    }
    return {
        "questions": frame["question_id"].nunique(),
        "stories": len(stories),
        "conditions": conditions,
        "comparisons": comparisons,
    }


def _ci(interval) -> str:
    return "[—, —]" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _value(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _condition_sort_key(condition: str) -> tuple:
    base = condition.removesuffix("_abstain")
    rank = CONDITION_ORDER.index(base) if base in CONDITION_ORDER else len(CONDITION_ORDER)
    return (condition.endswith("_abstain"), rank, condition)


def _model_block(model: str, summary: dict, partial: bool, fake: str | None) -> list[str]:
    tags = ("  [partial]" if partial else "") + (f"  [fake:{fake}]" if fake else "")
    short = model.split("/", 1)[-1]
    lines = [f"{short} · QuALITY dev · {summary['questions']:,} q · {summary['stories']} stories{tags}"]

    conditions = summary["conditions"]
    for condition in sorted(conditions, key=_condition_sort_key):
        entry = conditions[condition]
        line = (
            f"  {condition:<26}acc {_value(entry['accuracy'])} {_ci(entry.get('accuracy_ci'))}"
            f"   hard {_value(entry['hard_accuracy'])} {_ci(entry.get('hard_accuracy_ci'))}"
            f"   non-answer {entry['non_answer_rate'] * 100:.1f}%"
        )
        if entry["non_answer_rate"] > NON_ANSWER_FLAG:
            line += "  ⚠ above 2%"
        lines.append(line)
    for family in _families(conditions):
        for key, label, _, _ in COMPARISONS:
            result = summary["comparisons"].get(key + family)
            if result:
                label_text = label + (" [abstain]" if family else "")
                lines.append(f"  {label_text:<45}{result['value']:+.3f} {_ci(result['ci'])}")
    return lines


def _detail_table(summary: dict) -> list[str]:
    statuses = " | ".join(STATUSES)
    lines = [
        f"| condition | questions | accuracy | hard | answered-only | {statuses} | SAT | prompt tokens |",
        "|---|" + "---|" * (len(STATUSES) + 6),
    ]
    for condition, entry in sorted(summary["conditions"].items()):
        rates = " | ".join(f"{entry['status_rates'][s] * 100:.1f}%" for s in STATUSES)
        sat = _value(entry["sat_score"]) if "sat_score" in entry else "—"
        tokens = "—" if entry["prompt_tokens"] is None else f"{entry['prompt_tokens']:,}"
        lines.append(
            f"| {condition} | {entry['questions']:,} | {_value(entry['accuracy'])} "
            f"| {_value(entry['hard_accuracy'])} | {_value(entry['answered_accuracy'])} "
            f"| {rates} | {sat} | {tokens} |"
        )
    return lines


def _letter_lines(summary: dict) -> list[str]:
    lines = []
    for condition, entry in sorted(summary["conditions"].items()):
        if "letter_distribution" not in entry:
            continue
        distribution = ", ".join(f"{k} {v * 100:.1f}%" for k, v in entry["letter_distribution"].items())
        habit = entry.get("letter_habit")
        if habit:
            lines.append(
                f"- `{condition}`: accuracy {_value(entry['accuracy'])} against "
                f"{_value(habit['baseline'])} expected from letter habits alone, so "
                f"{habit['excess']:+.3f} {_ci(habit['excess_ci'])} above it. "
                f"Letter distribution: {distribution}."
            )
            continue
        share = entry["gold_original_letter_share"]
        lines.append(
            f"- `{condition}`: {_value(share)} {_ci(entry.get('gold_original_letter_share_ci'))} "
            f"of answers land on the gold's original letter. Letter distribution: {distribution}."
        )
    return lines


def _recall_lines(summary: dict) -> list[str]:
    recall = summary.get("verbatim_recall")
    if not recall:
        return []

    def row(label: str, key: str, baseline: str, difference: str) -> str:
        return (
            f"| {label} | {_value(recall[key]['value'])} {_ci(recall[key]['ci'])} "
            f"| {_value(recall[baseline]['value'])} {_ci(recall[baseline]['ci'])} "
            f"| {_value(recall[difference]['value'])} {_ci(recall[difference]['ci'])} |"
        )

    return [
        f"### Verbatim recall ({recall['long_options']:,} options of at least "
        f"{LONG_OPTION_WORDS} words; non-answers {recall['non_answer_rate'] * 100:.1f}%)",
        "",
        "| Measure | Removed option | Decoy (weak baseline) | Difference |",
        "|---|---|---|---|",
        row("Exact match", "exact", "decoy_exact", "exact_above_decoy"),
        row(f"Near match (similarity ≥ {NEAR_MATCH})", "near", "decoy_near", "near_above_decoy"),
        row("Mean similarity", "similarity", "decoy_similarity", "similarity_above_decoy"),
    ]


def build_report(summaries: dict[str, dict], runs: list[RunLog], excluded: list[str]) -> str:
    sections = ["# QuALITY contamination probe", ""]
    if excluded:
        names = ", ".join(excluded)
        sections += [f"**Excluded: context check failed and no study logs exist for {names}.**", ""]
    for model, summary in summaries.items():
        model_runs = [r for r in runs if r.model == model and r.condition != "context_check"]
        partial = any(run.partial for run in model_runs)
        fake = next((run.fake for run in model_runs if run.fake), None)
        sections += ["```text", *_model_block(model, summary, partial, fake), "```", ""]
        sections += ["### Details", "", *_detail_table(summary), ""]
        flagged = [
            c for c, e in summary["conditions"].items() if e["non_answer_rate"] > NON_ANSWER_FLAG
        ]
        if flagged:
            sections += [f"**Non-answer rate above 2%:** {', '.join(sorted(flagged))}.", ""]
        letters = _letter_lines(summary)
        if letters:
            sections += [*letters, ""]
        recall_lines = _recall_lines(summary)
        if recall_lines:
            sections += [*recall_lines, ""]
    present = {condition for summary in summaries.values() for condition in summary["conditions"]}
    extra_caveats = ""
    if any("_letters_" in condition for condition in present):
        extra_caveats += LETTERS_ONLY_CAVEAT
    if any("options_only" in condition for condition in present):
        extra_caveats += OPTIONS_ONLY_CAVEAT
    if any("paraphrased" in condition for condition in present):
        extra_caveats += PARAPHRASE_CAVEAT
    if any("verbatim_recall" in summary for summary in summaries.values()):
        extra_caveats += RECALL_CAVEAT
    caveats = CAVEATS.replace("- **Limit.**", extra_caveats + "- **Limit.**")
    sections += [
        f"Intervals: story-level paired bootstrap, B = {BOOTSTRAP_SAMPLES:,}, "
        "2.5th and 97.5th percentiles.",
        "",
        caveats,
    ]
    return "\n".join(sections)


def export_borrowed(runs: list[RunLog], path: Path) -> int:
    """Write every borrowed option for review. Returns the row count."""
    rows = []
    for run in runs:
        if run.condition is None or "swapped" not in run.condition:
            continue
        log = read_eval_log(run.path)
        for sample in log.samples or []:
            for item in sample.metadata.get("borrowed", []):
                rows.append(
                    {
                        "model": run.model,
                        "condition": run.condition,
                        "question_id": sample.id,
                        "question": sample.input,
                        "gold_letter": sample.target,
                        "gold_text": (sample.choices or [])[ord(str(sample.target)) - ord("A")],
                        "letter": item["letter"],
                        "borrowed_text": item["text"],
                        "source_question_id": item["source_question_id"],
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return len(rows)


def analyze(log_dir: str | Path, out_dir: str | Path | None = None) -> dict:
    """Run every check, then write the outputs. Raises AnalysisError with no report written."""
    out = Path(out_dir) if out_dir else Path(log_dir)
    # A refused run must not leave an earlier report looking current.
    for name in OUTPUT_FILES:
        (out / name).unlink(missing_ok=True)

    runs = read_run_logs(log_dir)
    check_logs(runs)
    check_context(runs)
    results = load_results(runs)
    recall = load_recall(runs)
    if results.empty and recall.empty:
        raise AnalysisError("No study logs found, only context checks.")
    check_pairing(results)

    study_models = {run.model for run in runs if run.condition != "context_check"}
    excluded = sorted({run.model for run in runs} - study_models)
    summaries = {
        model: summarize_model(frame)
        for model, frame in results.groupby("model", sort=True)
    }
    if not recall.empty:
        for model, frame in recall.groupby("model", sort=True):
            summary = summaries.setdefault(
                model,
                {
                    "questions": len(frame),
                    "stories": frame["article_id"].nunique(),
                    "conditions": {},
                    "comparisons": {},
                },
            )
            summary["verbatim_recall"] = summarize_recall(frame)

    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "results.csv", index=False, encoding="utf-8")
    if not recall.empty:
        recall.to_csv(out / "recall_results.csv", index=False, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (out / "report.md").write_text(build_report(summaries, runs, excluded), encoding="utf-8")
    export_borrowed(runs, out / "borrowed_options.csv")
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir")
    parser.add_argument("--out", help="output directory (default: the log directory)")
    args = parser.parse_args(argv)
    try:
        analyze(args.log_dir, args.out)
    except AnalysisError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    print(f"Wrote the report and results to {args.out or args.log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
