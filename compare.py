"""Compare a QuALITY run with a control-set run on the same stories. Reads reports only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import AnalysisError, bootstrap_ci


def _per_story(frame: pd.DataFrame, condition: str, stories: list[str]):
    rows = frame[frame["condition"] == condition]
    grouped = rows.groupby("article_id")["correct"]
    correct = grouped.sum().reindex(stories, fill_value=0).to_numpy(dtype=float)
    count = grouped.size().reindex(stories, fill_value=0).to_numpy(dtype=float)
    return correct, count


def compare_runs(benchmark: pd.DataFrame, control: pd.DataFrame) -> dict:
    """Per model: benchmark minus control, with and without the story, on shared stories.

    The difference of those differences is the part of the no-story advantage on the
    benchmark that a difference in reading difficulty can't explain.
    """
    models = sorted(set(benchmark["model"]) & set(control["model"]))
    if not models:
        raise AnalysisError("The two runs have no model in common.")

    summary = {}
    for model in models:
        a = benchmark[benchmark["model"] == model]
        b = control[control["model"] == model]
        for name, frame in (("benchmark", a), ("control", b)):
            missing = {"full", "no_story"} - set(frame["condition"])
            if missing:
                raise AnalysisError(f"{model}: the {name} run is missing {sorted(missing)}.")
        stories = sorted(set(a["article_id"]) & set(b["article_id"]))
        if not stories:
            raise AnalysisError(f"{model}: the two runs share no stories.")

        numerators, denominators = {}, {}
        for name, frame in (("benchmark", a), ("control", b)):
            for condition in ("full", "no_story"):
                correct, count = _per_story(frame, condition, stories)
                numerators[f"{name}_{condition}"] = correct
                denominators[f"{name}_{condition}"] = count
        differences = {
            "no_story_difference": ("benchmark_no_story", "control_no_story"),
            "full_difference": ("benchmark_full", "control_full"),
        }
        intervals = bootstrap_ci(numerators, denominators, differences)

        # Same resamples again, so the difference of differences stays paired.
        weights = np.random.default_rng(20240229).multinomial(
            len(stories), np.full(len(stories), 1 / len(stories)), size=10_000
        )
        draws = {
            name: (weights @ numerators[name]) / (weights @ denominators[name])
            for name in numerators
        }
        difference_of_differences = (draws["benchmark_no_story"] - draws["control_no_story"]) - (
            draws["benchmark_full"] - draws["control_full"]
        )

        values = {
            name: float(numerators[name].sum() / denominators[name].sum()) for name in numerators
        }
        no_story = values["benchmark_no_story"] - values["control_no_story"]
        full = values["benchmark_full"] - values["control_full"]
        model_summary = {
            "shared_stories": len(stories),
            "questions": {"benchmark": len(a), "control": len(b)},
        }
        for name, value in values.items():
            model_summary[name] = {"value": value, "ci": list(intervals[name])}
        model_summary["no_story_difference"] = {
            "value": no_story,
            "ci": list(intervals["no_story_difference"]),
        }
        model_summary["full_difference"] = {"value": full, "ci": list(intervals["full_difference"])}
        model_summary["difference_of_differences"] = {
            "value": no_story - full,
            "ci": [float(v) for v in np.nanpercentile(difference_of_differences, [2.5, 97.5])],
        }
        summary[model] = model_summary
    return summary


def _line(label: str, entry: dict) -> str:
    return f"| {label} | {entry['value']:+.3f} | [{entry['ci'][0]:.3f}, {entry['ci'][1]:.3f}] |"


def build_report(summary: dict) -> str:
    lines = ["# QuALITY vs control set", ""]
    for model, entry in summary.items():
        lines += [
            f"## {model} ({entry['shared_stories']} shared stories)",
            "",
            "| Measure | Value | 95% interval |",
            "|---|---|---|",
            _line("Benchmark, with story", entry["benchmark_full"]),
            _line("Control, with story", entry["control_full"]),
            _line("Benchmark, no story", entry["benchmark_no_story"]),
            _line("Control, no story", entry["control_no_story"]),
            _line("With-story difference (should be about 0)", entry["full_difference"]),
            _line("No-story difference", entry["no_story_difference"]),
            _line("Difference of differences", entry["difference_of_differences"]),
            "",
        ]
    lines += [
        "A no-story advantage on the benchmark that the control doesn't share points to exposure "
        "to the benchmark itself. Story familiarity can't explain it, because both sets use the "
        "same stories. Differences in how the two sets were written still can, so a with-story "
        "difference far from 0 means the sets aren't matched and this comparison can't be trusted.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir")
    parser.add_argument("control_dir")
    parser.add_argument("--out", default="comparison.md")
    args = parser.parse_args(argv)
    try:
        for directory in (args.benchmark_dir, args.control_dir):
            if not (Path(directory) / "results.csv").exists():
                raise AnalysisError(f"{directory} has no report. Run analysis.py on it first.")
        summary = compare_runs(
            pd.read_csv(Path(args.benchmark_dir) / "results.csv"),
            pd.read_csv(Path(args.control_dir) / "results.csv"),
        )
    except AnalysisError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    Path(args.out).write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({model: e["difference_of_differences"] for model, e in summary.items()}))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
