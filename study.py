"""Run the context check, then eval_set over conditions x models. Re-run to resume."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval_set, task_with
from inspect_ai._util.file import filesystem
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.model import Model, get_model

from quality_probe.dataset import (
    DEFAULT_DATA,
    load_paraphrases,
    load_questions,
    select_questions,
)
from quality_probe.task import DEFAULT_MAX_TOKENS, context_check, quality, verbatim_recall

# (name, story, options)
CONDITIONS = [
    ("full", True, "original"),
    ("no_story", False, "original"),
    ("no_story_shuffled", False, "shuffled"),
    ("no_story_swapped", False, "swapped"),
    ("full_shuffled", True, "shuffled"),
]

# Run with --letters-only. Same log directory as the main matrix is fine: they add to it.
LETTER_CONDITIONS = [
    ("full_letters_only", True, "letters_only"),
    ("no_story_letters_only", False, "letters_only"),
    ("full_letters_reordered", True, "letters_reordered"),
]

# Run with --probes, which also runs the verbatim recall task.
PROBE_CONDITIONS = [
    ("no_story_options_only", False, "options_only"),
    ("no_story_letters_recall", False, "letters_recall"),
    ("full_letters_recall", True, "letters_recall"),
]

# Run with --paraphrased, after paraphrase.py has written the reworded options.
PARAPHRASE_CONDITIONS = [
    ("no_story_paraphrased", False, "paraphrased"),
    ("full_paraphrased", True, "paraphrased"),
]

# Reasoning models spend max_tokens on thinking too. These Claude models think by default.
MAX_TOKENS_OVERRIDES = {
    "anthropic/claude-haiku-5-5": 4096,
    "anthropic/claude-sonnet-5-5": 4096,
    "anthropic/claude-opus-5-5": 4096,
    "openai/gpt-6-luna": 4096,
}


def max_tokens_for(model: str) -> int:
    return MAX_TOKENS_OVERRIDES.get(model, DEFAULT_MAX_TOKENS)


def parse_model_args(items: list[str], models: list[str]) -> dict[str, dict[str, str]]:
    """Parse repeated `-M <model>@key=value` into {model: {key: value}}."""
    parsed: dict[str, dict[str, str]] = {model: {} for model in models}
    for item in items:
        model, separator, setting = item.partition("@")
        key, equals, value = setting.partition("=")
        if not separator or not equals or model not in parsed:
            raise SystemExit(f"-M expects <model>@key=value for a model in --models, got {item!r}")
        parsed[model][key] = value
    return parsed


def check_passed(header) -> bool:
    """Whether a context_check log header records a pass."""
    if header.status != "success" or header.results is None or not header.results.scores:
        return False
    return header.results.scores[0].metrics["accuracy"].value == 1.0


def _read_logs(log_dir: str) -> list[tuple[str, EvalLog]]:
    """Return (path, header) for every log in the directory."""
    if not Path(log_dir).exists():
        return []
    return [
        (info.name, read_eval_log(info, header_only=True)) for info in list_eval_logs(log_dir)
    ]


def _context_headers(log_dir: str) -> list[EvalLog]:
    return [
        header
        for _, header in _read_logs(log_dir)
        if (header.eval.metadata or {}).get("condition") == "context_check"
    ]


def _remove_log(path: str) -> None:
    filesystem(path).rm(path)


def clear_failed_checks(log_dir: str, models: list[str]) -> None:
    """Remove failed checks so a fixed server is tested again.

    eval_set would otherwise reuse the failed log, whose status is still success.
    """
    for path, header in _read_logs(log_dir):
        is_check = (header.eval.metadata or {}).get("condition") == "context_check"
        if is_check and header.eval.model in models and not check_passed(header):
            _remove_log(path)


def clear_superseded_logs(log_dir: str) -> None:
    """Remove `started` logs left by an interrupted run once their task has succeeded.

    A retry's log is seeded from the prior log, so the old one holds nothing the new one
    lacks. Inspect keeps these because it can't tell whether another process still writes
    them. Call this only after eval_set succeeds. analysis.py rightly refuses them otherwise.
    """
    logs = _read_logs(log_dir)
    succeeded = {header.eval.task_id for _, header in logs if header.status == "success"}
    for path, header in logs:
        if header.status == "started" and header.eval.task_id in succeeded:
            _remove_log(path)


def run_context_checks(
    models: dict[str, Model], log_dir: str, fake: str | None, data: str | None
) -> list[str]:
    """Return the models whose server delivered the whole longest prompt."""
    clear_failed_checks(log_dir, list(models))
    kwargs = {"data": data} if data else {}
    tasks = [
        task_with(context_check(fake=fake, max_tokens=max_tokens_for(name), **kwargs), model=model)
        for name, model in models.items()
    ]
    # A server that rejects the prompt outright is a failed check, not a crash.
    eval_set(
        tasks,
        log_dir=log_dir,
        log_dir_allow_dirty=True,
        retry_on_error=3,
        retry_attempts=1,
        retry_wait=1,
    )
    passed = {h.eval.model for h in _context_headers(log_dir) if check_passed(h)}
    return [name for name in models if name in passed]


def run_study(
    model_names: list[str],
    log_dir: str,
    limit: int | None = None,
    seed: int = 0,
    abstain: bool = False,
    fake: str | None = None,
    data: str | None = None,
    model_args: dict[str, dict[str, str]] | None = None,
    retry_wait: float | None = None,
    sample_ids: list[str] | None = None,
    conditions: list[tuple[str, bool, str]] | None = None,
    recall: bool = False,
) -> bool:
    """Return True if every model finished the full matrix."""
    model_args = model_args or {}
    models = {name: get_model(name, **model_args.get(name, {})) for name in model_names}

    passed = run_context_checks(models, log_dir, fake, data)
    for name in models:
        if name not in passed:
            print(f"Context check FAILED for {name}: study stopped for it.", file=sys.stderr)
    if not passed:
        return False

    kwargs = {"data": data} if data else {}
    # Task-level models rather than eval_set(model=...), because max_tokens is per model.
    tasks = [
        task_with(
            quality(
                story=story,
                options=options,
                seed=seed,
                abstain=abstain,
                fake=fake,
                max_tokens=max_tokens_for(name),
                **kwargs,
            ),
            model=models[name],
        )
        for name in passed
        for _, story, options in (conditions or CONDITIONS)
    ]
    if recall:
        tasks += [
            task_with(
                verbatim_recall(seed=seed, fake=fake, max_tokens=max_tokens_for(name), **kwargs),
                model=models[name],
            )
            for name in passed
        ]
    # allow_dirty because the context-check logs share this directory, as analysis expects.
    extra = {} if retry_wait is None else {"retry_wait": retry_wait}
    if sample_ids:
        extra["sample_id"] = sample_ids
    success, _ = eval_set(
        tasks, log_dir=log_dir, log_dir_allow_dirty=True, retry_on_error=3, limit=limit, **extra
    )
    if success:
        clear_superseded_logs(log_dir)
    return success and len(passed) == len(models)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="anthropic/claude-haiku-5-5", help="comma-separated")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--limit", type=int, help="development only; reports are marked [partial]")
    parser.add_argument("--stories", type=int, help="development only: random stories to use")
    parser.add_argument("--per-story", type=int, help="development only: questions per story")
    extra_sets = parser.add_mutually_exclusive_group()
    extra_sets.add_argument(
        "--letters-only",
        action="store_true",
        help="run the hidden-option conditions instead of the main matrix",
    )
    extra_sets.add_argument(
        "--probes",
        action="store_true",
        help="run options-only, direct letter recall and verbatim recall instead",
    )
    extra_sets.add_argument(
        "--paraphrased",
        action="store_true",
        help="run the reworded-option conditions instead (run paraphrase.py first)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--abstain", action="store_true", help='add an "I don\'t know" option')
    parser.add_argument("--fake", help="tests only: oracle | always_a | random | undersized")
    parser.add_argument("--data", help="dataset path")
    parser.add_argument(
        "-M",
        dest="model_args",
        action="append",
        default=[],
        help="<model>@key=value, e.g. vllm/foo@max_model_len=16384",
    )
    args = parser.parse_args(argv)

    if (args.stories is None) != (args.per_story is None):
        parser.error("--stories and --per-story go together")
    if args.stories is not None and args.limit is not None:
        parser.error("use --limit or --stories, not both")
    sample_ids = None
    if args.stories is not None:
        data = load_questions(args.data or DEFAULT_DATA)
        sample_ids = select_questions(data, args.stories, args.per_story, args.seed)

    conditions = None
    if args.letters_only:
        conditions = LETTER_CONDITIONS
    elif args.probes:
        conditions = PROBE_CONDITIONS
    elif args.paraphrased:
        conditions = PARAPHRASE_CONDITIONS
        try:
            available, _ = load_paraphrases(args.data or DEFAULT_DATA)
        except FileNotFoundError as error:
            parser.error(str(error))
        if sample_ids:
            missing = [qid for qid in sample_ids if qid not in available]
            if missing:
                parser.error(
                    f"{len(missing)} selected questions have no paraphrase. "
                    "Run paraphrase.py with the same --stories, --per-story and --seed."
                )

    names = [name.strip() for name in args.models.split(",") if name.strip()]
    success = run_study(
        names,
        args.log_dir,
        limit=args.limit,
        seed=args.seed,
        abstain=args.abstain,
        fake=args.fake,
        data=args.data,
        model_args=parse_model_args(args.model_args, names),
        sample_ids=sample_ids,
        conditions=conditions,
        recall=args.probes,
    )
    return 0 if success else 1


if __name__ == "__main__":
    # Inspect reads .env only through its own CLI. Loaded here, not in main(), so the
    # tests that call main() stay keyless.
    load_dotenv()
    sys.exit(main())
