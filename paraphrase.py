"""Generate reworded options with a model and save them beside the dataset. Re-run to fill gaps."""

from __future__ import annotations

import argparse
import json
import re
import sys

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log

from quality_probe.dataset import (
    DEFAULT_DATA,
    LETTERS,
    load_questions,
    normalize_option,
    paraphrase_path,
    select_questions,
)
from quality_probe.task import paraphrase_options
from study import max_tokens_for

# Shorter options, like a name, may legitimately stay as they are.
MUST_CHANGE_WORDS = 3
PASSES = 3


def parse_paraphrases(completion: str, originals: list[str]) -> list[str] | None:
    """Return four usable reworded options, or None if the completion can't be used."""
    match = re.search(r"\[.*\]", completion, re.DOTALL)
    if match is None:
        return None
    try:
        options = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(options, list) or len(options) != len(LETTERS):
        return None
    if not all(isinstance(option, str) and option.strip() for option in options):
        return None

    for option, original in zip(options, originals):
        unchanged = normalize_option(option) == normalize_option(original)
        if unchanged and len(original.split()) >= MUST_CHANGE_WORDS:
            return None
    # Rewording must not merge two options that were different.
    if len({normalize_option(o) for o in options}) < len(
        {normalize_option(o) for o in originals}
    ):
        return None
    return [option.strip() for option in options]


def read_existing(path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    return {row["id"]: row["options"] for row in rows}


def write_all(path, paraphrases: dict[str, list[str]]) -> None:
    lines = [
        json.dumps({"id": qid, "options": options}, ensure_ascii=False)
        for qid, options in sorted(paraphrases.items())
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(
    model: str,
    data: str,
    log_dir: str,
    wanted: list[str],
    fake: bool = False,
) -> list[str]:
    """Fill in paraphrases for `wanted`. Returns the IDs still missing afterwards."""
    path = paraphrase_path(data)
    paraphrases = read_existing(path)
    originals = {q.id: list(q.options) for q in load_questions(data).questions}

    for _ in range(PASSES):
        missing = [qid for qid in wanted if qid not in paraphrases]
        if not missing:
            break
        task = paraphrase_options(fake=fake, data=data, max_tokens=max_tokens_for(model))
        logs = inspect_eval(task, model=model, log_dir=log_dir, sample_id=missing, retry_on_error=3)
        for sample in read_eval_log(logs[0].location).samples or []:
            parsed = parse_paraphrases(sample.output.completion, originals[str(sample.id)])
            if parsed is not None:
                paraphrases[str(sample.id)] = parsed
        write_all(path, paraphrases)
    return [qid for qid in wanted if qid not in paraphrases]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="anthropic/claude-haiku-5-5")
    parser.add_argument("--log-dir", required=True, help="not the study's log directory")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--stories", type=int)
    parser.add_argument("--per-story", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fake", action="store_true", help="tests only")
    args = parser.parse_args(argv)
    if (args.stories is None) != (args.per_story is None):
        parser.error("--stories and --per-story go together")

    data = load_questions(args.data)
    wanted = (
        select_questions(data, args.stories, args.per_story, args.seed)
        if args.stories is not None
        else [q.id for q in data.questions]
    )
    missing = generate(args.model, args.data, args.log_dir, wanted, fake=args.fake)
    print(f"{len(wanted) - len(missing)} of {len(wanted)} questions have paraphrases.")
    if missing:
        print(f"Still missing after {PASSES} passes: {', '.join(missing[:10])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())
