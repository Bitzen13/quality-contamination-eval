"""QuALITY loader, guardrails, and the shuffled and swapped option conditions."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

from inspect_ai.dataset import Sample

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = "data/QuALITY.v1.0.1.htmlstripped.dev"
LETTERS = "ABCD"
ABSTAIN_OPTION = "I don't know"
VALID_GOLD = {1, 2, 3, 4}
LETTERS_ONLY = "letters_only"
LETTERS_REORDERED = "letters_reordered"
LETTERS_RECALL = "letters_recall"
OPTIONS_ONLY = "options_only"
PARAPHRASED = "paraphrased"
HIDDEN_OPTION_MODES = (LETTERS_ONLY, LETTERS_REORDERED, LETTERS_RECALL)
OPTION_MODES = ("original", "shuffled", "swapped", OPTIONS_ONLY, PARAPHRASED, *HIDDEN_OPTION_MODES)
QUESTION_WITHHELD = (
    "The question has been withheld on purpose. Choose the option most likely to be the "
    "correct answer."
)


@dataclass(frozen=True)
class Question:
    """Only what a prompt or target may use. Writer and validation fields are dropped."""

    id: str
    article_id: str
    story: str
    question: str
    options: tuple[str, ...]
    gold_index: int
    difficult: int


@dataclass(frozen=True)
class QualityData:
    path: str
    sha256: str
    entries: int
    questions: list[Question]
    skipped: list[str] = field(default_factory=list)


def paraphrase_path(data: str | Path) -> Path:
    """Paraphrases live beside the dataset they were made from."""
    return Path(f"{resolve_data_path(data)}.paraphrases.jsonl")


def load_paraphrases(data: str | Path) -> tuple[dict[str, list[str]], str]:
    """Return ({question ID: four reworded options}, the file's SHA-256)."""
    path = paraphrase_path(data)
    if not path.exists():
        raise FileNotFoundError(f"No paraphrases at {path}. Run paraphrase.py first.")
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line.strip()]
    return {row["id"]: row["options"] for row in rows}, hashlib.sha256(raw).hexdigest()


def resolve_data_path(path: str | Path) -> Path:
    """Find a relative path from the project root if it isn't under the working directory.

    `inspect eval quality_probe/task.py@quality` runs the task from inside quality_probe/.
    """
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        from_root = PROJECT_ROOT / path
        if from_root.exists():
            return from_root
    return path


def load_questions(path: str | Path = DEFAULT_DATA) -> QualityData:
    """Load a QuALITY JSONL file, ordered by article, applying every load guardrail."""
    path = resolve_data_path(path)
    raw = path.read_bytes()
    entries = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line.strip()]
    rows = [(entry, q) for entry in entries for q in entry["questions"]]

    if not any("gold_label" in q for _, q in rows):
        raise ValueError(f"{path} has no gold_label (test split?). Refusing to load it.")

    seen: set[str] = set()
    questions: list[Question] = []
    skipped: list[str] = []
    for entry, q in rows:
        qid = q["question_unique_id"]
        if qid in seen:
            raise ValueError(f"Duplicate question_unique_id: {qid}")
        seen.add(qid)

        gold = q.get("gold_label")
        if type(gold) is not int or gold not in VALID_GOLD:
            logger.warning("Skipping %s: invalid gold_label %r", qid, gold)
            skipped.append(qid)
            continue
        if len(q["options"]) != len(LETTERS):
            raise ValueError(f"{qid} has {len(q['options'])} options, expected {len(LETTERS)}")

        questions.append(
            Question(
                id=qid,
                article_id=str(entry["article_id"]),
                story=entry["article"],
                question=q["question"],
                options=tuple(q["options"]),
                gold_index=gold - 1,
                difficult=1 if q.get("difficult") == 1 else 0,
            )
        )

    questions.sort(key=lambda q: q.article_id)  # stable: file order within an article
    return QualityData(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        entries=len(entries),
        questions=questions,
        skipped=skipped,
    )


def question_rng(seed: int, question_id: str) -> Random:
    """Per-question RNG, so results don't depend on sample order or --limit."""
    return Random(f"{seed}:{question_id}")


def shuffle_permutation(question_id: str, seed: int, size: int = len(LETTERS)) -> list[int]:
    """perm[shown_position] == original_position."""
    perm = list(range(size))
    question_rng(seed, question_id).shuffle(perm)
    return perm


def select_questions(data: QualityData, stories: int, per_story: int, seed: int = 0) -> list[str]:
    """Pick random stories and random questions from each. Same arguments, same questions."""
    by_story: dict[str, list[str]] = defaultdict(list)
    for question in data.questions:
        by_story[question.article_id].append(question.id)
    if stories > len(by_story):
        raise ValueError(f"Asked for {stories} stories, but the file has {len(by_story)}")

    rng = Random(f"{seed}:subset:{stories}:{per_story}")
    chosen: list[str] = []
    for article_id in rng.sample(sorted(by_story), stories):
        ids = sorted(by_story[article_id])
        if per_story > len(ids):
            raise ValueError(f"Story {article_id} has only {len(ids)} questions")
        chosen.extend(rng.sample(ids, per_story))
    return chosen


def normalize_option(text: str) -> str:
    """Casefold and collapse whitespace and punctuation."""
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


def swap_options(
    question: Question, pool: list[Question], seed: int
) -> tuple[list[str], list[dict]]:
    """Replace the wrong options with wrong options from other questions on this article.

    The gold keeps its letter. `pool` holds the article's questions with valid gold labels.
    """
    candidates = sorted(
        (other.id, index, text)
        for other in pool
        if other.id != question.id
        for index, text in enumerate(other.options)
        if index != other.gold_index
    )
    question_rng(seed, question.id).shuffle(candidates)

    gold = question.options[question.gold_index]
    taken = {normalize_option(gold)}
    chosen: list[tuple[str, str]] = []
    for source_id, _, text in candidates:
        key = normalize_option(text)
        if key in taken:
            continue
        taken.add(key)
        chosen.append((source_id, text))
        if len(chosen) == len(LETTERS) - 1:
            break
    if len(chosen) < len(LETTERS) - 1:
        raise ValueError(f"{question.id}: not enough distinct wrong options on its article")

    options: list[str] = []
    borrowed: list[dict] = []
    remaining = iter(chosen)
    for position in range(len(LETTERS)):
        if position == question.gold_index:
            options.append(gold)
            continue
        source_id, text = next(remaining)
        options.append(text)
        borrowed.append(
            {"letter": LETTERS[position], "text": text, "source_question_id": source_id}
        )
    return options, borrowed


def build_samples(
    data: QualityData,
    story: bool = True,
    options: str = "original",
    seed: int = 0,
    abstain: bool = False,
    paraphrases: dict[str, list[str]] | None = None,
) -> list[Sample]:
    """One Sample per question. gold_label reaches the Sample only as its target."""
    if options not in OPTION_MODES:
        raise ValueError(f"options must be one of {OPTION_MODES}, got {options!r}")
    if abstain and options in HIDDEN_OPTION_MODES:
        raise ValueError("abstain needs option text to add to, so it can't hide the options")
    if story and options == OPTIONS_ONLY:
        raise ValueError("options_only removes the question, so it runs without the story")
    if options == PARAPHRASED and paraphrases is None:
        raise ValueError("paraphrased needs paraphrases")

    pools: dict[str, list[Question]] = defaultdict(list)
    for question in data.questions:
        pools[question.article_id].append(question)

    samples: list[Sample] = []
    for question in data.questions:
        if options == PARAPHRASED and question.id not in paraphrases:
            continue  # analysis refuses the comparison if this leaves different question IDs
        shown = list(question.options)
        target_index = question.gold_index
        metadata: dict = {"article_id": question.article_id, "difficult": question.difficult}
        if story:
            metadata["story"] = question.story

        if options in HIDDEN_OPTION_MODES:
            # No option text anywhere in the sample, so it can't reach a prompt.
            if options == LETTERS_REORDERED:
                perm = shuffle_permutation(question.id, seed)
                metadata["letter_order"] = [LETTERS[i] for i in perm]
            if options == LETTERS_RECALL:
                metadata["recall"] = True
            samples.append(
                Sample(
                    id=question.id,
                    input=question.question,
                    target=LETTERS[question.gold_index],
                    metadata=metadata,
                )
            )
            continue

        if options == "shuffled":
            perm = shuffle_permutation(question.id, seed)
            shown = [question.options[original] for original in perm]
            target_index = perm.index(question.gold_index)
            metadata["perm"] = perm
        elif options == "swapped":
            shown, metadata["borrowed"] = swap_options(question, pools[question.article_id], seed)
        elif options == PARAPHRASED:
            shown = list(paraphrases[question.id])

        if abstain:
            shown.append(ABSTAIN_OPTION)
            metadata["abstain"] = True

        samples.append(
            Sample(
                id=question.id,
                input=QUESTION_WITHHELD if options == OPTIONS_ONLY else question.question,
                choices=shown,
                target=LETTERS[target_index],
                metadata=metadata,
            )
        )
    return samples


def _recall_decoy(
    question: Question, hidden: str, pool: list[Question], rng: Random
) -> tuple[str, str]:
    """A wrong option from another question on this article, closest in length to `hidden`.

    A weak baseline: it answers a different question, so a model can match the hidden option
    better just by staying on topic. Being above it is not evidence of memory on its own.
    """
    excluded = {normalize_option(text) for text in question.options}
    candidates = sorted(
        (other.id, text)
        for other in pool
        if other.id != question.id
        for index, text in enumerate(other.options)
        if index != other.gold_index and normalize_option(text) not in excluded
    )
    if not candidates:
        raise ValueError(f"{question.id}: no decoy available on its article")
    rng.shuffle(candidates)
    source, decoy = min(
        candidates, key=lambda c: abs(len(c[1].split()) - len(hidden.split()))
    )
    return decoy, source


def recall_samples(data: QualityData, seed: int = 0) -> list[Sample]:
    """Hide each question's longest wrong option and ask for it back.

    A long wrong option is hard to match by luck, but it can often be worked out from the
    options still shown, so compare the result with the same task on a control set.
    """
    pools: dict[str, list[Question]] = defaultdict(list)
    for question in data.questions:
        pools[question.article_id].append(question)

    samples: list[Sample] = []
    for question in data.questions:
        wrong = [i for i in range(len(LETTERS)) if i != question.gold_index]
        hidden_index = max(wrong, key=lambda i: (len(question.options[i]), -i))
        hidden = question.options[hidden_index]
        decoy, decoy_source = _recall_decoy(
            question, hidden, pools[question.article_id], question_rng(seed, question.id)
        )
        shown = "\n".join(
            f"{LETTERS[i]}) {'[removed]' if i == hidden_index else text}"
            for i, text in enumerate(question.options)
        )
        samples.append(
            Sample(
                id=question.id,
                input=f"{question.question}\n\n{shown}",
                target=hidden,
                metadata={
                    "article_id": question.article_id,
                    "difficult": question.difficult,
                    "decoy": decoy,
                    "decoy_source_question_id": decoy_source,
                    "hidden_letter": LETTERS[hidden_index],
                    "hidden_words": len(hidden.split()),
                },
            )
        )
    return samples
