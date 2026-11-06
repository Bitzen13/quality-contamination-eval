"""@task quality(...) and @task context_check(...)."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, multiple_choice, solver

from quality_probe.dataset import (
    DEFAULT_DATA,
    HIDDEN_OPTION_MODES,
    LETTERS,
    OPTION_MODES,
    PARAPHRASED,
    build_samples,
    load_paraphrases,
    load_questions,
    recall_samples,
)
from quality_probe.fakes import (
    fake_answer_generate,
    fake_context_generate,
    fake_multiple_choice,
    fake_paraphrase_generate,
    fake_recall_generate,
)
from quality_probe.prompts import (
    CONTEXT_CHECK_INSTRUCTION,
    PARAPHRASE_TEMPLATE,
    SENTINEL_END,
    SENTINEL_START,
    letters_only_prompt,
    story_system_message,
    verbatim_recall_prompt,
)
from quality_probe.scorer import context_scorer, quality_scorer, verbatim_scorer

DEFAULT_MAX_TOKENS = 512


def condition_name(story: bool, options: str, abstain: bool = False) -> str:
    base = "full" if story else "no_story"
    suffix = "" if options == "original" else f"_{options}"
    return f"{base}{suffix}{'_abstain' if abstain else ''}"


@task
def quality(
    story: bool = True,
    # "original" | "shuffled" | "swapped" | "options_only" | "paraphrased"
    # | "letters_only" | "letters_reordered" | "letters_recall"
    options: str = "original",
    seed: int = 0,
    abstain: bool = False,  # adds "I don't know" as a final option
    fake: str | None = None,  # "oracle" | "always_a" | "random" (tests only)
    data: str = DEFAULT_DATA,
    max_tokens: int = DEFAULT_MAX_TOKENS,  # a task's value beats a model's, so it lives here
) -> Task:
    if options not in OPTION_MODES:
        raise ValueError(f"options must be one of {OPTION_MODES}, got {options!r}")
    loaded = load_questions(data)
    paraphrases, paraphrases_sha256 = (None, None)
    if options == PARAPHRASED:
        paraphrases, paraphrases_sha256 = load_paraphrases(data)
    if options in HIDDEN_OPTION_MODES:
        answer_step = [
            letters_only_prompt(),
            fake_answer_generate(fake) if fake else _plain_generate(),
        ]
    else:
        answer_step = [fake_multiple_choice(fake) if fake else multiple_choice()]
    return Task(
        dataset=MemoryDataset(
            build_samples(
                loaded,
                story=story,
                options=options,
                seed=seed,
                abstain=abstain,
                paraphrases=paraphrases,
            ),
            name="QuALITY-dev",
            location=loaded.path,
        ),
        solver=[story_system_message(), *answer_step],
        scorer=quality_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=max_tokens),
        metadata={
            "condition": condition_name(story, options, abstain),
            "seed": seed,
            "fake": fake,
            "data_sha256": loaded.sha256,
            "skipped_question_ids": loaded.skipped,
            "paraphrases_sha256": paraphrases_sha256,
        },
    )


def context_sample(data: str = DEFAULT_DATA) -> Sample:
    """One sample built from the longest article and its longest question, with sentinels."""
    loaded = load_questions(data)
    longest_article = max(loaded.questions, key=lambda q: (len(q.story), q.article_id))
    questions = [q for q in loaded.questions if q.article_id == longest_article.article_id]
    longest = max(questions, key=lambda q: (len(q.question), q.id))

    for word in (SENTINEL_START, SENTINEL_END):
        if any(word.casefold() in q.story.casefold() for q in loaded.questions):
            raise ValueError(f"Sentinel {word} occurs in a story; pick another")

    options = "\n".join(f"{LETTERS[i]}) {text}" for i, text in enumerate(longest.options))
    return Sample(
        id=f"context_check:{longest.id}",
        input=f"{longest.question}\n\n{options}\n\n{CONTEXT_CHECK_INSTRUCTION}",
        target=f"{SENTINEL_START} {SENTINEL_END}",
        metadata={
            "article_id": longest.article_id,
            "story": f"{SENTINEL_START}\n\n{longest.story}\n\n{SENTINEL_END}",
        },
    )


@solver
def _plain_generate() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await generate(state)

    return solve


@task
def context_check(
    fake: str | None = None,  # any quality fake, or "undersized" (tests only)
    data: str = DEFAULT_DATA,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Task:
    """Fails when the server cut any part of the longest prompt in the study."""
    return Task(
        dataset=MemoryDataset([context_sample(data)], name="QuALITY-context-check"),
        solver=[
            story_system_message(),
            fake_context_generate(fake) if fake else _plain_generate(),
        ],
        scorer=context_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=max_tokens),
        metadata={"condition": "context_check", "fake": fake},
    )


@task
def verbatim_recall(
    seed: int = 0,
    fake: str | None = None,  # "oracle" | "always_a" | "random" (tests only)
    data: str = DEFAULT_DATA,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Task:
    """Ask for a removed wrong option back, word for word. Never shows the story."""
    loaded = load_questions(data)
    return Task(
        dataset=MemoryDataset(recall_samples(loaded, seed), name="QuALITY-verbatim-recall"),
        solver=[
            verbatim_recall_prompt(),
            fake_recall_generate(fake) if fake else _plain_generate(),
        ],
        scorer=verbatim_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=max_tokens),
        metadata={
            "condition": "verbatim_recall",
            "seed": seed,
            "fake": fake,
            "data_sha256": loaded.sha256,
        },
    )


@task
def paraphrase_options(
    fake: bool = False,  # tests only
    data: str = DEFAULT_DATA,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Task:
    """Generate reworded options. The story is not shown, and neither is the right answer."""
    loaded = load_questions(data)
    samples = [
        Sample(
            id=question.id,
            input=PARAPHRASE_TEMPLATE.format(
                question=question.question,
                options="\n".join(
                    f"{LETTERS[i]}) {text}" for i, text in enumerate(question.options)
                ),
            ),
            metadata={"options": list(question.options)},
        )
        for question in loaded.questions
    ]
    return Task(
        dataset=MemoryDataset(samples, name="QuALITY-paraphrase"),
        solver=[fake_paraphrase_generate() if fake else _plain_generate()],
        config=GenerateConfig(temperature=0, max_tokens=max_tokens),
        metadata={"condition": "paraphrase_generation", "data_sha256": loaded.sha256},
    )
