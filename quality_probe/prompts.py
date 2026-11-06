"""With-story and no-story system prompts, and the context-check prompt."""

from __future__ import annotations

from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import Generate, Solver, TaskState, solver

WITH_STORY = (
    "You are answering a reading comprehension question about the story below.\n\n"
    "<story>\n{story}\n</story>"
)
NO_STORY = (
    "You are answering a multiple-choice question about a story that has been "
    "withheld on purpose. Answer anyway with your best guess."
)

LETTERS_ONLY_TEMPLATE = (
    "Answer the following multiple choice question. The text of the answer options has "
    "been withheld on purpose, so answer anyway with your best guess. The entire content "
    "of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) "
    "where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)

LETTERS_RECALL_TEMPLATE = (
    "This question comes from the QuALITY reading comprehension benchmark (Pang et al., 2022). "
    "The text of its answer options has been withheld on purpose. From memory, give the "
    "letter of the correct answer in the published benchmark. If you do not remember it, "
    "give your best attempt. The entire content of your response should be of the following "
    "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}."
    "\n\n{question}\n\n{choices}"
)

VERBATIM_RECALL_TEMPLATE = (
    "This question comes from the QuALITY reading comprehension benchmark (Pang et al., 2022). "
    "One of its four answer options has been removed. From memory, write the removed option "
    "exactly as it appears in the benchmark. If you do not remember it, give your best "
    "attempt. Finish with a final line of the form 'OPTION: <text>'.\n\n{question}"
)

PARAPHRASE_TEMPLATE = (
    "Below are the four answer options to a multiple-choice question. Reword each option so "
    "it means exactly the same thing in different words. Keep names and numbers, keep each "
    "about the same length, and don't make any option more or less plausible. An option too "
    "short to reword, such as a name, may stay as it is. Reply with only a JSON array of the "
    "four reworded options, in the same order.\n\nQuestion: {question}\n\n{options}"
)

# Made-up words, so a model can't produce them without having received them. They are
# never named in the user message: a server that cut them off must fail the check.
SENTINEL_START = "VELLOXITE"
SENTINEL_END = "QUARMBLETON"

CONTEXT_CHECK_INSTRUCTION = (
    "Do not answer the question above. The story opens with a made-up word in capital "
    "letters and closes with a different made-up word in capital letters. Reply with "
    "only those two words, the opening word first."
)


def system_prompt(story: str | None) -> str:
    # replace, not format: stories may contain braces
    return NO_STORY if story is None else WITH_STORY.replace("{story}", story)


@solver
def story_system_message() -> Solver:
    """Put the story, or the withheld-story notice, in the system message.

    A plain insert rather than Inspect's system_message(), which formats a template
    with every metadata field. The story is present only in with-story conditions.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        content = system_prompt(state.metadata.get("story"))
        state.messages.insert(0, ChatMessageSystem(content=content))
        return state

    return solve


@solver
def letters_only_prompt() -> Solver:
    """Replace the user message with the question and bare letters, in the sample's order."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        order = state.metadata.get("letter_order") or list("ABCD")
        template = (
            LETTERS_RECALL_TEMPLATE if state.metadata.get("recall") else LETTERS_ONLY_TEMPLATE
        )
        state.user_prompt.text = template.format(
            letters=",".join(order),
            question=state.user_prompt.text,
            choices="\n".join(f"{letter})" for letter in order),
        )
        return state

    return solve


@solver
def verbatim_recall_prompt() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = VERBATIM_RECALL_TEMPLATE.format(question=state.user_prompt.text)
        return state

    return solve
