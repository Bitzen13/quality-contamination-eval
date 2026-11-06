"""quality_scorer(): parse the answer line, set a status, map back to the original option."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    grouped,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from quality_probe.dataset import LETTERS, normalize_option
from quality_probe.prompts import SENTINEL_END, SENTINEL_START

OK = "ok"
PARSE_FAIL = "parse_fail"
ABSTAIN = "abstain"
REFUSED = "refused"
TRUNCATED_OUTPUT = "truncated_output"
STATUSES = (OK, PARSE_FAIL, ABSTAIN, REFUSED, TRUNCATED_OUTPUT)

# Filler allowed around an answer line. Differs from \W in two ways: it can't cross a
# newline (the line must consist only of the answer) and it covers _emphasis_.
_FILLER = r"(?:[^\w\n]|_)*"
_ANSWER_LINE = re.compile(rf"(?im)^{_FILLER}answer{_FILLER}:{_FILLER}([a-z]){_FILLER}$")


_OPTION_LINE = re.compile(rf"(?im)^{_FILLER}option{_FILLER}:{_FILLER}(\S.*?)\s*$")
NEAR_MATCH = 0.8


def parse_option(completion: str) -> str | None:
    """The text on the last `OPTION: ...` line, or None."""
    matches = _OPTION_LINE.findall(completion)
    return matches[-1] if matches else None


def text_similarity(a: str, b: str) -> float:
    """1.0 for the same words, ignoring case and punctuation."""
    return SequenceMatcher(None, normalize_option(a), normalize_option(b)).ratio()


def parse_answer(completion: str, offered: str) -> str | None:
    """Return the letter on the last answer line, or None.

    The last answer line decides. If its letter isn't offered the result is None: an
    earlier line the model went on to replace is not reinstated.
    """
    matches = _ANSWER_LINE.findall(completion)
    if not matches:
        return None
    letter = matches[-1].upper()
    return letter if letter in offered else None


def classify(
    completion: str, stop_reason: str | None, offered: str, abstain_letter: str | None
) -> tuple[str, str | None]:
    """Return (status, letter). Each completion gets exactly one status."""
    if stop_reason == "content_filter":
        return REFUSED, None
    letter = parse_answer(completion, offered)
    if letter is None:
        # model_length is Inspect's report of an exhausted context window
        cut_off = stop_reason in ("max_tokens", "model_length")
        return (TRUNCATED_OUTPUT if cut_off else PARSE_FAIL), None
    return (ABSTAIN if letter == abstain_letter else OK), letter


def _prompt_tokens(state: TaskState) -> int | None:
    usage = state.output.usage
    if usage is None:
        return None
    return (
        usage.input_tokens
        + (usage.input_tokens_cache_read or 0)
        + (usage.input_tokens_cache_write or 0)
    )


@scorer(metrics=[accuracy(), grouped(accuracy(), "difficult"), stderr(cluster="article_id")])
def quality_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        abstain_letter = "E" if state.metadata.get("abstain") else None
        offered = LETTERS + (abstain_letter or "")
        completion = state.output.completion
        stop_reason = state.output.stop_reason
        status, letter = classify(completion, stop_reason, offered, abstain_letter)

        perm = state.metadata.get("perm") or list(range(len(LETTERS)))
        gold_original = perm[LETTERS.index(target.text)]

        correct = False
        chosen_original = None
        landed_on_gold_original_letter = None
        if status == OK:
            chosen_index = LETTERS.index(letter)
            chosen_original = LETTERS[perm[chosen_index]]
            correct = perm[chosen_index] == gold_original
            if "perm" in state.metadata:
                landed_on_gold_original_letter = chosen_index == gold_original

        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=letter,
            explanation=completion,
            metadata={
                "status": status,
                "chosen_letter": letter,
                "chosen_original": chosen_original,
                "landed_on_gold_original_letter": landed_on_gold_original_letter,
                "stop_reason": stop_reason,
                "prompt_tokens": _prompt_tokens(state),
            },
        )

    return score


def sentinels_found(completion: str) -> tuple[bool, bool]:
    def found(word: str) -> bool:
        return re.search(rf"\b{word}\b", completion, re.IGNORECASE) is not None

    return found(SENTINEL_START), found(SENTINEL_END)


@scorer(metrics=[accuracy()])
def context_scorer() -> Scorer:
    """Correct only if the model repeats both sentinels, so the whole story arrived."""

    async def score(state: TaskState, target: Target) -> Score:
        start, end = sentinels_found(state.output.completion)
        return Score(
            value=CORRECT if start and end else INCORRECT,
            explanation=state.output.completion,
            metadata={
                "start_sentinel_found": start,
                "end_sentinel_found": end,
                "stop_reason": state.output.stop_reason,
                "prompt_tokens": _prompt_tokens(state),
            },
        )

    return score


@scorer(metrics=[accuracy()])
def verbatim_scorer() -> Scorer:
    """Correct only on an exact match. The decoy fields give a weak baseline."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        stop_reason = state.output.stop_reason
        recalled = None if stop_reason == "content_filter" else parse_option(completion)
        if stop_reason == "content_filter":
            status = REFUSED
        elif recalled is not None:
            status = OK
        elif stop_reason in ("max_tokens", "model_length"):
            status = TRUNCATED_OUTPUT
        else:
            status = PARSE_FAIL

        similarity = decoy_similarity = 0.0
        if recalled is not None:
            similarity = text_similarity(recalled, target.text)
            decoy_similarity = text_similarity(recalled, state.metadata["decoy"])
        exact = recalled is not None and normalize_option(recalled) == normalize_option(target.text)
        return Score(
            value=CORRECT if exact else INCORRECT,
            answer=recalled,
            explanation=completion,
            metadata={
                "status": status,
                "recalled": recalled,
                "exact": exact,
                "decoy_exact": recalled is not None
                and normalize_option(recalled) == normalize_option(state.metadata["decoy"]),
                "similarity": similarity,
                "decoy_similarity": decoy_similarity,
                "stop_reason": stop_reason,
                "prompt_tokens": _prompt_tokens(state),
            },
        )

    return score
