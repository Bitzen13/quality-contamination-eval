"""The parse rule, statuses and mapping back to the original option."""

from __future__ import annotations

import asyncio

import pytest
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ContentReasoning,
    ContentText,
    ModelName,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from quality_probe.scorer import classify, context_scorer, quality_scorer

ABCD = "ABCD"
ABCDE = "ABCDE"

# (completion, stop_reason, offered, expected status, expected letter)
CASES = [
    # Plain forms
    ("ANSWER: A", "stop", ABCD, "ok", "A"),
    ("ANSWER: D", "stop", ABCD, "ok", "D"),
    ("answer: b", "stop", ABCD, "ok", "B"),
    ("Answer:C", "stop", ABCD, "ok", "C"),
    ("  ANSWER:   B  ", "stop", ABCD, "ok", "B"),
    ("ANSWER: A\r\n", "stop", ABCD, "ok", "A"),
    # Markdown, parentheses, trailing period
    ("**ANSWER: C**", "stop", ABCD, "ok", "C"),
    ("**ANSWER:** C", "stop", ABCD, "ok", "C"),
    ("ANSWER: **C**", "stop", ABCD, "ok", "C"),
    ("*ANSWER: C*", "stop", ABCD, "ok", "C"),
    ("_ANSWER: C_", "stop", ABCD, "ok", "C"),
    ("`ANSWER: C`", "stop", ABCD, "ok", "C"),
    ("### ANSWER: C", "stop", ABCD, "ok", "C"),
    ("> ANSWER: C", "stop", ABCD, "ok", "C"),
    ("ANSWER: (B)", "stop", ABCD, "ok", "B"),
    ("ANSWER: B.", "stop", ABCD, "ok", "B"),
    ("ANSWER: (B).", "stop", ABCD, "ok", "B"),
    # The article "a", and a lowercase letter, on an answer line
    ("ANSWER: a", "stop", ABCD, "ok", "A"),
    # Prose around the answer line, and the last line winning
    ("Let me think.\nThe story says so.\n\nANSWER: D", "stop", ABCD, "ok", "D"),
    ("ANSWER: A\nOn reflection, no.\nANSWER: C", "stop", ABCD, "ok", "C"),
    ("ANSWER: B\nThat is my final choice.", "stop", ABCD, "ok", "B"),
    # Not answer lines
    ("ANSWER: A careful reading shows B", "stop", ABCD, "parse_fail", None),
    ("ANSWER: e.g. the second one", "stop", ABCD, "parse_fail", None),
    ("ANSWER: a story about loss", "stop", ABCD, "parse_fail", None),
    ("The answer is A", "stop", ABCD, "parse_fail", None),
    ("The answer: A", "stop", ABCD, "parse_fail", None),
    ("Final ANSWER: A", "stop", ABCD, "parse_fail", None),
    ("ANSWER A", "stop", ABCD, "parse_fail", None),
    ("ANSWER: A, B", "stop", ABCD, "parse_fail", None),
    ("ANSWER: AB", "stop", ABCD, "parse_fail", None),
    ("ANSWER: 1", "stop", ABCD, "parse_fail", None),
    ("ANSWER:", "stop", ABCD, "parse_fail", None),
    ("A", "stop", ABCD, "parse_fail", None),
    ("A) 7 years", "stop", ABCD, "parse_fail", None),
    # Echoed options are not answers
    ("A) 7 years\nB) 10 hours\nC) 12 years\nD) 1 hour", "stop", ABCD, "parse_fail", None),
    ("ANSWER: B) 10 hours", "stop", ABCD, "parse_fail", None),
    # The answer must be on one line
    ("ANSWER:\nA", "stop", ABCD, "parse_fail", None),
    ("ANSWER\n: A", "stop", ABCD, "parse_fail", None),
    # Letters not offered, including when an earlier line was valid
    ("ANSWER: E", "stop", ABCD, "parse_fail", None),
    ("ANSWER: Z", "stop", ABCD, "parse_fail", None),
    ("ANSWER: A\nANSWER: E", "stop", ABCD, "parse_fail", None),
    # Empty output
    ("", "stop", ABCD, "parse_fail", None),
    ("   \n  ", "stop", ABCD, "parse_fail", None),
    # Cut off
    ("", "max_tokens", ABCD, "truncated_output", None),
    ("Let me think about whether", "max_tokens", ABCD, "truncated_output", None),
    ("ANSWER:", "max_tokens", ABCD, "truncated_output", None),
    ("", "model_length", ABCD, "truncated_output", None),
    # A cut-off that still produced a valid line is an answer
    ("ANSWER: B\nBecause", "max_tokens", ABCD, "ok", "B"),
    # Refusals, whatever the text says
    ("I can't help with that.", "content_filter", ABCD, "refused", None),
    ("", "content_filter", ABCD, "refused", None),
    ("ANSWER: A", "content_filter", ABCD, "refused", None),
    # Abstention
    ("ANSWER: E", "stop", ABCDE, "abstain", "E"),
    ("ANSWER: e", "stop", ABCDE, "abstain", "E"),
    ("ANSWER: D", "stop", ABCDE, "ok", "D"),
    ("I don't know", "stop", ABCDE, "parse_fail", None),
]


def test_case_table_covers_at_least_thirty_completions():
    assert len(CASES) >= 30


@pytest.mark.parametrize("completion,stop_reason,offered,status,letter", CASES)
def test_classify(completion, stop_reason, offered, status, letter):
    abstain_letter = "E" if "E" in offered else None
    assert classify(completion, stop_reason, offered, abstain_letter) == (status, letter)


def make_state(output: ModelOutput, target: str, metadata: dict | None = None, choices=4):
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="q",
        epoch=1,
        input="question",
        messages=[],
        target=Target(target),
        choices=[f"option {i}" for i in range(choices)],
        output=output,
        metadata=metadata or {},
    )
    return state


def run_scorer(completion: str, target: str, metadata: dict | None = None, stop_reason="stop"):
    output = ModelOutput.from_content("mockllm/model", completion, stop_reason=stop_reason)
    state = make_state(output, target, metadata, choices=5 if (metadata or {}).get("abstain") else 4)
    return asyncio.run(quality_scorer()(state, Target(target)))


def test_correct_answer_without_perm():
    score = run_scorer("ANSWER: B", "B")
    assert score.value == "C"
    assert score.metadata["status"] == "ok"
    assert score.metadata["chosen_letter"] == "B"
    assert score.metadata["chosen_original"] == "B"
    assert score.metadata["landed_on_gold_original_letter"] is None


def test_wrong_answer_is_ok_status_but_incorrect():
    score = run_scorer("ANSWER: A", "B")
    assert score.value == "I"
    assert score.metadata["status"] == "ok"


@pytest.mark.parametrize(
    "completion,status",
    [("nope", "parse_fail"), ("ANSWER: E", "parse_fail")],
)
def test_non_answers_count_as_incorrect(completion, status):
    score = run_scorer(completion, "A")
    assert (score.value, score.metadata["status"]) == ("I", status)


def test_refused_and_truncated_count_as_incorrect():
    assert run_scorer("no", "A", stop_reason="content_filter").value == "I"
    assert run_scorer("", "A", stop_reason="max_tokens").value == "I"


def test_abstain_is_incorrect_with_its_own_status():
    score = run_scorer("ANSWER: E", "A", {"abstain": True})
    assert (score.value, score.metadata["status"]) == ("I", "abstain")


def test_choice_maps_through_perm():
    # Shown A,B,C,D are original C,A,D,B. The gold is original A, shown at B.
    metadata = {"perm": [2, 0, 3, 1]}

    right = run_scorer("ANSWER: B", "B", metadata)
    assert right.value == "C"
    assert right.metadata["chosen_original"] == "A"
    assert right.metadata["landed_on_gold_original_letter"] is False

    # Choosing shown A is choosing the gold's original letter, and it is wrong.
    on_letter = run_scorer("ANSWER: A", "B", metadata)
    assert on_letter.value == "I"
    assert on_letter.metadata["chosen_original"] == "C"
    assert on_letter.metadata["landed_on_gold_original_letter"] is True

    other = run_scorer("ANSWER: D", "B", metadata)
    assert other.metadata["landed_on_gold_original_letter"] is False


def test_identity_perm_lands_on_gold_letter_only_when_correct():
    metadata = {"perm": [0, 1, 2, 3]}
    assert run_scorer("ANSWER: C", "C", metadata).metadata["landed_on_gold_original_letter"] is True
    assert run_scorer("ANSWER: A", "C", metadata).metadata["landed_on_gold_original_letter"] is False


def test_metadata_has_stop_reason_and_total_prompt_tokens():
    output = ModelOutput.from_content("mockllm/model", "ANSWER: A")
    output.usage = ModelUsage(
        input_tokens=10,
        output_tokens=3,
        total_tokens=13,
        input_tokens_cache_read=100,
        input_tokens_cache_write=1000,
    )
    score = asyncio.run(quality_scorer()(make_state(output, "A"), Target("A")))
    assert score.metadata["prompt_tokens"] == 1110
    assert score.metadata["stop_reason"] == "stop"


def test_reasoning_text_is_never_parsed():
    message = ChatMessageAssistant(
        content=[
            ContentReasoning(reasoning="ANSWER: A\nActually, thinking more."),
            ContentText(text="ANSWER: C"),
        ]
    )
    output = ModelOutput(
        model="mockllm/model",
        choices=[ChatCompletionChoice(message=message, stop_reason="stop")],
    )
    assert output.completion == "ANSWER: C"
    assert asyncio.run(quality_scorer()(make_state(output, "C"), Target("C"))).value == "C"

    reasoning_only = ModelOutput(
        model="mockllm/model",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(content=[ContentReasoning(reasoning="ANSWER: A")]),
                stop_reason="max_tokens",
            )
        ],
    )
    score = asyncio.run(quality_scorer()(make_state(reasoning_only, "A"), Target("A")))
    assert score.metadata["status"] == "truncated_output"


@pytest.mark.parametrize(
    "completion,value",
    [
        ("VELLOXITE QUARMBLETON", "C"),
        ("velloxite and quarmbleton", "C"),
        ("**VELLOXITE**\n**QUARMBLETON**", "C"),
        ("VELLOXITE", "I"),
        ("QUARMBLETON", "I"),
        ("", "I"),
    ],
)
def test_context_scorer_needs_both_sentinels(completion, value):
    output = ModelOutput.from_content("mockllm/model", completion)
    state = make_state(output, "x")
    assert asyncio.run(context_scorer()(state, Target("x"))).value == value
