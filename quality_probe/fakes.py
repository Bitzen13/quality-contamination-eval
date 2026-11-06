"""Deterministic stand-ins for the model call (tests only).

A fake replaces only the generate step, so prompt formatting, scoring and analysis run
unchanged. Fakes refuse to run on anything but mockllm, so a fake's answers can never
be logged under a real model's name.
"""

from __future__ import annotations

import json
from random import Random
from typing import Callable

from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, multiple_choice, solver

from quality_probe.prompts import SENTINEL_END, SENTINEL_START

ANSWER_FAKES = ("oracle", "always_a", "random")
# "undersized" behaves like a server that cut the start of the prompt (context_check only).
CONTEXT_FAKES = (*ANSWER_FAKES, "undersized")

# Test hook, called with the sample ID on every fake generate.
on_call: Callable[[str], None] | None = None


def _answer(fake: str, state: TaskState) -> str:
    if fake == "oracle":
        return f"ANSWER: {state.target.text}"
    if fake == "always_a":
        return "ANSWER: A"
    letters = "ABCDE"[: len(state.choices)] or "ABCD"
    return f"ANSWER: {Random(f'fake:{state.sample_id}').choice(letters)}"


def _repeat_sentinels(fake: str, state: TaskState) -> str:
    visible = "\n".join(message.text for message in state.messages)
    if fake == "undersized":
        visible = visible[len(visible) // 2 :]
    return " ".join(word for word in (SENTINEL_START, SENTINEL_END) if word in visible)


def _fake_generate(respond: Callable[[TaskState], str]) -> Generate:
    async def generate(state: TaskState, tool_calls="loop", **config) -> TaskState:
        if not str(state.model).startswith("mockllm/"):
            raise RuntimeError(f"Fakes only run on mockllm, not {state.model}")
        if on_call is not None:
            on_call(str(state.sample_id))
        state.output = ModelOutput.from_content(str(state.model), respond(state))
        state.messages.append(state.output.message)
        return state

    return generate


@solver
def fake_multiple_choice(fake: str) -> Solver:
    if fake not in ANSWER_FAKES:
        raise ValueError(f"fake must be one of {ANSWER_FAKES}, got {fake!r}")
    inner = multiple_choice()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await inner(state, _fake_generate(lambda s: _answer(fake, s)))

    return solve


@solver
def fake_answer_generate(fake: str) -> Solver:
    """For conditions that build their own prompt instead of using multiple_choice()."""
    if fake not in ANSWER_FAKES:
        raise ValueError(f"fake must be one of {ANSWER_FAKES}, got {fake!r}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await _fake_generate(lambda s: _answer(fake, s))(state)

    return solve


def _recall_answer(fake: str, state: TaskState) -> str:
    if fake == "oracle":
        return f"OPTION: {state.target.text}"
    if fake == "always_a":
        return f"OPTION: {state.metadata['decoy']}"  # no memory, but plausible-looking
    return "OPTION: zzz qqq xxx"


@solver
def fake_recall_generate(fake: str) -> Solver:
    if fake not in ANSWER_FAKES:
        raise ValueError(f"fake must be one of {ANSWER_FAKES}, got {fake!r}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await _fake_generate(lambda s: _recall_answer(fake, s))(state)

    return solve


@solver
def fake_paraphrase_generate() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        reworded = [f"Reworded: {text}" for text in state.metadata["options"]]
        return await _fake_generate(lambda s: json.dumps(reworded))(state)

    return solve


@solver
def fake_context_generate(fake: str) -> Solver:
    if fake not in CONTEXT_FAKES:
        raise ValueError(f"fake must be one of {CONTEXT_FAKES}, got {fake!r}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await _fake_generate(lambda s: _repeat_sentinels(fake, s))(state)

    return solve
