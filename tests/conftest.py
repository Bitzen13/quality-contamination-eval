"""Shared fixtures. All tests run offline with no API keys."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "QuALITY.v1.0.1.htmlstripped.dev"


def pytest_sessionstart(session):
    # Fail loudly rather than skip: skipped data tests would look like a pass.
    if not DATA.exists():
        pytest.exit(
            "Missing dataset. Download it first (see README, Setup): " + str(DATA), returncode=2
        )
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(name, None)


@pytest.fixture(autouse=True)
def _run_from_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def make_question(qid: str, gold=1, options=None, difficult=0, **extra) -> dict:
    question = {
        "question": f"Question {qid}?",
        "question_unique_id": qid,
        "options": options or [f"{qid} one", f"{qid} two", f"{qid} three", f"{qid} four"],
        "difficult": difficult,
        **extra,
    }
    if gold is not None:
        question["gold_label"] = gold
    return question


def write_jsonl(path: Path, entries: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return path


def make_entry(article_id: str, questions: list[dict], article="A story.") -> dict:
    return {"article_id": article_id, "article": article, "questions": questions}
