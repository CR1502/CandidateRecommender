"""Shared test helpers."""

from __future__ import annotations

from typing import Any

import pytest

from candidate_recommender.core.llm import LLMError


class FakeLLM:
    """
    Stands in for OllamaClient: `available` toggles is_available(), `replies`
    are returned (or raised, if an exception) in order, and every prompt is
    recorded in `calls`.
    """

    def __init__(self, available: bool = True, replies: list[Any] | None = None):
        self.available = available
        self.replies = list(replies or [])
        self.calls: list[dict[str, Any]] = []
        self.model = "fake-model"

    def is_available(self) -> bool:
        return self.available

    def generate_json(self, prompt: str, schema: dict, **kwargs: Any) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema, **kwargs})
        if not self.replies:
            raise LLMError("no reply configured")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def offline_llm() -> FakeLLM:
    return FakeLLM(available=False)
