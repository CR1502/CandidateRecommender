"""
Unit tests for the Ollama client. HTTP is mocked; no server needed.
"""

import json
from unittest.mock import Mock, patch

import pytest
import requests

from candidate_recommender.core.llm import LLMError, OllamaClient, model_is_installed

INSTALLED = ["gemma4:12b", "llama3.2:latest", "hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0"]


@pytest.mark.parametrize(
    "configured, expected",
    [
        ("gemma4:12b", True),
        ("gemma4", False),  # means gemma4:latest, which isn't installed
        ("gemma4:27b", False),
        ("llama3.2", True),  # bare name means :latest
        ("llama3.2:latest", True),
        ("hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0", True),
        ("hf.co/google/gemma-4-12B-it-qat-q4_0-gguf", False),
    ],
)
def test_model_is_installed_matches_tags(configured, expected):
    assert model_is_installed(configured, INSTALLED) is expected


def _tags_response(names):
    resp = Mock(status_code=200)
    resp.json.return_value = {"models": [{"name": n} for n in names]}
    resp.raise_for_status.return_value = None
    return resp


class TestAvailability:
    def test_cached_within_ttl_then_rechecked(self):
        client = OllamaClient(model="gemma4:12b", availability_ttl=60)
        with patch(
            "candidate_recommender.core.llm.requests.get", return_value=_tags_response(INSTALLED)
        ) as get:
            assert client.is_available() and client.is_available()
            assert get.call_count == 1
            client._checked_at -= 61  # TTL expired
            client.is_available()
            assert get.call_count == 2

    def test_model_not_pulled(self):
        client = OllamaClient(model="qwen3:8b")
        with patch(
            "candidate_recommender.core.llm.requests.get", return_value=_tags_response(INSTALLED)
        ):
            assert not client.is_available()

    def test_server_down(self):
        client = OllamaClient()
        with patch(
            "candidate_recommender.core.llm.requests.get", side_effect=requests.ConnectionError
        ):
            assert not client.is_available()


def _generate_response(payload: dict, done_reason="stop", status=200):
    resp = Mock(status_code=status, text="")
    resp.json.return_value = {"response": json.dumps(payload), "done_reason": done_reason}
    resp.raise_for_status.return_value = None
    return resp


class TestGenerateJson:
    SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    def test_request_shape_and_parse(self):
        client = OllamaClient(model="gemma4:12b", num_ctx=6144)
        with patch(
            "candidate_recommender.core.llm.requests.post",
            return_value=_generate_response({"ok": True}),
        ) as post:
            assert client.generate_json("hi", self.SCHEMA, system="sys", num_predict=100) == {
                "ok": True
            }
        body = post.call_args.kwargs["json"]
        assert body["think"] is False  # reasoning models would otherwise return nothing
        assert body["format"] == self.SCHEMA
        assert body["system"] == "sys"
        assert body["options"]["num_ctx"] == 6144 and body["options"]["num_predict"] == 100

    def test_identical_requests_are_cached(self):
        client = OllamaClient()
        with patch(
            "candidate_recommender.core.llm.requests.post",
            return_value=_generate_response({"ok": True}),
        ) as post:
            client.generate_json("same", self.SCHEMA)
            client.generate_json("same", self.SCHEMA)
            client.generate_json("different", self.SCHEMA)
        assert post.call_count == 2

    def test_truncated_reply_raises(self):
        client = OllamaClient()
        with patch(
            "candidate_recommender.core.llm.requests.post",
            return_value=_generate_response({}, "length"),
        ):
            with pytest.raises(LLMError, match="truncated"):
                client.generate_json("hi", self.SCHEMA)

    def test_invalid_json_raises(self):
        client = OllamaClient()
        resp = Mock(status_code=200, text="")
        resp.json.return_value = {"response": "not json", "done_reason": "stop"}
        with patch("candidate_recommender.core.llm.requests.post", return_value=resp):
            with pytest.raises(LLMError, match="not valid JSON"):
                client.generate_json("hi", self.SCHEMA)

    def test_retries_without_think_for_models_that_reject_it(self):
        client = OllamaClient()
        rejected = Mock(status_code=400, text='{"error":"model does not support thinking"}')
        ok = _generate_response({"ok": True})
        with patch(
            "candidate_recommender.core.llm.requests.post", side_effect=[rejected, ok]
        ) as post:
            assert client.generate_json("hi", self.SCHEMA) == {"ok": True}
        assert "think" not in post.call_args_list[1].kwargs["json"]

    def test_connection_error_marks_unavailable(self):
        client = OllamaClient()
        client._available, client._checked_at = True, 1e12  # "available", far from expiry
        with patch(
            "candidate_recommender.core.llm.requests.post",
            side_effect=requests.ConnectionError("down"),
        ):
            with pytest.raises(LLMError, match="not reachable"):
                client.generate_json("hi", self.SCHEMA)
        assert client._available is False
