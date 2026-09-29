"""
Minimal client for a local Ollama server (https://ollama.com).

- Availability is re-checked every `availability_ttl` seconds, so starting
  Ollama (or pulling the model) after the API is up takes effect without a
  restart.
- Model names are matched with their tag: "gemma4:12b" and
  "hf.co/org/repo:Q4_0" must match exactly, and a bare name like "llama3.2"
  means "llama3.2:latest", as it does for the Ollama CLI.
- Structured output: `generate_json` passes a JSON schema as Ollama's
  `format`, so the reply is guaranteed to parse, instead of scraping JSON out
  of prose with a regex.
- Thinking is turned off. Reasoning models (Gemma 4, Qwen3, ...) otherwise
  spend the whole token budget "thinking" and return an empty answer.
- Replies are cached in memory by request content, so re-ranking the same
  resumes against the same job doesn't re-run the model.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

import requests
from loguru import logger


class LLMError(RuntimeError):
    """The model could not produce a usable reply."""


def model_is_installed(configured: str, installed: list[str]) -> bool:
    """True if `configured` names one of Ollama's installed models (tag-aware)."""
    wanted = configured if ":" in configured.rsplit("/", 1)[-1] else f"{configured}:latest"
    return wanted in installed


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma4:12b",
        timeout: float = 180,
        num_ctx: int = 6144,
        availability_ttl: float = 30.0,
        cache_size: int = 256,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.availability_ttl = availability_ttl
        self._available: bool | None = None
        self._checked_at = 0.0
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Whether the server is reachable and the model is pulled (cached briefly)."""
        now = time.monotonic()
        with self._lock:
            if self._available is not None and now - self._checked_at < self.availability_ttl:
                return self._available
        available = self._check()
        with self._lock:
            if available != self._available:
                status = "available" if available else "unavailable"
                logger.info(f"Ollama model '{self.model}' is {status}")
            self._available, self._checked_at = available, now
        return available

    def _check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            resp.raise_for_status()
            installed = [m["name"] for m in resp.json().get("models", [])]
        except Exception as e:
            logger.debug(f"Ollama not reachable at {self.base_url}: {e}")
            return False
        if not model_is_installed(self.model, installed):
            logger.warning(
                f"Ollama is running but model '{self.model}' is not pulled "
                f"(installed: {', '.join(installed) or 'none'}). Run: ollama pull {self.model}"
            )
            return False
        return True

    def mark_unavailable(self) -> None:
        """Force the next is_available() to re-check (e.g. after a connection error)."""
        with self._lock:
            self._available, self._checked_at = False, time.monotonic()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: str | None = None,
        num_predict: int = 512,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Generate a reply constrained to `schema` and return it parsed."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                "num_predict": num_predict,
            },
        }
        if system:
            payload["system"] = system

        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        data = self._post(payload)
        if data.get("done_reason") == "length":
            raise LLMError(f"reply truncated at num_predict={num_predict}")
        try:
            result = json.loads(data.get("response", ""))
        except json.JSONDecodeError as e:
            raise LLMError(f"reply was not valid JSON: {e}") from e
        if not isinstance(result, dict):
            raise LLMError("reply was not a JSON object")

        with self._lock:
            self._cache[key] = result
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 400 and "think" in resp.text.lower():
                # Models without a thinking mode reject the flag; retry without it.
                payload = {k: v for k, v in payload.items() if k != "think"}
                resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError as e:
            self.mark_unavailable()
            raise LLMError(f"Ollama not reachable: {e}") from e
        except requests.RequestException as e:
            raise LLMError(f"Ollama request failed: {e}") from e
