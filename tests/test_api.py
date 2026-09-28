"""
API tests for /rank, /extract and /health with the ML model mocked and
Ollama unavailable (template summaries, dictionary skills).
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from candidate_recommender.api.dependencies import get_embedding_engine, get_summarizer
from candidate_recommender.api.main import app
from candidate_recommender.core.embeddings import EmbeddingEngine
from candidate_recommender.core.summarizer import CandidateSummarizer

JOB = (
    "Senior Python developer. Requirements: 5+ years of Python, Docker, "
    "Kubernetes and REST APIs experience."
)


def _resume(name: str, body: str) -> bytes:
    return f"{name}\n{body}\n".encode() * 3  # comfortably above the 50-char minimum


@pytest.fixture
def client():
    model = Mock()
    model.to.return_value = model
    model.max_seq_length = 512
    model.get_sentence_embedding_dimension.return_value = 4

    def encode(texts, **kwargs):
        rng = np.random.default_rng(0)
        shape = (len(texts), 4) if isinstance(texts, list) else (4,)
        vecs = rng.random(shape)
        return vecs / np.linalg.norm(vecs, axis=-1, keepdims=True)

    model.encode.side_effect = encode

    with patch("candidate_recommender.core.embeddings.SentenceTransformer", return_value=model):
        engine = EmbeddingEngine("test-model")
    with patch.object(CandidateSummarizer, "_check_ollama", return_value=False):
        summarizer = CandidateSummarizer()

    app.dependency_overrides[get_embedding_engine] = lambda: engine
    app.dependency_overrides[get_summarizer] = lambda: summarizer
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["summary_mode"] == "template"


def test_rank_reports_all_processed_files_not_just_top_k(client):
    files = [
        ("files", (f"candidate_{i}.txt", _resume(f"Person {i}", "Python and Docker, 6 years."), "text/plain"))
        for i in range(3)
    ]
    resp = client.post("/api/rank", data={"job_description": JOB, "top_k": 2}, files=files)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_processed"] == 3
    assert len(body["candidates"]) == 2
    assert [c["rank"] for c in body["candidates"]] == [1, 2]
    assert "Python" in body["candidates"][0]["matching_skills"]


def test_rank_rejects_unsupported_extension(client):
    files = [("files", ("resume.exe", b"binary", "application/octet-stream"))]
    resp = client.post("/api/rank", data={"job_description": JOB}, files=files)
    assert resp.status_code == 415


def test_rank_with_only_unreadable_files_returns_422_detail(client):
    files = [("files", ("empty.txt", b"too short", "text/plain"))]
    resp = client.post("/api/rank", data={"job_description": JOB}, files=files)
    assert resp.status_code == 422
    assert "No valid resume text" in resp.json()["detail"]


def test_extract(client):
    content = b"Jane Doe\njane@example.com\nSkills: Python, Docker, Kubernetes\n"
    with patch("candidate_recommender.core.text_cleaner.TextCleaner.extract_skills_with_llm",
               lambda self, text, **kw: self.extract_key_skills(text)):
        resp = client.post("/api/extract", files={"file": ("jane_doe.txt", content, "text/plain")})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["contact"]["email"] == "jane@example.com"
    assert {"Python", "Docker", "Kubernetes"} <= set(body["skills"])


def test_rank_rejects_too_many_files(client):
    from candidate_recommender.config import get_settings

    limit = get_settings().max_files_per_upload
    files = [
        ("files", (f"r{i}.txt", _resume(f"Person {i}", "Python"), "text/plain"))
        for i in range(limit + 1)
    ]
    resp = client.post("/api/rank", data={"job_description": JOB}, files=files)
    assert resp.status_code == 413


def test_internal_errors_do_not_leak_details(client):
    with patch(
        "candidate_recommender.api.routers.rank.run_ranking_pipeline",
        side_effect=RuntimeError("secret internal path /opt/models"),
    ):
        files = [("files", ("a.txt", _resume("A", "Python"), "text/plain"))]
        resp = client.post("/api/rank", data={"job_description": JOB}, files=files)

    assert resp.status_code == 500
    assert "secret" not in resp.json()["detail"]


def test_spa_fallback_serves_index_for_client_routes(tmp_path):
    from fastapi import FastAPI

    from candidate_recommender.api.main import SPAStaticFiles

    (tmp_path / "index.html").write_text("<div id=root></div>")
    (tmp_path / "app.js").write_text("console.log(1)")
    spa = FastAPI()
    spa.mount("/", SPAStaticFiles(directory=tmp_path, html=True))
    web = TestClient(spa)

    assert web.get("/app.js").text == "console.log(1)"
    assert web.get("/results").text == "<div id=root></div>"  # client route
    assert web.get("/api/missing").status_code == 404  # API 404s stay 404s
    assert web.get("/assets/missing.js").status_code == 404  # missing files stay 404s
