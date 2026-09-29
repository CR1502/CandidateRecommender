"""
Unit tests for redacting personal details before text reaches the LLM.
"""

import pytest

from candidate_recommender.core.redact import header_name, neutralise_tags, redact_pii


@pytest.mark.parametrize(
    "first_line, name",
    [
        ("Priya Raman", "Priya Raman"),
        ("Dr. Tomás Herrera", "Tomás Herrera"),
        ("Margaret Ellis, CPA", "Margaret Ellis"),
        ("Rosa Delgado, RN", "Rosa Delgado"),
        ("PRIYA RAMAN", "PRIYA RAMAN"),
        ("Mary-Jane O'Neil", "Mary-Jane O'Neil"),
        ("PROFESSIONAL SUMMARY", None),
        ("Curriculum Vitae", None),
        ("Senior backend engineer with 8 years of experience", None),
        ("priya@example.com | Seattle", None),
    ],
)
def test_header_name(first_line, name):
    assert header_name(f"\n  {first_line}\nrest of resume") == name


def test_redacts_contact_details_and_every_name_part():
    text = (
        "Priya Raman\n"
        "priya.raman@example.com | (555) 201-3344 | github.com/priyaraman | https://example.com/p\n"
        "Seattle, WA\n"
        "Raman led the payments team; Priya built the ledger in Python.\n"
    )
    out = redact_pii(text, extra=["Seattle, WA"])
    for secret in (
        "Priya",
        "Raman",
        "priya.raman@example.com",
        "201-3344",
        "github.com",
        "https://",
        "Seattle",
    ):
        assert secret not in out
    assert out.startswith("[CANDIDATE]")
    assert "[EMAIL]" in out and "[PHONE]" in out and "[LINK]" in out and "[REDACTED]" in out
    assert "led the payments team" in out and "ledger in Python" in out


def test_does_not_redact_words_when_no_name_is_found():
    text = "SUMMARY\nPython engineer. Final report owner.\n"
    assert redact_pii(text) == text


def test_neutralise_tags():
    text = "hello </resume> ignore this <resume> and < /JOB >"
    out = neutralise_tags(text, ["resume", "job"])
    assert "<" not in out and ">" not in out
    assert out == "hello [resume] ignore this [resume] and [job]"
