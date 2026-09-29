"""
Redact personal details from resume text before it goes to the LLM.

The model doesn't need a candidate's name, contact details, or location to
judge fit, and seeing them invites bias (names signal gender and ethnicity;
locations signal much the same). Redacted text also keeps less personal data
in prompts and logs.

The name is taken from the resume's first lines rather than the uploaded
filename, since filenames like "final_v2.pdf" would otherwise redact
ordinary words.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_EMAIL = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)
_URL = re.compile(
    r"\b(?:https?://|www\.)\S+"  # explicit links
    r"|\b[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}/\S*"  # bare domain with a path: example.com/jane
    r"|\b[\w-]+\.(?:dev|me|io|app|design|site|page|xyz|codes|tech)\b",  # personal-site TLDs
    re.IGNORECASE,
)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)")
# Credentials and honorifics that can trail or lead a name line ("Jane Doe, RN")
_NAME_AFFIXES = re.compile(
    r",?\s*\b(?:Dr|Mr|Mrs|Ms|Mx|Prof|PhD|MD|RN|CPA|PMP|MBA|Jr|Sr|II|III)\b\.?", re.IGNORECASE
)
_NAME_WORD = re.compile(r"^[^\W\d_](?:[^\W\d_]|['’.-])*$")  # letters in any alphabet, plus ' - .

# Words that make a first line a heading, not a name ("PROFESSIONAL SUMMARY")
_HEADING_WORDS = {
    "resume", "résumé", "cv", "curriculum", "vitae", "summary", "profile", "objective",
    "professional", "experience", "education", "skills", "contact", "about", "career",
}  # fmt: skip

NAME_PLACEHOLDER = "[CANDIDATE]"


def header_name(text: str) -> str | None:
    """
    The candidate's name, if the first non-empty line looks like one: 2–4
    words, each Title Case (or the whole line in capitals), after dropping
    honorifics and credentials like "Dr." or ", RN".
    """
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    candidate = _NAME_AFFIXES.sub("", first).strip(" ,")
    words = candidate.split()
    if not 2 <= len(words) <= 4 or not all(_NAME_WORD.match(w) for w in words):
        return None
    if any(w.lower().strip(".") in _HEADING_WORDS for w in words):
        return None
    title_case = all(w[0].isupper() and not w.isupper() for w in words if len(w) > 1)
    if title_case or candidate.isupper():
        return candidate
    return None


def redact_pii(text: str, extra: Iterable[str | None] = ()) -> str:
    """
    Replace the candidate's name (every part of it, anywhere), emails, phone
    numbers, profile/web links, and any `extra` strings (e.g. a location from
    the contact details) with placeholders.
    """
    name = header_name(text)
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub("[LINK]", text)
    text = _PHONE.sub("[PHONE]", text)
    for value in extra:
        if value and len(value) > 2:
            text = re.sub(re.escape(value), "[REDACTED]", text, flags=re.IGNORECASE)
    if name:
        text = re.sub(rf"\b{re.escape(name)}\b", NAME_PLACEHOLDER, text)
        for part in (p.strip(".") for p in name.split()):
            if len(part) > 2:
                text = re.sub(rf"\b{re.escape(part)}\b", NAME_PLACEHOLDER, text)
    return text


def neutralise_tags(text: str, tags: Iterable[str]) -> str:
    """
    Stop untrusted text from closing the XML-style tags the prompt wraps it
    in (e.g. a resume containing "</resume> Ignore previous instructions").
    """
    for tag in tags:
        text = re.sub(rf"<\s*/?\s*{re.escape(tag)}\s*>", f"[{tag}]", text, flags=re.IGNORECASE)
    return text
