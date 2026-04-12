"""
Candidate profile enrichment by following links found in resumes.

Sources:
  GitHub  — public REST API (no auth needed): profile, repos, languages, stars
  Other   — generic HTTP fetch + BeautifulSoup text extraction (portfolio sites,
             project pages, personal blogs, etc.)
  LinkedIn — intentionally skipped; they block automated fetches reliably

All network calls run in a thread pool with per-call timeouts.
Enrichment failure NEVER blocks the pipeline — worst case returns "".
"""

from __future__ import annotations

import re
import concurrent.futures
from typing import Optional
from loguru import logger

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

_TIMEOUT = 8  # seconds per request

_GITHUB_HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "CandidateRecommender/1.0",
}

# Domains to skip entirely
_SKIP_DOMAINS = frozenset({
    "linkedin.com", "twitter.com", "x.com",
    "facebook.com", "instagram.com", "tiktok.com",
    "youtube.com", "medium.com",         # rate-limits / paywalls
})


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def extract_raw_urls(text: str) -> list[str]:
    """Return all http(s) URLs found in the text, deduplicated, order preserved."""
    pattern = r'https?://[^\s\)\]\>\"\'\,]+'
    seen: set[str] = set()
    result = []
    for url in re.findall(pattern, text):
        url = url.rstrip(".")
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _github_username(contact_github: str, raw_text: str) -> Optional[str]:
    """
    Resolve a GitHub username from the contact dict value or raw URL in text.
    Returns None if nothing found or the URL looks like an org/repo path.
    """
    sources = []
    if contact_github:
        sources.append(contact_github)
    sources.extend(extract_raw_urls(raw_text))

    for src in sources:
        m = re.search(r'github\.com/([A-Za-z0-9][A-Za-z0-9\-_]{0,38})', src, re.IGNORECASE)
        if not m:
            continue
        after = src.split("github.com/")[-1].strip("/")
        # Skip if it looks like github.com/user/repo (we want just the profile)
        if after.count("/") == 0:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# GitHub API
# ---------------------------------------------------------------------------

def fetch_github_info(username: str) -> str:
    """
    Fetch public GitHub profile + recent repos.
    Returns a formatted text block or "" on failure.
    """
    if not _HAS_REQUESTS:
        return ""

    try:
        pr = _requests.get(
            f"https://api.github.com/users/{username}",
            headers=_GITHUB_HEADERS, timeout=_TIMEOUT,
        )
        if pr.status_code == 404:
            return ""
        if pr.status_code != 200:
            logger.debug(f"GitHub /users/{username} → {pr.status_code}")
            return ""

        profile = pr.json()

        rr = _requests.get(
            f"https://api.github.com/users/{username}/repos"
            "?sort=updated&per_page=8",
            headers=_GITHUB_HEADERS, timeout=_TIMEOUT,
        )
        repos = rr.json() if rr.status_code == 200 and isinstance(rr.json(), list) else []

        lines: list[str] = [f"[GitHub: @{username}]"]
        for field, label in [("name", "Name"), ("bio", "Bio"), ("company", "Company"), ("location", "Location")]:
            if profile.get(field):
                lines.append(f"{label}: {profile[field]}")
        lines.append(
            f"Public repos: {profile.get('public_repos', 0)}  |  "
            f"Followers: {profile.get('followers', 0)}"
        )

        # Aggregate languages + build repo list
        lang_counts: dict[str, int] = {}
        repo_lines: list[str] = []
        for repo in repos[:6]:
            if not isinstance(repo, dict) or repo.get("fork"):
                continue  # skip forks — own projects are more signal
            name = repo.get("name", "")
            desc = (repo.get("description") or "")[:80]
            lang = repo.get("language") or ""
            stars = repo.get("stargazers_count", 0)
            if lang:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            parts = [f"  • {name}"]
            if lang:
                parts.append(f"[{lang}]")
            if desc:
                parts.append(f"— {desc}")
            if stars > 0:
                parts.append(f"★{stars}")
            repo_lines.append(" ".join(parts))

        if lang_counts:
            sorted_langs = sorted(lang_counts.items(), key=lambda x: -x[1])
            lines.append("Primary languages: " + ", ".join(l for l, _ in sorted_langs))
        if repo_lines:
            lines.append("Recent (non-fork) repositories:")
            lines.extend(repo_lines)

        return "\n".join(lines)

    except Exception as e:
        logger.debug(f"GitHub fetch failed for {username}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Generic webpage fetch
# ---------------------------------------------------------------------------

def fetch_webpage_text(url: str) -> str:
    """
    Fetch a webpage and return its main body text.
    Returns "" if the page is blocked, too short, or non-HTML.
    """
    if not _HAS_REQUESTS or not _HAS_BS4:
        return ""

    if any(d in url for d in _SKIP_DOMAINS):
        return ""

    try:
        r = _requests.get(
            url, timeout=_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CandidateRecommender/1.0)"},
            allow_redirects=True,
        )
        if r.status_code != 200:
            return ""
        if "text/html" not in r.headers.get("Content-Type", ""):
            return ""

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 120:
            return ""

        return f"[{url}]\n{text[:1500]}"

    except Exception as e:
        logger.debug(f"Webpage fetch failed for {url}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def enrich_candidate(raw_text: str, contact: dict) -> str:
    """
    Follow links found in a resume and return a combined enrichment block.

    Args:
        raw_text: The original (uncleaned) resume text.
        contact:  Contact dict from TextCleaner.extract_contact_details().

    Returns:
        Multi-line string with enriched context, or "" if nothing was found.
    """
    tasks: list[callable] = []

    # --- GitHub (highest signal) ---
    username = _github_username(contact.get("github", ""), raw_text)
    if username:
        tasks.append(lambda u=username: fetch_github_info(u))

    # --- Other URLs (portfolio, personal site, project pages) ---
    seen_domains: set[str] = set()
    for url in extract_raw_urls(raw_text)[:6]:
        if any(d in url for d in _SKIP_DOMAINS | {"github.com"}):
            continue
        # One fetch per domain
        m = re.search(r'https?://([^/]+)', url)
        dom = m.group(1) if m else url
        if dom in seen_domains:
            continue
        seen_domains.add(dom)
        tasks.append(lambda u=url: fetch_webpage_text(u))

    if not tasks:
        return ""

    results: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fn) for fn in tasks]
        for future in concurrent.futures.as_completed(futures, timeout=_TIMEOUT + 3):
            try:
                text = future.result()
                if text:
                    results.append(text)
            except Exception as e:
                logger.debug(f"Enrichment task failed: {e}")

    return "\n\n".join(results)
