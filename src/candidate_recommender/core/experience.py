"""
Years of experience: stated ("8 years of experience") and computed from
employment date ranges.

Most resumes never write "N years of experience" — they list roles with date
ranges ("Mar 2021 – Present", "2017 – 2020"). This module finds those ranges,
skips lines that look like education, merges overlapping roles so concurrent
jobs aren't double-counted, and returns the total in years.
"""

from __future__ import annotations

import re
from datetime import date

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_RANGE = re.compile(
    rf"(?:(?P<m1>{_MONTH})\s+)?(?P<y1>(?:19|20)\d{{2}})"
    r"\s*(?:-|–|—|to)\s*"
    rf"(?:(?:(?P<m2>{_MONTH})\s+)?(?P<y2>(?:19|20)\d{{2}})|(?P<present>present|current|now|today))",
    re.IGNORECASE,
)

# Lines naming a degree or school are education, not work experience.
# Case-sensitive so "ms" or "ma" inside ordinary words don't match.
_EDUCATION = re.compile(
    r"\b(?:Ph\.?\s?D|B\.\s?S|M\.\s?S|B\.\s?A|M\.\s?A|B\.\s?Eng|M\.\s?Eng|MBA|"
    r"Bachelor|Master|Doctorate|University|College|Bootcamp|BOOTCAMP|EDUCATION|Education)\b"
)

# "5 years", "5+ yrs", "3-5 years", "3 to 5 years". Group 1 is the lower bound,
# group 2 the optional upper bound of a range.
_YEARS_PATTERN = re.compile(
    r"(\d{1,2})\+?(?:\s*(?:-|–|—|to)\s*(\d{1,2})\+?)?\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)
_MAX_PLAUSIBLE_YEARS = 50  # ignores "100 years of history" style numbers

# Year-only dates ("2019 – 2021") are counted from mid-year, the expected value
# when the month is unknown, rather than overstating tenure with Jan–Dec.
_UNKNOWN_MONTH = 7


def _month_index(year: str, month: str | None) -> int:
    m = _MONTHS[month[:3].lower()] if month else _UNKNOWN_MONTH
    return int(year) * 12 + (m - 1)


def employment_intervals(text: str, today: date | None = None) -> list[tuple[int, int]]:
    """(start, end) month indices for every work date range found in text."""
    today = today or date.today()
    now = today.year * 12 + (today.month - 1)
    intervals = []
    for line in text.splitlines() or [text]:
        if _EDUCATION.search(line):
            continue
        for m in _RANGE.finditer(line):
            start = _month_index(m["y1"], m["m1"])
            end = now if m["present"] else _month_index(m["y2"], m["m2"])
            if start <= end <= now:
                intervals.append((start, end))
    return intervals


def years_from_dates(text: str, today: date | None = None) -> float:
    """Total employment in years, with overlapping ranges merged."""
    intervals = sorted(employment_intervals(text, today))
    total = 0
    cur_start, cur_end = None, None
    for start, end in intervals:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total / 12


def stated_years(text: str, use_lower_bound: bool) -> int:
    """
    Largest plausible "N years" figure in the text. For ranges ("3-5 years")
    job descriptions use the lower bound as the requirement; resumes use the
    upper bound.
    """
    values = []
    for low, high in _YEARS_PATTERN.findall(text):
        value = int(low) if use_lower_bound or not high else int(high)
        if 0 < value <= _MAX_PLAUSIBLE_YEARS:
            values.append(value)
    return max(values, default=0)


def candidate_years(resume_text: str, today: date | None = None) -> float:
    """
    A candidate's years of experience: stated or computed from employment
    dates, whichever is larger (most resumes only give the dates). Pass raw
    text — cleaning joins lines and can turn "32% year over year" into
    "32 year".
    """
    return max(
        stated_years(resume_text, use_lower_bound=False), years_from_dates(resume_text, today)
    )
