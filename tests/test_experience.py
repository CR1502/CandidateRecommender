"""
Unit tests for years-of-experience from employment date ranges.
"""

from datetime import date

import pytest

from candidate_recommender.core.experience import years_from_dates

TODAY = date(2026, 9, 1)


@pytest.mark.parametrize(
    "text, years",
    [
        ("Engineer — Acme    Jan 2020 – Jan 2023", 3.0),
        ("Engineer — Acme    March 2020 - September 2021", 1.5),
        ("Engineer — Acme    Sept. 2019 to Mar 2020", 0.5),
        ("Engineer — Acme    2019 – 2021", 2.0),  # year-only: mid-year to mid-year
        ("Engineer — Acme    Sep 2023 – Present", 3.0),
        ("Engineer — Acme    Sep 2025 – current", 1.0),
    ],
)
def test_single_range(text, years):
    assert years_from_dates(text, TODAY) == pytest.approx(years)


def test_sums_separate_roles():
    text = "A — X    Jan 2015 – Jan 2017\nB — Y    Jan 2018 – Jan 2020"
    assert years_from_dates(text, TODAY) == pytest.approx(4.0)


def test_overlapping_roles_are_not_double_counted():
    text = "Full-time — X    Jan 2018 – Jan 2022\nConsulting — Y    Jan 2020 – Jan 2021"
    assert years_from_dates(text, TODAY) == pytest.approx(4.0)


@pytest.mark.parametrize(
    "line",
    [
        "Ph.D. Robotics, Carnegie Mellon University    2018 – 2023",
        "B.S. Computer Science    2014 – 2018",
        "Master of Science, Stanford    2019 – 2021",
        "Web Development Bootcamp    Jan 2024 – Apr 2024",
    ],
)
def test_education_lines_are_ignored(line):
    assert years_from_dates(line, TODAY) == 0.0


def test_future_and_inverted_ranges_are_ignored():
    assert years_from_dates("Offer — X    Jan 2027 – Present", TODAY) == 0.0
    assert years_from_dates("Oops — X    2022 – 2019", TODAY) == 0.0


def test_no_dates():
    assert years_from_dates("Experienced engineer who likes Python", TODAY) == 0.0
