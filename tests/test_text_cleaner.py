"""
Unit tests for skill, contact, and name extraction in text_cleaner.
"""

import pytest

from candidate_recommender.core.text_cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


class TestSkillExtraction:
    @pytest.mark.parametrize(
        "text, skill",
        [
            ("Senior C++ developer", "C++"),
            ("Worked in C# and .NET", "C#"),
            ("Languages: C++, Python", "C++"),
            ("Skills: Python, Go, Rust", "Go"),
            ("Built services in Golang", "Go"),
            ("Go developer at Acme", "Go"),
            ("Skills: Python, Rust", "Rust"),
            ("Stats in R, Python and SQL", "R"),
            ("Node.js, Express, MongoDB", "Express"),
            ("Spring Boot microservices", "Spring"),
            ("Designed RESTful services", "REST APIs"),
            ("Exposed a REST API", "REST APIs"),
            ("Fine-tuned LLMs", "LLMs"),
            ("Worked on ML pipelines", "Machine Learning"),
            ("OpenCV-based detection", "Computer Vision"),
            ("Deployed with Helm charts", "Helm"),
            ("Postgres and Redis", "PostgreSQL"),
        ],
    )
    def test_detects_skill(self, cleaner, text, skill):
        assert skill in cleaner.extract_key_skills(text)

    def test_ordinary_english_is_not_a_skill(self, cleaner):
        prose = (
            "Please send your CV. We go the extra mile, the rest is history. "
            "R&D team, Spring 2023, Shell Oil, swift turnaround, express delivery, "
            "at the helm, guard rails, rust stains."
        )
        assert cleaner.extract_key_skills(prose) == []

    def test_no_cap_on_matching(self, cleaner):
        text = (
            "Python, Java, Kotlin, Scala, Ruby, PHP, JavaScript, TypeScript, HTML, "
            "CSS, Bash, React, Vue, Angular, Svelte, Docker, Kubernetes, Terraform"
        )
        skills = cleaner.extract_key_skills(text)
        assert len(skills) > 15
        assert {"Docker", "Kubernetes", "Terraform"} <= set(skills)

    def test_limit_is_display_only(self, cleaner):
        text = "Python, Java, Docker, Kubernetes"
        assert cleaner.extract_key_skills(text, limit=2) == ["Python", "Java"]

    def test_skills_survive_cleaning(self, cleaner):
        cleaned = cleaner.prepare_for_embedding("Skills: C++ • C# • Go, Rust")
        assert {"C++", "C#", "Go", "Rust"} <= set(cleaner.extract_key_skills(cleaned))


class TestContactExtraction:
    def test_extracts_core_fields(self, cleaner):
        text = (
            "Jane Doe\n"
            "jane.doe@example.com | (555) 234-5678\n"
            "linkedin.com/in/janedoe | github.com/janedoe\n"
            "Boston, MA\n"
        )
        contact = cleaner.extract_contact_details(text)
        assert contact["email"] == "jane.doe@example.com"
        assert contact["phone"] == "(555) 234-5678"
        assert contact["linkedin"] == "linkedin.com/in/janedoe"
        assert contact["github"] == "github.com/janedoe"
        assert contact["location"] == "Boston, MA"

    def test_prefers_personal_email(self, cleaner):
        text = "noreply@corp.com and jane@example.com"
        assert cleaner.extract_contact_details(text)["email"] == "jane@example.com"


class TestNameExtraction:
    def test_from_filename(self, cleaner):
        assert cleaner.extract_candidate_name("", "john_smith_resume.pdf") == "John Smith"

    def test_from_first_lines(self, cleaner):
        assert cleaner.extract_candidate_name("Jane Doe\nEngineer", "cv.pdf") == "Jane Doe"
