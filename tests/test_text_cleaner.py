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


class TestJobSkillWeights:
    def test_nice_to_have_section_gets_lower_weight(self, cleaner):
        jd = "Requirements\n- Python and Docker\n\nNice to have\n- Redis, Kafka"
        assert cleaner.job_skill_weights(jd) == {
            "Python": 1.0,
            "Docker": 1.0,
            "Redis": 0.5,
            "Kafka": 0.5,
        }

    def test_skill_in_both_sections_keeps_full_weight(self, cleaner):
        jd = "Must know Python. Preferred qualifications: Python and Redis"
        assert cleaner.job_skill_weights(jd)["Python"] == 1.0

    def test_works_on_cleaned_single_line_text(self, cleaner):
        jd = cleaner.prepare_for_embedding("Requirements:\nPython\nBonus points:\nRedis")
        assert cleaner.job_skill_weights(jd) == {"Python": 1.0, "Redis": 0.5}


class TestSkillEvidence:
    def test_described_vs_listed(self, cleaner):
        resume = (
            "- Built payment APIs in Python with FastAPI\n"
            "SKILLS\n"
            "Python, FastAPI, Docker, Kubernetes, Terraform"
        )
        evidence = cleaner.skill_evidence(resume)
        assert evidence["Python"] == 1.0 and evidence["FastAPI"] == 1.0
        assert evidence["Docker"] == 0.5 and evidence["Terraform"] == 0.5

    def test_labelled_skill_lines_count_as_lists(self, cleaner):
        assert cleaner.skill_evidence("Languages: Python, Go")["Python"] == 0.5

    def test_prose_with_a_short_list_is_not_a_skills_list(self, cleaner):
        line = "- Moved services to Kubernetes on AWS with Docker and Terraform"
        assert cleaner.skill_evidence(line)["Kubernetes"] == 1.0


class TestCanonicalizeSkill:
    @pytest.mark.parametrize(
        "raw, canonical",
        [
            ("React.js", "React"),
            ("reactjs", "React"),
            ("Postgres", "PostgreSQL"),
            ("k8s", "Kubernetes"),
            ("Amazon Web Services", "AWS"),
            ("Go", "Go"),
            ("golang", "Go"),
            ("  Scikit-learn ", "scikit-learn"),
            ("Machine learning", "Machine Learning"),
        ],
    )
    def test_maps_aliases_to_registry_names(self, cleaner, raw, canonical):
        assert cleaner.canonicalize_skill(raw) == canonical

    def test_unknown_and_ambiguous_names_are_kept(self, cleaner):
        assert cleaner.canonicalize_skill(" Figma ") == "Figma"
        assert cleaner.canonicalize_skill("Docker and Kubernetes") == "Docker and Kubernetes"

    def test_canonicalize_skills_dedupes_case_insensitively(self, cleaner):
        names = ["React.js", "React", "figma", "Figma", "Postgres"]
        assert cleaner.canonicalize_skills(names) == ["React", "figma", "PostgreSQL"]


def test_clean_text_keeps_percent(cleaner):
    assert "32% year over year" in cleaner.clean_text("ROAS up 32% year over year")
