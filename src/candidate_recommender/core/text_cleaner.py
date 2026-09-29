"""
Text cleaning and preprocessing utilities.
"""

import hashlib
import re

from loguru import logger

# Explicit skill registry: (regex_pattern, display_name)
# Patterns are matched case-insensitively against the original text. Words that
# are also ordinary English ("go", "rust", "spring", "rest", ...) use (?-i:...)
# to require their usual capitalisation and/or list context, so prose like
# "we go the extra mile" or "Spring 2023" doesn't register as a skill.
# fmt: off
_LIST_BEFORE = r'(?:(?<=[,/;(:])\s*)'   # preceded by a list separator
_LIST_AFTER = r'(?=\s*[,/;)])'          # followed by a list separator

SKILL_REGISTRY = [
    # Languages
    (r'(?<![\w+])c\+\+(?![\w+])',   'C++'),
    (r'(?<!\w)c#(?!\w)',             'C#'),
    (r'(?-i:\bRust\b)',              'Rust'),
    (r'\bgolang\b|\bgo\s+lang\b'
     rf'|{_LIST_BEFORE}(?-i:Go)\b|(?-i:\bGo\b){_LIST_AFTER}'
     r'|(?-i:\bGo\b)(?=\s+(?:programming|developer|engineer|microservices)\b)', 'Go'),
    (r'\bpython\b',            'Python'),
    (r'\bjava\b',              'Java'),
    (r'\bkotlin\b',            'Kotlin'),
    (r'\bscala\b',             'Scala'),
    (r'(?-i:\bSwift\b)|\bswiftui\b', 'Swift'),
    (r'\bruby\b',              'Ruby'),
    (r'\bphp\b',               'PHP'),
    (rf'{_LIST_BEFORE}(?-i:R)\b(?![&\w\'])|(?-i:\bR\b)(?![&]){_LIST_AFTER}'
     r'|(?-i:\bR\b)(?=\s+(?:programming|language)\b)|\brstudio\b', 'R'),
    (r'\bjavascript\b|\bjs\b', 'JavaScript'),
    (r'\btypescript\b',        'TypeScript'),
    (r'\bhtml5?\b',            'HTML'),
    (r'\bcss3?\b',             'CSS'),
    (r'\bbash\b|\bzsh\b|\bshell\s+script(?:s|ing)?\b', 'Shell/Bash'),

    # Frontend
    (r'\breact\.?js\b|\breact\b', 'React'),
    (r'\bvue\.?js\b|\bvue\b',     'Vue'),
    (r'\bangular\b',              'Angular'),
    (r'\bnext\.?js\b',            'Next.js'),
    (r'\bsvelte\b',               'Svelte'),
    (r'\btailwind\b',             'Tailwind CSS'),

    # Backend frameworks
    (r'\bfastapi\b',         'FastAPI'),
    (r'\bdjango\b',          'Django'),
    (r'\bflask\b',           'Flask'),
    (r'\bnode\.?js\b',       'Node.js'),
    (rf'\bexpress\.?js\b|{_LIST_BEFORE}(?-i:Express)\b|(?-i:\bExpress\b){_LIST_AFTER}', 'Express'),
    (r'\bspring\s*(?:boot|framework|mvc|cloud|security|data)\b', 'Spring'),
    (r'\bruby\s+on\s+rails\b|(?-i:\bRails\b)', 'Rails'),
    (r'\blaravel\b',         'Laravel'),

    # Databases
    (r'\bpostgresql\b|\bpostgres\b', 'PostgreSQL'),
    (r'\bmysql\b',           'MySQL'),
    (r'\bsqlite\b',          'SQLite'),
    (r'\bmongodb\b',         'MongoDB'),
    (r'\bredis\b',           'Redis'),
    (r'\belasticsearch\b',   'Elasticsearch'),
    (r'\bcassandra\b',       'Cassandra'),
    (r'\bdynamodb\b',        'DynamoDB'),
    (r'\bsql\b',             'SQL'),
    (r'\bnosql\b',           'NoSQL'),

    # Cloud & DevOps
    (r'\baws\b|amazon web services',  'AWS'),
    (r'\bazure\b',           'Azure'),
    (r'\bgcp\b|google cloud', 'GCP'),
    (r'\bdocker\b',          'Docker'),
    (r'\bkubernetes\b|\bk8s\b', 'Kubernetes'),
    (r'\bterraform\b',       'Terraform'),
    (r'\bansible\b',         'Ansible'),
    (r'\bjenkins\b',         'Jenkins'),
    (r'\bgithub actions\b',  'GitHub Actions'),
    (r'\bci/cd\b|\bcontinuous integration\b', 'CI/CD'),
    (r'(?-i:\bHelm\b)|\bhelm\s+charts?\b', 'Helm'),

    # ML / AI
    (r'\btensorflow\b',      'TensorFlow'),
    (r'\bpytorch\b',         'PyTorch'),
    (r'\bscikit.learn\b|\bsklearn\b', 'scikit-learn'),
    (r'\bkeras\b',           'Keras'),
    (r'\bhugging face\b|\bhuggingface\b|\btransformers\b', 'HuggingFace'),
    (r'\blangchain\b',       'LangChain'),
    (r'\bllms?\b|large language models?', 'LLMs'),
    (r'\bmachine learning\b|(?-i:\bML\b)', 'Machine Learning'),
    (r'\bdeep learning\b',   'Deep Learning'),
    (r'\bnlp\b|natural language processing', 'NLP'),
    (r'\bcomputer vision\b|\bopencv\b', 'Computer Vision'),
    (r'\bmlops\b',           'MLOps'),
    (r'\bdata science\b',    'Data Science'),
    (r'\breinforcement learning\b', 'Reinforcement Learning'),

    # Data engineering
    (r'\bapache spark\b|\bpyspark\b|(?-i:\bSpark\b)', 'Spark'),
    (r'\bairflow\b',         'Airflow'),
    (r'\bkafka\b',           'Kafka'),
    (r'\bflink\b',           'Flink'),
    (r'\bdbt\b',             'dbt'),
    (r'\bsnowflake\b',       'Snowflake'),
    (r'\bbigquery\b',        'BigQuery'),
    (r'\bdatabricks\b',      'Databricks'),

    # General engineering
    (r'\brestful\b|\brest\s*apis?\b|(?-i:\bREST\b)', 'REST APIs'),
    (r'\bgraphql\b',         'GraphQL'),
    (r'\bgrpc\b',            'gRPC'),
    (r'\bmicroservices\b',   'Microservices'),
    (r'\bgit\b',             'Git'),
    (r'\blinux\b|\bunix\b',  'Linux'),
    (r'\bagile\b',           'Agile'),
    (r'\bscrum\b',           'Scrum'),

    # Python-specific
    (r'\bpandas\b',          'pandas'),
    (r'\bnumpy\b',           'NumPy'),
    (r'\bsqlalchemy\b',      'SQLAlchemy'),
    (r'\bcelery\b',          'Celery'),
    (r'\bpydantic\b',        'Pydantic'),
]
# fmt: on

_COMPILED_SKILLS = [(re.compile(pattern, re.IGNORECASE), name) for pattern, name in SKILL_REGISTRY]

# Job-description section headers after which skills are optional.
_NICE_TO_HAVE = re.compile(
    r"\b(?:nice[\s-]to[\s-]haves?|preferred\s+(?:qualifications|skills|experience)|bonus\s+points)\b",
    re.IGNORECASE,
)
NICE_TO_HAVE_WEIGHT = 0.5

# Resume lines that are a skills list rather than a description of work.
_SKILLS_HEADER = re.compile(
    r"^\s*(?:technical\s+)?(?:skills|technologies|tech\s+stack|tools|languages|frameworks|"
    r"core\s+competencies)\b",
    re.IGNORECASE,
)
LISTED_ONLY_CREDIT = 0.5


def _is_skill_list_line(line: str) -> bool:
    if _SKILLS_HEADER.match(line):
        return True
    items = [i.strip() for i in re.split(r"[,;|•·]", line) if i.strip()]
    return len(items) >= 5 and sum(len(i.split()) <= 3 for i in items) / len(items) >= 0.8


class TextCleaner:
    """Clean and preprocess text for embedding generation."""

    def __init__(self):
        self.min_length = 50
        self.max_length = 12000

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for embedding. Preserves characters that
        are meaningful in a technical context (C++, C#, @, /, +).

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        try:
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text)

            # Remove characters that are truly noise (control chars, zero-width etc.)
            # but keep: letters, digits, spaces, and common punctuation including
            # @, /, +, #, &, % which appear in skill names, contact info, and metrics
            text = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)\@\/\+\#\&\%]", "", text)

            # Collapse repeated punctuation (e.g. "..." → ".")
            text = re.sub(r"([.,;:!?])\1+", r"\1", text)

            text = text.strip()

            if len(text) > self.max_length:
                logger.warning(f"Text truncated from {len(text)} to {self.max_length} chars")
                text = text[: self.max_length]

            return text

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text

    def extract_candidate_name(self, text: str, filename: str | None = None) -> str:
        """
        Extract candidate name from resume text or filename.

        Priority: filename → first lines of text → hash fallback.
        """
        # Try filename first
        if filename:
            name = filename.rsplit(".", 1)[0]
            name = re.sub(r"[_\-]", " ", name)
            name = re.sub(r"resume|cv|curriculum|vitae", "", name, flags=re.IGNORECASE)
            name = name.strip()
            if len(name) > 2:
                return name.title()

        # Look in first 5 lines for a 2-4 word all-alpha sequence (likely a name)
        for line in text.split("\n")[:5]:
            line = line.strip()
            words = line.split()
            if 2 <= len(words) <= 4 and all(w.replace("-", "").isalpha() for w in words):
                return " ".join(words).title()

        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"Candidate_{text_hash}"

    def extract_key_skills(self, text: str, limit: int | None = None) -> list[str]:
        """
        Extract skills from text using the SKILL_REGISTRY.

        Returns every display-name skill found, in registry order. Pass
        `limit` only for display — scoring must see the full list, otherwise
        skill-rich resumes lose matches that sit later in the registry.
        """
        found = []

        for pattern, display_name in _COMPILED_SKILLS:
            if display_name not in found and pattern.search(text):
                found.append(display_name)

        return found[:limit] if limit else found

    def canonicalize_skill(self, name: str) -> str:
        """
        Map a free-form skill name (e.g. from the LLM) onto its SKILL_REGISTRY
        display name — "React.js" → "React", "Postgres" → "PostgreSQL",
        "k8s" → "Kubernetes" — so skills from different sources compare equal.
        When several registry skills match ("React.js" is also a JavaScript
        mention), the one whose match covers most of the name wins, provided
        it covers at least 60% of it; otherwise ("Docker and Kubernetes") the
        name is returned stripped, as it is when nothing matches.
        """
        name = name.strip()
        probe = f", {name},"  # list context, which some registry patterns require
        hits: dict[str, int] = {}
        for pattern, display in _COMPILED_SKILLS:
            m = pattern.search(probe)
            if m:
                hits[display] = max(hits.get(display, 0), len(m.group(0).strip(" ,")))
        if len(hits) == 1:
            return next(iter(hits))
        if hits:
            best, length = max(hits.items(), key=lambda kv: kv[1])
            if length >= 0.6 * len(name):
                return best
        return name

    def canonicalize_skills(self, names: list[str]) -> list[str]:
        """Canonicalise and de-duplicate (case-insensitively), keeping order."""
        seen: set[str] = set()
        result = []
        for name in names:
            canonical = self.canonicalize_skill(name)
            if canonical and canonical.lower() not in seen:
                seen.add(canonical.lower())
                result.append(canonical)
        return result

    def extract_required_skills(self, job_text: str) -> list[str]:
        """Skills a job description asks for, required or nice-to-have."""
        return list(self.job_skill_weights(job_text))

    def job_skill_weights(self, job_text: str) -> dict[str, float]:
        """
        Skills in a job description with an importance weight: 1.0 for
        skills in the main text, NICE_TO_HAVE_WEIGHT for skills that appear
        only after a "Nice to have" / "Preferred qualifications" header.
        Works on raw or cleaned (single-line) text.
        """
        marker = _NICE_TO_HAVE.search(job_text)
        core_text = job_text[: marker.start()] if marker else job_text
        nice_text = job_text[marker.end() :] if marker else ""

        weights = dict.fromkeys(self.extract_key_skills(core_text), 1.0)
        for skill in self.extract_key_skills(nice_text):
            weights.setdefault(skill, NICE_TO_HAVE_WEIGHT)
        return weights

    def skill_evidence(self, resume_text: str) -> dict[str, float]:
        """
        Credit per skill found in a resume: 1.0 when the skill appears in a
        description of work, LISTED_ONLY_CREDIT when it appears only in a
        skills list. A bare list of buzzwords shouldn't outscore someone who
        describes using the tools. Needs line breaks (raw text) to tell the
        two apart; single-line text gets full credit throughout.
        """
        evidence: dict[str, float] = {}
        for line in resume_text.splitlines() or [resume_text]:
            credit = LISTED_ONLY_CREDIT if _is_skill_list_line(line) else 1.0
            for skill in self.extract_key_skills(line):
                evidence[skill] = max(evidence.get(skill, 0.0), credit)
        return evidence

    def prepare_for_embedding(self, text: str) -> str:
        """Clean text for embedding generation."""
        text = self.clean_text(text)
        if len(text) < self.min_length:
            logger.warning(f"Short text ({len(text)} chars) may reduce embedding quality")
        return text

    def extract_contact_details(self, text: str) -> dict[str, str | None]:
        """
        Extract contact information from raw resume text.

        Always call this on the *original* (uncleaned) text so that
        characters like @ and / are still present.
        """
        contact: dict[str, str | None] = {
            "email": None,
            "phone": None,
            "linkedin": None,
            "github": None,
            "location": None,
            "website": None,
        }

        # Email
        emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b", text)
        if emails:
            noise = ("noreply", "support", "info@", "admin@", "no-reply")
            personal = [e for e in emails if not any(n in e.lower() for n in noise)]
            contact["email"] = personal[0] if personal else emails[0]

        # Phone (US + international, with optional extension)
        phone_patterns = [
            r"(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:x|ext\.?)\s*\d{1,6})?",
            r"\+?[1-9]\d{0,2}[-.\s]\d{2,4}[-.\s]\d{4,8}",
            r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
        ]
        for pattern in phone_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw = match.group(0).strip()
                digits = re.sub(r"\D", "", raw)
                if 10 <= len(digits) <= 15:
                    contact["phone"] = raw
                    break

        # LinkedIn
        for pattern in [
            r"linkedin\.com/in/([A-Za-z0-9\-_]+)",
            r"linkedin\.com/pub/([A-Za-z0-9\-_]+)",
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                contact["linkedin"] = f"linkedin.com/in/{m.group(1)}"
                break

        # GitHub
        m = re.search(r"github\.com/([A-Za-z0-9\-_]+)", text, re.IGNORECASE)
        if m:
            contact["github"] = f"github.com/{m.group(1)}"

        # Location — try labelled indicators first, then "City, ST" pattern
        for indicator in ["Location", "Address", "Based in", "Lives in", "Residing in"]:
            m = re.search(
                rf"{indicator}[\s:]*([A-Za-z][A-Za-z\s]+(?:,\s*[A-Za-z\s]{{2,}})?)",
                text,
                re.IGNORECASE,
            )
            if m:
                loc = m.group(1).strip()
                if len(loc) > 3 and "," in loc:
                    contact["location"] = loc
                    break

        if not contact["location"]:
            # US "City, ST" pattern
            m = re.search(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s*[A-Z]{2})\b", text)
            if m:
                contact["location"] = m.group(1)

        # Website / portfolio — match labelled URLs or bare domains
        m = re.search(
            r"(?:website|portfolio|personal site|www)[\s:]*(?:https?://)?([A-Za-z0-9\-]+\.[A-Za-z]{2,}(?:/[^\s]*)?)",
            text,
            re.IGNORECASE,
        )
        if m:
            contact["website"] = m.group(1)
        else:
            # Bare https:// URL that isn't LinkedIn/GitHub
            m = re.search(
                r"https?://(?!(?:www\.)?(linkedin|github))([A-Za-z0-9\-.]+\.[A-Za-z]{2,}(?:/[^\s]*)?)",
                text,
            )
            if m:
                contact["website"] = m.group(0)

        return contact
