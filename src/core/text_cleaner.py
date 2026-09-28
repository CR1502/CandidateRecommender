"""
Text cleaning and preprocessing utilities.
"""

import re
import hashlib
from typing import Optional, List, Dict
from loguru import logger


# Explicit skill registry: (regex_pattern, display_name)
# Patterns are matched case-insensitively against the original text. Words that
# are also ordinary English ("go", "rust", "spring", "rest", ...) use (?-i:...)
# to require their usual capitalisation and/or list context, so prose like
# "we go the extra mile" or "Spring 2023" doesn't register as a skill.
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

_COMPILED_SKILLS = [
    (re.compile(pattern, re.IGNORECASE), name) for pattern, name in SKILL_REGISTRY
]


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
            text = re.sub(r'\s+', ' ', text)

            # Remove characters that are truly noise (control chars, zero-width etc.)
            # but keep: letters, digits, spaces, and common punctuation including
            # @, /, +, #, & which appear in skill names and contact info
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\@\/\+\#\&]', '', text)

            # Collapse repeated punctuation (e.g. "..." → ".")
            text = re.sub(r'([.,;:!?])\1+', r'\1', text)

            text = text.strip()

            if len(text) > self.max_length:
                logger.warning(f"Text truncated from {len(text)} to {self.max_length} chars")
                text = text[:self.max_length]

            return text

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text

    def extract_candidate_name(self, text: str, filename: Optional[str] = None) -> str:
        """
        Extract candidate name from resume text or filename.

        Priority: filename → first lines of text → hash fallback.
        """
        # Try filename first
        if filename:
            name = filename.rsplit('.', 1)[0]
            name = re.sub(r'[_\-]', ' ', name)
            name = re.sub(r'resume|cv|curriculum|vitae', '', name, flags=re.IGNORECASE)
            name = name.strip()
            if len(name) > 2:
                return name.title()

        # Look in first 5 lines for a 2-4 word all-alpha sequence (likely a name)
        for line in text.split('\n')[:5]:
            line = line.strip()
            words = line.split()
            if 2 <= len(words) <= 4 and all(w.replace('-', '').isalpha() for w in words):
                return ' '.join(words).title()

        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"Candidate_{text_hash}"

    def extract_key_skills(self, text: str, limit: Optional[int] = None) -> List[str]:
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

    def extract_skills_with_llm(
        self,
        text: str,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
    ) -> List[str]:
        """
        Use Ollama to extract skills from text.

        This catches technologies not in SKILL_REGISTRY (newer frameworks,
        domain-specific tools, niche libraries) and normalises naming
        (e.g. "Postgres" → "PostgreSQL", "k8s" → "Kubernetes").

        Falls back to dictionary extraction if Ollama is unavailable or returns
        unparseable output.
        """
        import json as _json

        try:
            import requests as _req

            prompt = (
                "List every technical skill in this text: programming languages, "
                "frameworks, libraries, databases, cloud services, DevOps tools, "
                "ML frameworks, and methodologies. Use common canonical names "
                "(e.g. 'PostgreSQL' not 'postgres', 'Kubernetes' not 'k8s'). "
                "Return ONLY a JSON array of short strings. No explanation.\n\n"
                f"Text:\n{text[:1800]}\n\nJSON array:"
            )

            r = _req.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.05, "num_predict": 300},
                },
                timeout=20,
            )
            if r.status_code == 200:
                raw = r.json().get("response", "").strip()
                # Ollama sometimes wraps the array in prose — extract just the []
                m = re.search(r"\[.*\]", raw, re.DOTALL)
                if m:
                    skills = _json.loads(m.group(0))
                    if isinstance(skills, list):
                        cleaned = [
                            str(s).strip()
                            for s in skills
                            if s and isinstance(s, str) and len(str(s)) < 60
                        ]
                        return cleaned[:25]
        except Exception as e:
            logger.debug(f"LLM skill extraction failed: {e}")

        # Fallback
        return self.extract_key_skills(text)

    def extract_required_skills(self, job_text: str) -> List[str]:
        """
        Extract skills from a job description that appear to be requirements.

        Looks for skills near requirement signal words (required, must, need, etc.)
        as well as bare skill mentions, since most JDs list them explicitly.
        """
        # For now this is the same as extract_key_skills — job descriptions tend
        # to list skills directly. A future improvement would weight skills that
        # appear near "required" / "must have" higher.
        return self.extract_key_skills(job_text)

    def prepare_for_embedding(self, text: str) -> str:
        """Clean text for embedding generation."""
        text = self.clean_text(text)
        if len(text) < self.min_length:
            logger.warning(f"Short text ({len(text)} chars) may reduce embedding quality")
        return text

    def extract_contact_details(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract contact information from raw resume text.

        Always call this on the *original* (uncleaned) text so that
        characters like @ and / are still present.
        """
        contact: Dict[str, Optional[str]] = {
            'email': None,
            'phone': None,
            'linkedin': None,
            'github': None,
            'location': None,
            'website': None,
        }

        # Email
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b', text)
        if emails:
            noise = ('noreply', 'support', 'info@', 'admin@', 'no-reply')
            personal = [e for e in emails if not any(n in e.lower() for n in noise)]
            contact['email'] = personal[0] if personal else emails[0]

        # Phone (US + international, with optional extension)
        phone_patterns = [
            r'(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:x|ext\.?)\s*\d{1,6})?',
            r'\+?[1-9]\d{0,2}[-.\s]\d{2,4}[-.\s]\d{4,8}',
            r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b',
        ]
        for pattern in phone_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw = match.group(0).strip()
                digits = re.sub(r'\D', '', raw)
                if 10 <= len(digits) <= 15:
                    contact['phone'] = raw
                    break

        # LinkedIn
        for pattern in [r'linkedin\.com/in/([A-Za-z0-9\-_]+)',
                         r'linkedin\.com/pub/([A-Za-z0-9\-_]+)']:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                contact['linkedin'] = f"linkedin.com/in/{m.group(1)}"
                break

        # GitHub
        m = re.search(r'github\.com/([A-Za-z0-9\-_]+)', text, re.IGNORECASE)
        if m:
            contact['github'] = f"github.com/{m.group(1)}"

        # Location — try labelled indicators first, then "City, ST" pattern
        for indicator in ['Location', 'Address', 'Based in', 'Lives in', 'Residing in']:
            m = re.search(
                rf'{indicator}[\s:]*([A-Za-z][A-Za-z\s]+(?:,\s*[A-Za-z\s]{{2,}})?)',
                text, re.IGNORECASE
            )
            if m:
                loc = m.group(1).strip()
                if len(loc) > 3 and ',' in loc:
                    contact['location'] = loc
                    break

        if not contact['location']:
            # US "City, ST" pattern
            m = re.search(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s*[A-Z]{2})\b', text)
            if m:
                contact['location'] = m.group(1)

        # Website / portfolio — match labelled URLs or bare domains
        m = re.search(
            r'(?:website|portfolio|personal site|www)[\s:]*(?:https?://)?([A-Za-z0-9\-]+\.[A-Za-z]{2,}(?:/[^\s]*)?)',
            text, re.IGNORECASE
        )
        if m:
            contact['website'] = m.group(1)
        else:
            # Bare https:// URL that isn't LinkedIn/GitHub
            m = re.search(r'https?://(?!(?:www\.)?(linkedin|github))([A-Za-z0-9\-.]+\.[A-Za-z]{2,}(?:/[^\s]*)?)', text)
            if m:
                contact['website'] = m.group(0)

        return contact
