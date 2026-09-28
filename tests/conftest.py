"""
Shared pytest setup: make both the repo root (for `backend.*`) and `src/`
(for `core.*` / `config`) importable, mirroring what the backend does.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
