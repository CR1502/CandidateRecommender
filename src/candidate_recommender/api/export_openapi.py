"""
Write the API's OpenAPI schema, from which the frontend's TypeScript types
are generated:

    uv run python -m candidate_recommender.api.export_openapi frontend/src/api/openapi.json
    cd frontend && npm run gen:api
"""

import json
import sys
from pathlib import Path

from candidate_recommender.api.main import app


def main() -> None:
    schema = json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(schema)
    else:
        sys.stdout.write(schema)


if __name__ == "__main__":
    main()
