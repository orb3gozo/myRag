# Project checks, used by /ronglu:implement. They run from backend/.

set working-directory := "backend"

test:
    uv run pytest -q

lint:
    uv run ruff check .
    uv run ruff format --check .

typecheck:
    uv run mypy src tests

complexity:
    uv run ruff check --select C90 .

fetch-corpus:
    ./scripts/fetch_corpus.sh

# Rebuilds the logo and banner SVG files in assets/.
brand:
    node ../assets/src/build.mjs
