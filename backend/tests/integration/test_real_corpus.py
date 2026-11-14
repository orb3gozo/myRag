import re
from typing import Final

import pytest
from llama_index.core import Document

from myrag.config import Settings
from myrag.corpus import load_corpus

pytestmark = pytest.mark.integration

INCLUDE_DIRECTIVE: Final[re.Pattern[str]] = re.compile(r"^\{\*.*\*\}$", re.M)


@pytest.fixture(scope="module")
def real_corpus() -> list[Document]:
    settings = Settings(_env_file=None)
    if not settings.corpus_dir.is_dir() or not settings.docs_src_dir.is_dir():
        pytest.skip("corpus not downloaded; run `just fetch-corpus`")
    return load_corpus(settings.corpus_dir, docs_src_dir=settings.docs_src_dir)


def test_real_corpus_has_123_pages(real_corpus: list[Document]) -> None:
    assert len(real_corpus) == 123


def test_real_corpus_has_no_heading_anchors(real_corpus: list[Document]) -> None:
    assert [doc.metadata["source"] for doc in real_corpus if "{ #" in doc.text] == []


def test_real_corpus_leaves_only_the_include_outside_docs_src(
    real_corpus: list[Document],
) -> None:
    unresolved = {
        doc.metadata["source"]: INCLUDE_DIRECTIVE.findall(doc.text)
        for doc in real_corpus
        if INCLUDE_DIRECTIVE.search(doc.text)
    }

    # This directive includes FastAPI's own source, which `just fetch-corpus` doesn't fetch.
    assert unresolved == {
        "how-to/configure-swagger-ui.md": ["{* ../../fastapi/openapi/docs.py ln[9:24] hl[18:24] *}"]
    }


def test_real_corpus_sources_are_paths_relative_to_corpus(real_corpus: list[Document]) -> None:
    sources = {doc.metadata["source"] for doc in real_corpus}

    assert {"tutorial/path-params.md", "index.md", "advanced/templates.md"} <= sources
