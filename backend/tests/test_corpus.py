import logging
from pathlib import Path

import pytest

from myrag.corpus import load_corpus


def write_file(root: Path, relative_path: str, text: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_heading_anchor_is_stripped(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    write_file(corpus_dir, "page.md", "# Título\n\n## Validación de datos { #data-validation }\n")

    [document] = load_corpus(corpus_dir, docs_src_dir=tmp_path / "docs_src")

    assert document.text == "# Título\n\n## Validación de datos\n"


@pytest.mark.parametrize(
    "options",
    ["", " hl[6:7]", " ln[1:3] hl[6:7]", ' hl[1,3] title["app/routers/users.py"]'],
)
def test_include_directive_becomes_python_block_with_file_contents(
    tmp_path: Path, options: str
) -> None:
    corpus_dir = tmp_path / "corpus"
    docs_src_dir = tmp_path / "docs_src"
    write_file(docs_src_dir, "path_params/tutorial001.py", "app = FastAPI()\n")
    write_file(
        corpus_dir,
        "page.md",
        f"# Título\n\n{{* ../../docs_src/path_params/tutorial001.py{options} *}}\n\nFin.\n",
    )

    [document] = load_corpus(corpus_dir, docs_src_dir=docs_src_dir)

    assert document.text == "# Título\n\n```python\napp = FastAPI()\n```\n\nFin.\n"


@pytest.mark.parametrize(
    "directive",
    [
        "{* ../../docs_src/path_params/missing.py hl[6:7] *}",
        "{* ../../fastapi/openapi/docs.py ln[9:24] hl[18:24] *}",
    ],
)
def test_unresolved_include_is_kept_and_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, directive: str
) -> None:
    corpus_dir = tmp_path / "corpus"
    write_file(corpus_dir, "page.md", f"# Título\n\n{directive}\n")

    with caplog.at_level(logging.WARNING, logger="myrag.corpus"):
        [document] = load_corpus(corpus_dir, docs_src_dir=tmp_path / "docs_src")

    assert document.text == f"# Título\n\n{directive}\n"
    assert "page.md" in caplog.text


@pytest.mark.parametrize("skipped_name", ["_llm-test.md", "translation-banner.md"])
def test_auxiliary_page_is_skipped(tmp_path: Path, skipped_name: str) -> None:
    corpus_dir = tmp_path / "corpus"
    write_file(corpus_dir, "tutorial/page.md", "# Página\n")
    write_file(corpus_dir, f"tutorial/{skipped_name}", "# Auxiliar\n")

    documents = load_corpus(corpus_dir, docs_src_dir=tmp_path / "docs_src")

    assert [document.text for document in documents] == ["# Página\n"]


def test_metadata_records_source_title_and_language(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    write_file(
        corpus_dir,
        "tutorial/path-params.md",
        "---\nrender_macros: false\n---\n\n# Parámetros de Path { #path-parameters }\n\n## Otro\n",
    )

    [document] = load_corpus(corpus_dir, docs_src_dir=tmp_path / "docs_src")

    assert document.metadata == {
        "source": "tutorial/path-params.md",
        "title": "Parámetros de Path",
        "lang": "es",
    }


def test_title_is_empty_when_page_has_no_heading(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    write_file(corpus_dir, "page.md", "Solo texto.\n")

    [document] = load_corpus(corpus_dir, docs_src_dir=tmp_path / "docs_src")

    assert document.metadata["title"] == ""
