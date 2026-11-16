from pathlib import Path
from typing import Final

import lancedb
import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

from myrag.config import Settings
from myrag.engine.llamaindex_engine import LlamaIndexEngine, build_table_name

# Each heading section is far below CHUNK_SIZE, so it becomes exactly one chunk.
FIXTURE_PAGES: Final[dict[str, str]] = {
    "tutorial/path-params.md": (
        "# Parámetros de path\n\nDeclara parámetros de path con la sintaxis de Python.\n\n"
        "## Parámetros de path con tipos\n\nDeclara el tipo con anotaciones.\n"
    ),
    "index.md": "# FastAPI\n\nFastAPI es un framework web moderno.\n",
}
FIXTURE_CHUNK_COUNT: Final[int] = 3


@pytest.mark.parametrize(
    ("embed_model", "expected"),
    [
        ("intfloat/multilingual-e5-small", "intfloat_multilingual_e5_small__400_50"),
        ("text-embedding-nomic-embed-text-v1.5", "text_embedding_nomic_embed_text_v1_5__400_50"),
        ("BAAI/bge-small-en", "BAAI_bge_small_en__400_50"),
    ],
)
def test_table_name_keeps_only_letters_digits_and_underscores(
    embed_model: str, expected: str
) -> None:
    assert build_table_name(embed_model, chunk_size=400, chunk_overlap=50) == expected


def make_engine(tmp_path: Path, *, top_k: int = 4) -> LlamaIndexEngine:
    corpus_dir = tmp_path / "corpus"
    for relative_path, text in FIXTURE_PAGES.items():
        page_path = corpus_dir / relative_path
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_text(text, encoding="utf-8")
    settings = Settings(
        _env_file=None,
        embed_model="intfloat/multilingual-e5-small",
        chunk_size=400,
        chunk_overlap=50,
        top_k=top_k,
        lancedb_path=tmp_path / "lancedb",
        corpus_dir=corpus_dir,
        docs_src_dir=tmp_path / "docs_src",
    )
    return LlamaIndexEngine(settings, embed_model=MockEmbedding(embed_dim=8), llm=MockLLM())


def test_ingest_reports_files_chunks_and_table(tmp_path: Path) -> None:
    report = make_engine(tmp_path).ingest()

    assert (report.files, report.chunks, report.table_name) == (
        len(FIXTURE_PAGES),
        FIXTURE_CHUNK_COUNT,
        "intfloat_multilingual_e5_small__400_50",
    )


def test_ingest_stores_every_chunk_in_the_configuration_table(tmp_path: Path) -> None:
    make_engine(tmp_path).ingest()

    table = lancedb.connect(tmp_path / "lancedb").open_table(
        "intfloat_multilingual_e5_small__400_50"
    )
    assert table.count_rows() == FIXTURE_CHUNK_COUNT


def test_ingest_twice_replaces_the_table_instead_of_appending(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.ingest()

    engine.ingest()

    table = lancedb.connect(tmp_path / "lancedb").open_table(
        "intfloat_multilingual_e5_small__400_50"
    )
    assert table.count_rows() == FIXTURE_CHUNK_COUNT


def test_retrieve_returns_top_k_sources_numbered_from_one(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, top_k=2)
    engine.ingest()

    sources = engine.retrieve("¿Cómo declaro parámetros de path?")

    assert [source.number for source in sources] == [1, 2]


def test_retrieve_fills_source_path_and_heading_section(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, top_k=4)
    engine.ingest()

    sources = engine.retrieve("¿Cómo declaro parámetros de path?")

    assert {(source.source, source.section) for source in sources} == {
        ("tutorial/path-params.md", "Parámetros de path"),
        ("tutorial/path-params.md", "Parámetros de path > Parámetros de path con tipos"),
        ("index.md", "FastAPI"),
    }


def test_ingest_keeps_the_page_language_in_every_chunk(tmp_path: Path) -> None:
    make_engine(tmp_path).ingest()

    table = lancedb.connect(tmp_path / "lancedb").open_table(
        "intfloat_multilingual_e5_small__400_50"
    )
    languages = [row["lang"] for row in table.to_arrow().column("metadata").to_pylist()]
    assert languages == ["es"] * FIXTURE_CHUNK_COUNT
