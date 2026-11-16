"""LlamaIndex implementation of the RAG engine, storing chunks in LanceDB."""

import re
import time
from collections.abc import Sequence
from typing import Final

from lancedb.db import LanceDBConnection
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from myrag.config import Settings
from myrag.corpus import load_corpus
from myrag.engine.base import IngestReport, RagAnswer, Source

_SECTION_SEPARATOR: Final[str] = " > "

_DISALLOWED_TABLE_CHARACTERS: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_]")
# MarkdownNodeParser joins the parent headings into `header_path` with this separator.
# Headings are single lines, so a newline never occurs inside one, unlike in its default "/".
_HEADER_PATH_SEPARATOR: Final[str] = "\n"
_HEADING_LINE: Final[re.Pattern[str]] = re.compile(r"^#+\s(?P<title>.*)$")


class LlamaIndexEngine:
    """RAG engine built on LlamaIndex, with one LanceDB table per configuration."""

    def __init__(self, settings: Settings, *, embed_model: BaseEmbedding, llm: LLM) -> None:
        """Prepare the engine; nothing touches the corpus or LanceDB until a method runs."""
        self._settings = settings
        self._embed_model = embed_model
        self._llm = llm
        self._table_name = build_table_name(
            settings.embed_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def ingest(self) -> IngestReport:
        """Chunk, embed and store the corpus, replacing the configuration's table."""
        started_s = time.perf_counter()
        documents = load_corpus(self._settings.corpus_dir, docs_src_dir=self._settings.docs_src_dir)
        chunks = _split_into_chunks(
            documents,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        self._drop_table()
        storage_context = StorageContext.from_defaults(vector_store=self._open_store())
        VectorStoreIndex(chunks, storage_context=storage_context, embed_model=self._embed_model)
        return IngestReport(
            files=len(documents),
            chunks=len(chunks),
            table_name=self._table_name,
            seconds=time.perf_counter() - started_s,
        )

    def retrieve(self, question: str) -> list[Source]:
        """Return the `TOP_K` chunks closest to `question`, numbered from 1."""
        index = VectorStoreIndex.from_vector_store(
            self._open_store(), embed_model=self._embed_model
        )
        retriever = index.as_retriever(similarity_top_k=self._settings.top_k)
        return [
            _to_source(number, chunk)
            for number, chunk in enumerate(retriever.retrieve(question), start=1)
        ]

    def query(self, question: str) -> RagAnswer:
        """Not available yet; it raises `NotImplementedError` until #6."""
        # TODO(#6): answer from retrieve() with the RAG prompt and the LLM.
        raise NotImplementedError("query() llega con el issue #6.")

    def _drop_table(self) -> None:
        # LanceDBVectorStore opens an existing table and appends to it, even in its
        # default "overwrite" mode, so re-ingesting would duplicate every chunk.
        LanceDBConnection(self._settings.lancedb_path).drop_table(
            self._table_name, ignore_missing=True
        )

    def _open_store(self) -> LanceDBVectorStore:
        return LanceDBVectorStore(uri=str(self._settings.lancedb_path), table_name=self._table_name)


def build_table_name(embed_model: str, *, chunk_size: int, chunk_overlap: int) -> str:
    """Return the LanceDB table for one configuration, `{embed_model}__{chunk_size}_{overlap}`.

    Every character other than a letter, digit or underscore becomes an underscore.
    """
    raw_name = f"{embed_model}__{chunk_size}_{chunk_overlap}"
    return _DISALLOWED_TABLE_CHARACTERS.sub("_", raw_name)


def _split_into_chunks(
    documents: Sequence[Document], *, chunk_size: int, chunk_overlap: int
) -> list[BaseNode]:
    """Split pages on their headings, then cap each piece at `chunk_size` tokens.

    Each heading section gets a `section` metadata entry, which the SentenceSplitter
    copies to every chunk it cuts from that section.
    """
    heading_parser = MarkdownNodeParser(header_path_separator=_HEADER_PATH_SEPARATOR)
    sections = heading_parser.get_nodes_from_documents(documents)
    for section in sections:
        section.metadata["section"] = _heading_path(section)
        del section.metadata["header_path"]
    sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return sentence_splitter(sections)


def _heading_path(section: BaseNode) -> str:
    # `header_path` holds only the parent headings; the section's own heading is its first line.
    headings = [
        title for title in section.metadata["header_path"].split(_HEADER_PATH_SEPARATOR) if title
    ]
    first_line = section.get_content(metadata_mode=MetadataMode.NONE).partition("\n")[0]
    own_heading = _HEADING_LINE.match(first_line)
    if own_heading:
        headings.append(own_heading["title"].strip())
    return _SECTION_SEPARATOR.join(headings)


def _to_source(number: int, chunk: NodeWithScore) -> Source:
    return Source(
        number=number,
        source=chunk.node.metadata["source"],
        section=chunk.node.metadata["section"],
        score=chunk.get_score(),
        text=chunk.node.get_content(metadata_mode=MetadataMode.NONE),
    )
