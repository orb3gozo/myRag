"""The contract every RAG engine meets, so the CLI, API and evaluation work with any of them."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Source:
    """A retrieved chunk, numbered as the answer cites it.

    Attributes:
        number: Citation number, starting at 1 in retrieval order.
        source: File path of the document, relative to the corpus directory.
        section: Heading path of the chunk inside the document, such as `Título > Apartado`.
        score: Similarity to the question; higher means closer.
        text: The chunk content.
    """

    number: int
    source: str
    section: str
    score: float
    text: str


@dataclass(frozen=True)
class RagAnswer:
    """A generated answer and the sources it may cite."""

    answer: str
    sources: list[Source]


@dataclass(frozen=True)
class IngestReport:
    """What one ingestion run stored.

    Attributes:
        files: Corpus pages loaded.
        chunks: Chunks embedded and stored.
        table_name: Vector store table that now holds the chunks.
        seconds: Wall-clock duration of the run.
    """

    files: int
    chunks: int
    table_name: str
    seconds: float


class RagEngine(Protocol):
    """Indexes the corpus and answers questions from it (PLAN.md section 3.2)."""

    def ingest(self) -> IngestReport:
        """Index the whole corpus, replacing what the configuration's table held before."""
        ...

    def retrieve(self, question: str) -> list[Source]:
        """Return the chunks closest to `question`, without calling the LLM."""
        ...

    def query(self, question: str) -> RagAnswer:
        """Answer `question` from the retrieved chunks, citing them by number."""
        ...
