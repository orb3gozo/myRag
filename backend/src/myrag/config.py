"""Application settings read from the environment and the repository's `.env` file."""

from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Final

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives in backend/src/myrag/, three levels below the repository root.
REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


class RagEngine(StrEnum):
    """RAG framework that builds the index and answers queries."""

    LLAMAINDEX = "llamaindex"
    LANGCHAIN = "langchain"


class EmbedProvider(StrEnum):
    """Backend that computes embeddings."""

    FASTEMBED = "fastembed"
    OPENAI = "openai"


class Settings(BaseSettings):
    """Every configurable value, with the defaults from PLAN.md section 3.11.

    Environment variables take precedence over `.env`. Relative paths resolve against
    the repository root, so the result does not depend on the working directory.
    """

    model_config = SettingsConfigDict(env_file=REPOSITORY_ROOT / ".env", extra="ignore")

    rag_engine: RagEngine = RagEngine.LLAMAINDEX
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "qwen3-1.7b"
    llm_api_key: str = "lm-studio"
    llm_temperature: float = 0.1
    llm_no_think: bool = True
    embed_provider: EmbedProvider = EmbedProvider.FASTEMBED
    embed_model: str = "intfloat/multilingual-e5-small"
    embed_base_url: str = "http://localhost:1234/v1"
    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k: int = 4
    lancedb_path: Path = Path("backend/storage/lancedb")
    corpus_dir: Path = Path("data/fastapi-es")

    @field_validator("lancedb_path", "corpus_dir", mode="after")
    @classmethod
    def _resolve_against_repository_root(cls, path: Path) -> Path:
        return REPOSITORY_ROOT / path


@cache
def get_settings() -> Settings:
    """Return the process-wide settings, read once on first call."""
    return Settings()
