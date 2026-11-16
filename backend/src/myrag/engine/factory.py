"""Builds the RAG engine that `RAG_ENGINE` selects."""

from myrag import config
from myrag.embeddings import build_embed_model
from myrag.engine.base import RagEngine
from myrag.engine.llamaindex_engine import LlamaIndexEngine
from myrag.llm import build_llm


def build_engine(settings: config.Settings) -> RagEngine:
    """Return the configured engine, wired to the configured embedding model and LLM.

    Raises:
        ValueError: If `settings.rag_engine` names an engine without an implementation.
    """
    match settings.rag_engine:
        case config.RagEngine.LLAMAINDEX:
            return LlamaIndexEngine(
                settings, embed_model=build_embed_model(settings), llm=build_llm(settings)
            )
        # config.RagEngine also lists planned engines that have no implementation yet.
        case unsupported:
            raise ValueError(
                f"RAG_ENGINE no disponible todavía: {unsupported.value!r}. "
                f"Valores válidos: {config.RagEngine.LLAMAINDEX.value}."
            )
