import asyncio
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import pytest
from fastembed import TextEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from myrag.config import EmbedProvider, Settings
from myrag.embeddings import E5PrefixedEmbedding, build_embed_model

MINILM: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5_SMALL: str = "intfloat/multilingual-e5-small"


class FakeTextEmbedding:
    """Stands in for `fastembed.TextEmbedding` and records every text it is asked to embed."""

    received_texts: list[str]

    def __init__(self, model_name: str, **_: Any) -> None:
        self.model_name = model_name

    def embed(self, texts: Iterable[str], **_: Any) -> Iterator[np.ndarray]:
        for text in texts:
            self.received_texts.append(text)
            yield np.zeros(3)

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterator[np.ndarray]:
        yield from self.embed(texts, **kwargs)

    def query_embed(self, query: str, **kwargs: Any) -> Iterator[np.ndarray]:
        yield from self.embed([query], **kwargs)


@pytest.fixture(autouse=True)
def received_texts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    texts: list[str] = []
    fake_class = type("RecordingTextEmbedding", (FakeTextEmbedding,), {"received_texts": texts})
    monkeypatch.setattr("fastembed.TextEmbedding", fake_class)
    return texts


def build_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_fastembed_provider_returns_fastembed_model_with_configured_name() -> None:
    settings = build_settings(embed_provider=EmbedProvider.FASTEMBED, embed_model=MINILM)

    embed_model = build_embed_model(settings)

    assert isinstance(embed_model, FastEmbedEmbedding)
    assert embed_model.model_name == MINILM


def test_openai_provider_returns_openai_like_model_against_configured_server() -> None:
    settings = build_settings(
        embed_provider=EmbedProvider.OPENAI,
        embed_model="text-embedding-nomic-embed-text-v1.5",
        embed_base_url="http://embeddings.test:9000/v1",
    )

    embed_model = build_embed_model(settings)

    assert isinstance(embed_model, OpenAILikeEmbedding)
    assert embed_model.model_name == "text-embedding-nomic-embed-text-v1.5"
    assert embed_model.api_base == "http://embeddings.test:9000/v1"


def test_unknown_provider_raises_error_naming_it() -> None:
    # model_construct skips validation, the only way past the EmbedProvider enum.
    settings = Settings.model_construct(
        embed_provider="cohere",  # type: ignore[arg-type]
        embed_model=MINILM,
    )

    with pytest.raises(ValueError, match="cohere"):
        build_embed_model(settings)


def test_e5_model_prefixes_queries(received_texts: list[str]) -> None:
    embed_model = build_embed_model(build_settings(embed_model=E5_SMALL))

    embed_model.get_query_embedding("¿Cómo declaro un parámetro?")

    assert received_texts == ["query: ¿Cómo declaro un parámetro?"]


def test_e5_model_prefixes_single_document(received_texts: list[str]) -> None:
    embed_model = build_embed_model(build_settings(embed_model=E5_SMALL))

    embed_model.get_text_embedding("FastAPI usa type hints.")

    assert received_texts == ["passage: FastAPI usa type hints."]


def test_e5_model_prefixes_document_batch(received_texts: list[str]) -> None:
    embed_model = build_embed_model(build_settings(embed_model=E5_SMALL))

    embed_model.get_text_embedding_batch(["Primero.", "Segundo."])

    assert received_texts == ["passage: Primero.", "passage: Segundo."]


@pytest.mark.parametrize("model_name", [MINILM, "intfloat/multilingual-e5-large-instruct"])
def test_non_e5_model_leaves_texts_unprefixed(received_texts: list[str], model_name: str) -> None:
    embed_model = build_embed_model(build_settings(embed_model=model_name))

    embed_model.get_query_embedding("pregunta")
    embed_model.get_text_embedding("documento")

    assert received_texts == ["pregunta", "documento"]


def test_e5_model_prefixes_async_queries_and_documents(received_texts: list[str]) -> None:
    embed_model = build_embed_model(build_settings(embed_model=E5_SMALL))

    asyncio.run(embed_model.aget_query_embedding("pregunta"))
    asyncio.run(embed_model.aget_text_embedding_batch(["documento"]))

    assert received_texts == ["query: pregunta", "passage: documento"]


def test_e5_model_keeps_configured_name_on_openai_provider() -> None:
    settings = build_settings(embed_provider=EmbedProvider.OPENAI, embed_model=E5_SMALL)

    embed_model = build_embed_model(settings)

    assert isinstance(embed_model, E5PrefixedEmbedding)
    assert embed_model.model_name == E5_SMALL


def test_fastembed_e5_small_is_registered_with_384_dimensions() -> None:
    build_embed_model(build_settings(embed_model=E5_SMALL))

    registered = {model["model"]: model["dim"] for model in TextEmbedding.list_supported_models()}
    assert registered[E5_SMALL] == 384


def test_building_e5_small_twice_does_not_fail() -> None:
    settings = build_settings(embed_model=E5_SMALL)
    build_embed_model(settings)

    embed_model = build_embed_model(settings)

    assert embed_model.model_name == E5_SMALL


def test_lm_studio_e5_model_name_is_prefixed() -> None:
    settings = build_settings(
        embed_provider=EmbedProvider.OPENAI, embed_model="text-embedding-multilingual-e5-small"
    )

    assert isinstance(build_embed_model(settings), E5PrefixedEmbedding)
