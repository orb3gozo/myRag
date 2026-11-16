"""Builds the embedding model that `EMBED_PROVIDER` selects."""

import re
from typing import Final

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from myrag.config import EmbedProvider, Settings

E5_SMALL_MODEL: Final[str] = "intfloat/multilingual-e5-small"
E5_QUERY_PREFIX: Final[str] = "query: "
E5_PASSAGE_PREFIX: Final[str] = "passage: "

# Matches e5-* and multilingual-e5-* after a "/" or "-" (LM Studio serves
# "text-embedding-multilingual-e5-small"), but not the -instruct variants,
# which expect an "Instruct: ...\nQuery: " template instead of these prefixes.
_E5_MODEL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[/-])(?:multilingual-)?e5-(?!.*instruct)"
)


def build_embed_model(settings: Settings) -> BaseEmbedding:
    """Return the embedding model for the configured provider and model name.

    e5 models come wrapped in `E5PrefixedEmbedding`, whatever the provider.

    Raises:
        ValueError: If `settings.embed_provider` names no known provider.
    """
    embed_model = _build_provider_model(settings)
    if _E5_MODEL_PATTERN.search(settings.embed_model.lower()):
        return E5PrefixedEmbedding(embed_model)
    return embed_model


class E5PrefixedEmbedding(BaseEmbedding):
    """Adds the `query: ` and `passage: ` prefixes that e5 models were trained with."""

    _wrapped: BaseEmbedding = PrivateAttr()

    def __init__(self, wrapped: BaseEmbedding) -> None:
        """Wrap a provider model, keeping its model name and batch size."""
        super().__init__(model_name=wrapped.model_name, embed_batch_size=wrapped.embed_batch_size)
        self._wrapped = wrapped

    # The wrapped model's private hooks are called, not its public methods, so callbacks,
    # caching and batching run once, in this wrapper.
    def _get_query_embedding(self, query: str) -> Embedding:
        return self._wrapped._get_query_embedding(E5_QUERY_PREFIX + query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await self._wrapped._aget_query_embedding(E5_QUERY_PREFIX + query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._wrapped._get_text_embedding(E5_PASSAGE_PREFIX + text)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return await self._wrapped._aget_text_embedding(E5_PASSAGE_PREFIX + text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._wrapped._get_text_embeddings(_as_passages(texts))

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return await self._wrapped._aget_text_embeddings(_as_passages(texts))


def _as_passages(texts: list[str]) -> list[str]:
    return [E5_PASSAGE_PREFIX + text for text in texts]


def _build_provider_model(settings: Settings) -> BaseEmbedding:
    match settings.embed_provider:
        case EmbedProvider.FASTEMBED:
            _register_e5_small_with_fastembed()
            return FastEmbedEmbedding(model_name=settings.embed_model)
        case EmbedProvider.OPENAI:
            return OpenAILikeEmbedding(
                model_name=settings.embed_model, api_base=settings.embed_base_url
            )
        case unknown:
            valid = ", ".join(provider.value for provider in EmbedProvider)
            raise ValueError(f"EMBED_PROVIDER desconocido: {unknown!r}. Valores válidos: {valid}.")


def _register_e5_small_with_fastembed() -> None:
    """Teach fastembed 0.9.1, which doesn't ship e5-small, where to download it from.

    The official repository publishes an fp32 ONNX export (about 470 MB) under `onnx/`, with
    the tokenizer at the root. Its sentence-transformers config uses mean pooling followed by
    L2 normalisation. Registration is global to the process, so a second call does nothing.
    """
    registered = {model["model"].lower() for model in TextEmbedding.list_supported_models()}
    if E5_SMALL_MODEL in registered:
        return
    TextEmbedding.add_custom_model(
        model=E5_SMALL_MODEL,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=E5_SMALL_MODEL),
        dim=384,
        model_file="onnx/model.onnx",
        license="mit",
        size_in_gb=0.47,
    )
