import pytest

from myrag.config import RagEngine, Settings
from myrag.engine.factory import build_engine


def test_build_engine_rejects_an_engine_that_is_not_implemented_yet() -> None:
    settings = Settings(_env_file=None, rag_engine=RagEngine.LANGCHAIN)

    with pytest.raises(ValueError, match=r"RAG_ENGINE.*'langchain'.*llamaindex"):
        build_engine(settings)
