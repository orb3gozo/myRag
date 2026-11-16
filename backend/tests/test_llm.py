import pytest
from llama_index.llms.openai_like import OpenAILike

from myrag.config import Settings
from myrag.llm import build_llm, strip_think_blocks


def test_build_llm_points_an_openai_like_chat_client_at_the_configured_server() -> None:
    settings = Settings(
        _env_file=None,
        llm_base_url="http://localhost:11434/v1",
        llm_model="qwen3-8b",
        llm_api_key="ollama",
        llm_temperature=0.7,
    )

    llm = build_llm(settings)

    assert isinstance(llm, OpenAILike)
    assert llm.api_base == "http://localhost:11434/v1"
    assert llm.model == "qwen3-8b"
    assert llm.api_key == "ollama"
    assert llm.temperature == 0.7
    assert llm.metadata.is_chat_model


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        pytest.param(
            "Path parameters go in the URL [1].",
            "Path parameters go in the URL [1].",
            id="no-block",
        ),
        pytest.param("<think></think>Hola [2].", "Hola [2].", id="empty-block"),
        pytest.param(
            "<think>The user asks about\npath params.</think>Van en la URL [1].",
            "Van en la URL [1].",
            id="filled-block",
        ),
        pytest.param(
            "  \n<think>\n\n</think>\n\nNo lo sé con la documentación disponible.\n",
            "No lo sé con la documentación disponible.",
            id="surrounding-whitespace",
        ),
    ],
)
def test_strip_think_blocks_leaves_only_the_answer(generated: str, expected: str) -> None:
    assert strip_think_blocks(generated) == expected
