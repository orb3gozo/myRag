"""LLM client for the local OpenAI-compatible server, and cleanup of its generated text."""

import re
from typing import Final

from llama_index.core.llms import LLM
from llama_index.llms.openai_like import OpenAILike

from myrag.config import Settings

# Non-greedy and DOTALL so each block ends at its own closing tag, across lines.
_THINK_BLOCK: Final[re.Pattern[str]] = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_llm(settings: Settings) -> LLM:
    """Return a chat client for the server at `LLM_BASE_URL` (LM Studio or Ollama)."""
    return OpenAILike(
        api_base=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        is_chat_model=True,
    )


def strip_think_blocks(generated: str) -> str:
    """Remove Qwen3 `<think>...</think>` reasoning blocks and the whitespace around them."""
    return _THINK_BLOCK.sub("", generated).strip()
