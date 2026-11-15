"""The RAG prompt: answer rules from PLAN.md section 3.8 and numbered context chunks."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

# Runtime text the model reads, so it is in Spanish (PLAN.md). The rules ask for an
# answer in the question's language, so English questions still get English answers.
RAG_PROMPT_TEMPLATE: Final[str] = """\
Eres un asistente que responde preguntas sobre la documentación de FastAPI.

Reglas:
- Usa solo la información del contexto. No uses conocimiento propio.
- Si el contexto no contiene la respuesta, responde exactamente \
"No lo sé con la documentación disponible." o, si la pregunta está en inglés, \
"I don't know based on the available documentation."
- Responde en el mismo idioma que la pregunta.
- Cita los fragmentos que uses con su número entre corchetes, por ejemplo [1] o [2].

Contexto:
{context}

Pregunta: {question}
Respuesta:"""

# Qwen3 skips its <think> reasoning block when the prompt contains this switch.
NO_THINK_SWITCH: Final[str] = "/no_think"


@dataclass(frozen=True)
class ContextChunk:
    """A retrieved piece of the corpus, as the prompt shows it to the model.

    Attributes:
        text: The chunk content.
        source: File path of the document, relative to the corpus directory.
        section: Heading path of the chunk inside the document.
    """

    text: str
    source: str
    section: str


def build_rag_prompt(question: str, chunks: Sequence[ContextChunk], *, no_think: bool) -> str:
    """Return the full prompt for `question`, citing `chunks` as `[1]`, `[2]`, and so on.

    Args:
        question: The user's question, in any language.
        chunks: Retrieved context, in the order the citation numbers follow.
        no_think: Whether to end the prompt with `/no_think`, which turns off the Qwen3
            reasoning block.
    """
    prompt = RAG_PROMPT_TEMPLATE.format(context=format_context(chunks), question=question)
    if no_think:
        return f"{prompt}\n{NO_THINK_SWITCH}"
    return prompt


def format_context(chunks: Sequence[ContextChunk]) -> str:
    """Return the chunks as numbered blocks headed by their source and section."""
    return "\n\n".join(
        f"[{number}] {chunk.source} ({chunk.section})\n{chunk.text}"
        for number, chunk in enumerate(chunks, start=1)
    )
