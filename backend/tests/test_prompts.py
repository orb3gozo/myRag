from myrag.prompts import ContextChunk, build_rag_prompt

CHUNKS = [
    ContextChunk(
        text="Puedes declarar parámetros de path con la misma sintaxis.",
        source="tutorial/path-params.md",
        section="Parámetros de path",
    ),
    ContextChunk(
        text="Declara otros parámetros que no son parte del path.",
        source="tutorial/query-params.md",
        section="Parámetros de query > Valores por defecto",
    ),
]


def test_build_rag_prompt_numbers_each_chunk_with_its_source_and_section() -> None:
    prompt = build_rag_prompt("¿Qué es un parámetro de path?", CHUNKS, no_think=False)

    assert "[1] tutorial/path-params.md (Parámetros de path)" in prompt
    assert "Puedes declarar parámetros de path con la misma sintaxis." in prompt
    assert "[2] tutorial/query-params.md (Parámetros de query > Valores por defecto)" in prompt
    assert "Declara otros parámetros que no son parte del path." in prompt


def test_build_rag_prompt_includes_the_question() -> None:
    prompt = build_rag_prompt("How do I declare a query parameter?", CHUNKS, no_think=False)

    assert "How do I declare a query parameter?" in prompt


def test_build_rag_prompt_ends_with_no_think_when_enabled() -> None:
    prompt = build_rag_prompt("¿Qué es un parámetro de path?", CHUNKS, no_think=True)

    assert prompt.endswith("/no_think")


def test_build_rag_prompt_omits_no_think_when_disabled() -> None:
    prompt = build_rag_prompt("¿Qué es un parámetro de path?", CHUNKS, no_think=False)

    assert "/no_think" not in prompt


def test_build_rag_prompt_gives_the_exact_refusal_phrases() -> None:
    prompt = build_rag_prompt("¿Qué es un parámetro de path?", CHUNKS, no_think=False)

    assert "No lo sé con la documentación disponible." in prompt
    assert "I don't know based on the available documentation." in prompt
