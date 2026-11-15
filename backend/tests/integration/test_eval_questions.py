import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Final

import pytest

from myrag.config import REPOSITORY_ROOT

pytestmark = pytest.mark.integration

QUESTIONS_FILE: Final[Path] = REPOSITORY_ROOT / "eval" / "questions.jsonl"
CORPUS_DIR: Final[Path] = REPOSITORY_ROOT / "data" / "fastapi-es"
FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "id",
        "lang",
        "question",
        "expected_sources",
        "reference_answer",
        "answerable",
    }
)
# Files the corpus loader (#2) skips, so they can never be retrieved.
SKIPPED_FILE_NAME: Final[str] = "translation-banner.md"


def read_questions() -> list[dict[str, object]]:
    lines = QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def read_question_id(question: dict[str, object]) -> str:
    return str(question["id"])


def read_expected_sources() -> list[str]:
    sources: list[str] = []
    for question in read_questions():
        expected = question["expected_sources"]
        assert isinstance(expected, list)
        sources.extend(str(source) for source in expected)
    return sources


def test_file_has_twenty_json_objects_with_plan_fields() -> None:
    questions = read_questions()

    assert len(questions) == 20
    assert all(set(question) == FIELD_NAMES for question in questions)


def test_file_splits_questions_by_language_and_answerability() -> None:
    split = Counter((question["lang"], question["answerable"]) for question in read_questions())

    assert split[("es", True)] == 9
    assert split[("en", True)] == 8
    assert split[("es", False)] + split[("en", False)] == 3
    assert split.total() == 20


def test_ids_are_unique_and_prefixed_with_language() -> None:
    questions = read_questions()

    ids = [question["id"] for question in questions]
    assert len(set(ids)) == len(ids)
    assert all(str(question["id"]).startswith(f"{question['lang']}-") for question in questions)


@pytest.mark.parametrize("question", read_questions(), ids=read_question_id)
def test_question_and_reference_answer_are_non_empty_text(question: dict[str, object]) -> None:
    assert isinstance(question["question"], str) and question["question"].strip()
    assert isinstance(question["reference_answer"], str) and question["reference_answer"].strip()


@pytest.mark.parametrize("question", read_questions(), ids=read_question_id)
def test_only_answerable_questions_have_expected_sources(question: dict[str, object]) -> None:
    assert isinstance(question["answerable"], bool)
    assert isinstance(question["expected_sources"], list)
    assert bool(question["expected_sources"]) == question["answerable"]


@pytest.mark.parametrize("source", read_expected_sources())
def test_expected_source_is_a_file_the_corpus_loader_reads(source: str) -> None:
    path = PurePosixPath(source)

    assert not path.is_absolute()
    assert ".." not in path.parts
    assert path.suffix == ".md"
    assert not path.name.startswith("_")
    assert path.name != SKIPPED_FILE_NAME


@pytest.mark.skipif(not CORPUS_DIR.is_dir(), reason="corpus not downloaded (just fetch-corpus)")
@pytest.mark.parametrize("source", read_expected_sources())
def test_expected_source_exists_in_corpus(source: str) -> None:
    assert (CORPUS_DIR / source).is_file()
