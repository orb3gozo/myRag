"""Load the Markdown corpus as LlamaIndex documents, cleaned for retrieval and citation."""

import logging
import re
from pathlib import Path
from typing import Final

from llama_index.core import Document

_HEADING_ANCHOR: Final[re.Pattern[str]] = re.compile(r"^(#{1,6} .*?)[ \t]*\{ #[^}]*\}[ \t]*$", re.M)
# Matches `{* ../../docs_src/path/file.py hl[6:7] ln[1:9] title["file.py"] *}`. The options
# only control how the docs site renders the snippet, so they are ignored.
_INCLUDE_DIRECTIVE: Final[re.Pattern[str]] = re.compile(
    r"^\{\* +(?P<path>\S+)(?: +(?:hl|ln|title)\[[^\]]*\])* *\*\}$", re.M
)
_DOCS_SRC_PREFIX: Final[str] = "../../docs_src/"
_HEADING: Final[re.Pattern[str]] = re.compile(r"^#{1,6} +(?P<title>.+)$", re.M)
_CORPUS_LANGUAGE: Final[str] = "es"
# Snippets that the docs site embeds in other pages, not pages of their own.
_SNIPPET_PAGE_NAMES: Final[frozenset[str]] = frozenset({"translation-banner.md"})

_logger: Final[logging.Logger] = logging.getLogger(__name__)


def load_corpus(corpus_dir: Path, *, docs_src_dir: Path) -> list[Document]:
    """Read every Markdown page under `corpus_dir` and return one cleaned document per page.

    Include directives point into `docs_src_dir`. A directive whose file is missing stays
    in the text unchanged and is logged as a warning.
    """
    return [
        _load_page(path, corpus_dir=corpus_dir, docs_src_dir=docs_src_dir)
        for path in sorted(corpus_dir.rglob("*.md"))
        if _is_content_page(path)
    ]


def _is_content_page(path: Path) -> bool:
    return not path.name.startswith("_") and path.name not in _SNIPPET_PAGE_NAMES


def _load_page(page_path: Path, *, corpus_dir: Path, docs_src_dir: Path) -> Document:
    source = page_path.relative_to(corpus_dir).as_posix()
    text = _clean_markdown(
        page_path.read_text(encoding="utf-8"), source=source, docs_src_dir=docs_src_dir
    )
    metadata = {"source": source, "title": _find_title(text), "lang": _CORPUS_LANGUAGE}
    return Document(text=text, metadata=metadata)


def _clean_markdown(markdown: str, *, source: str, docs_src_dir: Path) -> str:
    without_anchors = _HEADING_ANCHOR.sub(r"\1", markdown)
    return _INCLUDE_DIRECTIVE.sub(
        lambda directive: _expand_include(directive, source, docs_src_dir), without_anchors
    )


def _expand_include(directive: re.Match[str], source: str, docs_src_dir: Path) -> str:
    included_path = _resolve_include(directive["path"], docs_src_dir)
    if included_path is None:
        _logger.warning("Include sin resolver en %s: %s", source, directive[0])
        return directive[0]
    code = included_path.read_text(encoding="utf-8").rstrip("\n")
    return f"```python\n{code}\n```"


def _resolve_include(directive_path: str, docs_src_dir: Path) -> Path | None:
    if not directive_path.startswith(_DOCS_SRC_PREFIX):
        return None
    included_path = docs_src_dir / directive_path.removeprefix(_DOCS_SRC_PREFIX)
    return included_path if included_path.is_file() else None


def _find_title(markdown: str) -> str:
    first_heading = _HEADING.search(markdown)
    return first_heading["title"].strip() if first_heading else ""
