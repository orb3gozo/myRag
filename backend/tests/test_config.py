from pathlib import Path

import pytest

from myrag.config import REPOSITORY_ROOT, EmbedProvider, RagEngine, Settings

SETTING_NAMES = [
    "RAG_ENGINE",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_TEMPERATURE",
    "LLM_NO_THINK",
    "EMBED_PROVIDER",
    "EMBED_MODEL",
    "EMBED_BASE_URL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "LANCEDB_PATH",
    "CORPUS_DIR",
]


@pytest.fixture
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def build_settings_without_env_file() -> Settings:
    return Settings(_env_file=None)


@pytest.mark.usefixtures("clean_environment")
def test_settings_use_plan_defaults_without_env_file() -> None:
    settings = build_settings_without_env_file()

    assert settings.rag_engine is RagEngine.LLAMAINDEX
    assert settings.llm_base_url == "http://localhost:1234/v1"
    assert settings.llm_model == "qwen3-1.7b"
    assert settings.llm_api_key == "lm-studio"
    assert settings.llm_temperature == 0.1
    assert settings.llm_no_think is True
    assert settings.embed_provider is EmbedProvider.FASTEMBED
    assert settings.embed_model == "intfloat/multilingual-e5-small"
    assert settings.embed_base_url == "http://localhost:1234/v1"
    assert settings.chunk_size == 400
    assert settings.chunk_overlap == 50
    assert settings.top_k == 4
    assert settings.lancedb_path == REPOSITORY_ROOT / "backend" / "storage" / "lancedb"
    assert settings.corpus_dir == REPOSITORY_ROOT / "data" / "fastapi-es"


@pytest.mark.parametrize(
    ("variable", "value", "field", "expected"),
    [
        ("LLM_MODEL", "qwen3-4b", "llm_model", "qwen3-4b"),
        ("LLM_NO_THINK", "false", "llm_no_think", False),
        ("TOP_K", "8", "top_k", 8),
        ("EMBED_PROVIDER", "openai", "embed_provider", EmbedProvider.OPENAI),
    ],
)
def test_environment_variable_overrides_default(
    clean_environment: pytest.MonkeyPatch,
    variable: str,
    value: str,
    field: str,
    expected: object,
) -> None:
    clean_environment.setenv(variable, value)

    settings = build_settings_without_env_file()

    assert getattr(settings, field) == expected


def test_relative_lancedb_path_resolves_under_repository_root(
    clean_environment: pytest.MonkeyPatch,
) -> None:
    clean_environment.setenv("LANCEDB_PATH", "backend/storage/lancedb")

    settings = build_settings_without_env_file()

    assert settings.lancedb_path.is_absolute()
    assert settings.lancedb_path == REPOSITORY_ROOT / "backend" / "storage" / "lancedb"


def test_relative_path_ignores_working_directory(
    clean_environment: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clean_environment.chdir(tmp_path)
    clean_environment.setenv("CORPUS_DIR", "data/other")

    settings = build_settings_without_env_file()

    assert settings.corpus_dir == REPOSITORY_ROOT / "data" / "other"


def test_absolute_path_is_kept(clean_environment: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_environment.setenv("CORPUS_DIR", str(tmp_path))

    settings = build_settings_without_env_file()

    assert settings.corpus_dir == tmp_path


def test_env_file_is_read_from_repository_root() -> None:
    assert Settings.model_config["env_file"] == REPOSITORY_ROOT / ".env"


def test_env_file_value_replaces_default(
    clean_environment: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-env-file\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.llm_model == "from-env-file"


def test_environment_variable_overrides_env_file(
    clean_environment: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-env-file\n", encoding="utf-8")
    clean_environment.setenv("LLM_MODEL", "from-environment")

    settings = Settings(_env_file=env_file)

    assert settings.llm_model == "from-environment"
