from pathlib import Path

import pytest

from myrag.config import REPOSITORY_ROOT, Settings

pytestmark = pytest.mark.integration


def read_variable_names(env_file: Path) -> set[str]:
    lines = env_file.read_text(encoding="utf-8").splitlines()
    return {
        line.split("=", 1)[0].strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_repository_root_contains_plan() -> None:
    assert (REPOSITORY_ROOT / "PLAN.md").is_file()


def test_env_example_lists_every_settings_field() -> None:
    documented = read_variable_names(REPOSITORY_ROOT / ".env.example")

    assert documented == {name.upper() for name in Settings.model_fields}
