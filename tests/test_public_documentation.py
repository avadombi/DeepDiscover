"""Check documentation metadata and the published mathematical notation."""

from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_version_metadata_matches_package():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    module = (ROOT / "src/deepdiscover/__init__.py").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    version = metadata["project"]["version"]
    assert f'__version__ = "{version}"' in module
    assert f'version: "{version}"' in citation


def test_display_equations_have_markdown_delimiters():
    text = (ROOT / "docs/scientific_formulation.md").read_text(encoding="utf-8")
    assert "\\[" not in text and "\\]" not in text
    assert len(re.findall(r"(?m)^\$\$$", text)) >= 8
    assert len(re.findall(r"(?m)^\$\$$", text)) % 2 == 0
    assert "$A_i(t)$" not in text  # the main balance is displayed separately


def test_public_instructions_are_current():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    scientific = (ROOT / "docs/scientific_formulation.md").read_text(encoding="utf-8")
    assert "python -m pip install deepdiscover-hydro" in readme
    assert "https://pypi.org/project/deepdiscover-hydro/" in readme
    assert "## Overall balance" in scientific
    assert "## Informational memory" in scientific
    assert "pymdownx.arithmatex" in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
