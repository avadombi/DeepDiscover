"""Checks for the public DeepDiscover documentation."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_PAGE = ROOT / "docs" / "scientific_formulation.md"


def test_display_equations_have_markdown_delimiters():
    scientific = SCIENTIFIC_PAGE.read_text(encoding="utf-8")

    # The Sphinx/MyST math directive supplies equation labels and references.
    blocks = re.findall(
        r"(?ms)^```\{math\}[ \t]*\n(.*?)^```[ \t]*$",
        scientific,
    )
    assert len(blocks) >= 8, "Expected at least eight displayed math directives."

    labels = []
    for block in blocks:
        match = re.search(r"(?m)^:label:[ \t]*([\w.-]+)[ \t]*$", block)
        assert match, "Each displayed equation must have a Sphinx label."
        assert re.search(r"(?m)^\\[A-Za-z]+", block), (
            "A math directive should contain a LaTeX expression."
        )
        labels.append(match.group(1))

    assert len(labels) == len(set(labels)), "Equation labels must be unique."


def test_public_instructions_are_current():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    scientific = SCIENTIFIC_PAGE.read_text(encoding="utf-8")

    assert "python -m pip install deepdiscover-hydro" in readme
    assert "https://pypi.org/project/deepdiscover-hydro/" in readme

    assert "## Conservative allocation" in scientific
    assert "## Global balance and observation" in scientific
    assert r"\mathrm{s}_i^{(t)}" in scientific
