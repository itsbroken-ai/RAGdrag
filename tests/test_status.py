"""Generated release status must follow executable capability metadata."""

import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from ragdrag.engine.phases import PHASE_METADATA


def status_module():
    # An absent feature is an assertion failure, not a collection error.
    assert importlib.util.find_spec("ragdrag.engine.status") is not None, "status renderer is missing"
    return importlib.import_module("ragdrag.engine.status")


def test_checked_in_status_matches_registry():
    expected = status_module().render_status_markdown()
    assert Path("docs/implementation-status.md").read_text(encoding="utf-8") == expected


def test_every_implemented_technique_has_capability_metadata():
    text = status_module().render_status_markdown()
    for phase in ("R1", "R2", "R3", "R4", "R5", "R6"):
        assert f"| {phase} |" in text


def test_table_has_contract_columns_and_only_registered_r6_techniques():
    text = status_module().render_status_markdown()
    assert text.startswith(
        "# RAGdrag Implementation Status\n\n"
        "| Phase | Capability | Technique IDs | Impact | Maturity | Release |\n"
        "|---|---|---|---|---|---|\n"
    )
    assert "| R6 | phase.r6 | RD-0601, RD-0603, RD-0604 | active-non-mutating | validated | 0.6.0 |\n" in text
    assert "RD-0602" not in text
    assert text == status_module().render_status_markdown()


@pytest.mark.parametrize("validation_test", [None, "missing-test.py", "."])
def test_validation_requires_a_real_declared_test_file(monkeypatch, validation_test):
    module = status_module()
    monkeypatch.setitem(PHASE_METADATA, "R6", replace(PHASE_METADATA["R6"], validation_test=validation_test))
    assert "| active-non-mutating | implemented | 0.6.0 |" in module.render_status_markdown()


@pytest.mark.parametrize("maturity", ["catalogued", "experimental", "implemented"])
def test_renderer_does_not_promote_declared_maturity(monkeypatch, maturity):
    module = status_module()
    monkeypatch.setitem(PHASE_METADATA, "R6", replace(PHASE_METADATA["R6"], maturity=maturity))
    assert f"| active-non-mutating | {maturity} | 0.6.0 |" in module.render_status_markdown()


def test_write_and_check_detect_artifact_drift(tmp_path):
    module = status_module()
    destination = tmp_path / "status.md"
    assert module.main(["--write", str(destination)]) == 0
    assert destination.read_text(encoding="utf-8").startswith("# RAGdrag Implementation Status\n")
    assert module.main(["--check", str(destination)]) == 0
    destination.write_text("stale\n", encoding="utf-8")
    assert module.main(["--check", str(destination)]) == 1
    assert destination.read_text(encoding="utf-8") == "stale\n"


def test_check_missing_artifact_fails_without_traceback(tmp_path, capsys):
    assert status_module().main(["--check", str(tmp_path / "missing.md")]) == 1
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [[], ["--check", "status.md", "--write", "status.md"]])
def test_exactly_one_status_mode_is_required(arguments):
    module = status_module()
    with pytest.raises(SystemExit) as exc:
        module.main(arguments)
    assert exc.value.code == 2
