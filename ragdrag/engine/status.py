"""Deterministic release status derived from executable phase metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from ragdrag import __version__
from ragdrag.engine.phases import PHASE_METADATA


HEADER = (
    "# RAGdrag Implementation Status\n\n"
    "| Phase | Capability | Technique IDs | Impact | Maturity | Release |\n"
    "|---|---|---|---|---|---|\n"
)


def render_status_markdown() -> str:
    """Render registry claims, downgrading validation without a test file.

    Validation paths are relative to the checkout, so installed distributions
    without the test suite do not claim local validation.
    """
    rows = []
    for phase, metadata in PHASE_METADATA.items():
        maturity = metadata.maturity
        if maturity == "validated" and (
            metadata.validation_test is None
            or not Path(metadata.validation_test).is_file()
        ):
            maturity = "implemented"
        rows.append(
            f"| {phase} | {metadata.capability_id} | "
            f"{', '.join(metadata.technique_ids)} | {metadata.impact.value} | "
            f"{maturity} | {__version__} |"
        )
    return HEADER + "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", type=Path)
    mode.add_argument("--write", type=Path)
    args = parser.parse_args(argv)
    rendered = render_status_markdown()
    if args.write is not None:
        args.write.write_text(rendered, encoding="utf-8")
        return 0
    try:
        return 0 if args.check.read_text(encoding="utf-8") == rendered else 1
    except (OSError, UnicodeError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
