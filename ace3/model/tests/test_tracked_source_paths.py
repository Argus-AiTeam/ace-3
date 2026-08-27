#!/usr/bin/env python3
"""Reject tracked publication sources that depend on a legacy checkout path."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


LEGACY_NAME = "ace" + "-2"
PATH_END = r"(?=/|\s|[\"'`),;:\]}]|$)"
ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:[^\s/]+/)*" + re.escape(LEGACY_NAME) + PATH_END,
    re.IGNORECASE,
)
PARENT_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])\.\./" + re.escape(LEGACY_NAME) + PATH_END,
    re.IGNORECASE,
)


def prohibited_reference(line: str) -> bool:
    return ABSOLUTE_PATH.search(line) is not None or PARENT_PATH.search(line) is not None


def tracked_paths(root: Path) -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return [root / name.decode("utf-8") for name in output.split(b"\0") if name]


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    forbidden_examples = (
        f"/home/example/{LEGACY_NAME}/fixture",
        f"../{LEGACY_NAME}/fixture",
    )
    allowed_examples = (
        f"{LEGACY_NAME.upper()} remains historical provenance",
        "/predecessor_reuse_audit",
    )
    if not all(prohibited_reference(value) for value in forbidden_examples):
        raise AssertionError("path regression does not detect every prohibited form")
    if any(prohibited_reference(value) for value in allowed_examples):
        raise AssertionError("path regression rejects allowed provenance text")

    paths = tracked_paths(root)
    violations: list[str] = []
    for path in paths:
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if prohibited_reference(line):
                violations.append(f"{path.relative_to(root)}:{line_number}:{line.strip()}")

    if violations:
        details = "\n".join(violations)
        raise SystemExit(f"TRACKED_SOURCE_PATH_REJECT\n{details}")
    print(
        "TRACKED_SOURCE_PATH_PASS "
        "absolute_legacy_paths=absent parent_legacy_paths=absent "
        f"tracked_files={len(paths)}"
    )


if __name__ == "__main__":
    main()
