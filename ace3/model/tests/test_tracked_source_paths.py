#!/usr/bin/env python3
"""Reject tracked publication sources that depend on local runtime paths."""

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
LOCAL_CHECKPOINT_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:home|dev/shm|tmp|run/user)/(?:[^\s/]+/)*[^\s/]+\.safetensors"
    + PATH_END,
    re.IGNORECASE,
)


def prohibited_reference(line: str) -> bool:
    return any(
        pattern.search(line) is not None
        for pattern in (ABSOLUTE_PATH, PARENT_PATH, LOCAL_CHECKPOINT_PATH)
    )


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
        "/home/example/ace3/model24_execution_vectors/model." + "safetensors",
    )
    allowed_examples = (
        f"{LEGACY_NAME.upper()} remains historical provenance",
        "/ace2_reuse_audit",
        '"rtl/ace2_rmsnorm_core.sv": "b09fe7073fd6509f0"',
        '"ace2_relationship": "No ACE-2 arithmetic source is copied."',
        "build/ace2_chat_demo/accepted-evidence.json",
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
        "local_checkpoint_paths=absent provenance_records=allowed "
        f"tracked_files={len(paths)}"
    )


if __name__ == "__main__":
    main()
