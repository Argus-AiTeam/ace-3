#!/usr/bin/env python3
"""Gate terminal/raw evidence before opening final-RMSNorm oracle files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HIDDEN_SIZE = 896
TERMINAL_SCHEMA = "ace3-final-rmsnorm-terminal-v1"


def read_ascii(path: Path) -> str:
    return path.read_bytes().decode("ascii", errors="strict")


def parse_terminal(path: Path, actual_exit_code: int) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in read_ascii(path).splitlines():
        if not line or line.count("=") != 1:
            raise ValueError("malformed terminal line")
        key, value = line.split("=", 1)
        if key in fields:
            raise ValueError(f"duplicate terminal field: {key}")
        fields[key] = value
    required = {
        "schema",
        "natural_terminal",
        "recorded_exit_code",
        "output_count",
        "case_index",
    }
    if set(fields) != required:
        raise ValueError("terminal field set mismatch")
    if fields["schema"] != TERMINAL_SCHEMA:
        raise ValueError("terminal schema mismatch")
    if fields["natural_terminal"] != "1":
        raise ValueError("simulator did not terminate naturally")
    if fields["recorded_exit_code"] != "0" or actual_exit_code != 0:
        raise ValueError("simulator exit-code disagreement")
    if fields["output_count"] != str(HIDDEN_SIZE):
        raise ValueError("terminal output count mismatch")
    return fields


def parse_raw(path: Path) -> list[int]:
    outputs: list[int] = []
    for expected_index, line in enumerate(read_ascii(path).splitlines()):
        parts = line.split()
        if len(parts) != 2 or not parts[0].isdigit() or len(parts[1]) != 4:
            raise ValueError("malformed raw row")
        if int(parts[0]) != expected_index:
            raise ValueError("raw index is missing, duplicated, or out of order")
        if any(char not in "0123456789abcdefABCDEF" for char in parts[1]):
            raise ValueError("raw row contains an unknown or non-hexadecimal value")
        outputs.append(int(parts[1], 16))
    if len(outputs) != HIDDEN_SIZE:
        raise ValueError("raw output is truncated or overlong")
    return outputs


def compare(
    terminal_path: Path,
    raw_path: Path,
    manifest_path: Path,
    expected_path: Path,
    report_path: Path,
    actual_exit_code: int,
) -> None:
    terminal = parse_terminal(terminal_path, actual_exit_code)
    outputs = parse_raw(raw_path)
    case_index = int(terminal["case_index"])

    manifest = json.loads(read_ascii(manifest_path))
    if manifest.get("schema") != "ace3-final-rmsnorm-vectors-v1":
        raise ValueError("vector manifest schema mismatch")
    if manifest.get("hidden_size") != HIDDEN_SIZE or manifest.get("case_count") != 4:
        raise ValueError("vector manifest geometry mismatch")
    expected_lines = read_ascii(expected_path).splitlines()
    if len(expected_lines) != 4 * HIDDEN_SIZE:
        raise ValueError("expected vector count mismatch")
    start = case_index * HIDDEN_SIZE
    expected = [int(value, 16) for value in expected_lines[start : start + HIDDEN_SIZE]]
    mismatches = [
        (index, actual, wanted)
        for index, (actual, wanted) in enumerate(zip(outputs, expected, strict=True))
        if actual != wanted
    ]
    if mismatches:
        index, actual, wanted = mismatches[0]
        raise ValueError(
            f"oracle mismatch at output {index}: actual={actual:04x} expected={wanted:04x}"
        )
    report = {
        "schema": "ace3-final-rmsnorm-comparison-v1",
        "case_index": case_index,
        "comparisons": HIDDEN_SIZE,
        "mismatches": 0,
    }
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--simulator-exit-code", type=int, required=True)
    args = parser.parse_args()
    compare(
        args.terminal,
        args.raw,
        args.manifest,
        args.expected,
        args.report,
        args.simulator_exit_code,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
