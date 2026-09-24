#!/usr/bin/env python3
"""Exactly-once launcher for a reviewed Model24 r11 durable package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_exclusive(path: Path, value: object, mode: int = 0o400) -> None:
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def load_validator(package: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "model24_r11_sealed_validator", package / "validate-package.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("sealed package validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--stdout-log", required=True, type=Path)
    parser.add_argument("--stderr-log", required=True, type=Path)
    args = parser.parse_args()

    package = Path(__file__).resolve().parent
    validator = load_validator(package)
    validator.validate_package(
        package,
        "launch",
        review_path=args.review,
        authority_path=args.authority,
        receipt_path=args.receipt,
        stdout_log=args.stdout_log,
        stderr_log=args.stderr_log,
    )
    manifest = validator.load_json(package / "package.json")
    paths = validator.expected_paths(package, manifest)

    consumed = paths["authority_consumed"]
    descriptor = os.open(consumed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(args.authority.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())

    os.mkdir(paths["output"], mode=0o700)
    environment = {
        "HOME": os.environ["HOME"],
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        **manifest["resources"]["environment"],
    }
    status = 125
    try:
        status = subprocess.run(
            manifest["execution"]["payload_argv"],
            cwd=manifest["execution"]["payload_cwd"],
            env=environment,
            check=False,
        ).returncode
    finally:
        terminal = {
            "schema_version": 1,
            "kind": "ace3_model24_r11_launch_terminal",
            "nonce": manifest["identity"]["nonce"],
            "task_id": manifest["identity"]["task_id"],
            "exit_code": status,
            "natural_terminal": status == 0,
            "cardinality": 1,
            "retry": False,
            "replay": False,
            "resume": False,
            "watcher": False,
            "package_manifest_sha256": digest(package / "package.json"),
            "package_seal_sha256": digest(package / "seal.json"),
            "review_sha256": digest(args.review),
            "authority_sha256": digest(args.authority),
            "authority_consumed_sha256": digest(consumed),
            "receipt_path": str(args.receipt),
            "stdout_log": str(args.stdout_log),
            "stderr_log": str(args.stderr_log),
        }
        write_exclusive(paths["output"] / "launch-terminal.json", terminal)
        directory = os.open(paths["output"], os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
