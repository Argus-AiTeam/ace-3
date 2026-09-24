#!/usr/bin/env python3
"""Submission, child launch, and terminal sealing for Model24 r13."""

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


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_validator(package: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "model24_r13_sealed_validator", package / "validate-package.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("sealed package validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def submit(package: Path, review: Path, authority: Path) -> None:
    validator = load_validator(package)
    validator.validate_package(
        package,
        "pre-submit",
        review_path=review,
        authority_path=authority,
    )
    manifest = validator.load_json(package / "package.json")
    paths = validator.expected_paths(package, manifest)
    os.mkdir(paths["terminal_root"], mode=0o700)
    fsync_directory(paths["terminal_root"].parent)
    result = subprocess.run(
        manifest["execution"]["submission_argv"],
        cwd=paths["submitter_root"],
        check=False,
        capture_output=True,
    )
    os.write(1, result.stdout)
    os.write(2, result.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    receipt = json.loads(result.stdout.decode("utf-8"))
    if (
        receipt.get("state") != "submitted"
        or receipt.get("task_id") != manifest["identity"]["task_id"]
        or receipt.get("run_id") is None
        or receipt.get("check_with") is None
    ):
        raise SystemExit("durable runner did not return a complete submitted receipt")


def launch(package: Path, review: Path, authority: Path) -> None:
    validator = load_validator(package)
    validator.validate_package(
        package,
        "launch",
        review_path=review,
        authority_path=authority,
    )
    manifest = validator.load_json(package / "package.json")
    paths = validator.expected_paths(package, manifest)
    consumed = paths["authority_consumed"]
    descriptor = os.open(consumed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(authority.read_bytes())
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
            cwd=manifest["execution"]["child_cwd"],
            env=environment,
            check=False,
        ).returncode
    finally:
        terminal = {
            "schema_version": 1,
            "kind": "ace3_model24_r13_launch_terminal",
            "nonce": manifest["identity"]["nonce"],
            "task_id": manifest["identity"]["task_id"],
            "exit_code": status,
            "natural_terminal": status == 0,
            "cardinality": 1,
            "invocation_count": 1,
            "retry": False,
            "replay": False,
            "resume": False,
            "watcher": False,
            "package_manifest_sha256": digest(package / "package.json"),
            "package_seal_sha256": digest(package / "seal.json"),
            "review_sha256": digest(review),
            "authority_sha256": digest(authority),
            "authority_consumed_sha256": digest(consumed),
        }
        write_exclusive(paths["output"] / "launch-terminal.json", terminal)
        fsync_directory(paths["output"])
    raise SystemExit(status)


def finalize(package: Path) -> None:
    validator = load_validator(package)
    seal, manifest = validator.validate_package_seal(package)
    validator.validate_source(package, seal)
    validator.validate_r12_evidence(package, manifest)
    validator.validate_contract(package, manifest)
    paths = validator.expected_paths(package, manifest)
    artifacts = validator.validate_terminal_layout(
        paths["terminal_root"],
        manifest["identity"]["task_id"],
        manifest["execution"]["durable_command"],
        paths["child_cwd"],
    )
    launch_terminal_path = paths["output"] / "launch-terminal.json"
    launch_terminal = (
        validator.load_json(launch_terminal_path)
        if launch_terminal_path.is_file()
        else None
    )
    if launch_terminal is not None:
        validator.validate_payload_cardinality(launch_terminal)
    records = {}
    for name, path in (
        ("runner_receipt", artifacts["receipt_path"]),
        ("stdout_log", artifacts["stdout"]),
        ("stderr_log", artifacts["stderr"]),
        ("exit_sidecar", artifacts["exit_sidecar"]),
        ("launch_terminal", launch_terminal_path),
    ):
        if path.is_file():
            records[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
    terminal = {
        "schema_version": 1,
        "kind": "ace3_model24_r13_terminal_evidence",
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "run_id": artifacts["receipt"]["run_id"],
        "runner_state": artifacts["receipt"]["state"],
        "runner_exit_code": artifacts["receipt"].get("exit_code"),
        "payload_invocation_count": (
            launch_terminal["invocation_count"] if launch_terminal else 0
        ),
        "records": records,
        "canonical_stdout_stderr_reconstructed": False,
    }
    write_exclusive(paths["terminal_manifest"], terminal)
    fsync_directory(paths["terminal_root"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("submit", "launch", "finalize"))
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    if args.action == "submit":
        if args.authority is None or args.review is None:
            raise SystemExit("submit requires --authority and --review")
        submit(package, args.review, args.authority)
    elif args.action == "launch":
        if args.authority is None or args.review is None:
            raise SystemExit("launch requires --authority and --review")
        launch(package, args.review, args.authority)
    else:
        if args.authority is not None or args.review is not None:
            raise SystemExit("finalize forbids authority and review arguments")
        finalize(package)


if __name__ == "__main__":
    main()
