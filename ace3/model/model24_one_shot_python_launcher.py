#!/usr/bin/env python3
"""Preflight the exact Python used by a future Model24 one-shot package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


class LauncherError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LauncherError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(1024 * 1024):
            digest.update(payload)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def canonical_json(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise LauncherError(f"{path.name} is not valid JSON") from error
    require(isinstance(document, dict), f"{path.name} is not a JSON object")
    return document


def probe_project_python(project_python: Path) -> dict[str, Any]:
    resolved = project_python.resolve(strict=True)
    require(resolved.is_file(), "project Python is not a file")
    require(os.access(resolved, os.X_OK), "project Python is not executable")
    probe = """
import json
import pathlib
import platform
import sys

import numpy
import torch

print(json.dumps({
    "executable": str(pathlib.Path(sys.executable).resolve()),
    "implementation": platform.python_implementation(),
    "numpy_version": numpy.__version__,
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(resolved), "-I", "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        reason = detail[-1] if detail else f"exit {completed.returncode}"
        raise LauncherError(f"project Python dependency probe failed: {reason}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise LauncherError("project Python dependency probe returned invalid JSON") from error
    require(isinstance(result, dict), "project Python dependency probe result is not an object")
    reported = Path(str(result.get("executable", ""))).resolve(strict=True)
    require(reported == resolved, "project Python probe reported a different executable")
    require(bool(result.get("numpy_version")), "project Python did not report NumPy")
    require(bool(result.get("torch_version")), "project Python did not report PyTorch")
    return result


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    require(path.parent.is_dir(), "receipt parent directory does not exist")
    payload = canonical_json(receipt)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise LauncherError("preflight receipt already exists") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def preflight(
    project_python: Path,
    entrypoint: Path,
    receipt_path: Path,
    entrypoint_args: list[str],
    dry_run: bool,
    runtime_contract_path: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    resolved_python = project_python.resolve(strict=True)
    resolved_entrypoint = entrypoint.resolve(strict=True)
    require(resolved_entrypoint.is_file(), "entrypoint is not a file")
    probe = probe_project_python(resolved_python)
    launch_argv = [str(resolved_python), str(resolved_entrypoint), *entrypoint_args]
    runtime_identity = {
        "project_python": {
            **file_record(resolved_python),
            "implementation": probe["implementation"],
            "python_version": probe["python_version"],
            "required_imports": {
                "numpy": probe["numpy_version"],
                "torch": probe["torch_version"],
            },
        },
        "entrypoint": file_record(resolved_entrypoint),
    }
    runtime_contract = None
    if runtime_contract_path is not None:
        contract_path = runtime_contract_path.resolve(strict=True)
        contract = load_json(contract_path)
        require(
            contract.get("schema_version") == 1
            and contract.get("kind") == "ace3_model24_python_runtime_contract",
            "runtime contract identity mismatch",
        )
        require(
            contract.get("runtime_identity") == runtime_identity,
            "stale Python/runtime contract mismatch",
        )
        runtime_contract = file_record(contract_path)
    receipt = {
        "schema_version": 1,
        "kind": "ace3_model24_one_shot_python_preflight",
        "mode": "dry-run" if dry_run else "launch",
        "launcher": file_record(Path(__file__)),
        **runtime_identity,
        "runtime_contract": runtime_contract,
        "launch_argv": launch_argv,
        "ordering": "receipt fsynced before entrypoint invocation",
        "authority_consumed_by_launcher": False,
        "entrypoint_invoked_at_receipt_creation": False,
    }
    write_receipt(receipt_path, receipt)
    return receipt, launch_argv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-python", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("entrypoint_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    entrypoint_args = args.entrypoint_args
    if entrypoint_args[:1] == ["--"]:
        entrypoint_args = entrypoint_args[1:]
    try:
        receipt, launch_argv = preflight(
            args.project_python,
            args.entrypoint,
            args.receipt,
            entrypoint_args,
            args.dry_run,
            args.runtime_contract,
        )
    except (LauncherError, OSError) as error:
        print(f"MODEL24_ONE_SHOT_PYTHON_PREFLIGHT_FAIL {error}", file=sys.stderr)
        return 2
    print(
        "MODEL24_ONE_SHOT_PYTHON_PREFLIGHT_PASS "
        f"python={receipt['project_python']['path']} mode={receipt['mode']}"
    )
    if args.dry_run:
        return 0
    os.execve(launch_argv[0], launch_argv, os.environ.copy())
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
