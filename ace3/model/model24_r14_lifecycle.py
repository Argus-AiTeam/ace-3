#!/usr/bin/env python3
"""Submission, child launch, and terminal sealing for Model24 r14."""

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


def _path_entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _create_directory_exclusive(path: Path) -> None:
    if _path_entry_exists(path):
        raise SystemExit(f"payload directory collision: {path}")
    os.mkdir(path, mode=0o700)


def validate_payload_directory_paths(
    output: Path,
    simulation_dir: Path,
    payload_output_dir: Path,
) -> None:
    if not all(path.is_absolute() for path in (output, simulation_dir, payload_output_dir)):
        raise SystemExit("payload directories must be absolute")
    if output != Path(os.path.abspath(output)):
        raise SystemExit("output root is not lexically canonical")
    if simulation_dir != output / "controller-simulation":
        raise SystemExit("simulation directory is outside the canonical output root")
    if payload_output_dir != output / "rtl-cascade":
        raise SystemExit("payload output directory is outside the canonical output root")
    if output.parent.resolve(strict=True) != output.parent:
        raise SystemExit("output root parent is not canonical")


def validate_payload_directory_layout(
    output: Path,
    simulation_dir: Path,
    payload_output_dir: Path,
    *,
    launch_terminal_expected: bool = False,
) -> None:
    validate_payload_directory_paths(output, simulation_dir, payload_output_dir)
    expected_entries = {"controller-simulation", "rtl-cascade"}
    if launch_terminal_expected:
        expected_entries.add("launch-terminal.json")
    if not output.is_dir() or output.is_symlink():
        raise SystemExit("canonical output root is absent, foreign, or symlinked")
    if {path.name for path in output.iterdir()} != expected_entries:
        raise SystemExit("canonical output root contains absent or foreign entries")
    resolved_output = output.resolve(strict=True)
    if resolved_output != output:
        raise SystemExit("canonical output root resolves outside its declared path")
    for name, path in (
        ("simulation", simulation_dir),
        ("payload output", payload_output_dir),
    ):
        if not path.is_dir() or path.is_symlink():
            raise SystemExit(f"{name} directory is absent, foreign, or symlinked")
        if path.resolve(strict=True) != resolved_output / path.name:
            raise SystemExit(f"{name} directory resolves outside the canonical output root")


def create_payload_directories(
    output: Path,
    simulation_dir: Path,
    payload_output_dir: Path,
) -> None:
    validate_payload_directory_paths(output, simulation_dir, payload_output_dir)
    for path in (output, simulation_dir, payload_output_dir):
        if _path_entry_exists(path):
            raise SystemExit(f"payload directory collision: {path}")
    _create_directory_exclusive(output)
    fsync_directory(output.parent)
    _create_directory_exclusive(simulation_dir)
    fsync_directory(simulation_dir)
    fsync_directory(output)
    _create_directory_exclusive(payload_output_dir)
    fsync_directory(payload_output_dir)
    fsync_directory(output)
    validate_payload_directory_layout(output, simulation_dir, payload_output_dir)


def load_validator(package: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "model24_r14_sealed_validator", package / "validate-package.py"
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
    create_payload_directories(
        paths["output"],
        paths["simulation_dir"],
        paths["payload_output_dir"],
    )
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
            "kind": "ace3_model24_r14_launch_terminal",
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
            "output_root": str(paths["output"]),
            "simulation_dir": str(paths["simulation_dir"]),
            "payload_output_dir": str(paths["payload_output_dir"]),
            "output_root_resolved": str(paths["output"].resolve(strict=True)),
            "simulation_dir_resolved": str(
                paths["simulation_dir"].resolve(strict=True)
            ),
            "payload_output_dir_resolved": str(
                paths["payload_output_dir"].resolve(strict=True)
            ),
            "payload_directories_created_before_invocation": True,
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
        validator.validate_launch_terminal(launch_terminal, paths)
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
        "kind": "ace3_model24_r14_terminal_evidence",
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
