#!/usr/bin/env python3
"""Validate the project-local durable-runner registry before authority use."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
from typing import Any


class RegistryPreflightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryPreflightError(message)


def path_exists(path: Path) -> bool:
    return os.path.lexists(path)


DEFAULT_SUBAGENT_PYTHON = Path(
    os.environ.get(
        "ARGUS_SKILL_PYTHON",
        "/home/argustest/miniconda3/bin/python3.13",
    )
)


_BOOTSTRAP_PROBE = r"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys


def fail(message: str) -> None:
    print(json.dumps({"status": "FAIL", "error": message}), file=sys.stderr)
    raise SystemExit(2)


payload = json.loads(sys.argv[1])
expected_root = Path(payload["project_root"]).resolve(strict=True)
actual_cwd = Path.cwd().resolve(strict=True)
if actual_cwd != expected_root:
    fail(f"bootstrap cwd mismatch: {actual_cwd} != {expected_root}")

try:
    from argus_skill.tools.subagent import _registry
except Exception as error:  # pragma: no cover - reported to caller verbatim
    fail(f"cannot import real subagent registry module: {error}")

registry_relative = _registry.REGISTRY_DIR
if registry_relative != Path(".argus_subagents"):
    fail(f"unexpected subagent registry relative path: {registry_relative}")

registry_relative.mkdir(parents=True, exist_ok=True)
registry_stat = os.lstat(registry_relative)
if not stat.S_ISDIR(registry_stat.st_mode):
    fail("real bootstrap registry is not a directory")

registry_path = (actual_cwd / registry_relative).resolve(strict=True)
expected_registry = expected_root / ".argus_subagents"
if registry_path != expected_registry:
    fail(f"registry resolved outside project root: {registry_path}")

canonical_task_id = payload.get("canonical_task_id") or ""
canonical_paths = []
if canonical_task_id:
    canonical_paths = [
        _registry._registry_path(canonical_task_id),
        *_registry._legacy_registry_paths(canonical_task_id),
        _registry._task_log_dir(canonical_task_id),
    ]
    existing = [str(path) for path in canonical_paths if os.path.lexists(path)]
    if existing:
        fail(f"canonical task id already allocated: {existing}")

probe_task_id = payload["probe_task_id"]
probe_path = _registry._registry_path(probe_task_id)
if probe_path.parent != registry_relative:
    fail(f"probe path escaped registry: {probe_path}")
if os.path.lexists(probe_path):
    fail(f"probe collision: {probe_path}")

probe_created = False
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
flags |= getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(probe_path, flags, 0o600)
probe_created = True
try:
    os.write(
        descriptor,
        json.dumps(
            {
                "kind": "ace3_argus_subagent_registry_bootstrap_probe",
                "task_id": probe_task_id,
                "cwd": str(actual_cwd),
            },
            sort_keys=True,
        ).encode("ascii") + b"\n",
    )
    os.fsync(descriptor)
finally:
    os.close(descriptor)

try:
    os.unlink(probe_path)
    probe_created = False
finally:
    if probe_created and os.path.lexists(probe_path):
        os.unlink(probe_path)

directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
directory_flags |= getattr(os, "O_NOFOLLOW", 0)
directory_fd = os.open(registry_relative, directory_flags)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)

if os.path.lexists(probe_path):
    fail(f"probe cleanup failed: {probe_path}")
if canonical_task_id:
    existing = [str(path) for path in canonical_paths if os.path.lexists(path)]
    if existing:
        fail(f"canonical task id changed during probe: {existing}")

print(json.dumps({
    "status": "PASS",
    "subagent_registry_module": str(Path(_registry.__file__).resolve(strict=True)),
    "cwd": str(actual_cwd),
    "registry": str(registry_path),
    "registry_mode": f"{stat.S_IMODE(registry_stat.st_mode):04o}",
    "registry_relative": str(registry_relative),
    "probe_task_id": probe_task_id,
    "probe_record": str((actual_cwd / probe_path).resolve(strict=False)),
    "canonical_task_id": canonical_task_id,
    "canonical_task_records": 0,
}))
"""


def _absent(path: Path, label: str) -> dict[str, Any]:
    require(not path_exists(path), f"{label} exists before preflight: {path}")
    return {"path": str(path), "state": "ABSENT_UNCHANGED"}


def _run_real_bootstrap_probe(
    root: Path,
    subagent_python: Path,
    canonical_task_id: str | None,
) -> dict[str, Any]:
    python = subagent_python.resolve(strict=True)
    probe_task_id = f".argus-registry-preflight-{secrets.token_hex(16)}"
    completed = subprocess.run(
        [
            str(python),
            "-B",
            "-c",
            _BOOTSTRAP_PROBE,
            json.dumps(
                {
                    "project_root": str(root),
                    "probe_task_id": probe_task_id,
                    "canonical_task_id": canonical_task_id or "",
                },
                sort_keys=True,
            ),
        ],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RegistryPreflightError(
            "real subagent registry bootstrap probe failed: "
            f"returncode={completed.returncode} "
            f"stdout={completed.stdout.strip()!r} "
            f"stderr={completed.stderr.strip()!r}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RegistryPreflightError(
            "real subagent registry bootstrap probe returned invalid JSON: "
            f"{completed.stdout!r}"
        ) from error
    require(result.get("status") == "PASS", "real bootstrap probe did not pass")
    require(
        result.get("probe_task_id") == probe_task_id,
        "real bootstrap probe task identity changed",
    )
    return {
        **result,
        "subagent_python": str(python),
    }


def preflight_registry(
    project_root: Path,
    authority_consumed_marker: Path,
    *,
    canonical_task_id: str | None = None,
    evidence_root: Path | None = None,
    forbidden_paths: tuple[Path, ...] = (),
    subagent_python: Path = DEFAULT_SUBAGENT_PYTHON,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    require(root.is_dir(), "project root is not a directory")
    require(
        not path_exists(authority_consumed_marker),
        "future authority is already consumed",
    )
    authority_state = {
        "path": str(authority_consumed_marker),
        "state": "ABSENT_UNCHANGED",
    }
    forbidden_state = [
        _absent(path, f"forbidden non-consuming path {index}")
        for index, path in enumerate(forbidden_paths)
    ]

    evidence_state: dict[str, Any] | None = None
    if evidence_root is not None:
        resolved_evidence = evidence_root.resolve(strict=True)
        require(
            resolved_evidence.is_dir(),
            "evidence root is not a directory",
        )
        evidence_mode = stat.S_IMODE(resolved_evidence.stat().st_mode)
        _absent(
            resolved_evidence / ".argus_subagents",
            "evidence-root registry",
        )
        evidence_state = {
            "path": str(resolved_evidence),
            "mode": f"{evidence_mode:04o}",
            "registry_state": "ABSENT_UNCHANGED",
        }

    bootstrap = _run_real_bootstrap_probe(
        root,
        subagent_python,
        canonical_task_id,
    )
    registry = Path(str(bootstrap["registry"]))
    registry_stat = os.lstat(registry)
    require(
        stat.S_ISDIR(registry_stat.st_mode),
        "project-local registry must be a real directory",
    )
    require(
        not registry.is_symlink(),
        "project-local registry must not be a symlink",
    )

    require(
        not path_exists(authority_consumed_marker),
        "authority marker changed during registry preflight",
    )
    for index, path in enumerate(forbidden_paths):
        require(
            not path_exists(path),
            f"forbidden non-consuming path changed during preflight: {index}",
        )
    if evidence_root is not None:
        require(
            not path_exists(Path(evidence_state["path"]) / ".argus_subagents"),
            "registry was created under immutable evidence root",
        )
    return {
        "schema_version": 1,
        "kind": "ace3_argus_subagent_registry_preflight",
        "status": "PASS",
        "project_root": str(root),
        "submitter_cwd": str(root),
        "registry": {
            "path": str(registry),
            "kind": "directory",
            "mode": f"{stat.S_IMODE(registry_stat.st_mode):04o}",
            "symlink": False,
            "exclusive_create_probe": "PASS_REMOVED",
            "probe_name": bootstrap["probe_task_id"],
            "probe_record": bootstrap["probe_record"],
        },
        "real_bootstrap": bootstrap,
        "evidence_root": evidence_state,
        "authority_consumed_marker": authority_state,
        "forbidden_paths": forbidden_state,
        "canonical_task_id": canonical_task_id,
        "canonical_task_allocations": 0,
        "durable_runner_submissions": 0,
        "payload_invocations": 0,
        "runtime_invocations": 0,
        "validator_invocations": 0,
        "authority_consumptions": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument(
        "--authority-consumed-marker",
        required=True,
        type=Path,
    )
    parser.add_argument("--canonical-task-id")
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument(
        "--forbid-path",
        action="append",
        default=[],
        type=Path,
    )
    parser.add_argument(
        "--subagent-python",
        default=DEFAULT_SUBAGENT_PYTHON,
        type=Path,
    )
    args = parser.parse_args()
    try:
        result = preflight_registry(
            args.project_root,
            args.authority_consumed_marker,
            canonical_task_id=args.canonical_task_id,
            evidence_root=args.evidence_root,
            forbidden_paths=tuple(args.forbid_path),
            subagent_python=args.subagent_python,
        )
    except (RegistryPreflightError, OSError) as error:
        print(f"ARGUS_SUBAGENT_REGISTRY_PREFLIGHT_FAIL {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
