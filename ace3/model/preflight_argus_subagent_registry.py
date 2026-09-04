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
RUNTIME_MAKE_TARGET = "model24-rtl-layer-compile"
RUNTIME_BOUNDARY_LAYERS = (0, 23)


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


_SOURCE_ARCHIVE_TARGET_PROBE = r"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile


def fail(message: str) -> None:
    print(json.dumps({"status": "FAIL", "error": message}), file=sys.stderr)
    raise SystemExit(2)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(8 * 1024 * 1024):
            hasher.update(payload)
    return hasher.hexdigest()


def safe_member_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    require(str(relative) not in {"", "."}, "empty source archive member")
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"unsafe source archive member: {name}",
    )
    target = root.joinpath(*relative.parts)
    require(
        target.resolve(strict=False).is_relative_to(root),
        f"source archive member escapes extraction root: {name}",
    )
    return target


def extract_archive(archive: Path, root: Path) -> tuple[int, int]:
    member_count = 0
    file_count = 0
    with tarfile.open(archive, "r") as stream:
        for member in stream.getmembers():
            member_count += 1
            target = safe_member_path(root, member.name)
            mode = stat.S_IMODE(member.mode) & 0o777
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                target.chmod(mode or 0o755)
                continue
            require(member.isfile(), f"unsupported source archive entry: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            require(source is not None, f"source archive file is unreadable: {member.name}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(mode or 0o644)
            file_count += 1
    return member_count, file_count


payload = json.loads(sys.argv[1])
expected_root = Path(payload["project_root"]).resolve(strict=True)
actual_cwd = Path.cwd().resolve(strict=True)
if actual_cwd != expected_root:
    fail(f"source archive probe cwd mismatch: {actual_cwd} != {expected_root}")

archive_input = Path(payload["source_archive"])
archive = archive_input if archive_input.is_absolute() else actual_cwd / archive_input
archive = archive.resolve(strict=True)
require(archive.is_file() and not archive.is_symlink(), f"sealed source archive is not a regular file: {archive}")
require(
    archive.is_relative_to(actual_cwd),
    f"sealed source archive is outside the project cwd: {archive}",
)

build_root_input = Path(payload["build_root"])
build_root = build_root_input if build_root_input.is_absolute() else actual_cwd / build_root_input
build_root = build_root.absolute()
target = payload["make_target"]
layers = payload["layers"]
archive_record = {
    "path": str(archive),
    "bytes": archive.stat().st_size,
    "sha256": sha256_file(archive),
}

with tempfile.TemporaryDirectory(prefix="ace3-source-archive-preflight-") as directory:
    extraction_root = Path(directory).resolve(strict=True) / "source"
    extraction_root.mkdir()
    require(
        build_root != extraction_root and extraction_root not in build_root.parents,
        "bound build root must be outside the extracted sealed source",
    )
    member_count, file_count = extract_archive(archive, extraction_root)
    makefile = extraction_root / "Makefile"
    require(
        makefile.is_file() and not makefile.is_symlink(),
        "sealed source archive does not contain a regular Makefile",
    )
    makefile_text = makefile.read_text(encoding="utf-8")
    require(
        re.search(rf"(?m)^{re.escape(target)}\s*:", makefile_text) is not None,
        f"sealed source Makefile target is missing: {target}",
    )
    resolution = {}
    source_build = extraction_root / "build"
    for layer in layers:
        command = [
            "make",
            "--no-print-directory",
            "--dry-run",
            target,
            f"MODEL24_RTL_LAYER_INDEX={layer}",
            "MODEL24_RTL_ACCURATE_SILU=1",
            f"MODEL24_RTL_CASCADE_DIR={build_root}",
        ]
        completed = subprocess.run(
            command,
            cwd=extraction_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        combined = completed.stdout + completed.stderr
        require(
            completed.returncode == 0,
            f"sealed source Makefile target is not invocable for layer {layer}: "
            f"returncode={completed.returncode} stderr={completed.stderr.strip()!r}",
        )
        require(
            str(build_root) in combined,
            f"sealed source Makefile dry-run did not route layer {layer} into the bound build root",
        )
        require(
            str(source_build) not in combined,
            f"sealed source Makefile dry-run routes layer {layer} into the extracted source tree",
        )
        resolution[str(layer)] = {
            "exit_code": completed.returncode,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
        }
    require(not source_build.exists(), "source archive probe created an extracted-source build directory")
    require(not build_root.exists(), "source archive dry-run unexpectedly created the bound build root")

print(json.dumps({
    "status": "PASS",
    "cwd": str(actual_cwd),
    "source_archive": archive_record,
    "extracted_source": "TEMP_REMOVED",
    "member_count": member_count,
    "file_count": file_count,
    "make_target": target,
    "build_root": str(build_root),
    "make_target_resolution": resolution,
    "build_root_state": "ABSENT_UNCHANGED",
    "runtime_invocations": 0,
    "validator_invocations": 0,
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


def _run_source_archive_target_probe(
    root: Path,
    source_archive: Path,
    build_root: Path,
    *,
    subagent_python: Path,
    make_target: str,
    layers: tuple[int, ...],
) -> dict[str, Any]:
    python = subagent_python.resolve(strict=True)
    completed = subprocess.run(
        [
            str(python),
            "-B",
            "-c",
            _SOURCE_ARCHIVE_TARGET_PROBE,
            json.dumps(
                {
                    "project_root": str(root),
                    "source_archive": str(source_archive),
                    "build_root": str(build_root),
                    "make_target": make_target,
                    "layers": list(layers),
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
            "source archive Makefile target preflight failed: "
            f"returncode={completed.returncode} "
            f"stdout={completed.stdout.strip()!r} "
            f"stderr={completed.stderr.strip()!r}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RegistryPreflightError(
            "source archive Makefile target preflight returned invalid JSON: "
            f"{completed.stdout!r}"
        ) from error
    require(result.get("status") == "PASS", "source archive target probe did not pass")
    require(
        result.get("make_target") == make_target,
        "source archive target identity changed",
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
    source_archive: Path | None = None,
    build_root: Path | None = None,
    make_target: str = RUNTIME_MAKE_TARGET,
    make_layers: tuple[int, ...] = RUNTIME_BOUNDARY_LAYERS,
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

    source_archive_state: dict[str, Any] | None = None
    if source_archive is not None:
        require(
            build_root is not None,
            "source archive preflight requires a bound build root",
        )
        source_archive_state = _run_source_archive_target_probe(
            root,
            source_archive,
            build_root,
            subagent_python=subagent_python,
            make_target=make_target,
            layers=make_layers,
        )

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
        "source_archive_preflight": source_archive_state,
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
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--make-target", default=RUNTIME_MAKE_TARGET)
    args = parser.parse_args()
    try:
        result = preflight_registry(
            args.project_root,
            args.authority_consumed_marker,
            canonical_task_id=args.canonical_task_id,
            evidence_root=args.evidence_root,
            forbidden_paths=tuple(args.forbid_path),
            subagent_python=args.subagent_python,
            source_archive=args.source_archive,
            build_root=args.build_root,
            make_target=args.make_target,
        )
    except (RegistryPreflightError, OSError) as error:
        print(f"ARGUS_SUBAGENT_REGISTRY_PREFLIGHT_FAIL {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
