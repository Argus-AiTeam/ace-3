#!/usr/bin/env python3
"""Prepare or validate a non-executing fresh position-2 runtime package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_RELATIVE = Path(
    "ace3/model/validate_selected_token_position2_traversal.py"
)
PACKAGE_TOOLING_RELATIVE = (
    Path("ace3/model/prepare_selected_token_position2_fresh_package.py"),
    Path(
        "ace3/model/tests/"
        "test_prepare_selected_token_position2_fresh_package.py"
    ),
)
CHECKPOINT_RELATIVE = Path(
    "build/model24_rtl_cascade/checkpoint/model.safetensors"
)
TENSOR_MAP_RELATIVE = Path("ace3/contracts/model24_tensor_map.json")
CANONICAL_EVIDENCE_RELATIVE = Path(
    "build/model24_selected_token_position2/evidence.json"
)
SOURCE_REVIEW_PARENT_RELATIVE = Path(
    "build/model24_selected_token_position2_source_reviews"
)
SOURCE_REVIEW_KIND = "ace3_position2_fresh_post_repair_source_review"
SOURCE_SET_SCHEMA = "ace3_position2_accepted_source_set_v1"
SOURCE_SET_DIGEST_ALGORITHM = (
    "sha256(canonical-json; UTF-8/ASCII; sorted keys; compact separators; "
    "no trailing newline)"
)
PACKAGE_PARENT_RELATIVE = Path(
    "build/model24_selected_token_position2_packages"
)
OUTPUT_PARENT_RELATIVE = Path(
    "build/model24_selected_token_position2_runs"
)
REVIEW_PARENT_RELATIVE = Path(
    "build/model24_selected_token_position2_package_reviews"
)
AUTHORITY_PARENT_RELATIVE = Path(
    "build/model24_selected_token_position2_authorities"
)
PACKAGE_FILES = (
    "execution-boundary.json",
    "package-manifest.json",
    "review-request.json",
    "source-input-bindings.json",
    "token-inputs.json",
)
SEALED_FILES = (*PACKAGE_FILES, "package-seal.json")
RUN_ID_PATTERN = re.compile(r"ace3-position2-fresh-[a-z0-9][a-z0-9-]{7,63}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
PACKAGE_SCHEMA_VERSION = 2
PERFORMED_ACTIONS = (
    "model_execution",
    "lm_head_work",
    "position3_work",
    "position4_work",
    "dialogue_work",
    "synthesis",
    "ppa_measurement",
    "fpga_work",
    "git_operation",
    "ace2_access",
    "direct_make_execution",
)


class PackageError(RuntimeError):
    """Raised when package preparation or validation is not fail-closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageError(message)


def canonical_json(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def compact_canonical_json(document: object) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    require(path.is_file(), f"{label} is missing: {path}")
    require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    return path.resolve(strict=True)


def repository_file_record(
    repository: Path,
    relative_path: Path,
    label: str,
) -> dict[str, Any]:
    require(not relative_path.is_absolute(), f"{label} path must be relative")
    resolved = _regular_file(repository / relative_path, label)
    require(
        resolved.is_relative_to(repository),
        f"{label} resolves outside the repository",
    )
    return {
        "path": relative_path.as_posix(),
        "absolute_path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def external_file_record(path: Path, label: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _regular_file(resolved, label)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def package_file_record(package_root: Path, name: str) -> dict[str, Any]:
    path = _regular_file(package_root / name, f"package file {name}")
    require(path.parent == package_root, f"package file escaped package root: {name}")
    return {
        "path": name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> dict[str, Any]:
    resolved = _regular_file(path, "JSON document")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackageError(f"invalid JSON document: {path}: {error}") from error
    require(isinstance(document, dict), f"JSON object required: {path}")
    return document


def _assigned_literal(tree: ast.Module, name: str) -> Any:
    matches: list[ast.expr] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            matches.append(node.value)
    require(len(matches) == 1, f"validator must assign {name} exactly once")
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError) as error:
        raise PackageError(f"validator {name} must be a literal") from error


def validator_contract(repository: Path) -> dict[str, Any]:
    validator = _regular_file(
        repository / VALIDATOR_RELATIVE,
        "fresh position-2 validator",
    )
    try:
        tree = ast.parse(validator.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise PackageError(f"cannot parse fresh position-2 validator: {error}") from error
    source_paths = _assigned_literal(tree, "CONSUMED_SOURCE_PATHS")
    require(
        isinstance(source_paths, dict)
        and source_paths
        and all(
            isinstance(label, str)
            and isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            for label, relative in source_paths.items()
        ),
        "validator consumed source mapping is malformed",
    )
    expected_validator = source_paths.get("validator")
    require(
        expected_validator == VALIDATOR_RELATIVE.as_posix(),
        "validator does not bind itself in its consumed source closure",
    )
    values = {
        name: _assigned_literal(tree, name)
        for name in (
            "CHECKPOINT_SHA256",
            "EMBEDDING_SHA256",
            "POSITION0_EMBEDDING_SHA256",
            "POSITION1_EMBEDDING_SHA256",
            "POSITION0_TOKEN_ID",
            "POSITION1_TOKEN_ID",
            "SELECTED_TOKEN_ID",
            "POSITION",
        )
    }
    for name in (
        "CHECKPOINT_SHA256",
        "EMBEDDING_SHA256",
        "POSITION0_EMBEDDING_SHA256",
        "POSITION1_EMBEDDING_SHA256",
    ):
        require(
            isinstance(values[name], str)
            and SHA256_PATTERN.fullmatch(values[name]) is not None,
            f"validator {name} is not a SHA-256",
        )
    require(
        values["POSITION0_TOKEN_ID"] == 151644
        and values["POSITION1_TOKEN_ID"] == 2114
        and values["SELECTED_TOKEN_ID"] == 271
        and values["POSITION"] == 2,
        "validator token/position contract differs from fresh position-2 scope",
    )
    return {"source_paths": source_paths, **values}


def _accepted_source_review(
    repository: Path,
    source_review_path: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    review_path = _regular_file(source_review_path, "accepted source review")
    review_parent = (repository / SOURCE_REVIEW_PARENT_RELATIVE).resolve(
        strict=True
    )
    require(
        review_path.parent == review_parent,
        "accepted source review is outside the canonical review namespace",
    )
    require(
        stat.S_IMODE(review_path.stat().st_mode) & 0o222 == 0,
        "accepted source review is not immutable",
    )
    review = load_json(review_path)
    require(
        review.get("schema_version") == 1
        and review.get("kind") == SOURCE_REVIEW_KIND
        and review.get("verdict") == "PASS",
        "accepted source review identity or verdict mismatch",
    )
    scope = review.get("scope")
    require(
        isinstance(scope, dict)
        and scope.get("position") == 2
        and scope.get("selected_token_id") == 271
        and scope.get("review_type")
        == "fresh read-only post-repair source review",
        "accepted source review scope mismatch",
    )

    accepted = review.get("accepted_source_set")
    require(isinstance(accepted, dict), "accepted source set is missing")
    source_files = accepted.get("files")
    narrow_target = accepted.get("narrow_makefile_target")
    digest = accepted.get("canonical_digest")
    require(
        accepted.get("schema") == SOURCE_SET_SCHEMA
        and isinstance(source_files, list)
        and source_files
        and isinstance(narrow_target, dict)
        and isinstance(digest, dict),
        "accepted source set structure mismatch",
    )
    canonical_source_set = {
        "files": source_files,
        "narrow_makefile_target": narrow_target,
        "schema": SOURCE_SET_SCHEMA,
    }
    canonical_payload = compact_canonical_json(canonical_source_set)
    require(
        digest
        == {
            "algorithm": SOURCE_SET_DIGEST_ALGORITHM,
            "canonical_bytes": len(canonical_payload),
            "sha256": sha256_bytes(canonical_payload),
        },
        "accepted source-set canonical digest mismatch",
    )

    reviewed_paths: list[str] = []
    for record in source_files:
        require(
            isinstance(record, dict)
            and set(record) == {"bytes", "path", "sha256"}
            and isinstance(record.get("path"), str),
            "accepted source file record is malformed",
        )
        relative = Path(record["path"])
        actual = repository_file_record(
            repository,
            relative,
            "accepted reviewed source",
        )
        require(
            {
                "bytes": actual["bytes"],
                "path": actual["path"],
                "sha256": actual["sha256"],
            }
            == record,
            f"accepted reviewed source changed: {relative.as_posix()}",
        )
        reviewed_paths.append(relative.as_posix())
    require(
        reviewed_paths == sorted(set(reviewed_paths)),
        "accepted source file paths are not unique and sorted",
    )

    declared_labels = accepted.get("declared_source_labels")
    require(
        declared_labels == contract["source_paths"],
        "accepted source labels differ from validator consumed sources",
    )
    require(
        reviewed_paths == sorted(set(contract["source_paths"].values())),
        "accepted source files do not close validator consumption",
    )
    require(
        accepted.get("local_python_import_closure")
        == {"missing": [], "status": "PASS"},
        "accepted source local import closure is incomplete",
    )

    require(
        set(narrow_target)
        == {
            "bytes",
            "end_line",
            "path",
            "sha256",
            "start_line",
            "target",
        }
        and narrow_target.get("path") == "Makefile"
        and narrow_target.get("target")
        == "model24-selected-token-position2-tests"
        and isinstance(narrow_target.get("start_line"), int)
        and isinstance(narrow_target.get("end_line"), int)
        and 1
        <= narrow_target["start_line"]
        <= narrow_target["end_line"],
        "accepted narrow Makefile target binding is malformed",
    )
    makefile = _regular_file(repository / "Makefile", "Makefile")
    target_bytes = b"".join(
        makefile.read_bytes().splitlines(keepends=True)[
            narrow_target["start_line"] - 1 : narrow_target["end_line"]
        ]
    )
    require(
        len(target_bytes) == narrow_target["bytes"]
        and sha256_bytes(target_bytes) == narrow_target["sha256"]
        and target_bytes.startswith(
            b"model24-selected-token-position2-tests:"
        ),
        "accepted narrow Makefile target changed",
    )

    relative_review = review_path.relative_to(repository)
    return {
        "artifact": {
            "path": relative_review.as_posix(),
            "absolute_path": str(review_path),
            "bytes": review_path.stat().st_size,
            "sha256": sha256_file(review_path),
        },
        "review_id": review.get("review_id"),
        "kind": review["kind"],
        "verdict": review["verdict"],
        "accepted_source_set": {
            "schema": SOURCE_SET_SCHEMA,
            "file_count": len(source_files),
            "canonical_bytes": len(canonical_payload),
            "sha256": sha256_bytes(canonical_payload),
        },
        "consumed_source_closure": {
            "status": "PASS",
            "declared_source_labels": dict(
                sorted(contract["source_paths"].items())
            ),
            "reviewed_file_paths": reviewed_paths,
            "local_python_import_closure": {
                "missing": [],
                "status": "PASS",
            },
            "narrow_makefile_target": narrow_target,
        },
    }


def _write_exclusive(path: Path, document: Mapping[str, Any]) -> None:
    payload = canonical_json(document)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_scoped_paths(
    repository: Path,
    package_root: Path,
    output_root: Path,
    run_id: str,
) -> None:
    require(
        RUN_ID_PATTERN.fullmatch(run_id) is not None,
        "run ID must be a fresh ace3-position2-fresh identity",
    )
    require(
        package_root
        == repository / PACKAGE_PARENT_RELATIVE / run_id,
        "package root is outside the canonical fresh package namespace",
    )
    require(
        output_root
        == repository / OUTPUT_PARENT_RELATIVE / run_id,
        "output root is outside the canonical fresh output namespace",
    )
    require(not package_root.exists(), f"package identity already exists: {package_root}")
    require(not output_root.exists(), f"output identity already exists: {output_root}")
    require(
        not (repository / CANONICAL_EVIDENCE_RELATIVE).exists(),
        "canonical position-2 evidence already exists",
    )


def _source_bindings(
    repository: Path,
    contract: Mapping[str, Any],
    source_review_path: Path,
) -> dict[str, Any]:
    accepted_review = _accepted_source_review(
        repository,
        source_review_path,
        contract,
    )
    source_records = {
        label: repository_file_record(repository, Path(relative), label)
        for label, relative in sorted(contract["source_paths"].items())
    }
    tooling_records = [
        repository_file_record(repository, relative, "package tooling")
        for relative in PACKAGE_TOOLING_RELATIVE
    ]
    checkpoint = repository_file_record(
        repository,
        CHECKPOINT_RELATIVE,
        "official checkpoint",
    )
    require(
        checkpoint["sha256"] == contract["CHECKPOINT_SHA256"],
        "official checkpoint SHA-256 differs from fresh validator binding",
    )
    tensor_map = repository_file_record(
        repository,
        TENSOR_MAP_RELATIVE,
        "Model24 tensor map",
    )
    interpreter = external_file_record(Path(sys.executable), "package interpreter")
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "kind": "ace3_position2_fresh_source_input_bindings",
        "accepted_source_review": accepted_review,
        "accepted_fresh_only_sources": source_records,
        "package_tooling": tooling_records,
        "execution_inputs": {
            "checkpoint": {
                **checkpoint,
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "provenance": (
                    "project-local authenticated official checkpoint consumed "
                    "by the fresh position-2 validator"
                ),
            },
            "tensor_map": tensor_map,
            "python_interpreter": interpreter,
        },
        "source_selection_basis": (
            "exact immutable PASS post-repair source review and canonical "
            "accepted source-set digest; package acceptance remains pending "
            "independent review"
        ),
    }


def _token_inputs(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "kind": "ace3_position2_fresh_token_inputs",
        "checkpoint": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "sha256": contract["CHECKPOINT_SHA256"],
            "embedding_tensor": "model.embed_tokens.weight",
        },
        "sequence": [
            {
                "position": 0,
                "token_id": contract["POSITION0_TOKEN_ID"],
                "embedding_sha256": contract["POSITION0_EMBEDDING_SHA256"],
                "role": "fresh causal replay seed",
            },
            {
                "position": 1,
                "token_id": contract["POSITION1_TOKEN_ID"],
                "embedding_sha256": contract["POSITION1_EMBEDDING_SHA256"],
                "role": "fresh causal replay seed",
            },
            {
                "position": contract["POSITION"],
                "token_id": contract["SELECTED_TOKEN_ID"],
                "embedding_sha256": contract["EMBEDDING_SHA256"],
                "role": "fixed authenticated position-2 input",
            },
        ],
        "provenance": {
            "token_ids": (
                "literal fresh-validator inputs; token 271 is not claimed as "
                "an accepted lm_head output"
            ),
            "embedding_hashes": (
                "expected FP16 row hashes bound by the fresh validator to the "
                "authenticated official checkpoint"
            ),
            "validator": VALIDATOR_RELATIVE.as_posix(),
        },
    }


def prepare_package(
    repository: Path,
    package_root: Path,
    output_root: Path,
    run_id: str,
    source_review_path: Path,
) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    package_root = package_root.resolve(strict=False)
    output_root = output_root.resolve(strict=False)
    _require_scoped_paths(repository, package_root, output_root, run_id)
    contract = validator_contract(repository)
    source_bindings = _source_bindings(
        repository,
        contract,
        source_review_path,
    )
    token_inputs = _token_inputs(contract)

    package_root.parent.mkdir(parents=True, exist_ok=True)
    package_root.mkdir(mode=0o755)
    _write_exclusive(
        package_root / "source-input-bindings.json",
        source_bindings,
    )
    _write_exclusive(package_root / "token-inputs.json", token_inputs)

    review_path = repository / REVIEW_PARENT_RELATIVE / f"{run_id}.json"
    authority_path = repository / AUTHORITY_PARENT_RELATIVE / f"{run_id}.json"
    future_evidence = output_root / "evidence.json"
    execution_boundary = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "kind": "ace3_position2_fresh_execution_boundary",
        "performed": {name: False for name in PERFORMED_ACTIONS},
        "allowed_package_actions": [
            "read and hash package source/input paths",
            "write and seal this preparation package",
            "run non-executing package schema/path/hash validation",
        ],
        "execution_prohibition": {
            "status": "PROHIBITED",
            "until": [
                "independent package review accepts this exact package seal",
                "separate exact execution authority is issued afterward",
            ],
            "review_does_not_grant_execution": True,
        },
        "claim_boundary": {
            "position2_runtime": "not run; no PASS claimed",
            "canonical_evidence": "absent",
            "lm_head": "not run and no parent claimed",
            "position3_or_position4": "not run",
            "dialogue": "not run",
            "synthesis_ppa_fpga": "not run or measured",
        },
    }
    _write_exclusive(
        package_root / "execution-boundary.json",
        execution_boundary,
    )

    review_request = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "kind": "ace3_position2_fresh_independent_package_review_request",
        "package_id": run_id,
        "package_root": str(package_root),
        "requested_reviewer_role": "independent reviewer",
        "scope": (
            "package schema, path, hash, namespace, provenance, and "
            "non-execution boundary only"
        ),
        "execution_during_review": "prohibited",
        "acceptance_effect": (
            "package acceptance only; execution remains prohibited without "
            "a later separate exact authority"
        ),
        "review_artifact": {
            "path": str(review_path),
            "state_at_seal": "ABSENT",
        },
        "accepted_source_review": source_bindings[
            "accepted_source_review"
        ],
    }
    _write_exclusive(package_root / "review-request.json", review_request)

    validator_path = repository / VALIDATOR_RELATIVE
    interpreter_path = Path(source_bindings["execution_inputs"]["python_interpreter"]["path"])
    generate_argv = [
        str(interpreter_path),
        "-B",
        str(validator_path),
        "generate",
        "--output",
        str(future_evidence),
    ]
    validate_argv = [
        str(interpreter_path),
        "-B",
        str(validator_path),
        "validate",
        "--output",
        str(future_evidence),
    ]
    manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "kind": "ace3_position2_fresh_runtime_package",
        "status": "SEALED_AWAITING_INDEPENDENT_PACKAGE_REVIEW",
        "package_id": run_id,
        "position": contract["POSITION"],
        "selected_token_id": contract["SELECTED_TOKEN_ID"],
        "package_root": str(package_root),
        "bindings": {
            name: package_file_record(package_root, name)
            for name in (
                "execution-boundary.json",
                "review-request.json",
                "source-input-bindings.json",
                "token-inputs.json",
            )
        },
        "accepted_source_review": source_bindings[
            "accepted_source_review"
        ],
        "namespaces": {
            "run_id": run_id,
            "fresh_output_root": str(output_root),
            "fresh_evidence": str(future_evidence),
            "fresh_output_state_at_seal": "ABSENT",
            "canonical_evidence": str(repository / CANONICAL_EVIDENCE_RELATIVE),
            "canonical_evidence_state_at_seal": "ABSENT",
        },
        "future_execution_plan": {
            "status": "PROHIBITED",
            "cwd": str(repository),
            "generate_argv": generate_argv,
            "validate_argv": validate_argv,
            "direct_make_execution": False,
        },
        "independent_review": {
            "status": "REQUIRED",
            "artifact_path": str(review_path),
            "artifact_state_at_seal": "ABSENT",
        },
        "execution_authority": {
            "status": "WITHHELD",
            "artifact_path": str(authority_path),
            "artifact_state_at_seal": "ABSENT",
            "must_postdate_accepted_review": True,
        },
        "runtime_evidence": {
            "status": "ABSENT",
            "position2_pass_claimed": False,
            "lm_head_parent_claimed": False,
        },
    }
    _write_exclusive(package_root / "package-manifest.json", manifest)

    records = [
        package_file_record(package_root, name)
        for name in PACKAGE_FILES
    ]
    seal = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "kind": "ace3_position2_fresh_package_seal",
        "package_id": run_id,
        "records": records,
        "records_sha256": sha256_bytes(canonical_json(records)),
        "accepted_source_review_sha256": source_bindings[
            "accepted_source_review"
        ]["artifact"]["sha256"],
        "accepted_source_set_sha256": source_bindings[
            "accepted_source_review"
        ]["accepted_source_set"]["sha256"],
        "fresh_output_root": str(output_root),
        "fresh_output_state_at_seal": "ABSENT",
        "canonical_evidence_state_at_seal": "ABSENT",
        "review_state_at_seal": "ABSENT",
        "authority_state_at_seal": "WITHHELD",
    }
    _write_exclusive(package_root / "package-seal.json", seal)
    for name in SEALED_FILES:
        (package_root / name).chmod(0o444)
    package_root.chmod(0o555)
    _fsync_directory(package_root)
    _fsync_directory(package_root.parent)
    return manifest


def _validate_record(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    require(actual == expected, f"{label} binding mismatch")


def _validate_readonly_package(package_root: Path) -> None:
    require(
        package_root.is_dir() and not package_root.is_symlink(),
        "regular package directory required",
    )
    require(
        stat.S_IMODE(package_root.stat().st_mode) & 0o222 == 0,
        "package root is not read-only",
    )
    actual_names = sorted(path.name for path in package_root.iterdir())
    require(
        actual_names == sorted(SEALED_FILES),
        "sealed package file set differs",
    )
    for name in SEALED_FILES:
        path = package_root / name
        _regular_file(path, f"sealed package file {name}")
        require(
            stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
            f"sealed package file is writable: {name}",
        )


def validate_package(repository: Path, package_root: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    package_root = package_root.resolve(strict=True)
    _validate_readonly_package(package_root)
    manifest = load_json(package_root / "package-manifest.json")
    seal = load_json(package_root / "package-seal.json")
    source_bindings = load_json(package_root / "source-input-bindings.json")
    token_inputs = load_json(package_root / "token-inputs.json")
    execution_boundary = load_json(package_root / "execution-boundary.json")
    review_request = load_json(package_root / "review-request.json")

    require(
        manifest.get("schema_version") == PACKAGE_SCHEMA_VERSION
        and manifest.get("kind") == "ace3_position2_fresh_runtime_package"
        and manifest.get("status")
        == "SEALED_AWAITING_INDEPENDENT_PACKAGE_REVIEW",
        "package manifest identity mismatch",
    )
    run_id = manifest.get("package_id")
    require(isinstance(run_id, str), "package ID is missing")
    output_root = repository / OUTPUT_PARENT_RELATIVE / run_id
    require(
        package_root == repository / PACKAGE_PARENT_RELATIVE / run_id,
        "package root does not match package ID",
    )
    require(
        manifest.get("package_root") == str(package_root),
        "manifest package root mismatch",
    )

    expected_records = [
        package_file_record(package_root, name)
        for name in PACKAGE_FILES
    ]
    require(
        seal
        == {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "kind": "ace3_position2_fresh_package_seal",
            "package_id": run_id,
            "records": expected_records,
            "records_sha256": sha256_bytes(canonical_json(expected_records)),
            "accepted_source_review_sha256": source_bindings[
                "accepted_source_review"
            ]["artifact"]["sha256"],
            "accepted_source_set_sha256": source_bindings[
                "accepted_source_review"
            ]["accepted_source_set"]["sha256"],
            "fresh_output_root": str(output_root),
            "fresh_output_state_at_seal": "ABSENT",
            "canonical_evidence_state_at_seal": "ABSENT",
            "review_state_at_seal": "ABSENT",
            "authority_state_at_seal": "WITHHELD",
        },
        "package seal mismatch",
    )
    expected_bindings = {
        name: package_file_record(package_root, name)
        for name in (
            "execution-boundary.json",
            "review-request.json",
            "source-input-bindings.json",
            "token-inputs.json",
        )
    }
    require(
        manifest.get("bindings") == expected_bindings,
        "manifest package bindings mismatch",
    )

    contract = validator_contract(repository)
    review_binding = source_bindings.get("accepted_source_review")
    require(
        isinstance(review_binding, dict)
        and isinstance(review_binding.get("artifact"), dict)
        and isinstance(
            review_binding["artifact"].get("absolute_path"),
            str,
        ),
        "accepted source review binding is missing",
    )
    expected_sources = _source_bindings(
        repository,
        contract,
        Path(review_binding["artifact"]["absolute_path"]),
    )
    _validate_record(
        source_bindings,
        expected_sources,
        "fresh source/input",
    )
    require(
        manifest.get("accepted_source_review")
        == source_bindings["accepted_source_review"],
        "manifest accepted source review binding mismatch",
    )
    _validate_record(
        token_inputs,
        _token_inputs(contract),
        "token input",
    )

    canonical_evidence = repository / CANONICAL_EVIDENCE_RELATIVE
    review_path = repository / REVIEW_PARENT_RELATIVE / f"{run_id}.json"
    authority_path = repository / AUTHORITY_PARENT_RELATIVE / f"{run_id}.json"
    fresh_evidence = output_root / "evidence.json"
    require(not output_root.exists(), "fresh output namespace is not absent")
    require(not canonical_evidence.exists(), "canonical evidence is not absent")
    require(not review_path.exists(), "independent review already exists")
    require(not authority_path.exists(), "execution authority already exists")
    require(
        manifest.get("namespaces")
        == {
            "run_id": run_id,
            "fresh_output_root": str(output_root),
            "fresh_evidence": str(fresh_evidence),
            "fresh_output_state_at_seal": "ABSENT",
            "canonical_evidence": str(canonical_evidence),
            "canonical_evidence_state_at_seal": "ABSENT",
        },
        "manifest namespace binding mismatch",
    )

    interpreter = source_bindings["execution_inputs"]["python_interpreter"]["path"]
    validator = str(repository / VALIDATOR_RELATIVE)
    require(
        manifest.get("future_execution_plan")
        == {
            "status": "PROHIBITED",
            "cwd": str(repository),
            "generate_argv": [
                interpreter,
                "-B",
                validator,
                "generate",
                "--output",
                str(fresh_evidence),
            ],
            "validate_argv": [
                interpreter,
                "-B",
                validator,
                "validate",
                "--output",
                str(fresh_evidence),
            ],
            "direct_make_execution": False,
        },
        "future execution plan mismatch",
    )
    require(
        execution_boundary.get("performed")
        == {name: False for name in PERFORMED_ACTIONS},
        "prohibited activity record differs",
    )
    prohibition = execution_boundary.get("execution_prohibition", {})
    require(
        prohibition.get("status") == "PROHIBITED"
        and prohibition.get("review_does_not_grant_execution") is True
        and len(prohibition.get("until", [])) == 2,
        "execution prohibition is incomplete",
    )
    require(
        manifest.get("execution_authority")
        == {
            "status": "WITHHELD",
            "artifact_path": str(authority_path),
            "artifact_state_at_seal": "ABSENT",
            "must_postdate_accepted_review": True,
        },
        "execution authority boundary mismatch",
    )
    require(
        manifest.get("independent_review")
        == {
            "status": "REQUIRED",
            "artifact_path": str(review_path),
            "artifact_state_at_seal": "ABSENT",
        },
        "independent review boundary mismatch",
    )
    require(
        review_request.get("package_id") == run_id
        and review_request.get("package_root") == str(package_root)
        and review_request.get("execution_during_review") == "prohibited"
        and review_request.get("review_artifact")
        == {"path": str(review_path), "state_at_seal": "ABSENT"}
        and review_request.get("accepted_source_review")
        == source_bindings["accepted_source_review"],
        "review request mismatch",
    )
    require(
        manifest.get("runtime_evidence")
        == {
            "status": "ABSENT",
            "position2_pass_claimed": False,
            "lm_head_parent_claimed": False,
        },
        "runtime evidence boundary mismatch",
    )
    return {
        "package_id": run_id,
        "package_root": str(package_root),
        "package_seal_sha256": sha256_file(
            package_root / "package-seal.json"
        ),
        "source_count": len(
            source_bindings["accepted_fresh_only_sources"]
        ),
        "execution_input_count": len(source_bindings["execution_inputs"]),
        "token_count": len(token_inputs["sequence"]),
        "execution_invocations": 0,
        "model_invocations": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--package-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--source-review", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--package-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.operation == "prepare":
            document = prepare_package(
                ROOT,
                arguments.package_root,
                arguments.output_root,
                arguments.run_id,
                arguments.source_review,
            )
            print(
                "POSITION2_FRESH_PACKAGE_PREPARED "
                f"package_id={document['package_id']} "
                f"status={document['status']} "
                "execution_invocations=0"
            )
        else:
            result = validate_package(ROOT, arguments.package_root)
            print(
                "POSITION2_FRESH_PACKAGE_VALIDATION_PASS "
                f"package_id={result['package_id']} "
                f"seal_sha256={result['package_seal_sha256']} "
                f"sources={result['source_count']} "
                f"inputs={result['execution_input_count']} "
                f"tokens={result['token_count']} "
                "execution_invocations=0 model_invocations=0"
            )
    except (OSError, PackageError) as error:
        raise SystemExit(
            f"POSITION2_FRESH_PACKAGE_{arguments.operation.upper()}_FAIL: {error}"
        ) from error


if __name__ == "__main__":
    main()
