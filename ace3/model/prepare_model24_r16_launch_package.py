#!/usr/bin/env python3
"""Construct and inertly validate a full Model24 r16 launch package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Any


REPOSITORY = Path("/home/argustest/ace3-argus")
ACCEPTED_COMMIT = "42c895ce1e5fea00525e9f2f7fef66f0fbb8e118"
R15_BUILDER = REPOSITORY / "ace3/model/prepare_model24_r15_launch_package.py"
R15_VALIDATOR = REPOSITORY / "ace3/model/validate_model24_r15_launch_package.py"
R15_LIFECYCLE = REPOSITORY / "ace3/model/model24_r15_lifecycle.py"
R15_CONTRACT = REPOSITORY / "ace3/contracts/model24_r15_durable_launch_contract.json"
R16_VALIDATOR = REPOSITORY / "ace3/model/validate_model24_r16_launch_package.py"
V2 = Path("/home/argustest/ace3-model24-r15-successor-v2-20260829-rtl-compile-closure")
V2_PACKAGE = V2 / "package"
V2_HASHES = {
    "package.json": "349a07566e2a19ffef929c45162e62f95e14ea4b982c1a7d8283f9dd033ef418",
    "seal.json": "d8dbad9f1184bb03b0bc3f245ebcc7014252e4772457d07ea689297a18b1ece1",
    "review-request.json": "61405da89e8c03798213ebb9c3b0d4e4607c47e38b7c33a5242a2fd52bc252ad",
    "validate-package.py": "2d4f612d3a80b9f195557d7e9ff49ccb15fc6910a1733f77b818c8daa2a48778",
    "source-tree.json": "23a6a0caa6cfeb453110a5a51e59fefb1a0495dde8522b478c75a826d238a278",
    "layer0-compile-result.json": "fd74ae247267f7cf8842885c0bc9fd041ca0015b179cbf19df039340a54b492f",
}
V2_SOURCE_FILES = {
    "package.json": V2_PACKAGE / "package.json",
    "seal.json": V2_PACKAGE / "seal.json",
    "review-request.json": V2_PACKAGE / "review-request.json",
    "validate-package.py": V2_PACKAGE / "validate-package.py",
    "source-tree.json": V2_PACKAGE / "source-tree.json",
    "layer0-compile-result.json": V2_PACKAGE / "evidence/layer0-compile/result.json",
}
OVERLAY_PATHS = (
    "Makefile",
    "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
    "ace3/rtl/ace3_fp16_silu_gate_core.sv",
    "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp",
)
COMPILE_ARGV = [
    "make",
    "--no-print-directory",
    "model24-rtl-layer-compile",
    "MODEL24_RTL_LAYER_INDEX=0",
    "MODEL24_RTL_ACCURATE_SILU=1",
]
EXPECTED_BINARY_RELATIVE = Path(
    "build/model24_rtl_cascade/compiled/layer0/obj_dir/"
    "Vace3_decoder_layer0_token_engine"
)
R14_NONCE = "03820ca8ec7835ca"
R15_NONCE = "f0742dfe9b4caa45"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write(path: Path, payload: bytes, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)


def writable_tree(root: Path) -> None:
    root.chmod(0o700)
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
        else:
            path.chmod(0o700 if path.stat().st_mode & 0o111 else 0o600)


def readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o500)
        else:
            path.chmod(0o500 if path.stat().st_mode & 0o111 else 0o400)
    root.chmod(0o500)


def tree_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        require(not path.is_symlink(), f"symlink forbidden in sealed tree: {path}")
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            records.append({"path": relative, "kind": "directory", "mode": mode})
        elif path.is_file():
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": mode,
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
        else:
            raise SystemExit(f"unsupported sealed-tree entry: {path}")
    return records


def tree_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(records)).hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transform_r15_text(payload: str) -> str:
    return payload.replace("r15", "r16")


def bootstrap_r16(nonce: str) -> Path:
    for path in (R15_BUILDER, R15_VALIDATOR, R15_LIFECYCLE, R15_CONTRACT, R16_VALIDATOR):
        require(path.is_file(), f"required r16 bootstrap input missing: {path}")
    with tempfile.TemporaryDirectory(prefix="ace3-model24-r16-bootstrap-") as scratch_name:
        scratch = Path(scratch_name)
        validator = scratch / "validate-base-package.py"
        lifecycle = scratch / "lifecycle.py"
        contract = scratch / "launch-contract.json"
        builder = scratch / "repository-builder.py"
        write(
            validator,
            transform_r15_text(R15_VALIDATOR.read_text(encoding="ascii")).encode("ascii"),
            0o500,
        )
        write(
            lifecycle,
            transform_r15_text(R15_LIFECYCLE.read_text(encoding="ascii")).encode("ascii"),
            0o500,
        )
        write(
            contract,
            transform_r15_text(R15_CONTRACT.read_text(encoding="ascii")).encode("ascii"),
        )
        builder_text = transform_r15_text(R15_BUILDER.read_text(encoding="ascii"))
        builder_text = builder_text.replace(
            'CONTRACT = REPOSITORY / "ace3/contracts/model24_r16_durable_launch_contract.json"',
            f"CONTRACT = Path({str(contract)!r})",
        ).replace(
            'VALIDATOR = REPOSITORY / "ace3/model/validate_model24_r16_launch_package.py"',
            f"VALIDATOR = Path({str(validator)!r})",
        ).replace(
            'LIFECYCLE = REPOSITORY / "ace3/model/model24_r16_lifecycle.py"',
            f"LIFECYCLE = Path({str(lifecycle)!r})",
        )
        write(builder, builder_text.encode("ascii"), 0o500)
        module = load_module(builder, "model24_r16_bootstrap_builder")
        result = module.prepare(nonce)
        require(
            result["status"] == "SEALED_AWAITING_INDEPENDENT_L2_REVIEW",
            "r16 lifecycle bootstrap did not seal",
        )
    return Path(f"/home/argustest/ace3-model24-r16-prep-20260829-{nonce}")


def git_blob(relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{ACCEPTED_COMMIT}:{relative}"],
        cwd=REPOSITORY,
        capture_output=True,
        check=False,
    )
    require(result.returncode == 0, f"accepted source unavailable: {relative}")
    return result.stdout


def authenticated_overlays(base: Path) -> list[dict[str, Any]]:
    package = base / "package"
    source = base / "source"
    declared = {
        record["path"]: record["successor_sha256"]
        for record in load_json(V2_PACKAGE / "source-overlays.json")["records"]
    }
    require(tuple(declared) == OVERLAY_PATHS, "v2 overlay inventory differs")
    records: list[dict[str, Any]] = []
    for relative in OVERLAY_PATHS:
        accepted_sha256 = hashlib.sha256(git_blob(relative)).hexdigest()
        v2_path = V2 / "source" / relative
        v2_sha256 = digest(v2_path)
        require(v2_sha256 == declared[relative], f"v2 overlay hash differs: {relative}")
        require(accepted_sha256 != v2_sha256, f"v2 overlay is not a change: {relative}")
        target = source / relative
        shutil.copy2(v2_path, target)
        materialized_sha256 = digest(target)
        require(materialized_sha256 == v2_sha256, f"materialized overlay differs: {relative}")
        records.append(
            {
                "path": relative,
                "accepted_sha256": accepted_sha256,
                "v2_declared_sha256": declared[relative],
                "v2_source_sha256": v2_sha256,
                "materialized_sha256": materialized_sha256,
                "accepted_differs": True,
                "v2_matches_materialized": True,
                "comparison": "PASS",
            }
        )
    write(
        package / "source-overlays.json",
        canonical_json(
            {
                "schema_version": 1,
                "kind": "ace3_model24_r16_exact_source_overlays",
                "base_commit": ACCEPTED_COMMIT,
                "overlay_set_exact": True,
                "records": records,
            }
        ),
    )
    return records


def extract_cascade_compile_argv(source: Path) -> list[str]:
    controller = source / "ace3/model/controller_model24_rtl_cascade.py"
    tree = ast.parse(controller.read_text(encoding="utf-8"), filename=str(controller))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "run_command":
            continue
        argument = node.args[0]
        if not isinstance(argument, ast.List):
            continue
        values: list[str] = []
        for item in argument.elts:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                values.append(item.value)
            elif isinstance(item, ast.JoinedStr):
                text = ""
                for part in item.values:
                    if isinstance(part, ast.Constant):
                        text += str(part.value)
                    elif isinstance(part, ast.FormattedValue):
                        text += "{layer_id}"
                values.append(text)
            else:
                values = []
                break
        if "model24-rtl-layer-compile" in values:
            return [
                value.replace("{layer_id}", "0")
                for value in values
            ]
    raise SystemExit("actual cascade compile command was not found")


def capture(command: list[str], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, cwd=cwd, capture_output=True, check=False)


def production_preflight(base: Path, source_tree_hash: str) -> dict[str, Any]:
    package = base / "package"
    source = base / "source"
    evidence = package / "evidence/production-preflight"
    evidence.mkdir(parents=True)
    cascade_argv = extract_cascade_compile_argv(source)
    require(cascade_argv == COMPILE_ARGV, "actual cascade command differs from r16 compile contract")
    dry_runs: dict[int, subprocess.CompletedProcess[bytes]] = {}
    for layer in (0, 23):
        command = [
            "make",
            "--no-print-directory",
            "--dry-run",
            "model24-rtl-layer-compile",
            f"MODEL24_RTL_LAYER_INDEX={layer}",
            "MODEL24_RTL_ACCURATE_SILU=1",
        ]
        completed = capture(command, source)
        require(completed.returncode == 0, f"layer {layer} Make target does not resolve")
        dry_runs[layer] = completed
        write(evidence / f"make-layer{layer}.stdout", completed.stdout)
        write(evidence / f"make-layer{layer}.stderr", completed.stderr)

    build_root = source / "build"
    if build_root.exists():
        shutil.rmtree(build_root)
    started_ns = time.time_ns()
    completed = capture(COMPILE_ARGV, source)
    write(evidence / "compile.stdout", completed.stdout)
    write(evidence / "compile.stderr", completed.stderr)
    binary = source / EXPECTED_BINARY_RELATIVE
    require(completed.returncode == 0, "fresh r16 layer-0 compile failed")
    require(binary.is_file() and binary.stat().st_mode & stat.S_IXUSR, "fresh r16 binary missing")
    require(binary.stat().st_mtime_ns >= started_ns, "compiled binary predates r16 compile")
    compiled_mtime_ns = binary.stat().st_mtime_ns
    binary_copy = evidence / binary.name
    shutil.copy2(binary, binary_copy)
    binary_copy.chmod(0o500)
    shutil.rmtree(build_root)
    result = {
        "schema_version": 1,
        "kind": "ace3_model24_r16_production_preflight",
        "status": "PASS",
        "source": str(source),
        "source_tree_sha256": source_tree_hash,
        "cascade_command_source": "ace3/model/controller_model24_rtl_cascade.py",
        "cascade_layer0_argv": cascade_argv,
        "compile_argv": COMPILE_ARGV,
        "compile_exit_code": completed.returncode,
        "accurate_silu": 1,
        "layer_index": 0,
        "make_target_resolution": {
            "layer0_exit_code": dry_runs[0].returncode,
            "layer23_exit_code": dry_runs[23].returncode,
        },
        "binary": {
            "name": binary_copy.name,
            "bytes": binary_copy.stat().st_size,
            "sha256": digest(binary_copy),
            "fresh_compile_started_ns": started_ns,
            "compiled_mtime_ns": compiled_mtime_ns,
        },
        "evidence_hashes": {
            name: digest(evidence / name)
            for name in (
                "compile.stdout",
                "compile.stderr",
                "make-layer0.stdout",
                "make-layer23.stdout",
            )
        },
        "model24_invocations": 0,
        "launcher_invocations": 0,
        "authority_invocations": 0,
    }
    write(evidence / "result.json", canonical_json(result))
    return result


def ancestry_artifacts(revision: int, nonce: str) -> dict[str, Path]:
    task_id = f"ace3-model24-r{revision}-42c895c-{nonce}"
    terminal = Path(f"/home/argustest/ace3-model24-r{revision}-terminal-20260829-{nonce}")
    registry = terminal / ".argus_subagents"
    receipt = registry / f"{task_id}.json"
    logs = registry / f"{task_id}_logs"
    exit_codes = list(logs.glob("exit_code.*"))
    require(len(exit_codes) == 1, f"r{revision} exit-code evidence is ambiguous")
    authority = Path(f"/home/argustest/ace3-model24-r{revision}-authority-20260829-{nonce}.json")
    return {
        "package.json": Path(f"/home/argustest/ace3-model24-r{revision}-prep-20260829-{nonce}/package/package.json"),
        "seal.json": Path(f"/home/argustest/ace3-model24-r{revision}-prep-20260829-{nonce}/package/seal.json"),
        "review.json": Path(f"/home/argustest/ace3-model24-r{revision}-review-20260829-{nonce}/review.json"),
        "authority.json": authority,
        "authority-consumed.json": Path(str(authority) + ".consumed"),
        "launch-terminal.json": Path(f"/home/argustest/ace3-model24-r{revision}-output-20260829-{nonce}/launch-terminal.json"),
        "terminal-manifest.json": terminal / "manifest.json",
        "runner-receipt.json": receipt,
        "stdout.log": logs / "stdout.log",
        "stderr.log": logs / "stderr.log",
        "exit-code.txt": exit_codes[0],
    }


def copy_unauthorized_ancestry(package: Path, revision: int, nonce: str) -> dict[str, Any]:
    artifacts = ancestry_artifacts(revision, nonce)
    root = package / f"provenance/r{revision}-unauthorized"
    root.mkdir(parents=True)
    hashes: dict[str, str] = {}
    original_paths: dict[str, str] = {}
    for name, source in artifacts.items():
        require(source.is_file() and not source.is_symlink(), f"r{revision} ancestry missing: {source}")
        shutil.copy2(source, root / name)
        (root / name).chmod(0o400)
        hashes[name] = digest(root / name)
        original_paths[name] = str(source)
        require(hashes[name] == digest(source), f"r{revision} ancestry copy differs: {name}")
    classification = {
        "schema_version": 1,
        "kind": f"ace3_model24_r{revision}_terminal_ancestry_classification",
        "classification": "UNAUTHORIZED_NO_MANAGER_DIRECTIVE",
        "manager_directive_present": False,
        "execution_authorized": False,
        "historical_execution_invocations": 1,
        "review_reusable": False,
        "authority_reusable": False,
        "copied_byte_for_byte": True,
        "original_paths": original_paths,
        "hashes": hashes,
    }
    write(root / "classification.json", canonical_json(classification))
    return classification


def extend_negative_tests(builder: Path) -> None:
    source = builder.read_text(encoding="ascii")
    marker = '    cases.extend(path_order["negative_cases"])\n    require(\n'
    require(marker in source, "r16 negative-suite extension point missing")
    extension = '''    cases.extend(path_order["negative_cases"])

    _expect_rejection(
        lambda: validator.reject_external_wrapper(True),
        "external wrapper",
        "external-wrapper",
    )
    cases.append("external-wrapper")

    wrong_overlay = validator.load_json(package / "source-overlays.json")
    wrong_overlay["records"][0]["v2_source_sha256"] = "0" * 64
    _expect_rejection(
        lambda: validator.validate_overlay_document(package, wrong_overlay),
        "overlay independent comparison",
        "wrong-overlay",
    )
    cases.append("wrong-overlay")

    makefile_text = (package.parent / "source/Makefile").read_text(encoding="utf-8")
    _expect_rejection(
        lambda: validator.validate_make_target_text(
            makefile_text.replace("model24-rtl-layer-compile:", "removed-target:")
        ),
        "compile target is missing",
        "missing-make-target",
    )
    cases.append("missing-make-target")

    with tempfile.TemporaryDirectory(
        prefix="ace3-model24-r16-binary-negative-"
    ) as binary_scratch_name:
        substituted_binary = Path(binary_scratch_name) / "substituted-binary"
        substituted_binary.write_bytes(b"not the compiled binary")
        substituted_binary.chmod(0o500)
        preflight = validator.load_json(
            package / "evidence/production-preflight/result.json"
        )
        _expect_rejection(
            lambda: validator.validate_compiled_binary(
                package, preflight, substituted_binary
            ),
            "stale or substituted",
            "stale-or-substituted-compiled-binary",
        )
    cases.append("stale-or-substituted-compiled-binary")

    _expect_rejection(
        lambda: validator.validate_predecessor_reuse(
            "/home/argustest/ace3-model24-r15-review-20260829-predecessor/review.json",
            "/home/argustest/ace3-model24-r15-authority-20260829-predecessor.json",
        ),
        "predecessor review or authority reuse",
        "predecessor-review-or-authority-reuse",
    )
    cases.append("predecessor-review-or-authority-reuse")

    require(
'''
    builder.write_text(source.replace(marker, extension), encoding="ascii")
    builder.chmod(0o400)


def materialize_full_package(base: Path) -> dict[str, Any]:
    package = base / "package"
    source = base / "source"
    validation = base / "validation"
    preparation = base / "preparation"
    writable_tree(base)
    baseline = load_json(package / "source-tree.json")
    write(
        package / "provenance/accepted-source-tree.json",
        canonical_json(baseline),
    )
    for name, source_path in V2_SOURCE_FILES.items():
        require(digest(source_path) == V2_HASHES[name], f"supplied v2 hash mismatch: {name}")
        target = package / "provenance/v2-compile-source" / name
        write(target, source_path.read_bytes(), 0o500 if name == "validate-package.py" else 0o400)
    overlays = authenticated_overlays(base)
    copy_unauthorized_ancestry(package, 14, R14_NONCE)
    copy_unauthorized_ancestry(package, 15, R15_NONCE)

    readonly_tree(source)
    source_records = tree_records(source)
    source_hash = tree_sha256(source_records)
    writable_tree(source)
    preflight = production_preflight(base, source_hash)
    readonly_tree(source)
    require(tree_records(source) == source_records, "production preflight changed sealed source")
    source_manifest = {
        "schema_version": 1,
        "kind": "ace3_model24_r16_materialized_source_tree",
        "base_commit": ACCEPTED_COMMIT,
        "base_tree_sha256": baseline["tree_sha256"],
        "overlay_paths": list(OVERLAY_PATHS),
        "records": source_records,
        "tree_sha256": source_hash,
    }
    write(package / "source-tree.json", canonical_json(source_manifest))

    base_validator = package / "validate-package.py"
    base_validator.rename(package / "validate-base-package.py")
    write(package / "validate-package.py", R16_VALIDATOR.read_bytes(), 0o500)
    extend_negative_tests(package / "ancestry/repository-builder.py")

    contract = load_json(package / "launch-contract.json")
    validator = load_module(package / "validate-package.py", "model24_r16_contract_validator")
    contract["review"]["required_negative_cases"] = list(validator.REQUIRED_NEGATIVE_CASES)
    contract["review"]["required_bound_hashes"] = list(validator.REQUIRED_REVIEW_HASHES)
    contract["source_materialization"] = {
        "accepted_commit": ACCEPTED_COMMIT,
        "v2_interpretation": "compile_evidence_only",
        "external_wrapper": False,
        "overlay_paths": list(OVERLAY_PATHS),
    }
    contract["production_preflight"] = {
        "required_before_model24": True,
        "layer0_compile_required": True,
        "accurate_silu": 1,
        "cascade_make_target_resolution_required": True,
        "fresh_compiled_binary_authentication_required": True,
    }
    contract["ancestry"] = {
        "r12_terminal_preserved": True,
        "r13_terminal_preserved": True,
        "r14_execution_authorized": False,
        "r15_execution_authorized": False,
        "predecessor_review_reusable": False,
        "predecessor_authority_reusable": False,
    }
    write(package / "launch-contract.json", canonical_json(contract))

    manifest = load_json(package / "package.json")
    manifest["provenance"]["source"].update(
        {
            "materialization": "git archive plus exact authenticated v2 overlays",
            "tree_sha256": source_hash,
            "overlay_paths": list(OVERLAY_PATHS),
        }
    )
    manifest["provenance"]["v2_compile_source"] = {
        "candidate": str(V2),
        "interpretation": "compile_evidence_only",
        "external_wrapper": False,
        "hashes": V2_HASHES,
        "source_tree_sha256": "b5228db8d3513ab29059bc55d446186fa8440f0cfc2283da28c61e1fe9eb156c",
    }
    manifest["provenance"]["source_overlays"] = {
        "paths": list(OVERLAY_PATHS),
        "records": overlays,
        "manifest_sha256": digest(package / "source-overlays.json"),
    }
    manifest["provenance"]["r14_terminal_ancestry"] = {
        "classification": "UNAUTHORIZED_NO_MANAGER_DIRECTIVE",
        "classification_sha256": digest(package / "provenance/r14-unauthorized/classification.json"),
    }
    manifest["provenance"]["r15_terminal_ancestry"] = {
        "classification": "UNAUTHORIZED_NO_MANAGER_DIRECTIVE",
        "classification_sha256": digest(package / "provenance/r15-unauthorized/classification.json"),
    }
    manifest["production_preflight"] = {
        "status": "PASS",
        "result_sha256": digest(package / "evidence/production-preflight/result.json"),
        "compile_argv": COMPILE_ARGV,
        "binary_sha256": preflight["binary"]["sha256"],
        "model24_invocations": 0,
    }
    manifest["review_policy"]["predecessor_review_reusable"] = False
    manifest["review_policy"]["predecessor_authority_reusable"] = False
    write(package / "package.json", canonical_json(manifest))

    paths = validator.expected_paths(package, manifest)
    authority_schema = {
        "schema_version": 1,
        "kind": "ace3_model24_r16_exact_manager_authority_schema",
        "document_kind": "ace3_model24_r16_manager_exactly_once_authority",
        "additional_fields_allowed": False,
        "required_exact_fields": [
            "schema_version",
            "kind",
            "nonce",
            "task_id",
            "review_path",
            "review_sha256",
            *validator.REQUIRED_REVIEW_HASHES,
            "submitter_cwd",
            "child_cwd",
            "submission_argv",
            "launch_argv",
            "durable_command",
            "terminal_root",
            "registry_root",
            "receipt_path",
            "stdout_log",
            "stderr_log",
            "terminal_manifest",
            "output_namespace",
            "simulation_namespace",
            "payload_output_namespace",
            "execution_cardinality",
            "manager_exactly_once_directive",
            "retry",
            "replay",
            "resume",
            "watcher",
        ],
        "canonical_paths": {
            "review_path": str(paths["review"] / "review.json"),
            "authority_path": str(paths["authority"]),
            "authority_consumed_path": str(paths["authority_consumed"]),
            "terminal_root": str(paths["terminal_root"]),
            "registry_root": str(paths["registry_root"]),
            "receipt_path": str(paths["receipt"]),
            "stdout_log": str(paths["stdout_log"]),
            "stderr_log": str(paths["stderr_log"]),
            "terminal_manifest": str(paths["terminal_manifest"]),
            "output_namespace": str(paths["output"]),
            "simulation_namespace": str(paths["simulation_dir"]),
            "payload_output_namespace": str(paths["payload_output_dir"]),
        },
        "exact_policy": {
            "execution_cardinality": 1,
            "manager_exactly_once_directive": True,
            "retry": False,
            "replay": False,
            "resume": False,
            "watcher": False,
        },
    }
    write(package / "authority-schema.json", canonical_json(authority_schema))

    review_request = load_json(package / "review-request.json")
    review_request["required_bound_hashes"] = list(validator.REQUIRED_REVIEW_HASHES)
    review_request["required_test_results"]["negative_cases"] = list(
        validator.REQUIRED_NEGATIVE_CASES
    )
    review_request.update(
        {
            "package_manifest_sha256": digest(package / "package.json"),
            "validator_sha256": digest(package / "validate-package.py"),
            "base_validator_sha256": digest(package / "validate-base-package.py"),
            "launch_contract_sha256": digest(package / "launch-contract.json"),
            "source_tree_sha256": digest(package / "source-tree.json"),
            "source_overlays_sha256": digest(package / "source-overlays.json"),
            "authority_schema_sha256": digest(package / "authority-schema.json"),
            "production_preflight_sha256": digest(
                package / "evidence/production-preflight/result.json"
            ),
            "v2_required_hashes": V2_HASHES,
            "v2_interpretation": "compile_evidence_only",
            "external_wrapper": False,
            "r14_execution_authorized": False,
            "r15_execution_authorized": False,
            "predecessor_review_reusable": False,
            "predecessor_authority_reusable": False,
        }
    )
    write(package / "review-request.json", canonical_json(review_request))

    seal_path = package / "seal.json"
    seal_path.unlink()
    readonly_tree(package)
    package.chmod(0o700)
    package_records = tree_records(package)
    seal = {
        "schema_version": 1,
        "kind": "ace3_model24_r16_package_seal",
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "package_manifest_sha256": digest(package / "package.json"),
        "review_request_sha256": digest(package / "review-request.json"),
        "source_tree_sha256": source_hash,
        "source_tree_manifest_sha256": digest(package / "source-tree.json"),
        "source_overlays_sha256": digest(package / "source-overlays.json"),
        "production_preflight_sha256": digest(
            package / "evidence/production-preflight/result.json"
        ),
        "v2_bound_hashes": V2_HASHES,
        "package_entries": package_records,
        "zero_state": True,
    }
    write(seal_path, canonical_json(seal))
    package.chmod(0o500)

    if validation.exists():
        shutil.rmtree(validation)
    validation.mkdir()
    commands = {
        "package-validation": [
            "/home/argustest/miniconda3/bin/python3",
            "-B",
            str(package / "validate-package.py"),
            "--package",
            str(package),
            "--mode",
            "package",
        ],
        "negative-tests": [
            "/home/argustest/miniconda3/bin/python3",
            "-B",
            str(package / "test-launch-contract.py"),
        ],
        "controller-entry-integration": [
            "/home/argustest/miniconda3/bin/python3",
            "-B",
            str(package / "test-controller-entry.py"),
        ],
    }
    outcomes: dict[str, dict[str, Any]] = {}
    for name, command in commands.items():
        completed = capture(command, base)
        write(validation / f"{name}.stdout", completed.stdout)
        write(validation / f"{name}.stderr", completed.stderr)
        write(validation / f"{name}.status", f"{completed.returncode}\n".encode("ascii"))
        require(completed.returncode == 0, f"{name} failed: {completed.stderr.decode(errors='replace')}")
        outcomes[name] = {
            "argv": command,
            "exit_code": completed.returncode,
            "stdout_sha256": digest(validation / f"{name}.stdout"),
            "stderr_sha256": digest(validation / f"{name}.stderr"),
        }

    forbidden = (
        "review",
        "authority",
        "authority_consumed",
        "output",
        "simulation_dir",
        "payload_output_dir",
        "terminal_root",
        "receipt",
        "stdout_log",
        "stderr_log",
        "terminal_manifest",
    )
    zero_state = {f"{name}_absent": not paths[name].exists() for name in forbidden}
    require(all(zero_state.values()), "canonical r16 review or execution namespace was created")
    zero_state.update(
        {
            "model24_invocations": 0,
            "launcher_invocations": 0,
            "authority_invocations": 0,
            "output_creations": 0,
        }
    )
    write(validation / "zero-state.json", canonical_json(zero_state))
    readonly_tree(validation)
    validation.chmod(0o700)
    validation_records = tree_records(validation)
    validation_seal = {
        "schema_version": 1,
        "kind": "ace3_model24_r16_inert_validation_seal",
        "package_seal_sha256": digest(seal_path),
        "negative_case_count": len(validator.REQUIRED_NEGATIVE_CASES),
        "positive_fixture_count": len(validator.REQUIRED_POSITIVE_FIXTURES),
        "outcomes": outcomes,
        "records": validation_records,
        "model24_invocations": 0,
        "launcher_invocations": 0,
        "authority_invocations": 0,
    }
    write(validation / "seal.json", canonical_json(validation_seal))
    validation.chmod(0o500)

    result = {
        "status": "SEALED_AWAITING_INDEPENDENT_L2_REVIEW",
        "candidate": str(base),
        "package": str(package),
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "package_manifest_sha256": digest(package / "package.json"),
        "package_seal_sha256": digest(seal_path),
        "review_request_sha256": digest(package / "review-request.json"),
        "source_tree_manifest_sha256": digest(package / "source-tree.json"),
        "source_tree_sha256": source_hash,
        "source_overlays_sha256": digest(package / "source-overlays.json"),
        "production_preflight_sha256": digest(
            package / "evidence/production-preflight/result.json"
        ),
        "validation_seal_sha256": digest(validation / "seal.json"),
        "negative_case_count": len(validator.REQUIRED_NEGATIVE_CASES),
        "positive_fixture_count": len(validator.REQUIRED_POSITIVE_FIXTURES),
        "production_preflight": "PASS",
        "v2_interpretation": "compile_evidence_only",
        "external_wrapper": False,
        "r14_execution_authorized": False,
        "r15_execution_authorized": False,
        "canonical_zero_state": zero_state,
        "execution_authority_withheld": True,
        "model24_invocations": 0,
    }
    if preparation.exists():
        shutil.rmtree(preparation)
    preparation.mkdir()
    write(preparation / "result.json", canonical_json(result))
    readonly_tree(preparation)
    base.chmod(0o500)
    return result


def prepare(nonce: str) -> dict[str, Any]:
    require(len(nonce) == 16 and all(character in "0123456789abcdef" for character in nonce), "nonce must be 16 lowercase hex characters")
    require(REPOSITORY.resolve() == Path.cwd().resolve(), "run from the ACE-3 worktree")
    for name, path in V2_SOURCE_FILES.items():
        require(path.is_file() and not path.is_symlink(), f"v2 input missing: {name}")
        require(digest(path) == V2_HASHES[name], f"v2 input hash mismatch: {name}")
    validated_v2 = subprocess.run(
        [str(V2_PACKAGE / "validate-package.py"), "--package", str(V2_PACKAGE)],
        capture_output=True,
        check=False,
    )
    require(validated_v2.returncode == 0, "sealed v2 validator failed")
    base = bootstrap_r16(nonce)
    return materialize_full_package(base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args()
    print(canonical_json(prepare(args.nonce)).decode("ascii"), end="")


if __name__ == "__main__":
    main()
