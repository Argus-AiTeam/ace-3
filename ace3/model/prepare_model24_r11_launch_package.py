#!/usr/bin/env python3
"""Construct and inertly validate a fresh Model24 r11 launch candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any
import unittest


REPOSITORY = Path("/home/argustest/ace3-argus")
PARENT = Path("/home/argustest/ace3-model24-r10-prep-20260829-ba953828cb457c4d")
PARENT_REVIEW = Path("/home/argustest/ace3-model24-r10-review-20260829-ba953828cb457c4d/review.json")
PARENT_MANIFEST_SHA256 = "3737400b2bcf839098e4cbd9bb1d392f5df6b383ceec3943ca603ce352c7c850"
PARENT_SEAL_SHA256 = "90b229697c77a0505945396533c4a74de8b9ec426ccd351fa3a7c4d5576580ff"
PARENT_REVIEW_SHA256 = "6efc30359df96af1a72f8f156416f4838069c0bedaf28161e42894bef1c46d05"
CHECKPOINT = Path("/home/argustest/ace3-argus/build/model24_rtl_cascade/checkpoint/model.safetensors")
CHECKPOINT_BYTES = 730652248
CHECKPOINT_SHA256 = "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
MODEL_PYTHON = Path("/home/argustest/miniconda3/bin/python3")
ARGUS_PYTHON = Path("/home/argustest/argustest2/argus-skill-latest/.venv/bin/python")
CONTRACT = REPOSITORY / "ace3/contracts/model24_r11_durable_launch_contract.json"
VALIDATOR = REPOSITORY / "ace3/model/validate_model24_r11_launch_package.py"
LAUNCHER = REPOSITORY / "ace3/model/model24_r11_launch.py"


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


def write_new(path: Path, payload: bytes, mode: int) -> None:
    require(path.parent.is_dir(), f"write parent is absent: {path.parent}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(mode)


def mutable_tree_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        require(not path.is_symlink(), f"symlink is forbidden in candidate: {path}")
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
    return records


def readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o500)
        else:
            executable = path.stat().st_mode & 0o111
            path.chmod(0o500 if executable else 0o400)
    root.chmod(0o500)


def snapshot_named_paths(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    return {str(path): mutable_tree_records(path) for path in paths}


def load_validator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("model24_r11_validator", path)
    require(spec is not None and spec.loader is not None, "validator import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_review(
    validator: Any,
    package: Path,
    manifest: dict[str, Any],
    review_root: Path,
    **changes: Any,
) -> Path:
    review = {
        "schema_version": 1,
        "kind": "ace3_model24_r11_independent_l2_review",
        "candidate": str(package.parent),
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "verdict": "ACCEPT",
        "producer_role": "reviewer",
        "active_role": "reviewer",
        "reviewer_level": "L2",
        "reviewer_identity": "role:independent-l2-reviewer",
        "independent": True,
        "preparation_participation": False,
        "execution_invocations": 0,
        "authority_created": False,
        "manager_may_issue_execution_authority": True,
        "bound_hashes": validator.package_review_hashes(package),
        "test_results": validator.passing_review_test_results(),
        "zero_state_observation": validator.clean_zero_state_observation(),
        "ba953_ancestry_statement": validator.BA953_ANCESTRY_STATEMENT,
    }
    review.update(changes)
    review_root.mkdir()
    path = review_root / "review.json"
    path.write_bytes(canonical_json(review))
    path.chmod(0o400)
    return path


def run_negative_tests(package: Path) -> dict[str, Any]:
    validator = load_validator(package / "validate-package.py")
    manifest = validator.load_json(package / "package.json")
    cases: list[str] = []

    with tempfile.TemporaryDirectory(prefix="ace3-model24-r11-inert-") as temporary:
        scratch = Path(temporary)
        original_namespaces = manifest["namespaces"]

        def review_manifest(review_root: Path) -> dict[str, Any]:
            value = copy.deepcopy(manifest)
            value["namespaces"]["review"] = str(review_root)
            return value

        positive_root = scratch / "positive-review"
        positive_manifest = review_manifest(positive_root)
        positive_review = write_review(
            validator, package, positive_manifest, positive_root
        )
        validator.validate_review_document(
            package, positive_manifest, positive_review
        )

        for name, changes, expected in (
            ("reviewer-producer-role", {"producer_role": "engineer"}, "producer_role"),
            ("reviewer-active-role", {"active_role": "engineer"}, "active_role"),
            ("reviewer-level", {"reviewer_level": "L1"}, "reviewer_level"),
            ("reviewer-preparation-participation", {"preparation_participation": True}, "preparation_participation"),
            ("reviewer-identity", {"reviewer_identity": manifest["review_policy"]["preparer_identity"]}, "not independent"),
        ):
            root = scratch / name
            local_manifest = review_manifest(root)
            review_path = write_review(validator, package, local_manifest, root, **changes)
            try:
                validator.validate_review_document(package, local_manifest, review_path)
            except SystemExit as error:
                require(expected in str(error), f"{name} rejected for wrong reason: {error}")
            else:
                raise SystemExit(f"negative probe unexpectedly accepted: {name}")
            cases.append(name)

        root = scratch / "reviewer-test-results"
        local_manifest = review_manifest(root)
        review_path = write_review(validator, package, local_manifest, root)
        review = validator.load_json(review_path)
        review["test_results"]["negative_cases"][0]["status"] = "FAIL"
        review_path.chmod(0o600)
        review_path.write_bytes(canonical_json(review))
        review_path.chmod(0o400)
        try:
            validator.validate_review_document(package, local_manifest, review_path)
        except SystemExit as error:
            require("ACCEPT review" in str(error), f"review evidence rejected for wrong reason: {error}")
        else:
            raise SystemExit("negative probe unexpectedly accepted: reviewer-test-results")
        cases.append("reviewer-test-results")

        root = scratch / "provenance-hash"
        local_manifest = review_manifest(root)
        review_path = write_review(validator, package, local_manifest, root)
        review = validator.load_json(review_path)
        review["bound_hashes"]["validator_sha256"] = "0" * 64
        review_path.chmod(0o600)
        review_path.write_bytes(canonical_json(review))
        review_path.chmod(0o400)
        try:
            validator.validate_review_document(package, local_manifest, review_path)
        except SystemExit as error:
            require("six sealed artifacts" in str(error), f"provenance probe rejected for wrong reason: {error}")
        else:
            raise SystemExit("negative probe unexpectedly accepted: provenance-hash")
        cases.append("provenance-hash")

        root = scratch / "authority-binding"
        local_manifest = review_manifest(root)
        review_path = write_review(validator, package, local_manifest, root)
        authority = validator.expected_authority(package, local_manifest, review_path)
        authority["package_seal_sha256"] = "0" * 64
        authority_path = scratch / "authority.json"
        authority_path.write_bytes(canonical_json(authority))
        authority_path.chmod(0o400)
        try:
            validator.validate_authority_document(package, local_manifest, review_path, authority_path)
        except SystemExit as error:
            require("exactly bind" in str(error), f"authority probe rejected for wrong reason: {error}")
        else:
            raise SystemExit("negative probe unexpectedly accepted: authority-binding")
        cases.append("authority-binding")

        for semantic in ("retry", "replay", "resume", "watcher"):
            mutated = scratch / f"resealed-{semantic}"
            shutil.copytree(package, mutated)
            mutated.chmod(0o700)
            manifest_path = mutated / "package.json"
            manifest_path.chmod(0o600)
            local_manifest = validator.load_json(manifest_path)
            local_manifest["execution"][semantic] = True
            manifest_path.write_bytes(canonical_json(local_manifest))
            manifest_path.chmod(0o400)
            seal_path = mutated / "seal.json"
            seal_path.chmod(0o600)
            seal = validator.load_json(seal_path)
            seal["package_manifest_sha256"] = digest(manifest_path)
            seal["package_entries"] = [
                record
                for record in mutable_tree_records(mutated)
                if record["path"] != "seal.json"
            ]
            seal_path.write_bytes(canonical_json(seal))
            seal_path.chmod(0o400)
            mutated.chmod(0o500)
            validator.validate_package_seal(mutated)
            try:
                validator.validate_contract(mutated, local_manifest)
            except SystemExit as error:
                require("policy is open" in str(error), f"{semantic} probe rejected for wrong reason: {error}")
            else:
                raise SystemExit(f"negative probe unexpectedly accepted: {semantic}")
            cases.append(semantic)

        local_manifest = copy.deepcopy(manifest)
        local_manifest["execution"]["exact_argv"] = local_manifest["execution"]["exact_argv"][:-1]
        try:
            validator.validate_contract(package, local_manifest)
        except SystemExit as error:
            require("argv differs" in str(error), f"wrong-argv probe rejected for wrong reason: {error}")
        else:
            raise SystemExit("negative probe unexpectedly accepted: wrong-argv")
        cases.append("wrong-argv")

        local_manifest = copy.deepcopy(manifest)
        local_manifest["execution"]["exact_cwd"] = str(scratch)
        try:
            validator.validate_contract(package, local_manifest)
        except SystemExit as error:
            require("cwd differs" in str(error), f"launch-contract probe rejected for wrong reason: {error}")
        else:
            raise SystemExit("negative probe unexpectedly accepted: launch-contract")
        cases.append("launch-contract")

        receipt_root = scratch / "receipt-state"
        log_dir = receipt_root / f"{manifest['identity']['task_id']}_logs"
        log_dir.mkdir(parents=True)
        stdout = log_dir / "stdout.log"
        stderr = log_dir / "stderr.log"
        stdout.write_bytes(b"")
        stderr.write_bytes(b"")
        receipt = receipt_root / f"{manifest['identity']['task_id']}.json"
        local_manifest = copy.deepcopy(manifest)
        local_manifest["namespaces"].update({
            "receipt_namespace": str(receipt_root),
            "receipt": str(receipt),
            "log_dir": str(log_dir),
            "stdout_log": str(stdout),
            "stderr_log": str(stderr),
        })
        receipt.write_bytes(canonical_json({
            "state": "starting",
            "task_id": manifest["identity"]["task_id"],
            "run_id": manifest["identity"]["task_id"] + "-123456789",
            "description": "future independent launch",
            "command": manifest["execution"]["durable_command"],
            "mode": "direct",
            "run_dir": None,
            "cwd": manifest["execution"]["exact_cwd"],
            "submitted_at": time.time(),
            "submitter_pid": os.getpid(),
        }))
        validator.validate_receipt_document(local_manifest, receipt, stdout, stderr)
        valid_authority = validator.expected_authority(
            package, positive_manifest, positive_review
        )
        valid_authority_path = scratch / "positive-authority.json"
        valid_authority_path.write_bytes(canonical_json(valid_authority))
        valid_authority_path.chmod(0o400)
        validator.validate_authority_document(
            package,
            positive_manifest,
            positive_review,
            valid_authority_path,
        )
        canonical = validator.expected_paths(package, manifest)
        for name, supplied, expected in (
            ("wrong-receipt-path", (scratch / "wrong.json", canonical["stdout_log"], canonical["stderr_log"]), "wrong durable receipt"),
            ("wrong-stdout-path", (canonical["receipt"], scratch / "stdout.log", canonical["stderr_log"]), "wrong durable stdout"),
            ("wrong-stderr-path", (canonical["receipt"], canonical["stdout_log"], scratch / "stderr.log"), "wrong durable stderr"),
        ):
            try:
                validator.validate_runner_arguments(manifest, *supplied)
            except SystemExit as error:
                require(expected in str(error), f"{name} rejected for wrong reason: {error}")
            else:
                raise SystemExit(f"negative probe unexpectedly accepted: {name}")
            cases.append(name)
        receipt_value = validator.load_json(receipt)
        receipt_value["replay"] = True
        receipt.write_bytes(canonical_json(receipt_value))
        try:
            validator.validate_receipt_document(local_manifest, receipt, stdout, stderr)
        except SystemExit as error:
            require("receipt enables" in str(error), f"receipt replay rejected for wrong reason: {error}")
        else:
            raise SystemExit("negative probe unexpectedly accepted: receipt-replay")
        cases.append("receipt-replay")

        preexisting_paths = {
            name: scratch / "pre-existing-state" / name
            for name in (
                "authority",
                "authority_consumed",
                "output",
                "receipt_namespace",
                "review",
            )
        }
        preexisting_paths["output"].mkdir(parents=True)
        try:
            validator.validate_zero_state_paths(
                preexisting_paths, manifest["zero_state"], "package"
            )
        except SystemExit as error:
            require("absent output" in str(error), f"pre-existing-state probe rejected for wrong reason: {error}")
        else:
            raise SystemExit("negative probe unexpectedly accepted: pre-existing-state")
        cases.append("pre-existing-state")

        mutated = scratch / "mutated-package"
        shutil.copytree(package, mutated)
        mutated.chmod(0o700)
        target = mutated / "review-request.json"
        target.chmod(0o600)
        target.write_bytes(target.read_bytes() + b" ")
        try:
            validator.validate_package_seal(mutated)
        except SystemExit as error:
            require("differs from seal" in str(error), f"mutation probe rejected for wrong reason: {error}")
        else:
            raise SystemExit("negative probe unexpectedly accepted: package-mutation")
        cases.append("package-mutation")
        mutated_source = scratch / "mutated-source"
        shutil.copytree(package.parent / "source", mutated_source)
        source_target = next(path for path in mutated_source.rglob("*") if path.is_file())
        source_target.chmod(0o600)
        source_target.write_bytes(source_target.read_bytes() + b" ")
        source_records = validator.load_json(package / "source-tree.json")["records"]
        try:
            validator.validate_recorded_tree(
                mutated_source, source_records, require_read_only=True
            )
        except SystemExit as error:
            require("differs from manifest" in str(error), f"source mutation rejected for wrong reason: {error}")
        else:
            raise SystemExit("negative probe unexpectedly accepted: source-mutation")
        cases.append("source-mutation")
        manifest["namespaces"] = original_namespaces

    require(
        cases == list(validator.REQUIRED_NEGATIVE_CASES),
        "inert negative case order differs from the review evidence contract",
    )
    return {
        "status": "PASS",
        "kind": "ace3_model24_r11_inert_negative_tests",
        "case_count": len(cases),
        "cases": cases,
        "positive_fixtures": ["package", "review", "launch"],
        "execution_invocations": 0,
        "model_invocations": 0,
        "simulator_invocations": 0,
        "launcher_invocations": 0,
        "authority_creations": 0,
        "output_creations": 0,
        "receipt_creations": 0,
    }


def capture_result(validation: Path, stem: str, command: list[str], result: subprocess.CompletedProcess[bytes]) -> None:
    write_new(validation / f"{stem}.command.json", canonical_json(command), 0o400)
    write_new(validation / f"{stem}.stdout", result.stdout, 0o400)
    write_new(validation / f"{stem}.stderr", result.stderr, 0o400)
    write_new(validation / f"{stem}.status", f"{result.returncode}\n".encode("ascii"), 0o400)


def prepare(nonce: str) -> dict[str, Any]:
    base = Path(f"/home/argustest/ace3-model24-r11-prep-20260829-{nonce}")
    package = base / "package"
    source = base / "source"
    validation = base / "validation"
    review = Path(f"/home/argustest/ace3-model24-r11-review-20260829-{nonce}")
    authority = Path(f"/home/argustest/ace3-model24-r11-authority-20260829-{nonce}.json")
    output = Path(f"/home/argustest/ace3-model24-r11-output-20260829-{nonce}")
    task_id = f"ace3-model24-r11-42c895c-{nonce}"
    for path in (base, review, authority, Path(str(authority) + ".consumed"), output):
        require(not path.exists(), f"fresh r11 namespace already exists: {path}")
    require(PARENT.is_dir(), "ba953 parent candidate is unavailable")
    require(digest(PARENT / "package/package.json") == PARENT_MANIFEST_SHA256, "ba953 manifest changed")
    require(digest(PARENT / "package/seal.json") == PARENT_SEAL_SHA256, "ba953 seal changed")
    require(digest(PARENT_REVIEW) == PARENT_REVIEW_SHA256, "ba953 review changed")
    require(CHECKPOINT.stat().st_size == CHECKPOINT_BYTES, "checkpoint size changed")
    for path in (CONTRACT, VALIDATOR, LAUNCHER, MODEL_PYTHON, ARGUS_PYTHON):
        require(path.is_file(), f"required construction input missing: {path}")

    r10_paths = sorted(
        path for path in Path("/home/argustest").glob("ace3-model24-r10-*")
        if path.is_dir()
    )
    protected_before = snapshot_named_paths(r10_paths)

    base.mkdir(mode=0o700)
    package.mkdir(mode=0o700)
    validation.mkdir(mode=0o700)
    shutil.copytree(PARENT / "source", source, symlinks=True)
    ancestry = package / "ancestry"
    ancestry.mkdir()
    shutil.copytree(PARENT, ancestry / "ba953-candidate", symlinks=True)
    shutil.copytree(PARENT_REVIEW.parent, ancestry / "ba953-review", symlinks=True)
    write_new(ancestry / "repository-contract.json", CONTRACT.read_bytes(), 0o400)
    write_new(ancestry / "repository-builder.py", Path(__file__).read_bytes(), 0o400)

    write_new(package / "launch-contract.json", CONTRACT.read_bytes(), 0o400)
    write_new(package / "launch-once.py", LAUNCHER.read_bytes(), 0o500)
    write_new(package / "validate-package.py", VALIDATOR.read_bytes(), 0o500)

    readonly_tree(source)
    source_records = mutable_tree_records(source)
    source_tree_sha256 = hashlib.sha256(canonical_json(source_records)).hexdigest()
    source_manifest = {
        "schema_version": 1,
        "kind": "ace3_model24_r11_source_tree",
        "records": source_records,
        "tree_sha256": source_tree_sha256,
    }
    write_new(package / "source-tree.json", canonical_json(source_manifest), 0o400)

    receipt_root = base / ".argus_subagents"
    log_dir = receipt_root / f"{task_id}_logs"
    namespaces = {
        "review": str(review),
        "authority": str(authority),
        "authority_consumed": str(authority) + ".consumed",
        "output": str(output),
        "receipt_namespace": str(receipt_root),
        "receipt": str(receipt_root / f"{task_id}.json"),
        "log_dir": str(log_dir),
        "stdout_log": str(log_dir / "stdout.log"),
        "stderr_log": str(log_dir / "stderr.log"),
    }
    exact_argv = [
        str(MODEL_PYTHON),
        "-B",
        str(package / "launch-once.py"),
        "--authority",
        str(authority),
        "--review",
        str(review / "review.json"),
        "--receipt",
        namespaces["receipt"],
        "--stdout-log",
        namespaces["stdout_log"],
        "--stderr-log",
        namespaces["stderr_log"],
    ]
    payload_argv = [
        str(MODEL_PYTHON),
        "-B",
        str(source / "ace3/model/controller_model24_rtl_cascade.py"),
        "--repository-root",
        str(source),
        "--checkpoint",
        str(CHECKPOINT),
        "--tensor-map",
        str(source / "ace3/contracts/model24_tensor_map.json"),
        "--bindings",
        str(package / "ancestry/ba953-candidate/package/evidence/bindings.json"),
        "--simulation-dir",
        str(output / "controller-simulation"),
        "--output-dir",
        str(output / "rtl-cascade"),
        "--fresh",
    ]
    manifest = {
        "schema_version": 1,
        "kind": "ace3_model24_r11_durable_launch_package",
        "identity": {"candidate": str(base), "nonce": nonce, "task_id": task_id},
        "provenance": {
            "ba953_history": {
                "status": "NONCONFORMING_REJECTED_HISTORY_ONLY",
                "authoritative": False,
                "trust_root": False,
                "package_acceptance": False,
                "execution_authority": False,
                "external_parent_review_dependency": False,
                "package_manifest_sha256": PARENT_MANIFEST_SHA256,
                "package_seal_sha256": PARENT_SEAL_SHA256,
                "review_sha256": PARENT_REVIEW_SHA256,
                "source_commit": "42c895ce1e5fea00525e9f2f7fef66f0fbb8e118",
                "source_tree": "93f57e9e0c87c3f8638282475e12f8f5289b498b",
            },
            "checkpoint": {
                "path": str(CHECKPOINT),
                "bytes": CHECKPOINT_BYTES,
                "sha256": CHECKPOINT_SHA256,
            },
        },
        "review_policy": {
            "preparer_identity": "role:engineer",
            "required_reviewer_level": "L2",
            "required_independent": True,
            "exactly_one_future_verdict": True,
        },
        "execution": {
            "cardinality": 1,
            "current_invocation_count": 0,
            "durable_command": shlex.join(exact_argv),
            "exact_argv": exact_argv,
            "exact_cwd": str(base),
            "payload_argv": payload_argv,
            "payload_cwd": str(source),
            "retry": False,
            "replay": False,
            "resume": False,
            "watcher": False,
            "timeout_seconds": 18000,
        },
        "resources": {
            "environment": {
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            "memory_is_finite": True,
        },
        "namespaces": namespaces,
        "zero_state": {
            "authority_consumptions": 0,
            "authority_creations": 0,
            "durable_submissions": 0,
            "execution_invocations": 0,
            "launcher_invocations": 0,
            "model_invocations": 0,
            "output_creations": 0,
            "receipt_creations": 0,
            "simulator_invocations": 0,
        },
    }
    write_new(package / "package.json", canonical_json(manifest), 0o400)
    review_request = {
        "schema_version": 1,
        "kind": "ace3_model24_r11_independent_l2_review_request",
        "candidate": str(base),
        "nonce": nonce,
        "task_id": task_id,
        "review_namespace": str(review),
        "review_namespace_must_be_exclusive_created": True,
        "reviewer_level": "L2",
        "reviewer_must_be_independent": True,
        "producer_role": "reviewer",
        "active_role": "reviewer",
        "preparation_participation": False,
        "exactly_one_future_verdict": True,
        "required_verdicts": ["ACCEPT", "REJECT"],
        "required_bound_hashes": [
            "package_manifest_sha256",
            "package_seal_sha256",
            "review_request_sha256",
            "launcher_sha256",
            "validator_sha256",
            "launch_contract_sha256",
        ],
        "package_manifest_sha256": digest(package / "package.json"),
        "launcher_sha256": digest(package / "launch-once.py"),
        "validator_sha256": digest(package / "validate-package.py"),
        "launch_contract_sha256": digest(package / "launch-contract.json"),
        "source_tree_sha256": source_tree_sha256,
        "embedded_ba953_package_seal_sha256": PARENT_SEAL_SHA256,
        "embedded_ba953_review_sha256": PARENT_REVIEW_SHA256,
        "ba953_history_status": "NONCONFORMING_REJECTED_HISTORY_ONLY",
        "ba953_history_is_authoritative": False,
        "external_parent_review_dependency": False,
        "required_test_results": {
            "positive_fixtures": list(
                load_validator(VALIDATOR).REQUIRED_POSITIVE_FIXTURES
            ),
            "negative_cases": list(
                load_validator(VALIDATOR).REQUIRED_NEGATIVE_CASES
            ),
        },
        "required_zero_state_observation": load_validator(
            VALIDATOR
        ).clean_zero_state_observation(),
        "required_ba953_ancestry_statement": load_validator(
            VALIDATOR
        ).BA953_ANCESTRY_STATEMENT,
        "execution_authority_withheld": True,
        "execution_invocations": 0,
        "stage_transition": False,
    }
    write_new(package / "review-request.json", canonical_json(review_request), 0o400)

    test_source = (
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import importlib.util\n"
        "spec=importlib.util.spec_from_file_location('builder', "
        + repr(str(package / "ancestry/repository-builder.py"))
        + ")\n"
        "module=importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "result=module.run_negative_tests(Path(__file__).resolve().parent)\n"
        "print(module.canonical_json(result).decode('ascii'), end='')\n"
    ).encode("ascii")
    write_new(package / "test-launch-contract.py", test_source, 0o500)

    readonly_tree(package / "ancestry")
    package_records = mutable_tree_records(package)
    package_records = [record for record in package_records if record["path"] != "seal.json"]
    seal = {
        "schema_version": 1,
        "kind": "ace3_model24_r11_package_seal",
        "nonce": nonce,
        "task_id": task_id,
        "package_manifest_sha256": digest(package / "package.json"),
        "review_request_sha256": digest(package / "review-request.json"),
        "source_tree_sha256": source_tree_sha256,
        "package_entries": package_records,
        "zero_state": True,
    }
    write_new(package / "seal.json", canonical_json(seal), 0o400)
    readonly_tree(package)

    positive_command = [
        str(MODEL_PYTHON),
        "-B",
        str(package / "validate-package.py"),
        "--package",
        str(package),
        "--mode",
        "package",
    ]
    positive = subprocess.run(positive_command, cwd=base, capture_output=True, check=False)
    capture_result(validation, "positive-validation", positive_command, positive)
    require(positive.returncode == 0, positive.stderr.decode("utf-8", errors="replace"))

    negative_command = [str(MODEL_PYTHON), "-B", str(package / "test-launch-contract.py")]
    negative = subprocess.run(negative_command, cwd=base, capture_output=True, check=False)
    capture_result(validation, "negative-tests", negative_command, negative)
    require(negative.returncode == 0, negative.stderr.decode("utf-8", errors="replace"))

    protected_after = snapshot_named_paths(r10_paths)
    require(protected_after == protected_before, "r10 or ba953 evidence changed during r11 construction")
    for path in (review, authority, Path(str(authority) + ".consumed"), output, receipt_root):
        require(not path.exists(), f"forbidden execution/review state was created: {path}")
    validation_records = mutable_tree_records(validation)
    validation_seal = {
        "schema_version": 1,
        "kind": "ace3_model24_r11_inert_validation_seal",
        "package_seal_sha256": digest(package / "seal.json"),
        "records": validation_records,
        "execution_invocations": 0,
        "model_invocations": 0,
        "simulator_invocations": 0,
        "launcher_invocations": 0,
        "authority_creations": 0,
        "output_creations": 0,
        "receipt_creations": 0,
    }
    write_new(validation / "seal.json", canonical_json(validation_seal), 0o400)
    readonly_tree(validation)
    base.chmod(0o500)
    return {
        "status": "SEALED_AWAITING_INDEPENDENT_L2_REVIEW",
        "candidate": str(base),
        "package": str(package),
        "review_namespace": str(review),
        "task_id": task_id,
        "nonce": nonce,
        "package_manifest_sha256": digest(package / "package.json"),
        "package_seal_sha256": digest(package / "seal.json"),
        "review_request_sha256": digest(package / "review-request.json"),
        "validation_seal_sha256": digest(validation / "seal.json"),
        "review_namespace_absent": not review.exists(),
        "zero_state": True,
        "execution_authority_withheld": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", default=secrets.token_hex(8))
    args = parser.parse_args()
    require(
        len(args.nonce) == 16
        and all(character in "0123456789abcdef" for character in args.nonce),
        "nonce must be 16 lowercase hexadecimal characters",
    )
    print(json.dumps(prepare(args.nonce), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
