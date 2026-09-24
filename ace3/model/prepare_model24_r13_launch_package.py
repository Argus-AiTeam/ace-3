#!/usr/bin/env python3
"""Construct and inertly validate a fresh Model24 r13 launch candidate."""

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
import tarfile
import tempfile
import time
from typing import Any


REPOSITORY = Path("/home/argustest/ace3-argus")
COMMIT = "42c895ce1e5fea00525e9f2f7fef66f0fbb8e118"
TREE = "93f57e9e0c87c3f8638282475e12f8f5289b498b"
CHECKPOINT = REPOSITORY / "build/model24_rtl_cascade/checkpoint/model.safetensors"
CHECKPOINT_BYTES = 730652248
CHECKPOINT_SHA256 = "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
MODEL_PYTHON = Path("/home/argustest/miniconda3/bin/python3")
ARGUS_PYTHON = Path("/home/argustest/argustest2/argus-skill-latest/.venv/bin/python")
RUNNER_ROOT = Path(
    "/home/argustest/argustest2/argus-skill-latest/argus_skill/tools/subagent"
)
CONTRACT = REPOSITORY / "ace3/contracts/model24_r13_durable_launch_contract.json"
VALIDATOR = REPOSITORY / "ace3/model/validate_model24_r13_launch_package.py"
LIFECYCLE = REPOSITORY / "ace3/model/model24_r13_lifecycle.py"
R12_PACKAGE = Path(
    "/home/argustest/ace3-model24-r12-prep-20260829-d4e81c72a593bf06/package"
)
R12_PACKAGE_SEAL_SHA256 = "3d326b3c6c51d3b9ddfc8f292054c0a9ffa922adec8ef3a8e1a6082ee2e13c06"
R12_TERMINAL = Path(
    "/home/argustest/ace3-model24-r12-terminal-20260829-d4e81c72a593bf06"
)
R12_TERMINAL_HASHES = {
    "manifest.json": "0c4699aac58439bc10a180cfad63fd1875c8a1559ee2030ff237720477b2b2f5",
    "runner-receipt.json": "71cea3855a0aa57d33bfacc1cafa0f738fd909384f75340742c9978edaeeade1",
    "exit-code.txt": "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
}
BINDINGS_SOURCE = (
    R12_PACKAGE / "ancestry/ba953-candidate/package/evidence/bindings.json"
)


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
            path.chmod(0o500 if path.stat().st_mode & 0o111 else 0o400)
    root.chmod(0o500)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"import failed: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_files(root: Path) -> dict[str, dict[str, Any]]:
    require(root.is_dir(), f"protected directory missing: {root}")
    return {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def capture_result(
    validation: Path,
    stem: str,
    command: list[str],
    result: subprocess.CompletedProcess[bytes],
) -> None:
    write_new(validation / f"{stem}.command.json", canonical_json(command), 0o400)
    write_new(validation / f"{stem}.stdout", result.stdout, 0o400)
    write_new(validation / f"{stem}.stderr", result.stderr, 0o400)
    write_new(
        validation / f"{stem}.status",
        f"{result.returncode}\n".encode("ascii"),
        0o400,
    )


def write_review(
    validator: Any,
    package: Path,
    manifest: dict[str, Any],
    review_root: Path,
    **changes: Any,
) -> Path:
    review = {
        "schema_version": 1,
        "kind": "ace3_model24_r13_independent_l2_review",
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
    }
    review.update(changes)
    review_root.mkdir()
    path = review_root / "review.json"
    path.write_bytes(canonical_json(review))
    path.chmod(0o400)
    return path


def _expect_rejection(callable_object: Any, expected: str, name: str) -> None:
    try:
        callable_object()
    except SystemExit as error:
        require(expected in str(error), f"{name} rejected for wrong reason: {error}")
    else:
        raise SystemExit(f"negative probe unexpectedly accepted: {name}")


def _fake_terminal(
    root: Path,
    task_id: str,
    command: str,
    child_cwd: Path,
    *,
    replay: bool = False,
    invocation_count: int = 1,
) -> Path:
    registry = root / ".argus_subagents"
    log_dir = registry / f"{task_id}_logs"
    log_dir.mkdir(parents=True)
    run_id = f"{task_id}-123456789"
    (log_dir / "stdout.log").write_bytes(b"")
    (log_dir / "stderr.log").write_bytes(b"")
    (log_dir / f"exit_code.{run_id}").write_text("0\n", encoding="ascii")
    receipt = {
        "state": "done",
        "task_id": task_id,
        "run_id": run_id,
        "command": command,
        "mode": "direct",
        "cwd": str(child_cwd),
        "exit_code": 0,
        "stdout_log": f".argus_subagents/{task_id}_logs/stdout.log",
        "stderr_log": f".argus_subagents/{task_id}_logs/stderr.log",
        "replay": replay,
    }
    (registry / f"{task_id}.json").write_bytes(canonical_json(receipt))
    launch_terminal = root.parent / f"{root.name}-launch-terminal.json"
    launch_terminal.write_bytes(
        canonical_json({"invocation_count": invocation_count, "cardinality": 1})
    )
    return launch_terminal


def run_negative_tests(package: Path) -> dict[str, Any]:
    validator = load_module(package / "validate-package.py", "model24_r13_validator")
    manifest = validator.load_json(package / "package.json")
    cases: list[str] = []
    canonical = validator.expected_paths(package, manifest)

    with tempfile.TemporaryDirectory(prefix="ace3-model24-r13-inert-") as temporary:
        scratch = Path(temporary)
        positive_review = write_review(
            validator, package, manifest, scratch / "positive-review"
        )
        validator.validate_review_document(package, manifest, positive_review)

        for name, changes, expected in (
            ("reviewer-producer-role", {"producer_role": "engineer"}, "producer_role"),
            ("reviewer-active-role", {"active_role": "engineer"}, "active_role"),
            ("reviewer-level", {"reviewer_level": "L1"}, "reviewer_level"),
            (
                "reviewer-preparation-participation",
                {"preparation_participation": True},
                "preparation_participation",
            ),
            (
                "reviewer-identity",
                {"reviewer_identity": manifest["review_policy"]["preparer_identity"]},
                "not independent",
            ),
        ):
            path = write_review(
                validator, package, manifest, scratch / name, **changes
            )
            _expect_rejection(
                lambda path=path: validator.validate_review_document(
                    package, manifest, path
                ),
                expected,
                name,
            )
            cases.append(name)

        path = write_review(
            validator, package, manifest, scratch / "reviewer-test-results"
        )
        review = validator.load_json(path)
        review["test_results"]["negative_cases"][0]["status"] = "FAIL"
        path.chmod(0o600)
        path.write_bytes(canonical_json(review))
        path.chmod(0o400)
        _expect_rejection(
            lambda: validator.validate_review_document(package, manifest, path),
            "ACCEPT review",
            "reviewer-test-results",
        )
        cases.append("reviewer-test-results")

        path = write_review(
            validator, package, manifest, scratch / "provenance-hash"
        )
        review = validator.load_json(path)
        review["bound_hashes"]["validator_sha256"] = "0" * 64
        path.chmod(0o600)
        path.write_bytes(canonical_json(review))
        path.chmod(0o400)
        _expect_rejection(
            lambda: validator.validate_review_document(package, manifest, path),
            "sealed artifacts",
            "provenance-hash",
        )
        cases.append("provenance-hash")

        authority = validator.expected_authority(
            package, manifest, positive_review
        )
        authority["package_seal_sha256"] = "0" * 64
        authority_path = scratch / "authority-binding.json"
        authority_path.write_bytes(canonical_json(authority))
        authority_path.chmod(0o400)
        _expect_rejection(
            lambda: validator.validate_authority_document(
                package, manifest, positive_review, authority_path
            ),
            "exactly bind",
            "authority-binding",
        )
        cases.append("authority-binding")

        for semantic in ("retry", "replay", "resume", "watcher"):
            local = copy.deepcopy(manifest)
            local["execution"][semantic] = True
            _expect_rejection(
                lambda local=local: validator.validate_contract(package, local),
                "policy is open",
                semantic,
            )
            cases.append(semantic)

        local = copy.deepcopy(manifest)
        local["execution"]["launch_argv"] = local["execution"]["launch_argv"][:-1]
        _expect_rejection(
            lambda: validator.validate_contract(package, local),
            "argv differs",
            "wrong-argv",
        )
        cases.append("wrong-argv")

        local = copy.deepcopy(manifest)
        local["execution"]["child_cwd"] = str(scratch)
        _expect_rejection(
            lambda: validator.validate_contract(package, local),
            "child cwd differs",
            "launch-contract",
        )
        cases.append("launch-contract")

        for name, args, expected in (
            (
                "wrong-receipt-path",
                (scratch / "wrong.json", canonical["stdout_log"], canonical["stderr_log"]),
                "wrong durable receipt",
            ),
            (
                "wrong-stdout-path",
                (canonical["receipt"], scratch / "stdout.log", canonical["stderr_log"]),
                "wrong durable stdout",
            ),
            (
                "wrong-stderr-path",
                (canonical["receipt"], canonical["stdout_log"], scratch / "stderr.log"),
                "wrong durable stderr",
            ),
        ):
            _expect_rejection(
                lambda args=args: validator.validate_terminal_artifact_paths(
                    manifest, *args
                ),
                expected,
                name,
            )
            cases.append(name)

        replay_root = scratch / "receipt-replay"
        _fake_terminal(
            replay_root,
            manifest["identity"]["task_id"],
            manifest["execution"]["durable_command"],
            canonical["child_cwd"],
            replay=True,
        )
        _expect_rejection(
            lambda: validator.validate_terminal_layout(
                replay_root,
                manifest["identity"]["task_id"],
                manifest["execution"]["durable_command"],
                canonical["child_cwd"],
            ),
            "receipt enables",
            "receipt-replay",
        )
        cases.append("receipt-replay")

        existing = scratch / "pre-existing-output"
        existing.mkdir()
        _expect_rejection(
            lambda: validator.validate_absent_paths(
                {"output": existing}, ("output",), "package"
            ),
            "absent output",
            "pre-existing-state",
        )
        cases.append("pre-existing-state")

        mutated = scratch / "mutated-package"
        shutil.copytree(package, mutated)
        mutated.chmod(0o700)
        target = mutated / "review-request.json"
        target.chmod(0o600)
        target.write_bytes(target.read_bytes() + b" ")
        _expect_rejection(
            lambda: validator.validate_package_seal(mutated),
            "differs from seal",
            "package-mutation",
        )
        cases.append("package-mutation")

        mutated_source = scratch / "mutated-source"
        shutil.copytree(package.parent / "source", mutated_source)
        source_target = next(path for path in mutated_source.rglob("*") if path.is_file())
        source_target.chmod(0o600)
        source_target.write_bytes(source_target.read_bytes() + b" ")
        records = validator.load_json(package / "source-tree.json")["records"]
        _expect_rejection(
            lambda: validator.require(
                validator.tree_records(mutated_source) == records,
                "sealed source differs from manifest",
            ),
            "differs from manifest",
            "source-mutation",
        )
        cases.append("source-mutation")

        local = copy.deepcopy(manifest)
        local["namespaces"]["child_cwd"] = local["namespaces"]["submitter_root"]
        _expect_rejection(
            lambda: validator.validate_root_separation(local),
            "conflated",
            "conflated-registry-child-roots",
        )
        cases.append("conflated-registry-child-roots")

        _expect_rejection(
            lambda: validator.validate_submitter_cwd(manifest, scratch),
            "wrong submitter cwd",
            "wrong-submitter-cwd",
        )
        cases.append("wrong-submitter-cwd")

        _expect_rejection(
            lambda: validator.validate_terminal_root_value(
                Path("/tmp/ace3-model24-r13-terminal-fixture")
            ),
            "temporary or noncanonical",
            "temporary-terminal-root",
        )
        cases.append("temporary-terminal-root")

        valid_authority = validator.expected_authority(
            package, manifest, positive_review
        )
        valid_authority_path = scratch / "prelaunch-authority.json"
        valid_authority_path.write_bytes(canonical_json(valid_authority))
        valid_authority_path.chmod(0o400)
        require(not canonical["receipt"].exists(), "prelaunch fixture created a receipt")
        validator.validate_review_document(package, manifest, positive_review)
        validator.validate_authority_document(
            package, manifest, positive_review, valid_authority_path
        )
        cases.append("pre-launch-receipt-dependence")

        foreign_root = scratch / "foreign-terminal"
        _fake_terminal(
            foreign_root,
            manifest["identity"]["task_id"],
            manifest["execution"]["durable_command"],
            canonical["child_cwd"],
        )
        (foreign_root / "foreign.json").write_text("{}\n", encoding="ascii")
        _expect_rejection(
            lambda: validator.validate_terminal_layout(
                foreign_root,
                manifest["identity"]["task_id"],
                manifest["execution"]["durable_command"],
                canonical["child_cwd"],
            ),
            "foreign terminal",
            "foreign-terminal-artifact",
        )
        cases.append("foreign-terminal-artifact")

        missing_root = scratch / "missing-terminal"
        _fake_terminal(
            missing_root,
            manifest["identity"]["task_id"],
            manifest["execution"]["durable_command"],
            canonical["child_cwd"],
        )
        (
            missing_root
            / ".argus_subagents"
            / f"{manifest['identity']['task_id']}_logs"
            / "stderr.log"
        ).unlink()
        _expect_rejection(
            lambda: validator.validate_terminal_layout(
                missing_root,
                manifest["identity"]["task_id"],
                manifest["execution"]["durable_command"],
                canonical["child_cwd"],
            ),
            "missing terminal log",
            "missing-terminal-artifact",
        )
        cases.append("missing-terminal-artifact")

        _expect_rejection(
            lambda: validator.validate_payload_cardinality(
                {"invocation_count": 2, "cardinality": 1}
            ),
            "duplicate payload",
            "duplicate-payload-invocation",
        )
        cases.append("duplicate-payload-invocation")

        require(
            all(
                not canonical[name].exists()
                for name in (
                    "review",
                    "authority",
                    "authority_consumed",
                    "output",
                    "terminal_root",
                )
            ),
            "inert fixture created a canonical r13 artifact",
        )
        cases.append("canonical-artifact-from-inert-fixture")

    require(
        cases == list(validator.REQUIRED_NEGATIVE_CASES),
        "negative case order differs from r13 contract",
    )
    return {
        "status": "PASS",
        "kind": "ace3_model24_r13_inert_negative_tests",
        "case_count": len(cases),
        "cases": cases,
        "execution_invocations": 0,
        "launcher_invocations": 0,
        "model_invocations": 0,
        "simulator_invocations": 0,
        "canonical_artifact_creations": 0,
    }


def run_installed_runner_probe(
    package: Path,
    nonce: str,
    runner_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    probe = package / "inert-runner-probe"
    child = probe / "child"
    submitter = probe / "submitter"
    result_root = probe / "result"
    child.mkdir(parents=True)
    submitter.mkdir()
    result_root.mkdir()
    marker = result_root / "invocation.json"
    payload = child / "payload.py"
    payload_source = """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
path = Path(sys.argv[1])
value = {"invocation_count": 1, "cwd": os.getcwd(), "pid": os.getpid()}
with path.open("x", encoding="ascii") as stream:
    json.dump(value, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\\n")
"""
    write_new(payload, payload_source.encode("ascii"), 0o500)
    task_id = f"ace3-model24-r13-inert-probe-{nonce}"
    command = shlex.join([str(MODEL_PYTHON), "-B", str(payload), str(marker)])
    submission = [
        str(ARGUS_PYTHON),
        "-m",
        "argus_skill.tools.subagent",
        "submit",
        "--task-id",
        task_id,
        "--description",
        "ACE-3 r13 inert submitter-root and child-cwd integration probe",
        "--mode",
        "direct",
        "--timeout",
        "30",
        "--command",
        command,
        "--cwd",
        str(child),
        "--cpu-count",
        "1",
    ]
    submitted = subprocess.run(
        submission,
        cwd=submitter,
        capture_output=True,
        check=False,
    )
    require(
        submitted.returncode == 0,
        submitted.stderr.decode("utf-8", errors="replace"),
    )
    submit_receipt = json.loads(submitted.stdout.decode("utf-8"))
    require(submit_receipt.get("state") == "submitted", "probe was not submitted")
    require(submit_receipt.get("task_id") == task_id, "probe submit task mismatch")
    require(submit_receipt.get("run_id"), "probe submit run id missing")
    require(submit_receipt.get("check_with"), "probe check_with command missing")
    registry = submitter / ".argus_subagents"
    receipt_path = registry / f"{task_id}.json"
    deadline = time.monotonic() + 30
    receipt: dict[str, Any] = {}
    while time.monotonic() < deadline:
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("state") in {"done", "error", "timeout"}:
                break
        time.sleep(0.05)
    require(receipt.get("state") == "done", f"inert runner probe failed: {receipt}")
    require(receipt.get("exit_code") == 0, "inert runner probe exit code is nonzero")
    require(receipt.get("cwd") == str(child), "inert runner probe child cwd mismatch")
    require(not (child / ".argus_subagents").exists(), "runner registry followed child --cwd")
    require(registry.is_dir(), "runner registry missing from submitter root")
    observed = json.loads(marker.read_text(encoding="ascii"))
    require(observed["invocation_count"] == 1, "inert payload ran more than once")
    require(observed["cwd"] == str(child), "inert payload did not run under child cwd")
    log_dir = registry / f"{task_id}_logs"
    exit_sidecar = log_dir / f"exit_code.{receipt['run_id']}"
    require(exit_sidecar.read_text(encoding="ascii") == "0\n", "probe exit sidecar mismatch")
    require(
        {path.name for path in registry.iterdir()}
        == {receipt_path.name, log_dir.name},
        "probe registry contains foreign artifacts",
    )
    require(
        {path.name for path in log_dir.iterdir()}
        == {"stdout.log", "stderr.log", exit_sidecar.name},
        "probe log directory contains foreign artifacts",
    )
    evidence = {
        "schema_version": 1,
        "kind": "ace3_model24_r13_installed_direct_runner_probe",
        "status": "PASS",
        "task_id": task_id,
        "run_id": receipt["run_id"],
        "submit_receipt": submit_receipt,
        "submission_argv": submission,
        "submitter_cwd": str(submitter),
        "child_cwd": str(child),
        "registry_root": str(registry),
        "registry_under_submitter_root": True,
        "registry_under_child_cwd": False,
        "payload_reached": True,
        "payload_invocation_count": 1,
        "runner_state": receipt["state"],
        "runner_exit_code": receipt["exit_code"],
        "runner_sources": runner_sources,
        "receipt_sha256": digest(receipt_path),
        "stdout_sha256": digest(log_dir / "stdout.log"),
        "stderr_sha256": digest(log_dir / "stderr.log"),
        "exit_sidecar_sha256": digest(exit_sidecar),
    }
    write_new(probe / "evidence.json", canonical_json(evidence), 0o400)
    return evidence


def prepare(nonce: str) -> dict[str, Any]:
    base = Path(f"/home/argustest/ace3-model24-r13-prep-20260829-{nonce}")
    package = base / "package"
    source = base / "source"
    validation = base / "validation"
    preparation = base / "preparation"
    review = Path(f"/home/argustest/ace3-model24-r13-review-20260829-{nonce}")
    authority = Path(f"/home/argustest/ace3-model24-r13-authority-20260829-{nonce}.json")
    consumed = Path(str(authority) + ".consumed")
    output = Path(f"/home/argustest/ace3-model24-r13-output-20260829-{nonce}")
    terminal_root = Path(
        f"/home/argustest/ace3-model24-r13-terminal-20260829-{nonce}"
    )
    task_id = f"ace3-model24-r13-42c895c-{nonce}"
    for path in (base, review, authority, consumed, output, terminal_root):
        require(not path.exists(), f"fresh r13 namespace already exists: {path}")
    require(
        {path.name for path in R12_TERMINAL.iterdir()}
        == set(R12_TERMINAL_HASHES),
        "r12 terminal evidence file set changed",
    )
    for name, expected in R12_TERMINAL_HASHES.items():
        require(digest(R12_TERMINAL / name) == expected, f"r12 evidence changed: {name}")
    require(digest(R12_PACKAGE / "seal.json") == R12_PACKAGE_SEAL_SHA256, "r12 package seal changed")
    require(BINDINGS_SOURCE.is_file(), "sealed r12 bindings are unavailable")
    require(CHECKPOINT.stat().st_size == CHECKPOINT_BYTES, "checkpoint size changed")
    require(digest(CHECKPOINT) == CHECKPOINT_SHA256, "checkpoint hash changed")
    for path in (CONTRACT, VALIDATOR, LIFECYCLE, MODEL_PYTHON, ARGUS_PYTHON):
        require(path.is_file(), f"required construction input missing: {path}")
    r12_before = snapshot_files(R12_TERMINAL)

    commit_result = subprocess.run(
        ["git", "rev-parse", f"{COMMIT}^{{commit}}"],
        cwd=REPOSITORY,
        capture_output=True,
        check=False,
    )
    require(commit_result.returncode == 0, commit_result.stderr.decode())
    require(commit_result.stdout.decode().strip() == COMMIT, "accepted commit changed")
    tree_result = subprocess.run(
        ["git", "rev-parse", f"{COMMIT}^{{tree}}"],
        cwd=REPOSITORY,
        capture_output=True,
        check=False,
    )
    require(tree_result.returncode == 0, tree_result.stderr.decode())
    require(tree_result.stdout.decode().strip() == TREE, "accepted tree changed")

    base.mkdir(mode=0o700)
    package.mkdir(mode=0o700)
    source.mkdir(mode=0o700)
    validation.mkdir(mode=0o700)
    preparation.mkdir(mode=0o700)
    capture_result(preparation, "accepted-commit", ["git", "rev-parse", f"{COMMIT}^{{commit}}"], commit_result)
    capture_result(preparation, "accepted-tree", ["git", "rev-parse", f"{COMMIT}^{{tree}}"], tree_result)

    archive = preparation / "source-export.tar"
    archive_command = [
        "git",
        "archive",
        "--format=tar",
        f"--output={archive}",
        COMMIT,
    ]
    archive_result = subprocess.run(
        archive_command,
        cwd=REPOSITORY,
        capture_output=True,
        check=False,
    )
    capture_result(preparation, "source-archive", archive_command, archive_result)
    require(archive_result.returncode == 0, archive_result.stderr.decode())
    with tarfile.open(archive, "r") as stream:
        stream.extractall(source, filter="data")
    readonly_tree(source)
    source_records = mutable_tree_records(source)
    source_tree_sha256 = hashlib.sha256(canonical_json(source_records)).hexdigest()

    write_new(package / "launch-contract.json", CONTRACT.read_bytes(), 0o400)
    write_new(package / "validate-package.py", VALIDATOR.read_bytes(), 0o500)
    write_new(package / "lifecycle.py", LIFECYCLE.read_bytes(), 0o500)
    write_new(package / "bindings.json", BINDINGS_SOURCE.read_bytes(), 0o400)
    ancestry = package / "ancestry"
    ancestry.mkdir()
    write_new(ancestry / "repository-builder.py", Path(__file__).read_bytes(), 0o400)
    provenance_root = package / "provenance"
    provenance_root.mkdir()
    r12_copy = provenance_root / "r12-terminal"
    r12_copy.mkdir()
    for name in R12_TERMINAL_HASHES:
        write_new(r12_copy / name, (R12_TERMINAL / name).read_bytes(), 0o400)
    r12_evidence = {
        "schema_version": 1,
        "kind": "ace3_model24_r13_preserved_r12_terminal_evidence",
        "source": str(R12_TERMINAL),
        "original_hashes": R12_TERMINAL_HASHES,
        "copied_byte_for_byte": True,
        "canonical_stdout_stderr_existed": False,
        "canonical_stdout_stderr_reconstructed": False,
        "observed": {
            "state": "error",
            "exit_code": 1,
            "elapsed_seconds": 1.2,
            "stderr_tail": "durable receipt namespace missing\n",
            "registry_submitter_cwd": "/tmp",
        },
    }
    write_new(r12_copy / "evidence.json", canonical_json(r12_evidence), 0o400)
    source_manifest = {
        "schema_version": 1,
        "kind": "ace3_model24_r13_source_tree",
        "commit": COMMIT,
        "git_tree": TREE,
        "records": source_records,
        "tree_sha256": source_tree_sha256,
    }
    write_new(package / "source-tree.json", canonical_json(source_manifest), 0o400)

    runner_sources = []
    for name in ("_cli.py", "_direct_run.py", "_registry.py"):
        path = RUNNER_ROOT / name
        require(path.is_file(), f"installed runner source missing: {path}")
        runner_sources.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )

    child_cwd = source
    registry_root = terminal_root / ".argus_subagents"
    log_dir = registry_root / f"{task_id}_logs"
    namespaces = {
        "child_cwd": str(child_cwd),
        "review": str(review),
        "authority": str(authority),
        "authority_consumed": str(consumed),
        "output": str(output),
        "terminal_root": str(terminal_root),
        "submitter_root": str(terminal_root),
        "registry_root": str(registry_root),
        "receipt": str(registry_root / f"{task_id}.json"),
        "log_dir": str(log_dir),
        "stdout_log": str(log_dir / "stdout.log"),
        "stderr_log": str(log_dir / "stderr.log"),
        "terminal_manifest": str(terminal_root / "manifest.json"),
    }
    launch_argv = [
        str(MODEL_PYTHON),
        "-B",
        str(package / "lifecycle.py"),
        "launch",
        "--authority",
        str(authority),
        "--review",
        str(review / "review.json"),
    ]
    durable_command = shlex.join(launch_argv)
    submission_argv = [
        str(ARGUS_PYTHON),
        "-m",
        "argus_skill.tools.subagent",
        "submit",
        "--task-id",
        task_id,
        "--description",
        "ACE-3 r13 Model24 native AWQ cascade, independently accepted and Manager-authorized exactly once",
        "--mode",
        "direct",
        "--timeout",
        "18000",
        "--command",
        durable_command,
        "--cwd",
        str(child_cwd),
        "--cpu-count",
        "1",
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
        str(package / "bindings.json"),
        "--simulation-dir",
        str(output / "controller-simulation"),
        "--output-dir",
        str(output / "rtl-cascade"),
        "--fresh",
    ]
    manifest = {
        "schema_version": 1,
        "kind": "ace3_model24_r13_durable_launch_package",
        "identity": {"candidate": str(base), "nonce": nonce, "task_id": task_id},
        "provenance": {
            "source": {
                "repository": str(REPOSITORY),
                "commit": COMMIT,
                "tree": TREE,
                "materialization": "git archive",
                "tree_sha256": source_tree_sha256,
            },
            "checkpoint": {
                "path": str(CHECKPOINT),
                "bytes": CHECKPOINT_BYTES,
                "sha256": CHECKPOINT_SHA256,
            },
            "bindings": {
                "source": str(BINDINGS_SOURCE),
                "sha256": digest(package / "bindings.json"),
                "r12_package_seal_sha256": R12_PACKAGE_SEAL_SHA256,
            },
            "r12_terminal_evidence": {
                "source": str(R12_TERMINAL),
                "hashes": R12_TERMINAL_HASHES,
                "stdout_stderr_reconstructed": False,
            },
            "installed_runner_sources": runner_sources,
        },
        "review_policy": {
            "preparer_identity": f"role:engineer/session:r13-{nonce}",
            "required_reviewer_level": "L2",
            "required_independent": True,
            "exactly_one_future_verdict": True,
        },
        "execution": {
            "cardinality": 1,
            "current_invocation_count": 0,
            "submission_argv": submission_argv,
            "submitter_cwd": str(terminal_root),
            "launch_argv": launch_argv,
            "durable_command": durable_command,
            "child_cwd": str(child_cwd),
            "payload_argv": payload_argv,
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
            "terminal_manifest_creations": 0,
        },
    }
    write_new(package / "package.json", canonical_json(manifest), 0o400)

    validator = load_module(VALIDATOR, "repository_r13_validator")
    review_request = {
        "schema_version": 1,
        "kind": "ace3_model24_r13_independent_l2_review_request",
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
        "required_bound_hashes": list(validator.REQUIRED_REVIEW_HASHES),
        "required_test_results": {
            "positive_fixtures": list(validator.REQUIRED_POSITIVE_FIXTURES),
            "negative_cases": list(validator.REQUIRED_NEGATIVE_CASES),
        },
        "required_zero_state_observation": validator.clean_zero_state_observation(),
        "package_manifest_sha256": digest(package / "package.json"),
        "lifecycle_sha256": digest(package / "lifecycle.py"),
        "validator_sha256": digest(package / "validate-package.py"),
        "launch_contract_sha256": digest(package / "launch-contract.json"),
        "source_tree_sha256": digest(package / "source-tree.json"),
        "r12_terminal_evidence_sha256": digest(r12_copy / "evidence.json"),
        "prelaunch_receipt_dependency": False,
        "submitter_cwd": str(terminal_root),
        "child_cwd": str(child_cwd),
        "terminal_evidence_root": str(terminal_root),
        "execution_authority_withheld": True,
        "execution_invocations": 0,
        "stage_transition": False,
    }
    write_new(package / "review-request.json", canonical_json(review_request), 0o400)
    test_source = (
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import importlib.util\n"
        f"spec=importlib.util.spec_from_file_location('builder',{str(ancestry / 'repository-builder.py')!r})\n"
        "module=importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "result=module.run_negative_tests(Path(__file__).resolve().parent)\n"
        "print(module.canonical_json(result).decode('ascii'),end='')\n"
    ).encode("ascii")
    write_new(package / "test-launch-contract.py", test_source, 0o500)

    probe = run_installed_runner_probe(package, nonce, runner_sources)
    require(probe["payload_invocation_count"] == 1, "inert probe cardinality mismatch")
    readonly_tree(package / "inert-runner-probe")
    readonly_tree(package / "ancestry")
    readonly_tree(package / "provenance")

    package_records = [
        record
        for record in mutable_tree_records(package)
        if record["path"] != "seal.json"
    ]
    seal = {
        "schema_version": 1,
        "kind": "ace3_model24_r13_package_seal",
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
    positive = subprocess.run(
        positive_command, cwd=base, capture_output=True, check=False
    )
    capture_result(validation, "package-validation", positive_command, positive)
    require(positive.returncode == 0, positive.stderr.decode("utf-8", errors="replace"))

    negative_command = [
        str(MODEL_PYTHON),
        "-B",
        str(package / "test-launch-contract.py"),
    ]
    negative = subprocess.run(
        negative_command, cwd=base, capture_output=True, check=False
    )
    capture_result(validation, "negative-tests", negative_command, negative)
    require(negative.returncode == 0, negative.stderr.decode("utf-8", errors="replace"))

    require(snapshot_files(R12_TERMINAL) == r12_before, "r12 terminal evidence changed")
    require(CHECKPOINT.stat().st_size == CHECKPOINT_BYTES, "checkpoint size changed after preparation")
    require(digest(CHECKPOINT) == CHECKPOINT_SHA256, "checkpoint changed after preparation")
    for path in (review, authority, consumed, output, terminal_root):
        require(not path.exists(), f"canonical review or execution artifact created: {path}")
    zero_state = {
        "review_absent": not review.exists(),
        "authority_absent": not authority.exists(),
        "authority_consumed_absent": not consumed.exists(),
        "output_absent": not output.exists(),
        "terminal_root_absent": not terminal_root.exists(),
        "receipt_absent": not Path(namespaces["receipt"]).exists(),
        "stdout_log_absent": not Path(namespaces["stdout_log"]).exists(),
        "stderr_log_absent": not Path(namespaces["stderr_log"]).exists(),
        "terminal_manifest_absent": not Path(namespaces["terminal_manifest"]).exists(),
        "payload_execution_absent": True,
        "model24_execution_invocations": 0,
    }
    write_new(validation / "zero-state.json", canonical_json(zero_state), 0o400)
    validation_records = mutable_tree_records(validation)
    validation_seal = {
        "schema_version": 1,
        "kind": "ace3_model24_r13_inert_validation_seal",
        "package_seal_sha256": digest(package / "seal.json"),
        "records": validation_records,
        "negative_case_count": len(validator.REQUIRED_NEGATIVE_CASES),
        "installed_runner_probe_invocations": 1,
        "execution_invocations": 0,
        "launcher_invocations": 0,
        "model_invocations": 0,
        "simulator_invocations": 0,
        "canonical_artifact_creations": 0,
    }
    write_new(validation / "seal.json", canonical_json(validation_seal), 0o400)
    result = {
        "status": "SEALED_AWAITING_INDEPENDENT_L2_REVIEW",
        "candidate": str(base),
        "package": str(package),
        "source_commit": COMMIT,
        "source_tree": TREE,
        "task_id": task_id,
        "nonce": nonce,
        "package_manifest_sha256": digest(package / "package.json"),
        "package_seal_sha256": digest(package / "seal.json"),
        "review_request_sha256": digest(package / "review-request.json"),
        "validation_seal_sha256": digest(validation / "seal.json"),
        "negative_case_count": len(validator.REQUIRED_NEGATIVE_CASES),
        "installed_runner_probe": "PASS",
        "installed_runner_probe_invocations": 1,
        "r12_terminal_evidence_preserved": True,
        "canonical_zero_state": zero_state,
        "execution_authority_withheld": True,
        "execution_invocations": 0,
    }
    write_new(preparation / "result.json", canonical_json(result), 0o400)
    readonly_tree(validation)
    readonly_tree(preparation)
    base.chmod(0o500)
    return result


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
