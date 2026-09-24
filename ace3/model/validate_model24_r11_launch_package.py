#!/usr/bin/env python3
"""Fail-closed validation for a sealed Model24 r11 durable-launch package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import time
from typing import Any


REQUIRED_REVIEW_HASHES = (
    "package_manifest_sha256",
    "package_seal_sha256",
    "review_request_sha256",
    "launcher_sha256",
    "validator_sha256",
    "launch_contract_sha256",
)
FORBIDDEN_SEMANTICS = ("retry", "replay", "resume", "watcher")
MODEL_PYTHON = "/home/argustest/miniconda3/bin/python3"
BA953_REVIEW_SHA256 = "6efc30359df96af1a72f8f156416f4838069c0bedaf28161e42894bef1c46d05"
BA953_ANCESTRY_STATEMENT = (
    "The embedded ba953 review is nonconforming rejected history only and is "
    "not a trust root, authorization, package acceptance, or execution "
    "authority; launch acceptance has no external parent-review dependency."
)
REQUIRED_NEGATIVE_CASES = (
    "reviewer-producer-role",
    "reviewer-active-role",
    "reviewer-level",
    "reviewer-preparation-participation",
    "reviewer-identity",
    "reviewer-test-results",
    "provenance-hash",
    "authority-binding",
    "retry",
    "replay",
    "resume",
    "watcher",
    "wrong-argv",
    "launch-contract",
    "wrong-receipt-path",
    "wrong-stdout-path",
    "wrong-stderr-path",
    "receipt-replay",
    "pre-existing-state",
    "package-mutation",
    "source-mutation",
)
REQUIRED_POSITIVE_FIXTURES = ("package", "review", "launch")


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
    require(path.is_file() and not path.is_symlink(), f"regular JSON file required: {path}")
    value = json.loads(path.read_text(encoding="ascii"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def tree_records(root: Path) -> list[dict[str, Any]]:
    require(root.is_dir() and not root.is_symlink(), f"regular directory required: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        require(not path.is_symlink(), f"symlink is forbidden in sealed tree: {relative}")
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
            raise SystemExit(f"unsupported sealed-tree entry: {relative}")
    return records


def tree_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(records)).hexdigest()


def validate_recorded_tree(
    root: Path,
    expected_records: list[dict[str, Any]],
    *,
    require_read_only: bool,
) -> None:
    actual = tree_records(root)
    require(actual == expected_records, f"sealed tree differs from manifest: {root}")
    if require_read_only:
        require(root.stat().st_mode & 0o222 == 0, f"sealed root is writable: {root}")
        for record in actual:
            require(record["mode"] & 0o222 == 0, f"sealed entry is writable: {record['path']}")


def package_review_hashes(package: Path) -> dict[str, str]:
    return {
        "package_manifest_sha256": digest(package / "package.json"),
        "package_seal_sha256": digest(package / "seal.json"),
        "review_request_sha256": digest(package / "review-request.json"),
        "launcher_sha256": digest(package / "launch-once.py"),
        "validator_sha256": digest(package / "validate-package.py"),
        "launch_contract_sha256": digest(package / "launch-contract.json"),
    }


def passing_review_test_results() -> dict[str, Any]:
    return {
        "positive_fixtures": [
            {"name": name, "status": "PASS"} for name in REQUIRED_POSITIVE_FIXTURES
        ],
        "negative_cases": [
            {"name": name, "status": "PASS"} for name in REQUIRED_NEGATIVE_CASES
        ],
        "execution_invocations": 0,
        "launcher_invocations": 0,
        "model_invocations": 0,
        "simulator_invocations": 0,
    }


def clean_zero_state_observation() -> dict[str, Any]:
    return {
        "observed_before_review_creation": True,
        "authority_paths": 0,
        "authority_consumption_paths": 0,
        "output_paths": 0,
        "durable_receipt_paths": 0,
        "review_paths": 0,
        "matching_processes": 0,
        "execution_invocations": 0,
        "launcher_invocations": 0,
        "model_invocations": 0,
        "simulator_invocations": 0,
    }


def validate_test_results(value: object, verdict: str) -> None:
    require(isinstance(value, dict), "review test results must be an object")
    require(
        set(value)
        == {
            "positive_fixtures",
            "negative_cases",
            "execution_invocations",
            "launcher_invocations",
            "model_invocations",
            "simulator_invocations",
        },
        "review test-result fields differ from the canonical schema",
    )
    for field, names in (
        ("positive_fixtures", REQUIRED_POSITIVE_FIXTURES),
        ("negative_cases", REQUIRED_NEGATIVE_CASES),
    ):
        results = value[field]
        require(isinstance(results, list), f"review {field} must be a list")
        require(
            [result.get("name") for result in results if isinstance(result, dict)]
            == list(names),
            f"review {field} names or order differ from the required inert suite",
        )
        require(
            all(
                isinstance(result, dict)
                and set(result) == {"name", "status"}
                and result["status"] in {"PASS", "FAIL"}
                for result in results
            ),
            f"review {field} entries must contain exact name and PASS/FAIL status",
        )
    for field in (
        "execution_invocations",
        "launcher_invocations",
        "model_invocations",
        "simulator_invocations",
    ):
        require(value[field] == 0, f"review reports nonzero {field}")
    statuses = [
        result["status"]
        for field in ("positive_fixtures", "negative_cases")
        for result in value[field]
    ]
    if verdict == "ACCEPT":
        require(all(status == "PASS" for status in statuses), "ACCEPT review contains a failed inert result")
    else:
        require(any(status == "FAIL" for status in statuses), "REJECT review must identify a failed inert result")


def validate_package_seal(package: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    seal = load_json(package / "seal.json")
    require(
        set(seal)
        == {
            "schema_version",
            "kind",
            "nonce",
            "task_id",
            "package_manifest_sha256",
            "review_request_sha256",
            "source_tree_sha256",
            "package_entries",
            "zero_state",
        },
        "package seal fields differ from the r11 schema",
    )
    require(seal["schema_version"] == 1, "unsupported package seal schema")
    require(seal["kind"] == "ace3_model24_r11_package_seal", "package seal kind mismatch")
    records = seal["package_entries"]
    require(isinstance(records, list), "package seal entry list missing")
    actual = tree_records(package)
    actual = [record for record in actual if record["path"] != "seal.json"]
    require(actual == records, "package file set, mode, size, or hash differs from seal")
    require(package.stat().st_mode & 0o222 == 0, "package directory is writable")
    require((package / "seal.json").stat().st_mode & 0o222 == 0, "package seal is writable")
    for record in actual:
        require(record["mode"] & 0o222 == 0, f"sealed package entry is writable: {record['path']}")

    manifest = load_json(package / "package.json")
    require(digest(package / "package.json") == seal["package_manifest_sha256"], "manifest seal binding mismatch")
    require(digest(package / "review-request.json") == seal["review_request_sha256"], "review request seal binding mismatch")
    require(manifest["identity"]["nonce"] == seal["nonce"], "seal nonce mismatch")
    require(manifest["identity"]["task_id"] == seal["task_id"], "seal task mismatch")
    require(seal["zero_state"] is True, "package seal is not zero-state")
    return seal, manifest


def validate_review_request(package: Path, manifest: dict[str, Any]) -> None:
    request = load_json(package / "review-request.json")
    require(request["candidate"] == str(package.parent), "review request candidate mismatch")
    require(request["nonce"] == manifest["identity"]["nonce"], "review request nonce mismatch")
    require(request["task_id"] == manifest["identity"]["task_id"], "review request task mismatch")
    require(request["review_namespace"] == manifest["namespaces"]["review"], "review request namespace mismatch")
    require(request["reviewer_level"] == "L2", "review request level mismatch")
    require(request["producer_role"] == "reviewer", "review request producer role mismatch")
    require(request["active_role"] == "reviewer", "review request active role mismatch")
    require(request["reviewer_must_be_independent"] is True, "review request independence is open")
    require(request["preparation_participation"] is False, "review request permits preparation participation")
    require(request["required_bound_hashes"] == list(REQUIRED_REVIEW_HASHES), "review request hash contract mismatch")
    require(request["required_test_results"] == {
        "positive_fixtures": list(REQUIRED_POSITIVE_FIXTURES),
        "negative_cases": list(REQUIRED_NEGATIVE_CASES),
    }, "review request inert-test contract mismatch")
    require(request["required_zero_state_observation"] == clean_zero_state_observation(), "review request zero-state contract mismatch")
    require(request["required_ba953_ancestry_statement"] == BA953_ANCESTRY_STATEMENT, "review request ba953 statement mismatch")
    require(request["ba953_history_status"] == "NONCONFORMING_REJECTED_HISTORY_ONLY", "review request ba953 status mismatch")
    require(request["ba953_history_is_authoritative"] is False, "review request makes ba953 authoritative")
    require(request["external_parent_review_dependency"] is False, "review request has an external parent-review dependency")
    require(request["package_manifest_sha256"] == digest(package / "package.json"), "review request manifest binding mismatch")
    require(request["launcher_sha256"] == digest(package / "launch-once.py"), "review request launcher binding mismatch")
    require(request["validator_sha256"] == digest(package / "validate-package.py"), "review request validator binding mismatch")
    require(request["launch_contract_sha256"] == digest(package / "launch-contract.json"), "review request launch-contract binding mismatch")
    require(request["execution_authority_withheld"] is True, "review request grants execution authority")
    require(request["execution_invocations"] == 0, "review request reports execution")


def expected_paths(package: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    base = package.parent
    nonce = manifest["identity"]["nonce"]
    task_id = manifest["identity"]["task_id"]
    receipt_root = base / ".argus_subagents"
    log_dir = receipt_root / f"{task_id}_logs"
    return {
        "base": base,
        "source": base / "source",
        "review": Path(f"/home/argustest/ace3-model24-r11-review-20260829-{nonce}"),
        "authority": Path(f"/home/argustest/ace3-model24-r11-authority-20260829-{nonce}.json"),
        "authority_consumed": Path(f"/home/argustest/ace3-model24-r11-authority-20260829-{nonce}.json.consumed"),
        "output": Path(f"/home/argustest/ace3-model24-r11-output-20260829-{nonce}"),
        "receipt_namespace": receipt_root,
        "receipt": receipt_root / f"{task_id}.json",
        "log_dir": log_dir,
        "stdout_log": log_dir / "stdout.log",
        "stderr_log": log_dir / "stderr.log",
    }


def validate_contract(package: Path, manifest: dict[str, Any]) -> None:
    contract = load_json(package / "launch-contract.json")
    repository_contract = package / "ancestry/repository-contract.json"
    require(digest(package / "launch-contract.json") == digest(repository_contract), "repository launch-contract copy mismatch")
    require(contract["kind"] == "ace3_model24_r11_durable_launch_contract", "launch contract kind mismatch")
    require(contract["review"]["required_bound_hashes"] == list(REQUIRED_REVIEW_HASHES), "review hash contract mismatch")
    require(contract["review"]["required_positive_fixtures"] == list(REQUIRED_POSITIVE_FIXTURES), "positive fixture contract mismatch")
    require(contract["review"]["required_negative_cases"] == list(REQUIRED_NEGATIVE_CASES), "negative case contract mismatch")
    require(contract["review"]["required_ba953_ancestry_statement"] == BA953_ANCESTRY_STATEMENT, "ba953 review statement contract mismatch")
    require(contract["provenance"] == {
        "ba953_status": "NONCONFORMING_REJECTED_HISTORY_ONLY",
        "ba953_authoritative": False,
        "external_parent_review_dependency": False,
    }, "ba953 provenance contract mismatch")
    require(contract["launch"]["forbidden_semantics"] == list(FORBIDDEN_SEMANTICS), "forbidden launch semantics mismatch")
    require(contract["durable_runner"]["mode"] == "direct", "durable runner must be direct")
    require(contract["durable_runner"]["timeout_seconds"] == 18000, "durable timeout must be five hours")

    execution = manifest["execution"]
    require(all(execution[name] is False for name in FORBIDDEN_SEMANTICS), "retry/replay/resume/watcher policy is open")

    paths = expected_paths(package, manifest)
    identity = manifest["identity"]
    require(
        set(identity) == {"candidate", "nonce", "task_id"},
        "identity fields differ from the r11 schema",
    )
    require(identity["candidate"] == str(paths["base"]), "candidate path mismatch")
    require(identity["task_id"] == f"ace3-model24-r11-42c895c-{identity['nonce']}", "task identity mismatch")

    namespaces = manifest["namespaces"]
    require(set(namespaces) == set(paths) - {"base", "source"}, "namespace fields differ from the r11 schema")
    for name, expected in paths.items():
        if name not in {"base", "source"}:
            require(Path(namespaces[name]) == expected, f"canonical namespace mismatch: {name}")

    launcher_argv = [
        MODEL_PYTHON,
        "-B",
        str(package / "launch-once.py"),
        "--authority",
        str(paths["authority"]),
        "--review",
        str(paths["review"] / "review.json"),
        "--receipt",
        str(paths["receipt"]),
        "--stdout-log",
        str(paths["stdout_log"]),
        "--stderr-log",
        str(paths["stderr_log"]),
    ]
    output = paths["output"]
    payload_argv = [
        MODEL_PYTHON,
        "-B",
        str(paths["source"] / "ace3/model/controller_model24_rtl_cascade.py"),
        "--repository-root",
        str(paths["source"]),
        "--checkpoint",
        manifest["provenance"]["checkpoint"]["path"],
        "--tensor-map",
        str(paths["source"] / "ace3/contracts/model24_tensor_map.json"),
        "--bindings",
        str(package / "ancestry/ba953-candidate/package/evidence/bindings.json"),
        "--simulation-dir",
        str(output / "controller-simulation"),
        "--output-dir",
        str(output / "rtl-cascade"),
        "--fresh",
    ]
    require(
        set(execution)
        == {
            "cardinality",
            "current_invocation_count",
            "durable_command",
            "exact_argv",
            "exact_cwd",
            "payload_argv",
            "payload_cwd",
            "retry",
            "replay",
            "resume",
            "watcher",
            "timeout_seconds",
        },
        "execution fields differ from the r11 schema",
    )
    require(execution["cardinality"] == 1, "execution cardinality is not one")
    require(execution["current_invocation_count"] == 0, "execution invocation count is not zero")
    require(execution["exact_argv"] == launcher_argv, "launcher argv differs from canonical contract")
    require(execution["durable_command"] == shlex.join(launcher_argv), "durable command differs from canonical argv")
    require(execution["exact_cwd"] == str(paths["base"]), "durable cwd differs from candidate root")
    require(execution["payload_argv"] == payload_argv, "payload argv differs from fresh Model24 contract")
    require(execution["payload_cwd"] == str(paths["source"]), "payload cwd mismatch")
    require(execution["timeout_seconds"] == 18000, "execution timeout mismatch")
    joined = "\0".join(launcher_argv + payload_argv).lower()
    for forbidden in ("--retry", "--replay", "--resume", "--watch", "watcher"):
        require(forbidden not in joined, f"forbidden launch token present: {forbidden}")

    resources = manifest["resources"]
    require(resources == {
        "environment": {
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "memory_is_finite": True,
    }, "resource contract mismatch")


def validate_provenance(package: Path, manifest: dict[str, Any], seal: dict[str, Any]) -> None:
    source_manifest = load_json(package / "source-tree.json")
    require(
        set(source_manifest) == {"schema_version", "kind", "records", "tree_sha256"},
        "source-tree manifest fields differ from schema",
    )
    records = source_manifest["records"]
    require(isinstance(records, list), "source-tree records missing")
    require(tree_sha256(records) == source_manifest["tree_sha256"], "source-tree manifest self-hash mismatch")
    require(source_manifest["tree_sha256"] == seal["source_tree_sha256"], "source tree is not bound by package seal")
    validate_recorded_tree(package.parent / "source", records, require_read_only=True)

    provenance = manifest["provenance"]
    history = provenance["ba953_history"]
    require(
        set(history)
        == {
            "status",
            "authoritative",
            "trust_root",
            "package_acceptance",
            "execution_authority",
            "external_parent_review_dependency",
            "package_manifest_sha256",
            "package_seal_sha256",
            "review_sha256",
            "source_commit",
            "source_tree",
        },
        "ba953 history fields differ from the self-contained schema",
    )
    require(history["status"] == "NONCONFORMING_REJECTED_HISTORY_ONLY", "ba953 history status mismatch")
    for field in (
        "authoritative",
        "trust_root",
        "package_acceptance",
        "execution_authority",
        "external_parent_review_dependency",
    ):
        require(history[field] is False, f"ba953 history is authoritative: {field}")
    require(history["package_manifest_sha256"] == "3737400b2bcf839098e4cbd9bb1d392f5df6b383ceec3943ca603ce352c7c850", "ba953 manifest provenance mismatch")
    require(history["package_seal_sha256"] == "90b229697c77a0505945396533c4a74de8b9ec426ccd351fa3a7c4d5576580ff", "ba953 seal provenance mismatch")
    require(history["review_sha256"] == BA953_REVIEW_SHA256, "ba953 review provenance mismatch")
    ancestry_candidate = package / "ancestry/ba953-candidate"
    ancestry_review = package / "ancestry/ba953-review/review.json"
    require(digest(ancestry_candidate / "package/package.json") == history["package_manifest_sha256"], "embedded ba953 manifest mismatch")
    require(digest(ancestry_candidate / "package/seal.json") == history["package_seal_sha256"], "embedded ba953 seal mismatch")
    require(digest(ancestry_review) == history["review_sha256"], "embedded ba953 review mismatch")
    checkpoint = provenance["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    require(checkpoint_path.is_file() and not checkpoint_path.is_symlink(), "bound checkpoint is unavailable")
    require(checkpoint_path.stat().st_size == checkpoint["bytes"], "bound checkpoint size mismatch")
    require(digest(checkpoint_path) == checkpoint["sha256"], "bound checkpoint hash mismatch")


def validate_review(
    package: Path,
    manifest: dict[str, Any],
    review_path: Path,
) -> dict[str, Any]:
    paths = expected_paths(package, manifest)
    require(review_path == paths["review"] / "review.json", "review path is not canonical")
    review_root = paths["review"]
    require(review_root.is_dir() and not review_root.is_symlink(), "canonical review namespace missing")
    require({entry.name for entry in review_root.iterdir()} == {"review.json"}, "review namespace is not exactly one review.json")
    return validate_review_document(package, manifest, review_path)


def validate_review_document(
    package: Path,
    manifest: dict[str, Any],
    review_path: Path,
) -> dict[str, Any]:
    review = load_json(review_path)
    required_fields = {
        "schema_version",
        "kind",
        "candidate",
        "nonce",
        "task_id",
        "verdict",
        "producer_role",
        "active_role",
        "reviewer_level",
        "reviewer_identity",
        "independent",
        "preparation_participation",
        "execution_invocations",
        "authority_created",
        "manager_may_issue_execution_authority",
        "bound_hashes",
        "test_results",
        "zero_state_observation",
        "ba953_ancestry_statement",
    }
    require(set(review) == required_fields, "review fields differ from the canonical schema")
    require(review_path.stat().st_mode & 0o222 == 0, "review receipt is writable")
    expected = {
        "schema_version": 1,
        "kind": "ace3_model24_r11_independent_l2_review",
        "candidate": str(package.parent),
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "producer_role": "reviewer",
        "active_role": "reviewer",
        "reviewer_level": "L2",
        "independent": True,
        "preparation_participation": False,
        "execution_invocations": 0,
        "authority_created": False,
    }
    for key, value in expected.items():
        require(review[key] == value, f"review provenance or package binding mismatch: {key}")
    identity = review["reviewer_identity"]
    require(isinstance(identity, str) and identity.strip(), "reviewer identity is missing")
    require(identity != manifest["review_policy"]["preparer_identity"], "reviewer is not independent of package preparation")
    verdict = review["verdict"]
    require(verdict in {"ACCEPT", "REJECT"}, "review verdict must be ACCEPT or REJECT")
    validate_test_results(review["test_results"], verdict)
    require(
        review["zero_state_observation"] == clean_zero_state_observation(),
        "review zero-state observation differs from the canonical clean state",
    )
    require(
        review["ba953_ancestry_statement"] == BA953_ANCESTRY_STATEMENT,
        "review does not declare ba953 ancestry non-authoritative",
    )
    require(review["manager_may_issue_execution_authority"] is (verdict == "ACCEPT"), "review authority disposition mismatch")
    require(review["bound_hashes"] == package_review_hashes(package), "review does not bind all six sealed artifacts")
    return review


def expected_authority(
    package: Path,
    manifest: dict[str, Any],
    review_path: Path,
) -> dict[str, Any]:
    paths = expected_paths(package, manifest)
    hashes = package_review_hashes(package)
    return {
        "schema_version": 1,
        "kind": "ace3_model24_r11_manager_exactly_once_authority",
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "review_path": str(review_path),
        "review_sha256": digest(review_path),
        **hashes,
        "launch_cwd": str(paths["base"]),
        "launch_argv": manifest["execution"]["exact_argv"],
        "durable_command": manifest["execution"]["durable_command"],
        "receipt_path": str(paths["receipt"]),
        "stdout_log": str(paths["stdout_log"]),
        "stderr_log": str(paths["stderr_log"]),
        "output_namespace": str(paths["output"]),
        "execution_cardinality": 1,
        "manager_exactly_once_directive": True,
        "retry": False,
        "replay": False,
        "resume": False,
        "watcher": False,
    }


def validate_authority(
    package: Path,
    manifest: dict[str, Any],
    review_path: Path,
    authority_path: Path,
) -> dict[str, Any]:
    paths = expected_paths(package, manifest)
    require(authority_path == paths["authority"], "authority path is not canonical")
    return validate_authority_document(package, manifest, review_path, authority_path)


def validate_authority_document(
    package: Path,
    manifest: dict[str, Any],
    review_path: Path,
    authority_path: Path,
) -> dict[str, Any]:
    paths = expected_paths(package, manifest)
    authority = load_json(authority_path)
    require(authority_path.stat().st_mode & 0o222 == 0, "execution authority is writable")
    require(authority == expected_authority(package, manifest, review_path), "authority does not exactly bind review, package, hashes, and launch")
    require(not paths["authority_consumed"].exists(), "execution authority was already consumed")
    return authority


def pid_alive(value: object) -> bool:
    try:
        pid = int(value)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def validate_runner_state(
    manifest: dict[str, Any],
    receipt_path: Path,
    stdout_log: Path,
    stderr_log: Path,
) -> dict[str, Any]:
    package = Path(manifest["identity"]["candidate"]) / "package"
    paths = expected_paths(package, manifest)
    validate_runner_arguments(manifest, receipt_path, stdout_log, stderr_log)
    root = paths["receipt_namespace"]
    require(root.is_dir() and not root.is_symlink(), "durable receipt namespace missing")
    require({entry.name for entry in root.iterdir()} == {paths["receipt"].name, paths["log_dir"].name}, "receipt namespace contains foreign or missing entries")
    require(paths["log_dir"].is_dir() and not paths["log_dir"].is_symlink(), "task log directory missing")
    require({entry.name for entry in paths["log_dir"].iterdir()} == {"stdout.log", "stderr.log"}, "task log directory is not canonical")
    for log in (stdout_log, stderr_log):
        require(log.is_file() and not log.is_symlink(), f"durable log missing: {log}")
    return validate_receipt_document(manifest, receipt_path, stdout_log, stderr_log)


def validate_runner_arguments(
    manifest: dict[str, Any],
    receipt_path: Path,
    stdout_log: Path,
    stderr_log: Path,
) -> None:
    package = Path(manifest["identity"]["candidate"]) / "package"
    paths = expected_paths(package, manifest)
    require(receipt_path == paths["receipt"], "wrong durable receipt path")
    require(stdout_log == paths["stdout_log"], "wrong durable stdout log path")
    require(stderr_log == paths["stderr_log"], "wrong durable stderr log path")


def validate_receipt_document(
    manifest: dict[str, Any],
    receipt_path: Path,
    stdout_log: Path,
    stderr_log: Path,
) -> dict[str, Any]:
    receipt = load_json(receipt_path)
    require(receipt.get("task_id") == manifest["identity"]["task_id"], "durable receipt task mismatch")
    require(receipt.get("mode") == "direct", "durable runner mode is not direct")
    require(receipt.get("command") == manifest["execution"]["durable_command"], "durable command receipt mismatch")
    require(all(receipt.get(name, False) is False for name in FORBIDDEN_SEMANTICS), "receipt enables retry/replay/resume/watcher")
    run_id = receipt.get("run_id")
    task_id = manifest["identity"]["task_id"]
    require(isinstance(run_id, str) and run_id.startswith(task_id + "-") and run_id[len(task_id) + 1 :].isdigit(), "durable run id mismatch")
    state = receipt.get("state")
    require(state in {"starting", "running"}, "durable receipt is not an active first launch")
    if state == "starting":
        require(receipt.get("cwd") == manifest["execution"]["exact_cwd"], "starting receipt cwd mismatch")
        submitted = receipt.get("submitted_at")
        require(
            isinstance(submitted, (int, float))
            and 0 <= time.time() - float(submitted) <= 120
            and pid_alive(receipt.get("submitter_pid")),
            "starting receipt does not have a recent live submitter",
        )
    else:
        require(pid_alive(receipt.get("pid")) and pid_alive(receipt.get("worker_pid")), "running receipt lacks live runner processes")
        require(Path(receipt.get("stdout_log", "")) == stdout_log, "running receipt stdout path mismatch")
        require(Path(receipt.get("stderr_log", "")) == stderr_log, "running receipt stderr path mismatch")
    return receipt


def validate_zero_state_paths(paths: dict[str, Path], zero: dict[str, Any], mode: str) -> None:
    require(all(value == 0 for value in zero.values()), "manifest reports nonzero execution state")
    require(not paths["authority"].exists(), f"{mode} mode requires absent authority") if mode != "launch" else None
    require(not paths["authority_consumed"].exists(), "authority consumption already exists")
    require(not paths["output"].exists(), f"{mode} mode requires absent output namespace")
    require(not paths["receipt_namespace"].exists(), f"{mode} mode requires absent receipt namespace") if mode != "launch" else None
    require(not paths["review"].exists(), "package mode requires fresh absent review namespace") if mode == "package" else None


def validate_zero_state(package: Path, manifest: dict[str, Any], mode: str) -> None:
    validate_zero_state_paths(expected_paths(package, manifest), manifest["zero_state"], mode)


def validate_package(
    package: Path,
    mode: str,
    *,
    review_path: Path | None = None,
    authority_path: Path | None = None,
    receipt_path: Path | None = None,
    stdout_log: Path | None = None,
    stderr_log: Path | None = None,
) -> dict[str, Any]:
    package = package.resolve(strict=True)
    seal, manifest = validate_package_seal(package)
    validate_review_request(package, manifest)
    validate_contract(package, manifest)
    validate_provenance(package, manifest, seal)
    validate_zero_state(package, manifest, mode)
    review: dict[str, Any] | None = None
    if mode in {"review", "launch"}:
        require(review_path is not None, f"{mode} mode requires canonical review path")
        review = validate_review(package, manifest, review_path)
    else:
        require(review_path is None, "package mode forbids a review path")
    if mode == "launch":
        require(review is not None and review["verdict"] == "ACCEPT", "launch requires independent L2 ACCEPT")
        require(authority_path is not None, "launch requires manager authority")
        require(receipt_path is not None and stdout_log is not None and stderr_log is not None, "launch requires exact receipt and log paths")
        validate_authority(package, manifest, review_path, authority_path)
        validate_runner_state(manifest, receipt_path, stdout_log, stderr_log)
    else:
        require(authority_path is None and receipt_path is None and stdout_log is None and stderr_log is None, f"{mode} mode forbids launch artifacts")
    for script in ("launch-once.py", "validate-package.py", "test-launch-contract.py"):
        compile((package / script).read_text(encoding="ascii"), script, "exec")
    return {
        "status": "PASS",
        "mode": mode,
        "nonce": manifest["identity"]["nonce"],
        "package_seal_sha256": digest(package / "seal.json"),
        "zero_state": mode != "launch",
        "execution_authority_withheld": mode != "launch",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=("package", "review", "launch"), required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    args = parser.parse_args()
    result = validate_package(
        args.package,
        args.mode,
        review_path=args.review,
        authority_path=args.authority,
        receipt_path=args.receipt,
        stdout_log=args.stdout_log,
        stderr_log=args.stderr_log,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
