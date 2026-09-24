#!/usr/bin/env python3
"""Fail-closed validation for a Model24 r13 durable-launch package."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
from typing import Any


MODEL_PYTHON = "/home/argustest/miniconda3/bin/python3"
ARGUS_PYTHON = "/home/argustest/argustest2/argus-skill-latest/.venv/bin/python"
ARGUS_MODULE = "argus_skill.tools.subagent"
SOURCE_COMMIT = "42c895ce1e5fea00525e9f2f7fef66f0fbb8e118"
SOURCE_TREE = "93f57e9e0c87c3f8638282475e12f8f5289b498b"
CHECKPOINT_BYTES = 730652248
CHECKPOINT_SHA256 = "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
R12_TERMINAL_HASHES = {
    "manifest.json": "0c4699aac58439bc10a180cfad63fd1875c8a1559ee2030ff237720477b2b2f5",
    "runner-receipt.json": "71cea3855a0aa57d33bfacc1cafa0f738fd909384f75340742c9978edaeeade1",
    "exit-code.txt": "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
}
REQUIRED_REVIEW_HASHES = (
    "package_manifest_sha256",
    "package_seal_sha256",
    "review_request_sha256",
    "lifecycle_sha256",
    "validator_sha256",
    "launch_contract_sha256",
    "source_tree_sha256",
    "r12_terminal_evidence_sha256",
)
REQUIRED_POSITIVE_FIXTURES = (
    "package",
    "review",
    "pre-submit",
    "launch",
    "post-terminal",
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
    "conflated-registry-child-roots",
    "wrong-submitter-cwd",
    "temporary-terminal-root",
    "pre-launch-receipt-dependence",
    "foreign-terminal-artifact",
    "missing-terminal-artifact",
    "duplicate-payload-invocation",
    "canonical-artifact-from-inert-fixture",
)
FORBIDDEN_SEMANTICS = ("retry", "replay", "resume", "watcher")


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


def package_review_hashes(package: Path) -> dict[str, str]:
    return {
        "package_manifest_sha256": digest(package / "package.json"),
        "package_seal_sha256": digest(package / "seal.json"),
        "review_request_sha256": digest(package / "review-request.json"),
        "lifecycle_sha256": digest(package / "lifecycle.py"),
        "validator_sha256": digest(package / "validate-package.py"),
        "launch_contract_sha256": digest(package / "launch-contract.json"),
        "source_tree_sha256": digest(package / "source-tree.json"),
        "r12_terminal_evidence_sha256": digest(
            package / "provenance/r12-terminal/evidence.json"
        ),
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
        "terminal_manifest_paths": 0,
        "review_paths": 0,
        "execution_invocations": 0,
        "launcher_invocations": 0,
        "model_invocations": 0,
        "simulator_invocations": 0,
    }


def expected_paths(package: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    base = package.parent
    nonce = manifest["identity"]["nonce"]
    task_id = manifest["identity"]["task_id"]
    terminal_root = Path(
        f"/home/argustest/ace3-model24-r13-terminal-20260829-{nonce}"
    )
    registry_root = terminal_root / ".argus_subagents"
    log_dir = registry_root / f"{task_id}_logs"
    return {
        "base": base,
        "child_cwd": base / "source",
        "review": Path(
            f"/home/argustest/ace3-model24-r13-review-20260829-{nonce}"
        ),
        "authority": Path(
            f"/home/argustest/ace3-model24-r13-authority-20260829-{nonce}.json"
        ),
        "authority_consumed": Path(
            f"/home/argustest/ace3-model24-r13-authority-20260829-{nonce}.json.consumed"
        ),
        "output": Path(
            f"/home/argustest/ace3-model24-r13-output-20260829-{nonce}"
        ),
        "terminal_root": terminal_root,
        "submitter_root": terminal_root,
        "registry_root": registry_root,
        "receipt": registry_root / f"{task_id}.json",
        "log_dir": log_dir,
        "stdout_log": log_dir / "stdout.log",
        "stderr_log": log_dir / "stderr.log",
        "terminal_manifest": terminal_root / "manifest.json",
    }


def validate_terminal_root_value(root: Path) -> None:
    require(root.is_absolute(), "terminal root must be absolute")
    require(root.parent == Path("/home/argustest"), "terminal root is temporary or noncanonical")
    require(root.name.startswith("ace3-model24-r13-terminal-"), "terminal root name is noncanonical")


def validate_root_separation(manifest: dict[str, Any]) -> None:
    namespaces = manifest["namespaces"]
    submitter = Path(namespaces["submitter_root"])
    child = Path(namespaces["child_cwd"])
    registry = Path(namespaces["registry_root"])
    require(submitter != child, "submitter registry root is conflated with child cwd")
    require(registry == submitter / ".argus_subagents", "registry root is not derived from submitter cwd")
    validate_terminal_root_value(Path(namespaces["terminal_root"]))


def validate_submitter_cwd(manifest: dict[str, Any], actual_cwd: Path) -> None:
    expected = Path(manifest["execution"]["submitter_cwd"])
    require(actual_cwd.resolve() == expected.resolve(), "wrong submitter cwd")


def validate_package_seal(package: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    seal = load_json(package / "seal.json")
    require(seal["kind"] == "ace3_model24_r13_package_seal", "package seal kind mismatch")
    records = seal.get("package_entries")
    require(isinstance(records, list), "package seal entry list missing")
    actual = [
        record for record in tree_records(package) if record["path"] != "seal.json"
    ]
    require(actual == records, "package file set, mode, size, or hash differs from seal")
    require(package.stat().st_mode & 0o222 == 0, "package directory is writable")
    require((package / "seal.json").stat().st_mode & 0o222 == 0, "package seal is writable")
    for record in actual:
        require(record["mode"] & 0o222 == 0, f"sealed package entry is writable: {record['path']}")
    manifest = load_json(package / "package.json")
    require(digest(package / "package.json") == seal["package_manifest_sha256"], "manifest seal binding mismatch")
    require(digest(package / "review-request.json") == seal["review_request_sha256"], "review request seal binding mismatch")
    require(seal["zero_state"] is True, "package seal is not zero-state")
    return seal, manifest


def validate_source(package: Path, seal: dict[str, Any]) -> None:
    source_manifest = load_json(package / "source-tree.json")
    records = source_manifest.get("records")
    require(isinstance(records, list), "source-tree records missing")
    require(tree_sha256(records) == source_manifest["tree_sha256"], "source-tree manifest self-hash mismatch")
    require(source_manifest["tree_sha256"] == seal["source_tree_sha256"], "source tree is not bound by package seal")
    actual = tree_records(package.parent / "source")
    require(actual == records, "sealed source differs from manifest")
    require((package.parent / "source").stat().st_mode & 0o222 == 0, "sealed source root is writable")
    require(all(record["mode"] & 0o222 == 0 for record in actual), "sealed source entry is writable")


def validate_r12_evidence(package: Path, manifest: dict[str, Any]) -> None:
    root = package / "provenance/r12-terminal"
    require(
        {path.name for path in root.iterdir()}
        == {"manifest.json", "runner-receipt.json", "exit-code.txt", "evidence.json"},
        "r12 terminal evidence file set differs",
    )
    for name, expected in R12_TERMINAL_HASHES.items():
        require(digest(root / name) == expected, f"r12 terminal evidence changed: {name}")
    evidence = load_json(root / "evidence.json")
    require(evidence["original_hashes"] == R12_TERMINAL_HASHES, "r12 evidence hash manifest mismatch")
    require(evidence["canonical_stdout_stderr_existed"] is False, "r12 evidence reconstructs nonexistent logs")
    require(evidence["copied_byte_for_byte"] is True, "r12 evidence is not byte-preserving")
    provenance = manifest["provenance"]
    require(provenance["source"]["commit"] == SOURCE_COMMIT, "accepted source commit mismatch")
    require(provenance["source"]["tree"] == SOURCE_TREE, "accepted source tree mismatch")
    require(provenance["r12_terminal_evidence"]["hashes"] == R12_TERMINAL_HASHES, "manifest r12 evidence mismatch")
    checkpoint = provenance["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    require(checkpoint_path.is_file() and not checkpoint_path.is_symlink(), "bound checkpoint is unavailable")
    require(checkpoint_path.stat().st_size == CHECKPOINT_BYTES, "bound checkpoint size mismatch")
    require(checkpoint["bytes"] == CHECKPOINT_BYTES, "checkpoint byte binding mismatch")
    require(checkpoint["sha256"] == CHECKPOINT_SHA256, "checkpoint hash binding mismatch")
    require(digest(checkpoint_path) == CHECKPOINT_SHA256, "bound checkpoint changed")


def validate_contract(package: Path, manifest: dict[str, Any]) -> None:
    contract = load_json(package / "launch-contract.json")
    require(contract["kind"] == "ace3_model24_r13_durable_launch_contract", "launch contract kind mismatch")
    require(contract["review"]["required_negative_cases"] == list(REQUIRED_NEGATIVE_CASES), "negative case contract mismatch")
    require(contract["review"]["required_positive_fixtures"] == list(REQUIRED_POSITIVE_FIXTURES), "positive fixture contract mismatch")
    require(contract["review"]["required_bound_hashes"] == list(REQUIRED_REVIEW_HASHES), "review hash contract mismatch")
    require(contract["durable_runner"]["prelaunch_receipt_dependency"] is False, "pre-launch validation depends on a receipt")
    identity = manifest["identity"]
    require(identity["candidate"] == str(package.parent), "candidate path mismatch")
    require(identity["task_id"] == f"ace3-model24-r13-42c895c-{identity['nonce']}", "task identity mismatch")
    paths = expected_paths(package, manifest)
    namespaces = manifest["namespaces"]
    require(set(namespaces) == set(paths) - {"base"}, "namespace fields differ from r13 schema")
    for name, expected in paths.items():
        if name != "base":
            require(Path(namespaces[name]) == expected, f"canonical namespace mismatch: {name}")
    validate_root_separation(manifest)

    execution = manifest["execution"]
    require(execution["cardinality"] == 1, "execution cardinality is not one")
    require(execution["current_invocation_count"] == 0, "execution invocation count is not zero")
    require(all(execution[name] is False for name in FORBIDDEN_SEMANTICS), "retry/replay/resume/watcher policy is open")
    launch_argv = [
        MODEL_PYTHON,
        "-B",
        str(package / "lifecycle.py"),
        "launch",
        "--authority",
        str(paths["authority"]),
        "--review",
        str(paths["review"] / "review.json"),
    ]
    require(execution["launch_argv"] == launch_argv, "launcher argv differs from canonical contract")
    require(execution["durable_command"] == shlex.join(launch_argv), "durable command differs from canonical argv")
    require(execution["child_cwd"] == str(paths["child_cwd"]), "child cwd differs from sealed source")
    require(execution["submitter_cwd"] == str(paths["submitter_root"]), "submitter cwd differs from terminal root")
    submission_argv = [
        ARGUS_PYTHON,
        "-m",
        ARGUS_MODULE,
        "submit",
        "--task-id",
        identity["task_id"],
        "--description",
        "ACE-3 r13 Model24 native AWQ cascade, independently accepted and Manager-authorized exactly once",
        "--mode",
        "direct",
        "--timeout",
        "18000",
        "--command",
        execution["durable_command"],
        "--cwd",
        str(paths["child_cwd"]),
        "--cpu-count",
        "1",
    ]
    require(execution["submission_argv"] == submission_argv, "submission argv differs from canonical contract")
    output = paths["output"]
    payload_argv = [
        MODEL_PYTHON,
        "-B",
        str(paths["child_cwd"] / "ace3/model/controller_model24_rtl_cascade.py"),
        "--repository-root",
        str(paths["child_cwd"]),
        "--checkpoint",
        manifest["provenance"]["checkpoint"]["path"],
        "--tensor-map",
        str(paths["child_cwd"] / "ace3/contracts/model24_tensor_map.json"),
        "--bindings",
        str(package / "bindings.json"),
        "--simulation-dir",
        str(output / "controller-simulation"),
        "--output-dir",
        str(output / "rtl-cascade"),
        "--fresh",
    ]
    require(execution["payload_argv"] == payload_argv, "payload argv differs from fresh Model24 contract")
    require(execution["timeout_seconds"] == 18000, "execution timeout mismatch")


def validate_test_results(value: object, verdict: str) -> None:
    require(isinstance(value, dict), "review test results must be an object")
    for field, names in (
        ("positive_fixtures", REQUIRED_POSITIVE_FIXTURES),
        ("negative_cases", REQUIRED_NEGATIVE_CASES),
    ):
        results = value.get(field)
        require(isinstance(results, list), f"review {field} must be a list")
        require(
            [result.get("name") for result in results if isinstance(result, dict)]
            == list(names),
            f"review {field} names or order differ from required suite",
        )
        require(
            all(
                isinstance(result, dict)
                and set(result) == {"name", "status"}
                and result["status"] in {"PASS", "FAIL"}
                for result in results
            ),
            f"review {field} entries are malformed",
        )
    for field in (
        "execution_invocations",
        "launcher_invocations",
        "model_invocations",
        "simulator_invocations",
    ):
        require(value.get(field) == 0, f"review reports nonzero {field}")
    statuses = [
        result["status"]
        for field in ("positive_fixtures", "negative_cases")
        for result in value[field]
    ]
    if verdict == "ACCEPT":
        require(all(status == "PASS" for status in statuses), "ACCEPT review contains a failed inert result")
    else:
        require(any(status == "FAIL" for status in statuses), "REJECT review must identify a failed result")


def validate_review_document(
    package: Path,
    manifest: dict[str, Any],
    review_path: Path,
) -> dict[str, Any]:
    review = load_json(review_path)
    expected = {
        "schema_version": 1,
        "kind": "ace3_model24_r13_independent_l2_review",
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
    for key, expected_value in expected.items():
        require(review.get(key) == expected_value, f"review provenance or package binding mismatch: {key}")
    identity = review.get("reviewer_identity")
    require(isinstance(identity, str) and identity.strip(), "reviewer identity is missing")
    require(identity != manifest["review_policy"]["preparer_identity"], "reviewer is not independent of package preparation")
    verdict = review.get("verdict")
    require(verdict in {"ACCEPT", "REJECT"}, "review verdict must be ACCEPT or REJECT")
    validate_test_results(review.get("test_results"), verdict)
    require(review.get("zero_state_observation") == clean_zero_state_observation(), "review zero-state observation differs")
    require(review.get("bound_hashes") == package_review_hashes(package), "review does not bind all sealed artifacts")
    require(review.get("manager_may_issue_execution_authority") is (verdict == "ACCEPT"), "review authority disposition mismatch")
    require(review_path.stat().st_mode & 0o222 == 0, "review receipt is writable")
    return review


def expected_authority(
    package: Path,
    manifest: dict[str, Any],
    review_path: Path,
) -> dict[str, Any]:
    paths = expected_paths(package, manifest)
    execution = manifest["execution"]
    return {
        "schema_version": 1,
        "kind": "ace3_model24_r13_manager_exactly_once_authority",
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "review_path": str(review_path),
        "review_sha256": digest(review_path),
        **package_review_hashes(package),
        "submitter_cwd": str(paths["submitter_root"]),
        "child_cwd": str(paths["child_cwd"]),
        "submission_argv": execution["submission_argv"],
        "launch_argv": execution["launch_argv"],
        "durable_command": execution["durable_command"],
        "terminal_root": str(paths["terminal_root"]),
        "registry_root": str(paths["registry_root"]),
        "receipt_path": str(paths["receipt"]),
        "stdout_log": str(paths["stdout_log"]),
        "stderr_log": str(paths["stderr_log"]),
        "terminal_manifest": str(paths["terminal_manifest"]),
        "output_namespace": str(paths["output"]),
        "execution_cardinality": 1,
        "manager_exactly_once_directive": True,
        "retry": False,
        "replay": False,
        "resume": False,
        "watcher": False,
    }


def validate_authority_document(
    package: Path,
    manifest: dict[str, Any],
    review_path: Path,
    authority_path: Path,
) -> dict[str, Any]:
    authority = load_json(authority_path)
    require(authority == expected_authority(package, manifest, review_path), "authority does not exactly bind review, package, roots, and launch")
    require(authority_path.stat().st_mode & 0o222 == 0, "execution authority is writable")
    return authority


def validate_absent_paths(paths: dict[str, Path], names: tuple[str, ...], mode: str) -> None:
    for name in names:
        require(not paths[name].exists(), f"{mode} mode requires absent {name}")


def validate_terminal_artifact_paths(
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


def _receipt_path(root: Path, value: object) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else root / path


def validate_terminal_layout(
    root: Path,
    task_id: str,
    command: str,
    child_cwd: Path,
) -> dict[str, Any]:
    registry = root / ".argus_subagents"
    receipt_path = registry / f"{task_id}.json"
    log_dir = registry / f"{task_id}_logs"
    require(
        {path.name for path in root.iterdir()} == {".argus_subagents"},
        "foreign terminal artifact",
    )
    require(
        {path.name for path in registry.iterdir()} == {receipt_path.name, log_dir.name},
        "foreign or missing terminal registry artifact",
    )
    receipt = load_json(receipt_path)
    require(receipt.get("task_id") == task_id, "terminal receipt task mismatch")
    require(receipt.get("mode") == "direct", "terminal receipt mode mismatch")
    require(receipt.get("command") == command, "terminal receipt command mismatch")
    require(receipt.get("cwd") == str(child_cwd), "terminal receipt child cwd mismatch")
    require(all(receipt.get(name, False) is False for name in FORBIDDEN_SEMANTICS), "receipt enables retry/replay/resume/watcher")
    require(receipt.get("state") in {"done", "error", "timeout"}, "runner receipt is not terminal")
    run_id = receipt.get("run_id")
    require(
        isinstance(run_id, str)
        and run_id.startswith(task_id + "-")
        and run_id[len(task_id) + 1 :].isdigit(),
        "terminal run id mismatch",
    )
    stdout = log_dir / "stdout.log"
    stderr = log_dir / "stderr.log"
    exit_sidecar = log_dir / f"exit_code.{run_id}"
    require(
        {path.name for path in log_dir.iterdir()}
        == {stdout.name, stderr.name, exit_sidecar.name},
        "foreign or missing terminal log artifact",
    )
    require(_receipt_path(root, receipt.get("stdout_log")) == stdout, "terminal stdout receipt path mismatch")
    require(_receipt_path(root, receipt.get("stderr_log")) == stderr, "terminal stderr receipt path mismatch")
    require(exit_sidecar.read_text(encoding="ascii") == f"{receipt.get('exit_code')}\n", "terminal exit sidecar mismatch")
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "stdout": stdout,
        "stderr": stderr,
        "exit_sidecar": exit_sidecar,
    }


def validate_payload_cardinality(launch_terminal: dict[str, Any]) -> None:
    require(launch_terminal.get("invocation_count") == 1, "duplicate payload invocation")
    require(launch_terminal.get("cardinality") == 1, "payload cardinality mismatch")


def validate_package(
    package: Path,
    mode: str,
    *,
    review_path: Path | None = None,
    authority_path: Path | None = None,
) -> dict[str, Any]:
    package = package.resolve(strict=True)
    seal, manifest = validate_package_seal(package)
    validate_source(package, seal)
    validate_r12_evidence(package, manifest)
    validate_contract(package, manifest)
    paths = expected_paths(package, manifest)
    require(all(value == 0 for value in manifest["zero_state"].values()), "manifest reports nonzero execution state")
    review: dict[str, Any] | None = None
    if mode == "package":
        require(review_path is None and authority_path is None, "package mode forbids review or authority")
        validate_absent_paths(
            paths,
            ("review", "authority", "authority_consumed", "output", "terminal_root"),
            mode,
        )
    elif mode in {"review", "pre-submit", "launch"}:
        require(review_path == paths["review"] / "review.json", f"{mode} mode requires canonical review")
        review = validate_review_document(package, manifest, review_path)
        if mode == "review":
            require(authority_path is None, "review mode forbids authority")
            validate_absent_paths(
                paths,
                ("authority", "authority_consumed", "output", "terminal_root"),
                mode,
            )
        else:
            require(review["verdict"] == "ACCEPT", f"{mode} requires independent L2 ACCEPT")
            require(authority_path == paths["authority"], f"{mode} mode requires canonical authority")
            validate_authority_document(package, manifest, review_path, authority_path)
            validate_absent_paths(paths, ("authority_consumed", "output"), mode)
            if mode == "pre-submit":
                validate_absent_paths(paths, ("terminal_root",), mode)
            else:
                require(paths["terminal_root"].is_dir(), "launch terminal root is missing")
                require(not paths["terminal_manifest"].exists(), "terminal manifest already exists")
                validate_submitter_cwd(manifest, paths["submitter_root"])
                require(Path.cwd().resolve() == paths["child_cwd"].resolve(), "launch child cwd mismatch")
    else:
        raise SystemExit(f"unsupported validation mode: {mode}")
    for script in ("lifecycle.py", "validate-package.py", "test-launch-contract.py"):
        compile((package / script).read_text(encoding="ascii"), script, "exec")
    return {
        "status": "PASS",
        "mode": mode,
        "nonce": manifest["identity"]["nonce"],
        "package_seal_sha256": digest(package / "seal.json"),
        "receipt_required": False,
        "execution_authority_withheld": mode in {"package", "review"},
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=("package", "review", "pre-submit", "launch"), required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--authority", type=Path)
    args = parser.parse_args()
    result = validate_package(
        args.package,
        args.mode,
        review_path=args.review,
        authority_path=args.authority,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
