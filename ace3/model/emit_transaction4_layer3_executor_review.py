#!/usr/bin/env python3
"""Independently review the transaction004/layer03 executor package."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


ROOT = Path("/home/argustest/ace3-argus")
MISSION_ID = "06a9a74e72a0"
RUNTIME_IDENTITY = "ace3-position3-fresh-r11-20260831t215500z"
RUNTIME = ROOT / "build/model24_selected_token_position3_runs" / RUNTIME_IDENTITY
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
POINTER = ADOPTION / "authoritative-state.json"
GENERATION3 = ADOPTION / "state-generations/generation-0000000003"
GENERATION4 = ADOPTION / "state-generations/generation-0000000004"
TRANSACTIONS = RUNTIME / "transactions"
GENERATION5 = ADOPTION / "state-generations/generation-0000000005"
GENERATION5_STAGING = ADOPTION / "state-generations/.generation-0000000005.prepared"
FUTURE_ROOT = RUNTIME / "transaction4-authoritative-generation5"
AUTHORITY = (
    ROOT
    / "build/model24_selected_token_position3_transaction4_authorities"
    / "generation4-cursor4-transaction004-layer03-r1"
    / "manager-authority.json"
)
TRANSACTION_INDEX = 4
LAYER_INDEX = 3
FORBIDDEN_INDICES = [*range(4), *range(5, 26)]
ZERO_COUNTERS = {
    "authority_issuance": 0,
    "authority_consumption": 0,
    "model_execution": 0,
    "payload_execution": 0,
    "reference_execution": 0,
    "rtl_compile": 0,
    "rtl_simulation": 0,
    "runtime_mutation": 0,
    "submission": 0,
    "transaction_execution": 0,
}
MEMBERS = {
    "authoritative_baseline": "authoritative-baseline.json",
    "executor": "transaction4_executor.py",
    "package_manifest": "package-manifest.json",
    "review_emitter": "review-emitter.py",
    "review_request": "review-request.json",
    "source_manifest": "source-manifest.json",
}


class ReviewError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def canonical_json(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("ascii")


def load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in document, f"duplicate JSON key {key}: {path}")
            document[key] = value
        return document

    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"regular file required: {path}")
    require(not stat.S_ISLNK(metadata.st_mode), f"symlink rejected: {path}")
    document = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    require(isinstance(document, dict), f"JSON object required: {path}")
    return document


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"regular file required: {path}")
    require(not stat.S_ISLNK(metadata.st_mode), f"symlink rejected: {path}")
    return {
        "path": str(path),
        "bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def authenticate(
    record: Mapping[str, Any],
    label: str,
    expected_path: Path | None = None,
) -> None:
    require(
        set(record) == {"path", "bytes", "sha256"},
        f"{label} record malformed",
    )
    path = Path(record["path"])
    if expected_path is not None:
        require(path == expected_path, f"{label} path differs")
    require(file_record(path) == dict(record), f"{label} binding differs")


def tree_digest(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"tree root invalid: {root}")
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        record = file_record(path)
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
        count += 1
        total_bytes += record["bytes"]
    return {
        "root": str(root),
        "file_count": count,
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def validate_executor_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    require(
        constants.get("TRANSACTION_INDEX") == TRANSACTION_INDEX
        and constants.get("LAYER_INDEX") == LAYER_INDEX
        and constants.get("START_CURSOR") == 4
        and constants.get("EXIT_CURSOR") == 5,
        "executor transaction/layer/cursor constants differ",
    )
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    require(
        {"validate_generation4", "validate_absence", "execute_once"}
        <= function_names,
        "executor lacks bounded parent, absence, or execute gate",
    )
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    require(
        "execute_exact_layer_transaction" in calls,
        "executor does not invoke the accepted exact layer primitive",
    )


def validate_manifest(document: Mapping[str, Any]) -> None:
    require(
        document.get("kind")
        == "ace3_position3_transaction4_layer3_executor_package"
        and document.get("status") == "SEALED_REVIEW_REQUIRED"
        and document.get("mission_id") == MISSION_ID
        and document.get("runtime_identity") == RUNTIME_IDENTITY
        and document.get("authoritative_generation") == 4
        and document.get("authoritative_cursor") == 4
        and document.get("required_parent_checkpoint_index") == 3
        and document.get("transaction_index") == TRANSACTION_INDEX
        and document.get("layer_index") == LAYER_INDEX
        and document.get("permitted_transaction_indices") == [4]
        and document.get("forbidden_transaction_indices") == FORBIDDEN_INDICES
        and document.get("transaction_descriptor", {}).get("transaction_index")
        == TRANSACTION_INDEX
        and document.get("transaction_descriptor", {}).get("layer_index")
        == LAYER_INDEX
        and document.get("transaction_descriptor", {})
        .get("inputs", {})
        .get("predecessor", {})
        .get("source_transaction_index")
        == 3
        and document.get("execution_authorized") is False
        and document.get("authority_created") is False
        and document.get("activity_counters") == ZERO_COUNTERS,
        "manifest transaction scope, parent, or zero state differs",
    )
    launch = document.get("launch", {})
    require(
        launch.get("cwd") == str(ROOT)
        and launch.get("transaction_index") == TRANSACTION_INDEX
        and launch.get("layer_index") == LAYER_INDEX
        and launch.get("start_generation") == 4
        and launch.get("start_cursor") == 4
        and launch.get("exit_generation") == 5
        and launch.get("exit_cursor") == 5
        and launch.get("environment")
        == {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
        and launch.get("argv", [None])[-1] == "4",
        "exact launch contract differs",
    )


def validate_package(package: Path) -> dict[str, Any]:
    manifest = load_json(package / "package-manifest.json")
    seal = load_json(package / "package-seal.json")
    baseline = load_json(package / "authoritative-baseline.json")
    sources = load_json(package / "source-manifest.json")
    validate_manifest(manifest)
    validate_executor_source(package / "transaction4_executor.py")
    require(
        seal.get("kind")
        == "ace3_position3_transaction4_layer3_executor_seal"
        and seal.get("status") == "SEALED_REVIEW_REQUIRED"
        and seal.get("execution_authorized") is False
        and seal.get("authority_created") is False
        and seal.get("activity_counters") == ZERO_COUNTERS
        and set(seal.get("members", {})) == set(MEMBERS),
        "package seal differs",
    )
    for label, relative in MEMBERS.items():
        authenticate(
            seal["members"][label],
            f"sealed {label}",
            package / relative,
        )
    for label, record in sources.get("sources", {}).items():
        authenticate(record, f"source {label}")
    for label, record in sources.get("toolchain", {}).items():
        requested = Path(record["requested_path"])
        require(
            requested.resolve(strict=True) == Path(record["resolved"]["path"]),
            f"tool resolution differs: {label}",
        )
        authenticate(record["resolved"], f"tool {label}")
    require(
        baseline.get("authoritative_generation") == 4
        and baseline.get("authoritative_cursor") == 4
        and baseline.get("activity_counters") == ZERO_COUNTERS
        and baseline.get("checkpoint003_parent")
        == file_record(GENERATION4 / "checkpoints/transaction-003.json")
        and baseline.get("transactions000_003")
        == [
            tree_digest(TRANSACTIONS / f"transaction-{index:03d}")
            for index in range(4)
        ],
        "baseline parent, preserved transactions, or counters differ",
    )
    authenticate(baseline["pointer"], "baseline pointer", POINTER)
    for section, generation, count in (
        ("generation3", GENERATION3, 3),
        ("generation4", GENERATION4, 4),
    ):
        authenticate(
            baseline[section]["manifest"],
            f"{section} manifest",
            generation / "generation-manifest.json",
        )
        authenticate(
            baseline[section]["ledger"],
            f"{section} ledger",
            generation / "ledger.json",
        )
        require(
            len(baseline[section]["checkpoints"]) == count,
            f"{section} checkpoint count differs",
        )
        for index, record in enumerate(baseline[section]["checkpoints"]):
            authenticate(
                record,
                f"{section} checkpoint{index:03d}",
                generation / f"checkpoints/transaction-{index:03d}.json",
            )
    pointer = load_json(POINTER)
    ledger = load_json(GENERATION4 / "ledger.json")
    require(
        pointer.get("generation") == 4
        and pointer.get("status") == "COMMITTED"
        and pointer.get("generation_manifest")
        == file_record(GENERATION4 / "generation-manifest.json")
        and ledger.get("state_generation") == 4
        and ledger.get("completed_transaction_count") == 4
        and ledger.get("next_transaction_index") == 4,
        "live authoritative generation4/cursor4 differs",
    )
    require(
        all(
            not (TRANSACTIONS / f"transaction-{index:03d}").exists()
            for index in range(4, 26)
        )
        and not GENERATION5.exists()
        and not GENERATION5_STAGING.exists()
        and not FUTURE_ROOT.exists()
        and not AUTHORITY.exists(),
        "authority, transaction004-025, or generation5 activity exists",
    )
    return manifest


def adversarial_controls(manifest: Mapping[str, Any]) -> list[str]:
    controls = []
    mutations = (
        ("wrong-transaction", lambda item: item.update({"transaction_index": 5})),
        ("wrong-layer", lambda item: item.update({"layer_index": 4})),
        (
            "wrong-parent",
            lambda item: item.update({"required_parent_checkpoint_index": 2}),
        ),
        (
            "broad-scope",
            lambda item: item.update({"permitted_transaction_indices": [4, 5]}),
        ),
        (
            "nonzero-counter",
            lambda item: item["activity_counters"].update(
                {"transaction_execution": 1}
            ),
        ),
        (
            "premature-authority",
            lambda item: item.update({"execution_authorized": True}),
        ),
    )
    for name, mutate in mutations:
        candidate = copy.deepcopy(dict(manifest))
        mutate(candidate)
        try:
            validate_manifest(candidate)
        except ReviewError:
            controls.append(name)
        else:
            raise ReviewError(f"adversarial control accepted: {name}")
    return controls


def assess_package(package: Path) -> tuple[str, list[str], str | None]:
    try:
        manifest = validate_package(package)
        controls = adversarial_controls(manifest)
    except (ReviewError, OSError, ValueError, KeyError, TypeError) as error:
        return "REJECT", [], str(error)
    return "PASS", controls, None


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def emit_review(package: Path, output: Path) -> None:
    require(package.is_absolute(), "package path must be absolute")
    require(output.is_absolute(), "review output path must be absolute")
    require(not output.exists(), f"review output already exists: {output}")
    status, controls, reason = assess_package(package)
    output.parent.mkdir(parents=True, exist_ok=True)
    review = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction4_layer3_independent_review",
        "status": status,
        "producer_role": "reviewer",
        "mission_id": MISSION_ID,
        "preparation_participation": False,
        "package_seal": file_record(package / "package-seal.json"),
        "source_manifest": file_record(package / "source-manifest.json"),
        "authoritative_baseline": file_record(
            package / "authoritative-baseline.json"
        ),
        "permitted_launch": load_json(package / "package-manifest.json")["launch"],
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        "transaction004_executed": False,
        "transactions005_025_absent": True,
        "this_review_authorizes_execution": False,
        "adversarial_controls": controls,
        "reason": reason,
    }
    write_new(output, canonical_json(review))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    emit_review(arguments.package, arguments.output)
    print(f"TRANSACTION004_LAYER03_INDEPENDENT_REVIEW_WRITTEN output={arguments.output}")


if __name__ == "__main__":
    try:
        main()
    except (
        ReviewError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise SystemExit(f"TRANSACTION004_LAYER03_REVIEW_REFUSED {error}") from error
