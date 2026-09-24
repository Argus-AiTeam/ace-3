#!/usr/bin/env python3
"""Source-disjoint review of the transaction005/layer04 executor package."""

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
MISSION_ID = "43c0b6ae6c52"
PARENT_MISSION_ID = "d560dc6d3138"
RUNTIME_IDENTITY = "ace3-position3-fresh-r11-20260831t215500z"
RUNTIME = ROOT / "build/model24_selected_token_position3_runs" / RUNTIME_IDENTITY
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
POINTER = ADOPTION / "authoritative-state.json"
GENERATIONS = ADOPTION / "state-generations"
GENERATION3 = GENERATIONS / "generation-0000000003"
GENERATION4 = GENERATIONS / "generation-0000000004"
GENERATION5 = GENERATIONS / "generation-0000000005"
GENERATION6 = GENERATIONS / "generation-0000000006"
GENERATION6_STAGING = GENERATIONS / ".generation-0000000006.prepared"
TRANSACTIONS = RUNTIME / "transactions"
TRANSACTION5 = TRANSACTIONS / "transaction-005"
FUTURE_ROOT = RUNTIME / "transaction5-authoritative-generation6"
PACKAGE_ID = "generation5-cursor5-checkpoint004-transaction005-layer04-r1"
PACKAGE_ROOT = (
    ROOT / "build/model24_selected_token_position3_transaction5_packages" / PACKAGE_ID
)
REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction5_reviews"
    / PACKAGE_ID
    / "independent-review.json"
)
AUTHORITY = (
    ROOT
    / "build/model24_selected_token_position3_transaction5_authorities"
    / PACKAGE_ID
    / "manager-authority.json"
)
ORIGINAL_INVOCATION = RUNTIME / "invocation.json"
MODEL_CHECKPOINT = ROOT / "build/model24_rtl_cascade/checkpoint/model.safetensors"
PARENT_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/d560dc6d3138/latest.json"
)
PARENT_REVIEW_DECISION = PARENT_HANDOFF.with_name("round-0001.json")
PYTHON = Path("/home/argustest/miniconda3/bin/python3")

TRANSACTION_INDEX = 5
LAYER_INDEX = 4
START_CURSOR = 5
EXIT_CURSOR = 6
POSITION = 3
PERMITTED_INDICES = [TRANSACTION_INDEX]
FORBIDDEN_INDICES = [*range(5), *range(6, 26)]

ZERO_COUNTERS = {
    "authority_issuance": 0,
    "authority_consumption": 0,
    "model_execution": 0,
    "oracle_execution": 0,
    "payload_execution": 0,
    "rtl_compile": 0,
    "rtl_simulation": 0,
    "runtime_mutation": 0,
    "submission": 0,
    "transaction_execution": 0,
    "vector_generation": 0,
}

FIXED_SHA256 = {
    PARENT_HANDOFF: "8191adb23a56b8ddc463595bdfbd4c414550cfbe8a5c19c11a633f00f255168c",
    PARENT_REVIEW_DECISION: "b6d400df449af5699ce3c84ad001d8e687b54746b3be348ad354b9cc415e902d",
    POINTER: "adee838870e758a5cc32a1a79248938b22d8e9d701501dce525c9098958bdda0",
    GENERATION5
    / "generation-manifest.json": "03f73918812f160cb4b9b7f94b239c5d3efb9cab9d4d41bd621a59eadfc9b599",
    GENERATION5
    / "ledger.json": "ca9397fc658ffe0629c8ba4cade0af5640d85298106b4dfc0dfbeb2ca039af14",
    GENERATION5
    / "checkpoints/transaction-004.json": "ff157b01b83d07d8bf167c420115659072a54415054454674625503d7decd610",
    TRANSACTIONS
    / "transaction-004/position004.state": "bc17b1015874eba58edd49c6b29c74b37057576823711f5983476264de0c65ac",
    ORIGINAL_INVOCATION: "9fb5418fe06c492633399ea79791a08ae25507f3e74b0cf0c003bdb79b43919d",
    MODEL_CHECKPOINT: "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b",
}

MEMBERS = {
    "authoritative_baseline": "authoritative-baseline.json",
    "executor": "transaction5_executor.py",
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
        set(record) == {"path", "bytes", "sha256"}
        and isinstance(record.get("path"), str)
        and type(record.get("bytes")) is int
        and isinstance(record.get("sha256"), str),
        f"{label} record malformed",
    )
    path = Path(record["path"])
    if expected_path is not None:
        require(path == expected_path, f"{label} path differs")
    require(file_record(path) == dict(record), f"{label} binding differs")


def require_fixed(path: Path) -> dict[str, Any]:
    record = file_record(path)
    require(record["sha256"] == FIXED_SHA256[path], f"fixed hash mismatch: {path}")
    return record


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


def expected_descriptor() -> dict[str, Any]:
    require_fixed(ORIGINAL_INVOCATION)
    invocation = load_json(ORIGINAL_INVOCATION)
    descriptor = copy.deepcopy(invocation["transactions"][TRANSACTION_INDEX])
    require(
        descriptor.get("transaction_index") == TRANSACTION_INDEX
        and descriptor.get("layer_index") == LAYER_INDEX
        and descriptor.get("operation") == "position3-decoder-layer"
        and descriptor.get("inputs", {})
        .get("predecessor", {})
        .get("source_transaction_index")
        == 4
        and descriptor.get("inputs", {}).get("transaction_position") == POSITION,
        "transaction005 descriptor differs",
    )
    authenticate(descriptor["inputs"]["fixture_manifest"], "layer04 fixture")
    authenticate(descriptor["inputs"]["position2_kv_parent"], "layer04 K/V parent")
    return descriptor


def expected_official_evidence() -> dict[str, Any]:
    invocation = load_json(ORIGINAL_INVOCATION)
    checkpoint = invocation["transactions"][0]["inputs"]["checkpoint"]
    projection = invocation["transactions"][0]["inputs"]["tied_weight"]
    authenticate(checkpoint, "official model checkpoint", MODEL_CHECKPOINT)
    require_fixed(MODEL_CHECKPOINT)
    require(
        projection
        == {
            "dtype": "FP16",
            "sha256": "d74257dc547b48be5ae7b93f1c9af072c0c42dbbb85503078e25c59cd09e68d0",
            "shape": [151936, 896],
            "tensor": "model.embed_tokens.weight",
            "tied_peer": "lm_head.weight",
        },
        "official tied projection differs",
    )
    descriptor = expected_descriptor()
    return {
        "invocation": file_record(ORIGINAL_INVOCATION),
        "model_checkpoint": copy.deepcopy(checkpoint),
        "tied_projection": copy.deepcopy(projection),
        "layer04_fixture_manifest": copy.deepcopy(
            descriptor["inputs"]["fixture_manifest"]
        ),
        "layer04_position2_kv_parent": copy.deepcopy(
            descriptor["inputs"]["position2_kv_parent"]
        ),
    }


def expected_compile_argv() -> list[str]:
    return [
        "make",
        "--no-print-directory",
        "model24-rtl-layer-compile",
        "MODEL24_RTL_LAYER_INDEX=4",
        "MODEL24_RTL_ACCURATE_SILU=1",
        f"MODEL24_RTL_CASCADE_DIR={TRANSACTION5 / 'build'}",
    ]


def expected_simulation_argv() -> list[str]:
    descriptor = expected_descriptor()
    return [
        str(
            TRANSACTION5
            / "build/compiled/layer4/obj_dir/Vace3_decoder_layer0_token_engine"
        ),
        "--layer-index",
        "4",
        "--vector-dir",
        str(TRANSACTION5 / "vectors"),
        "--tensor-dir",
        str(TRANSACTION5 / "vectors"),
        "--raw-dir",
        str(TRANSACTION5 / "position003/raw"),
        "--transaction-position",
        "3",
        "--state-out",
        str(TRANSACTION5 / "position004.state"),
        "--progress-interval",
        "1000000",
        "--state-in",
        str(descriptor["inputs"]["position2_kv_parent"]["path"]),
    ]


def expected_launch(package: Path) -> dict[str, Any]:
    return {
        "cwd": str(ROOT),
        "argv": [
            str(PYTHON),
            str(package / "transaction5_executor.py"),
            "execute",
            "--package",
            str(package),
            "--review",
            str(REVIEW),
            "--authorization",
            str(AUTHORITY),
            "--runtime-root",
            str(RUNTIME),
            "--transaction-index",
            "5",
        ],
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        },
        "interpreter": str(PYTHON),
        "transaction_index": TRANSACTION_INDEX,
        "layer_index": LAYER_INDEX,
        "start_generation": START_CURSOR,
        "start_cursor": START_CURSOR,
        "exit_generation": EXIT_CURSOR,
        "exit_cursor": EXIT_CURSOR,
    }


def validate_live_parent() -> None:
    require_fixed(PARENT_HANDOFF)
    require_fixed(PARENT_REVIEW_DECISION)
    handoff = load_json(PARENT_HANDOFF)
    decision = load_json(PARENT_REVIEW_DECISION)
    require(
        handoff.get("kind") == "handoff_ref"
        and handoff.get("handoff", {}).get("path") == str(PARENT_REVIEW_DECISION)
        and decision.get("kind") == "round_reviewed_handoff"
        and decision.get("mission_id") == PARENT_MISSION_ID
        and decision.get("producer_role") == "reviewer"
        and decision.get("review", {}).get("status") == "done",
        "parent Reviewer acceptance differs",
    )
    manifest_record = require_fixed(GENERATION5 / "generation-manifest.json")
    require_fixed(GENERATION5 / "ledger.json")
    require_fixed(GENERATION5 / "checkpoints/transaction-004.json")
    position004_record = require_fixed(
        TRANSACTIONS / "transaction-004/position004.state"
    )
    require_fixed(POINTER)
    pointer = load_json(POINTER)
    manifest = load_json(GENERATION5 / "generation-manifest.json")
    ledger = load_json(GENERATION5 / "ledger.json")
    checkpoint4 = load_json(GENERATION5 / "checkpoints/transaction-004.json")
    checkpoints = [
        file_record(GENERATION5 / f"checkpoints/transaction-{index:03d}.json")
        for index in range(5)
    ]
    require(
        pointer.get("kind")
        == "ace3_position3_transaction4_generation5_authoritative_pointer"
        and pointer.get("status") == "COMMITTED"
        and pointer.get("generation") == START_CURSOR
        and pointer.get("generation_manifest") == manifest_record
        and manifest.get("kind")
        == "ace3_transaction4_publication_recovery_generation5_state"
        and manifest.get("status") == "PREPARED"
        and manifest.get("generation") == START_CURSOR
        and manifest.get("transaction004_retry_replay_resume") is False
        and manifest.get("transactions005_025_executed") is False
        and ledger.get("state_generation") == START_CURSOR
        and ledger.get("completed_transaction_count") == START_CURSOR
        and ledger.get("next_transaction_index") == TRANSACTION_INDEX
        and ledger.get("completed_receipts") == checkpoints
        and checkpoint4.get("kind") == "ace3_position3_transaction_completion"
        and checkpoint4.get("status") == "COMPLETE"
        and checkpoint4.get("transaction_index") == 4
        and checkpoint4.get("result", {}).get("exact_integer_oracle_match") is True
        and checkpoint4.get("result", {}).get("natural_rtl_terminal") is True
        and checkpoint4.get("outputs", {}).get("state") == position004_record,
        "generation5/cursor5/checkpoint004 parent differs",
    )
    for record in manifest.get("files", {}).values():
        authenticate(record, "generation5 manifest file")
    authenticate(
        {
            key: checkpoint4["outputs"]["hidden"][key]
            for key in ("path", "bytes", "sha256")
        },
        "checkpoint004 hidden output",
    )
    expected_official_evidence()


def validate_executor_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants: dict[str, object] = {}
    functions: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
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
        and constants.get("START_CURSOR") == START_CURSOR
        and constants.get("EXIT_CURSOR") == EXIT_CURSOR
        and constants.get("POSITION") == POSITION,
        "executor transaction/layer/parent constants differ",
    )
    require(
        {
            "validate_parent_acceptance",
            "validate_absence",
            "prepare_package",
            "execute_once",
            "publish_generation6",
        }
        <= set(functions),
        "executor lacks additive-generation gates",
    )
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    require(
        "execute_exact_layer_transaction" in calls,
        "executor does not call the accepted exact layer primitive",
    )
    prepare_calls = {
        node.func.id
        for node in ast.walk(functions["prepare_package"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    require(
        {
            "execute_once",
            "publish_generation6",
            "import_module",
            "validate_authority",
        }.isdisjoint(prepare_calls),
        "package preparation reaches execution or authority code",
    )


def validate_manifest(document: Mapping[str, Any], package: Path) -> None:
    checkpoint4 = load_json(GENERATION5 / "checkpoints/transaction-004.json")
    require(
        document.get("schema_version") == 1
        and document.get("kind")
        == "ace3_position3_transaction5_layer4_executor_package"
        and document.get("status") == "SEALED_REVIEW_REQUIRED"
        and document.get("mission_id") == MISSION_ID
        and document.get("parent_mission_id") == PARENT_MISSION_ID
        and document.get("runtime_identity") == RUNTIME_IDENTITY
        and document.get("authoritative_generation") == START_CURSOR
        and document.get("authoritative_cursor") == START_CURSOR
        and document.get("required_parent_checkpoint_index") == 4
        and document.get("transaction_index") == TRANSACTION_INDEX
        and document.get("layer_index") == LAYER_INDEX
        and document.get("transaction_position") == POSITION
        and document.get("permitted_transaction_indices") == PERMITTED_INDICES
        and document.get("forbidden_transaction_indices") == FORBIDDEN_INDICES
        and document.get("transaction_descriptor") == expected_descriptor()
        and document.get("position004_input_state") == checkpoint4["outputs"]["state"]
        and document.get("official_frozen_evidence") == expected_official_evidence()
        and document.get("compile")
        == {"cwd": str(ROOT), "argv": expected_compile_argv()}
        and document.get("simulation")
        == {"cwd": str(ROOT), "argv": expected_simulation_argv()}
        and document.get("launch") == expected_launch(package)
        and document.get("transaction005_output_namespace") == str(TRANSACTION5)
        and document.get("generation6_output_namespace") == str(GENERATION6)
        and document.get("review_output") == str(REVIEW)
        and document.get("future_manager_authority") == str(AUTHORITY)
        and document.get("execution_authorized") is False
        and document.get("authority_created") is False
        and document.get("activity_counters") == ZERO_COUNTERS,
        "manifest identity, evidence, argv, scope, or zero state differs",
    )
    require(
        document.get("prohibitions")
        == [
            "transaction000-004 replay",
            "transaction006-025 execution",
            "model, oracle, or vector execution during package review",
            "RTL compile or simulation during package review",
            "generation6 publication during package review",
            "authority creation by this package stage",
            "execution before separate Manager authority and Reviewer PASS",
        ],
        "manifest prohibitions differ",
    )


def validate_package(package: Path) -> dict[str, Any]:
    require(package == PACKAGE_ROOT, "package path differs")
    manifest = load_json(package / "package-manifest.json")
    seal = load_json(package / "package-seal.json")
    baseline = load_json(package / "authoritative-baseline.json")
    sources = load_json(package / "source-manifest.json")
    validate_manifest(manifest, package)
    validate_executor_source(package / "transaction5_executor.py")
    require(
        seal.get("kind") == "ace3_position3_transaction5_layer4_executor_seal"
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
    require(
        sources.get("kind")
        == "ace3_position3_transaction5_layer4_exact_source_toolchain_closure",
        "source/toolchain closure kind differs",
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
    authenticate(
        manifest["source_manifest"],
        "manifest source closure",
        package / "source-manifest.json",
    )
    authenticate(
        manifest["authoritative_baseline"],
        "manifest baseline",
        package / "authoritative-baseline.json",
    )
    require(
        baseline.get("kind")
        == "ace3_position3_transaction5_layer4_authoritative_baseline"
        and baseline.get("authoritative_generation") == START_CURSOR
        and baseline.get("authoritative_cursor") == START_CURSOR
        and baseline.get("activity_counters") == ZERO_COUNTERS
        and baseline.get("transaction005_absent") is True
        and baseline.get("transactions006_025_absent") is True
        and baseline.get("generation6_absent") is True
        and baseline.get("generation6_staging_absent") is True
        and baseline.get("transaction005_output_namespace") == str(TRANSACTION5),
        "baseline identity or zero state differs",
    )
    authenticate(
        baseline["parent_acceptance"]["index"],
        "parent acceptance index",
        PARENT_HANDOFF,
    )
    authenticate(
        baseline["parent_acceptance"]["reviewer_decision"],
        "parent Reviewer decision",
        PARENT_REVIEW_DECISION,
    )
    authenticate(baseline["pointer"], "baseline pointer", POINTER)
    authenticate(
        baseline["checkpoint004_parent"],
        "baseline checkpoint004",
        GENERATION5 / "checkpoints/transaction-004.json",
    )
    checkpoint4 = load_json(GENERATION5 / "checkpoints/transaction-004.json")
    require(
        baseline.get("position004_input_state") == checkpoint4["outputs"]["state"]
        and baseline.get("predecessor_hidden") == checkpoint4["outputs"]["hidden"]
        and baseline.get("official_frozen_evidence") == expected_official_evidence()
        and baseline.get("generation_trees")
        == [
            tree_digest(generation)
            for generation in (GENERATION3, GENERATION4, GENERATION5)
        ]
        and baseline.get("transactions000_004")
        == [
            tree_digest(TRANSACTIONS / f"transaction-{index:03d}")
            for index in range(5)
        ],
        "baseline parent evidence or preserved tree differs",
    )
    validate_live_parent()
    require(
        all(
            not (TRANSACTIONS / f"transaction-{index:03d}").exists()
            for index in range(5, 26)
        )
        and not GENERATION6.exists()
        and not GENERATION6_STAGING.exists()
        and not FUTURE_ROOT.exists()
        and not AUTHORITY.exists(),
        "authority, transaction005-025, or generation6 activity exists",
    )
    return manifest


def adversarial_controls(
    manifest: Mapping[str, Any],
    package: Path,
) -> list[str]:
    controls: list[str] = []
    mutations = (
        ("wrong-transaction", lambda item: item.update({"transaction_index": 6})),
        ("wrong-layer", lambda item: item.update({"layer_index": 5})),
        (
            "wrong-parent",
            lambda item: item.update(
                {
                    "authoritative_generation": 4,
                    "required_parent_checkpoint_index": 3,
                }
            ),
        ),
        (
            "wrong-position004-state",
            lambda item: item["position004_input_state"].update(
                {"sha256": "0" * 64}
            ),
        ),
        (
            "wrong-model",
            lambda item: item["official_frozen_evidence"][
                "model_checkpoint"
            ].update({"sha256": "0" * 64}),
        ),
        (
            "wrong-projection",
            lambda item: item["official_frozen_evidence"][
                "tied_projection"
            ].update({"sha256": "0" * 64}),
        ),
        (
            "wrong-compile-argv",
            lambda item: item["compile"]["argv"].__setitem__(
                3,
                "MODEL24_RTL_LAYER_INDEX=3",
            ),
        ),
        (
            "wrong-simulation-argv",
            lambda item: item["simulation"]["argv"].__setitem__(2, "3"),
        ),
        (
            "wrong-output-namespace",
            lambda item: item.update(
                {
                    "transaction005_output_namespace": str(
                        TRANSACTIONS / "transaction-006"
                    )
                }
            ),
        ),
        (
            "broad-scope",
            lambda item: item.update({"permitted_transaction_indices": [5, 6]}),
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
            validate_manifest(candidate, package)
        except ReviewError:
            controls.append(name)
        else:
            raise ReviewError(f"adversarial control accepted: {name}")
    return controls


def assess_package(package: Path) -> tuple[str, list[str], str | None]:
    try:
        manifest = validate_package(package)
        controls = adversarial_controls(manifest, package)
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
    require(output == REVIEW, "canonical review output path differs")
    require(not output.exists(), f"review output already exists: {output}")
    status, controls, reason = assess_package(package)
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline = load_json(package / "authoritative-baseline.json")
    review = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction5_layer4_independent_review",
        "status": status,
        "producer_role": "reviewer",
        "mission_id": MISSION_ID,
        "preparation_participation": False,
        "execution_import_participation": False,
        "package_seal": file_record(package / "package-seal.json"),
        "source_manifest": file_record(package / "source-manifest.json"),
        "authoritative_baseline": file_record(
            package / "authoritative-baseline.json"
        ),
        "parent_reviewer_decision": file_record(PARENT_REVIEW_DECISION),
        "permitted_launch": load_json(package / "package-manifest.json")["launch"],
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        "preserved_generation_trees": baseline["generation_trees"],
        "preserved_transaction_trees": baseline["transactions000_004"],
        "transaction005_executed": False,
        "generation6_published": False,
        "transactions006_025_absent": True,
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
    print(
        "TRANSACTION005_LAYER04_INDEPENDENT_REVIEW_WRITTEN "
        f"output={arguments.output}"
    )


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
        raise SystemExit(f"TRANSACTION005_LAYER04_REVIEW_REFUSED {error}") from error
