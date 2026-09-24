#!/usr/bin/env python3
"""Seal transaction007's post-computation publication-recovery evidence."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Mapping, Sequence


ROOT = Path("/home/argustest/ace3-argus")
RUNTIME_IDENTITY = "ace3-position3-fresh-r11-20260831t215500z"
RUNTIME = ROOT / "build/model24_selected_token_position3_runs" / RUNTIME_IDENTITY
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
GENERATIONS = ADOPTION / "state-generations"
POINTER = ADOPTION / "authoritative-state.json"
GENERATION7 = GENERATIONS / "generation-0000000007"
GENERATION8 = GENERATIONS / "generation-0000000008"
GENERATION8_STAGING = GENERATIONS / ".generation-0000000008.prepared"
TRANSACTIONS = RUNTIME / "transactions"
TRANSACTION7 = TRANSACTIONS / "transaction-007"
FUTURE = RUNTIME / "transaction7-authoritative-generation8"
CONSUMPTION = FUTURE / "manager-authorization-consumption.json"
EXECUTION_START = FUTURE / "execution-start.json"
FAILURE_TERMINAL = FUTURE / "fail-closed-terminal.json"
PACKAGE = RUNTIME / "transaction7-layer6-continuation-package-r1"
PACKAGE_MANIFEST = PACKAGE / "package-manifest.json"
PACKAGE_SEAL = PACKAGE / "package-seal.json"
EXACT_EXECUTOR = RUNTIME / "recovery-r12/recovery_executor.py"
TERMINAL_PARSER = ROOT / "ace3/model/validate_selected_token_position2_traversal.py"
OUTPUT = (
    ROOT
    / "build/model24_selected_token_position3_transaction7_recovery"
    / "r1-postconsume-natural-terminal-mapping"
)
REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction7_recovery"
    / "reviews/r1-postconsume-natural-terminal-mapping/independent-review.json"
)
REVIEW_SOURCE = ROOT / "ace3/model/emit_transaction7_publication_recovery_review.py"
HIDDEN_ELEMENTS = 896
ZERO_ACTIVITY = {
    "authority_consumption": 0,
    "generation8_publication": 0,
    "model_execution": 0,
    "oracle_execution": 0,
    "rtl_compile": 0,
    "rtl_simulation": 0,
    "synthesis": 0,
    "transaction_execution": 0,
    "transaction_replay": 0,
    "transaction_resume": 0,
    "transaction_retry": 0,
    "transactions008_025_execution": 0,
    "u280": 0,
}
PACKAGE_MEMBERS = {
    "adjudication.json",
    "package-manifest.json",
    "recovery-adjudication.py",
    "review-emitter.py",
    "review-request.json",
}
FORBIDDEN_IMPORTS = {"numpy", "safetensors", "subprocess", "torch"}
FORBIDDEN_CALLS = {
    "execute_exact_layer_transaction",
    "execute_layer_transaction",
    "popen",
    "run",
    "system",
}


class RecoveryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


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
    require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"regular non-symlink file required: {path}",
    )
    document = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    require(isinstance(document, dict), f"JSON object required: {path}")
    return document


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, published_path: Path | None = None) -> dict[str, Any]:
    metadata = path.lstat()
    require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"regular non-symlink file required: {path}",
    )
    return {
        "path": str(published_path or path),
        "bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def authenticate(record: Mapping[str, Any], label: str) -> None:
    require(
        set(record) == {"path", "bytes", "sha256"},
        f"{label} record malformed",
    )
    require(file_record(Path(record["path"])) == dict(record), f"{label} differs")


def tree_inventory(root: Path) -> list[dict[str, Any]]:
    require(root.is_dir() and not root.is_symlink(), f"tree root invalid: {root}")
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        record = file_record(path)
        inventory.append(
            {"relative_path": path.relative_to(root).as_posix(), **record}
        )
    return inventory


def tree_summary(root: Path) -> dict[str, Any]:
    inventory = tree_inventory(root)
    digest = hashlib.sha256()
    for record in inventory:
        digest.update(record["relative_path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "root": str(root),
        "file_count": len(inventory),
        "total_bytes": sum(record["bytes"] for record in inventory),
        "tree_sha256": digest.hexdigest(),
    }


def parse_terminal(payload: bytes) -> dict[str, int]:
    try:
        fields = dict(
            field.split("=", 1) for field in payload.decode("ascii").strip().split()
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise RecoveryError("RTL terminal is malformed") from error
    require(
        fields
        == {
            "schema": "ace3_decoder_token_transaction_v1",
            "layer_index": "6",
            "position": "3",
            "natural_terminal": "1",
            "exit_code": "0",
            "trace_count": "23408",
            "final_count": "896",
            "done_count": "1",
        },
        "RTL terminal fields differ",
    )
    return {
        name: int(fields[name])
        for name in (
            "natural_terminal",
            "exit_code",
            "trace_count",
            "final_count",
            "done_count",
        )
    }


def semantic_hidden_sha256(payload: bytes) -> str:
    try:
        rows = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise RecoveryError("hidden output is not ASCII") from error
    require(len(rows) == HIDDEN_ELEMENTS, "hidden output row count differs")
    binary = bytearray()
    for index, row in enumerate(rows):
        require(
            len(row) == 10
            and row[:2] == "00"
            and int(row[2:6], 16) == index,
            f"hidden output row {index} ordering differs",
        )
        binary.extend(int(row[6:], 16).to_bytes(2, "little"))
    return sha256_bytes(bytes(binary))


def validate_source_boundary(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    require(
        imports.isdisjoint(FORBIDDEN_IMPORTS),
        f"computation import in publication-recovery source: {path}",
    )
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    require(
        {name.lower() for name in calls}.isdisjoint(FORBIDDEN_CALLS),
        f"execution call in publication-recovery source: {path}",
    )


def validate_live_boundary() -> dict[str, Any]:
    require(not GENERATION8.exists(), "generation8 already exists")
    require(not GENERATION8_STAGING.exists(), "generation8 staging exists")
    require(
        all(
            not (TRANSACTIONS / f"transaction-{index:03d}").exists()
            for index in range(8, 26)
        ),
        "transaction008-025 artifact exists",
    )
    require(
        FUTURE.is_dir()
        and {path.name for path in FUTURE.iterdir()}
        == {
            EXECUTION_START.name,
            CONSUMPTION.name,
            FAILURE_TERMINAL.name,
        },
        "transaction007 fail-closed namespace differs",
    )
    pointer = load_json(POINTER)
    ledger = load_json(GENERATION7 / "ledger.json")
    require(
        pointer.get("status") == "COMMITTED"
        and pointer.get("generation") == 7
        and pointer.get("runtime_identity") == RUNTIME_IDENTITY
        and ledger.get("state_generation") == 7
        and ledger.get("next_transaction_index") == 7
        and ledger.get("completed_transaction_count") == 7
        and len(ledger.get("completed_receipts", [])) == 7,
        "generation7/checkpoint006/cursor7 parent differs",
    )
    consumption = load_json(CONSUMPTION)
    require(
        consumption.get("kind")
        == "ace3_position3_transaction7_layer6_authority_consumption"
        and consumption.get("status") == "CONSUMED_ONCE"
        and consumption.get("transaction_index") == 7
        and consumption.get("layer_index") == 6
        and consumption.get("replay_authorized") is False,
        "transaction007 authority consumption differs",
    )
    for label in ("manager_authorization", "independent_review", "package_seal"):
        authenticate(consumption[label], f"authority consumption {label}")
    require(
        consumption["package_seal"] == file_record(PACKAGE_SEAL),
        "consumed package seal differs",
    )
    failure = load_json(FAILURE_TERMINAL)
    require(
        failure
        == {
            "schema_version": 1,
            "kind": "ace3_position3_transaction7_layer6_fail_closed_terminal",
            "status": "FAIL",
            "error_type": "ControlError",
            "transaction_index": 7,
            "layer_index": 6,
        },
        "transaction007 fail-closed terminal differs",
    )
    return {
        "pointer": file_record(POINTER),
        "ledger": file_record(GENERATION7 / "ledger.json"),
        "checkpoint006": file_record(
            GENERATION7 / "checkpoints/transaction-006.json"
        ),
        "execution_start": file_record(EXECUTION_START),
        "authority_consumption": file_record(CONSUMPTION),
        "fail_closed_terminal": file_record(FAILURE_TERMINAL),
    }


def validate_mapping_failure() -> dict[str, Any]:
    exact = EXACT_EXECUTOR.read_text(encoding="utf-8")
    parser = TERMINAL_PARSER.read_text(encoding="utf-8")
    require(
        'natural_terminal = rtl.get("natural_terminal")' in exact
        and 'natural_terminal = rtl["raw"].get("natural_terminal")' in exact
        and '"natural_rtl_terminal": natural_terminal is True' in exact,
        "receipt natural-terminal mapping source differs",
    )
    require(
        'for name in ("trace_count", "final_count", "done_count")' in parser
        and "return counts" in parser,
        "terminal parser return boundary differs",
    )
    return {
        "exact_executor": file_record(EXACT_EXECUTOR),
        "terminal_parser": file_record(TERMINAL_PARSER),
        "authenticated_terminal_natural_terminal": 1,
        "returned_rtl_natural_terminal": None,
        "failing_expression": "natural_terminal is True",
        "failing_expression_result": False,
        "corrected_receipt_value": True,
    }


def computation_evidence() -> dict[str, Any]:
    raw = TRANSACTION7 / "position003/raw"
    oracle = TRANSACTION7 / "position003/exact_oracle"
    vectors = TRANSACTION7 / "vectors"
    terminal = raw / "terminal.txt"
    counts = parse_terminal(terminal.read_bytes())
    raw_trace = raw / "trace.hex"
    raw_final = raw / "final.hex"
    require(
        raw_trace.read_bytes()
        == (oracle / "trace.hex").read_bytes()
        == (vectors / "trace.hex").read_bytes(),
        "RTL, vector, and integer-oracle traces differ",
    )
    final_payload = raw_final.read_bytes()
    require(
        final_payload
        == (oracle / "final.hex").read_bytes()
        == (vectors / "final.hex").read_bytes(),
        "RTL, vector, and integer-oracle final outputs differ",
    )
    semantic = semantic_hidden_sha256(final_payload)
    comparison = load_json(TRANSACTION7 / "comparison.json")
    require(
        comparison
        == {
            "exact_integer_oracle_output_sha256": semantic,
            "implementation": "independent ACE-3 integer W4A16 oracle",
            "integer_mismatches": 0,
            "inter_layer_boundary": "binary16 round after every RTL layer",
            "rtl_matches_exact_integer_oracle": True,
        },
        "independent integer-oracle comparison differs",
    )
    simulation = (TRANSACTION7 / "position003/simulation.log").read_text(
        encoding="ascii"
    )
    pass_lines = [
        line
        for line in simulation.splitlines()
        if line.startswith("DECODER_LAYER_TOKEN_TRANSACTION_PASS ")
    ]
    require(
        len(pass_lines) == 1
        and "layer=6 position=3" in pass_lines[0]
        and "trace_count=23408 final_count=896" in pass_lines[0],
        "natural RTL completion log differs",
    )
    state = TRANSACTION7 / "position004.state"
    require(
        state.read_bytes().startswith(b"verilatorsave01\n"),
        "transaction007 output state is not a Verilator save",
    )
    return {
        "terminal": file_record(terminal),
        "terminal_counts": counts,
        "simulation_log": file_record(TRANSACTION7 / "position003/simulation.log"),
        "comparison": file_record(TRANSACTION7 / "comparison.json"),
        "raw_trace": file_record(raw_trace),
        "oracle_trace": file_record(oracle / "trace.hex"),
        "raw_final": file_record(raw_final),
        "oracle_final": file_record(oracle / "final.hex"),
        "output_state": file_record(state),
        "hidden_semantic_sha256": semantic,
    }


def transaction_descriptor() -> dict[str, Any]:
    descriptor = load_json(PACKAGE_MANIFEST)["transaction_descriptor"]
    require(
        descriptor.get("transaction_index") == 7
        and descriptor.get("layer_index") == 6
        and descriptor.get("operation") == "position3-decoder-layer"
        and descriptor.get("required_result")
        == {
            "exact_integer_oracle_match": True,
            "natural_rtl_terminal": True,
            "output_hidden_elements": 896,
            "output_state_position": 4,
        },
        "transaction007 descriptor differs",
    )
    return descriptor


def reconstructed_receipt(
    descriptor: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    semantic_sha256: str,
) -> dict[str, Any]:
    hidden = file_record(TRANSACTION7 / "position003/raw/final.hex")
    hidden.update(
        {
            "dtype": "FP16",
            "elements": HIDDEN_ELEMENTS,
            "semantic_sha256": semantic_sha256,
        }
    )
    return {
        "schema_version": 1,
        "kind": "ace3_position3_transaction_completion",
        "status": "COMPLETE",
        "runtime_identity": RUNTIME_IDENTITY,
        "transaction_index": 7,
        "operation": descriptor["operation"],
        "input_binding_sha256": descriptor["input_binding_sha256"],
        "authenticated_inputs": copy.deepcopy(descriptor["inputs"]),
        "transaction7_authority_consumption": file_record(CONSUMPTION),
        "candidate_manifest": file_record(PACKAGE_MANIFEST),
        "result": copy.deepcopy(descriptor["required_result"]),
        "outputs": {
            "hidden": hidden,
            "state": file_record(TRANSACTION7 / "position004.state"),
        },
        "output_semantics": {
            "hidden": {
                "dtype": "FP16",
                "elements": HIDDEN_ELEMENTS,
                "semantic_sha256": semantic_sha256,
            },
            "state": {
                "layer_index": 6,
                "position": 4,
                "parent_position": 2,
                "parent": copy.deepcopy(
                    descriptor["inputs"]["position2_kv_parent"]
                ),
            },
        },
        "artifacts": {
            record["relative_path"]: {
                key: record[key] for key in ("path", "bytes", "sha256")
            }
            for record in inventory
        },
        "rtl_reference_agreement": {"mismatches": 0, "agreement": True},
        "timing": {
            "started_at": None,
            "completed_at": None,
            "transaction_seconds": None,
            "cumulative_execution_seconds": load_json(
                GENERATION7 / "ledger.json"
            )["cumulative_execution_seconds"],
            "basis": (
                "transaction007 completion time was not durably preserved before "
                "receipt validation failed; no transaction007 latency is claimed"
            ),
        },
        "publication_recovery": {
            "kind": "frozen-evidence-receipt-reconstruction",
            "adapts_only": (
                "authenticated terminal natural_terminal=1 to receipt Boolean true"
            ),
            "model_rerun": False,
            "oracle_rerun": False,
            "rtl_rerun": False,
            "transaction_retry": False,
            "transaction_replay": False,
            "transaction_resume": False,
        },
    }


def adjudicate() -> dict[str, Any]:
    validate_source_boundary(Path(__file__).resolve())
    validate_source_boundary(REVIEW_SOURCE)
    boundary = validate_live_boundary()
    mapping = validate_mapping_failure()
    evidence = computation_evidence()
    inventory = tree_inventory(TRANSACTION7)
    descriptor = transaction_descriptor()
    receipt = reconstructed_receipt(
        descriptor, inventory, evidence["hidden_semantic_sha256"]
    )
    return {
        "schema_version": 1,
        "kind": "ace3_transaction7_postconsume_publication_recovery_adjudication",
        "status": "PASS",
        "runtime_identity": RUNTIME_IDENTITY,
        "transaction_index": 7,
        "layer_index": 6,
        "authoritative_parent": {
            "generation": 7,
            "cursor": 7,
            **{key: boundary[key] for key in ("pointer", "ledger", "checkpoint006")},
        },
        "execution_boundary": {
            key: boundary[key]
            for key in (
                "execution_start",
                "authority_consumption",
                "fail_closed_terminal",
            )
        },
        "source_boundary": mapping,
        "computation_evidence": evidence,
        "preserved_history": {
            "generations": {
                str(index): tree_summary(
                    GENERATIONS / f"generation-{index:010d}"
                )
                for index in range(3, 8)
            },
            "transactions": {
                str(index): tree_summary(
                    TRANSACTIONS / f"transaction-{index:03d}"
                )
                for index in range(7)
            },
        },
        "frozen_transaction007": {
            "tree": tree_summary(TRANSACTION7),
            "inventory": inventory,
        },
        "finding": {
            "natural_terminal": 1,
            "rtl_exit_code": 0,
            "done_count": 1,
            "final_count": 896,
            "integer_mismatches": 0,
            "exact_oracle_agreement": True,
            "sole_blocking_defect": (
                "authenticated terminal natural_terminal=1 was not represented "
                "as receipt Boolean true"
            ),
            "generation8_publication_attempted": False,
        },
        "reconstructed_receipt": receipt,
        "recovery_boundary": {
            "activity_counters": copy.deepcopy(ZERO_ACTIVITY),
            "authority_consumed_before_recovery": True,
            "authority_reusable": False,
            "new_authority_created": False,
            "generation8_publication_authorized": False,
            "generation8_publication_performed": False,
            "separately_reviewed_publication_authority_required": True,
            "transactions008_025_absent": True,
        },
    }


def validate_adjudication(document: Mapping[str, Any]) -> None:
    require(
        document.get("kind")
        == "ace3_transaction7_postconsume_publication_recovery_adjudication"
        and document.get("status") == "PASS"
        and document.get("transaction_index") == 7
        and document.get("layer_index") == 6,
        "adjudication identity differs",
    )
    require(
        document.get("finding")
        == {
            "natural_terminal": 1,
            "rtl_exit_code": 0,
            "done_count": 1,
            "final_count": 896,
            "integer_mismatches": 0,
            "exact_oracle_agreement": True,
            "sole_blocking_defect": (
                "authenticated terminal natural_terminal=1 was not represented "
                "as receipt Boolean true"
            ),
            "generation8_publication_attempted": False,
        },
        "adjudicated finding differs",
    )
    recovery = document["recovery_boundary"]
    require(
        recovery.get("activity_counters") == ZERO_ACTIVITY
        and recovery.get("authority_consumed_before_recovery") is True
        and recovery.get("authority_reusable") is False
        and recovery.get("new_authority_created") is False
        and recovery.get("generation8_publication_authorized") is False
        and recovery.get("generation8_publication_performed") is False
        and recovery.get("separately_reviewed_publication_authority_required")
        is True
        and recovery.get("transactions008_025_absent") is True,
        "recovery execution boundary differs",
    )
    receipt = document["reconstructed_receipt"]
    require(
        receipt.get("status") == "COMPLETE"
        and receipt.get("transaction_index") == 7
        and receipt.get("result", {}).get("natural_rtl_terminal") is True
        and receipt.get("result", {}).get("exact_integer_oracle_match") is True
        and receipt.get("result", {}).get("output_hidden_elements") == 896
        and receipt.get("timing", {}).get("transaction_seconds") is None,
        "reconstructed transaction007 receipt differs",
    )
    validate_live_boundary()
    evidence = computation_evidence()
    require(
        document.get("computation_evidence") == evidence,
        "frozen computation evidence differs",
    )
    require(
        document.get("source_boundary") == validate_mapping_failure(),
        "source mapping evidence differs",
    )
    require(
        document.get("frozen_transaction007", {}).get("tree")
        == tree_summary(TRANSACTION7),
        "transaction007 tree differs",
    )
    for record in document["frozen_transaction007"]["inventory"]:
        authenticate(
            {key: record[key] for key in ("path", "bytes", "sha256")},
            "transaction007 inventory",
        )
    for index in range(3, 8):
        require(
            document["preserved_history"]["generations"][str(index)]
            == tree_summary(GENERATIONS / f"generation-{index:010d}"),
            f"generation{index} preservation differs",
        )
    for index in range(7):
        require(
            document["preserved_history"]["transactions"][str(index)]
            == tree_summary(TRANSACTIONS / f"transaction-{index:03d}"),
            f"transaction{index:03d} preservation differs",
        )


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path: Path, payload: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare(output: Path = OUTPUT, review: Path = REVIEW) -> Path:
    require(output.is_absolute(), "recovery output must be absolute")
    require(review.is_absolute(), "recovery review must be absolute")
    require(not output.exists(), f"recovery package already exists: {output}")
    require(not review.exists(), f"recovery review already exists: {review}")
    document = adjudicate()
    validate_adjudication(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    review.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.preparing")
    require(not staging.exists(), f"recovery staging exists: {staging}")
    staging.mkdir(mode=0o700)
    try:
        write_new(staging / "recovery-adjudication.py", Path(__file__).read_bytes())
        write_new(staging / "review-emitter.py", REVIEW_SOURCE.read_bytes())
        write_new(staging / "adjudication.json", canonical_json(document))
        manifest = {
            "schema_version": 1,
            "kind": "ace3_transaction7_publication_recovery_package",
            "status": "SEALED_REVIEW_REQUIRED",
            "runtime_identity": RUNTIME_IDENTITY,
            "transaction_index": 7,
            "layer_index": 6,
            "review_path": str(review),
            "publication_entrypoint": None,
            "publication_authority": None,
            "activity_counters": copy.deepcopy(ZERO_ACTIVITY),
            "generation8_publication_authorized": False,
            "generation8_publication_performed": False,
            "separately_reviewed_publication_authority_required": True,
        }
        write_new(staging / "package-manifest.json", canonical_json(manifest))
        request = {
            "schema_version": 1,
            "kind": "ace3_transaction7_publication_recovery_review_request",
            "required_role": "reviewer",
            "review_output": str(review),
            "review_invocation": [
                "/usr/bin/python3",
                str(output / "review-emitter.py"),
                "--package",
                str(output),
                "--output",
                str(review),
            ],
            "requested_judgment": "PASS_OR_REJECT",
            "this_review_authorizes_publication": False,
            "review_must_not": [
                "execute model, oracle, vectors, RTL, synthesis, or U280",
                "retry, replay, or resume transaction007",
                "publish or stage generation8",
                "create transaction008-025 artifacts",
            ],
        }
        write_new(staging / "review-request.json", canonical_json(request))
        members = {
            name: file_record(staging / name, output / name)
            for name in PACKAGE_MEMBERS
        }
        seal = {
            "schema_version": 1,
            "kind": "ace3_transaction7_publication_recovery_adjudication_seal",
            "status": "SEALED_REVIEW_REQUIRED",
            "members": members,
            "activity_counters": copy.deepcopy(ZERO_ACTIVITY),
            "generation8_publication_authorized": False,
            "generation8_publication_performed": False,
        }
        write_new(staging / "package-seal.json", canonical_json(seal))
        fsync_directory(staging)
        os.rename(staging, output)
        fsync_directory(output.parent)
        output.chmod(0o555)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--review", type=Path, default=REVIEW)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        package = prepare(arguments.output, arguments.review)
        print(
            "TRANSACTION007_PUBLICATION_RECOVERY_SEALED "
            f"package={package} execution=0 publication=0"
        )
    else:
        validate_adjudication(load_json(arguments.output / "adjudication.json"))
        print(
            "TRANSACTION007_PUBLICATION_RECOVERY_VALID "
            f"package={arguments.output} execution=0 publication=0"
        )


if __name__ == "__main__":
    try:
        main()
    except (
        RecoveryError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise SystemExit(
            f"TRANSACTION007_PUBLICATION_RECOVERY_REFUSED {error}"
        ) from error
