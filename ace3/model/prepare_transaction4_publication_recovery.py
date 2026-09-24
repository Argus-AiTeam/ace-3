#!/usr/bin/env python3
"""Seal a no-execution adjudication of transaction004 publication failure."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

ROOT = Path("/home/argustest/ace3-argus")
RUNTIME_IDENTITY = "ace3-position3-fresh-r11-20260831t215500z"
RUNTIME = ROOT / "build/model24_selected_token_position3_runs" / RUNTIME_IDENTITY
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
POINTER = ADOPTION / "authoritative-state.json"
GENERATION4 = ADOPTION / "state-generations/generation-0000000004"
GENERATION5 = ADOPTION / "state-generations/generation-0000000005"
GENERATION5_STAGING = ADOPTION / "state-generations/.generation-0000000005.prepared"
TRANSACTIONS = RUNTIME / "transactions"
TRANSACTION4 = TRANSACTIONS / "transaction-004"
FUTURE = RUNTIME / "transaction4-authoritative-generation5"
EXECUTION_START = FUTURE / "execution-start.json"
CONSUMPTION = FUTURE / "manager-authorization-consumption.json"
FAILURE_TERMINAL = FUTURE / "fail-closed-terminal.json"
RECEIPT_CANDIDATE = FUTURE / "receipt-candidate.json"
EXECUTION_EVIDENCE = FUTURE / "execution-evidence.json"
SUCCESS_TERMINAL = FUTURE / "terminal.json"
PACKAGE = (
    ROOT
    / "build/model24_selected_token_position3_transaction4_packages"
    / "generation4-cursor4-transaction004-layer03-r1"
)
PACKAGE_MANIFEST = PACKAGE / "package-manifest.json"
PACKAGE_SEAL = PACKAGE / "package-seal.json"
SOURCE_MANIFEST = PACKAGE / "source-manifest.json"
WRAPPER_SOURCE = PACKAGE / "transaction4_executor.py"
EXACT_SOURCE = RUNTIME / "recovery-r12/recovery_executor.py"
TERMINAL_PARSER_SOURCE = (
    ROOT / "ace3/model/validate_selected_token_position2_traversal.py"
)
PRODUCTION_CONTROL = (
    ROOT
    / "build/model24_selected_token_position3_continuations"
    / "v10-accepted-review-gated-r10-readable-authority-schema"
    / "continuation_control.py"
)
RECOVERY_ROOT = ROOT / "build/model24_selected_token_position3_transaction4_recovery"
DEFAULT_OUTPUT = RECOVERY_ROOT / "r3-postconsume-none-to-false-sourcebound"
DEFAULT_REVIEW = (
    RECOVERY_ROOT
    / "reviews/r3-postconsume-none-to-false-sourcebound/independent-review.json"
)
REVIEW_EMITTER_SOURCE = (
    ROOT / "ace3/model/emit_transaction4_publication_recovery_review.py"
)

TRANSACTION_INDEX = 4
LAYER_INDEX = 3
HIDDEN_SIZE = 896
TRACE_COUNT = 23408
EXPECTED_HIDDEN_SEMANTIC_SHA256 = (
    "152347b94d879e16b8e5d514ac7a0b38c954336f138bd3f5e27b57965d350cfa"
)
EXPECTED_EXACT_SOURCE_SHA256 = (
    "8308de743d5dbb7e3ff9d74080b70834c9bb945a94460090d564b53ba52ef726"
)
EXPECTED_WRAPPER_SOURCE_SHA256 = (
    "a4c89cb1f33551e33920f10cc7c91e6fe51603307e66346e5974ca3c6a381395"
)
EXPECTED_TERMINAL_PARSER_SOURCE_SHA256 = (
    "6b9fe66a6ba3fd0249ce7e618a6fd395907707a39225d8d56565e4ba8f6ec00c"
)
EXPECTED_PRODUCTION_CONTROL = {
    "path": str(PRODUCTION_CONTROL),
    "bytes": 151006,
    "sha256": "001f26990ee7628106600100c2de91e7305723c38efe31a3c81b2fe044e09db6",
}
PACKAGE_MEMBERS = {
    "adjudication.json",
    "package-manifest.json",
    "recovery-adjudication.py",
    "review-emitter.py",
    "review-request.json",
}
ZERO_ACTIVITY = {
    "authority_consumption": 0,
    "generation5_publication": 0,
    "model_execution": 0,
    "oracle_execution": 0,
    "rtl_compile": 0,
    "rtl_simulation": 0,
    "transaction_execution": 0,
    "transaction_replay": 0,
    "transaction_resume": 0,
    "transaction_retry": 0,
    "transactions005_025_execution": 0,
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


def file_record(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"regular non-symlink file required: {path}",
    )
    return {
        "path": str(path),
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
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        record = file_record(path)
        record["relative_path"] = path.relative_to(root).as_posix()
        records.append(record)
    return records


def tree_summary(inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for record in inventory:
        digest.update(record["relative_path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "root": str(TRANSACTION4),
        "file_count": len(inventory),
        "total_bytes": sum(record["bytes"] for record in inventory),
        "tree_sha256": digest.hexdigest(),
    }


def authenticate_inventory(
    root: Path, inventory: Sequence[Mapping[str, Any]]
) -> None:
    require(list(inventory) == tree_inventory(root), "transaction004 tree differs")


def parse_terminal(payload: bytes) -> dict[str, int]:
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise RecoveryError("RTL terminal is not ASCII") from error
    require(len(lines) == 1, "RTL terminal line count mismatch")
    fields: dict[str, str] = {}
    for item in lines[0].split():
        require(item.count("=") == 1, "RTL terminal field malformed")
        name, value = item.split("=", 1)
        require(name not in fields, f"duplicate RTL terminal field: {name}")
        fields[name] = value
    require(
        set(fields)
        == {
            "schema",
            "layer_index",
            "position",
            "natural_terminal",
            "exit_code",
            "trace_count",
            "final_count",
            "done_count",
        },
        "RTL terminal schema differs",
    )
    require(
        fields["schema"] == "ace3_decoder_token_transaction_v1"
        and fields["layer_index"] == "3"
        and fields["position"] == "3"
        and fields["natural_terminal"] == "1"
        and fields["exit_code"] == "0",
        "RTL terminal is not a natural successful completion",
    )
    counts = {
        name: int(fields[name])
        for name in ("trace_count", "final_count", "done_count")
    }
    require(
        counts
        == {
            "trace_count": TRACE_COUNT,
            "final_count": HIDDEN_SIZE,
            "done_count": 1,
        },
        "RTL terminal counts differ",
    )
    return counts


def semantic_hidden_sha256(payload: bytes) -> str:
    try:
        rows = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise RecoveryError("hidden output is not ASCII") from error
    require(len(rows) == HIDDEN_SIZE, "hidden output row count differs")
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


def validate_comparison(document: Mapping[str, Any]) -> None:
    require(
        document
        == {
            "exact_integer_oracle_output_sha256": EXPECTED_HIDDEN_SEMANTIC_SHA256,
            "implementation": "independent ACE-3 integer W4A16 oracle",
            "integer_mismatches": 0,
            "inter_layer_boundary": "binary16 round after every RTL layer",
            "rtl_matches_exact_integer_oracle": True,
        },
        "comparison is not exact zero-mismatch integer-oracle agreement",
    )


def validate_source_boundary(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
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
        {
            "fcntl",
            "numpy",
            "safetensors",
            "shutil",
            "subprocess",
            "torch",
        }.isdisjoint(imported),
        "recovery package imports computation or execution support",
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
        {
            "execute_exact_layer_transaction",
            "execute_layer_transaction",
            "publish_generation5",
            "run",
            "system",
        }.isdisjoint(calls),
        "recovery package contains an execution or publication call",
    )


def validate_mapping_sources() -> dict[str, Any]:
    source_manifest = load_json(SOURCE_MANIFEST)
    exact_record = source_manifest["sources"]["exact_transaction_executor"]
    wrapper_record = source_manifest["sources"]["transaction004_executor"]
    parser_record = source_manifest["sources"]["position2_exact_helpers"]
    require(
        exact_record == file_record(EXACT_SOURCE)
        and exact_record["sha256"] == EXPECTED_EXACT_SOURCE_SHA256,
        "exact executor source binding differs",
    )
    require(
        wrapper_record == file_record(WRAPPER_SOURCE)
        and wrapper_record["sha256"] == EXPECTED_WRAPPER_SOURCE_SHA256,
        "transaction004 wrapper source binding differs",
    )
    require(
        parser_record == file_record(TERMINAL_PARSER_SOURCE)
        and parser_record["sha256"] == EXPECTED_TERMINAL_PARSER_SOURCE_SHA256,
        "terminal parser source binding differs",
    )
    require(
        file_record(PRODUCTION_CONTROL) == EXPECTED_PRODUCTION_CONTROL,
        "production completion validator source binding differs",
    )
    parser_source = TERMINAL_PARSER_SOURCE.read_text(encoding="utf-8")
    parser_tree = ast.parse(parser_source)
    parser_function = next(
        (
            node
            for node in parser_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "parse_natural_terminal"
        ),
        None,
    )
    require(parser_function is not None, "terminal parser function is absent")
    parser_segment = ast.get_source_segment(parser_source, parser_function) or ""
    require(
        'fields["natural_terminal"] == "1"' in parser_segment
        and 'fields["exit_code"] == "0"' in parser_segment
        and 'for name in ("trace_count", "final_count", "done_count")'
        in parser_segment
        and "return counts" in parser_segment,
        "terminal parser authentication/return boundary differs",
    )
    record_function = next(
        (
            node
            for node in parser_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "execute_transaction"
        ),
        None,
    )
    require(record_function is not None, "RTL record builder function is absent")
    record_segment = ast.get_source_segment(parser_source, record_function) or ""
    raw_segment = (
        '"raw": {\n'
        '            "terminal": file_record(raw_dir / "terminal.txt"),\n'
        '            "trace": file_record(raw_dir / "trace.hex"),\n'
        "            **counts,\n"
        "        }"
    )
    require(
        raw_segment in record_segment
        and '"natural_terminal":' not in record_segment,
        "returned RTL record boundary differs",
    )
    exact = EXACT_SOURCE.read_text(encoding="utf-8")
    mapping = (
        'natural_terminal = rtl.get("natural_terminal")\n'
        "    if natural_terminal is None:\n"
        '        natural_terminal = rtl["raw"].get("natural_terminal")'
    )
    require(mapping in exact, "natural terminal extraction source differs")
    require(
        '"natural_rtl_terminal": natural_terminal is True' in exact,
        "natural_rtl_terminal identity mapping source differs",
    )
    wrapper = WRAPPER_SOURCE.read_text(encoding="utf-8")
    call_offset = wrapper.index("receipt = exact.execute_exact_layer_transaction(")
    validation_offset = wrapper.index(
        "original.candidate_control.validate_completion_receipt(", call_offset
    )
    write_offset = wrapper.index("write_exclusive_json(RECEIPT_CANDIDATE", call_offset)
    require(
        call_offset < validation_offset < write_offset,
        "receipt validation/publication order differs",
    )
    counts = parse_terminal(
        (TRANSACTION4 / "position003/raw/terminal.txt").read_bytes()
    )
    returned_rtl_record = {
        "raw": {
            "terminal": file_record(
                TRANSACTION4 / "position003/raw/terminal.txt"
            ),
            "trace": file_record(TRANSACTION4 / "position003/raw/trace.hex"),
            **counts,
        }
    }
    top_level_value = returned_rtl_record.get("natural_terminal")
    raw_value = returned_rtl_record["raw"].get("natural_terminal")
    adapter_value = top_level_value
    if adapter_value is None:
        adapter_value = raw_value
    require(
        top_level_value is None
        and raw_value is None
        and adapter_value is None
        and (adapter_value is True) is False,
        "None-to-False mapping failure was not reproduced",
    )
    return {
        "source_manifest": file_record(SOURCE_MANIFEST),
        "terminal_parser": parser_record,
        "returned_rtl_record_builder": parser_record,
        "exact_executor": exact_record,
        "transaction004_wrapper": wrapper_record,
        "production_completion_validator": copy.deepcopy(
            EXPECTED_PRODUCTION_CONTROL
        ),
        "terminal_parser_authenticates": {
            "natural_terminal": 1,
            "exit_code": 0,
        },
        "terminal_parser_returns": [
            "trace_count",
            "final_count",
            "done_count",
        ],
        "returned_rtl_record_top_level_natural_terminal": None,
        "returned_rtl_record_raw_natural_terminal": None,
        "receipt_adapter_observed_value_type": "NoneType",
        "receipt_adapter_observed_value": None,
        "failing_expression": "natural_terminal is True",
        "failing_expression_result": False,
        "corrected_reconstructed_receipt_value": True,
    }


def validate_with_production_completion_validator(
    transaction: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    require(
        file_record(PRODUCTION_CONTROL) == EXPECTED_PRODUCTION_CONTROL,
        "production completion validator source binding differs",
    )
    spec = importlib.util.spec_from_file_location(
        "ace3_transaction4_recovery_production_control",
        PRODUCTION_CONTROL,
    )
    require(
        spec is not None and spec.loader is not None,
        "production completion validator could not be loaded",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        validated = module.validate_completion_receipt(transaction, receipt)
    except module.ControlError as error:
        raise RecoveryError(
            f"production completion validator rejected reconstructed receipt: {error}"
        ) from error
    require(
        isinstance(validated, dict)
        and validated.get("result") == receipt.get("result"),
        "production completion validator returned an unexpected receipt",
    )
    return {
        "status": "PASS",
        "validator": copy.deepcopy(EXPECTED_PRODUCTION_CONTROL),
        "transaction_descriptor": file_record(PACKAGE_MANIFEST),
        "validated_result": copy.deepcopy(validated["result"]),
    }


def validate_namespace(
    *,
    generation5_exists: bool,
    generation5_staging_exists: bool,
    receipt_candidate_exists: bool,
    execution_evidence_exists: bool,
    success_terminal_exists: bool,
    later_transactions: Sequence[int],
) -> None:
    require(not generation5_exists, "generation5 publication exists")
    require(not generation5_staging_exists, "generation5 staging exists")
    require(not receipt_candidate_exists, "receipt candidate unexpectedly exists")
    require(not execution_evidence_exists, "execution evidence unexpectedly exists")
    require(not success_terminal_exists, "success terminal unexpectedly exists")
    require(not later_transactions, "transaction005-025 artifacts exist")


def later_transactions() -> list[int]:
    return [
        index
        for index in range(5, 26)
        if (TRANSACTIONS / f"transaction-{index:03d}").exists()
    ]


def validate_live_namespace() -> None:
    validate_namespace(
        generation5_exists=GENERATION5.exists(),
        generation5_staging_exists=GENERATION5_STAGING.exists(),
        receipt_candidate_exists=RECEIPT_CANDIDATE.exists(),
        execution_evidence_exists=EXECUTION_EVIDENCE.exists(),
        success_terminal_exists=SUCCESS_TERMINAL.exists(),
        later_transactions=later_transactions(),
    )
    require(
        FUTURE.is_dir()
        and {path.name for path in FUTURE.iterdir()}
        == {
            EXECUTION_START.name,
            CONSUMPTION.name,
            FAILURE_TERMINAL.name,
        },
        "post-consumption failure namespace differs",
    )
    pointer = load_json(POINTER)
    ledger = load_json(GENERATION4 / "ledger.json")
    require(
        pointer.get("status") == "COMMITTED"
        and pointer.get("generation") == 4
        and pointer.get("runtime_identity") == RUNTIME_IDENTITY
        and ledger.get("state_generation") == 4
        and ledger.get("next_transaction_index") == 4
        and ledger.get("completed_transaction_count") == 4
        and len(ledger.get("completed_receipts", [])) == 4,
        "authoritative generation4/cursor4 parent differs",
    )


def validate_execution_boundary() -> dict[str, Any]:
    require(
        load_json(EXECUTION_START)
        == {
            "schema_version": 1,
            "kind": "ace3_position3_transaction4_layer3_execution_start",
            "transaction_index": 4,
            "layer_index": 3,
            "started_ns": 1788266333018994600,
        },
        "transaction004 execution-start differs",
    )
    consumption = load_json(CONSUMPTION)
    require(
        consumption.get("kind")
        == "ace3_position3_transaction4_layer3_authority_consumption"
        and consumption.get("status") == "CONSUMED_ONCE"
        and consumption.get("transaction_index") == 4
        and consumption.get("layer_index") == 3
        and consumption.get("replay_authorized") is False,
        "transaction004 authority consumption differs",
    )
    for label in ("manager_authorization", "independent_review", "package_seal"):
        authenticate(consumption[label], f"consumption {label}")
    require(
        consumption["package_seal"] == file_record(PACKAGE_SEAL),
        "consumption package seal differs",
    )
    failure = load_json(FAILURE_TERMINAL)
    require(
        failure
        == {
            "schema_version": 1,
            "kind": "ace3_position3_transaction4_layer3_fail_closed_terminal",
            "status": "FAIL",
            "error_type": "ControlError",
            "transaction_index": 4,
            "layer_index": 3,
        },
        "transaction004 fail-closed terminal differs",
    )
    return {
        "execution_start": file_record(EXECUTION_START),
        "authority_consumption": file_record(CONSUMPTION),
        "fail_closed_terminal": file_record(FAILURE_TERMINAL),
        "authority_state": "CONSUMED_ONCE_NON_REPLAYABLE",
    }


def validate_computation_evidence() -> dict[str, Any]:
    terminal = TRANSACTION4 / "position003/raw/terminal.txt"
    counts = parse_terminal(terminal.read_bytes())
    comparison_path = TRANSACTION4 / "comparison.json"
    validate_comparison(load_json(comparison_path))
    raw_trace = TRANSACTION4 / "position003/raw/trace.hex"
    oracle_trace = TRANSACTION4 / "position003/exact_oracle/trace.hex"
    vector_trace = TRANSACTION4 / "vectors/trace.hex"
    raw_final = TRANSACTION4 / "position003/raw/final.hex"
    oracle_final = TRANSACTION4 / "position003/exact_oracle/final.hex"
    vector_final = TRANSACTION4 / "vectors/final.hex"
    require(
        raw_trace.read_bytes()
        == oracle_trace.read_bytes()
        == vector_trace.read_bytes(),
        "raw, vector, and exact-oracle traces differ",
    )
    final_payload = raw_final.read_bytes()
    require(
        final_payload == oracle_final.read_bytes() == vector_final.read_bytes(),
        "raw, vector, and exact-oracle hidden outputs differ",
    )
    semantic = semantic_hidden_sha256(final_payload)
    require(
        semantic == EXPECTED_HIDDEN_SEMANTIC_SHA256,
        "hidden output semantic hash differs",
    )
    simulation = (TRANSACTION4 / "position003/simulation.log").read_text(
        encoding="ascii"
    )
    require(
        simulation.count("DECODER_LAYER_TOKEN_TRANSACTION_PASS") == 1
        and simulation.rstrip().endswith(
            "DECODER_LAYER_TOKEN_TRANSACTION_PASS layer=3 position=3 "
            "trace_count=23408 final_count=896 cycles=15399417 stalls=214569"
        ),
        "simulation completion log differs",
    )
    output_state = TRANSACTION4 / "position004.state"
    require(
        output_state.read_bytes().startswith(b"verilatorsave01\n"),
        "output state is not a Verilator save",
    )
    return {
        "terminal": file_record(terminal),
        "terminal_counts": counts,
        "comparison": file_record(comparison_path),
        "raw_trace": file_record(raw_trace),
        "oracle_trace": file_record(oracle_trace),
        "raw_final": file_record(raw_final),
        "oracle_final": file_record(oracle_final),
        "simulation_log": file_record(
            TRANSACTION4 / "position003/simulation.log"
        ),
        "output_state": file_record(output_state),
        "hidden_semantic_sha256": semantic,
    }


def descriptor() -> dict[str, Any]:
    document = load_json(PACKAGE_MANIFEST)["transaction_descriptor"]
    require(
        document.get("transaction_index") == 4
        and document.get("layer_index") == 3
        and document.get("operation") == "position3-decoder-layer"
        and document.get("required_result")
        == {
            "exact_integer_oracle_match": True,
            "natural_rtl_terminal": True,
            "output_hidden_elements": 896,
            "output_state_position": 4,
        },
        "transaction004 descriptor differs",
    )
    return document


def reconstructed_receipt(
    transaction: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    hidden = file_record(TRANSACTION4 / "position003/raw/final.hex")
    hidden.update(
        {
            "dtype": "FP16",
            "elements": HIDDEN_SIZE,
            "semantic_sha256": EXPECTED_HIDDEN_SEMANTIC_SHA256,
        }
    )
    receipt = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction_completion",
        "status": "COMPLETE",
        "runtime_identity": RUNTIME_IDENTITY,
        "transaction_index": 4,
        "operation": transaction["operation"],
        "input_binding_sha256": transaction["input_binding_sha256"],
        "authenticated_inputs": copy.deepcopy(transaction["inputs"]),
        "transaction4_authority_consumption": file_record(CONSUMPTION),
        "candidate_manifest": file_record(PACKAGE_MANIFEST),
        "result": copy.deepcopy(transaction["required_result"]),
        "outputs": {
            "hidden": hidden,
            "state": file_record(TRANSACTION4 / "position004.state"),
        },
        "output_semantics": {
            "hidden": {
                "dtype": "FP16",
                "elements": HIDDEN_SIZE,
                "semantic_sha256": EXPECTED_HIDDEN_SEMANTIC_SHA256,
            },
            "state": {
                "layer_index": 3,
                "position": 4,
                "parent_position": 2,
                "parent": copy.deepcopy(transaction["inputs"]["position2_kv_parent"]),
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
                GENERATION4 / "ledger.json"
            )["cumulative_execution_seconds"],
            "basis": (
                "transaction004 timing was not durably preserved before receipt "
                "validation failed; no transaction004 latency is claimed"
            ),
        },
        "publication_recovery": {
            "kind": "frozen-evidence-receipt-reconstruction",
            "adapts_only": "natural_rtl_terminal None-to-True reconstruction",
            "model_rerun": False,
            "oracle_rerun": False,
            "rtl_rerun": False,
            "transaction_retry": False,
            "transaction_replay": False,
            "transaction_resume": False,
        },
    }
    validate_receipt(transaction, receipt)
    return receipt


def validate_receipt(
    transaction: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    require(
        receipt.get("status") == "COMPLETE"
        and receipt.get("transaction_index") == 4
        and receipt.get("operation") == transaction["operation"]
        and receipt.get("input_binding_sha256")
        == transaction["input_binding_sha256"]
        and receipt.get("result") == transaction["required_result"],
        "reconstructed receipt identity or result differs",
    )
    hidden = receipt.get("outputs", {}).get("hidden", {})
    state = receipt.get("outputs", {}).get("state", {})
    authenticate(
        {key: hidden[key] for key in ("path", "bytes", "sha256")},
        "receipt hidden output",
    )
    authenticate(state, "receipt state output")
    require(
        receipt.get("output_semantics", {}).get("hidden")
        == {
            "dtype": "FP16",
            "elements": HIDDEN_SIZE,
            "semantic_sha256": EXPECTED_HIDDEN_SEMANTIC_SHA256,
        }
        and receipt.get("output_semantics", {}).get("state", {}).get(
            "layer_index"
        )
        == 3
        and receipt.get("output_semantics", {}).get("state", {}).get("position")
        == 4,
        "reconstructed output semantics differ",
    )
    timing = receipt.get("timing", {})
    require(
        timing.get("started_at") is None
        and timing.get("completed_at") is None
        and timing.get("transaction_seconds") is None
        and "not durably preserved" in timing.get("basis", ""),
        "reconstructed receipt makes an unsupported timing claim",
    )
    recovery = receipt.get("publication_recovery", {})
    require(
        recovery.get("model_rerun") is False
        and recovery.get("oracle_rerun") is False
        and recovery.get("rtl_rerun") is False
        and recovery.get("transaction_retry") is False
        and recovery.get("transaction_replay") is False
        and recovery.get("transaction_resume") is False,
        "reconstructed receipt permits new execution",
    )


def adjudicate() -> dict[str, Any]:
    validate_live_namespace()
    transaction = descriptor()
    source_boundary = validate_mapping_sources()
    execution_boundary = validate_execution_boundary()
    computation = validate_computation_evidence()
    inventory = tree_inventory(TRANSACTION4)
    receipt = reconstructed_receipt(transaction, inventory)
    production_validation = validate_with_production_completion_validator(
        transaction, receipt
    )
    return {
        "schema_version": 1,
        "kind": "ace3_transaction4_postconsume_publication_recovery_adjudication",
        "status": "PASS",
        "runtime_identity": RUNTIME_IDENTITY,
        "transaction_index": 4,
        "layer_index": 3,
        "authoritative_parent": {
            "generation": 4,
            "cursor": 4,
            "pointer": file_record(POINTER),
            "ledger": file_record(GENERATION4 / "ledger.json"),
            "checkpoint003": file_record(
                GENERATION4 / "checkpoints/transaction-003.json"
            ),
        },
        "execution_boundary": execution_boundary,
        "source_boundary": source_boundary,
        "computation_evidence": computation,
        "frozen_transaction004": {
            "tree": tree_summary(inventory),
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
                "completion receipt natural_rtl_terminal None-to-False mapping"
            ),
            "generation5_publication_attempted": False,
        },
        "reconstructed_receipt": receipt,
        "production_completion_validation": production_validation,
        "recovery_boundary": {
            "activity_counters": copy.deepcopy(ZERO_ACTIVITY),
            "authority_consumed_before_recovery": True,
            "authority_reusable": False,
            "new_authority_created": False,
            "generation5_publication_authorized": False,
            "generation5_publication_performed": False,
            "separately_reviewed_successor_required": True,
            "transactions005_025_absent": True,
        },
    }


def validate_adjudication(
    document: Mapping[str, Any], authenticate_evidence: bool = True
) -> None:
    require(
        document.get("kind")
        == "ace3_transaction4_postconsume_publication_recovery_adjudication"
        and document.get("status") == "PASS"
        and document.get("transaction_index") == 4
        and document.get("layer_index") == 3,
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
                "completion receipt natural_rtl_terminal None-to-False mapping"
            ),
            "generation5_publication_attempted": False,
        },
        "adjudicated finding differs",
    )
    boundary = document.get("recovery_boundary", {})
    require(
        boundary.get("activity_counters") == ZERO_ACTIVITY
        and boundary.get("authority_consumed_before_recovery") is True
        and boundary.get("authority_reusable") is False
        and boundary.get("new_authority_created") is False
        and boundary.get("generation5_publication_authorized") is False
        and boundary.get("generation5_publication_performed") is False
        and boundary.get("separately_reviewed_successor_required") is True
        and boundary.get("transactions005_025_absent") is True,
        "recovery no-execution/publication boundary differs",
    )
    validate_receipt(descriptor(), document["reconstructed_receipt"])
    require(
        document.get("source_boundary") == validate_mapping_sources(),
        "source mapping adjudication differs",
    )
    production_validation = document.get("production_completion_validation", {})
    require(
        production_validation
        == validate_with_production_completion_validator(
            descriptor(), document["reconstructed_receipt"]
        ),
        "production completion validation proof differs",
    )
    if authenticate_evidence:
        validate_live_namespace()
        authenticate_inventory(
            TRANSACTION4, document["frozen_transaction004"]["inventory"]
        )
        for section in (
            "authoritative_parent",
            "execution_boundary",
            "source_boundary",
            "computation_evidence",
        ):
            for value in document[section].values():
                if (
                    isinstance(value, dict)
                    and set(value) == {"path", "bytes", "sha256"}
                ):
                    authenticate(value, f"adjudication {section}")
        validate_execution_boundary()
        validate_computation_evidence()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path: Path, payload: bytes) -> None:
    descriptor_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor_fd, payload[offset:])
        os.fsync(descriptor_fd)
    finally:
        os.close(descriptor_fd)


def prepare_package(
    output: Path = DEFAULT_OUTPUT, review_path: Path = DEFAULT_REVIEW
) -> Path:
    require(output.is_absolute(), "package output must be absolute")
    require(review_path.is_absolute(), "review output must be absolute")
    require(not output.exists(), f"package output already exists: {output}")
    require(not review_path.exists(), f"review output already exists: {review_path}")
    validate_source_boundary(Path(__file__).resolve())
    validate_source_boundary(REVIEW_EMITTER_SOURCE)
    adjudication = adjudicate()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.preparing")
    require(not staging.exists(), f"package staging already exists: {staging}")
    staging.mkdir(mode=0o700)
    write_new(staging / "recovery-adjudication.py", Path(__file__).read_bytes())
    write_new(staging / "review-emitter.py", REVIEW_EMITTER_SOURCE.read_bytes())
    write_new(staging / "adjudication.json", canonical_json(adjudication))
    manifest = {
        "schema_version": 1,
        "kind": "ace3_transaction4_publication_recovery_adjudication_package",
        "status": "SEALED_REVIEW_REQUIRED",
        "runtime_identity": RUNTIME_IDENTITY,
        "transaction_index": 4,
        "layer_index": 3,
        "source_generation": 4,
        "source_cursor": 4,
        "review_path": str(review_path),
        "activity_counters": copy.deepcopy(ZERO_ACTIVITY),
        "publication_entrypoint": None,
        "recovery_authority": None,
        "generation5_publication_authorized": False,
        "separately_reviewed_successor_required": True,
        "forbidden": [
            "transaction004 retry",
            "transaction004 replay",
            "transaction004 resume",
            "transaction005-025 execution",
            "authority consumption",
            "generation5 publication",
            "model, oracle, RTL, or transaction invocation",
        ],
    }
    write_new(staging / "package-manifest.json", canonical_json(manifest))
    request = {
        "schema_version": 1,
        "kind": "ace3_transaction4_publication_recovery_review_request",
        "requested_judgment": "PASS_OR_REJECT",
        "review_output": str(review_path),
        "reviewer_emitter": file_record(staging / "review-emitter.py"),
        "adjudication": file_record(staging / "adjudication.json"),
        "review_must_not": [
            "consume authority",
            "execute or resume any transaction",
            "publish or stage generation5",
        ],
    }
    request["reviewer_emitter"]["path"] = str(output / "review-emitter.py")
    request["adjudication"]["path"] = str(output / "adjudication.json")
    write_new(staging / "review-request.json", canonical_json(request))
    members = {}
    for name in sorted(PACKAGE_MEMBERS):
        record = file_record(staging / name)
        record["path"] = str(output / name)
        members[name] = record
    seal = {
        "schema_version": 1,
        "kind": "ace3_transaction4_publication_recovery_adjudication_seal",
        "status": "SEALED_REVIEW_REQUIRED",
        "members": members,
        "activity_counters": copy.deepcopy(ZERO_ACTIVITY),
        "generation5_publication_authorized": False,
        "generation5_publication_performed": False,
    }
    write_new(staging / "package-seal.json", canonical_json(seal))
    fsync_directory(staging)
    os.rename(staging, output)
    fsync_directory(output.parent)
    output.chmod(0o555)
    validate_package(output)
    return output


def validate_package(package: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    require(
        package.is_dir()
        and not package.is_symlink()
        and stat.S_IMODE(package.stat().st_mode) & 0o222 == 0,
        "sealed package directory is absent or writable",
    )
    seal = load_json(package / "package-seal.json")
    require(
        seal.get("kind")
        == "ace3_transaction4_publication_recovery_adjudication_seal"
        and seal.get("status") == "SEALED_REVIEW_REQUIRED"
        and seal.get("activity_counters") == ZERO_ACTIVITY
        and seal.get("generation5_publication_authorized") is False
        and seal.get("generation5_publication_performed") is False
        and set(seal.get("members", {})) == PACKAGE_MEMBERS,
        "package seal differs",
    )
    actual = {path.name for path in package.iterdir() if path.is_file()}
    require(
        actual == PACKAGE_MEMBERS | {"package-seal.json"},
        "sealed package file set differs",
    )
    for name, record in seal["members"].items():
        require(record["path"] == str(package / name), f"sealed path differs: {name}")
        authenticate(record, f"sealed member {name}")
        require(
            stat.S_IMODE((package / name).stat().st_mode) & 0o222 == 0,
            f"sealed member is writable: {name}",
        )
    manifest = load_json(package / "package-manifest.json")
    require(
        manifest.get("activity_counters") == ZERO_ACTIVITY
        and manifest.get("publication_entrypoint") is None
        and manifest.get("recovery_authority") is None
        and manifest.get("generation5_publication_authorized") is False
        and manifest.get("separately_reviewed_successor_required") is True,
        "package execution/publication boundary differs",
    )
    validate_source_boundary(package / "recovery-adjudication.py")
    validate_source_boundary(package / "review-emitter.py")
    adjudication = load_json(package / "adjudication.json")
    validate_adjudication(adjudication)
    return adjudication


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "validate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    arguments = parser.parse_args()
    if arguments.operation == "prepare":
        package = prepare_package(arguments.output, arguments.review)
        print(
            "TRANSACTION004_RECOVERY_ADJUDICATION_SEALED "
            f"package={package} execution=0 authority_consumption=0 publication=0"
        )
    else:
        validate_package(arguments.output)
        print(
            "TRANSACTION004_RECOVERY_ADJUDICATION_VALID "
            "transaction004_retry=0 transaction005_025=absent generation5=absent"
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
        raise SystemExit(f"TRANSACTION004_RECOVERY_ADJUDICATION_REFUSED {error}") from error
