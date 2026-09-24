#!/usr/bin/env python3
"""Seal and apply the transaction-003 publication-only receipt recovery."""

from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path("/home/argustest/ace3-argus")
RUNTIME_IDENTITY = "ace3-position3-fresh-r11-20260831t215500z"
RUNTIME = (
    ROOT / "build/model24_selected_token_position3_runs" / RUNTIME_IDENTITY
)
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
GENERATIONS = ADOPTION / "state-generations"
GENERATION3 = GENERATIONS / "generation-0000000003"
GENERATION4 = GENERATIONS / "generation-0000000004"
GENERATION4_STAGING = GENERATIONS / ".generation-0000000004.prepared"
POINTER = ADOPTION / "authoritative-state.json"
RUNTIME_LOCK = ADOPTION / "execution.lock"
TRANSACTIONS = RUNTIME / "transactions"
TRANSACTION3 = TRANSACTIONS / "transaction-003"
FUTURE = RUNTIME / "transaction3-authoritative-generation4"
CONSUMPTION = FUTURE / "manager-authorization-consumption.json"
EXECUTION_START = FUTURE / "execution-start.json"
RECEIPT_CANDIDATE = FUTURE / "receipt-candidate.json"
EXECUTION_EVIDENCE = FUTURE / "execution-evidence.json"
SUCCESS_TERMINAL = FUTURE / "terminal.json"
FAILURE_SEAL = FUTURE / "fail-closed-terminal.json"
R5_PACKAGE = (
    ROOT
    / "build/model24_selected_token_position3_transaction3_reviews"
    / "authoritative-generation3-transaction3-only-r5-absence-predicate-repair-final"
)
R5_MANIFEST = R5_PACKAGE / "package-manifest.json"
R5_SEAL = R5_PACKAGE / "package-seal.json"
R5_REVIEW = R5_PACKAGE / "independent-review.json"
R5_AUTHORIZATION = R5_PACKAGE / "manager-transaction3-authorization.json"
R5_EXACT_EXECUTOR = RUNTIME / "recovery-r12/recovery_executor.py"
R5_TERMINAL_PARSER = ROOT / "ace3/model/validate_selected_token_position2_traversal.py"
RUNNER_STATUS = (
    ROOT
    / "build/ace3-tx3-r5-execution-evidence"
    / "05-durable-terminal-status.stdout.json"
)
RUNNER_STDOUT = (
    ROOT / ".argus_subagents/ace3-tx3-r5-execute-once_logs/stdout.log"
)
RUNNER_STDERR = (
    ROOT / ".argus_subagents/ace3-tx3-r5-execute-once_logs/stderr.log"
)
DEFAULT_OUTPUT = (
    ROOT
    / "build/model24_selected_token_position3_transaction3_publication_recovery"
    / "r5-postconsume-natural-terminal-mapping-r7"
)
DEFAULT_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction3_publication_recovery_reviews"
    / "r5-postconsume-natural-terminal-mapping-r7"
    / "independent-review.json"
)
DEFAULT_AUTHORITY_PACKAGE = (
    ROOT
    / "build/model24_selected_token_position3_transaction3_publication_recovery_authorities"
    / "r5-postconsume-natural-terminal-mapping-r7"
)
DEFAULT_AUTHORITY_REVIEW = (
    ROOT
    / "build/model24_selected_token_position3_transaction3_publication_recovery_authority_reviews"
    / "r5-postconsume-natural-terminal-mapping-r7"
    / "independent-review.json"
)
REVIEW_EMITTER_SOURCE = (
    ROOT / "ace3/model/emit_transaction3_publication_recovery_review.py"
)
AUTHORITY_REVIEW_EMITTER_SOURCE = (
    ROOT
    / "ace3/model/emit_transaction3_publication_recovery_authority_review.py"
)

START_CURSOR = 3
EXIT_CURSOR = 4
TRANSACTION_INDEX = 3
LAYER_INDEX = 2
HIDDEN_SIZE = 896
TRACE_COUNT = 23408

EXPECTED_SHA256 = {
    POINTER: "9489297fb3230c5b281c746c86a4bde6fba92cbb8af1d339f81178591c3de13d",
    GENERATION3
    / "generation-manifest.json": "bade4d17d6d9fe627ecb9c34f39b08b50cad85341f897be6d38ffbd75096b02c",
    GENERATION3
    / "ledger.json": "f4c44a7717d392dd8e328d4430a62d233c23865efb0bd59f48718d0f31af0af2",
    GENERATION3
    / "checkpoints/transaction-000.json": "9fabdd9a91f1331220ca4e2c5682cc17e8be18fed444799f97cd3c66276cbdd0",
    GENERATION3
    / "checkpoints/transaction-001.json": "6019478026ea51b2cefd34e4a2e261edca8cb1f02034c1c5ed8fb20e0c3250a7",
    GENERATION3
    / "checkpoints/transaction-002.json": "31330e3e421ced0289415a2e864ca428535c01a97f2c7e157fa2bfeda82a7552",
    R5_SEAL: "b3b7d7747cb54570cee28b07f16664b63a3b53a8b75757b23d7652b9aa5740d4",
    R5_REVIEW: "18205ace0fa938079db085ece7f5124b1922e7e6d60c698b1c5eb524ce8913c1",
    R5_AUTHORIZATION: "05105bf6f7f04252f04f13c5b5de584d2112558df6c7b17af8f2971922d2e6d0",
    CONSUMPTION: "803a89340967094fd4e2875635ab27e94a3314cab31c9795d7a948d617b66a89",
    EXECUTION_START: "24be70a905665174391b63653b1a0c7521c850bb78d502e4553a2973a09ef320",
    FAILURE_SEAL: "7c0245d0aafe3a2065bf01659c5a81b4ee3b40da03bccc3776c5ebaf15ad98ab",
    RUNNER_STATUS: "6c1c866bc65c945b081649d99b1dd87a01d73b0223286a9e570c78442d39e816",
    RUNNER_STDOUT: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    RUNNER_STDERR: "36ca188e9101a20ef3891f25bc6c1a7ed62d8052babb628d4098cd9a9483b2f8",
    R5_EXACT_EXECUTOR: "8308de743d5dbb7e3ff9d74080b70834c9bb945a94460090d564b53ba52ef726",
    R5_TERMINAL_PARSER: "6b9fe66a6ba3fd0249ce7e618a6fd395907707a39225d8d56565e4ba8f6ec00c",
    TRANSACTION3
    / "comparison.json": "23c5792963b9e4bf58abe4ec948982608a4b7bbc80f80ff77fcd7af36d70ead2",
    TRANSACTION3
    / "position003/raw/terminal.txt": "d94b645e69d075ca952d0bc13af43c699a72158700592c94028897478c4f807c",
    TRANSACTION3
    / "position003/raw/trace.hex": "d90d1382eec19f296571fe639cc16312364c55e31a24296cfedd2601ca260f84",
    TRANSACTION3
    / "position003/raw/final.hex": "07457cc78c7064b0c9802233e6f2c7a3e18e634b1f66acdb84f4001f084a7a9c",
    TRANSACTION3
    / "position003/exact_oracle/trace.hex": "d90d1382eec19f296571fe639cc16312364c55e31a24296cfedd2601ca260f84",
    TRANSACTION3
    / "position003/exact_oracle/final.hex": "07457cc78c7064b0c9802233e6f2c7a3e18e634b1f66acdb84f4001f084a7a9c",
    TRANSACTION3
    / "position003/simulation.log": "bfc8383cd69a0b3f29fc3823768dd680eab0e32f2dbb353754ea87ca09089395",
}
EXPECTED_TRANSACTION3_TREE = {
    "file_count": 69,
    "total_bytes": 25208783,
    "tree_sha256": "3be5e041f076b16d14553de04865c1beb4a7833b9d4dace01c69bbd4e3f3accb",
}
EXPECTED_HIDDEN_SEMANTIC_SHA256 = (
    "95e7911ce6afbc8dba048e3910b22489288fedfdeee33b47021622d937b67463"
)
PACKAGE_MEMBERS = (
    "adjudication.json",
    "package-manifest.json",
    "publication_recovery.py",
    "review-emitter.py",
    "review-request.json",
)
AUTHORITY_PACKAGE_MEMBERS = (
    "manager-publication-recovery-authority.json",
    "review-emitter.py",
    "review-request.json",
)
EXPECTED_R7_SHA256 = {
    "adjudication": "850026965fa2c55b902756e6e59b5a4140cd3785997c34f7e618da4a4e894b7e",
    "package_manifest": "87c6f0aaf03316834e01b209de073bd610fda482ce3e6ff7028f4991f7ca212b",
    "package_seal": "dcb02706a3daaf510771d7cccc5bc782fa83c8c4aaee8ff6876dca0840183587",
    "publication_recovery": "89f6ecbda9cf10dc972974a706136a69a38db12bb091ba8c3aea0ff837a058a6",
    "independent_review": "cf6f92873167909c8799d68920d83b7cf55edfefd598f3b658cbbe908ea46baf",
}
AUTHORITY_PROHIBITIONS = (
    "model generation",
    "oracle generation",
    "RTL compilation",
    "RTL simulation",
    "transaction003 replay",
    "new transaction identity",
    "transaction004-025 execution",
)


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
    require(stat.S_ISREG(metadata.st_mode), f"regular file required: {path}")
    require(not stat.S_ISLNK(metadata.st_mode), f"symlink rejected: {path}")
    document = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    require(isinstance(document, dict), f"JSON object required: {path}")
    return document


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(1024 * 1024):
            digest.update(payload)
    return digest.hexdigest()


def file_record(path: Path, published_path: Path | None = None) -> dict[str, Any]:
    metadata = path.lstat()
    require(not stat.S_ISLNK(metadata.st_mode), f"symlink rejected: {path}")
    require(stat.S_ISREG(metadata.st_mode), f"regular file required: {path}")
    return {
        "path": str(published_path if published_path is not None else path),
        "bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def inventory_record(path: Path, relative: Path) -> dict[str, Any]:
    record = file_record(path)
    return {
        "relative_path": relative.as_posix(),
        "path": record["path"],
        "bytes": record["bytes"],
        "mode": f"{stat.S_IMODE(path.lstat().st_mode):04o}",
        "sha256": record["sha256"],
    }


def tree_inventory(root: Path) -> list[dict[str, Any]]:
    require(root.is_dir() and not root.is_symlink(), f"directory required: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        records.append(inventory_record(path, path.relative_to(root)))
    return records


def tree_summary(root: Path) -> dict[str, Any]:
    records = tree_inventory(root)
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["relative_path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "root": str(root),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "tree_sha256": digest.hexdigest(),
    }


def authenticate_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    require(
        isinstance(record.get("path"), str)
        and type(record.get("bytes")) is int
        and isinstance(record.get("sha256"), str),
        f"{label} record is malformed",
    )
    actual = file_record(Path(record["path"]))
    require(actual == dict(record), f"{label} content binding mismatch")
    return actual


def authenticate_inventory(
    root: Path, expected: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    actual = tree_inventory(root)
    require(actual == list(expected), f"frozen inventory differs: {root}")
    return actual


def parse_terminal(payload: bytes) -> dict[str, int]:
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise RecoveryError("RTL terminal is not ASCII") from error
    require(len(lines) == 1, "RTL terminal line count mismatch")
    fields: dict[str, str] = {}
    for field in lines[0].split():
        require(field.count("=") == 1, "RTL terminal field is malformed")
        key, value = field.split("=", 1)
        require(key not in fields, f"duplicate RTL terminal field: {key}")
        fields[key] = value
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
        and fields["layer_index"] == str(LAYER_INDEX)
        and fields["position"] == "3"
        and fields["natural_terminal"] == "1"
        and fields["exit_code"] == "0",
        "RTL did not naturally terminate",
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
        binary.extend(int(row[6:10], 16).to_bytes(2, "little"))
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
        "comparison does not report exact zero-mismatch oracle agreement",
    )


def validate_parent(pointer: Mapping[str, Any], ledger: Mapping[str, Any]) -> None:
    require(
        pointer.get("schema_version") == 1
        and pointer.get("status") == "COMMITTED"
        and pointer.get("generation") == START_CURSOR
        and pointer.get("runtime_identity") == RUNTIME_IDENTITY,
        "wrong authoritative parent generation",
    )
    require(
        ledger.get("state_generation") == START_CURSOR
        and ledger.get("next_transaction_index") == START_CURSOR
        and ledger.get("completed_transaction_count") == START_CURSOR
        and ledger.get("authoritative_state_root") == str(GENERATION3)
        and len(ledger.get("completed_receipts", [])) == START_CURSOR,
        "generation3 ledger is not authoritative cursor3",
    )


def require_publishable_namespace(
    pointer: Mapping[str, Any],
    *,
    generation4_exists: bool,
    staging_exists: bool,
    receipt_candidate_exists: bool,
    execution_evidence_exists: bool,
    success_terminal_exists: bool,
    later_transactions: Sequence[int],
) -> None:
    validate_parent(pointer, load_json(GENERATION3 / "ledger.json"))
    require(not generation4_exists, "generation4 already exists; replay rejected")
    require(not staging_exists, "partial generation4 staging exists")
    require(not receipt_candidate_exists, "legacy receipt candidate exists")
    require(not execution_evidence_exists, "legacy execution evidence exists")
    require(not success_terminal_exists, "legacy success terminal exists")
    require(not later_transactions, "transaction004-025 evidence exists")


def validate_source_boundary() -> dict[str, Any]:
    exact_source = R5_EXACT_EXECUTOR.read_text(encoding="utf-8")
    parser_source = R5_TERMINAL_PARSER.read_text(encoding="utf-8")
    require(
        exact_source.count('natural_terminal = rtl.get("natural_terminal")') == 1
        and exact_source.count(
            'natural_terminal = rtl["raw"].get("natural_terminal")'
        )
        == 1
        and exact_source.count(
            '"natural_rtl_terminal": natural_terminal is True'
        )
        == 1,
        "receipt mapping source differs",
    )
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
        and "return counts" in parser_segment
        and '"natural_terminal":' not in parser_segment,
        "terminal parser no longer exhibits the frozen mapping boundary",
    )
    return {
        "terminal_parser": file_record(R5_TERMINAL_PARSER),
        "receipt_adapter": file_record(R5_EXACT_EXECUTOR),
        "established_mapping": (
            "the parser authenticates natural_terminal=1 and exit_code=0 but "
            "returns only trace/final/done counts; the adapter consequently "
            "maps the absent natural_terminal field to false"
        ),
    }


def validate_package_source_boundary(source_path: Path) -> None:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_imports = {
        "importlib",
        "numpy",
        "safetensors",
        "subprocess",
        "torch",
    }
    forbidden_calls = {
        "compile",
        "eval",
        "exec",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "popen",
        "posix_spawn",
        "posix_spawnp",
        "system",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            require(
                all(alias.name.split(".", 1)[0] not in forbidden_imports for alias in node.names),
                "publication package imports executable/model/RTL support",
            )
        elif isinstance(node, ast.ImportFrom):
            require(
                (node.module or "").split(".", 1)[0] not in forbidden_imports,
                "publication package imports executable/model/RTL support",
            )
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            require(name.lower() not in forbidden_calls, f"forbidden call in package: {name}")


def validate_fixed_inputs() -> None:
    for path, expected in EXPECTED_SHA256.items():
        require(file_record(path)["sha256"] == expected, f"frozen input changed: {path}")
    require(
        {
            key: value
            for key, value in tree_summary(TRANSACTION3).items()
            if key != "root"
        }
        == EXPECTED_TRANSACTION3_TREE,
        "transaction003 tree differs",
    )


def validate_r5_package() -> None:
    seal = load_json(R5_SEAL)
    for label, record in seal.get("members", {}).items():
        authenticate_record(record, f"r5 package {label}")
    review = load_json(R5_REVIEW)
    require(
        review.get("status") == "PASS"
        and review.get("producer_role") == "reviewer"
        and review.get("this_review_authorizes_execution") is False,
        "consumed r5 independent review differs",
    )
    authorization = load_json(R5_AUTHORIZATION)
    require(
        authorization.get("decision") == "AUTHORIZE_TRANSACTION3_ONCE"
        and authorization.get("transaction_index") == TRANSACTION_INDEX
        and authorization.get("replay_authorized") is False,
        "consumed r5 authorization differs",
    )
    consumption = load_json(CONSUMPTION)
    require(
        consumption.get("status") == "CONSUMED_ONCE"
        and consumption.get("transaction_index") == TRANSACTION_INDEX
        and consumption.get("layer_index") == LAYER_INDEX
        and consumption.get("replay_authorized") is False
        and consumption.get("model_execution_authorized") is False
        and consumption.get("package_seal") == file_record(R5_SEAL)
        and consumption.get("independent_review") == file_record(R5_REVIEW)
        and consumption.get("manager_authorization") == file_record(R5_AUTHORIZATION),
        "r5 authority consumption differs",
    )


def validate_execution_boundary() -> dict[str, Any]:
    start = load_json(EXECUTION_START)
    require(
        start
        == {
            "schema_version": 1,
            "kind": "ace3_position3_transaction3_execution_start",
            "transaction_index": TRANSACTION_INDEX,
            "layer_index": LAYER_INDEX,
            "started_ns": 1788258054718722992,
            "cached_evidence": False,
        },
        "execution-start differs",
    )
    runner = load_json(RUNNER_STATUS)
    require(
        runner.get("task_id") == "ace3-tx3-r5-execute-once"
        and runner.get("run_id")
        == "ace3-tx3-r5-execute-once-1788258054528560482"
        and runner.get("state") == "error"
        and runner.get("live") is False
        and runner.get("exit_code") == 1
        and "transaction3_executor.py execute" in runner.get("command", ""),
        "durable one-shot terminal differs",
    )
    traceback = RUNNER_STDERR.read_text(encoding="utf-8")
    expected_message = (
        "continuation_control.ControlError: transaction 3 required result "
        "natural_rtl_terminal mismatch"
    )
    require(
        traceback.count("Traceback (most recent call last):") == 1
        and traceback.rstrip().endswith(expected_message),
        "failing receipt-validation traceback differs",
    )
    require(RUNNER_STDOUT.read_bytes() == b"", "one-shot stdout is not empty")
    failure = load_json(FAILURE_SEAL)
    require(
        failure.get("status") == "FAIL_CLOSED_NON_REPLAYABLE"
        and failure.get("authority_state") == "CONSUMED_ONCE"
        and failure.get("authoritative_generation_preserved") == START_CURSOR
        and failure.get("authoritative_cursor_preserved") == START_CURSOR
        and failure.get("generation4_published") is False
        and failure.get("replay_authorized") is False,
        "fail-closed terminal differs",
    )
    return {
        "execution_start": file_record(EXECUTION_START),
        "durable_runner_status": file_record(RUNNER_STATUS),
        "stdout": file_record(RUNNER_STDOUT),
        "traceback": file_record(RUNNER_STDERR),
        "fail_closed_terminal": file_record(FAILURE_SEAL),
    }


def validate_computation_evidence() -> dict[str, Any]:
    terminal_path = TRANSACTION3 / "position003/raw/terminal.txt"
    counts = parse_terminal(terminal_path.read_bytes())
    comparison_path = TRANSACTION3 / "comparison.json"
    comparison = load_json(comparison_path)
    validate_comparison(comparison)
    raw_trace = TRANSACTION3 / "position003/raw/trace.hex"
    raw_final = TRANSACTION3 / "position003/raw/final.hex"
    oracle_trace = TRANSACTION3 / "position003/exact_oracle/trace.hex"
    oracle_final = TRANSACTION3 / "position003/exact_oracle/final.hex"
    vector_trace = TRANSACTION3 / "vectors/trace.hex"
    vector_final = TRANSACTION3 / "vectors/final.hex"
    require(
        raw_trace.read_bytes()
        == oracle_trace.read_bytes()
        == vector_trace.read_bytes(),
        "raw/vector/oracle trace bytes differ",
    )
    final_payload = raw_final.read_bytes()
    require(
        final_payload == oracle_final.read_bytes() == vector_final.read_bytes(),
        "raw/vector/oracle final bytes differ",
    )
    semantic_sha256 = semantic_hidden_sha256(final_payload)
    require(
        semantic_sha256 == EXPECTED_HIDDEN_SEMANTIC_SHA256,
        "hidden semantic hash differs",
    )
    boundary = load_json(TRANSACTION3 / "vectors/boundary_manifest.json")
    require(
        boundary.get("layer_index") == LAYER_INDEX
        and boundary.get("position") == 3
        and boundary.get("trace_records") == TRACE_COUNT
        and boundary.get("final_records") == HIDDEN_SIZE
        and boundary.get("trace") == file_record(vector_trace)
        and boundary.get("final_hidden") == file_record(vector_final),
        "vector boundary manifest differs",
    )
    simulation = (
        TRANSACTION3 / "position003/simulation.log"
    ).read_text(encoding="ascii")
    require(
        simulation.count("DECODER_LAYER_TOKEN_TRANSACTION_PASS") == 1
        and simulation.rstrip().endswith(
            "DECODER_LAYER_TOKEN_TRANSACTION_PASS layer=2 position=3 "
            "trace_count=23408 final_count=896 cycles=15399417 stalls=214569"
        ),
        "simulation terminal log differs",
    )
    output_state = TRANSACTION3 / "position004.state"
    require(
        output_state.read_bytes().startswith(b"verilatorsave01\n"),
        "output state is not a Verilator save",
    )
    return {
        "terminal": file_record(terminal_path),
        "terminal_counts": counts,
        "comparison": file_record(comparison_path),
        "raw_trace": file_record(raw_trace),
        "raw_final": file_record(raw_final),
        "oracle_trace": file_record(oracle_trace),
        "oracle_final": file_record(oracle_final),
        "vector_trace": file_record(vector_trace),
        "vector_final": file_record(vector_final),
        "simulation_log": file_record(
            TRANSACTION3 / "position003/simulation.log"
        ),
        "output_state": file_record(output_state),
        "hidden_semantic_sha256": semantic_sha256,
    }


def current_later_transactions() -> list[int]:
    return [
        index
        for index in range(4, 26)
        if (TRANSACTIONS / f"transaction-{index:03d}").exists()
    ]


def validate_live_namespace() -> None:
    pointer = load_json(POINTER)
    require_publishable_namespace(
        pointer,
        generation4_exists=GENERATION4.exists(),
        staging_exists=GENERATION4_STAGING.exists(),
        receipt_candidate_exists=RECEIPT_CANDIDATE.exists(),
        execution_evidence_exists=EXECUTION_EVIDENCE.exists(),
        success_terminal_exists=SUCCESS_TERMINAL.exists(),
        later_transactions=current_later_transactions(),
    )


def receipt_from_evidence(
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = load_json(R5_MANIFEST)
    descriptor = manifest["transaction_descriptor"]
    source_manifest = load_json(R5_PACKAGE / "source-manifest.json")
    artifacts = {
        record["relative_path"]: {
            "path": record["path"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        for record in inventory
    }
    hidden = file_record(TRANSACTION3 / "position003/raw/final.hex")
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
        "transaction_index": TRANSACTION_INDEX,
        "operation": descriptor["operation"],
        "input_binding_sha256": descriptor["input_binding_sha256"],
        "authenticated_inputs": copy.deepcopy(descriptor["inputs"]),
        "transaction3_authority_consumption": file_record(CONSUMPTION),
        "candidate_manifest": file_record(R5_MANIFEST),
        "source_bindings": copy.deepcopy(source_manifest["sources"]),
        "result": {
            "exact_integer_oracle_match": True,
            "natural_rtl_terminal": True,
            "output_hidden_elements": HIDDEN_SIZE,
            "output_state_position": 4,
        },
        "outputs": {
            "hidden": hidden,
            "state": file_record(TRANSACTION3 / "position004.state"),
        },
        "output_semantics": {
            "hidden": {
                "dtype": "FP16",
                "elements": HIDDEN_SIZE,
                "semantic_sha256": EXPECTED_HIDDEN_SEMANTIC_SHA256,
            },
            "state": {
                "layer_index": LAYER_INDEX,
                "position": 4,
                "parent_position": 2,
                "parent": copy.deepcopy(descriptor["inputs"]["position2_kv_parent"]),
            },
        },
        "artifacts": artifacts,
        "rtl_reference_agreement": {
            "mismatches": 0,
            "agreement": True,
        },
        "timing": {
            "started_at": None,
            "completed_at": None,
            "transaction_seconds": None,
            "cumulative_execution_seconds": load_json(
                GENERATION3 / "ledger.json"
            )["cumulative_execution_seconds"],
            "basis": (
                "transaction-003 timing was not durably preserved before receipt "
                "validation failed; no transaction-003 latency is claimed"
            ),
        },
        "publication_recovery": {
            "kind": "frozen-evidence-receipt-reconstruction",
            "adapts_only": "natural_rtl_terminal",
            "model_rerun": False,
            "oracle_rerun": False,
            "rtl_rerun": False,
            "transaction_replay": False,
        },
    }
    validate_receipt(descriptor, receipt)
    return receipt


def validate_receipt(
    descriptor: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    require(
        receipt.get("schema_version") == 1
        and receipt.get("kind") == "ace3_position3_transaction_completion"
        and receipt.get("status") == "COMPLETE"
        and receipt.get("transaction_index") == TRANSACTION_INDEX
        and receipt.get("operation") == descriptor["operation"]
        and receipt.get("input_binding_sha256")
        == descriptor["input_binding_sha256"],
        "reconstructed receipt identity differs",
    )
    require(
        receipt.get("result") == descriptor["required_result"],
        "reconstructed receipt result differs",
    )
    outputs = receipt.get("outputs", {})
    semantics = receipt.get("output_semantics", {})
    authenticate_record(
        {
            key: outputs["hidden"][key]
            for key in ("path", "bytes", "sha256")
        },
        "receipt hidden output",
    )
    authenticate_record(outputs["state"], "receipt state output")
    require(
        semantics.get("hidden")
        == {
            "dtype": "FP16",
            "elements": HIDDEN_SIZE,
            "semantic_sha256": EXPECTED_HIDDEN_SEMANTIC_SHA256,
        }
        and semantics.get("state", {}).get("layer_index") == LAYER_INDEX
        and semantics.get("state", {}).get("position") == 4,
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


def ledger4_document(receipt_payload: bytes) -> dict[str, Any]:
    ledger = copy.deepcopy(load_json(GENERATION3 / "ledger.json"))
    checkpoint_records = [
        file_record(
            GENERATION3 / f"checkpoints/transaction-{index:03d}.json",
            GENERATION4 / f"checkpoints/transaction-{index:03d}.json",
        )
        for index in range(3)
    ]
    checkpoint_records.append(
        {
            "path": str(GENERATION4 / "checkpoints/transaction-003.json"),
            "bytes": len(receipt_payload),
            "sha256": sha256_bytes(receipt_payload),
        }
    )
    ledger.update(
        {
            "authoritative_state_root": str(GENERATION4),
            "completed_receipts": checkpoint_records,
            "completed_transaction_count": EXIT_CURSOR,
            "next_transaction_index": EXIT_CURSOR,
            "state_generation": EXIT_CURSOR,
            "status": "IN_PROGRESS",
            "cumulative_execution_seconds_incomplete": True,
            "missing_transaction_timing_indices": [2, 3],
        }
    )
    return ledger


def adjudicate() -> dict[str, Any]:
    validate_fixed_inputs()
    validate_r5_package()
    validate_live_namespace()
    source_boundary = validate_source_boundary()
    execution_boundary = validate_execution_boundary()
    computation = validate_computation_evidence()
    inventory = tree_inventory(TRANSACTION3)
    receipt = receipt_from_evidence(inventory)
    receipt_payload = canonical_json(receipt)
    ledger_payload = canonical_json(ledger4_document(receipt_payload))
    return {
        "schema_version": 1,
        "kind": "ace3_transaction3_postconsume_publication_recovery_adjudication",
        "status": "PASS",
        "runtime_identity": RUNTIME_IDENTITY,
        "transaction_index": TRANSACTION_INDEX,
        "authoritative_parent": {
            "generation": START_CURSOR,
            "cursor": START_CURSOR,
            "pointer": file_record(POINTER),
            "ledger": file_record(GENERATION3 / "ledger.json"),
            "checkpoints": [
                file_record(
                    GENERATION3 / f"checkpoints/transaction-{index:03d}.json"
                )
                for index in range(3)
            ],
        },
        "consumed_r5": {
            "package_seal": file_record(R5_SEAL),
            "independent_review": file_record(R5_REVIEW),
            "manager_authorization": file_record(R5_AUTHORIZATION),
            "authority_consumption": file_record(CONSUMPTION),
            "replay_authorized": False,
        },
        "execution_boundary": execution_boundary,
        "computation_evidence": computation,
        "source_boundary": source_boundary,
        "frozen_transaction003": {
            "tree": tree_summary(TRANSACTION3),
            "inventory": inventory,
        },
        "finding": {
            "natural_terminal": 1,
            "rtl_exit_code": 0,
            "done_count": 1,
            "final_count": HIDDEN_SIZE,
            "integer_mismatches": 0,
            "exact_oracle_agreement": True,
            "sole_failure": "completion receipt natural_rtl_terminal mapping",
            "generation4_publication_attempted": False,
        },
        "reconstructed_receipt": receipt,
        "publication_preview": {
            "generation": EXIT_CURSOR,
            "cursor": EXIT_CURSOR,
            "checkpoint003": {
                "bytes": len(receipt_payload),
                "sha256": sha256_bytes(receipt_payload),
            },
            "ledger": {
                "bytes": len(ledger_payload),
                "sha256": sha256_bytes(ledger_payload),
            },
            "live_publication_performed": False,
        },
        "recovery_authority_created": False,
        "independent_reviewer_required": True,
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path: Path, payload: bytes, mode: int = 0o400) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_package(output: Path = DEFAULT_OUTPUT, review_path: Path = DEFAULT_REVIEW) -> Path:
    require(output.is_absolute(), "package output must be absolute")
    require(review_path.is_absolute(), "review path must be absolute")
    require(not output.exists(), f"package output already exists: {output}")
    require(not review_path.exists(), f"review output already exists: {review_path}")
    validate_package_source_boundary(Path(__file__).resolve())
    validate_package_source_boundary(REVIEW_EMITTER_SOURCE)
    adjudication = adjudicate()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.preparing")
    require(not staging.exists(), f"package staging already exists: {staging}")
    staging.mkdir(mode=0o700)
    try:
        source_payload = Path(__file__).read_bytes()
        write_new(staging / "publication_recovery.py", source_payload)
        write_new(staging / "review-emitter.py", REVIEW_EMITTER_SOURCE.read_bytes())
        write_new(staging / "adjudication.json", canonical_json(adjudication))
        package_manifest = {
            "schema_version": 1,
            "kind": "ace3_transaction3_publication_only_recovery_package",
            "status": "SEALED_REVIEW_REQUIRED",
            "runtime_identity": RUNTIME_IDENTITY,
            "transaction_index": TRANSACTION_INDEX,
            "source_generation": START_CURSOR,
            "destination_generation": EXIT_CURSOR,
            "publication": {
                "visibility_point": str(POINTER),
                "generation": str(GENERATION4),
                "checkpoint": str(
                    GENERATION4 / "checkpoints/transaction-003.json"
                ),
                "cursor": EXIT_CURSOR,
            },
            "review_path": str(review_path),
            "reviewer_emitter": str(output / "review-emitter.py"),
            "recovery_authority": None,
            "computation": {
                "model": 0,
                "oracle": 0,
                "rtl": 0,
                "transaction": 0,
            },
            "forbidden": [
                "model import or invocation",
                "oracle import or invocation",
                "RTL build or invocation",
                "transaction executor import or invocation",
                "child process or external executable",
                "transaction003 replay",
                "transaction004-025 execution",
            ],
        }
        write_new(staging / "package-manifest.json", canonical_json(package_manifest))
        review_request = {
            "schema_version": 1,
            "kind": "ace3_transaction3_publication_recovery_review_request",
            "required_role": "reviewer",
            "preparation_participation_required": False,
            "review_output": str(review_path),
            "reviewer_emitter": file_record(
                staging / "review-emitter.py",
                output / "review-emitter.py",
            ),
            "review_invocation": [
                "/usr/bin/python3",
                str(output / "review-emitter.py"),
                "--package",
                str(output),
                "--output",
                str(review_path),
            ],
            "adjudication": file_record(
                staging / "adjudication.json",
                output / "adjudication.json",
            ),
            "requested_judgment": "PASS_OR_REJECT",
            "review_must_not": [
                "create recovery authority",
                "publish generation4",
                "execute model, oracle, RTL, or transaction",
                "mutate frozen r5 evidence",
            ],
        }
        write_new(staging / "review-request.json", canonical_json(review_request))
        members = {
            name: file_record(staging / name, output / name)
            for name in PACKAGE_MEMBERS
        }
        seal = {
            "schema_version": 1,
            "kind": "ace3_transaction3_publication_only_recovery_seal",
            "status": "SEALED_REVIEW_REQUIRED",
            "members": members,
            "adjudication_status": "PASS",
            "publication_performed": False,
            "recovery_authority_created": False,
        }
        write_new(staging / "package-seal.json", canonical_json(seal))
        fsync_directory(staging)
        os.rename(staging, output)
        fsync_directory(output.parent)
        output.chmod(0o555)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_package(output)
    return output


def validate_package(package: Path) -> dict[str, Any]:
    require(package.is_dir() and not package.is_symlink(), "package directory is absent")
    require(stat.S_IMODE(package.stat().st_mode) & 0o222 == 0, "package is writable")
    seal = load_json(package / "package-seal.json")
    require(
        seal.get("kind") == "ace3_transaction3_publication_only_recovery_seal"
        and seal.get("status") == "SEALED_REVIEW_REQUIRED"
        and seal.get("publication_performed") is False
        and seal.get("recovery_authority_created") is False,
        "package seal identity differs",
    )
    members = seal.get("members")
    require(isinstance(members, dict), "package seal members are absent")
    require(set(members) == set(PACKAGE_MEMBERS), "package member set differs")
    for name, record in members.items():
        expected_path = package / name
        require(record.get("path") == str(expected_path), f"sealed path differs: {name}")
        authenticate_record(record, f"package member {name}")
        require(
            stat.S_IMODE(expected_path.stat().st_mode) & 0o222 == 0,
            f"package member is writable: {name}",
        )
    actual = {
        path.name for path in package.iterdir() if path.is_file()
    }
    require(
        actual == set(PACKAGE_MEMBERS) | {"package-seal.json"},
        "package file set differs",
    )
    manifest = load_json(package / "package-manifest.json")
    require(
        manifest.get("status") == "SEALED_REVIEW_REQUIRED"
        and manifest.get("recovery_authority") is None
        and manifest.get("computation")
        == {"model": 0, "oracle": 0, "rtl": 0, "transaction": 0},
        "package computation or authority boundary differs",
    )
    adjudication = load_json(package / "adjudication.json")
    review_request = load_json(package / "review-request.json")
    require(
        adjudication.get("status") == "PASS"
        and adjudication.get("recovery_authority_created") is False
        and adjudication.get("publication_preview", {}).get(
            "live_publication_performed"
        )
        is False,
        "adjudication is not a no-publication PASS",
    )
    require(
        review_request.get("required_role") == "reviewer"
        and review_request.get("review_output") == manifest.get("review_path")
        and review_request.get("reviewer_emitter")
        == file_record(package / "review-emitter.py")
        and review_request.get("review_invocation")
        == [
            "/usr/bin/python3",
            str(package / "review-emitter.py"),
            "--package",
            str(package),
            "--output",
            manifest.get("review_path"),
        ]
        and review_request.get("adjudication")
        == file_record(package / "adjudication.json"),
        "review request final-package binding differs",
    )
    authenticate_inventory(
        TRANSACTION3,
        adjudication["frozen_transaction003"]["inventory"],
    )
    validate_package_source_boundary(package / "publication_recovery.py")
    validate_package_source_boundary(package / "review-emitter.py")
    return adjudication


def validate_review(package: Path, review_path: Path) -> dict[str, Any]:
    review = load_json(review_path)
    require(
        stat.S_IMODE(review_path.stat().st_mode) & 0o222 == 0,
        "independent review is writable",
    )
    require(
        review.get("schema_version") == 1
        and review.get("kind")
        == "ace3_transaction3_publication_recovery_independent_review"
        and review.get("status") in {"PASS", "REJECT"}
        and review.get("producer_role") == "reviewer"
        and review.get("preparation_participation") is False
        and review.get("recovery_authority_created") is False
        and review.get("publication_performed") is False
        and review.get("package_seal") == file_record(package / "package-seal.json")
        and review.get("adjudication")
        == file_record(package / "adjudication.json"),
        "independent review identity or package binding differs",
    )
    require(review["status"] == "PASS", "independent Reviewer rejected recovery")
    return review


def zero_activity_counters() -> dict[str, int]:
    return {
        "authority_consumption": 0,
        "model_generation": 0,
        "new_transaction_identity": 0,
        "oracle_generation": 0,
        "publication": 0,
        "rtl_compilation": 0,
        "rtl_simulation": 0,
        "transaction003_replay": 0,
        "transaction004_025_execution": 0,
    }


def recovery_argv(package: Path, review_path: Path) -> list[str]:
    return [
        "/usr/bin/python3",
        str(package / "publication_recovery.py"),
        "publish",
        "--package",
        str(package),
        "--review",
        str(review_path),
    ]


def validate_accepted_r7(
    package: Path, review_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    adjudication = validate_package(package)
    review = validate_review(package, review_path)
    fixed = {
        "adjudication": package / "adjudication.json",
        "package_manifest": package / "package-manifest.json",
        "package_seal": package / "package-seal.json",
        "publication_recovery": package / "publication_recovery.py",
        "independent_review": review_path,
    }
    for label, path in fixed.items():
        require(
            sha256_file(path) == EXPECTED_R7_SHA256[label],
            f"accepted r7 {label} hash differs",
        )
    require(
        review.get("status") == "PASS"
        and adjudication.get("status") == "PASS",
        "accepted r7 gate is not PASS",
    )
    validate_live_namespace()
    return adjudication, review


def authority_document(
    package: Path,
    review_path: Path,
    authority_review_path: Path,
    adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    computation = adjudication["computation_evidence"]
    execution = adjudication["execution_boundary"]
    parent = adjudication["authoritative_parent"]
    return {
        "schema_version": 1,
        "kind": "ace3_transaction3_publication_only_recovery_manager_authority",
        "status": "AUTHORIZED_NOT_CONSUMED",
        "producer_role": "manager",
        "scope": "publication-only frozen-evidence receipt recovery",
        "runtime_identity": RUNTIME_IDENTITY,
        "transaction_index": TRANSACTION_INDEX,
        "new_transaction_identity": None,
        "consumed_original_r5": copy.deepcopy(adjudication["consumed_r5"]),
        "execution_identity": {
            "runtime_identity": RUNTIME_IDENTITY,
            "transaction_index": TRANSACTION_INDEX,
            "execution_start": copy.deepcopy(execution["execution_start"]),
            "durable_runner_status": copy.deepcopy(
                execution["durable_runner_status"]
            ),
        },
        "sealed_failure_terminal": copy.deepcopy(
            execution["fail_closed_terminal"]
        ),
        "frozen_evidence": {
            "raw_rtl_terminal": copy.deepcopy(computation["terminal"]),
            "raw_rtl_trace": copy.deepcopy(computation["raw_trace"]),
            "raw_rtl_final": copy.deepcopy(computation["raw_final"]),
            "oracle_trace": copy.deepcopy(computation["oracle_trace"]),
            "oracle_final": copy.deepcopy(computation["oracle_final"]),
            "comparison": copy.deepcopy(computation["comparison"]),
            "output_state": copy.deepcopy(computation["output_state"]),
            "hidden_semantic_sha256": computation["hidden_semantic_sha256"],
        },
        "accepted_recovery": {
            "package_root": str(package),
            "adjudication": file_record(package / "adjudication.json"),
            "package_manifest": file_record(package / "package-manifest.json"),
            "package_seal": file_record(package / "package-seal.json"),
            "publication_recovery": file_record(
                package / "publication_recovery.py"
            ),
            "independent_reviewer_artifact": file_record(review_path),
        },
        "authoritative_parent": {
            "generation": START_CURSOR,
            "cursor": START_CURSOR,
            "latest_checkpoint_index": 2,
            "pointer": copy.deepcopy(parent["pointer"]),
            "ledger": copy.deepcopy(parent["ledger"]),
            "checkpoint002": copy.deepcopy(parent["checkpoints"][2]),
        },
        "authorized_argv": recovery_argv(package, review_path),
        "prohibitions": list(AUTHORITY_PROHIBITIONS),
        "activity_counters": zero_activity_counters(),
        "authority_cardinality": 1,
        "authority_consumed": False,
        "publication_performed": False,
        "reviewer_acceptance": {
            "required": True,
            "path": str(authority_review_path),
            "present_at_issuance": False,
        },
    }


def validate_authority_package(
    authority_package: Path,
    package: Path = DEFAULT_OUTPUT,
    review_path: Path = DEFAULT_REVIEW,
    authority_review_path: Path = DEFAULT_AUTHORITY_REVIEW,
) -> dict[str, Any]:
    require(
        authority_package.is_dir() and not authority_package.is_symlink(),
        "authority package directory is absent",
    )
    require(
        stat.S_IMODE(authority_package.stat().st_mode) & 0o222 == 0,
        "authority package is writable",
    )
    actual_files = {
        path.name for path in authority_package.iterdir() if path.is_file()
    }
    require(
        actual_files == set(AUTHORITY_PACKAGE_MEMBERS) | {"authority-seal.json"},
        "authority package file set differs",
    )
    seal = load_json(authority_package / "authority-seal.json")
    require(
        seal.get("schema_version") == 1
        and seal.get("kind")
        == "ace3_transaction3_publication_recovery_authority_seal"
        and seal.get("status") == "AUTHORIZED_NOT_CONSUMED"
        and seal.get("authority_cardinality") == 1
        and seal.get("activity_counters") == zero_activity_counters(),
        "authority seal differs",
    )
    members = seal.get("members")
    require(
        isinstance(members, dict)
        and set(members) == set(AUTHORITY_PACKAGE_MEMBERS),
        "authority seal member set differs",
    )
    for name, record in members.items():
        path = authority_package / name
        require(record.get("path") == str(path), f"authority path differs: {name}")
        authenticate_record(record, f"authority member {name}")
        require(
            stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
            f"authority member is writable: {name}",
        )
    authority = load_json(
        authority_package / "manager-publication-recovery-authority.json"
    )
    adjudication, _ = validate_accepted_r7(package, review_path)
    require(
        authority
        == authority_document(
            package,
            review_path,
            authority_review_path,
            adjudication,
        ),
        "Manager authority content differs",
    )
    request = load_json(authority_package / "review-request.json")
    require(
        request.get("required_role") == "reviewer"
        and request.get("authority")
        == file_record(
            authority_package
            / "manager-publication-recovery-authority.json"
        )
        and request.get("review_output") == str(authority_review_path)
        and request.get("review_invocation")
        == [
            "/usr/bin/python3",
            str(authority_package / "review-emitter.py"),
            "--authority-package",
            str(authority_package),
            "--output",
            str(authority_review_path),
        ],
        "authority review request differs",
    )
    return authority


def issue_authority(
    authority_package: Path = DEFAULT_AUTHORITY_PACKAGE,
    package: Path = DEFAULT_OUTPUT,
    review_path: Path = DEFAULT_REVIEW,
    authority_review_path: Path = DEFAULT_AUTHORITY_REVIEW,
) -> Path:
    for path in (
        authority_package,
        package,
        review_path,
        authority_review_path,
    ):
        require(path.is_absolute(), f"absolute path required: {path}")
    require(
        not authority_package.exists(),
        f"authority package already exists: {authority_package}",
    )
    require(
        not authority_review_path.exists(),
        f"authority review already exists: {authority_review_path}",
    )
    validate_package_source_boundary(AUTHORITY_REVIEW_EMITTER_SOURCE)
    adjudication, _ = validate_accepted_r7(package, review_path)
    authority_package.parent.mkdir(parents=True, exist_ok=True)
    competing = list(
        authority_package.parent.parent.rglob(
            "manager-publication-recovery-authority.json"
        )
    )
    require(not competing, "a Manager publication-recovery authority already exists")
    authority_review_path.parent.mkdir(parents=True, exist_ok=True)
    staging = authority_package.with_name(f".{authority_package.name}.preparing")
    require(not staging.exists(), f"authority staging already exists: {staging}")
    staging.mkdir(mode=0o700)
    try:
        authority = authority_document(
            package,
            review_path,
            authority_review_path,
            adjudication,
        )
        write_new(
            staging / "manager-publication-recovery-authority.json",
            canonical_json(authority),
        )
        write_new(
            staging / "review-emitter.py",
            AUTHORITY_REVIEW_EMITTER_SOURCE.read_bytes(),
        )
        request = {
            "schema_version": 1,
            "kind": (
                "ace3_transaction3_publication_recovery_authority_review_request"
            ),
            "required_role": "reviewer",
            "manager_participation_required": False,
            "review_output": str(authority_review_path),
            "authority": file_record(
                staging / "manager-publication-recovery-authority.json",
                authority_package
                / "manager-publication-recovery-authority.json",
            ),
            "reviewer_emitter": file_record(
                staging / "review-emitter.py",
                authority_package / "review-emitter.py",
            ),
            "review_invocation": [
                "/usr/bin/python3",
                str(authority_package / "review-emitter.py"),
                "--authority-package",
                str(authority_package),
                "--output",
                str(authority_review_path),
            ],
            "requested_judgment": "PASS_OR_REJECT",
            "review_must_not": [
                "consume the Manager authority",
                "publish generation4",
                "generate model or oracle evidence",
                "compile or simulate RTL",
                "replay transaction003",
                "create a transaction identity",
                "execute transaction004-025",
            ],
        }
        write_new(staging / "review-request.json", canonical_json(request))
        members = {
            name: file_record(staging / name, authority_package / name)
            for name in AUTHORITY_PACKAGE_MEMBERS
        }
        seal = {
            "schema_version": 1,
            "kind": "ace3_transaction3_publication_recovery_authority_seal",
            "status": "AUTHORIZED_NOT_CONSUMED",
            "members": members,
            "authority_cardinality": 1,
            "activity_counters": zero_activity_counters(),
            "authority_consumed": False,
            "publication_performed": False,
        }
        write_new(staging / "authority-seal.json", canonical_json(seal))
        fsync_directory(staging)
        os.rename(staging, authority_package)
        fsync_directory(authority_package.parent)
        authority_package.chmod(0o555)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_authority_package(
        authority_package,
        package,
        review_path,
        authority_review_path,
    )
    return authority_package


def generation_payloads(
    package: Path, review_path: Path
) -> tuple[dict[str, bytes], bytes, bytes]:
    adjudication = validate_package(package)
    validate_review(package, review_path)
    validate_fixed_inputs()
    validate_live_namespace()
    authenticate_inventory(
        TRANSACTION3,
        adjudication["frozen_transaction003"]["inventory"],
    )
    receipt_payload = canonical_json(adjudication["reconstructed_receipt"])
    ledger = ledger4_document(receipt_payload)
    ledger_payload = canonical_json(ledger)
    files = {
        f"checkpoints/transaction-{index:03d}.json": (
            GENERATION3 / f"checkpoints/transaction-{index:03d}.json"
        ).read_bytes()
        for index in range(3)
    }
    files["checkpoints/transaction-003.json"] = receipt_payload
    files["ledger.json"] = ledger_payload
    records = {
        name: {
            "path": str(GENERATION4 / name),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
        for name, payload in files.items()
    }
    manifest = {
        "schema_version": 1,
        "kind": "ace3_transaction3_publication_recovery_state_generation",
        "status": "PREPARED",
        "runtime_identity": RUNTIME_IDENTITY,
        "generation": EXIT_CURSOR,
        "parent_generation": START_CURSOR,
        "parent_pointer": file_record(POINTER),
        "package_seal": file_record(package / "package-seal.json"),
        "adjudication": file_record(package / "adjudication.json"),
        "independent_review": file_record(review_path),
        "files": records,
        "model_execution": 0,
        "oracle_execution": 0,
        "rtl_execution": 0,
        "transaction_execution": 0,
        "recovery_authority": None,
    }
    manifest_payload = canonical_json(manifest)
    pointer = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction3_generation4_authoritative_pointer",
        "status": "COMMITTED",
        "runtime_identity": RUNTIME_IDENTITY,
        "generation": EXIT_CURSOR,
        "generation_manifest": {
            "path": str(GENERATION4 / "generation-manifest.json"),
            "bytes": len(manifest_payload),
            "sha256": sha256_bytes(manifest_payload),
        },
        "previous_authoritative_pointer_sha256": EXPECTED_SHA256[POINTER],
        "atomic_visibility_contract": (
            "pointer replacement is the sole visibility point from "
            "generation3/cursor3 to generation4/cursor4"
        ),
    }
    return files, manifest_payload, canonical_json(pointer)


def publish(package: Path, review_path: Path) -> None:
    with RUNTIME_LOCK.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RecoveryError("transaction or publication invocation is active") from error
        files, manifest_payload, pointer_payload = generation_payloads(
            package, review_path
        )
        GENERATION4_STAGING.mkdir(mode=0o700)
        (GENERATION4_STAGING / "checkpoints").mkdir(mode=0o700)
        for relative, payload in files.items():
            write_new(GENERATION4_STAGING / relative, payload)
        write_new(
            GENERATION4_STAGING / "generation-manifest.json",
            manifest_payload,
        )
        fsync_directory(GENERATION4_STAGING / "checkpoints")
        fsync_directory(GENERATION4_STAGING)
        os.rename(GENERATION4_STAGING, GENERATION4)
        fsync_directory(GENERATIONS)
        pointer_temporary = POINTER.with_name(
            ".authoritative-state.transaction3-publication.tmp"
        )
        write_new(pointer_temporary, pointer_payload, 0o600)
        os.replace(pointer_temporary, POINTER)
        fsync_directory(POINTER.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=(
            "prepare",
            "validate-package",
            "issue-authority",
            "validate-authority",
            "publish",
        ),
    )
    parser.add_argument("--package", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument(
        "--authority-package",
        type=Path,
        default=DEFAULT_AUTHORITY_PACKAGE,
    )
    parser.add_argument(
        "--authority-review",
        type=Path,
        default=DEFAULT_AUTHORITY_REVIEW,
    )
    arguments = parser.parse_args()
    if arguments.operation == "prepare":
        package = prepare_package(arguments.package, arguments.review)
        print(
            "TRANSACTION3_PUBLICATION_RECOVERY_PREPARED "
            f"package={package} generation=3 cursor=3 publication=0 "
            "model=0 oracle=0 rtl=0 transaction=0 authority=0"
        )
    elif arguments.operation == "validate-package":
        adjudication = validate_package(arguments.package)
        print(
            "TRANSACTION3_PUBLICATION_RECOVERY_VALID "
            f"package={arguments.package} adjudication={adjudication['status']} "
            "generation=3 cursor=3 publication=0"
        )
    elif arguments.operation == "issue-authority":
        authority_package = issue_authority(
            arguments.authority_package,
            arguments.package,
            arguments.review,
            arguments.authority_review,
        )
        print(
            "TRANSACTION3_PUBLICATION_RECOVERY_AUTHORIZED "
            f"authority_package={authority_package} "
            "status=AUTHORIZED_NOT_CONSUMED generation=3 cursor=3 "
            "publication=0 model=0 oracle=0 rtl_compile=0 rtl_simulation=0 "
            "transaction_replay=0 later_transactions=0"
        )
    elif arguments.operation == "validate-authority":
        authority = validate_authority_package(
            arguments.authority_package,
            arguments.package,
            arguments.review,
            arguments.authority_review,
        )
        print(
            "TRANSACTION3_PUBLICATION_RECOVERY_AUTHORITY_VALID "
            f"authority_package={arguments.authority_package} "
            f"status={authority['status']} publication=0 consumption=0"
        )
    else:
        publish(arguments.package, arguments.review)
        print(
            "TRANSACTION3_PUBLICATION_RECOVERY_PUBLISHED "
            "generation=4 cursor=4 checkpoint=003"
        )


if __name__ == "__main__":
    main()
