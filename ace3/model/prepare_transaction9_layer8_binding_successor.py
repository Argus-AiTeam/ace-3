#!/usr/bin/env python3
"""Prepare the transaction009/layer8 binding-closed successor without execution."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping


sys.dont_write_bytecode = True

ROOT = Path("/home/argustest/ace3-argus")
RUNTIME = ROOT / (
    "build/model24_selected_token_position3_runs/"
    "ace3-position3-fresh-r11-20260831t215500z"
)
REVISION = "r10"
MISSION_ID = "e4892be03408"
NODE_KEY = "tx009-r9-position2-review-authority-operator-answer"

PREDECESSOR_PACKAGE = RUNTIME / "transaction9-layer8-continuation-package-r5"
PREDECESSOR_REVIEW = (
    RUNTIME
    / "transaction9-layer8-independent-review-r5"
    / "independent-review.json"
)
PREDECESSOR_AUTHORITY = (
    RUNTIME
    / "transaction9-layer8-launch-authority-r5"
    / "launch-authority.json"
)

PACKAGE = RUNTIME / f"transaction9-layer8-continuation-package-{REVISION}"
BINDING_ROOT = (
    RUNTIME / f"transaction9-layer8-pre-authority-binding-{REVISION}"
)
BINDING = BINDING_ROOT / "binding-check.json"
REVIEW_ROOT = RUNTIME / f"transaction9-layer8-independent-review-{REVISION}"
REVIEW_EMITTER = REVIEW_ROOT / "review-emitter.py"
REVIEW = REVIEW_ROOT / "independent-review.json"
AUTHORITY_ROOT = RUNTIME / f"transaction9-layer8-launch-authority-{REVISION}"
AUTHORITY = AUTHORITY_ROOT / "launch-authority.json"
AUTHORITY_SEAL = AUTHORITY_ROOT / "authority-seal.json"
GATE_ROOT = RUNTIME / f"transaction9-layer8-read-only-gates-{REVISION}"
GATE_CHECK = GATE_ROOT / "gate-check.json"
READINESS_ROOT = RUNTIME / f"transaction9-layer8-successor-readiness-{REVISION}"
READINESS = READINESS_ROOT / "readiness.json"

ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
POINTER = ADOPTION / "authoritative-state.json"
GENERATION9 = ADOPTION / "state-generations/generation-0000000009"
GENERATION10 = ADOPTION / "state-generations/generation-0000000010"
GENERATION10_STAGING = (
    ADOPTION / "state-generations/.generation-0000000010.prepared"
)
TRANSACTIONS = RUNTIME / "transactions"
TRANSACTION8 = TRANSACTIONS / "transaction-008"
TRANSACTION9 = TRANSACTIONS / "transaction-009"
FUTURE_ROOT = RUNTIME / "transaction9-authoritative-generation10"
CONSUMPTION = FUTURE_ROOT / "manager-authorization-consumption.json"
TERMINAL = FUTURE_ROOT / "terminal.json"

PYTHON = Path("/home/argustest/miniconda3/bin/python3")
TRANSACTION_INDEX = 9
LAYER_INDEX = 8
ZERO_COUNTERS = {
    "authority_consumption": 0,
    "authority_issuance": 0,
    "model_execution": 0,
    "oracle_execution": 0,
    "payload_execution": 0,
    "rtl_compile": 0,
    "rtl_simulation": 0,
    "runtime_mutation": 0,
    "submission": 0,
    "transaction_execution": 0,
    "transaction_replay": 0,
    "transaction_resume": 0,
    "transaction_retry": 0,
    "vector_generation": 0,
}
PACKAGE_MEMBERS = {
    "authoritative_baseline": "authoritative-baseline.json",
    "binding_spec": "binding-spec.json",
    "executor": "transaction9_executor.py",
    "package_manifest": "package-manifest.json",
    "review_emitter": "review-emitter.py",
    "review_request": "review-request.json",
    "source_manifest": "source-manifest.json",
}


class SuccessorError(RuntimeError):
    """Raised when the successor cannot be prepared without side effects."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorError(message)


def canonical_json(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
        "ascii"
    )


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
        f"regular non-symlink JSON required: {path}",
    )
    document = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    require(isinstance(document, dict), f"JSON object required: {path}")
    return document


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(
    path: Path, published_path: Path | None = None
) -> dict[str, Any]:
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


def authenticate(
    record: Mapping[str, Any], path: Path, label: str
) -> None:
    require(
        set(record) == {"path", "bytes", "sha256"}
        and record.get("path") == str(path)
        and type(record.get("bytes")) is int
        and isinstance(record.get("sha256"), str)
        and dict(record) == file_record(path),
        f"{label} differs",
    )


def tree_record(root: Path) -> dict[str, Any]:
    require(
        root.is_dir() and not root.is_symlink(),
        f"real directory required: {root}",
    )
    files = sorted(path for path in root.iterdir() if path.is_file())
    require(
        all(not path.is_symlink() for path in files),
        f"symlinked tree member: {root}",
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.name.encode("utf-8")
        record = file_record(path)
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(record["bytes"].to_bytes(8, "big"))
        digest.update(bytes.fromhex(record["sha256"]))
        total_bytes += record["bytes"]
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def make_directory(path: Path) -> None:
    path.mkdir(mode=0o700)
    fsync_directory(path.parent)


def write_new(path: Path, payload: bytes, mode: int = 0o400) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal_directory(path: Path) -> None:
    for member in path.iterdir():
        if member.is_file():
            member.chmod(0o400)
    fsync_directory(path)
    path.chmod(0o500)
    fsync_directory(path.parent)


def canonical_launch() -> dict[str, Any]:
    return {
        "argv": [
            str(PYTHON),
            str(PACKAGE / "transaction9_executor.py"),
            "execute",
            "--package",
            str(PACKAGE),
            "--review",
            str(REVIEW),
            "--authorization",
            str(AUTHORITY),
            "--runtime-root",
            str(RUNTIME),
            "--transaction-index",
            str(TRANSACTION_INDEX),
        ],
        "cwd": str(ROOT),
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        },
        "exit_cursor": 10,
        "exit_generation": 10,
        "interpreter": str(PYTHON),
        "layer_index": LAYER_INDEX,
        "start_cursor": 9,
        "start_generation": 9,
        "transaction_index": TRANSACTION_INDEX,
    }


def expected_generation10_receipts() -> dict[str, Any]:
    return {
        "count": 10,
        "transaction_indices": list(range(10)),
        "copied_transaction_indices": list(range(9)),
        "new_transaction_index": 9,
        "checkpoint_paths": [
            f"checkpoints/transaction-{index:03d}.json"
            for index in range(10)
        ],
        "checkpoint009_included": True,
    }


def validate_parent() -> dict[str, Any]:
    pointer = load_json(POINTER)
    manifest = load_json(GENERATION9 / "generation-manifest.json")
    ledger = load_json(GENERATION9 / "ledger.json")
    checkpoint = load_json(
        GENERATION9 / "checkpoints/transaction-008.json"
    )
    require(
        pointer.get("kind")
        == "ace3_position3_transaction8_generation9_authoritative_pointer"
        and pointer.get("status") == "COMMITTED"
        and pointer.get("generation") == 9
        and pointer.get("generation_manifest")
        == file_record(GENERATION9 / "generation-manifest.json"),
        "authoritative pointer is not generation9",
    )
    require(
        manifest.get("kind")
        == "ace3_transaction8_publication_recovery_generation9_state"
        and manifest.get("generation") == 9
        and manifest.get("transactions009_025_executed") is False,
        "generation9 manifest differs",
    )
    require(
        ledger.get("kind") == "ace3_position3_r11_transaction_ledger"
        and ledger.get("state_generation") == 9
        and ledger.get("completed_transaction_count") == 9
        and ledger.get("next_transaction_index") == 9
        and ledger.get("transaction_count") == 26,
        "generation9 ledger is not cursor9",
    )
    require(
        checkpoint.get("kind")
        == "ace3_position3_transaction8_reconstructed_completion_receipt"
        and checkpoint.get("status") == "COMPLETE"
        and checkpoint.get("transaction_index") == 8
        and checkpoint.get("layer_index") == 7,
        "checkpoint008 is not accepted",
    )
    launch_consumption = (
        RUNTIME
        / "transaction8-authoritative-generation9"
        / "manager-authorization-consumption.json"
    )
    publication_consumption = (
        RUNTIME
        / "transaction8-authoritative-generation9"
        / "publication-authority-consumption.json"
    )
    publication_terminal = (
        RUNTIME
        / "transaction8-authoritative-generation9"
        / "publication-authority-terminal.json"
    )
    require(
        TRANSACTION8.is_dir()
        and launch_consumption.is_file()
        and publication_consumption.is_file()
        and publication_terminal.is_file(),
        "transaction008 consumed-non-reusable evidence is incomplete",
    )
    return {
        "pointer": file_record(POINTER),
        "generation_manifest": file_record(
            GENERATION9 / "generation-manifest.json"
        ),
        "ledger": file_record(GENERATION9 / "ledger.json"),
        "checkpoint008": file_record(
            GENERATION9 / "checkpoints/transaction-008.json"
        ),
        "transaction008_launch_consumption": file_record(
            launch_consumption
        ),
        "transaction008_publication_consumption": file_record(
            publication_consumption
        ),
        "transaction008_publication_terminal": file_record(
            publication_terminal
        ),
    }


def validate_zero_effects() -> dict[str, Any]:
    require(not CONSUMPTION.exists(), "transaction009 authority is consumed")
    require(not TERMINAL.exists(), "transaction009 terminal exists")
    require(not FUTURE_ROOT.exists(), "transaction009 runtime root exists")
    require(not TRANSACTION9.exists(), "transaction009 output exists")
    require(
        not GENERATION10.exists() and not GENERATION10_STAGING.exists(),
        "generation10 publication exists",
    )
    future_paths: list[str] = []
    for index in range(10, 26):
        transaction = TRANSACTIONS / f"transaction-{index:03d}"
        runtime_effect = (
            RUNTIME / f"transaction{index}-authoritative-generation{index + 1}"
        )
        generation = (
            ADOPTION
            / "state-generations"
            / f"generation-{index + 1:010d}"
        )
        generation_staging = generation.with_name(
            f".{generation.name}.prepared"
        )
        for path in (
            transaction,
            runtime_effect,
            generation,
            generation_staging,
        ):
            require(
                not path.exists(),
                f"transaction{index:03d} effect exists: {path}",
            )
            future_paths.append(str(path))
    return {
        "authority_consumption_absent": True,
        "terminal_absent": True,
        "transaction009_absent": True,
        "generation10_absent": True,
        "generation10_staging_absent": True,
        "transaction010_025_effects_absent": True,
        "checked_transaction010_025_paths": future_paths,
    }


def package_transaction_descriptor() -> dict[str, Any]:
    descriptor = copy.deepcopy(
        load_json(PREDECESSOR_PACKAGE / "package-manifest.json")[
            "transaction_descriptor"
        ]
    )
    require(
        descriptor.get("transaction_index") == TRANSACTION_INDEX
        and descriptor.get("layer_index") == LAYER_INDEX
        and descriptor.get("operation") == "position3-decoder-layer"
        and descriptor.get("inputs", {})
        .get("predecessor", {})
        .get("source_transaction_index")
        == 8
        and descriptor.get("inputs", {}).get("transaction_position") == 3,
        "predecessor transaction009 descriptor differs",
    )
    return descriptor


def binding_specification(
    parent: Mapping[str, Any],
    zero_effects: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor = package_transaction_descriptor()
    consumed_inputs = {
        "authoritative_parent": copy.deepcopy(dict(parent)),
        "generation9_completed_receipts": {
            f"checkpoint{index:03d}": file_record(
                GENERATION9
                / f"checkpoints/transaction-{index:03d}.json"
            )
            for index in range(9)
        },
        "fixture_manifest": copy.deepcopy(
            descriptor["inputs"]["fixture_manifest"]
        ),
        "position2_kv_parent": copy.deepcopy(
            descriptor["inputs"]["position2_kv_parent"]
        ),
        "position004_predecessor_state": copy.deepcopy(
            descriptor["inputs"]["predecessor"]["state"]
        ),
    }
    return {
        "schema_version": 1,
        "kind": "ace3_transaction009_layer8_canonical_binding_spec",
        "status": "SEALED_INPUT",
        "binding_mission_id": MISSION_ID,
        "binding_node_key": NODE_KEY,
        "execution_mission_id": "6b14d6917e3c",
        "parent_mission_id": "a9a6a6378b72",
        "package_id": (
            "generation9-cursor9-checkpoint008-"
            f"transaction009-layer08-binding-{REVISION}"
        ),
        "runtime_identity": RUNTIME.name,
        "authoritative_generation": 9,
        "authoritative_cursor": 9,
        "required_parent_checkpoint_index": 8,
        "transaction_index": TRANSACTION_INDEX,
        "layer_index": LAYER_INDEX,
        "transaction_position": 3,
        "target_generation": 10,
        "target_cursor": 10,
        "target_checkpoint_index": 9,
        "transaction_descriptor": descriptor,
        "compile": {
            "cwd": str(ROOT),
            "argv": [
                "make",
                "--no-print-directory",
                "model24-rtl-layer-compile",
                "MODEL24_RTL_LAYER_INDEX=8",
                "MODEL24_RTL_ACCURATE_SILU=1",
                f"MODEL24_RTL_CASCADE_DIR={TRANSACTION9 / 'build'}",
            ],
        },
        "simulation": {
            "cwd": str(ROOT),
            "argv": [
                str(
                    TRANSACTION9
                    / (
                        "build/compiled/layer8/obj_dir/"
                        "Vace3_decoder_layer0_token_engine"
                    )
                ),
                "--layer-index",
                "8",
                "--vector-dir",
                str(TRANSACTION9 / "vectors"),
                "--tensor-dir",
                str(TRANSACTION9 / "vectors"),
                "--raw-dir",
                str(TRANSACTION9 / "position003/raw"),
                "--transaction-position",
                "3",
                "--state-out",
                str(TRANSACTION9 / "position004.state"),
                "--progress-interval",
                "1000000",
                "--state-in",
                descriptor["inputs"]["position2_kv_parent"]["path"],
            ],
        },
        "launch": canonical_launch(),
        "generation10_completed_receipts": (
            expected_generation10_receipts()
        ),
        "zero_activity_counters": copy.deepcopy(ZERO_COUNTERS),
        "absence_predicates": copy.deepcopy(dict(zero_effects)),
        "consumed_binding_inputs": consumed_inputs,
    }


def _evaluate_constant(
    node: ast.expr, values: Mapping[str, object]
) -> object:
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, int)
    ):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and len(node.args) == 1
        and not node.keywords
    ):
        return Path(str(_evaluate_constant(node.args[0], values)))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _evaluate_constant(node.left, values)
        right = _evaluate_constant(node.right, values)
        return Path(left) / str(right)
    raise SuccessorError(
        f"unsupported executor constant expression: {ast.dump(node)}"
    )


def executor_constants(path: Path) -> dict[str, object]:
    wanted = {
        "ROOT",
        "RUNTIME",
        "PACKAGE_ROOT",
        "BINDING_SPEC",
        "REVIEW",
        "AUTHORITY",
        "EXECUTOR",
        "TRANSACTION_INDEX",
    }
    values: dict[str, object] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            try:
                values[name] = _evaluate_constant(node.value, values)
            except SuccessorError:
                if name in wanted:
                    raise
    require(wanted.issubset(values), "executor binding constants are absent")
    return {name: values[name] for name in wanted}


def serialized_constants(constants: Mapping[str, object]) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in sorted(constants.items())
    }


def require_binding(
    constants: Mapping[str, object], launch: Mapping[str, Any]
) -> None:
    expected = {
        "ROOT": ROOT,
        "RUNTIME": RUNTIME,
        "PACKAGE_ROOT": PACKAGE,
        "BINDING_SPEC": PACKAGE / "binding-spec.json",
        "REVIEW": REVIEW,
        "AUTHORITY": AUTHORITY,
        "EXECUTOR": PACKAGE / "transaction9_executor.py",
        "TRANSACTION_INDEX": TRANSACTION_INDEX,
    }
    require(dict(constants) == expected, "executor constants differ")
    require(dict(launch) == canonical_launch(), "canonical launch differs")
    argv = launch["argv"]
    require(
        argv[0] == str(PYTHON)
        and argv[1] == str(constants["EXECUTOR"])
        and argv[3:5] == ["--package", str(constants["PACKAGE_ROOT"])]
        and argv[5:7] == ["--review", str(constants["REVIEW"])]
        and argv[7:9] == ["--authorization", str(constants["AUTHORITY"])]
        and argv[9:11] == ["--runtime-root", str(constants["RUNTIME"])]
        and argv[11:13]
        == ["--transaction-index", str(constants["TRANSACTION_INDEX"])],
        "executor constants do not equal canonical argv",
    )


def successor_executor() -> bytes:
    source = (
        PREDECESSOR_PACKAGE / "transaction9_executor.py"
    ).read_text(encoding="utf-8")
    replacements = (
        (
            'PACKAGE_ID = "generation9-cursor9-checkpoint008-'
            'transaction009-layer08-binding-r5"',
            'PACKAGE_ID = "generation9-cursor9-checkpoint008-'
            'transaction009-layer08-binding-r10"',
        ),
        (
            'RUNTIME / "transaction9-layer8-continuation-package-r5"',
            'RUNTIME / "transaction9-layer8-continuation-package-r10"',
        ),
        (
            '/ "transaction9-layer8-independent-review-r5"',
            '/ "transaction9-layer8-independent-review-r10"',
        ),
        (
            '/ "transaction9-layer8-launch-authority-r5"',
            '/ "transaction9-layer8-launch-authority-r10"',
        ),
        (
            f'REVIEW_EMITTER_SOURCE = Path("{PREDECESSOR_REVIEW.parent}/'
            'review-emitter.py")',
            f'REVIEW_EMITTER_SOURCE = Path("{REVIEW_EMITTER}")',
        ),
        (
            'SUCCESSOR_MISSION_ID = "c600b9afe908"',
            f'SUCCESSOR_MISSION_ID = "{MISSION_ID}"',
        ),
        (
            'SUCCESSOR_NODE_KEY = "tx009-layer8-fresh-binding-authority"',
            f'SUCCESSOR_NODE_KEY = "{NODE_KEY}"',
        ),
        (
            '"--layer-index",\n        "7",',
            '"--layer-index",\n        "8",',
        ),
        (
            '    "transaction_execution": 0,\n'
            '    "vector_generation": 0,',
            '    "transaction_execution": 0,\n'
            '    "transaction_replay": 0,\n'
            '    "transaction_resume": 0,\n'
            '    "transaction_retry": 0,\n'
            '    "vector_generation": 0,',
        ),
    )
    for old, new in replacements:
        count = source.count(old)
        require(count == 1, f"executor replacement count differs: {old}")
        source = source.replace(old, new)
    source = source.replace("TRANSACTION8", "TRANSACTION9")
    package_root = (
        "PACKAGE_ROOT = (\n"
        '    RUNTIME / "transaction9-layer8-continuation-package-r10"\n'
        ")\n"
    )
    require(
        source.count(package_root) == 1,
        "executor package root declaration differs",
    )
    source = source.replace(
        package_root,
        package_root + 'BINDING_SPEC = PACKAGE_ROOT / "binding-spec.json"\n',
    )
    package_members = (
        'PACKAGE_MEMBERS = {\n'
        '    "authoritative_baseline": "authoritative-baseline.json",\n'
    )
    require(
        source.count(package_members) == 1,
        "executor package member declaration differs",
    )
    source = source.replace(
        package_members,
        package_members + '    "binding_spec": "binding-spec.json",\n',
    )

    def replace_region(
        text: str,
        start_name: str,
        end_name: str,
        replacement: str,
    ) -> str:
        start = text.index(f"def {start_name}")
        end = text.index(f"def {end_name}", start)
        return text[:start] + replacement + "\n\n" + text[end:]

    parent_gates = '''def authoritative_parent_records() -> dict[str, Any]:
    transaction8_root = RUNTIME / "transaction8-authoritative-generation9"
    return {
        "pointer": file_record(POINTER),
        "generation_manifest": file_record(
            GENERATION9 / "generation-manifest.json"
        ),
        "ledger": file_record(GENERATION9 / "ledger.json"),
        "checkpoint008": file_record(
            GENERATION9 / "checkpoints/transaction-008.json"
        ),
        "transaction008_launch_consumption": file_record(
            transaction8_root / "manager-authorization-consumption.json"
        ),
        "transaction008_publication_consumption": file_record(
            transaction8_root / "publication-authority-consumption.json"
        ),
        "transaction008_publication_terminal": file_record(
            transaction8_root / "publication-authority-terminal.json"
        ),
    }


def validate_parent_acceptance() -> dict[str, Any]:
    pointer = load_json(POINTER)
    manifest = load_json(GENERATION9 / "generation-manifest.json")
    ledger = load_json(GENERATION9 / "ledger.json")
    checkpoint = load_json(
        GENERATION9 / "checkpoints/transaction-008.json"
    )
    parent = authoritative_parent_records()
    require(
        pointer.get("kind")
        == "ace3_position3_transaction8_generation9_authoritative_pointer"
        and pointer.get("status") == "COMMITTED"
        and pointer.get("generation") == START_CURSOR
        and pointer.get("generation_manifest")
        == parent["generation_manifest"],
        "authoritative pointer is not generation9/cursor9",
    )
    require(
        manifest.get("kind")
        == "ace3_transaction8_publication_recovery_generation9_state"
        and manifest.get("generation") == START_CURSOR
        and manifest.get("transactions009_025_executed") is False,
        "generation9 manifest semantics differ",
    )
    require(
        ledger.get("kind") == "ace3_position3_r11_transaction_ledger"
        and ledger.get("state_generation") == START_CURSOR
        and ledger.get("completed_transaction_count") == START_CURSOR
        and ledger.get("next_transaction_index") == TRANSACTION_INDEX
        and ledger.get("transaction_count") == 26,
        "generation9 ledger is not cursor9",
    )
    require(
        checkpoint.get("kind")
        == "ace3_position3_transaction8_reconstructed_completion_receipt"
        and checkpoint.get("status") == "COMPLETE"
        and checkpoint.get("transaction_index") == 8
        and checkpoint.get("layer_index") == 7,
        "checkpoint008 is not accepted",
    )
    return ledger
'''
    source = replace_region(
        source,
        "validate_parent_acceptance()",
        "validate_absence()",
        parent_gates,
    )

    absence_gate = '''def validate_absence() -> None:
    require(
        not TRANSACTION9.exists()
        and not GENERATION10.exists()
        and not GENERATION10_STAGING.exists()
        and not FUTURE_ROOT.exists(),
        "transaction009 or generation10 namespace exists",
    )
    for index in range(10, 26):
        require(
            not (TRANSACTIONS / f"transaction-{index:03d}").exists()
            and not (
                RUNTIME
                / f"transaction{index}-authoritative-generation{index + 1}"
            ).exists()
            and not (
                GENERATIONS / f"generation-{index + 1:010d}"
            ).exists()
            and not (
                GENERATIONS / f".generation-{index + 1:010d}.prepared"
            ).exists(),
            f"transaction{index:03d} effect exists",
        )
'''
    source = replace_region(
        source,
        "validate_absence()",
        "baseline_document()",
        absence_gate,
    )

    baseline_gate = '''def validate_baseline(document: Mapping[str, Any]) -> None:
    require(
        document.get("kind")
        == "ace3_position3_transaction9_layer8_authoritative_baseline"
        and document.get("status") == "AUTHENTICATED_PARENT_NO_EXECUTION"
        and document.get("successor_mission_id") == SUCCESSOR_MISSION_ID
        and document.get("successor_node_key") == SUCCESSOR_NODE_KEY
        and document.get("runtime_identity") == RUNTIME_IDENTITY
        and document.get("authoritative_generation") == START_CURSOR
        and document.get("authoritative_cursor") == START_CURSOR
        and document.get("required_parent_checkpoint_index") == 8
        and document.get("transaction_index") == TRANSACTION_INDEX
        and document.get("layer_index") == LAYER_INDEX
        and document.get("authoritative_parent")
        == authoritative_parent_records()
        and document.get("transaction008_consumed_non_reusable") is True
        and document.get("transaction009_absent") is True
        and document.get("generation10_absent") is True
        and document.get("transactions010_025_absent") is True
        and document.get("activity_counters") == ZERO_COUNTERS,
        "authoritative baseline differs",
    )
'''
    source = replace_region(
        source,
        "baseline_document()",
        "validate_baseline(document: Mapping[str, Any])",
        "",
    )
    source = replace_region(
        source,
        "validate_baseline(document: Mapping[str, Any])",
        "build_source_manifest(",
        baseline_gate,
    )

    manifest_gate = '''def validate_manifest_document(
    document: Mapping[str, Any],
) -> None:
    spec = load_json(BINDING_SPEC)
    descriptor = transaction_descriptor()
    descriptor.pop("completion_receipt", None)
    descriptor["inputs"]["predecessor"]["state"] = file_record(
        TRANSACTIONS / "transaction-008/position004.state"
    )
    descriptor["status"] = "PENDING_AUTHORIZED_NOT_CONSUMED"
    require(
        spec.get("kind")
        == "ace3_transaction009_layer8_canonical_binding_spec"
        and spec.get("status") == "SEALED_INPUT"
        and spec.get("binding_mission_id") == SUCCESSOR_MISSION_ID
        and spec.get("binding_node_key") == SUCCESSOR_NODE_KEY
        and spec.get("execution_mission_id") == MISSION_ID
        and spec.get("parent_mission_id") == PARENT_MISSION_ID
        and spec.get("package_id") == PACKAGE_ID
        and spec.get("runtime_identity") == RUNTIME_IDENTITY
        and spec.get("authoritative_generation") == START_CURSOR
        and spec.get("authoritative_cursor") == START_CURSOR
        and spec.get("required_parent_checkpoint_index") == 8
        and spec.get("transaction_index") == TRANSACTION_INDEX
        and spec.get("layer_index") == LAYER_INDEX
        and spec.get("transaction_position") == POSITION
        and spec.get("target_generation") == EXIT_CURSOR
        and spec.get("target_cursor") == EXIT_CURSOR
        and spec.get("target_checkpoint_index") == 9
        and spec.get("transaction_descriptor") == descriptor
        and spec.get("compile")
        == {"cwd": str(ROOT), "argv": compile_argv()}
        and spec.get("simulation")
        == {"cwd": str(ROOT), "argv": simulation_argv()}
        and spec.get("launch") == launch_contract()
        and spec.get("generation10_completed_receipts")
        == {
            "count": EXIT_CURSOR,
            "transaction_indices": list(range(EXIT_CURSOR)),
            "copied_transaction_indices": list(range(START_CURSOR)),
            "new_transaction_index": TRANSACTION_INDEX,
            "checkpoint_paths": [
                f"checkpoints/transaction-{index:03d}.json"
                for index in range(EXIT_CURSOR)
            ],
            "checkpoint009_included": True,
        }
        and spec.get("zero_activity_counters") == ZERO_COUNTERS,
        "canonical binding specification differs",
    )
    require(
        document.get("schema_version") == 1
        and document.get("kind")
        == "ace3_position3_transaction9_layer8_executor_package"
        and document.get("status") == "SEALED_REVIEW_REQUIRED"
        and document.get("mission_id") == spec["binding_mission_id"]
        and document.get("node_key") == spec["binding_node_key"]
        and document.get("successor_mission_id")
        == spec["binding_mission_id"]
        and document.get("successor_node_key")
        == spec["binding_node_key"]
        and document.get("execution_mission_id")
        == spec["execution_mission_id"]
        and document.get("parent_mission_id")
        == spec["parent_mission_id"]
        and document.get("package_id") == spec["package_id"]
        and document.get("runtime_identity") == spec["runtime_identity"]
        and document.get("authoritative_generation")
        == spec["authoritative_generation"]
        and document.get("authoritative_cursor")
        == spec["authoritative_cursor"]
        and document.get("required_parent_checkpoint_index")
        == spec["required_parent_checkpoint_index"]
        and document.get("transaction_index")
        == spec["transaction_index"]
        and document.get("layer_index") == spec["layer_index"]
        and document.get("transaction_position")
        == spec["transaction_position"]
        and document.get("permitted_transaction_indices")
        == PERMITTED_INDICES
        and document.get("forbidden_transaction_indices")
        == FORBIDDEN_INDICES
        and document.get("transaction_descriptor")
        == spec["transaction_descriptor"]
        and document.get("authoritative_generation9")
        == authoritative_parent_records()
        and document.get("full_output_comparison")
        == full_output_comparison_contract()
        and document.get("binding_spec") == file_record(BINDING_SPEC)
        and document.get("compile") == spec["compile"]
        and document.get("simulation") == spec["simulation"]
        and document.get("launch") == spec["launch"]
        and document.get("transaction009_output_namespace")
        == str(TRANSACTION9)
        and document.get("generation10_output_namespace")
        == str(GENERATION10)
        and document.get("generation10_completed_receipts")
        == spec["generation10_completed_receipts"]
        and document.get("transaction009_absence_predicates")
        == spec["absence_predicates"]
        and document.get("restart_adoption")
        == restart_adoption_contract()
        and document.get("review_output") == str(REVIEW)
        and document.get("future_manager_authority") == str(AUTHORITY)
        and document.get("execution_authorized") is False
        and document.get("authority_created") is False
        and document.get("activity_counters")
        == spec["zero_activity_counters"],
        "package identity, parent, argv, receipt contract, or zero state differs",
    )
    require(
        document.get("prohibitions")
        == [
            "transaction000-008 retry, replay, or resume",
            "transaction010-025 execution or authority",
            "model, oracle, or vector execution during package review",
            "RTL compile or simulation during package review",
            "generation10 publication during package review",
            "restart or adoption of partial transaction009 state",
            "execution before separate Manager authority and Reviewer PASS",
            "authority consumption or publication by this bounded task",
        ],
        "package prohibitions differ",
    )
'''
    source = replace_region(
        source,
        "validate_manifest_document(document: Mapping[str, Any])",
        "validate_package(",
        manifest_gate,
    )

    gate_start = source.index("def validate_review() -> dict[str, Any]:")
    gate_end = source.index(
        "def write_exclusive_json(path: Path, document: object) -> None:"
    )
    successor_gates = '''def validate_review() -> dict[str, Any]:
    review = load_json(REVIEW)
    authority = load_json(AUTHORITY)
    expected_counters = {
        **ZERO_COUNTERS,
        "transaction_replay": 0,
        "transaction_resume": 0,
        "transaction_retry": 0,
    }
    require(
        review.get("kind")
        == "ace3_transaction009_layer8_binding_successor_independent_review"
        and review.get("status") == "PASS"
        and review.get("producer_role") == "reviewer"
        and review.get("mission_id") == SUCCESSOR_MISSION_ID
        and review.get("node_key") == SUCCESSOR_NODE_KEY
        and review.get("package_manifest")
        == file_record(OUTPUT_PACKAGE_MANIFEST)
        and review.get("package_seal") == file_record(PACKAGE_SEAL)
        and review.get("pre_authority_binding_check")
        == authority.get("pre_authority_binding_check")
        and review.get("authority") == file_record(AUTHORITY)
        and review.get("authority_seal")
        == file_record(AUTHORITY.parent / "authority-seal.json")
        and review.get("authorized_launch") == launch_contract()
        and review.get("executor_constants_equal_canonical_argv") is True
        and review.get("authoritative_generation") == START_CURSOR
        and review.get("authoritative_cursor") == START_CURSOR
        and review.get("target_generation") == EXIT_CURSOR
        and review.get("target_cursor") == EXIT_CURSOR
        and review.get("transaction_index") == TRANSACTION_INDEX
        and review.get("layer_index") == LAYER_INDEX
        and review.get("transaction008_consumed_non_reusable") is True
        and review.get("transaction009_authority_consumed") is False
        and review.get("transaction009_execution_performed") is False
        and review.get("transaction009_terminal_present") is False
        and review.get("generation10_present") is False
        and review.get("transaction010_025_authority_or_effects") is False
        and review.get("activity_counters") == expected_counters
        and review.get("total_workload_calls") == 0,
        "independent transaction009 successor review differs",
    )
    return review


def validate_authority() -> dict[str, Any]:
    authority = load_json(AUTHORITY)
    validate_review()
    transaction8_root = RUNTIME / "transaction8-authoritative-generation9"
    expected_parent = {
        "pointer": file_record(POINTER),
        "generation_manifest": file_record(
            GENERATION9 / "generation-manifest.json"
        ),
        "ledger": file_record(GENERATION9 / "ledger.json"),
        "checkpoint008": file_record(
            GENERATION9 / "checkpoints/transaction-008.json"
        ),
        "transaction008_launch_consumption": file_record(
            transaction8_root / "manager-authorization-consumption.json"
        ),
        "transaction008_publication_consumption": file_record(
            transaction8_root / "publication-authority-consumption.json"
        ),
        "transaction008_publication_terminal": file_record(
            transaction8_root / "publication-authority-terminal.json"
        ),
    }
    expected_consumed = {
        "status": "CONSUMED_ONCE_NON_REUSABLE",
        "launch_authority_consumption": expected_parent[
            "transaction008_launch_consumption"
        ],
        "publication_authority_consumption": expected_parent[
            "transaction008_publication_consumption"
        ],
        "publication_terminal": expected_parent[
            "transaction008_publication_terminal"
        ],
    }
    expected_counters = {
        **ZERO_COUNTERS,
        "transaction_replay": 0,
        "transaction_resume": 0,
        "transaction_retry": 0,
    }
    expected_namespaces = {
        "authority": str(AUTHORITY),
        "authority_consumption": str(CONSUMPTION),
        "checkpoint009": str(
            GENERATION10 / "checkpoints/transaction-009.json"
        ),
        "generation10": str(GENERATION10),
        "transaction009": str(TRANSACTION9),
    }
    require(
        authority.get("kind")
        == "ace3_position3_transaction9_layer8_launch_authority"
        and authority.get("status") == "RUNTIME_PASS_ACCEPTED"
        and authority.get("manager_status") == "MANAGER_RELEASE_ACCEPTED"
        and authority.get("mission_id") == SUCCESSOR_MISSION_ID
        and authority.get("node_key") == SUCCESSOR_NODE_KEY
        and authority.get("producer_role") == "manager"
        and authority.get("runtime_identity") == RUNTIME_IDENTITY
        and authority.get("authoritative_generation") == START_CURSOR
        and authority.get("authoritative_cursor") == START_CURSOR
        and authority.get("required_parent_checkpoint_index") == 8
        and authority.get("target_generation") == EXIT_CURSOR
        and authority.get("target_cursor") == EXIT_CURSOR
        and authority.get("target_checkpoint_index") == 9
        and authority.get("transaction_index") == TRANSACTION_INDEX
        and authority.get("layer_index") == LAYER_INDEX
        and authority.get("permitted_transaction_indices")
        == PERMITTED_INDICES
        and authority.get("forbidden_transaction_indices")
        == FORBIDDEN_INDICES
        and authority.get("package_manifest")
        == file_record(OUTPUT_PACKAGE_MANIFEST)
        and authority.get("package_seal") == file_record(PACKAGE_SEAL)
        and authority.get("source_manifest") == file_record(SOURCE_MANIFEST)
        and authority.get("authoritative_baseline") == file_record(BASELINE)
        and authority.get("aggregate_preflight")
        == authority.get("pre_authority_binding_check")
        and authority.get("authorized_launch") == launch_contract()
        and authority.get("authoritative_generation9") == expected_parent
        and authority.get("transaction008_consumed_non_reusable")
        == expected_consumed
        and authority.get("output_namespaces") == expected_namespaces
        and authority.get("reviewer_acceptance")
        == {
            "required": True,
            "path": str(REVIEW),
            "present_at_issuance": False,
        }
        and authority.get("activity_counters") == expected_counters
        and authority.get("authority_consumed") is False
        and authority.get("transaction009_executed") is False
        and authority.get("generation10_exists") is False
        and authority.get("transactions010_025_absent") is True
        and authority.get("preconsumption_workload_authorized") is False
        and authority.get("retry_authorized") is False
        and authority.get("replay_authorized") is False
        and authority.get("resume_authorized") is False,
        "transaction009 successor launch authority differs",
    )
    return authority


'''
    source = source[:gate_start] + successor_gates + source[gate_end:]

    publication_helpers = '''def validate_generation10_publication_structure() -> dict[str, Any]:
    expected = {
        "count": EXIT_CURSOR,
        "transaction_indices": list(range(EXIT_CURSOR)),
        "copied_transaction_indices": list(range(START_CURSOR)),
        "new_transaction_index": TRANSACTION_INDEX,
        "checkpoint_paths": [
            f"checkpoints/transaction-{index:03d}.json"
            for index in range(EXIT_CURSOR)
        ],
        "checkpoint009_included": True,
    }
    spec = load_json(BINDING_SPEC)
    contract = spec.get("generation10_completed_receipts")
    bound_receipts = spec.get("consumed_binding_inputs", {}).get(
        "generation9_completed_receipts"
    )
    require(contract == expected, "generation10 receipt contract differs")
    require(
        isinstance(bound_receipts, dict)
        and set(bound_receipts)
        == {f"checkpoint{index:03d}" for index in range(START_CURSOR)},
        "generation9 completed receipt bindings differ",
    )
    for index in range(START_CURSOR):
        authenticate(
            bound_receipts[f"checkpoint{index:03d}"],
            f"generation9 checkpoint{index:03d}",
            GENERATION9 / f"checkpoints/transaction-{index:03d}.json",
        )
    return contract


def generation10_checkpoint_payloads(
    receipt: Mapping[str, Any],
) -> dict[str, bytes]:
    contract = validate_generation10_publication_structure()
    checkpoint_payloads: dict[str, bytes] = {}
    for index in contract["copied_transaction_indices"]:
        relative = f"checkpoints/transaction-{index:03d}.json"
        checkpoint_payloads[relative] = (GENERATION9 / relative).read_bytes()
    new_relative = (
        f"checkpoints/transaction-{contract['new_transaction_index']:03d}.json"
    )
    checkpoint_payloads[new_relative] = canonical_json(receipt)
    require(
        list(checkpoint_payloads) == contract["checkpoint_paths"],
        "generation10 publication payload is not exactly receipts 000-009",
    )
    return checkpoint_payloads
'''
    publication_marker = "def publish_generation10(\n"
    require(
        source.count(publication_marker) == 1,
        "executor publication function differs",
    )
    source = source.replace(
        publication_marker,
        publication_helpers + "\n\n" + publication_marker,
    )

    old_publication_assembly = '''    checkpoint_payloads = {
        f"checkpoints/transaction-{index:03d}.json": (
            GENERATION9 / f"checkpoints/transaction-{index:03d}.json"
        ).read_bytes()
        for index in range(8)
    }
    checkpoint_payloads["checkpoints/transaction-009.json"] = canonical_json(
        receipt
    )
    checkpoint_records = [
        {
            "path": str(GENERATION10 / f"checkpoints/transaction-{index:03d}.json"),
            "bytes": len(checkpoint_payloads[f"checkpoints/transaction-{index:03d}.json"]),
            "sha256": sha256_bytes(
                checkpoint_payloads[f"checkpoints/transaction-{index:03d}.json"]
            ),
        }
        for index in range(EXIT_CURSOR)
    ]
    require(
        len(checkpoint_records) == EXIT_CURSOR
        and checkpoint_records[-1]["path"]
        == str(GENERATION10 / "checkpoints/transaction-009.json"),
        "generation10 completed receipts must include transactions000-008",
    )
'''
    new_publication_assembly = '''    receipt_contract = validate_generation10_publication_structure()
    checkpoint_payloads = generation10_checkpoint_payloads(receipt)
    checkpoint_records = [
        {
            "path": str(GENERATION10 / relative),
            "bytes": len(checkpoint_payloads[relative]),
            "sha256": sha256_bytes(checkpoint_payloads[relative]),
        }
        for relative in receipt_contract["checkpoint_paths"]
    ]
    require(
        len(checkpoint_records) == receipt_contract["count"]
        and [record["path"] for record in checkpoint_records]
        == [
            str(GENERATION10 / relative)
            for relative in receipt_contract["checkpoint_paths"]
        ],
        "generation10 completed receipts must be exactly transactions000-009",
    )
'''
    require(
        source.count(old_publication_assembly) == 1,
        "executor generation10 publication assembly differs",
    )
    source = source.replace(
        old_publication_assembly,
        new_publication_assembly,
    )
    source = source.replace(
        "generation=8 cursor=8 checkpoint=006",
        "generation=9 cursor=9 checkpoint=008",
    )
    source = source.replace(
        "transaction=7 layer=6 generation=8 cursor=8",
        "transaction=9 layer=8 generation=10 cursor=10",
    )
    require(
        "transaction9-layer8-continuation-package-r5" not in source
        and "transaction9-layer8-independent-review-r5" not in source
        and "transaction9-layer8-launch-authority-r5" not in source,
        "predecessor r5 binding remains in successor executor",
    )
    return source.encode("utf-8")


def review_emitter_source() -> bytes:
    source = f'''#!/usr/bin/env python3
"""Source-disjoint review of the transaction009/layer8 binding successor."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping
ROOT = Path({str(ROOT)!r})
RUNTIME = Path({str(RUNTIME)!r})
PACKAGE = Path({str(PACKAGE)!r})
BINDING = Path({str(BINDING)!r})
AUTHORITY = Path({str(AUTHORITY)!r})
AUTHORITY_SEAL = Path({str(AUTHORITY_SEAL)!r})
REVIEW = Path({str(REVIEW)!r})
TRANSACTION9 = Path({str(TRANSACTION9)!r})
GENERATION10 = Path({str(GENERATION10)!r})
GENERATION10_STAGING = Path({str(GENERATION10_STAGING)!r})
FUTURE_ROOT = Path({str(FUTURE_ROOT)!r})
TRANSACTION_INDEX = 9
LAYER_INDEX = 8
ZERO_COUNTERS = {ZERO_COUNTERS!r}
class ReviewError(RuntimeError):
    pass
def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)
def canonical_json(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\\n").encode("ascii")
def load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {{}}
        for key, value in pairs:
            require(key not in out, f"duplicate JSON key {{key}}: {{path}}")
            out[key] = value
        return out
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"regular non-symlink JSON required: {{path}}")
    document = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    require(isinstance(document, dict), f"JSON object required: {{path}}")
    return document
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
def file_record(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"regular non-symlink file required: {{path}}")
    return {{"path": str(path), "bytes": metadata.st_size, "sha256": sha256_file(path)}}
def authenticate(record: Mapping[str, Any], path: Path, label: str) -> None:
    require(set(record) == {{"path", "bytes", "sha256"}} and dict(record) == file_record(path), f"{{label}} differs")
def tree_record(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.iterdir() if path.is_file())
    digest = hashlib.sha256()
    total = 0
    for path in files:
        record = file_record(path)
        relative = path.name.encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(record["bytes"].to_bytes(8, "big"))
        digest.update(bytes.fromhex(record["sha256"]))
        total += record["bytes"]
    return {{"root": str(root), "file_count": len(files), "total_bytes": total, "tree_sha256": digest.hexdigest()}}
def evaluate(node: ast.expr, values: Mapping[str, object]) -> object:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path" and len(node.args) == 1:
        return Path(str(evaluate(node.args[0], values)))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return Path(evaluate(node.left, values)) / str(evaluate(node.right, values))
    raise ReviewError("unsupported executor constant expression")
def constants(path: Path) -> dict[str, object]:
    wanted = {{"ROOT", "RUNTIME", "PACKAGE_ROOT", "REVIEW", "AUTHORITY", "EXECUTOR", "TRANSACTION_INDEX"}}
    values: dict[str, object] = {{}}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            try:
                values[name] = evaluate(node.value, values)
            except ReviewError:
                if name in wanted:
                    raise
    require(wanted.issubset(values), "executor constants absent")
    return {{name: values[name] for name in wanted}}
def write_new(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REVIEW)
    arguments = parser.parse_args()
    manifest = load_json(PACKAGE / "package-manifest.json")
    seal = load_json(PACKAGE / "package-seal.json")
    request = load_json(PACKAGE / "review-request.json")
    executor = PACKAGE / "transaction9_executor.py"
    observed = constants(executor)
    expected = {{"ROOT": ROOT, "RUNTIME": RUNTIME, "PACKAGE_ROOT": PACKAGE, "REVIEW": REVIEW, "AUTHORITY": AUTHORITY, "EXECUTOR": executor, "TRANSACTION_INDEX": TRANSACTION_INDEX}}
    require(observed == expected, "executor constants differ from successor paths")
    require(manifest.get("launch", {{}}).get("argv") == [{str(PYTHON)!r}, str(executor), "execute", "--package", str(PACKAGE), "--review", str(REVIEW), "--authorization", str(AUTHORITY), "--runtime-root", str(RUNTIME), "--transaction-index", "9"], "manifest canonical argv differs")
    for label, relative in {PACKAGE_MEMBERS!r}.items():
        authenticate(seal["members"][label], PACKAGE / relative, f"sealed {{label}}")
    require(request.get("kind") == "ace3_transaction009_layer8_binding_successor_review_request" and request.get("review_phase") == "PRE_AUTHORITY_PACKAGE_ACCEPTANCE" and request.get("launch_authority_required") is False and request.get("execution_binding_required") is False, "review request does not preserve PASS-before-authority ordering")
    require(not BINDING.exists() and not AUTHORITY.exists() and not AUTHORITY_SEAL.exists(), "authority or execution binding exists before package review")
    require(not (FUTURE_ROOT / "manager-authorization-consumption.json").exists() and not FUTURE_ROOT.exists() and not TRANSACTION9.exists() and not GENERATION10.exists() and not GENERATION10_STAGING.exists(), "transaction009 consumption, terminal, output, or generation10 exists")
    for index in range(10, 26):
        require(not (RUNTIME / "transactions" / f"transaction-{{index:03d}}").exists(), f"transaction{{index:03d}} exists")
        require(not (RUNTIME / f"transaction{{index}}-authoritative-generation{{index + 1}}").exists(), f"transaction{{index:03d}} runtime effect exists")
    review = {{
        "schema_version": 1,
        "kind": "ace3_transaction009_layer8_binding_successor_independent_review",
        "status": "PASS",
        "producer_role": "reviewer",
        "review_type": "source-disjoint-static-binding-and-zero-effect-review",
        "mission_id": {MISSION_ID!r},
        "node_key": {NODE_KEY!r},
        "package_manifest": file_record(PACKAGE / "package-manifest.json"),
        "package_seal": file_record(PACKAGE / "package-seal.json"),
        "package_tree": tree_record(PACKAGE),
        "review_request": file_record(PACKAGE / "review-request.json"),
        "canonical_future_launch": manifest["launch"],
        "launch_authority_present": False,
        "execution_binding_present": False,
        "executor_constants_equal_canonical_argv": True,
        "authoritative_generation": 9,
        "authoritative_cursor": 9,
        "target_generation": 10,
        "target_cursor": 10,
        "transaction_index": 9,
        "layer_index": 8,
        "transaction008_consumed_non_reusable": True,
        "transaction009_authority_created": False,
        "transaction009_authority_consumed": False,
        "transaction009_execution_performed": False,
        "transaction009_terminal_present": False,
        "generation10_present": False,
        "transaction010_025_authority_or_effects": False,
        "activity_counters": dict(ZERO_COUNTERS),
        "total_workload_calls": 0,
        "claim_boundary": "Independent metadata and static AST binding review only; no authority consumption, payload, model, oracle, RTL, vector, transaction, terminal, or publication execution was performed.",
    }}
    write_new(arguments.output, canonical_json(review))
    print("TX009_BINDING_SUCCESSOR_REVIEW_PASS authority_consumed=0 payload=0 generation10=0 transaction010_025=0 artifact=" + str(arguments.output))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
'''
    return source.encode("utf-8")


def materialize_package(
    parent: Mapping[str, Any],
    zero_effects: Mapping[str, Any],
) -> None:
    make_directory(PACKAGE)
    spec = binding_specification(parent, zero_effects)
    write_new(PACKAGE / "binding-spec.json", canonical_json(spec))
    executor_path = PACKAGE / "transaction9_executor.py"
    write_new(executor_path, successor_executor())
    emitter = review_emitter_source()
    write_new(PACKAGE / "review-emitter.py", emitter)

    baseline = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction9_layer8_authoritative_baseline",
        "status": "AUTHENTICATED_PARENT_NO_EXECUTION",
        "successor_mission_id": MISSION_ID,
        "successor_node_key": NODE_KEY,
        "runtime_identity": RUNTIME.name,
        "authoritative_generation": 9,
        "authoritative_cursor": 9,
        "required_parent_checkpoint_index": 8,
        "transaction_index": 9,
        "layer_index": 8,
        "authoritative_parent": copy.deepcopy(dict(parent)),
        "transaction008_consumed_non_reusable": True,
        "transaction009_absent": True,
        "generation10_absent": True,
        "transactions010_025_absent": True,
        "zero_effects": copy.deepcopy(dict(zero_effects)),
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        "predecessor_package_manifest": file_record(
            PREDECESSOR_PACKAGE / "package-manifest.json"
        ),
        "binding_repair": (
            "sealed executor parent, descriptor, simulation argv, receipt "
            "contract, counter schema, and launch binding are coherent"
        ),
    }
    write_new(
        PACKAGE / "authoritative-baseline.json", canonical_json(baseline)
    )

    sources = copy.deepcopy(
        load_json(PREDECESSOR_PACKAGE / "source-manifest.json")
    )
    sources["sources"]["predecessor_transaction009_executor"] = file_record(
        PREDECESSOR_PACKAGE / "transaction9_executor.py"
    )
    sources["sources"]["transaction009_executor"] = file_record(executor_path)
    sources["sources"]["binding_successor_materializer"] = file_record(
        Path(__file__).resolve()
    )
    write_new(PACKAGE / "source-manifest.json", canonical_json(sources))

    manifest = copy.deepcopy(
        load_json(PREDECESSOR_PACKAGE / "package-manifest.json")
    )
    manifest.update(
        {
            "package_id": (
                "generation9-cursor9-checkpoint008-"
                f"transaction009-layer08-binding-{REVISION}"
            ),
            "mission_id": MISSION_ID,
            "node_key": NODE_KEY,
            "successor_mission_id": MISSION_ID,
            "successor_node_key": NODE_KEY,
            "execution_mission_id": spec["execution_mission_id"],
            "parent_mission_id": spec["parent_mission_id"],
            "manager_directive": (
                "MANAGER DIRECTIVE TRANSACTION009 LAYER8 EXECUTE-ONCE V3"
            ),
            "runtime_identity": spec["runtime_identity"],
            "authoritative_generation": spec["authoritative_generation"],
            "authoritative_cursor": spec["authoritative_cursor"],
            "required_parent_checkpoint_index": (
                spec["required_parent_checkpoint_index"]
            ),
            "transaction_index": spec["transaction_index"],
            "layer_index": spec["layer_index"],
            "transaction_position": spec["transaction_position"],
            "target_generation": spec["target_generation"],
            "target_cursor": spec["target_cursor"],
            "target_checkpoint_index": spec["target_checkpoint_index"],
            "transaction_descriptor": copy.deepcopy(
                spec["transaction_descriptor"]
            ),
            "authoritative_generation9": copy.deepcopy(dict(parent)),
            "authoritative_baseline": file_record(
                PACKAGE / "authoritative-baseline.json"
            ),
            "binding_spec": file_record(PACKAGE / "binding-spec.json"),
            "source_manifest": file_record(
                PACKAGE / "source-manifest.json"
            ),
            "compile": copy.deepcopy(spec["compile"]),
            "simulation": copy.deepcopy(spec["simulation"]),
            "launch": copy.deepcopy(spec["launch"]),
            "review_output": str(REVIEW),
            "future_manager_authority": str(AUTHORITY),
            "pre_authority_binding_check": str(BINDING),
            "transaction009_output_namespace": str(TRANSACTION9),
            "generation10_output_namespace": str(GENERATION10),
            "generation10_completed_receipts": copy.deepcopy(
                spec["generation10_completed_receipts"]
            ),
            "transaction009_absence_predicates": copy.deepcopy(
                spec["absence_predicates"]
            ),
            "prohibitions": [
                "transaction000-008 retry, replay, or resume",
                "transaction010-025 execution or authority",
                "model, oracle, or vector execution during package review",
                "RTL compile or simulation during package review",
                "generation10 publication during package review",
                "restart or adoption of partial transaction009 state",
                "execution before separate Manager authority and Reviewer PASS",
                "authority consumption or publication by this bounded task",
            ],
            "predecessor_package": {
                "package_manifest": file_record(
                    PREDECESSOR_PACKAGE / "package-manifest.json"
                ),
                "executor": file_record(
                    PREDECESSOR_PACKAGE / "transaction9_executor.py"
                ),
                "binding_status": (
                    "REJECTED_READ_ONLY_PACKAGE_GATES"
                ),
            },
            "execution_authorized": False,
            "authority_created": False,
            "activity_counters": copy.deepcopy(
                spec["zero_activity_counters"]
            ),
        }
    )
    write_new(PACKAGE / "package-manifest.json", canonical_json(manifest))

    review_request = {
        "schema_version": 1,
        "kind": "ace3_transaction009_layer8_binding_successor_review_request",
        "review_phase": "PRE_AUTHORITY_PACKAGE_ACCEPTANCE",
        "launch_authority_required": False,
        "execution_binding_required": False,
        "mission_id": MISSION_ID,
        "node_key": NODE_KEY,
        "required_role": "reviewer",
        "review_output": str(REVIEW),
        "reviewer_emitter": file_record(PACKAGE / "review-emitter.py"),
        "review_invocation": [
            "/usr/bin/python3",
            str(REVIEW_EMITTER),
            "--output",
            str(REVIEW),
        ],
        "pre_authority_binding_check": str(BINDING),
        "requested_judgment": "PASS_OR_REJECT",
        "this_review_consumes_authority": False,
        "this_review_executes_payload": False,
    }
    write_new(
        PACKAGE / "review-request.json", canonical_json(review_request)
    )
    members = {
        label: file_record(PACKAGE / relative)
        for label, relative in PACKAGE_MEMBERS.items()
    }
    seal = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction9_layer8_executor_seal",
        "status": "SEALED_REVIEW_REQUIRED",
        "successor_mission_id": MISSION_ID,
        "successor_node_key": NODE_KEY,
        "members": members,
        "execution_authorized": False,
        "authority_created": False,
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
    }
    write_new(PACKAGE / "package-seal.json", canonical_json(seal))
    validate_binding_documents(parent, zero_effects, require_sealed=False)
    seal_directory(PACKAGE)


def authenticate_record_tree(value: object, label: str) -> int:
    if isinstance(value, dict) and set(value) == {
        "path",
        "bytes",
        "sha256",
    }:
        authenticate(value, Path(value["path"]), label)
        return 1
    require(isinstance(value, dict), f"binding input record differs: {label}")
    return sum(
        authenticate_record_tree(child, f"{label}.{name}")
        for name, child in sorted(value.items())
    )


def validate_binding_documents(
    parent: Mapping[str, Any],
    zero_effects: Mapping[str, Any],
    *,
    require_sealed: bool,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    spec = load_json(PACKAGE / "binding-spec.json")
    manifest = load_json(PACKAGE / "package-manifest.json")
    expected_spec = binding_specification(parent, zero_effects)
    require(spec == expected_spec, "sealed binding specification differs")
    require(
        manifest.get("mission_id") == spec["binding_mission_id"]
        and manifest.get("node_key") == spec["binding_node_key"]
        and manifest.get("successor_mission_id")
        == spec["binding_mission_id"]
        and manifest.get("successor_node_key")
        == spec["binding_node_key"]
        and manifest.get("execution_mission_id")
        == spec["execution_mission_id"]
        and manifest.get("parent_mission_id")
        == spec["parent_mission_id"]
        and manifest.get("package_id") == spec["package_id"]
        and manifest.get("runtime_identity") == spec["runtime_identity"]
        and manifest.get("authoritative_generation")
        == spec["authoritative_generation"]
        and manifest.get("authoritative_cursor")
        == spec["authoritative_cursor"]
        and manifest.get("required_parent_checkpoint_index")
        == spec["required_parent_checkpoint_index"]
        and manifest.get("transaction_index")
        == spec["transaction_index"]
        and manifest.get("layer_index") == spec["layer_index"]
        and manifest.get("transaction_position")
        == spec["transaction_position"]
        and manifest.get("target_generation")
        == spec["target_generation"]
        and manifest.get("target_cursor") == spec["target_cursor"]
        and manifest.get("target_checkpoint_index")
        == spec["target_checkpoint_index"]
        and manifest.get("transaction_descriptor")
        == spec["transaction_descriptor"]
        and manifest.get("compile") == spec["compile"]
        and manifest.get("simulation") == spec["simulation"]
        and manifest.get("launch") == spec["launch"]
        and manifest.get("generation10_completed_receipts")
        == spec["generation10_completed_receipts"]
        and manifest.get("activity_counters")
        == spec["zero_activity_counters"]
        and manifest.get("transaction009_absence_predicates")
        == spec["absence_predicates"]
        and manifest.get("transaction009_output_namespace")
        == str(TRANSACTION9)
        and manifest.get("generation10_output_namespace")
        == str(GENERATION10)
        and manifest.get("binding_spec")
        == file_record(PACKAGE / "binding-spec.json"),
        "manifest does not derive from canonical binding specification",
    )
    input_count = authenticate_record_tree(
        spec["consumed_binding_inputs"], "consumed_binding_inputs"
    )
    require(input_count == 19, "consumed binding input count differs")
    executor_source = (
        PACKAGE / "transaction9_executor.py"
    ).read_text(encoding="utf-8")
    require(
        "TRANSACTION8" not in executor_source
        and 'BINDING_SPEC = PACKAGE_ROOT / "binding-spec.json"'
        in executor_source,
        "executor transaction namespace is not canonical",
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    probe = """
import importlib.util
import sys
from pathlib import Path
path = Path(sys.argv[1])
module_spec = importlib.util.spec_from_file_location(
    "ace3_tx009_r10_binding_probe", path
)
if module_spec is None or module_spec.loader is None:
    raise RuntimeError("executor import specification unavailable")
module = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = module
module_spec.loader.exec_module(module)
manifest = module.load_json(module.OUTPUT_PACKAGE_MANIFEST)
module.validate_manifest_document(manifest)
receipt = {"kind": "read_only_structural_probe", "transaction_index": 9}
contract = module.validate_generation10_publication_structure()
payloads = module.generation10_checkpoint_payloads(receipt)
if list(payloads) != contract["checkpoint_paths"]:
    raise RuntimeError("generation10 checkpoint payload paths differ")
for index in range(module.START_CURSOR):
    relative = f"checkpoints/transaction-{index:03d}.json"
    if payloads[relative] != (module.GENERATION9 / relative).read_bytes():
        raise RuntimeError(f"generation9 checkpoint{index:03d} copy differs")
if payloads[contract["checkpoint_paths"][-1]] != module.canonical_json(receipt):
    raise RuntimeError("checkpoint009 payload differs")
print("TX009_R9_EXECUTOR_MANIFEST_AND_PUBLICATION_STRUCTURE_PASS")
"""
    completed = subprocess.run(
        [
            str(PYTHON),
            "-c",
            probe,
            str(PACKAGE / "transaction9_executor.py"),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        completed.returncode == 0
        and completed.stdout.strip()
        == "TX009_R9_EXECUTOR_MANIFEST_AND_PUBLICATION_STRUCTURE_PASS",
        "executor canonical-spec gate failed: "
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}",
    )
    if require_sealed:
        require(
            stat.S_IMODE(PACKAGE.stat().st_mode) & 0o222 == 0,
            "successor package is writable",
        )
    return spec, manifest, input_count


def validate_package() -> tuple[dict[str, Any], dict[str, object]]:
    parent = validate_parent()
    zero_effects = validate_zero_effects()
    validate_binding_documents(
        parent, zero_effects, require_sealed=True
    )
    manifest = load_json(PACKAGE / "package-manifest.json")
    seal = load_json(PACKAGE / "package-seal.json")
    require(
        stat.S_IMODE(PACKAGE.stat().st_mode) & 0o222 == 0,
        "successor package is writable",
    )
    require(
        manifest.get("kind")
        == "ace3_position3_transaction9_layer8_executor_package"
        and manifest.get("successor_mission_id") == MISSION_ID
        and manifest.get("successor_node_key") == NODE_KEY
        and manifest.get("authoritative_generation") == 9
        and manifest.get("authoritative_cursor") == 9
        and manifest.get("required_parent_checkpoint_index") == 8
        and manifest.get("transaction_index") == 9
        and manifest.get("layer_index") == 8
        and manifest.get("activity_counters") == ZERO_COUNTERS,
        "successor package identity differs",
    )
    for label, relative in PACKAGE_MEMBERS.items():
        authenticate(
            seal["members"][label],
            PACKAGE / relative,
            f"sealed package member {label}",
        )
    constants = executor_constants(PACKAGE / "transaction9_executor.py")
    require_binding(constants, manifest["launch"])
    return manifest, constants


def materialize_read_only_inspection(
    parent: Mapping[str, Any],
    zero_effects: Mapping[str, Any],
) -> None:
    before = copy.deepcopy(dict(zero_effects))
    spec, manifest, input_count = validate_binding_documents(
        parent, before, require_sealed=True
    )
    after = validate_zero_effects()
    require(before == after, "read-only inspection changed runtime frontier")
    document = {
        "schema_version": 1,
        "kind": "ace3_transaction009_layer8_canonical_binding_inspection",
        "status": "PASS",
        "mission_id": MISSION_ID,
        "node_key": NODE_KEY,
        "binding_spec": file_record(PACKAGE / "binding-spec.json"),
        "executor": file_record(PACKAGE / "transaction9_executor.py"),
        "package_manifest": file_record(
            PACKAGE / "package-manifest.json"
        ),
        "package_seal": file_record(PACKAGE / "package-seal.json"),
        "package_tree": tree_record(PACKAGE),
        "consumed_binding_input_count": input_count,
        "consumed_binding_inputs": copy.deepcopy(
            spec["consumed_binding_inputs"]
        ),
        "generation10_publication_payload_structure": {
            "source_generation": 9,
            "copied_checkpoint_records": copy.deepcopy(
                spec["consumed_binding_inputs"][
                    "generation9_completed_receipts"
                ]
            ),
            "new_transaction_index": spec[
                "generation10_completed_receipts"
            ]["new_transaction_index"],
            "exact_checkpoint_paths": copy.deepcopy(
                spec["generation10_completed_receipts"][
                    "checkpoint_paths"
                ]
            ),
            "checkpoint_count": spec[
                "generation10_completed_receipts"
            ]["count"],
        },
        "predicate_results": {
            "parent_mission_identity_consistent": True,
            "transaction_descriptor_consistent": True,
            "layer_index_consistent": True,
            "simulation_argv_consistent": True,
            "generation10_receipt_contract_consistent": True,
            "generation10_publication_payload_structure_passed": True,
            "activity_counters_consistent": True,
            "transaction009_absence_predicates_consistent": True,
            "executor_manifest_spec_gate_passed": True,
            "all_consumed_binding_inputs_authenticated": True,
        },
        "canonical_launch": copy.deepcopy(manifest["launch"]),
        "before_zero_effects": before,
        "after_zero_effects": after,
        "independent_review_created": False,
        "launch_authority_created": False,
        "authority_consumption_performed": False,
        "payload_execution_performed": False,
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        "claim_boundary": (
            "Read-only package inspection only. No independent review, "
            "launch authority, authority consumption, compile, simulation, "
            "payload, terminal, generation10, or transaction010-025 effect."
        ),
    }
    make_directory(GATE_ROOT)
    write_new(GATE_CHECK, canonical_json(document))
    seal_directory(GATE_ROOT)


def materialize_review_source() -> None:
    make_directory(REVIEW_ROOT)
    write_new(REVIEW_EMITTER, (PACKAGE / "review-emitter.py").read_bytes())
    REVIEW_EMITTER.chmod(0o500)
    fsync_directory(REVIEW_ROOT)


def materialize_binding_check(
    parent: Mapping[str, Any],
    zero_effects: Mapping[str, Any],
) -> None:
    manifest, constants = validate_package()
    require(not AUTHORITY.exists(), "successor authority exists before check")
    require(not REVIEW.exists(), "successor review exists before check")
    predecessor_constants = executor_constants(
        PREDECESSOR_PACKAGE / "transaction9_executor.py"
    )
    predecessor_launch = load_json(
        PREDECESSOR_PACKAGE / "package-manifest.json"
    )["launch"]
    predecessor_matches = True
    try:
        expected_predecessor = {
            "ROOT": ROOT,
            "RUNTIME": RUNTIME,
            "PACKAGE_ROOT": PREDECESSOR_PACKAGE,
            "REVIEW": PREDECESSOR_REVIEW,
            "AUTHORITY": PREDECESSOR_AUTHORITY,
            "EXECUTOR": PREDECESSOR_PACKAGE / "transaction9_executor.py",
            "TRANSACTION_INDEX": TRANSACTION_INDEX,
        }
        predecessor_matches = (
            predecessor_constants == expected_predecessor
            and predecessor_launch["argv"][1]
            == str(predecessor_constants["EXECUTOR"])
            and predecessor_launch["argv"][4]
            == str(predecessor_constants["PACKAGE_ROOT"])
            and predecessor_launch["argv"][6]
            == str(predecessor_constants["REVIEW"])
            and predecessor_launch["argv"][8]
            == str(predecessor_constants["AUTHORITY"])
        )
    except (KeyError, TypeError):
        predecessor_matches = False
    require(predecessor_matches, "r3 predecessor static binding differs")
    document = {
        "schema_version": 1,
        "kind": "ace3_transaction009_layer8_pre_authority_binding_check",
        "status": "PASS",
        "mission_id": MISSION_ID,
        "node_key": NODE_KEY,
        "runtime_identity": manifest["runtime_identity"],
        "authoritative_generation": 9,
        "authoritative_cursor": 9,
        "required_parent_checkpoint_index": 8,
        "transaction_index": 9,
        "layer_index": 8,
        "package_manifest": file_record(
            PACKAGE / "package-manifest.json"
        ),
        "package_seal": file_record(PACKAGE / "package-seal.json"),
        "package_tree": tree_record(PACKAGE),
        "executor": file_record(PACKAGE / "transaction9_executor.py"),
        "executor_constants": serialized_constants(constants),
        "canonical_launch": manifest["launch"],
        "executor_constants_equal_canonical_argv": True,
        "predecessor_package_manifest": file_record(
            PREDECESSOR_PACKAGE / "package-manifest.json"
        ),
        "predecessor_executor": file_record(
            PREDECESSOR_PACKAGE / "transaction9_executor.py"
        ),
        "predecessor_executor_constants_equal_canonical_argv": True,
        "predecessor_executor_gate_status": "REJECTED_PACKAGE_PREDICATES",
        "authoritative_parent": copy.deepcopy(dict(parent)),
        "transaction008_consumed_non_reusable": True,
        "authority_absent_at_check": True,
        "independent_review_absent_at_check": True,
        "authority_consumption_performed": False,
        "payload_execution_performed": False,
        "terminal_created": False,
        "zero_effects": copy.deepcopy(dict(zero_effects)),
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        "claim_boundary": (
            "Static sealed-executor binding and namespace check before "
            "authority issuance only; no consumption or payload execution."
        ),
    }
    make_directory(BINDING_ROOT)
    write_new(BINDING, canonical_json(document))
    seal_directory(BINDING_ROOT)


def materialize_authority(
    parent: Mapping[str, Any],
) -> None:
    manifest, constants = validate_package()
    require_binding(constants, manifest["launch"])
    binding = load_json(BINDING)
    require(
        binding.get("status") == "PASS"
        and binding.get("authority_absent_at_check") is True
        and binding.get("executor_constants_equal_canonical_argv") is True,
        "pre-authority binding check is not PASS",
    )
    validate_zero_effects()
    predecessor = load_json(PREDECESSOR_AUTHORITY)
    authority = copy.deepcopy(predecessor)
    authority.update(
        {
            "status": "RUNTIME_PASS_ACCEPTED",
            "mission_id": MISSION_ID,
            "node_key": NODE_KEY,
            "producer_role": "manager",
            "package_manifest": file_record(
                PACKAGE / "package-manifest.json"
            ),
            "package_seal": file_record(PACKAGE / "package-seal.json"),
            "source_manifest": file_record(
                PACKAGE / "source-manifest.json"
            ),
            "authoritative_baseline": file_record(
                PACKAGE / "authoritative-baseline.json"
            ),
            "aggregate_preflight": file_record(BINDING),
            "pre_authority_binding_check": file_record(BINDING),
            "authorized_launch": manifest["launch"],
            "authoritative_generation9": copy.deepcopy(dict(parent)),
            "output_namespaces": {
                "authority": str(AUTHORITY),
                "authority_consumption": str(CONSUMPTION),
                "checkpoint009": str(
                    GENERATION10 / "checkpoints/transaction-009.json"
                ),
                "generation10": str(GENERATION10),
                "transaction009": str(TRANSACTION9),
            },
            "reviewer_acceptance": {
                "required": True,
                "path": str(REVIEW),
                "present_at_issuance": False,
            },
            "supersedes": {
                "authority": file_record(PREDECESSOR_AUTHORITY),
                "reason": (
                    "r5 static argv binding passed but its sealed executor "
                    "retained obsolete parent, simulation, receipt, and "
                    "counter predicates"
                ),
                "predecessor_consumed": False,
                "predecessor_launchable": False,
            },
            "activity_counters": copy.deepcopy(ZERO_COUNTERS),
            "authority_consumed": False,
            "transaction009_executed": False,
            "generation10_exists": False,
            "transactions010_025_absent": True,
            "claim_boundary": (
                "Unconsumed transaction009/layer8 launch authority bound "
                "to a pre-authority static executor/argv PASS. No payload, "
                "terminal, generation10, or transaction010-025 effect."
            ),
        }
    )
    make_directory(AUTHORITY_ROOT)
    write_new(AUTHORITY, canonical_json(authority))
    request = {
        "schema_version": 1,
        "kind": "ace3_transaction009_layer8_authority_review_request",
        "mission_id": MISSION_ID,
        "node_key": NODE_KEY,
        "authority": file_record(AUTHORITY),
        "package_manifest": file_record(
            PACKAGE / "package-manifest.json"
        ),
        "pre_authority_binding_check": file_record(BINDING),
        "reviewer_emitter": file_record(REVIEW_EMITTER),
        "review_output": str(REVIEW),
        "review_invocation": [
            "/usr/bin/python3",
            str(REVIEW_EMITTER),
            "--output",
            str(REVIEW),
        ],
        "this_review_consumes_authority": False,
        "this_review_executes_payload": False,
    }
    write_new(
        AUTHORITY_ROOT / "review-request.json", canonical_json(request)
    )
    seal = {
        "schema_version": 1,
        "kind": "ace3_position3_transaction9_layer8_authority_seal",
        "status": "RUNTIME_PASS_ACCEPTED",
        "mission_id": MISSION_ID,
        "node_key": NODE_KEY,
        "producer_role": "manager",
        "members": {
            "authority": file_record(AUTHORITY),
            "review_request": file_record(
                AUTHORITY_ROOT / "review-request.json"
            ),
        },
        "pre_authority_binding_check": file_record(BINDING),
        "authority_consumed": False,
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
    }
    write_new(AUTHORITY_SEAL, canonical_json(seal))
    seal_directory(AUTHORITY_ROOT)


def run_independent_review() -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            "/usr/bin/python3",
            str(REVIEW_EMITTER),
            "--output",
            str(REVIEW),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        completed.returncode == 0,
        "independent review failed: "
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}",
    )
    require(REVIEW.is_file(), "independent review did not create its output")
    seal_directory(REVIEW_ROOT)


def exercise_executor_gates() -> None:
    before = validate_zero_effects()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    probe = """
import importlib.util
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("ace3_tx009_r7_gate_probe", path)
if spec is None or spec.loader is None:
    raise RuntimeError("executor import specification unavailable")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
package = module.validate_package(require_zero_runtime=False)
review = module.validate_review()
authority = module.validate_authority()
print(json.dumps({
    "package_kind": package["kind"],
    "package_status": package["status"],
    "review_kind": review["kind"],
    "review_status": review["status"],
    "authority_kind": authority["kind"],
    "authority_status": authority["status"],
}, sort_keys=True))
"""
    completed = subprocess.run(
        [
            str(PYTHON),
            "-c",
            probe,
            str(PACKAGE / "transaction9_executor.py"),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        completed.returncode == 0,
        "sealed executor read-only gates failed: "
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}",
    )
    observed = json.loads(completed.stdout)
    require(
        observed
        == {
            "authority_kind": (
                "ace3_position3_transaction9_layer8_launch_authority"
            ),
            "authority_status": "RUNTIME_PASS_ACCEPTED",
            "package_kind": (
                "ace3_position3_transaction9_layer8_executor_package"
            ),
            "package_status": "SEALED_REVIEW_REQUIRED",
            "review_kind": (
                "ace3_transaction009_layer8_binding_successor_"
                "independent_review"
            ),
            "review_status": "PASS",
        },
        "sealed executor gate probe output differs",
    )
    after = validate_zero_effects()
    require(before == after, "read-only gate probe changed zero-effect state")
    document = {
        "schema_version": 1,
        "kind": "ace3_transaction009_layer8_read_only_gate_check",
        "status": "PASS",
        "mission_id": MISSION_ID,
        "node_key": NODE_KEY,
        "executor": file_record(PACKAGE / "transaction9_executor.py"),
        "package_manifest": file_record(
            PACKAGE / "package-manifest.json"
        ),
        "package_seal": file_record(PACKAGE / "package-seal.json"),
        "independent_review": file_record(REVIEW),
        "launch_authority": file_record(AUTHORITY),
        "authority_seal": file_record(AUTHORITY_SEAL),
        "canonical_launch": canonical_launch(),
        "validate_package_require_zero_runtime_false_passed": True,
        "validate_review_passed": True,
        "validate_authority_passed": True,
        "probe_interpreter": str(PYTHON),
        "probe_import_only": True,
        "execute_once_called": False,
        "authority_consumption_performed": False,
        "payload_execution_performed": False,
        "before_zero_effects": before,
        "after_zero_effects": after,
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        "claim_boundary": (
            "Read-only import and direct invocation of the sealed executor "
            "validate_review and validate_authority gates only. The execute "
            "entrypoint, authority consumption, payload, terminal, and "
            "generation publication were not invoked."
        ),
    }
    make_directory(GATE_ROOT)
    write_new(GATE_CHECK, canonical_json(document))
    seal_directory(GATE_ROOT)


def materialize_readiness(
    parent: Mapping[str, Any],
    zero_effects: Mapping[str, Any],
) -> None:
    manifest, constants = validate_package()
    review = load_json(REVIEW)
    authority = load_json(AUTHORITY)
    require(
        review.get("status") == "PASS"
        and review.get("executor_constants_equal_canonical_argv") is True
        and authority.get("status") == "RUNTIME_PASS_ACCEPTED"
        and authority.get("authority_consumed") is False,
        "review or authority is not ready",
    )
    document = {
        "schema_version": 1,
        "kind": "ace3_transaction009_layer8_binding_successor_readiness",
        "status": "RUNTIME_PASS_ACCEPTED_UNCONSUMED",
        "mission_id": MISSION_ID,
        "node_key": NODE_KEY,
        "runtime_identity": manifest["runtime_identity"],
        "authoritative_generation": 9,
        "authoritative_cursor": 9,
        "target_generation": 10,
        "target_cursor": 10,
        "transaction_index": 9,
        "layer_index": 8,
        "package_manifest": file_record(
            PACKAGE / "package-manifest.json"
        ),
        "package_seal": file_record(PACKAGE / "package-seal.json"),
        "package_tree": tree_record(PACKAGE),
        "pre_authority_binding_check": file_record(BINDING),
        "independent_review": file_record(REVIEW),
        "review_emitter": file_record(REVIEW_EMITTER),
        "launch_authority": file_record(AUTHORITY),
        "authority_seal": file_record(AUTHORITY_SEAL),
        "read_only_gate_check": file_record(GATE_CHECK),
        "canonical_launch": manifest["launch"],
        "executor_constants": serialized_constants(constants),
        "executor_constants_equal_canonical_argv": True,
        "authoritative_parent": copy.deepcopy(dict(parent)),
        "transaction008_consumed_non_reusable": True,
        "authority_consumed": False,
        "payload_execution_performed": False,
        "terminal_created": False,
        "zero_effects": copy.deepcopy(dict(zero_effects)),
        "activity_counters": copy.deepcopy(ZERO_COUNTERS),
        "claim_boundary": (
            "Fresh successor package, static pre-authority binding PASS, "
            "source-disjoint review, and unconsumed launch authority only. "
            "No transaction009 payload or terminal, generation10/cursor10, "
            "or transaction010-025 effect exists."
        ),
    }
    make_directory(READINESS_ROOT)
    write_new(READINESS, canonical_json(document))
    seal_directory(READINESS_ROOT)


def prepare() -> None:
    for path in (
        PACKAGE,
        BINDING_ROOT,
        REVIEW_ROOT,
        AUTHORITY_ROOT,
        GATE_ROOT,
        READINESS_ROOT,
    ):
        require(not path.exists(), f"successor output already exists: {path}")
    parent = validate_parent()
    zero_effects = validate_zero_effects()
    materialize_package(parent, zero_effects)
    materialize_read_only_inspection(parent, zero_effects)


def validate() -> dict[str, Any]:
    parent = validate_parent()
    zero_effects = validate_zero_effects()
    manifest, constants = validate_package()
    spec = load_json(PACKAGE / "binding-spec.json")
    spec_receipts = spec["consumed_binding_inputs"][
        "generation9_completed_receipts"
    ]
    gate_check = load_json(GATE_CHECK)
    require_binding(constants, manifest["launch"])
    require(
        gate_check.get("kind")
        == "ace3_transaction009_layer8_canonical_binding_inspection"
        and gate_check.get("status") == "PASS"
        and gate_check.get("binding_spec")
        == file_record(PACKAGE / "binding-spec.json")
        and gate_check.get("executor")
        == file_record(PACKAGE / "transaction9_executor.py")
        and gate_check.get("package_manifest")
        == file_record(PACKAGE / "package-manifest.json")
        and gate_check.get("canonical_launch") == manifest["launch"]
        and gate_check.get("predicate_results")
        == {
            "parent_mission_identity_consistent": True,
            "transaction_descriptor_consistent": True,
            "layer_index_consistent": True,
            "simulation_argv_consistent": True,
            "generation10_receipt_contract_consistent": True,
            "generation10_publication_payload_structure_passed": True,
            "activity_counters_consistent": True,
            "transaction009_absence_predicates_consistent": True,
            "executor_manifest_spec_gate_passed": True,
            "all_consumed_binding_inputs_authenticated": True,
        }
        and gate_check.get("consumed_binding_input_count") == 19
        and gate_check.get("generation10_publication_payload_structure")
        == {
            "source_generation": 9,
            "copied_checkpoint_records": spec_receipts,
            "new_transaction_index": 9,
            "exact_checkpoint_paths": (
                expected_generation10_receipts()["checkpoint_paths"]
            ),
            "checkpoint_count": 10,
        }
        and gate_check.get("before_zero_effects") == zero_effects
        and gate_check.get("after_zero_effects") == zero_effects
        and gate_check.get("independent_review_created") is False
        and gate_check.get("launch_authority_created") is False
        and gate_check.get("authority_consumption_performed") is False
        and gate_check.get("payload_execution_performed") is False
        and gate_check.get("activity_counters") == ZERO_COUNTERS,
        "canonical binding inspection differs",
    )
    require(
        not BINDING_ROOT.exists()
        and not REVIEW_ROOT.exists()
        and not AUTHORITY_ROOT.exists()
        and not READINESS_ROOT.exists(),
        "package-only task created review or authority metadata",
    )
    validate_binding_documents(
        parent, zero_effects, require_sealed=True
    )
    return gate_check


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "validate"))
    arguments = parser.parse_args()
    if arguments.operation == "prepare":
        prepare()
        print(
            "TX009_BINDING_SUCCESSOR_PREPARED "
            f"package={PACKAGE} inspection={GATE_CHECK} "
            "consumption=0 payload=0 terminal=0 generation10=0 "
            "transaction010_025=0"
        )
    else:
        inspection = validate()
        print(
            "TX009_BINDING_SUCCESSOR_VALID "
            f"package_sha256={inspection['package_manifest']['sha256']} "
            f"binding_spec_sha256={inspection['binding_spec']['sha256']} "
            f"executor_sha256={inspection['executor']['sha256']} "
            "consumption=0 payload=0 terminal=0 generation10=0 "
            "transaction010_025=0"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        SuccessorError,
        OSError,
        UnicodeDecodeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise SystemExit(
            f"TX009_BINDING_SUCCESSOR_REFUSED {error}"
        ) from error
