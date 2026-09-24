from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


GENERATION8_POINTER_SHA256 = (
    "e2aa2a2230e0855cd0cea0a84bc7c619d028a1984868fae137c58f29144f2578"
)
SOURCE_READINESS_RELATIVE = Path(
    "build/cursor9_next_position_runtime_readiness/readiness.json"
)


def _canonical_pointer(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
        "ascii"
    )


def _canonical_readiness(document: object) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _file_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_cursor8_readiness(root: Path, output_dir: Path) -> Path:
    source_path = root / SOURCE_READINESS_RELATIVE
    if not source_path.is_file():
        raise FileNotFoundError("recorded selected-token readiness is absent")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    runtime_identity = source["runtime_identity"]

    generation9_ledger = Path(source["cursor"]["ledger"]["path"])
    state_generations = generation9_ledger.parents[1]
    generation8 = state_generations / "generation-0000000008"
    manifest_path = generation8 / "generation-manifest.json"
    ledger_path = generation8 / "ledger.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    invocation_record = ledger["invocation"]
    invocation = json.loads(
        Path(invocation_record["path"]).read_text(encoding="utf-8")
    )
    descriptor = invocation["transactions"][8]

    manifest_record = _file_record(manifest_path)
    pointer = {
        "schema_version": 1,
        "kind": (
            "ace3_position3_transaction7_generation8_"
            "authoritative_pointer"
        ),
        "status": "COMMITTED",
        "runtime_identity": runtime_identity,
        "generation": 8,
        "generation_manifest": manifest_record,
        "previous_authoritative_pointer_sha256": manifest["parent_pointer"][
            "sha256"
        ],
        "atomic_visibility_contract": (
            "pointer replacement is the sole visibility point from "
            "generation7/cursor7 to generation8/cursor8"
        ),
    }
    pointer_payload = _canonical_pointer(pointer)
    if hashlib.sha256(pointer_payload).hexdigest() != GENERATION8_POINTER_SHA256:
        raise AssertionError("generation8 authoritative pointer differs")
    pointer_path = output_dir / "generation8-authoritative-pointer.json"
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_bytes(pointer_payload)

    checkpoint000 = manifest["files"]["checkpoints/transaction-000.json"]
    checkpoint001 = manifest["files"]["checkpoints/transaction-001.json"]
    checkpoint007 = manifest["files"]["checkpoints/transaction-007.json"]
    checkpoint007_document = json.loads(
        Path(checkpoint007["path"]).read_text(encoding="utf-8")
    )
    selected_token = copy.deepcopy(source["selected_token"])
    selected_token["lm_head_receipt"] = checkpoint000
    embedding_feedback = copy.deepcopy(source["embedding_feedback"])
    embedding_feedback["layer0_consumed_receipt"] = checkpoint001

    readiness = {
        "schema_version": 1,
        "kind": "ace3_cursor8_next_position_runtime_readiness",
        "status": "NOT_READY",
        "runtime_identity": runtime_identity,
        "cursor": {
            "authoritative_pointer": _file_record(pointer_path),
            "generation_manifest": manifest_record,
            "ledger": _file_record(ledger_path),
            "checkpoint007": checkpoint007,
            "generation": 8,
            "parent_generation": 7,
            "checkpoint": 7,
            "next_transaction_index": 8,
        },
        "next_layer_inputs": {
            "transaction_index": 8,
            "layer_index": 7,
            "transaction_position": descriptor["inputs"][
                "transaction_position"
            ],
            "required_result": descriptor["required_result"],
            "template_input_binding_sha256": descriptor[
                "template_input_binding_sha256"
            ],
            "layer7_fixture_manifest": descriptor["inputs"][
                "fixture_manifest"
            ],
            "layer7_position2_kv_parent": descriptor["inputs"][
                "position2_kv_parent"
            ],
            "checkpoint007_hidden_kv_state": (
                checkpoint007_document["outputs"]["state"]
            ),
        },
        "selected_token": selected_token,
        "embedding_feedback": embedding_feedback,
        "data_bindings_ready": True,
        "launchable": False,
        "missing_required_bindings": [
            "an unconsumed transaction008-specific execution authority",
            (
                "a bounded transaction008 launcher command that cannot "
                "execute transactions009-025"
            ),
        ],
        "execution_performed_by_this_check": False,
        "ace2_dependency": False,
        "claim_boundary": (
            "Authenticated reviewer-sealed generation8/cursor8 parent and "
            "selected-token input inspection only. No authority, traversal, "
            "runtime mutation, consumption, execution, generation "
            "publication, synthesis, PPA, FPGA, latency, or dialogue result "
            "was created or newly claimed."
        ),
    }
    readiness_path = output_dir / "readiness.json"
    readiness_path.write_bytes(_canonical_readiness(readiness))
    return readiness_path
