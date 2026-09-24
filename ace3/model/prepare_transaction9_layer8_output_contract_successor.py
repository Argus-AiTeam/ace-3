#!/usr/bin/env python3
"""Prepare the review-only transaction009/layer8 r14 output-contract repair."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from typing import Any, Mapping

import prepare_transaction9_layer8_binding_successor as base
import prepare_transaction9_layer8_review_first_successor as review_first


REVISION = "r14"
RUNTIME = base.RUNTIME
PACKAGE = RUNTIME / f"transaction9-layer8-continuation-package-{REVISION}"
REVIEW_ROOT = RUNTIME / f"transaction9-layer8-independent-review-{REVISION}"
REVIEW_EMITTER = REVIEW_ROOT / "review-emitter.py"
REVIEW = REVIEW_ROOT / "independent-review.json"
EXECUTION_BINDING_ROOT = (
    RUNTIME / f"transaction9-layer8-execution-binding-{REVISION}"
)
EXECUTION_BINDING = EXECUTION_BINDING_ROOT / "execution-binding.json"
AUTHORITY_ROOT = RUNTIME / f"transaction9-layer8-launch-authority-{REVISION}"
AUTHORITY = AUTHORITY_ROOT / "launch-authority.json"
AUTHORITY_SEAL = AUTHORITY_ROOT / "authority-seal.json"
GATE_ROOT = RUNTIME / f"transaction9-layer8-read-only-gates-{REVISION}"
GATE_CHECK = GATE_ROOT / "gate-check.json"
AUTHORITY_REVIEW = GATE_ROOT / "authority-readiness-review.json"
READINESS_ROOT = RUNTIME / f"transaction9-layer8-successor-readiness-{REVISION}"
READINESS = READINESS_ROOT / "readiness.json"

EXPECTED_REVIEW_SHA256 = {
    PACKAGE / "package-manifest.json": (
        "cc5c43c7947bcb1a6ba7f8fcceba5e7febf4d47c0643b69a4a9a620d3dbaf5a4"
    ),
    PACKAGE / "package-seal.json": (
        "fbc7f0ffc120ce3f500aac061c0306363681e41bc21c4df5d0c29e819d5e2b6c"
    ),
    PACKAGE / "review-request.json": (
        "ae02c3d29baf5d72f6c6f79beadd4b324a15f3e358689c2c35cc091fca95b1f3"
    ),
    GATE_CHECK: (
        "316c8a9996899d116bb1842c26c3ed1be515de8b305c97aeb8ce205d0b0a4732"
    ),
}

R12_PACKAGE = RUNTIME / "transaction9-layer8-continuation-package-r12"
R12_RUNTIME = RUNTIME / "transaction9-authoritative-generation10"
R12_FAILURE = R12_RUNTIME / "fail-closed-terminal.json"
R12_FIXED_SHA256 = {
    R12_PACKAGE / "binding-spec.json": (
        "a10240af062ca53ffd04f340a14f735d63f5368c86a78cf32555cf5b75ea89c2"
    ),
    R12_PACKAGE / "package-manifest.json": (
        "63eed8861a398508602d5ec1d650755a778b4c97798fd821c4425b2a8d59846f"
    ),
    R12_PACKAGE / "package-seal.json": (
        "552f497e473a7efd9b05818ce775590df74de9970e653d7989236c8974044cc0"
    ),
    R12_PACKAGE / "review-request.json": (
        "2446e0f6f3f379939e0d0728ab97cd794311c502d6c6267d71dce1d78d0efd74"
    ),
    R12_PACKAGE / "transaction9_executor.py": (
        "e0f257dc4741715549db6a8e005a8ee5c5315115b75eaf0f26d5e75127bdf4d1"
    ),
    R12_RUNTIME / "manager-authorization-consumption.json": (
        "28e95467f6d261240ec6c1d6df9e7d780a46fffaef6638bea6885fecc9ea4523"
    ),
    R12_RUNTIME / "project-authority-consumption.json": (
        "9ef7e0f791ad2788d3dae27e872ccec49ef5f80f099a2ec33e5224889ce7e90d"
    ),
    R12_RUNTIME / "execution-start.json": (
        "039a7619a217c199aaac1e1ab78529277d58b135890d877cbe0fc94ee28171d6"
    ),
    R12_FAILURE: (
        "1d0b7c3534bd745cc3cb0e92f0e755bf9d44c4dd84c645a6d5119ccf8dec9734"
    ),
    R12_RUNTIME / "authorized-invocation.stderr.bin": (
        "083ce53887d2028387d3233ed5f39254c928166801f2c80aaaf3b94d849d5819"
    ),
    R12_RUNTIME / "authorized-invocation.exit.txt": (
        "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865"
    ),
}
R12_RUNTIME_MEMBERS = {
    "authorized-invocation.argv.json",
    "authorized-invocation.command.txt",
    "authorized-invocation.exit.txt",
    "authorized-invocation.stderr.bin",
    "authorized-invocation.stdout.bin",
    "execution-start.json",
    "fail-closed-terminal.json",
    "manager-authorization-consumption.json",
    "project-authority-consumption.json",
}

ORIGINAL_WRITE_NEW = base.write_new
ORIGINAL_BINDING_SPECIFICATION = base.binding_specification
ORIGINAL_EXPECTED_RECEIPTS = base.expected_generation10_receipts


def hidden_semantic_sha256(path: Path) -> str:
    rows = path.read_text(encoding="ascii").splitlines()
    base.require(len(rows) == 896, "generation9 parent hidden length differs")
    payload = b"".join(struct.pack("<H", int(row[-4:], 16)) for row in rows)
    return hashlib.sha256(payload).hexdigest()


def generation9_parent_receipt_contract() -> dict[str, Any]:
    checkpoint_path = (
        base.GENERATION9 / "checkpoints/transaction-008.json"
    )
    checkpoint = base.load_json(checkpoint_path)
    hidden = copy.deepcopy(
        checkpoint["rtl_reference_agreement"]["rtl_final"]
    )
    state = copy.deepcopy(checkpoint["result"]["output_state"])
    hidden_path = Path(hidden["path"])
    state_path = Path(state["path"])
    base.authenticate(hidden, hidden_path, "generation9 parent hidden")
    base.authenticate(state, state_path, "generation9 parent state")
    semantic_sha256 = hidden_semantic_sha256(hidden_path)
    comparison = base.load_json(
        Path(checkpoint["rtl_reference_agreement"]["comparison"]["path"])
    )
    base.require(
        comparison.get("exact_integer_oracle_output_sha256")
        == semantic_sha256
        and checkpoint.get("result", {}).get("exact_integer_oracle_match")
        is True
        and checkpoint.get("result", {}).get("natural_rtl_terminal") is True,
        "generation9 parent hidden semantics differ",
    )
    hidden_record = {
        key: hidden[key] for key in ("path", "bytes", "sha256")
    }
    state_record = {
        key: state[key] for key in ("path", "bytes", "sha256")
    }
    return {
        "source_checkpoint": base.file_record(checkpoint_path),
        "required_receipt_keys": ["outputs", "output_semantics"],
        "outputs": {"hidden": hidden_record, "state": state_record},
        "output_semantics": {
            "hidden": {
                "dtype": "FP16",
                "elements": 896,
                "semantic_sha256": semantic_sha256,
            },
            "state": {"layer_index": 7, "position": 4},
        },
    }


def transaction009_completion_receipt_contract() -> dict[str, Any]:
    return {
        "required_top_level_keys": [
            "kind",
            "status",
            "transaction_index",
            "layer_index",
            "result",
            "outputs",
            "output_semantics",
            "timing",
        ],
        "kind": "ace3_position3_transaction_completion",
        "status": "COMPLETE",
        "transaction_index": 9,
        "layer_index": 8,
        "result": {
            "exact_integer_oracle_match": True,
            "natural_rtl_terminal": True,
            "output_hidden_elements": 896,
            "output_state_position": 4,
        },
        "outputs": {
            "required_keys": ["hidden", "state"],
            "hidden": {
                "path": str(
                    base.TRANSACTION9 / "position003/raw/final.hex"
                ),
                "required_record_keys": [
                    "path",
                    "bytes",
                    "sha256",
                    "dtype",
                    "elements",
                    "semantic_sha256",
                ],
                "dtype": "FP16",
                "elements": 896,
            },
            "state": {
                "path": str(base.TRANSACTION9 / "position004.state"),
                "required_record_keys": ["path", "bytes", "sha256"],
            },
        },
        "output_semantics": {
            "required_keys": ["hidden", "state"],
            "hidden": {"dtype": "FP16", "elements": 896},
            "state": {"layer_index": 8, "position": 4},
        },
    }


def expected_generation10_receipts() -> dict[str, Any]:
    document = ORIGINAL_EXPECTED_RECEIPTS()
    document["new_receipt_contract"] = (
        transaction009_completion_receipt_contract()
    )
    return document


def validate_r12_terminal_history() -> dict[str, Any]:
    for path, expected in R12_FIXED_SHA256.items():
        base.require(
            base.sha256_file(path) == expected,
            f"sealed r12 evidence differs: {path}",
        )
    base.require(
        {path.name for path in R12_RUNTIME.iterdir()} == R12_RUNTIME_MEMBERS,
        "sealed r12 terminal namespace differs",
    )
    failure = base.load_json(R12_FAILURE)
    consumption = base.load_json(
        R12_RUNTIME / "project-authority-consumption.json"
    )
    base.require(
        failure
        == {
            "schema_version": 1,
            "kind": (
                "ace3_position3_transaction9_layer8_fail_closed_terminal"
            ),
            "status": "FAIL",
            "error_type": "KeyError",
            "transaction_index": 9,
            "layer_index": 8,
        }
        and consumption.get("status") == "CONSUMED_ONCE"
        and consumption.get("retry_authorized") is False
        and consumption.get("replay_authorized") is False
        and consumption.get("resume_authorized") is False
        and consumption.get("workload_calls_before_consumption") == 0,
        "sealed r12 terminal semantics differ",
    )
    return {
        "package_manifest": base.file_record(
            R12_PACKAGE / "package-manifest.json"
        ),
        "package_seal": base.file_record(R12_PACKAGE / "package-seal.json"),
        "executor": base.file_record(
            R12_PACKAGE / "transaction9_executor.py"
        ),
        "authority_consumption": base.file_record(
            R12_RUNTIME / "manager-authorization-consumption.json"
        ),
        "project_authority_consumption": base.file_record(
            R12_RUNTIME / "project-authority-consumption.json"
        ),
        "execution_start": base.file_record(
            R12_RUNTIME / "execution-start.json"
        ),
        "terminal_failure": base.file_record(R12_FAILURE),
        "failure_type": "KeyError",
        "retry_replay_resume_authorized": False,
    }


def validate_successor_zero_effects() -> dict[str, Any]:
    history = validate_r12_terminal_history()
    base.require(
        base.TRANSACTION9.is_dir()
        and not any(base.TRANSACTION9.iterdir()),
        "r12 transaction009 namespace is not the sealed empty failure state",
    )
    base.require(
        not base.GENERATION10.exists()
        and not base.GENERATION10_STAGING.exists(),
        "generation10 publication exists",
    )
    for index in range(10, 26):
        transaction = base.TRANSACTIONS / f"transaction-{index:03d}"
        runtime_effect = (
            RUNTIME / f"transaction{index}-authoritative-generation{index + 1}"
        )
        generation = (
            base.ADOPTION
            / "state-generations"
            / f"generation-{index + 1:010d}"
        )
        base.require(
            not transaction.exists()
            and not runtime_effect.exists()
            and not generation.exists()
            and not generation.with_name(
                f".{generation.name}.prepared"
            ).exists(),
            f"transaction{index:03d} effect exists",
        )
    base.require(
        not REVIEW_ROOT.exists()
        and not EXECUTION_BINDING_ROOT.exists()
        and not AUTHORITY_ROOT.exists()
        and not READINESS_ROOT.exists(),
        "r14 review, execution binding, authority, or readiness exists",
    )
    return {
        "sealed_r12_terminal_failure": history,
        "r12_transaction009_namespace_empty": True,
        "r12_workload_calls": 0,
        "r12_oracle_execution": 0,
        "r12_payload_execution": 0,
        "r12_rtl_compile": 0,
        "r12_rtl_simulation": 0,
        "generation10_absent": True,
        "generation10_staging_absent": True,
        "transaction010_025_effects_absent": True,
        "successor_review_absent": True,
        "successor_execution_binding_absent": True,
        "successor_authority_absent": True,
        "successor_authority_consumption": 0,
        "successor_payload_execution": 0,
        "successor_terminal_creation": 0,
        "successor_generation10_publication": 0,
    }


def binding_specification(
    parent: Mapping[str, Any],
    zero_effects: Mapping[str, Any],
) -> dict[str, Any]:
    document = review_first.corrected_binding_specification(
        dict(parent), dict(zero_effects)
    )
    document["generation9_parent_receipt_contract"] = (
        generation9_parent_receipt_contract()
    )
    document["transaction009_completion_receipt_contract"] = (
        transaction009_completion_receipt_contract()
    )
    document["generation10_completed_receipts"] = (
        expected_generation10_receipts()
    )
    document["sealed_r12_terminal_failure"] = copy.deepcopy(
        zero_effects["sealed_r12_terminal_failure"]
    )
    document["execution_disposition"] = (
        "NON_EXECUTABLE_REVIEW_ONLY_R12_AUTHORITY_CONSUMED_NO_RETRY"
    )
    document["ordering_contract"] = {
        "phase_one": "IMMUTABLE_REVIEW_ONLY_OUTPUT_CONTRACT_REPAIR",
        "phase_two": "NO_AUTHORITY_WITHIN_THIS_SUCCESSOR",
        "review_required_before_any_later_operator_decision": True,
        "review_accepted_unchanged_by_later_gates": True,
    }
    return document


def corrected_write_new(
    path: Path, payload: bytes, mode: int = 0o400
) -> None:
    if path.parent == PACKAGE and path.suffix == ".json":
        document = json.loads(payload)
        spec = (
            base.load_json(PACKAGE / "binding-spec.json")
            if (PACKAGE / "binding-spec.json").exists()
            else None
        )
        if path.name == "binding-spec.json":
            document["ordering_contract"] = {
                "phase_one": "IMMUTABLE_REVIEW_ONLY_OUTPUT_CONTRACT_REPAIR",
                "phase_two": "NO_AUTHORITY_WITHIN_THIS_SUCCESSOR",
                "review_required_before_any_later_operator_decision": True,
                "review_accepted_unchanged_by_later_gates": True,
            }
        elif path.name == "authoritative-baseline.json":
            document["sealed_r12_terminal_failure"] = (
                validate_r12_terminal_history()
            )
            document["binding_repair"] = (
                "generation9 checkpoint008 is normalized to authenticated "
                "outputs and output_semantics before the exact helper boundary"
            )
        elif path.name == "package-manifest.json":
            assert spec is not None
            document["manager_directive"] = (
                "R12 TERMINAL FAILURE REPAIR; REVIEW ONLY; NO RETRY AUTHORITY"
            )
            document["generation9_parent_receipt_contract"] = copy.deepcopy(
                spec["generation9_parent_receipt_contract"]
            )
            document["transaction009_completion_receipt_contract"] = (
                copy.deepcopy(
                    spec["transaction009_completion_receipt_contract"]
                )
            )
            document["execution_disposition"] = spec["execution_disposition"]
            document["sealed_r12_terminal_failure"] = copy.deepcopy(
                spec["sealed_r12_terminal_failure"]
            )
            document["ordering_contract"] = {
                "phase_one": "IMMUTABLE_REVIEW_ONLY_OUTPUT_CONTRACT_REPAIR",
                "phase_two": "NO_AUTHORITY_WITHIN_THIS_SUCCESSOR",
                "review_required_before_any_later_operator_decision": True,
                "review_accepted_unchanged_by_later_gates": True,
            }
            document["predecessor_package"]["binding_status"] = (
                "SEALED_TERMINAL_FAILURE_OUTPUT_SCHEMA_MISMATCH"
            )
            document["prohibitions"].append(
                "transaction009 retry, replay, resume, or new authority"
            )
        elif path.name == "review-request.json":
            assert spec is not None
            document["generation9_parent_receipt_contract"] = copy.deepcopy(
                spec["generation9_parent_receipt_contract"]
            )
            document["transaction009_completion_receipt_contract"] = (
                copy.deepcopy(
                    spec["transaction009_completion_receipt_contract"]
                )
            )
            document["generation10_completed_receipts"] = copy.deepcopy(
                spec["generation10_completed_receipts"]
            )
            document["sealed_r12_terminal_failure"] = copy.deepcopy(
                spec["sealed_r12_terminal_failure"]
            )
            document["requested_judgment"] = (
                "PASS_OR_REJECT_OUTPUT_CONTRACT_REPAIR"
            )
            document["authority_after_review"] = (
                "NOT_AUTHORIZED_R12_CONSUMED_NO_RETRY"
            )
        payload = base.canonical_json(document)
    ORIGINAL_WRITE_NEW(path, payload, mode)


def successor_executor() -> bytes:
    source = (R12_PACKAGE / "transaction9_executor.py").read_text(
        encoding="utf-8"
    )
    replacements = (
        (
            "generation9-cursor9-checkpoint008-"
            "transaction009-layer08-binding-r12",
            "generation9-cursor9-checkpoint008-"
            "transaction009-layer08-binding-r14",
        ),
        (
            "transaction9-layer8-continuation-package-r12",
            "transaction9-layer8-continuation-package-r14",
        ),
        (
            "transaction9-layer8-independent-review-r12",
            "transaction9-layer8-independent-review-r14",
        ),
        (
            "transaction9-layer8-execution-binding-r12",
            "transaction9-layer8-execution-binding-r14",
        ),
        (
            "transaction9-layer8-launch-authority-r12",
            "transaction9-layer8-launch-authority-r14",
        ),
    )
    for old, new in replacements:
        base.require(old in source, f"r12 executor token absent: {old}")
        source = source.replace(old, new)

    contracts = json.dumps(
        {
            "parent": generation9_parent_receipt_contract(),
            "transaction009": transaction009_completion_receipt_contract(),
        },
        sort_keys=True,
    )
    helpers = f'''OUTPUT_CONTRACTS = json.loads({contracts!r})


def generation9_parent_receipt() -> dict[str, Any]:
    checkpoint = load_json(GENERATION9 / "checkpoints/transaction-008.json")
    contract = OUTPUT_CONTRACTS["parent"]
    outputs = contract.get("outputs")
    semantics = contract.get("output_semantics")
    require(
        isinstance(outputs, dict)
        and isinstance(outputs.get("hidden"), dict)
        and isinstance(outputs.get("state"), dict)
        and isinstance(semantics, dict)
        and isinstance(semantics.get("hidden"), dict),
        "generation9 parent output contract differs",
    )
    authenticate(outputs["hidden"], "generation9 parent hidden")
    authenticate(outputs["state"], "generation9 parent state")
    normalized = copy.deepcopy(checkpoint)
    normalized["outputs"] = copy.deepcopy(outputs)
    normalized["output_semantics"] = copy.deepcopy(semantics)
    return normalized


def validate_transaction009_completion_receipt(
    receipt: Mapping[str, Any],
) -> None:
    contract = OUTPUT_CONTRACTS["transaction009"]
    outputs = receipt.get("outputs")
    semantics = receipt.get("output_semantics")
    result = receipt.get("result")
    timing = receipt.get("timing")
    require(
        isinstance(outputs, dict)
        and isinstance(outputs.get("hidden"), dict)
        and isinstance(outputs.get("state"), dict)
        and isinstance(semantics, dict)
        and isinstance(semantics.get("hidden"), dict)
        and isinstance(semantics.get("state"), dict)
        and isinstance(result, dict)
        and isinstance(timing, dict),
        "transaction009 receipt outputs or output semantics absent",
    )
    hidden = outputs["hidden"]
    state = outputs["state"]
    hidden_semantics = semantics["hidden"]
    state_semantics = semantics["state"]
    require(
        receipt.get("kind") == contract["kind"]
        and receipt.get("status") == contract["status"]
        and receipt.get("transaction_index") == TRANSACTION_INDEX
        and receipt.get("layer_index") == LAYER_INDEX
        and result == contract["result"]
        and set(outputs) == set(contract["outputs"]["required_keys"])
        and set(semantics)
        == set(contract["output_semantics"]["required_keys"])
        and set(contract["outputs"]["hidden"]["required_record_keys"])
        <= set(hidden)
        and set(contract["outputs"]["state"]["required_record_keys"])
        <= set(state)
        and hidden.get("path") == contract["outputs"]["hidden"]["path"]
        and hidden.get("dtype") == "FP16"
        and hidden.get("elements") == 896
        and isinstance(hidden.get("sha256"), str)
        and len(hidden["sha256"]) == 64
        and isinstance(hidden.get("semantic_sha256"), str)
        and len(hidden["semantic_sha256"]) == 64
        and state.get("path") == contract["outputs"]["state"]["path"]
        and isinstance(state.get("sha256"), str)
        and len(state["sha256"]) == 64
        and hidden_semantics
        == {{
            "dtype": "FP16",
            "elements": 896,
            "semantic_sha256": hidden["semantic_sha256"],
        }}
        and state_semantics.get("layer_index") == LAYER_INDEX
        and state_semantics.get("position") == 4
        and isinstance(timing.get("cumulative_execution_seconds"), (int, float)),
        "transaction009 completion receipt contract differs",
    )


'''
    marker = "def tree_digest(root: Path) -> dict[str, Any]:\n"
    base.require(source.count(marker) == 1, "executor helper insertion differs")
    source = source.replace(marker, helpers + marker)

    old_expected = '''        "checkpoint009_included": True,
    }'''
    new_expected = '''        "checkpoint009_included": True,
        "new_receipt_contract": OUTPUT_CONTRACTS["transaction009"],
    }'''
    base.require(
        source.count(old_expected) == 1,
        "generation10 receipt contract insertion differs",
    )
    source = source.replace(old_expected, new_expected)
    manifest_receipt_gate = '''            "checkpoint009_included": True,
        }
        and spec.get("zero_activity_counters")'''
    base.require(
        source.count(manifest_receipt_gate) == 1,
        "manifest generation10 receipt contract gate differs",
    )
    source = source.replace(
        manifest_receipt_gate,
        '''            "checkpoint009_included": True,
            "new_receipt_contract": OUTPUT_CONTRACTS["transaction009"],
        }
        and spec.get("zero_activity_counters")''',
    )

    old_call = '''                load_json(GENERATION9 / "checkpoints/transaction-008.json"),'''
    base.require(
        source.count(old_call) == 1,
        "generation9 parent receipt call differs",
    )
    source = source.replace(
        old_call, "                generation9_parent_receipt(),"
    )

    publication = '''    receipt_contract = validate_generation10_publication_structure()
    checkpoint_payloads = generation10_checkpoint_payloads(receipt)'''
    base.require(
        source.count(publication) == 1,
        "generation10 publication receipt gate differs",
    )
    source = source.replace(
        publication,
        '''    validate_transaction009_completion_receipt(receipt)
    receipt_contract = validate_generation10_publication_structure()
    checkpoint_payloads = generation10_checkpoint_payloads(receipt)''',
    )

    spec_gate = '''        and spec.get("zero_activity_counters") == ZERO_COUNTERS,'''
    base.require(source.count(spec_gate) == 1, "executor spec gate differs")
    source = source.replace(
        spec_gate,
        '''        and spec.get("generation9_parent_receipt_contract")
        == OUTPUT_CONTRACTS["parent"]
        and spec.get("transaction009_completion_receipt_contract")
        == OUTPUT_CONTRACTS["transaction009"]
        and spec.get("execution_disposition")
        == "NON_EXECUTABLE_REVIEW_ONLY_R12_AUTHORITY_CONSUMED_NO_RETRY"
        and spec.get("zero_activity_counters") == ZERO_COUNTERS,''',
    )
    manifest_gate = '''        and document.get("generation10_completed_receipts")
        == spec["generation10_completed_receipts"]
        and document.get("transaction009_absence_predicates")'''
    base.require(
        source.count(manifest_gate) == 1,
        "executor manifest output gate differs",
    )
    source = source.replace(
        manifest_gate,
        '''        and document.get("generation10_completed_receipts")
        == spec["generation10_completed_receipts"]
        and document.get("generation9_parent_receipt_contract")
        == spec["generation9_parent_receipt_contract"]
        and document.get("transaction009_completion_receipt_contract")
        == spec["transaction009_completion_receipt_contract"]
        and document.get("execution_disposition")
        == spec["execution_disposition"]
        and document.get("transaction009_absence_predicates")''',
    )
    prohibition = '''            "authority consumption or publication by this bounded task",
        ],'''
    base.require(
        source.count(prohibition) == 1,
        "executor package prohibition gate differs",
    )
    source = source.replace(
        prohibition,
        '''            "authority consumption or publication by this bounded task",
            "transaction009 retry, replay, resume, or new authority",
        ],''',
    )
    source = source.replace(
        "def execute_once() -> None:\n"
        "    validate_package(require_zero_runtime=False)\n",
        "def execute_once() -> None:\n"
        "    raise Transaction9Error(\n"
        '        "r14 is review-only; r12 authority was consumed and retry is prohibited"\n'
        "    )\n",
    )
    return source.encode("utf-8")


def review_emitter_source() -> bytes:
    source = review_first.corrected_review_emitter_source().decode("utf-8")
    marker = "    review = {\n"
    static_gate = '''    required_output_functions = {
        "generation9_parent_receipt",
        "validate_transaction009_completion_receipt",
        "validate_generation10_publication_structure",
        "publish_generation10",
    }
    require(
        required_output_functions <= set(functions),
        "executor output-contract gates absent",
    )
'''
    base.require(source.count(marker) == 1, "review output gate insertion differs")
    source = source.replace(marker, static_gate + marker)
    source = source.replace(
        '        "executor_constants_equal_canonical_argv": True,\n',
        '        "executor_constants_equal_canonical_argv": True,\n'
        '        "generation9_parent_outputs_bound": True,\n'
        '        "transaction009_receipt_outputs_bound": True,\n'
        '        "generation10_publication_receipt_gate_present": True,\n'
        '        "r14_execution_disposition": "REVIEW_ONLY_NO_RETRY",\n',
    )
    return source.encode("utf-8")


def configure_base() -> None:
    review_first.REVISION = REVISION
    review_first.PACKAGE = PACKAGE
    review_first.REVIEW_ROOT = REVIEW_ROOT
    review_first.REVIEW_EMITTER = REVIEW_EMITTER
    review_first.REVIEW = REVIEW
    review_first.EXECUTION_BINDING_ROOT = EXECUTION_BINDING_ROOT
    review_first.EXECUTION_BINDING = EXECUTION_BINDING
    review_first.AUTHORITY_ROOT = AUTHORITY_ROOT
    review_first.AUTHORITY = AUTHORITY
    review_first.AUTHORITY_SEAL = AUTHORITY_SEAL
    review_first.GATE_ROOT = GATE_ROOT
    review_first.GATE_CHECK = GATE_CHECK
    review_first.READINESS_ROOT = READINESS_ROOT
    review_first.READINESS = READINESS

    base.REVISION = REVISION
    base.PREDECESSOR_PACKAGE = R12_PACKAGE
    base.PACKAGE = PACKAGE
    base.BINDING_ROOT = EXECUTION_BINDING_ROOT
    base.BINDING = EXECUTION_BINDING
    base.REVIEW_ROOT = REVIEW_ROOT
    base.REVIEW_EMITTER = REVIEW_EMITTER
    base.REVIEW = REVIEW
    base.AUTHORITY_ROOT = AUTHORITY_ROOT
    base.AUTHORITY = AUTHORITY
    base.AUTHORITY_SEAL = AUTHORITY_SEAL
    base.GATE_ROOT = GATE_ROOT
    base.GATE_CHECK = GATE_CHECK
    base.READINESS_ROOT = READINESS_ROOT
    base.READINESS = READINESS
    base.__file__ = str(Path(__file__).resolve())
    base.expected_generation10_receipts = expected_generation10_receipts
    base.validate_zero_effects = validate_successor_zero_effects
    base.binding_specification = binding_specification
    base.successor_executor = successor_executor
    base.review_emitter_source = review_emitter_source
    base.write_new = corrected_write_new


def validate_structural_output_contract() -> dict[str, Any]:
    inspection = base.validate()
    source = (PACKAGE / "transaction9_executor.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source, filename=str(PACKAGE / "transaction9_executor.py"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "generation9_parent_receipt",
        "validate_transaction009_completion_receipt",
        "validate_generation10_publication_structure",
        "generation10_checkpoint_payloads",
        "publish_generation10",
    }
    base.require(required <= set(functions), "sealed output-contract gates absent")
    probe = r'''
import importlib.util
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("ace3_tx009_r14_probe", path)
if spec is None or spec.loader is None:
    raise RuntimeError("executor import specification unavailable")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.validate_manifest_document(module.load_json(module.OUTPUT_PACKAGE_MANIFEST))
parent = module.generation9_parent_receipt()
if "outputs" not in parent or "output_semantics" not in parent:
    raise RuntimeError("normalized generation9 parent outputs absent")
contract = module.OUTPUT_CONTRACTS["transaction009"]
receipt = {
    "kind": contract["kind"],
    "status": contract["status"],
    "transaction_index": 9,
    "layer_index": 8,
    "result": contract["result"],
    "outputs": {
        "hidden": {
            "path": contract["outputs"]["hidden"]["path"],
            "bytes": 9856,
            "sha256": "0" * 64,
            "dtype": "FP16",
            "elements": 896,
            "semantic_sha256": "1" * 64,
        },
        "state": {
            "path": contract["outputs"]["state"]["path"],
            "bytes": 249565,
            "sha256": "2" * 64,
        },
    },
    "output_semantics": {
        "hidden": {
            "dtype": "FP16",
            "elements": 896,
            "semantic_sha256": "1" * 64,
        },
        "state": {"layer_index": 8, "position": 4},
    },
    "timing": {"cumulative_execution_seconds": 1.0},
}
module.validate_transaction009_completion_receipt(receipt)
payloads = module.generation10_checkpoint_payloads(receipt)
contract10 = module.validate_generation10_publication_structure()
if list(payloads) != contract10["checkpoint_paths"]:
    raise RuntimeError("generation10 payload paths differ")
try:
    module.validate_transaction009_completion_receipt(
        {key: value for key, value in receipt.items() if key != "outputs"}
    )
except module.Transaction9Error:
    pass
except KeyError as error:
    raise RuntimeError("missing outputs still raises KeyError") from error
else:
    raise RuntimeError("missing outputs was accepted")
print(json.dumps({
    "generation9_parent_outputs": sorted(parent["outputs"]),
    "generation9_parent_output_record_keys": {
        key: sorted(value) for key, value in parent["outputs"].items()
    },
    "generation9_parent_semantics": sorted(parent["output_semantics"]),
    "generation10_checkpoint_count": len(payloads),
    "missing_outputs_failure": "Transaction9Error",
    "workload_calls": 0,
}, sort_keys=True))
'''
    completed = subprocess.run(
        [
            str(base.PYTHON),
            "-c",
            probe,
            str(PACKAGE / "transaction9_executor.py"),
        ],
        cwd=base.ROOT,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"},
        check=False,
        capture_output=True,
        text=True,
    )
    base.require(
        completed.returncode == 0,
        "r14 structural output probe failed: "
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}",
    )
    observed = json.loads(completed.stdout)
    base.require(
        observed
        == {
            "generation9_parent_outputs": ["hidden", "state"],
            "generation9_parent_output_record_keys": {
                "hidden": ["bytes", "path", "sha256"],
                "state": ["bytes", "path", "sha256"],
            },
            "generation9_parent_semantics": ["hidden", "state"],
            "generation10_checkpoint_count": 10,
            "missing_outputs_failure": "Transaction9Error",
            "workload_calls": 0,
        },
        "r14 structural output probe result differs",
    )
    manifest = base.load_json(PACKAGE / "package-manifest.json")
    request = base.load_json(PACKAGE / "review-request.json")
    spec_document = base.load_json(PACKAGE / "binding-spec.json")
    for label, document in (
        ("manifest", manifest),
        ("review request", request),
    ):
        base.require(
            document.get("generation9_parent_receipt_contract")
            == spec_document["generation9_parent_receipt_contract"]
            and document.get("transaction009_completion_receipt_contract")
            == spec_document["transaction009_completion_receipt_contract"],
            f"{label} output contract differs",
        )
    base.require(
        request.get("generation10_completed_receipts")
        == spec_document["generation10_completed_receipts"]
        and request.get("authority_after_review")
        == "NOT_AUTHORIZED_R12_CONSUMED_NO_RETRY",
        "review request publication or authority contract differs",
    )
    after = validate_successor_zero_effects()
    base.require(
        inspection["before_zero_effects"] == after
        and inspection["after_zero_effects"] == after,
        "structural output probe changed the sealed runtime frontier",
    )
    return {
        "inspection": inspection,
        "structural_probe": observed,
        "package_manifest": base.file_record(
            PACKAGE / "package-manifest.json"
        ),
        "package_seal": base.file_record(PACKAGE / "package-seal.json"),
        "review_request": base.file_record(PACKAGE / "review-request.json"),
    }


def sealed_authority_conflicts() -> list[dict[str, str]]:
    for path, expected_sha256 in EXPECTED_REVIEW_SHA256.items():
        base.require(
            base.sha256_file(path) == expected_sha256,
            f"r14 reviewed artifact differs: {path}",
        )
    manifest = base.load_json(PACKAGE / "package-manifest.json")
    seal = base.load_json(PACKAGE / "package-seal.json")
    request = base.load_json(PACKAGE / "review-request.json")
    gate = base.load_json(GATE_CHECK)
    source = (PACKAGE / "transaction9_executor.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source, filename=str(PACKAGE / "transaction9_executor.py"))
    execute_once = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "execute_once"
        ),
        None,
    )
    refuse_message = (
        "r14 is review-only; r12 authority was consumed and retry is prohibited"
    )
    refusal = (
        execute_once is not None
        and len(execute_once.body) >= 1
        and isinstance(execute_once.body[0], ast.Raise)
        and isinstance(execute_once.body[0].exc, ast.Call)
        and isinstance(execute_once.body[0].exc.func, ast.Name)
        and execute_once.body[0].exc.func.id == "Transaction9Error"
        and len(execute_once.body[0].exc.args) == 1
        and isinstance(execute_once.body[0].exc.args[0], ast.Constant)
        and execute_once.body[0].exc.args[0].value == refuse_message
    )
    base.require(
        seal.get("status") == "SEALED_REVIEW_REQUIRED"
        and seal.get("authority_created") is False
        and seal.get("execution_authorized") is False
        and manifest.get("status") == "SEALED_REVIEW_REQUIRED"
        and manifest.get("execution_disposition")
        == "NON_EXECUTABLE_REVIEW_ONLY_R12_AUTHORITY_CONSUMED_NO_RETRY"
        and manifest.get("authority_created") is False
        and manifest.get("execution_authorized") is False
        and request.get("requested_judgment")
        == "PASS_OR_REJECT_OUTPUT_CONTRACT_REPAIR"
        and request.get("authority_after_review")
        == "NOT_AUTHORIZED_R12_CONSUMED_NO_RETRY"
        and request.get("launch_authority_required") is False
        and request.get("this_review_consumes_authority") is False
        and request.get("this_review_executes_payload") is False
        and gate.get("status") == "PASS"
        and gate.get("launch_authority_created") is False
        and gate.get("authority_consumption_performed") is False
        and gate.get("payload_execution_performed") is False
        and all(
            value == 0
            for value in gate.get("activity_counters", {}).values()
        )
        and refusal,
        "sealed r14 authority prohibition differs",
    )
    return [
        {
            "reason_code": "R14_PACKAGE_NON_EXECUTABLE",
            "evidence": manifest["execution_disposition"],
        },
        {
            "reason_code": "R14_REVIEW_WITHHOLDS_AUTHORITY",
            "evidence": request["authority_after_review"],
        },
        {
            "reason_code": "R12_AUTHORITY_CONSUMED_NO_RETRY",
            "evidence": refuse_message,
        },
    ]


def build_fail_closed_authority_review(
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    conflicts = sealed_authority_conflicts()
    zero_effects = validation["inspection"]["after_zero_effects"]
    gate = base.load_json(GATE_CHECK)
    base.require(
        validation["package_manifest"]
        == base.file_record(PACKAGE / "package-manifest.json")
        and validation["package_seal"]
        == base.file_record(PACKAGE / "package-seal.json")
        and validation["review_request"]
        == base.file_record(PACKAGE / "review-request.json")
        and zero_effects["r12_workload_calls"] == 0
        and zero_effects["successor_authority_consumption"] == 0
        and zero_effects["successor_payload_execution"] == 0
        and zero_effects["successor_generation10_publication"] == 0
        and zero_effects["transaction010_025_effects_absent"] is True,
        "r14 fail-closed review inputs differ",
    )
    return {
        "schema_version": 1,
        "kind": (
            "ace3_transaction009_layer8_r14_"
            "fail_closed_authority_review"
        ),
        "status": "REJECT",
        "result": "ambiguous_objective",
        "producer_role": "engineer",
        "independent_reviewer_acceptance_present": False,
        "reviewed_bindings": {
            "package_manifest": validation["package_manifest"],
            "package_seal": validation["package_seal"],
            "review_request": validation["review_request"],
            "read_only_gate": base.file_record(GATE_CHECK),
        },
        "generation9_parent_receipt_contract": (
            generation9_parent_receipt_contract()
        ),
        "required_generation10_receipt_contract": (
            transaction009_completion_receipt_contract()
        ),
        "conflicts": conflicts,
        "authority_decision": {
            "created": False,
            "consumed": False,
            "execution_authorized": False,
            "reason_code": "SEALED_R14_FORBIDS_NEW_AUTHORITY",
        },
        "activity_counters": copy.deepcopy(gate["activity_counters"]),
        "zero_effects": {
            "r12_workload_calls": zero_effects["r12_workload_calls"],
            "successor_authority_consumption": (
                zero_effects["successor_authority_consumption"]
            ),
            "successor_payload_execution": (
                zero_effects["successor_payload_execution"]
            ),
            "successor_generation10_publication": (
                zero_effects["successor_generation10_publication"]
            ),
            "generation10_absent": zero_effects["generation10_absent"],
            "transaction010_025_effects_absent": (
                zero_effects["transaction010_025_effects_absent"]
            ),
        },
        "transaction010_025": {
            "transaction_indices": list(range(10, 26)),
            "authority_or_effects_present": False,
        },
        "claim_boundary": (
            "Fail-closed r14 authority review only. The immutable r14 "
            "package and review request repair output/output_semantics but "
            "explicitly prohibit a new transaction009 authority after the "
            "consumed r12 attempt. No authority was created or consumed, no "
            "workload ran, generation10 remains absent, and transaction010-025 "
            "remain untouched."
        ),
    }


def emit_fail_closed_authority_review(
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    base.require(
        not AUTHORITY_REVIEW.exists(),
        f"r14 fail-closed authority review already exists: {AUTHORITY_REVIEW}",
    )
    document = build_fail_closed_authority_review(validation)
    ORIGINAL_WRITE_NEW(AUTHORITY_REVIEW, base.canonical_json(document))
    base.require(
        base.load_json(AUTHORITY_REVIEW) == document,
        "r14 fail-closed authority review did not persist",
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=("prepare", "validate", "review"),
    )
    arguments = parser.parse_args()
    configure_base()
    if arguments.operation == "prepare":
        base.prepare()
    result = validate_structural_output_contract()
    review = None
    if arguments.operation == "review":
        review = emit_fail_closed_authority_review(result)
    print(
        "TX009_OUTPUT_CONTRACT_SUCCESSOR_"
        + (
            "PREPARED"
            if arguments.operation == "prepare"
            else "REVIEW_REJECTED"
            if arguments.operation == "review"
            else "VALID"
        )
        + f" revision={REVISION}"
        + f" package_sha256={result['package_manifest']['sha256']}"
        + f" package_seal_sha256={result['package_seal']['sha256']}"
        + f" review_request_sha256={result['review_request']['sha256']}"
        + (
            f" review_result={review['result']}"
            if review is not None
            else ""
        )
        + " review=absent execution_binding=absent authority=absent"
        + " successor_consumption=0 workload=0 payload=0 terminal=0"
        + " generation10=0 transaction010_025=0"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        base.SuccessorError,
        OSError,
        UnicodeDecodeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise SystemExit(
            f"TX009_OUTPUT_CONTRACT_SUCCESSOR_REFUSED {error}"
        ) from error
