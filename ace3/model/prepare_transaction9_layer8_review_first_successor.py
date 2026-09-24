#!/usr/bin/env python3
"""Prepare the review-first transaction009/layer8 r11 successor."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
from typing import Any

import prepare_transaction9_layer8_binding_successor as base


REVISION = "r11"
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
READINESS_ROOT = (
    RUNTIME / f"transaction9-layer8-successor-readiness-{REVISION}"
)
READINESS = READINESS_ROOT / "readiness.json"

R9_PACKAGE = RUNTIME / "transaction9-layer8-continuation-package-r9"
R9_GATE = RUNTIME / "transaction9-layer8-read-only-gates-r9"
R9_REVIEW_ROOT = RUNTIME / "transaction9-layer8-independent-review-r9"
R9_AUTHORITY_ROOT = RUNTIME / "transaction9-layer8-launch-authority-r9"
R10_PACKAGE = RUNTIME / "transaction9-layer8-continuation-package-r10"
R10_GATE = RUNTIME / "transaction9-layer8-read-only-gates-r10"
R10_REVIEW_ROOT = RUNTIME / "transaction9-layer8-independent-review-r10"
R10_AUTHORITY_ROOT = RUNTIME / "transaction9-layer8-launch-authority-r10"
R10_BINDING_ROOT = RUNTIME / "transaction9-layer8-pre-authority-binding-r10"

SEALED_HISTORY = {
    R9_PACKAGE / "package-manifest.json": (
        "5a2587048c1c78526ac85180cb957e7c560677c2c0cf775b5885e2c6fae29b19"
    ),
    R9_PACKAGE / "package-seal.json": (
        "a75db403114dca975c45201953d249ba7dc508fa42e8e6005db7543b424cd977"
    ),
    R9_GATE / "gate-check.json": (
        "7fd010bc433c7d52112157abd44961dac12777e16305e911d9bfae09b2fbdf9f"
    ),
    R10_PACKAGE / "package-manifest.json": (
        "68653fc5ba8ac25123d9ae11d8fa44e5b1df747cc0a091ff50da4ba10504c1a6"
    ),
    R10_PACKAGE / "package-seal.json": (
        "accd81c438099189362a2effca51446c0bc2ce268fbbc90e361069f03b56bdd6"
    ),
    R10_GATE / "gate-check.json": (
        "cad2df6ce493d29174021682d028f77cdd794e336a2249afb1bf7f70c1ebffdd"
    ),
}

ORIGINAL_WRITE_NEW = base.write_new
ORIGINAL_REVIEW_EMITTER_SOURCE = base.review_emitter_source
ORIGINAL_BINDING_SPECIFICATION = base.binding_specification


def replace_once(source: str, old: str, new: str, label: str) -> str:
    base.require(source.count(old) == 1, f"{label} replacement count differs")
    return source.replace(old, new)


def history_records() -> dict[str, Any]:
    return {
        "r9": {
            "package_manifest": base.file_record(
                R9_PACKAGE / "package-manifest.json"
            ),
            "package_seal": base.file_record(R9_PACKAGE / "package-seal.json"),
            "read_only_gate": base.file_record(R9_GATE / "gate-check.json"),
            "review_emitter": base.file_record(
                R9_REVIEW_ROOT / "review-emitter.py"
            ),
            "review_status": "ABSENT_FAILED_AUTHORITY_BEFORE_REVIEW_ORDERING",
            "authority_status": "ABSENT_NON_EXECUTABLE_HISTORY",
        },
        "r10": {
            "package_manifest": base.file_record(
                R10_PACKAGE / "package-manifest.json"
            ),
            "package_seal": base.file_record(
                R10_PACKAGE / "package-seal.json"
            ),
            "read_only_gate": base.file_record(R10_GATE / "gate-check.json"),
            "review_status": "ABSENT_REJECTED_TWO_PHASE_SCHEMA",
            "authority_status": "ABSENT_NON_EXECUTABLE_HISTORY",
        },
    }


def validate_sealed_history() -> dict[str, Any]:
    for path, expected_sha256 in SEALED_HISTORY.items():
        base.require(
            base.sha256_file(path) == expected_sha256,
            f"sealed history differs: {path}",
        )
    base.require(
        not (R9_REVIEW_ROOT / "independent-review.json").exists()
        and not R9_AUTHORITY_ROOT.exists()
        and not R10_REVIEW_ROOT.exists()
        and not R10_BINDING_ROOT.exists()
        and not R10_AUTHORITY_ROOT.exists(),
        "r9 or r10 non-executable history was extended",
    )
    return history_records()


def corrected_executor_source() -> bytes:
    source = (R10_PACKAGE / "transaction9_executor.py").read_text(
        encoding="utf-8"
    )
    replacements = (
        (
            "generation9-cursor9-checkpoint008-"
            "transaction009-layer08-binding-r10",
            "generation9-cursor9-checkpoint008-"
            "transaction009-layer08-binding-r11",
            "package identity",
        ),
        (
            "transaction9-layer8-continuation-package-r10",
            "transaction9-layer8-continuation-package-r11",
            "package path",
        ),
        (
            "transaction9-layer8-independent-review-r10",
            "transaction9-layer8-independent-review-r11",
            "review path",
        ),
        (
            "transaction9-layer8-launch-authority-r10",
            "transaction9-layer8-launch-authority-r11",
            "authority path",
        ),
    )
    for old, new, label in replacements:
        base.require(old in source, f"{label} predecessor token absent")
        source = source.replace(old, new)

    authority_declaration = (
        "AUTHORITY = (\n"
        "    RUNTIME\n"
        '    / "transaction9-layer8-launch-authority-r11"\n'
        '    / "launch-authority.json"\n'
        ")\n"
    )
    source = replace_once(
        source,
        authority_declaration,
        authority_declaration
        + "EXECUTION_BINDING = (\n"
        + "    RUNTIME\n"
        + '    / "transaction9-layer8-execution-binding-r11"\n'
        + '    / "execution-binding.json"\n'
        + ")\n",
        "execution binding declaration",
    )

    review_start = source.index("def validate_review() -> dict[str, Any]:")
    authority_start = source.index(
        "def validate_authority() -> dict[str, Any]:", review_start
    )
    gate_end = source.index(
        "def write_exclusive_json(path: Path, document: object) -> None:",
        authority_start,
    )
    authority_gate = source[authority_start:gate_end]
    authority_gate = replace_once(
        authority_gate,
        "    authority = load_json(AUTHORITY)\n    validate_review()\n",
        "    authority = load_json(AUTHORITY)\n"
        "    review = validate_review()\n"
        "    binding = validate_execution_binding(review)\n",
        "authority review dependency",
    )
    authority_gate = replace_once(
        authority_gate,
        '        and authority.get("aggregate_preflight")\n'
        '        == authority.get("pre_authority_binding_check")\n',
        '        and authority.get("aggregate_preflight")\n'
        "        == file_record(EXECUTION_BINDING)\n"
        '        and authority.get("execution_binding")\n'
        "        == file_record(EXECUTION_BINDING)\n"
        '        and authority.get("pre_authority_binding_check")\n'
        "        == file_record(EXECUTION_BINDING)\n",
        "authority execution binding",
    )
    authority_gate = replace_once(
        authority_gate,
        '        and authority.get("reviewer_acceptance")\n'
        "        == {\n"
        '            "required": True,\n'
        '            "path": str(REVIEW),\n'
        '            "present_at_issuance": False,\n'
        "        }\n",
        '        and authority.get("reviewer_acceptance")\n'
        "        == {\n"
        '            "required": True,\n'
        '            "path": str(REVIEW),\n'
        '            "present_at_issuance": True,\n'
        '            "review": file_record(REVIEW),\n'
        '            "review_status": "PASS",\n'
        "        }\n",
        "authority immutable review binding",
    )

    corrected_gates = '''def validate_review() -> dict[str, Any]:
    review = load_json(REVIEW)
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
        and review.get("package_tree") == tree_record(PACKAGE_ROOT)
        and review.get("review_request") == file_record(REVIEW_REQUEST)
        and review.get("canonical_future_launch") == launch_contract()
        and review.get("launch_authority_present") is False
        and review.get("execution_binding_present") is False
        and review.get("review_gate_authority_independent") is True
        and review.get("execution_binding_binds_immutable_review") is True
        and review.get("authority_binds_review_and_execution_binding") is True
        and review.get("executor_constants_equal_canonical_argv") is True
        and review.get("authoritative_generation") == START_CURSOR
        and review.get("authoritative_cursor") == START_CURSOR
        and review.get("target_generation") == EXIT_CURSOR
        and review.get("target_cursor") == EXIT_CURSOR
        and review.get("transaction_index") == TRANSACTION_INDEX
        and review.get("layer_index") == LAYER_INDEX
        and review.get("transaction008_consumed_non_reusable") is True
        and review.get("transaction009_authority_created") is False
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


def validate_execution_binding(
    review: Mapping[str, Any],
) -> dict[str, Any]:
    binding = load_json(EXECUTION_BINDING)
    require(
        binding.get("schema_version") == 1
        and binding.get("kind")
        == "ace3_transaction009_layer8_post_review_execution_binding"
        and binding.get("status") == "BOUND_AFTER_EXACT_REVIEW_PASS"
        and binding.get("producer_role") == "manager"
        and binding.get("mission_id") == SUCCESSOR_MISSION_ID
        and binding.get("node_key") == SUCCESSOR_NODE_KEY
        and binding.get("package_manifest")
        == file_record(OUTPUT_PACKAGE_MANIFEST)
        and binding.get("package_seal") == file_record(PACKAGE_SEAL)
        and binding.get("independent_review") == file_record(REVIEW)
        and binding.get("review_status") == review["status"]
        and binding.get("authorized_launch") == launch_contract()
        and binding.get("authority_namespace") == str(AUTHORITY)
        and binding.get("authoritative_generation") == START_CURSOR
        and binding.get("authoritative_cursor") == START_CURSOR
        and binding.get("target_generation") == EXIT_CURSOR
        and binding.get("target_cursor") == EXIT_CURSOR
        and binding.get("transaction_index") == TRANSACTION_INDEX
        and binding.get("layer_index") == LAYER_INDEX
        and binding.get("authority_issued") is False
        and binding.get("authority_consumed") is False
        and binding.get("transaction009_executed") is False
        and binding.get("generation10_exists") is False
        and binding.get("transactions010_025_absent") is True
        and binding.get("activity_counters") == ZERO_COUNTERS,
        "post-review execution binding differs",
    )
    return binding


'''
    source = (
        source[:review_start]
        + corrected_gates
        + authority_gate
        + source[gate_end:]
    )
    return source.encode("utf-8")


def corrected_review_emitter_source() -> bytes:
    source = ORIGINAL_REVIEW_EMITTER_SOURCE().decode("utf-8")
    marker = "    review = {\n"
    static_gate = '''    executor_tree = ast.parse(
        executor.read_text(encoding="utf-8"), filename=str(executor)
    )
    functions = {
        node.name: node
        for node in executor_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    review_gate = functions.get("validate_review")
    binding_gate = functions.get("validate_execution_binding")
    authority_gate = functions.get("validate_authority")
    require(
        review_gate is not None
        and binding_gate is not None
        and authority_gate is not None,
        "two-phase executor gates absent",
    )
    review_names = {
        node.id for node in ast.walk(review_gate) if isinstance(node, ast.Name)
    }
    binding_names = {
        node.id for node in ast.walk(binding_gate) if isinstance(node, ast.Name)
    }
    authority_calls = {
        node.func.id
        for node in ast.walk(authority_gate)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    require(
        "AUTHORITY" not in review_names
        and "EXECUTION_BINDING" not in review_names,
        "review gate depends on post-review artifacts",
    )
    require(
        "REVIEW" in binding_names
        and "EXECUTION_BINDING" in binding_names
        and {"validate_review", "validate_execution_binding"}
        <= authority_calls,
        "later gates do not bind the immutable review",
    )
'''
    source = replace_once(
        source,
        marker,
        static_gate + marker,
        "review emitter static ordering gate",
    )
    source = replace_once(
        source,
        '        "execution_binding_present": False,\n',
        '        "execution_binding_present": False,\n'
        '        "review_gate_authority_independent": True,\n'
        '        "execution_binding_binds_immutable_review": True,\n'
        '        "authority_binds_review_and_execution_binding": True,\n',
        "review ordering findings",
    )
    return source.encode("utf-8")


def corrected_write_new(
    path: Path, payload: bytes, mode: int = 0o400
) -> None:
    if path.parent == PACKAGE and path.suffix == ".json":
        document = json.loads(payload)
        if path.name == "binding-spec.json":
            document["ordering_contract"] = {
                "phase_one": "IMMUTABLE_PRE_AUTHORITY_REVIEW",
                "phase_two": "POST_PASS_EXECUTION_BINDING_AND_AUTHORITY",
                "review_required_before_execution_binding": True,
                "review_accepted_unchanged_by_later_gates": True,
            }
        elif path.name == "authoritative-baseline.json":
            document["preserved_non_executable_history"] = history_records()
            document["binding_repair"] = (
                "review acceptance is authority-independent; the later "
                "execution binding and authority bind the unchanged review"
            )
        elif path.name == "package-manifest.json":
            document["manager_directive"] = (
                "AUTHORITATIVE MANAGER REVIEW-FIRST SUCCESSOR DECISION"
            )
            document["future_execution_binding"] = str(EXECUTION_BINDING)
            document["ordering_contract"] = {
                "phase_one": "IMMUTABLE_PRE_AUTHORITY_REVIEW",
                "phase_two": "POST_PASS_EXECUTION_BINDING_AND_AUTHORITY",
                "review_required_before_execution_binding": True,
                "review_accepted_unchanged_by_later_gates": True,
            }
            document["preserved_non_executable_history"] = history_records()
            document["predecessor_package"]["binding_status"] = (
                "REJECTED_TWO_PHASE_ORDERING_SCHEMA"
            )
        elif path.name == "review-request.json":
            document["future_execution_binding"] = str(EXECUTION_BINDING)
            document["review_accepted_unchanged_by_later_gates"] = True
            document["authority_phase"] = (
                "SEPARATE_POST_PASS_CREATE_EXCLUSIVE_MANAGER_TASK"
            )
        payload = base.canonical_json(document)
    ORIGINAL_WRITE_NEW(path, payload, mode)


def corrected_binding_specification(
    parent: dict[str, Any], zero_effects: dict[str, Any]
) -> dict[str, Any]:
    document = ORIGINAL_BINDING_SPECIFICATION(parent, zero_effects)
    document["ordering_contract"] = {
        "phase_one": "IMMUTABLE_PRE_AUTHORITY_REVIEW",
        "phase_two": "POST_PASS_EXECUTION_BINDING_AND_AUTHORITY",
        "review_required_before_execution_binding": True,
        "review_accepted_unchanged_by_later_gates": True,
    }
    return document


def configure_base() -> None:
    base.REVISION = REVISION
    base.PREDECESSOR_PACKAGE = R10_PACKAGE
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
    base.binding_specification = corrected_binding_specification
    base.successor_executor = corrected_executor_source
    base.review_emitter_source = corrected_review_emitter_source
    base.write_new = corrected_write_new


def validate_two_phase_package() -> dict[str, Any]:
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
    review_gate = functions["validate_review"]
    binding_gate = functions["validate_execution_binding"]
    authority_gate = functions["validate_authority"]
    review_names = {
        node.id for node in ast.walk(review_gate) if isinstance(node, ast.Name)
    }
    binding_names = {
        node.id for node in ast.walk(binding_gate) if isinstance(node, ast.Name)
    }
    authority_calls = {
        node.func.id
        for node in ast.walk(authority_gate)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    base.require(
        "AUTHORITY" not in review_names
        and "EXECUTION_BINDING" not in review_names,
        "sealed review gate depends on post-review artifacts",
    )
    base.require(
        "REVIEW" in binding_names
        and "EXECUTION_BINDING" in binding_names
        and {"validate_review", "validate_execution_binding"}
        <= authority_calls,
        "sealed later gates do not bind the immutable review",
    )
    manifest_sha256 = base.sha256_file(PACKAGE / "package-manifest.json")
    seal_sha256 = base.sha256_file(PACKAGE / "package-seal.json")
    executor_sha256 = base.sha256_file(PACKAGE / "transaction9_executor.py")
    historical_hashes = set(SEALED_HISTORY.values())
    base.require(
        {manifest_sha256, seal_sha256, executor_sha256}.isdisjoint(
            historical_hashes
        ),
        "r11 package is not hash-disjoint from sealed history",
    )
    base.require(
        not REVIEW_ROOT.exists()
        and not EXECUTION_BINDING_ROOT.exists()
        and not AUTHORITY_ROOT.exists()
        and not READINESS_ROOT.exists(),
        "phase one created review, execution binding, or authority",
    )
    return inspection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "validate"))
    arguments = parser.parse_args()
    configure_base()
    validate_sealed_history()
    if arguments.operation == "prepare":
        base.prepare()
    inspection = validate_two_phase_package()
    print(
        "TX009_REVIEW_FIRST_SUCCESSOR_"
        + ("PREPARED" if arguments.operation == "prepare" else "VALID")
        + f" revision={REVISION}"
        + f" package_sha256={inspection['package_manifest']['sha256']}"
        + f" package_seal_sha256={base.sha256_file(PACKAGE / 'package-seal.json')}"
        + f" review_request={PACKAGE / 'review-request.json'}"
        + " review=absent execution_binding=absent authority=absent"
        + " consumption=0 payload=0 terminal=0 generation10=0"
        + " transaction010_025=0"
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
            f"TX009_REVIEW_FIRST_SUCCESSOR_REFUSED {error}"
        ) from error
