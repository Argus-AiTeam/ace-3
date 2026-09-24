#!/usr/bin/env python3
"""Create and independently review the transaction007/layer06 package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


sys.dont_write_bytecode = True

ROOT = Path("/home/argustest/ace3-argus")
MISSION_ID = "cd582996d710"
PARENT_MISSION_ID = "51e05603a20c"
RUNTIME_IDENTITY = "ace3-position3-fresh-r11-20260831t215500z"
RUNTIME = ROOT / "build/model24_selected_token_position3_runs" / RUNTIME_IDENTITY
PREPARATION = RUNTIME / "transaction7-layer6-preparation-r1"
PACKAGE = RUNTIME / "transaction7-layer6-continuation-package-r1"
REVIEW = (
    RUNTIME
    / "transaction7-layer6-continuation-review-r1"
    / "independent-review.json"
)
AUTHORITY = (
    RUNTIME
    / "transaction7-layer6-continuation-authority-r1"
    / "manager-authority.json"
)
BASE_PACKAGE = RUNTIME / "transaction6-layer5-continuation-package-r3"
BASE_EXECUTOR = BASE_PACKAGE / "transaction6_executor.py"
BASE_REVIEWER = BASE_PACKAGE / "review-emitter.py"
ENGINEER_SOURCE = PREPARATION / "prepare_transaction7_layer6_executor.py"
REVIEWER_SOURCE = PREPARATION / "emit_transaction7_layer6_executor_review.py"
MISSION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/cd582996d710/mission.json"
)
PARENT_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
    "handoffs/51e05603a20c/latest.json"
)
PARENT_REVIEW = PARENT_HANDOFF.with_name("round-0001.json")
ADOPTION = RUNTIME / "transaction2-receipt-adoption-r4"
POINTER = ADOPTION / "authoritative-state.json"
GENERATION7 = ADOPTION / "state-generations/generation-0000000007"
GENERATION8 = ADOPTION / "state-generations/generation-0000000008"
GENERATION8_STAGING = ADOPTION / "state-generations/.generation-0000000008.prepared"
TRANSACTIONS = RUNTIME / "transactions"
TRANSACTION7 = TRANSACTIONS / "transaction-007"
FUTURE_ROOT = RUNTIME / "transaction7-authoritative-generation8"

FIXED_SHA256 = {
    BASE_EXECUTOR: "8b5ab23311674c824e10f7a6faa202e4118f8f6d03be1f792a2da94d33aa8af6",
    BASE_REVIEWER: "5102136a58d7dcddab2b0d3bf4bfa9dd8fd95fdf44cad76d556321c520cb1f2f",
    MISSION: "df93998131e0f9e2111aa92432ef18a9aa36b3c21058589246202848c18ba9c0",
    PARENT_HANDOFF: "deb9cf4fd1faa71af8ca9e49e84b97a1e7d9ee934b9fbba2ad35d9e767c9f2a2",
    PARENT_REVIEW: "5dc0031bcf4ec67244da03813b8a8434bc236b5b6699b8bd08c1e034c784158b",
    POINTER: "87727196763ad11b666d07b14cf9d885ff58169277b98503d8407b1262d7f60d",
    GENERATION7
    / "generation-manifest.json": "55f2c189cd8694e72a8156c8f6e4fcf29eb8be0963ef04334c52956e21a2ea28",
    GENERATION7
    / "ledger.json": "784dbeeafded4b0d956601c4697a535848ea99ec1350cbdee25bdc4a2d35cc6a",
    GENERATION7
    / "checkpoints/transaction-006.json": "9a2aa573c27e703970d3badb61a2ecbb416ff711c0dc5beb4a7285d0e6bb6929",
    TRANSACTIONS
    / "transaction-006/position004.state": "d8445c2badf3e544e5affd9942f66e2efd71e23647cc20f66e1656ed6589ebb0",
}


class PackageError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require_fixed(path: Path) -> None:
    metadata = path.lstat()
    require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"fixed regular non-symlink file required: {path}",
    )
    require(sha256_file(path) == FIXED_SHA256[path], f"fixed hash differs: {path}")


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


def replace(
    source: str,
    old: str,
    new: str,
    label: str,
    *,
    minimum: int = 1,
) -> str:
    count = source.count(old)
    require(count >= minimum, f"{label} replacement count is {count}")
    return source.replace(old, new)


def manager_validation_source() -> str:
    return '''
def validate_manager_mission() -> None:
    require_fixed(MANAGER_MISSION)
    mission = load_json(MANAGER_MISSION)
    objective = mission.get("objective", "")
    require(
        mission.get("kind") == "mission_context"
        and mission.get("mission_id") == MISSION_ID
        and mission.get("node_key") == "tx007-layer06-package-review"
        and mission.get("scope") == "bounded"
        and mission.get("stage") == "rtl"
        and isinstance(objective, str)
        and "generation7/checkpoint006/cursor7" in objective
        and "transaction007/layer06" in objective
        and "generation8/checkpoint007/cursor8" in objective
        and "zero production RTL transaction" in objective,
        "Manager transaction007 package mission differs",
    )

'''


def fixture_validation_source() -> str:
    return '''
def validate_layer06_fixture(record: Mapping[str, Any]) -> dict[str, Any]:
    authenticate(record, "layer06 fixture manifest")
    fixture = load_json(Path(record["path"]))
    required_names = {
        "model.layers.6.input_layernorm.weight",
        "model.layers.6.post_attention_layernorm.weight",
        *{
            f"model.layers.6.mlp.{projection}.{field}"
            for projection in ("gate_proj", "up_proj", "down_proj")
            for field in ("qweight", "qzeros", "scales")
        },
        *{
            f"model.layers.6.self_attn.{projection}.{field}"
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            for field in ("qweight", "qzeros", "scales")
        },
        *{
            f"model.layers.6.self_attn.{projection}.bias"
            for projection in ("q_proj", "k_proj", "v_proj")
        },
    }
    tensors = fixture.get("tensors")
    require(
        fixture.get("schema_version") == 1
        and fixture.get("kind") == "ace3_position2_live_transaction_vectors"
        and fixture.get("layer_index") == 6
        and fixture.get("position") == 2
        and isinstance(fixture.get("input_activation_sha256"), str)
        and len(fixture["input_activation_sha256"]) == 64
        and isinstance(tensors, list)
        and len(tensors) == len(required_names),
        "layer06 fixture identity differs",
    )
    authenticate(fixture["input"], "layer06 fixture input")
    authenticate(fixture["rope_coefficients"], "layer06 RoPE coefficients")
    observed: set[str] = set()
    for index, tensor in enumerate(tensors):
        checkpoint = tensor.get("checkpoint_tensor")
        serialized = tensor.get("serialized")
        require(
            isinstance(checkpoint, dict)
            and set(checkpoint) == {"bytes", "dtype", "name", "sha256", "shape"}
            and isinstance(serialized, dict)
            and isinstance(checkpoint.get("name"), str)
            and checkpoint["name"] not in observed
            and checkpoint["name"] in required_names
            and checkpoint.get("dtype") in {"F16", "I32"}
            and isinstance(checkpoint.get("shape"), list)
            and checkpoint["shape"]
            and all(type(dimension) is int and dimension > 0 for dimension in checkpoint["shape"])
            and isinstance(checkpoint.get("sha256"), str)
            and len(checkpoint["sha256"]) == 64,
            f"layer06 checkpoint tensor {index} metadata differs",
        )
        elements = 1
        for dimension in checkpoint["shape"]:
            elements *= dimension
        expected_bytes = elements * (2 if checkpoint["dtype"] == "F16" else 4)
        require(
            checkpoint.get("bytes") == expected_bytes,
            f"layer06 checkpoint tensor {index} size differs",
        )
        name = checkpoint["name"]
        expected_dtype = (
            "I32"
            if name.endswith((".qweight", ".qzeros"))
            else "F16"
        )
        require(
            checkpoint["dtype"] == expected_dtype,
            f"layer06 checkpoint tensor {name} dtype differs",
        )
        authenticate(serialized, f"layer06 serialized tensor {name}")
        observed.add(name)
    require(observed == required_names, "layer06 tensor evidence set differs")
    return fixture


def full_output_comparison_contract() -> dict[str, Any]:
    return {
        "actual_output": str(TRANSACTION7 / "position003/raw/final.hex"),
        "dtype": "FP16",
        "elements": 896,
        "comparison": "element-by-element exact uint16 equality",
        "oracle": "independent exact integer W4A16 oracle",
        "required_array_equal": True,
        "required_integer_mismatches": 0,
        "required_result": {
            "exact_integer_oracle_match": True,
            "natural_rtl_terminal": True,
            "output_hidden_elements": 896,
            "output_state_position": 4,
        },
    }

'''


def common_transform(source: str) -> str:
    placeholders = (
        ("transactions007_025_absent", "__CURRENT_FUTURE_ABSENT__"),
        ("transaction007-025", "__CURRENT_FUTURE_RANGE__"),
        ("transactions006_025_executed", "__PARENT_FUTURE_EXECUTED__"),
        ("TRANSACTION6", "__CURRENT_TRANSACTION_UPPER__"),
        ("Transaction6", "__CURRENT_TRANSACTION_TITLE__"),
        ("transaction006", "__CURRENT_TRANSACTION_PADDED__"),
        ("TRANSACTION006", "__CURRENT_TRANSACTION_PADDED_UPPER__"),
        ("transaction-006", "__CURRENT_TRANSACTION_PATH__"),
        ("transaction6", "__CURRENT_TRANSACTION__"),
        ("LAYER05", "__CURRENT_LAYER_PADDED_UPPER__"),
        ("layer05", "__CURRENT_LAYER_PADDED__"),
        ("layer5", "__CURRENT_LAYER__"),
        ('"MODEL24_RTL_LAYER_INDEX=5"', '"__CURRENT_COMPILE_LAYER__"'),
        ("GENERATION7", "__OUTPUT_GENERATION_UPPER__"),
        ("generation-0000000007", "__OUTPUT_GENERATION_PATH__"),
        ("generation7", "__OUTPUT_GENERATION__"),
        ("cursor7", "__OUTPUT_CURSOR__"),
    )
    for old, new in placeholders:
        source = replace(source, old, new, f"placeholder {old}", minimum=0)

    substitutions = (
        ("transaction005", "transaction006"),
        ("TRANSACTION005", "TRANSACTION006"),
        ("transaction-005", "transaction-006"),
        ("transaction5", "transaction6"),
        ("layer04", "layer05"),
        ("layer4", "layer5"),
        ("GENERATION6", "GENERATION7"),
        ("generation-0000000006", "generation-0000000007"),
        ("generation6", "generation7"),
        ("cursor6", "cursor7"),
        ("checkpoint005", "checkpoint006"),
        ("checkpoint5", "checkpoint6"),
        ("__CURRENT_TRANSACTION_UPPER__", "TRANSACTION7"),
        ("__CURRENT_TRANSACTION_TITLE__", "Transaction7"),
        ("__CURRENT_TRANSACTION_PADDED__", "transaction007"),
        ("__CURRENT_TRANSACTION_PADDED_UPPER__", "TRANSACTION007"),
        ("__CURRENT_TRANSACTION_PATH__", "transaction-007"),
        ("__CURRENT_TRANSACTION__", "transaction7"),
        ("__CURRENT_LAYER_PADDED_UPPER__", "LAYER06"),
        ("__CURRENT_LAYER_PADDED__", "layer06"),
        ("__CURRENT_LAYER__", "layer6"),
        ('"__CURRENT_COMPILE_LAYER__"', '"MODEL24_RTL_LAYER_INDEX=6"'),
        ("__OUTPUT_GENERATION_UPPER__", "GENERATION8"),
        ("__OUTPUT_GENERATION_PATH__", "generation-0000000008"),
        ("__OUTPUT_GENERATION__", "generation8"),
        ("__OUTPUT_CURSOR__", "cursor8"),
        ("__PARENT_FUTURE_EXECUTED__", "transactions007_025_executed"),
        ("__CURRENT_FUTURE_ABSENT__", "transactions008_025_absent"),
        ("__CURRENT_FUTURE_RANGE__", "transaction008-025"),
        ("transactions000_005", "transactions000_006"),
        ("transaction000-005", "transaction000-006"),
        ('MISSION_ID = "12fb90389c40"', f'MISSION_ID = "{MISSION_ID}"'),
        ('PARENT_MISSION_ID = "d560dc6d3138"', f'PARENT_MISSION_ID = "{PARENT_MISSION_ID}"'),
        ("TRANSACTION_INDEX = 6", "TRANSACTION_INDEX = 7"),
        ("LAYER_INDEX = 5", "LAYER_INDEX = 6"),
        ("START_CURSOR = 6", "START_CURSOR = 7"),
        ("EXIT_CURSOR = 7", "EXIT_CURSOR = 8"),
        ("[*range(6), *range(7, 26)]", "[*range(7), *range(8, 26)]"),
        ("range(6, 26)", "range(7, 26)"),
        ("range(6)", "range(7)"),
        ('"required_parent_checkpoint_index": 5', '"required_parent_checkpoint_index": 6'),
        ('required_parent_checkpoint_index") == 5', 'required_parent_checkpoint_index") == 6'),
        ('checkpoint6.get("transaction_index") == 5', 'checkpoint6.get("transaction_index") == 6'),
        ('== 5\n        and descriptor.get("inputs"', '== 6\n        and descriptor.get("inputs"'),
        ('{"transaction_index": 7}', '{"transaction_index": 8}'),
        ('{"layer_index": 6}', '{"layer_index": 7}'),
        ('"authoritative_generation": 5,\n                    "required_parent_checkpoint_index": 4,',
         '"authoritative_generation": 6,\n                    "required_parent_checkpoint_index": 5,'),
        ('"MODEL24_RTL_LAYER_INDEX=4"', '"MODEL24_RTL_LAYER_INDEX=5"'),
        ('__setitem__(2, "4")', '__setitem__(2, "5")'),
        ('{"permitted_transaction_indices": [6, 7]}',
         '{"permitted_transaction_indices": [7, 8]}'),
        (
            "generation6-cursor6-checkpoint005-transaction006-layer05-r3",
            "generation7-cursor7-checkpoint006-transaction007-layer06-r1",
        ),
        ("transaction6-layer5-continuation-package-r3", "transaction7-layer6-continuation-package-r1"),
        ("transaction6-layer5-continuation-review-r3", "transaction7-layer6-continuation-review-r1"),
        ("transaction6-layer5-continuation-authority-r3", "transaction7-layer6-continuation-authority-r1"),
        ("transaction6-layer5-preparation-r1", "transaction7-layer6-preparation-r1"),
        ("emit_transaction6_layer5_executor_review.py", "emit_transaction7_layer6_executor_review.py"),
        ("prepare_transaction6_layer5_executor.py", "prepare_transaction7_layer6_executor.py"),
        ("generation6/cursor6/checkpoint005", "generation7/cursor7/checkpoint006"),
        ("generation6/cursor6", "generation7/cursor7"),
        ("generation=6 cursor=6 checkpoint=005", "generation=7 cursor=7 checkpoint=006"),
        ("generation=6 cursor=6", "generation=7 cursor=7"),
        ("generation7=0", "generation8=0"),
        ("generation=7 cursor=7", "generation=8 cursor=8"),
        ("generation6 parent", "generation7 parent"),
        ("generation6 ledger", "generation7 ledger"),
        ("generation6 manifest", "generation7 manifest"),
        ("fixed generation6", "fixed generation7"),
        ("generations3-6", "generations3-7"),
        ("authenticated generation6 parent acceptance", "authenticated generation7 parent acceptance"),
        ("d560dc6d3138", PARENT_MISSION_ID),
        (
            "/handoffs/51e05603a20c/latest.json",
            "/handoffs/51e05603a20c/latest.json",
        ),
        ("8191adb23a56b8ddc463595bdfbd4c414550cfbe8a5c19c11a633f00f255168c",
         FIXED_SHA256[PARENT_HANDOFF]),
        ("b6d400df449af5699ce3c84ad001d8e687b54746b3be348ad354b9cc415e902d",
         FIXED_SHA256[PARENT_REVIEW]),
        ("d3042e1e75afebb7996aa618390f5dc8dfada73bfcdae028189d19d100fb0e47",
         FIXED_SHA256[POINTER]),
        ("4991277d157154eb564f6ebf7a442fa9d7d1375c58025b1bef22d477aaf1981c",
         FIXED_SHA256[GENERATION7 / "generation-manifest.json"]),
        ("73ae94073b546410f60fae32bcdcf00b99b2819c5a2ca16e9ffa6613bf4892c0",
         FIXED_SHA256[GENERATION7 / "ledger.json"]),
        ("4145c04113658114bfe0e428661212914231465baf4e221737d16d9e6821f896",
         FIXED_SHA256[GENERATION7 / "checkpoints/transaction-006.json"]),
        ("0b35421e91650d2c131116bade95af38d90230cecf5d78ca6abdc99890d0f865",
         FIXED_SHA256[TRANSACTIONS / "transaction-006/position004.state"]),
    )
    for old, new in substitutions:
        source = replace(source, old, new, f"substitution {old}", minimum=0)

    source = source.replace(
        "generation7-cursor7-checkpoint006-transaction007-layer06-r3",
        "generation7-cursor7-checkpoint006-transaction007-layer06-r1",
    )
    source = source.replace(
        "transaction7-layer6-continuation-package-r3",
        "transaction7-layer6-continuation-package-r1",
    )
    source = source.replace(
        "transaction7-layer6-continuation-review-r3",
        "transaction7-layer6-continuation-review-r1",
    )
    source = source.replace(
        "transaction7-layer6-continuation-authority-r3",
        "transaction7-layer6-continuation-authority-r1",
    )
    source = replace(
        source,
        'GENERATION5 = GENERATIONS / "generation-0000000005"\n'
        'GENERATION7 = GENERATIONS / "generation-0000000007"',
        'GENERATION5 = GENERATIONS / "generation-0000000005"\n'
        'GENERATION6 = GENERATIONS / "generation-0000000006"\n'
        'GENERATION7 = GENERATIONS / "generation-0000000007"',
        "preserved generation6 constant",
    )
    source = source.replace(
        "(GENERATION3, GENERATION4, GENERATION5, GENERATION7)",
        "(GENERATION3, GENERATION4, GENERATION5, GENERATION6, GENERATION7)",
    )
    source = replace(
        source,
        '"--layer-index",\n        "5",',
        '"--layer-index",\n        "6",',
        "simulation layer",
    )
    source = replace(
        source,
        '"--transaction-index",\n            "6",',
        '"--transaction-index",\n            "7",',
        "launch transaction",
        minimum=0,
    )
    source = replace(
        source,
        '"transaction-007"\n                    )',
        '"transaction-008"\n                    )',
        "hostile output namespace",
        minimum=0,
    )
    source = replace(
        source,
        'PARENT_HANDOFF = Path(\n'
        '    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"\n'
        '    "handoffs/51e05603a20c/latest.json"\n'
        ')\n'
        'PARENT_REVIEW_DECISION = PARENT_HANDOFF.with_name("round-0001.json")',
        'PARENT_HANDOFF = Path(\n'
        '    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"\n'
        '    "handoffs/51e05603a20c/latest.json"\n'
        ')\n'
        'PARENT_REVIEW_DECISION = PARENT_HANDOFF.with_name("round-0001.json")\n'
        'MANAGER_MISSION = Path(\n'
        '    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/"\n'
        '    "handoffs/cd582996d710/mission.json"\n'
        ')',
        "Manager mission constant",
    )
    source = replace(
        source,
        f'    PARENT_REVIEW_DECISION: "{FIXED_SHA256[PARENT_REVIEW]}",',
        f'    PARENT_REVIEW_DECISION: "{FIXED_SHA256[PARENT_REVIEW]}",\n'
        f'    MANAGER_MISSION: "{FIXED_SHA256[MISSION]}",',
        "Manager mission hash",
    )
    source = replace(
        source,
        "\ndef tree_digest(",
        manager_validation_source() + fixture_validation_source() + "def tree_digest(",
        "shared validation functions",
    )
    source = replace(
        source,
        '    authenticate(descriptor["inputs"]["fixture_manifest"], "layer06 fixture")',
        '    validate_layer06_fixture(descriptor["inputs"]["fixture_manifest"])',
        "layer06 fixture validation",
        minimum=0,
    )
    source = replace(
        source,
        '    authenticate(\n'
        '        descriptor["inputs"]["fixture_manifest"],\n'
        '        "official layer06 fixture",\n'
        '    )',
        '    validate_layer06_fixture(descriptor["inputs"]["fixture_manifest"])',
        "official layer06 fixture validation",
        minimum=0,
    )
    source = replace(
        source,
        'and document.get("mission_id") == MISSION_ID\n'
        '        and document.get("parent_mission_id") == PARENT_MISSION_ID',
        'and document.get("mission_id") == MISSION_ID\n'
        '        and document.get("package_id") == PACKAGE_ID\n'
        '        and document.get("parent_mission_id") == PARENT_MISSION_ID',
        "sealed package identity validation",
    )
    source = replace(
        source,
        "def validate_parent_acceptance() -> dict[str, Any]:\n",
        "def validate_parent_acceptance() -> dict[str, Any]:\n"
        "    validate_manager_mission()\n",
        "engineer Manager validation",
        minimum=0,
    )
    source = replace(
        source,
        "def validate_live_parent() -> None:\n",
        "def validate_live_parent() -> None:\n"
        "    validate_manager_mission()\n",
        "Reviewer Manager validation",
        minimum=0,
    )
    return source


def engineer_transform(source: str) -> str:
    source = common_transform(source)
    source = replace(
        source,
        '            "mission_id": MISSION_ID,\n'
        '            "parent_mission_id": PARENT_MISSION_ID,',
        '            "mission_id": MISSION_ID,\n'
        '            "package_id": PACKAGE_ID,\n'
        '            "parent_mission_id": PARENT_MISSION_ID,',
        "package identity field",
    )
    source = replace(
        source,
        'ROOT\n'
        '    / "build/model24_selected_token_position3_transaction6_packages"\n'
        '    / "generation5-cursor5-checkpoint004-transaction006-layer05-r1"\n'
        '    / "source-manifest.json"',
        'RUNTIME / "transaction6-layer5-continuation-package-r4" / "source-manifest.json"',
        "accepted transaction006 source closure path",
    )
    source = replace(
        source,
        'ROOT\n    / "build/model24_selected_token_position3_transaction7_packages"\n'
        '    / PACKAGE_ID',
        'RUNTIME / "transaction7-layer6-continuation-package-r1"',
        "transaction007 package path",
        minimum=0,
    )
    source = replace(
        source,
        'RUNTIME\n    / "transaction7-layer6-preparation-r1"\n'
        '    / "emit_transaction7_layer6_executor_review.py"',
        f'Path("{REVIEWER_SOURCE}")',
        "Reviewer source path",
    )
    accepted_kind = '"ace3_position3_transaction6_layer5_exact_source_toolchain_closure"'
    require(
        source.count(accepted_kind) == 1
        and source.count(
            '"ace3_position3_transaction7_layer6_exact_source_toolchain_closure"'
        )
        == 2,
        "source closure kind count differs",
    )
    source = replace(
        source,
        'sources["accepted_transaction006_source_closure"] = file_record(',
        'sources["accepted_transaction006_source_closure"] = file_record(',
        "accepted source closure label",
        minimum=0,
    )
    source = replace(
        source,
        '"official_frozen_evidence": baseline["official_frozen_evidence"],',
        '"official_frozen_evidence": baseline["official_frozen_evidence"],\n'
        '            "full_output_comparison": full_output_comparison_contract(),\n'
        '            "manager_mission": file_record(MANAGER_MISSION),',
        "manifest explicit contracts",
    )
    source = replace(
        source,
        '        "official_frozen_evidence": official_frozen_evidence(),',
        '        "official_frozen_evidence": official_frozen_evidence(),\n'
        '        "full_output_comparison": full_output_comparison_contract(),\n'
        '        "manager_mission": file_record(MANAGER_MISSION),',
        "baseline explicit contracts",
    )
    source = replace(
        source,
        'and document.get("official_frozen_evidence") == official_frozen_evidence()',
        'and document.get("official_frozen_evidence") == official_frozen_evidence()\n'
        '        and document.get("full_output_comparison")\n'
        '        == full_output_comparison_contract()\n'
        '        and document.get("manager_mission") == file_record(MANAGER_MISSION)',
        "manifest contract validation",
    )
    source = replace(
        source,
        '"transaction=5 layer=4 generation=8 cursor=8"',
        '"transaction=7 layer=6 generation=8 cursor=8"',
        "terminal status",
        minimum=0,
    )
    ast.parse(source)
    return source


def reviewer_transform(source: str) -> str:
    source = common_transform(source)
    source = replace(
        source,
        '"official_frozen_evidence") == expected_official_evidence()',
        '"official_frozen_evidence") == expected_official_evidence()\n'
        '        and document.get("full_output_comparison")\n'
        '        == full_output_comparison_contract()\n'
        '        and document.get("manager_mission") == file_record(MANAGER_MISSION)',
        "Reviewer manifest contract validation",
    )
    source = source.replace(
        'and baseline.get("official_frozen_evidence") == expected_official_evidence()\n'
        '        and document.get("full_output_comparison")\n'
        '        == full_output_comparison_contract()\n'
        '        and document.get("manager_mission") == file_record(MANAGER_MISSION)',
        'and baseline.get("official_frozen_evidence") == expected_official_evidence()\n'
        '        and baseline.get("full_output_comparison")\n'
        '        == full_output_comparison_contract()\n'
        '        and baseline.get("manager_mission") == file_record(MANAGER_MISSION)',
    )
    extra_mutations = '''
        (
            "wrong-fixture-manifest",
            lambda item: item["official_frozen_evidence"][
                "layer06_fixture_manifest"
            ].update({"sha256": "0" * 64}),
        ),
        (
            "wrong-kv-parent",
            lambda item: item["official_frozen_evidence"][
                "layer06_position2_kv_parent"
            ].update({"sha256": "0" * 64}),
        ),
        (
            "wrong-launch-argv",
            lambda item: item["launch"]["argv"].__setitem__(-1, "8"),
        ),
        (
            "wrong-generation8-namespace",
            lambda item: item.update(
                {"generation8_output_namespace": str(GENERATION7)}
            ),
        ),
        (
            "partial-output-contract",
            lambda item: item["full_output_comparison"].update(
                {"elements": 895}
            ),
        ),
        (
            "mismatch-tolerant-output-contract",
            lambda item: item["full_output_comparison"].update(
                {"required_integer_mismatches": 1}
            ),
        ),
        (
            "wrong-actual-output",
            lambda item: item["full_output_comparison"].update(
                {"actual_output": str(TRANSACTION7 / "wrong.hex")}
            ),
        ),
        (
            "authority-created",
            lambda item: item.update({"authority_created": True}),
        ),
        (
            "wrong-package-identity",
            lambda item: item.update({"package_id": "wrong"}),
        ),
        (
            "wrong-manager-mission",
            lambda item: item["manager_mission"].update(
                {"sha256": "0" * 64}
            ),
        ),
'''
    source = replace(
        source,
        '        (\n'
        '            "premature-authority",\n'
        '            lambda item: item.update({"execution_authorized": True}),\n'
        '        ),',
        '        (\n'
        '            "premature-authority",\n'
        '            lambda item: item.update({"execution_authorized": True}),\n'
        '        ),\n'
        + extra_mutations.rstrip(),
        "extended hostile mutations",
    )
    source = replace(
        source,
        "    return controls\n\n\ndef assess_package",
        '''    seal = load_json(package / "package-seal.json")
    member = copy.deepcopy(seal["members"]["executor"])
    member["sha256"] = "0" * 64
    try:
        authenticate(member, "mutated sealed executor")
    except ReviewError:
        controls.append("sealed-member-hash")
    else:
        raise ReviewError("mutated sealed executor hash accepted")

    scratch = package.parent / ".transaction7-review-hostile-control"
    require(not scratch.exists(), "hostile-control scratch path exists")
    scratch.mkdir(mode=0o700)
    try:
        duplicate = scratch / "duplicate.json"
        duplicate.write_text('{"value": 1, "value": 2}\\n', encoding="ascii")
        try:
            load_json(duplicate)
        except ReviewError:
            controls.append("duplicate-json-key")
        else:
            raise ReviewError("duplicate JSON key accepted")
        target = scratch / "target.json"
        target.write_text('{"value": 1}\\n', encoding="ascii")
        symlink = scratch / "symlink.json"
        symlink.symlink_to(target)
        try:
            load_json(symlink)
        except ReviewError:
            controls.append("symlink-json")
        else:
            raise ReviewError("symlink JSON accepted")
    finally:
        for path in (scratch / "symlink.json", scratch / "target.json", scratch / "duplicate.json"):
            if path.exists() or path.is_symlink():
                path.unlink()
        scratch.rmdir()
    return controls


def assess_package''',
        "filesystem hostile controls",
    )
    source = replace(
        source,
        '        "permitted_launch": load_json(package / "package-manifest.json")["launch"],',
        '        "permitted_launch": load_json(package / "package-manifest.json")["launch"],\n'
        '        "full_output_comparison": full_output_comparison_contract(),\n'
        '        "layer06_authenticated_tensor_count": 26,\n'
        '        "manager_mission": file_record(MANAGER_MISSION),',
        "review explicit contracts",
    )
    ast.parse(source)
    return source


def write_new(path: Path, payload: bytes, mode: int = 0o400) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        mode,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def import_source(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    require(
        specification is not None and specification.loader is not None,
        f"module import failed: {path}",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def validate_parent_and_absence() -> None:
    for path in FIXED_SHA256:
        require_fixed(path)
    mission = load_json(MISSION)
    pointer = load_json(POINTER)
    generation = load_json(GENERATION7 / "generation-manifest.json")
    ledger = load_json(GENERATION7 / "ledger.json")
    checkpoint = load_json(GENERATION7 / "checkpoints/transaction-006.json")
    require(
        mission.get("kind") == "mission_context"
        and mission.get("mission_id") == MISSION_ID
        and mission.get("node_key") == "tx007-layer06-package-review"
        and pointer.get("kind")
        == "ace3_position3_transaction6_generation7_authoritative_pointer"
        and pointer.get("status") == "COMMITTED"
        and pointer.get("generation") == 7
        and pointer.get("generation_manifest")
        == {
            "path": str(GENERATION7 / "generation-manifest.json"),
            "bytes": (GENERATION7 / "generation-manifest.json").stat().st_size,
            "sha256": FIXED_SHA256[GENERATION7 / "generation-manifest.json"],
        }
        and generation.get("kind")
        == "ace3_transaction6_publication_recovery_generation7_state"
        and generation.get("status") == "PREPARED"
        and generation.get("generation") == 7
        and generation.get("transaction006_retry_replay_resume") is False
        and generation.get("transactions007_025_executed") is False
        and ledger.get("state_generation") == 7
        and ledger.get("completed_transaction_count") == 7
        and ledger.get("next_transaction_index") == 7
        and checkpoint.get("kind") == "ace3_position3_transaction_completion"
        and checkpoint.get("status") == "COMPLETE"
        and checkpoint.get("transaction_index") == 6
        and checkpoint.get("result", {}).get("exact_integer_oracle_match") is True
        and checkpoint.get("result", {}).get("output_hidden_elements") == 896,
        "generation7/checkpoint006/cursor7 parent differs",
    )
    require(
        not TRANSACTION7.exists()
        and all(
            not (TRANSACTIONS / f"transaction-{index:03d}").exists()
            for index in range(8, 26)
        )
        and not GENERATION8.exists()
        and not GENERATION8_STAGING.exists()
        and not FUTURE_ROOT.exists()
        and not AUTHORITY.exists(),
        "transaction007-025, authority, or generation8 namespace exists",
    )


def derived_sources() -> tuple[str, str]:
    validate_parent_and_absence()
    engineer = engineer_transform(BASE_EXECUTOR.read_text(encoding="utf-8"))
    reviewer = reviewer_transform(BASE_REVIEWER.read_text(encoding="utf-8"))
    require(
        MISSION_ID in engineer
        and MISSION_ID in reviewer
        and "validate_layer06_fixture" in engineer
        and "validate_layer06_fixture" in reviewer
        and "full_output_comparison_contract" in engineer
        and "full_output_comparison_contract" in reviewer,
        "derived source bindings are incomplete",
    )
    return engineer, reviewer


def prepare() -> None:
    engineer, reviewer = derived_sources()
    require(not PREPARATION.exists(), f"preparation namespace exists: {PREPARATION}")
    require(not PACKAGE.exists(), f"package namespace exists: {PACKAGE}")
    require(not REVIEW.exists(), f"review namespace exists: {REVIEW}")
    PREPARATION.mkdir(mode=0o700)
    try:
        write_new(ENGINEER_SOURCE, engineer.encode("utf-8"))
        write_new(REVIEWER_SOURCE, reviewer.encode("utf-8"))
        directory = os.open(PREPARATION, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        executor_module = import_source(ENGINEER_SOURCE, "ace3_tx007_layer06_executor")
        require(
            executor_module.prepare_package() == PACKAGE,
            "transaction007 package path differs",
        )
        reviewer_module = import_source(REVIEWER_SOURCE, "ace3_tx007_layer06_reviewer")
        reviewer_module.emit_review(PACKAGE, REVIEW)
    except BaseException:
        if not PACKAGE.exists() and not REVIEW.exists():
            for path in (REVIEWER_SOURCE, ENGINEER_SOURCE):
                if path.exists():
                    path.unlink()
            PREPARATION.rmdir()
        raise
    validate()


def validate() -> None:
    validate_parent_and_absence()
    require(
        PACKAGE.is_dir()
        and not PACKAGE.is_symlink()
        and stat.S_IMODE(PACKAGE.stat().st_mode) & 0o222 == 0,
        "sealed transaction007 package is absent or writable",
    )
    require(REVIEW.is_file() and not REVIEW.is_symlink(), "independent review absent")
    reviewer_module = import_source(
        PACKAGE / "review-emitter.py",
        "ace3_tx007_layer06_sealed_reviewer",
    )
    status, controls, reason = reviewer_module.assess_package(PACKAGE)
    review = load_json(REVIEW)
    require(
        status == "PASS"
        and reason is None
        and len(controls) >= 27
        and review.get("kind")
        == "ace3_position3_transaction7_layer6_independent_review"
        and review.get("status") == "PASS"
        and review.get("mission_id") == MISSION_ID
        and review.get("preparation_participation") is False
        and review.get("execution_import_participation") is False
        and review.get("activity_counters")
        == {
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
        and review.get("transaction007_executed") is False
        and review.get("generation8_published") is False
        and review.get("transactions008_025_absent") is True
        and review.get("this_review_authorizes_execution") is False
        and review.get("full_output_comparison", {}).get("elements") == 896
        and review.get("full_output_comparison", {}).get(
            "required_integer_mismatches"
        )
        == 0
        and review.get("layer06_authenticated_tensor_count") == 26
        and review.get("adversarial_controls") == controls,
        f"independent review differs: {reason}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("preflight", "prepare", "validate"))
    arguments = parser.parse_args()
    if arguments.operation == "preflight":
        derived_sources()
        require(not PREPARATION.exists(), "preparation namespace already exists")
        require(not PACKAGE.exists(), "package namespace already exists")
        require(not REVIEW.exists(), "review namespace already exists")
        print("TRANSACTION007_LAYER06_PACKAGE_PREFLIGHT_PASS")
    elif arguments.operation == "prepare":
        prepare()
        print(
            "TRANSACTION007_LAYER06_PACKAGE_REVIEW_PASS "
            f"package={PACKAGE} review={REVIEW} outputs=896 mismatches=0 "
            "authority=0 model=0 oracle=0 vectors=0 rtl_compile=0 "
            "rtl_simulation=0 transaction=0 generation8=0"
        )
    else:
        validate()
        print(
            "TRANSACTION007_LAYER06_PACKAGE_VALID "
            "generation=7 cursor=7 checkpoint=006 controls>=27 "
            "authority=0 model=0 oracle=0 vectors=0 rtl_compile=0 "
            "rtl_simulation=0 transaction=0 generation8=0"
        )


if __name__ == "__main__":
    try:
        main()
    except (
        PackageError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise SystemExit(f"TRANSACTION007_LAYER06_PACKAGE_REFUSED {error}") from error
