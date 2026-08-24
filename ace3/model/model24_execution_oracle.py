#!/usr/bin/env python3
"""Decoder-independent oracle for the pinned model24 execution contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PARENT_COMMIT = "3cf65b762d928e02e2b64fbba4389e294e1aa2c5"
MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
TENSOR_MAP_SHA256 = (
    "11a03bed8049cd815ac2c37384a7ba15d71d2f69ee397110d1cd443193474624"
)
CONTROL_MAP_SHA256 = (
    "3364dc4c2c585f4687d8ad7943792ca4c44265b85463b91ab2a1c6866690b611"
)
DECODER_SOURCE_SHA256 = (
    "595a2374dec03ee5b6ae85d65758e9e64648342c36b9c8af96c7d3ae70572bd2"
)
DECODER_INTERFACE_SHA256 = (
    "6d3177c5a3fdff424037ace6ce3162085a6c5cb15f15b7771edbf2396198a467"
)
TIED_WEIGHT_SHA256 = (
    "d74257dc547b48be5ae7b93f1c9af072c0c42dbbb85503078e25c59cd09e68d0"
)

PER_LAYER_OPERATIONS = (
    "input_rmsnorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_rope",
    "k_rope",
    "kv_write",
    "kv_read",
    "attention_qk",
    "attention_softmax",
    "attention_value",
    "o_proj",
    "attention_residual_add",
    "post_attention_rmsnorm",
    "gate_proj",
    "up_proj",
    "silu",
    "gated_multiply",
    "down_proj",
    "mlp_residual_add",
)

OPERATION_TENSOR_SUFFIXES: dict[str, tuple[str, ...]] = {
    "input_rmsnorm": ("input_layernorm.weight",),
    "q_proj": (
        "self_attn.q_proj.qweight",
        "self_attn.q_proj.qzeros",
        "self_attn.q_proj.scales",
        "self_attn.q_proj.bias",
    ),
    "k_proj": (
        "self_attn.k_proj.qweight",
        "self_attn.k_proj.qzeros",
        "self_attn.k_proj.scales",
        "self_attn.k_proj.bias",
    ),
    "v_proj": (
        "self_attn.v_proj.qweight",
        "self_attn.v_proj.qzeros",
        "self_attn.v_proj.scales",
        "self_attn.v_proj.bias",
    ),
    "o_proj": (
        "self_attn.o_proj.qweight",
        "self_attn.o_proj.qzeros",
        "self_attn.o_proj.scales",
    ),
    "post_attention_rmsnorm": ("post_attention_layernorm.weight",),
    "gate_proj": (
        "mlp.gate_proj.qweight",
        "mlp.gate_proj.qzeros",
        "mlp.gate_proj.scales",
    ),
    "up_proj": (
        "mlp.up_proj.qweight",
        "mlp.up_proj.qzeros",
        "mlp.up_proj.scales",
    ),
    "down_proj": (
        "mlp.down_proj.qweight",
        "mlp.down_proj.qzeros",
        "mlp.down_proj.scales",
    ),
}

LAYER_DESCRIPTOR_SHA256 = (
    "12c974f9aee6aa0ef84ecf483bac9f26186ff717f9a79ae10f44c332f75a069a",
    "c8a037c0043ededc764f02b14671781ceeb1fb5be3fa6b7f8e114d75a98ad8f4",
    "07b907a2f7a800af011b630ce2a026593f05fbd9447e3f106e8970be7888d916",
    "b2c2e5088b133b7ed99ac6615cc401d835775e356b1e0c56c47ffcf13cd0f000",
    "11b3bbde0e323e62992361dbeac9be0e25c75e92047a9773feee80f03cbeeab3",
    "438b0999de6d7d672877fda658d9f29b7ae031181530e55fe83f2d070239f1e1",
    "2f94cbd039aee46ec65b50c5f6ba339fb3a7a8d26fe14926f299add23bb5e3e7",
    "8de17beec10f3cebf6c355998fde0c2cd74228d03fdf089482026e4284ea6ca6",
    "e5c9b4522a33a7c01f6fe10e7f317a24cf6d26598773298f451c3789a631f757",
    "28f3aad0a3a79a814ea99f4c3edb1653ad97fbfb39dba23ec8525a5236e4cd63",
    "4b1070d1c34b1b572983b26f165f4d3f0c12b5f50aaca2c11e4e797934209a9f",
    "49893e2cdb478ba2f15ef93abac3e76c2be2307f650c4040f69e0614f77f26e0",
    "d5be9c0d3c9880575d3cdcde6f99fadd175d51cd60d52aff11093633e38b1dec",
    "f6aa64c1c0e66cf38e4168d50f7647910ee9aa4ed72c4f8b22ef4f8c117e8a42",
    "097759351e6eefc24d1a145e23d914dc4f3ee18a4205c4a25f6e277efed9ee5e",
    "739ffe2c23ed03ebaabcaf21cb3bcfb0418f2d26e34154a12de1d8ef275604e5",
    "d4ef31966fda5c4600a6978160055841813458a9ffa7ab9e25acc81d009161c5",
    "96b412fa9a8212bc41ac2aac2b58e093de6895ba441969d98619e7f15611224a",
    "81b33427c26bb8c92e038184a4f5350255167f4d8618f66126afa63fa32d41d2",
    "71282897c309636540962c48e65f9eb554c8225329de1708e33f8f3b8e080036",
    "90d923c29e25be17677293712cbb99342a208b860d65dfe75081b1b5c0a8486e",
    "af3c61e73991d0c38a61756beeaab9e1664ceca6376cc48d1a3b2cdd20bb5b36",
    "e9cf2e8e77a07c47bc331eca6d87093ef9ad515c84665c7727740df586571bc4",
    "9c1eef81355729ab3a32f6b21b58a5a17b3f6477f18f8553fe4bd74e7ea1592d",
)


class ContractError(RuntimeError):
    """Raised when an execution contract or vector fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_object_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is not valid JSON: {error}") from error


def load_json_document(path: Path) -> Any:
    return load_json_bytes(path.read_bytes(), str(path))


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class Geometry:
    layers: int
    cache_slots: int
    positions: int
    kv_heads: int
    head_dim: int
    hidden_size: int
    group_size: int
    vocab_size: int

    def validate(self) -> None:
        for name, value in (
            ("layers", self.layers),
            ("cache_slots", self.cache_slots),
            ("positions", self.positions),
            ("kv_heads", self.kv_heads),
            ("head_dim", self.head_dim),
            ("hidden_size", self.hidden_size),
            ("group_size", self.group_size),
            ("vocab_size", self.vocab_size),
        ):
            require(type(value) is int and value > 0, f"{name} must be positive")
        require(
            self.hidden_size % self.group_size == 0,
            "lm_head group geometry is fractional",
        )

    def document(self) -> dict[str, int]:
        return {
            "layers": self.layers,
            "cache_slots": self.cache_slots,
            "positions": self.positions,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "group_size": self.group_size,
            "vocab_size": self.vocab_size,
        }


OFFICIAL_GEOMETRY = Geometry(
    layers=24,
    cache_slots=2,
    positions=128,
    kv_heads=2,
    head_dim=64,
    hidden_size=896,
    group_size=128,
    vocab_size=151936,
)


def layer_bindings() -> list[dict[str, Any]]:
    return [
        {
            "layer_id": layer_id,
            "namespace": f"model.layers.{layer_id}.",
            "descriptor_sha256": descriptor_sha256,
        }
        for layer_id, descriptor_sha256 in enumerate(LAYER_DESCRIPTOR_SHA256)
    ]


def residual_handoffs(geometry: Geometry) -> list[dict[str, Any]]:
    geometry.validate()
    return [
        {
            "layer_id": layer_id,
            "input": (
                "embedding.output"
                if layer_id == 0
                else f"residual.layer.{layer_id - 1}.output"
            ),
            "attention_result": f"residual.layer.{layer_id}.attention",
            "output": f"residual.layer.{layer_id}.output",
        }
        for layer_id in range(geometry.layers)
    ]


def execution_contract_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ace3_model24_decoder_independent_execution",
        "parent_commit": PARENT_COMMIT,
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "tensor_map": {
                "path": "ace3/contracts/model24_tensor_map.json",
                "sha256": TENSOR_MAP_SHA256,
            },
            "control_map": {
                "path": "ace3/contracts/model24_control.json",
                "sha256": CONTROL_MAP_SHA256,
            },
        },
        "decoder_snapshot_compatibility": {
            "source": {
                "path": "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
                "sha256": DECODER_SOURCE_SHA256,
            },
            "interface": {
                "path": "ace3/tb/ace3_decoder_width_boundary_tb.sv",
                "sha256": DECODER_INTERFACE_SHA256,
            },
            "review_provenance": "decoder_wall_repair_review_cf03",
            "status": "hash compatibility only; not decoder acceptance",
        },
        "numeric_profile": {
            "projection": "native asymmetric packed INT4 AWQ W4A16 G128 GEMM",
            "packed_nibble_order": [0, 4, 1, 5, 2, 6, 3, 7],
            "qzero_adjustment": "none",
            "scales": "FP16",
            "activations": "FP16",
            "kv": "FP16",
        },
        "layers": {
            "count": 24,
            "bindings": layer_bindings(),
            "ownership": "a layer consumes only its bound namespace",
        },
        "schedule": {
            "event_count": 483,
            "initial": "embedding_lookup",
            "per_layer": list(PER_LAYER_OPERATIONS),
            "final": ["final_rmsnorm", "lm_head"],
            "ordering": "strict; any missing, duplicate, reordered, or extra event faults",
        },
        "kv_cache": {
            "format": "FP16",
            "geometry": OFFICIAL_GEOMETRY.document(),
            "owner_key": ["cache_slot", "layer_id", "K_or_V"],
            "address_order": [
                "K_or_V",
                "cache_slot",
                "layer_id",
                "position",
                "kv_head",
                "dimension",
            ],
            "bounds": "reject without masking, wrapping, clamping, or aliasing",
            "clear": "invalidate every slot and layer owner",
        },
        "residual_handoff": {
            "format": "FP16",
            "bindings": residual_handoffs(OFFICIAL_GEOMETRY),
            "rule": "layer output is the sole residual input of the next layer",
        },
        "final_interfaces": {
            "rmsnorm": {
                "tensor": "model.norm.weight",
                "dtype": "F16",
                "features": 896,
                "epsilon": 1e-6,
            },
            "lm_head": {
                "checkpoint_tensor": "lm_head.weight",
                "checkpoint_dtype": "F16",
                "tied_to": "model.embed_tokens.weight",
                "tied_value_sha256": TIED_WEIGHT_SHA256,
                "input_features": 896,
                "output_logits": 151936,
                "execution_group_size": 128,
                "execution_groups_per_logit": 7,
                "execution_stream": (
                    "grouped native asymmetric AWQ W4A16 G128 interface"
                ),
                "storage_boundary": (
                    "grouped execution interface only; no AWQ-packed lm_head "
                    "checkpoint-storage claim"
                ),
            },
            "argmax": {
                "input_order": "token_id ascending from 0 through 151935",
                "selection": "greatest valid logit",
                "tie_break": "lowest token_id",
                "completion": "exactly 151936 logits accepted",
                "invalid": "fault; no token result",
            },
        },
        "reset_and_error": {
            "fault_policy": "latched fail-closed until reset or clear",
            "suppressed_while_faulted": [
                "event_accept",
                "kv_write",
                "residual_handoff",
                "logit_accept",
                "argmax_valid",
                "done",
            ],
            "reset": "return to embedding_lookup and invalidate all KV ownership",
        },
        "verification_boundary": {
            "established": [
                "24 namespace bindings and 483-event control trajectory",
                "per-layer FP16 KV ownership and residual lineage",
                "final RMSNorm, tied grouped lm_head, and deterministic argmax interfaces",
                "fixed parent, reviewed map, and decoder snapshot hash compatibility",
            ],
            "excluded": [
                "decoder RTL implementation or execution",
                "official numerical model execution or dialogue",
                "latency, throughput, synthesis, PPA, FPGA, or silicon",
                "decoder acceptance",
            ],
        },
    }


def vector_bindings_document(contract_sha256: str) -> dict[str, Any]:
    require(
        len(contract_sha256) == 64,
        "execution contract SHA256 must contain 64 hex characters",
    )
    return {
        "schema_version": 1,
        "kind": "ace3_model24_execution_vector_bindings",
        "generator": {
            "algorithm": "model24-execution-v1",
            "seed": 240483,
            "serialization": "UTF-8 canonical JSON, sorted keys, compact separators, LF",
        },
        "inputs": {
            "parent_commit": PARENT_COMMIT,
            "execution_contract_sha256": contract_sha256,
            "tensor_map_sha256": TENSOR_MAP_SHA256,
            "control_map_sha256": CONTROL_MAP_SHA256,
            "decoder_source_sha256": DECODER_SOURCE_SHA256,
            "decoder_interface_sha256": DECODER_INTERFACE_SHA256,
        },
        "artifact_set": [
            "manifest.json",
            "official_schedule.json",
            "small_geometry.json",
        ],
        "validation": {
            "exact_artifact_set": True,
            "duplicate_json_keys": "reject",
            "recompute_vectors": True,
            "mutation_policy": "reject any byte or semantic drift",
        },
    }


def _first_difference(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
        return f"{path}: expected {type(expected).__name__}, got {type(actual).__name__}"
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            return f"{path}: key mismatch missing={missing} extra={extra}"
        for key in expected:
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: expected {len(expected)} items, got {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            difference = _first_difference(
                expected_item,
                actual_item,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return None


def require_document(expected: Any, actual: Any, label: str) -> None:
    difference = _first_difference(expected, actual)
    require(difference is None, f"{label} mismatch: {difference}")


def validate_reviewed_maps(tensor_payload: bytes, control_payload: bytes) -> None:
    require(
        sha256_bytes(tensor_payload) == TENSOR_MAP_SHA256,
        "reviewed tensor map SHA256 mismatch",
    )
    require(
        sha256_bytes(control_payload) == CONTROL_MAP_SHA256,
        "reviewed control map SHA256 mismatch",
    )
    tensor_map = load_json_bytes(tensor_payload, "tensor map")
    control = load_json_bytes(control_payload, "control map")
    require(tensor_map.get("kind") == "ace3_model24_official_tensor_address_map", "tensor map kind")
    require(control.get("kind") == "ace3_model24_layer_and_kv_control", "control map kind")
    require(
        tensor_map["provenance"]["model_revision"] == MODEL_REVISION,
        "tensor map revision mismatch",
    )
    require(
        control["model_binding"]["revision"] == MODEL_REVISION,
        "control map revision mismatch",
    )
    expected_map_bindings = [
        {
            "layer_id": item["layer_id"],
            "namespace": item["namespace"],
            "tensor_count": 26,
            "descriptor_sha256": item["descriptor_sha256"],
        }
        for item in layer_bindings()
    ]
    require_document(
        expected_map_bindings,
        tensor_map["layer_namespaces"],
        "tensor namespace bindings",
    )
    require_document(
        expected_map_bindings,
        control["layer_namespaces"]["bindings"],
        "control namespace bindings",
    )
    require(
        control["schedule"]["operation_count"] == 483,
        "control event count mismatch",
    )
    require_document(
        list(PER_LAYER_OPERATIONS),
        control["schedule"]["per_layer_operation_order"],
        "control per-layer schedule",
    )
    require(
        control["kv_cache"]["format"] == "FP16"
        and control["kv_cache"]["limits"]["layers"] == 24,
        "control KV ownership mismatch",
    )
    require(
        control["final_projection"]["rmsnorm"]["tensor"] == "model.norm.weight",
        "final RMSNorm binding mismatch",
    )
    require(
        control["final_projection"]["lm_head"]["tied_to"]
        == "model.embed_tokens.weight",
        "lm_head tie mismatch",
    )


def validate_execution_contract(
    contract: Any,
    tensor_payload: bytes,
    control_payload: bytes,
) -> None:
    require(isinstance(contract, dict), "execution contract root must be an object")
    require_document(
        execution_contract_document(),
        contract,
        "execution contract",
    )
    validate_reviewed_maps(tensor_payload, control_payload)


def validate_vector_bindings(bindings: Any, contract_sha256: str) -> None:
    require(isinstance(bindings, dict), "vector bindings root must be an object")
    require_document(
        vector_bindings_document(contract_sha256),
        bindings,
        "execution vector bindings",
    )


def require_parent_commit(repository_root: Path) -> None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD^"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    require(
        result.stdout.strip() == PARENT_COMMIT,
        f"publication parent mismatch: expected {PARENT_COMMIT}, got {result.stdout.strip()}",
    )


def kv_owner(layer_id: int, kind: str, geometry: Geometry = OFFICIAL_GEOMETRY) -> str:
    geometry.validate()
    require(type(layer_id) is int and 0 <= layer_id < geometry.layers, "layer_id out of range")
    require(kind in ("K", "V"), "kind must be K or V")
    return f"kv.layer.{layer_id}.{kind}"


def kv_address(
    cache_slot: int,
    layer_id: int,
    position: int,
    kv_head: int,
    dimension: int,
    kind: str,
    geometry: Geometry = OFFICIAL_GEOMETRY,
) -> int:
    geometry.validate()
    for name, value, limit in (
        ("cache_slot", cache_slot, geometry.cache_slots),
        ("layer_id", layer_id, geometry.layers),
        ("position", position, geometry.positions),
        ("kv_head", kv_head, geometry.kv_heads),
        ("dimension", dimension, geometry.head_dim),
    ):
        require(
            type(value) is int and 0 <= value < limit,
            f"{name} outside supported range [0, {limit})",
        )
    require(kind in ("K", "V"), "kind must be K or V")
    entries_per_layer = geometry.positions * geometry.kv_heads * geometry.head_dim
    entries_per_slot = geometry.layers * entries_per_layer
    entries_per_bank = geometry.cache_slots * entries_per_slot
    return (
        (0 if kind == "K" else entries_per_bank)
        + cache_slot * entries_per_slot
        + layer_id * entries_per_layer
        + position * geometry.kv_heads * geometry.head_dim
        + kv_head * geometry.head_dim
        + dimension
    )


def expected_schedule(geometry: Geometry = OFFICIAL_GEOMETRY) -> list[dict[str, Any]]:
    geometry.validate()
    events: list[dict[str, Any]] = []

    def append(
        operation: str,
        layer_id: int | None,
        tensor_names: list[str],
    ) -> None:
        event: dict[str, Any] = {
            "ordinal": len(events),
            "operation": operation,
            "layer_id": layer_id,
            "tensor_names": tensor_names,
        }
        if layer_id is not None:
            event["namespace"] = f"model.layers.{layer_id}."
        if operation in ("kv_write", "kv_read"):
            event["kv_owners"] = [
                kv_owner(layer_id, "K", geometry),
                kv_owner(layer_id, "V", geometry),
            ]
        if operation == "attention_residual_add":
            handoff = residual_handoffs(geometry)[layer_id]
            event["residual_input"] = handoff["input"]
            event["residual_output"] = handoff["attention_result"]
        if operation == "mlp_residual_add":
            handoff = residual_handoffs(geometry)[layer_id]
            event["residual_input"] = handoff["attention_result"]
            event["residual_output"] = handoff["output"]
        events.append(event)

    append("embedding_lookup", None, ["model.embed_tokens.weight"])
    for layer_id in range(geometry.layers):
        prefix = f"model.layers.{layer_id}."
        for operation in PER_LAYER_OPERATIONS:
            append(
                operation,
                layer_id,
                [
                    prefix + suffix
                    for suffix in OPERATION_TENSOR_SUFFIXES.get(operation, ())
                ],
            )
    append("final_rmsnorm", None, ["model.norm.weight"])
    append("lm_head", None, ["lm_head.weight"])
    require(
        len(events) == 1 + geometry.layers * len(PER_LAYER_OPERATIONS) + 2,
        "schedule length mismatch",
    )
    return events


class ExecutionMachine:
    """Strict event acceptor with a latched fail-closed fault."""

    def __init__(self, geometry: Geometry = OFFICIAL_GEOMETRY) -> None:
        self.geometry = geometry
        self._expected = expected_schedule(geometry)
        self.reset()

    def reset(self) -> None:
        self.cursor = 0
        self.faulted = False
        self.done = False

    def clear(self) -> None:
        self.reset()

    def accept(self, event: Mapping[str, Any]) -> None:
        require(not self.faulted, "execution machine is faulted")
        require(not self.done, "execution machine already completed")
        if dict(event) != self._expected[self.cursor]:
            self.faulted = True
            raise ContractError(f"event {self.cursor} violates strict schedule")
        self.cursor += 1
        self.done = self.cursor == len(self._expected)


def validate_trajectory(
    events: Iterable[Mapping[str, Any]],
    geometry: Geometry = OFFICIAL_GEOMETRY,
) -> None:
    machine = ExecutionMachine(geometry)
    for event in events:
        machine.accept(event)
    require(machine.done, "trajectory ended before completion")


def grouped_lm_head_interface(geometry: Geometry = OFFICIAL_GEOMETRY) -> dict[str, Any]:
    geometry.validate()
    return {
        "input_features": geometry.hidden_size,
        "output_logits": geometry.vocab_size,
        "group_size": geometry.group_size,
        "groups_per_logit": geometry.hidden_size // geometry.group_size,
        "output_order": list(range(geometry.vocab_size)),
        "group_order": list(range(geometry.hidden_size // geometry.group_size)),
        "tie_break": "lowest token_id",
    }


def argmax_first(logits: Sequence[int]) -> int:
    require(len(logits) > 0, "argmax requires at least one logit")
    require(all(type(value) is int for value in logits), "argmax test logits must be integers")
    best_index = 0
    best_value = logits[0]
    for index, value in enumerate(logits[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def _illegal_kv_cases(geometry: Geometry) -> list[dict[str, Any]]:
    candidates = [
        (-1, 0, 0, 0, 0, "K"),
        (geometry.cache_slots, 0, 0, 0, 0, "K"),
        (0, -1, 0, 0, 0, "K"),
        (0, geometry.layers, 0, 0, 0, "K"),
        (0, 0, -1, 0, 0, "K"),
        (0, 0, geometry.positions, 0, 0, "K"),
        (0, 0, 0, -1, 0, "K"),
        (0, 0, 0, geometry.kv_heads, 0, "K"),
        (0, 0, 0, 0, -1, "K"),
        (0, 0, 0, 0, geometry.head_dim, "K"),
        (0, 0, 0, 0, 0, "X"),
        (False, 0, 0, 0, 0, "K"),
    ]
    rejected: list[dict[str, Any]] = []
    for arguments in candidates:
        try:
            kv_address(*arguments, geometry=geometry)
        except ContractError as error:
            rejected.append(
                {
                    "arguments": list(arguments),
                    "error": str(error),
                }
            )
        else:
            raise ContractError(f"illegal KV case was accepted: {arguments}")
    return rejected


def _small_geometry_document() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for layers in (1, 2, 3):
        geometry = Geometry(
            layers=layers,
            cache_slots=2,
            positions=2,
            kv_heads=2,
            head_dim=2,
            hidden_size=4,
            group_size=2,
            vocab_size=5,
        )
        schedule = expected_schedule(geometry)
        validate_trajectory(schedule, geometry)
        addresses = [
            {
                "arguments": [slot, layer, position, head, dimension, kind],
                "address": kv_address(
                    slot,
                    layer,
                    position,
                    head,
                    dimension,
                    kind,
                    geometry,
                ),
                "owner": kv_owner(layer, kind, geometry),
            }
            for kind in ("K", "V")
            for slot in range(geometry.cache_slots)
            for layer in range(geometry.layers)
            for position in range(geometry.positions)
            for head in range(geometry.kv_heads)
            for dimension in range(geometry.head_dim)
        ]
        machine = ExecutionMachine(geometry)
        bad_first = dict(schedule[0])
        bad_first["operation"] = "q_proj"
        try:
            machine.accept(bad_first)
        except ContractError as error:
            transition_rejection = {
                "error": str(error),
                "faulted": machine.faulted,
                "cursor": machine.cursor,
            }
        else:
            raise ContractError("illegal first transition was accepted")
        cases.append(
            {
                "geometry": geometry.document(),
                "schedule": schedule,
                "residual_handoffs": residual_handoffs(geometry),
                "legal_kv_addresses": addresses,
                "illegal_kv_cases": _illegal_kv_cases(geometry),
                "transition_rejection": transition_rejection,
                "lm_head": grouped_lm_head_interface(geometry),
                "argmax_cases": [
                    {"logits": [9, 9, 8, 7, 6], "token_id": 0},
                    {"logits": [-5, 4, 4, 3, 4], "token_id": 1},
                    {"logits": [-9, -8, -7, -6, -5], "token_id": 4},
                ],
            }
        )
    return {
        "schema_version": 1,
        "kind": "ace3_model24_execution_small_geometry_vectors",
        "exhaustive_dimensions": [
            "every legal event in each complete trajectory",
            "every legal KV address",
            "every KV bound and kind rejection",
            "strict first-transition fault and reset semantics",
        ],
        "cases": cases,
    }


def _official_schedule_document() -> dict[str, Any]:
    schedule = expected_schedule()
    validate_trajectory(schedule)
    return {
        "schema_version": 1,
        "kind": "ace3_model24_execution_official_schedule",
        "geometry": OFFICIAL_GEOMETRY.document(),
        "event_count": len(schedule),
        "events": schedule,
        "layer_bindings": layer_bindings(),
        "residual_handoffs": residual_handoffs(OFFICIAL_GEOMETRY),
        "kv_boundary_samples": [
            {
                "arguments": [0, 0, 0, 0, 0, "K"],
                "address": kv_address(0, 0, 0, 0, 0, "K"),
            },
            {
                "arguments": [1, 23, 127, 1, 63, "K"],
                "address": kv_address(1, 23, 127, 1, 63, "K"),
            },
            {
                "arguments": [0, 0, 0, 0, 0, "V"],
                "address": kv_address(0, 0, 0, 0, 0, "V"),
            },
            {
                "arguments": [1, 23, 127, 1, 63, "V"],
                "address": kv_address(1, 23, 127, 1, 63, "V"),
            },
        ],
        "lm_head": grouped_lm_head_interface(),
        "argmax_cases": [
            {"logits": [7, 7, 6, 5], "token_id": argmax_first([7, 7, 6, 5])},
            {"logits": [-8, 3, 3, 2], "token_id": argmax_first([-8, 3, 3, 2])},
        ],
        "claim_boundary": "structural and control vectors only; no official logits",
    }


def build_vector_artifacts(
    contract_sha256: str,
    bindings_sha256: str,
) -> dict[str, bytes]:
    documents = {
        "official_schedule.json": _official_schedule_document(),
        "small_geometry.json": _small_geometry_document(),
    }
    artifacts = {
        name: canonical_json_bytes(document)
        for name, document in documents.items()
    }
    manifest = {
        "schema_version": 1,
        "kind": "ace3_model24_execution_vector_manifest",
        "algorithm": "model24-execution-v1",
        "seed": 240483,
        "inputs": {
            "parent_commit": PARENT_COMMIT,
            "execution_contract_sha256": contract_sha256,
            "vector_bindings_sha256": bindings_sha256,
            "tensor_map_sha256": TENSOR_MAP_SHA256,
            "control_map_sha256": CONTROL_MAP_SHA256,
            "decoder_source_sha256": DECODER_SOURCE_SHA256,
            "decoder_interface_sha256": DECODER_INTERFACE_SHA256,
        },
        "artifacts": {
            name: {
                "sha256": sha256_bytes(payload),
                "bytes": len(payload),
            }
            for name, payload in sorted(artifacts.items())
        },
        "summary": {
            "official_layers": 24,
            "official_events": 483,
            "official_tensor_count": 627,
            "small_geometry_cases": 3,
            "small_geometry_max_layers": 3,
        },
        "verification_boundary": (
            "decoder-independent structural execution vectors; no RTL or "
            "official numerical model execution"
        ),
    }
    return {"manifest.json": canonical_json_bytes(manifest), **artifacts}
