#!/usr/bin/env python3
"""Decoder-independent oracle for the pinned model24 execution contract."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PARENT_COMMIT = "3cf65b762d928e02e2b64fbba4389e294e1aa2c5"
MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
CONFIG_SHA256 = (
    "bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090"
)
CHECKPOINT_SHA256 = (
    "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
)
CHECKPOINT_SIZE = 730_652_248
CHECKPOINT_HEADER_SHA256 = (
    "2aeb5b461191aee401c2c3ac0a5b393a8f3a630ffd5571ddbc8c57ca7a149362"
)
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


@dataclass(frozen=True, slots=True)
class OracleGeometry:
    layers: int = 24
    cache_slots: int = 1
    positions: int = 2
    hidden_size: int = 256
    intermediate_size: int = 256
    query_heads: int = 4
    kv_heads: int = 1
    head_dim: int = 64
    group_size: int = 128
    vocab_size: int = 8

    def validate(self) -> None:
        for name, value in self.document().items():
            require(type(value) is int and value > 0, f"{name} must be positive")
        require(self.layers == 24, "software oracle must execute all 24 layers")
        require(
            self.hidden_size == self.query_heads * self.head_dim,
            "query head geometry mismatch",
        )
        require(
            self.kv_heads <= self.query_heads
            and self.query_heads % self.kv_heads == 0,
            "grouped-query head geometry mismatch",
        )
        require(
            self.hidden_size % self.group_size == 0
            and self.intermediate_size % self.group_size == 0,
            "native AWQ G128 geometry mismatch",
        )
        require(
            self.hidden_size % 8 == 0
            and self.intermediate_size % 8 == 0
            and self.kv_heads * self.head_dim % 8 == 0,
            "packed AWQ output geometry mismatch",
        )

    def schedule_geometry(self) -> Geometry:
        return Geometry(
            layers=self.layers,
            cache_slots=self.cache_slots,
            positions=self.positions,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            group_size=self.group_size,
            vocab_size=self.vocab_size,
        )

    def document(self) -> dict[str, int]:
        return {
            "layers": self.layers,
            "cache_slots": self.cache_slots,
            "positions": self.positions,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "group_size": self.group_size,
            "vocab_size": self.vocab_size,
        }


ORACLE_GEOMETRY = OracleGeometry()


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
            "config": {
                "filename": "config.json",
                "sha256": CONFIG_SHA256,
            },
            "checkpoint": {
                "filename": "model.safetensors",
                "lfs_sha256": CHECKPOINT_SHA256,
                "file_size": CHECKPOINT_SIZE,
                "header_sha256": CHECKPOINT_HEADER_SHA256,
            },
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
        "software_oracle_execution": {
            "algorithm": "model24-execution-v2",
            "geometry": ORACLE_GEOMETRY.document(),
            "input": {
                "initial_token_id": 2,
                "generated_token_count": 2,
                "cache_slot": 0,
            },
            "numeric_semantics": {
                "rounding_boundary": "binary16 after every operator output",
                "projection": (
                    "native asymmetric packed INT4 AWQ GEMM with G128 groups"
                ),
                "qweight_nibble_order": [0, 4, 1, 5, 2, 6, 3, 7],
                "qzero_adjustment": "none",
                "kv_storage": "FP16 values owned by cache slot, layer, and K/V kind",
                "attention": "grouped-query causal scaled dot product with softmax",
                "rope_theta": 1_000_000.0,
                "lm_head": (
                    "G128 grouped dot products over values tied to "
                    "model.embed_tokens.weight"
                ),
                "argmax_tie_break": "lowest token_id",
            },
            "evidence": {
                "control_events_per_token": 483,
                "tensor_descriptors": 627,
                "artifacts": [
                    "official_schedule.json",
                    "reduced_execution.json",
                    "fault_rejections.json",
                ],
                "serialization": (
                    "UTF-8 canonical JSON, sorted keys, compact separators, LF"
                ),
            },
        },
        "verification_boundary": {
            "established": [
                "24 namespace bindings and 483-event control trajectory",
                "per-layer FP16 KV ownership and residual lineage",
                "final RMSNorm, tied grouped lm_head, and deterministic argmax interfaces",
                "fixed parent, reviewed map, and decoder snapshot hash compatibility",
                "deterministic reduced-geometry software/oracle numerical execution",
            ],
            "excluded": [
                "decoder RTL implementation or execution",
                "official-geometry checkpoint numerical execution or readable dialogue",
                "latency, throughput, synthesis, PPA, FPGA, or silicon",
                "decoder acceptance",
            ],
        },
    }


def vector_bindings_document(
    contract_sha256: str,
    oracle_source_sha256: str,
) -> dict[str, Any]:
    require(
        len(contract_sha256) == 64,
        "execution contract SHA256 must contain 64 hex characters",
    )
    require(
        len(oracle_source_sha256) == 64,
        "oracle source SHA256 must contain 64 hex characters",
    )
    return {
        "schema_version": 2,
        "kind": "ace3_model24_execution_vector_bindings",
        "generator": {
            "algorithm": "model24-execution-v2",
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
            "config_sha256": CONFIG_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_size": CHECKPOINT_SIZE,
            "tied_weight_sha256": TIED_WEIGHT_SHA256,
            "oracle_source_sha256": oracle_source_sha256,
        },
        "artifact_set": [
            "fault_rejections.json",
            "manifest.json",
            "official_schedule.json",
            "reduced_execution.json",
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


def validate_vector_bindings(
    bindings: Any,
    contract_sha256: str,
    oracle_source_payload: bytes,
) -> None:
    require(isinstance(bindings, dict), "vector bindings root must be an object")
    require_document(
        vector_bindings_document(
            contract_sha256,
            sha256_bytes(oracle_source_payload),
        ),
        bindings,
        "execution vector bindings",
    )


def require_provenance_commit(
    repository_root: Path,
    provenance_commit: str = PARENT_COMMIT,
) -> None:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", provenance_commit, "HEAD"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
    except OSError as error:
        raise ContractError(
            f"unable to execute git while verifying required provenance commit "
            f"{provenance_commit}: {error}"
        ) from error
    if result.returncode == 0:
        return
    if result.returncode == 1:
        raise ContractError(
            f"required provenance commit {provenance_commit} is not an ancestor of HEAD"
        )
    detail = result.stderr.strip() or result.stdout.strip() or "no git diagnostic"
    raise ContractError(
        f"unable to verify required provenance commit {provenance_commit}: "
        f"git exited {result.returncode}: {detail}"
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


def argmax_lowest(logits: Sequence[int | float]) -> int:
    require(len(logits) > 0, "argmax requires at least one logit")
    require(
        all(type(value) in (int, float) and math.isfinite(value) for value in logits),
        "argmax logits must be finite numbers",
    )
    best_index = 0
    for index in range(1, len(logits)):
        if logits[index] > logits[best_index]:
            best_index = index
    return best_index


def _f16(value: int | float) -> float:
    require(type(value) in (int, float) and math.isfinite(value), "non-finite FP16 input")
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError as error:
        raise ContractError("FP16 operator output overflow") from error


def _f16_bits(value: int | float) -> int:
    return struct.unpack("<H", struct.pack("<e", _f16(value)))[0]


def _fp16_vector(values: Sequence[int | float]) -> list[float]:
    return [_f16(value) for value in values]


def _vector_record(values: Sequence[int | float]) -> dict[str, Any]:
    bits = [_f16_bits(value) for value in values]
    payload = b"".join(struct.pack("<H", value) for value in bits)
    return {
        "dtype": "FP16",
        "elements": len(bits),
        "sha256": sha256_bytes(payload),
        "sample_bits": bits[:8],
    }


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("ascii")).digest()[:4], "little")


def _awq_nibble(word: int, logical_lane: int) -> int:
    require(0 <= logical_lane < 8, "AWQ logical lane outside [0, 8)")
    return (word >> (4 * (0, 4, 1, 5, 2, 6, 3, 7)[logical_lane])) & 0xF


class ReducedTensorFixtures:
    """Deterministic reduced tensors with native packed-AWQ projection semantics."""

    def __init__(self, geometry: OracleGeometry = ORACLE_GEOMETRY) -> None:
        geometry.validate()
        self.geometry = geometry
        self._projection_cache: dict[
            tuple[int, str, int, int],
            tuple[list[list[int]], list[list[int]], list[list[float]]],
        ] = {}

    def embedding(self, token_id: int) -> list[float]:
        require(
            type(token_id) is int and 0 <= token_id < self.geometry.vocab_size,
            "token_id out of range",
        )
        tied_row = 0 if token_id in (0, 1) else token_id
        return [
            _f16((((tied_row + 3) * (index + 5) + index // 7) % 31 - 15) / 64)
            for index in range(self.geometry.hidden_size)
        ]

    def norm_weights(self, layer_id: int | None, post_attention: bool) -> list[float]:
        label = (
            "model.norm.weight"
            if layer_id is None
            else (
                f"model.layers.{layer_id}."
                f"{'post_attention' if post_attention else 'input'}_layernorm.weight"
            )
        )
        seed = _stable_seed(label)
        return [
            _f16(1 + ((seed + index * 13) % 9 - 4) / 256)
            for index in range(self.geometry.hidden_size)
        ]

    @staticmethod
    def _quantized_value(seed: int, input_index: int, output_index: int) -> int:
        return (
            seed
            + input_index * 5
            + output_index * 11
            + (input_index * output_index) % 17
        ) & 0xF

    @staticmethod
    def _zero_value(seed: int, group_index: int, output_index: int) -> int:
        return (seed // 17 + group_index * 3 + output_index * 7) & 0xF

    def _qweight_word(self, seed: int, input_index: int, packed_output: int) -> int:
        word = 0
        for lane in range(8):
            output_index = packed_output * 8 + lane
            value = self._quantized_value(seed, input_index, output_index)
            word |= value << (4 * (0, 4, 1, 5, 2, 6, 3, 7)[lane])
        return word

    def _qzero_word(self, seed: int, group_index: int, packed_output: int) -> int:
        word = 0
        for lane in range(8):
            output_index = packed_output * 8 + lane
            value = self._zero_value(seed, group_index, output_index)
            word |= value << (4 * (0, 4, 1, 5, 2, 6, 3, 7)[lane])
        return word

    def projection(
        self,
        layer_id: int,
        operation: str,
        inputs: Sequence[int | float],
        output_features: int,
        bias: bool,
    ) -> list[float]:
        require(0 <= layer_id < self.geometry.layers, "projection layer out of range")
        require(
            len(inputs) % self.geometry.group_size == 0,
            "projection input is not composed of G128 groups",
        )
        require(output_features % 8 == 0, "projection output is not INT4 packed")
        seed = _stable_seed(f"model.layers.{layer_id}.{operation}")
        packed_outputs = output_features // 8
        group_count = len(inputs) // self.geometry.group_size
        cache_key = (layer_id, operation, len(inputs), output_features)
        packed = self._projection_cache.get(cache_key)
        if packed is None:
            qweights = [
                [
                    self._qweight_word(seed, input_index, packed_output)
                    for packed_output in range(packed_outputs)
                ]
                for input_index in range(len(inputs))
            ]
            qzeros = [
                [
                    self._qzero_word(seed, group_index, packed_output)
                    for packed_output in range(packed_outputs)
                ]
                for group_index in range(group_count)
            ]
            scales = [
                [
                    _f16(
                        (
                            1
                            + (
                                (
                                    seed
                                    + group_index * 5
                                    + output_index * 3
                                )
                                % 7
                            )
                        )
                        / 2048
                    )
                    for output_index in range(output_features)
                ]
                for group_index in range(group_count)
            ]
            packed = (qweights, qzeros, scales)
            self._projection_cache[cache_key] = packed
        qweights, qzeros, scales = packed
        outputs: list[float] = []
        for output_index in range(output_features):
            packed_output, logical_lane = divmod(output_index, 8)
            accumulator = 0.0
            for group_start in range(0, len(inputs), self.geometry.group_size):
                group_index = group_start // self.geometry.group_size
                qzero = _awq_nibble(
                    qzeros[group_index][packed_output],
                    logical_lane,
                )
                integer_dot = 0.0
                for input_index in range(
                    group_start,
                    group_start + self.geometry.group_size,
                ):
                    qweight = _awq_nibble(
                        qweights[input_index][packed_output],
                        logical_lane,
                    )
                    integer_dot += _f16(inputs[input_index]) * (qweight - qzero)
                scale = scales[group_index][output_index]
                accumulator += integer_dot * scale
            if bias:
                accumulator += _f16(((seed + output_index * 19) % 17 - 8) / 512)
            outputs.append(_f16(accumulator))
        return outputs


def _rmsnorm(
    values: Sequence[int | float],
    weights: Sequence[int | float],
    epsilon: float = 1e-6,
) -> list[float]:
    require(len(values) == len(weights) and len(values) > 0, "RMSNorm shape mismatch")
    mean_square = sum(_f16(value) * _f16(value) for value in values) / len(values)
    inverse_rms = 1.0 / math.sqrt(mean_square + epsilon)
    return [_f16(_f16(value) * inverse_rms * _f16(weight)) for value, weight in zip(values, weights, strict=True)]


def _rope(
    values: Sequence[int | float],
    heads: int,
    head_dim: int,
    position: int,
) -> list[float]:
    require(len(values) == heads * head_dim, "RoPE shape mismatch")
    result = [0.0] * len(values)
    for head in range(heads):
        base = head * head_dim
        for pair in range(0, head_dim, 2):
            angle = position / (1_000_000.0 ** (pair / head_dim))
            cosine = math.cos(angle)
            sine = math.sin(angle)
            even = _f16(values[base + pair])
            odd = _f16(values[base + pair + 1])
            result[base + pair] = _f16(even * cosine - odd * sine)
            result[base + pair + 1] = _f16(even * sine + odd * cosine)
    return result


def _softmax(rows: Sequence[Sequence[int | float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for row in rows:
        require(len(row) > 0, "softmax row is empty")
        maximum = max(row)
        exponentials = [math.exp(value - maximum) for value in row]
        denominator = sum(exponentials)
        result.append([_f16(value / denominator) for value in exponentials])
    return result


class FP16KVCache:
    def __init__(self, geometry: OracleGeometry = ORACLE_GEOMETRY) -> None:
        self.geometry = geometry
        self.clear()

    def clear(self) -> None:
        self._owners: dict[tuple[int, int, str], str] = {}
        self._values: dict[tuple[int, int, int, str], tuple[float, ...]] = {}

    def write(
        self,
        cache_slot: int,
        layer_id: int,
        position: int,
        keys: Sequence[int | float],
        values: Sequence[int | float],
    ) -> None:
        schedule_geometry = self.geometry.schedule_geometry()
        for kind, vector in (("K", keys), ("V", values)):
            kv_address(
                cache_slot,
                layer_id,
                position,
                0,
                0,
                kind,
                schedule_geometry,
            )
            require(
                len(vector) == self.geometry.kv_heads * self.geometry.head_dim,
                f"{kind} cache vector shape mismatch",
            )
            owner_key = (cache_slot, layer_id, kind)
            expected_owner = kv_owner(layer_id, kind, schedule_geometry)
            current_owner = self._owners.get(owner_key)
            require(
                current_owner in (None, expected_owner),
                f"stale {kind} KV ownership for layer {layer_id}",
            )
            self._owners[owner_key] = expected_owner
            self._values[(cache_slot, layer_id, position, kind)] = tuple(
                _fp16_vector(vector)
            )

    def read(
        self,
        cache_slot: int,
        layer_id: int,
        position: int,
    ) -> tuple[list[list[float]], list[list[float]]]:
        result: list[list[list[float]]] = [[], []]
        schedule_geometry = self.geometry.schedule_geometry()
        for kind_index, kind in enumerate(("K", "V")):
            owner_key = (cache_slot, layer_id, kind)
            require(
                self._owners.get(owner_key) == kv_owner(layer_id, kind, schedule_geometry),
                f"{kind} KV owner is missing or stale for layer {layer_id}",
            )
            for cached_position in range(position + 1):
                value = self._values.get(
                    (cache_slot, layer_id, cached_position, kind)
                )
                require(
                    value is not None,
                    f"{kind} KV value missing at layer {layer_id} position {cached_position}",
                )
                result[kind_index].append(list(value))
        return result[0], result[1]

    def owner_document(self, cache_slot: int, layer_id: int) -> dict[str, str]:
        return {
            kind: self._owners[(cache_slot, layer_id, kind)]
            for kind in ("K", "V")
        }


def _projection_spec(
    operation: str,
    geometry: OracleGeometry,
) -> tuple[int, int, bool]:
    if operation == "q_proj":
        return geometry.hidden_size, geometry.hidden_size, True
    if operation in ("k_proj", "v_proj"):
        return geometry.hidden_size, geometry.kv_heads * geometry.head_dim, True
    if operation == "o_proj":
        return geometry.hidden_size, geometry.hidden_size, False
    if operation in ("gate_proj", "up_proj"):
        return geometry.hidden_size, geometry.intermediate_size, False
    if operation == "down_proj":
        return geometry.intermediate_size, geometry.hidden_size, False
    raise ContractError(f"unsupported projection operation: {operation}")


def reduced_tensor_inventory(
    geometry: OracleGeometry = ORACLE_GEOMETRY,
) -> list[dict[str, Any]]:
    geometry.validate()
    records: list[dict[str, Any]] = []
    names = sorted(
        {
            tensor_name
            for event in expected_schedule(geometry.schedule_geometry())
            for tensor_name in event["tensor_names"]
        }
    )
    for name in names:
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            shape = [geometry.vocab_size, geometry.hidden_size]
            dtype = "F16"
            value_key = "tied_embedding_values"
            tensor_class = "tied_embedding"
        elif name == "model.norm.weight":
            shape = [geometry.hidden_size]
            dtype = "F16"
            value_key = name
            tensor_class = "final_norm"
        else:
            suffix = name.split(".", 3)[3]
            if suffix.endswith("layernorm.weight"):
                shape = [geometry.hidden_size]
                dtype = "F16"
                tensor_class = "layer_norm"
            else:
                projection_name, tensor_class = suffix.rsplit(".", 1)
                operation = projection_name.rsplit(".", 1)[-1]
                input_features, output_features, _ = _projection_spec(
                    operation,
                    geometry,
                )
                if tensor_class == "qweight":
                    shape = [input_features, output_features // 8]
                    dtype = "I32"
                elif tensor_class == "qzeros":
                    shape = [input_features // geometry.group_size, output_features // 8]
                    dtype = "I32"
                elif tensor_class == "scales":
                    shape = [input_features // geometry.group_size, output_features]
                    dtype = "F16"
                elif tensor_class == "bias":
                    shape = [output_features]
                    dtype = "F16"
                else:
                    raise ContractError(f"unsupported tensor fixture: {name}")
            value_key = name
        descriptor = {
            "algorithm": "model24-reduced-fixture-v2",
            "value_key": value_key,
            "dtype": dtype,
            "shape": shape,
        }
        records.append(
            {
                "name": name,
                "tensor_class": tensor_class,
                "dtype": dtype,
                "shape": shape,
                "fixture_descriptor_sha256": sha256_bytes(
                    canonical_json_bytes(descriptor)
                ),
            }
        )
    require(len(records) == 627, "reduced tensor inventory must cover 627 tensors")
    return records


class SoftwareOracleEngine:
    def __init__(
        self,
        fixtures: ReducedTensorFixtures | None = None,
        geometry: OracleGeometry = ORACLE_GEOMETRY,
        *,
        tied_head: bool = True,
    ) -> None:
        geometry.validate()
        require(tied_head, "lm_head values are not tied to embedding values")
        self.geometry = geometry
        self.fixtures = fixtures or ReducedTensorFixtures(geometry)
        self.cache = FP16KVCache(geometry)

    def reset(self) -> None:
        self.cache.clear()

    def _attention_scores(
        self,
        queries: Sequence[float],
        key_history: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        rows: list[list[float]] = []
        heads_per_kv = self.geometry.query_heads // self.geometry.kv_heads
        for query_head in range(self.geometry.query_heads):
            query_base = query_head * self.geometry.head_dim
            kv_head = query_head // heads_per_kv
            kv_base = kv_head * self.geometry.head_dim
            row = []
            for keys in key_history:
                dot = sum(
                    queries[query_base + index] * keys[kv_base + index]
                    for index in range(self.geometry.head_dim)
                )
                row.append(_f16(dot / math.sqrt(self.geometry.head_dim)))
            rows.append(row)
        return rows

    def _attention_values(
        self,
        probabilities: Sequence[Sequence[float]],
        value_history: Sequence[Sequence[float]],
    ) -> list[float]:
        outputs: list[float] = []
        heads_per_kv = self.geometry.query_heads // self.geometry.kv_heads
        for query_head, row in enumerate(probabilities):
            kv_head = query_head // heads_per_kv
            kv_base = kv_head * self.geometry.head_dim
            for dimension in range(self.geometry.head_dim):
                value = sum(
                    probability * cached[kv_base + dimension]
                    for probability, cached in zip(row, value_history, strict=True)
                )
                outputs.append(_f16(value))
        return outputs

    def _lm_head(
        self,
        values: Sequence[float],
    ) -> tuple[list[list[float]], list[float], int]:
        partials: list[list[float]] = []
        logits: list[float] = []
        for token_id in range(self.geometry.vocab_size):
            weights = self.fixtures.embedding(token_id)
            token_partials = []
            for group_start in range(
                0,
                self.geometry.hidden_size,
                self.geometry.group_size,
            ):
                partial = sum(
                    values[index] * weights[index]
                    for index in range(
                        group_start,
                        group_start + self.geometry.group_size,
                    )
                )
                token_partials.append(_f16(partial))
            partials.append(token_partials)
            logits.append(_f16(sum(token_partials)))
        return partials, logits, argmax_lowest(logits)

    def run_token(
        self,
        token_id: int,
        position: int,
        cache_slot: int = 0,
    ) -> dict[str, Any]:
        require(
            type(position) is int and 0 <= position < self.geometry.positions,
            "generation position out of range",
        )
        schedule_geometry = self.geometry.schedule_geometry()
        schedule = expected_schedule(schedule_geometry)
        machine = ExecutionMachine(schedule_geometry)
        trace: list[dict[str, Any]] = []
        layer_input: list[float] = []
        for event in schedule:
            machine.accept(event)
            operation = event["operation"]
            details: dict[str, Any] = {}
            if operation == "embedding_lookup":
                current = self.fixtures.embedding(token_id)
            elif operation == "input_rmsnorm":
                layer_id = event["layer_id"]
                layer_input = list(current)
                current = _rmsnorm(
                    layer_input,
                    self.fixtures.norm_weights(layer_id, False),
                )
                normed_input = current
            elif operation in ("q_proj", "k_proj", "v_proj"):
                _, output_features, bias = _projection_spec(operation, self.geometry)
                projected = self.fixtures.projection(
                    event["layer_id"],
                    operation,
                    normed_input,
                    output_features,
                    bias,
                )
                if operation == "q_proj":
                    queries = projected
                elif operation == "k_proj":
                    keys = projected
                else:
                    values = projected
                current = projected
            elif operation == "q_rope":
                queries = _rope(
                    queries,
                    self.geometry.query_heads,
                    self.geometry.head_dim,
                    position,
                )
                current = queries
            elif operation == "k_rope":
                keys = _rope(
                    keys,
                    self.geometry.kv_heads,
                    self.geometry.head_dim,
                    position,
                )
                current = keys
            elif operation == "kv_write":
                layer_id = event["layer_id"]
                self.cache.write(cache_slot, layer_id, position, keys, values)
                current = keys + values
                details["kv_transition"] = {
                    "action": "write",
                    "cache_slot": cache_slot,
                    "layer_id": layer_id,
                    "position": position,
                    "owners": self.cache.owner_document(cache_slot, layer_id),
                    "format": "FP16",
                }
            elif operation == "kv_read":
                layer_id = event["layer_id"]
                key_history, value_history = self.cache.read(
                    cache_slot,
                    layer_id,
                    position,
                )
                current = [
                    value
                    for history in (key_history, value_history)
                    for vector in history
                    for value in vector
                ]
                details["kv_transition"] = {
                    "action": "read",
                    "cache_slot": cache_slot,
                    "layer_id": layer_id,
                    "positions": list(range(position + 1)),
                    "owners": self.cache.owner_document(cache_slot, layer_id),
                    "format": "FP16",
                }
            elif operation == "attention_qk":
                scores = self._attention_scores(queries, key_history)
                current = [value for row in scores for value in row]
            elif operation == "attention_softmax":
                probabilities = _softmax(scores)
                current = [value for row in probabilities for value in row]
            elif operation == "attention_value":
                attention_values = self._attention_values(
                    probabilities,
                    value_history,
                )
                current = attention_values
            elif operation == "o_proj":
                current = self.fixtures.projection(
                    event["layer_id"],
                    operation,
                    attention_values,
                    self.geometry.hidden_size,
                    False,
                )
            elif operation == "attention_residual_add":
                residual_input = _vector_record(layer_input)
                current = [
                    _f16(residual + attention_output)
                    for residual, attention_output in zip(
                        layer_input,
                        current,
                        strict=True,
                    )
                ]
                attention_residual = current
                details["residual_transition"] = {
                    "input": event["residual_input"],
                    "input_sha256": residual_input["sha256"],
                    "output": event["residual_output"],
                    "output_sha256": _vector_record(current)["sha256"],
                    "format": "FP16",
                }
            elif operation == "post_attention_rmsnorm":
                current = _rmsnorm(
                    attention_residual,
                    self.fixtures.norm_weights(event["layer_id"], True),
                )
                mlp_input = current
            elif operation in ("gate_proj", "up_proj"):
                _, output_features, _ = _projection_spec(operation, self.geometry)
                projected = self.fixtures.projection(
                    event["layer_id"],
                    operation,
                    mlp_input,
                    output_features,
                    False,
                )
                if operation == "gate_proj":
                    gate = projected
                else:
                    up = projected
                current = projected
            elif operation == "silu":
                gate_silu = [_f16(value / (1.0 + math.exp(-value))) for value in gate]
                current = gate_silu
            elif operation == "gated_multiply":
                current = [
                    _f16(left * right)
                    for left, right in zip(gate_silu, up, strict=True)
                ]
            elif operation == "down_proj":
                current = self.fixtures.projection(
                    event["layer_id"],
                    operation,
                    current,
                    self.geometry.hidden_size,
                    False,
                )
            elif operation == "mlp_residual_add":
                residual_input = _vector_record(attention_residual)
                current = [
                    _f16(residual + mlp_output)
                    for residual, mlp_output in zip(
                        attention_residual,
                        current,
                        strict=True,
                    )
                ]
                details["residual_transition"] = {
                    "input": event["residual_input"],
                    "input_sha256": residual_input["sha256"],
                    "output": event["residual_output"],
                    "output_sha256": _vector_record(current)["sha256"],
                    "format": "FP16",
                }
            elif operation == "final_rmsnorm":
                current = _rmsnorm(
                    current,
                    self.fixtures.norm_weights(None, False),
                )
                details["final_norm"] = {
                    "tensor": "model.norm.weight",
                    "epsilon": 1e-6,
                    "result": _vector_record(current),
                }
            elif operation == "lm_head":
                partials, logits, selected_token = self._lm_head(current)
                current = logits
                details["lm_head"] = {
                    "tied_to": "model.embed_tokens.weight",
                    "tied": True,
                    "group_size": self.geometry.group_size,
                    "groups_per_logit": (
                        self.geometry.hidden_size // self.geometry.group_size
                    ),
                    "partial_bits": [
                        [_f16_bits(value) for value in row] for row in partials
                    ],
                    "logit_bits": [_f16_bits(value) for value in logits],
                    "argmax_token_id": selected_token,
                    "tie_break": "lowest token_id",
                }
            else:
                raise ContractError(f"unimplemented execution operation: {operation}")
            trace.append(
                {
                    "ordinal": event["ordinal"],
                    "operation": operation,
                    "layer_id": event["layer_id"],
                    "tensor_names": event["tensor_names"],
                    "output": _vector_record(current),
                    **details,
                }
            )
        require(machine.done and len(trace) == 483, "software execution did not complete")
        return {
            "input_token_id": token_id,
            "position": position,
            "cache_slot": cache_slot,
            "control_event_count": len(trace),
            "events": trace,
            "final_norm": trace[-2]["final_norm"]["result"],
            "lm_head": trace[-1]["lm_head"],
            "output_token_id": selected_token,
        }


def reduced_execution_document() -> dict[str, Any]:
    engine = SoftwareOracleEngine()
    executions = []
    token_id = 2
    for position in range(2):
        execution = engine.run_token(token_id, position)
        executions.append(execution)
        token_id = execution["output_token_id"]
    inventory = reduced_tensor_inventory()
    consumed = sorted(
        {
            tensor_name
            for execution in executions
            for event in execution["events"]
            for tensor_name in event["tensor_names"]
        }
    )
    require(
        consumed == [record["name"] for record in inventory],
        "execution tensor consumption does not match fixture inventory",
    )
    tied_rows = [
        value
        for token_id in range(ORACLE_GEOMETRY.vocab_size)
        for value in engine.fixtures.embedding(token_id)
    ]
    return {
        "schema_version": 1,
        "kind": "ace3_model24_reduced_software_oracle_execution",
        "algorithm": "model24-execution-v2",
        "official_model_provenance": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "config_sha256": CONFIG_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_size": CHECKPOINT_SIZE,
            "checkpoint_header_sha256": CHECKPOINT_HEADER_SHA256,
            "official_tied_weight_sha256": TIED_WEIGHT_SHA256,
        },
        "geometry": ORACLE_GEOMETRY.document(),
        "numeric_profile": {
            "activations": "FP16",
            "kv": "FP16",
            "awq": "native asymmetric packed INT4 GEMM G128",
            "packed_nibble_order": [0, 4, 1, 5, 2, 6, 3, 7],
            "qzero_adjustment": "none",
            "scales": "FP16",
        },
        "tensor_inventory": inventory,
        "tensor_count": len(inventory),
        "consumed_tensor_names": consumed,
        "reduced_tied_embedding_value_sha256": _vector_record(tied_rows)["sha256"],
        "executions": executions,
        "generated_tokens": [execution["output_token_id"] for execution in executions],
        "coverage": {
            "layers": 24,
            "control_events_per_execution": 483,
            "execution_count": len(executions),
            "residual_transitions_per_execution": 48,
            "kv_write_read_transitions_per_execution": 48,
            "final_rmsnorm_per_execution": 1,
            "tied_grouped_lm_head_per_execution": 1,
        },
        "claim_boundary": (
            "deterministic reduced-geometry software/oracle execution only; "
            "not official-geometry logits, readable dialogue, RTL acceptance, "
            "latency, synthesis, PPA, or FPGA evidence"
        ),
    }


def _rejection(function: Any) -> str:
    try:
        function()
    except ContractError as error:
        return str(error)
    raise ContractError("negative test was unexpectedly accepted")


def fault_rejections_document(
    contract_sha256: str,
    oracle_source_payload: bytes,
) -> dict[str, Any]:
    geometry = ORACLE_GEOMETRY.schedule_geometry()
    schedule = expected_schedule(geometry)
    mutations: dict[str, list[dict[str, Any]]] = {
        "missing": schedule[:10] + schedule[11:],
        "duplicate": schedule[:11] + [schedule[10]] + schedule[11:],
        "reordered": schedule[:10] + [schedule[11], schedule[10]] + schedule[12:],
        "extra": schedule + [schedule[-1]],
    }
    schedule_rejections = {
        name: _rejection(lambda events=events: validate_trajectory(events, geometry))
        for name, events in mutations.items()
    }
    machine = ExecutionMachine(geometry)
    invalid_first = dict(schedule[0])
    invalid_first["operation"] = "q_proj"
    initial_fault = _rejection(lambda: machine.accept(invalid_first))
    while_faulted = _rejection(lambda: machine.accept(schedule[0]))
    machine.reset()
    for event in schedule:
        machine.accept(event)
    require(machine.done, "reset did not restore schedule execution")

    cache = FP16KVCache()
    cache._owners[(0, 0, "K")] = "kv.layer.23.K"
    zero_kv = [0.0] * (ORACLE_GEOMETRY.kv_heads * ORACLE_GEOMETRY.head_dim)
    stale_kv = _rejection(lambda: cache.write(0, 0, 0, zero_kv, zero_kv))
    untied_head = _rejection(lambda: SoftwareOracleEngine(tied_head=False))

    oracle_sha256 = sha256_bytes(oracle_source_payload)
    bindings = vector_bindings_document(contract_sha256, oracle_sha256)
    source_tamper = _rejection(
        lambda: validate_vector_bindings(
            bindings,
            contract_sha256,
            oracle_source_payload + b"\n# tampered\n",
        )
    )
    hash_tamper = _rejection(
        lambda: validate_vector_bindings(
            bindings,
            "0" * 64,
            oracle_source_payload,
        )
    )
    return {
        "schema_version": 1,
        "kind": "ace3_model24_execution_fault_rejections",
        "schedule": schedule_rejections,
        "fault_latch_and_reset": {
            "initial_rejection": initial_fault,
            "while_faulted_rejection": while_faulted,
            "reset_completed_all_events": machine.done,
        },
        "stale_kv_owner_rejection": stale_kv,
        "untied_lm_head_rejection": untied_head,
        "source_tamper_rejection": source_tamper,
        "contract_hash_tamper_rejection": hash_tamper,
        "argmax_tie_case": {
            "logits": [7, 7, 6, 5],
            "selected_token_id": argmax_lowest([7, 7, 6, 5]),
            "rule": "lowest token_id",
        },
    }


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
    oracle_source_payload: bytes,
) -> dict[str, bytes]:
    documents = {
        "fault_rejections.json": fault_rejections_document(
            contract_sha256,
            oracle_source_payload,
        ),
        "official_schedule.json": _official_schedule_document(),
        "reduced_execution.json": reduced_execution_document(),
    }
    artifacts = {
        name: canonical_json_bytes(document)
        for name, document in documents.items()
    }
    manifest = {
        "schema_version": 2,
        "kind": "ace3_model24_execution_vector_manifest",
        "algorithm": "model24-execution-v2",
        "seed": 240483,
        "inputs": {
            "parent_commit": PARENT_COMMIT,
            "execution_contract_sha256": contract_sha256,
            "vector_bindings_sha256": bindings_sha256,
            "tensor_map_sha256": TENSOR_MAP_SHA256,
            "control_map_sha256": CONTROL_MAP_SHA256,
            "decoder_source_sha256": DECODER_SOURCE_SHA256,
            "decoder_interface_sha256": DECODER_INTERFACE_SHA256,
            "config_sha256": CONFIG_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_size": CHECKPOINT_SIZE,
            "tied_weight_sha256": TIED_WEIGHT_SHA256,
            "oracle_source_sha256": sha256_bytes(oracle_source_payload),
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
            "reduced_execution_count": 2,
            "generated_token_count": 2,
            "reduced_geometry_layers": 24,
        },
        "verification_boundary": (
            "deterministic reduced-geometry software/oracle execution; no RTL, "
            "official-geometry logits, dialogue, latency, synthesis, PPA, or FPGA claim"
        ),
    }
    return {"manifest.json": canonical_json_bytes(manifest), **artifacts}
