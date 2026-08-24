#!/usr/bin/env python3
"""Independent structural oracle for the pinned 24-layer ACE-3 model."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, BinaryIO, Mapping

MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
CONFIG_SHA256 = (
    "bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090"
)
CHECKPOINT_SHA256 = (
    "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
)
CHECKPOINT_SIZE = 730_652_248
SAFETENSORS_HEADER_LENGTH = 68_432
SAFETENSORS_HEADER_SHA256 = (
    "2aeb5b461191aee401c2c3ac0a5b393a8f3a630ffd5571ddbc8c57ca7a149362"
)
SAFETENSORS_DATA_BASE = 8 + SAFETENSORS_HEADER_LENGTH
SAFETENSORS_DATA_LENGTH = 730_583_808
TIED_WEIGHT_SHA256 = (
    "d74257dc547b48be5ae7b93f1c9af072c0c42dbbb85503078e25c59cd09e68d0"
)

OFFICIAL_CONFIG: dict[str, Any] = {
    "hidden_size": 896,
    "intermediate_size": 4864,
    "max_position_embeddings": 32768,
    "num_attention_heads": 14,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "float16",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
    "quantization_config": {
        "bits": 4,
        "group_size": 128,
        "quant_method": "awq",
        "version": "gemm",
        "zero_point": True,
    },
}

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

DTYPE_BYTES = {"I32": 4, "F16": 2}
DTYPE_STORAGE_ORDER = {"I32": 0, "F16": 1}
KV_CACHE_SLOTS = 2
KV_MAX_TOKENS = 128


class ContractError(RuntimeError):
    """Raised when an authenticated artifact or model-control contract is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is not valid JSON: {error}") from error


def load_json_document(path: Path) -> Any:
    return _load_json_bytes(path.read_bytes(), str(path))


def canonical_sha256(domain: str, value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + payload).hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    for key, expected in OFFICIAL_CONFIG.items():
        if key == "quantization_config":
            continue
        require(config.get(key) == expected, f"config {key} mismatch")
    quantization = config.get("quantization_config")
    require(isinstance(quantization, Mapping), "quantization_config missing")
    for key, expected in OFFICIAL_CONFIG["quantization_config"].items():
        require(
            quantization.get(key) == expected,
            f"quantization_config {key} mismatch",
        )
    hidden_size = config["hidden_size"]
    query_heads = config["num_attention_heads"]
    require(
        hidden_size % query_heads == 0
        and hidden_size // query_heads == 64,
        "head dimension mismatch",
    )
    require(
        hidden_size % quantization["group_size"] == 0
        and config["intermediate_size"] % quantization["group_size"] == 0,
        "AWQ group geometry mismatch",
    )


def load_authenticated_config(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    require(
        actual_sha256 == CONFIG_SHA256,
        f"config SHA256 mismatch: expected {CONFIG_SHA256}, got {actual_sha256}",
    )
    config = _load_json_bytes(payload, str(path))
    require(isinstance(config, dict), "config root must be an object")
    validate_config(config)
    return config


def _add_projection_specs(
    specs: dict[str, dict[str, Any]],
    prefix: str,
    in_features: int,
    out_features: int,
    group_size: int,
    bias: bool,
) -> None:
    require(in_features % group_size == 0, f"{prefix} group count is fractional")
    require(out_features % 8 == 0, f"{prefix} packed width is fractional")
    groups = in_features // group_size
    packed_outputs = out_features // 8
    specs[f"{prefix}.qweight"] = {
        "dtype": "I32",
        "shape": [in_features, packed_outputs],
    }
    specs[f"{prefix}.qzeros"] = {
        "dtype": "I32",
        "shape": [groups, packed_outputs],
    }
    specs[f"{prefix}.scales"] = {
        "dtype": "F16",
        "shape": [groups, out_features],
    }
    if bias:
        specs[f"{prefix}.bias"] = {
            "dtype": "F16",
            "shape": [out_features],
        }


def expected_tensor_specs(
    config: Mapping[str, Any] = OFFICIAL_CONFIG,
) -> dict[str, dict[str, Any]]:
    validate_config(config)
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    layer_count = config["num_hidden_layers"]
    head_dim = hidden_size // config["num_attention_heads"]
    kv_size = config["num_key_value_heads"] * head_dim
    group_size = config["quantization_config"]["group_size"]
    specs: dict[str, dict[str, Any]] = {
        "model.embed_tokens.weight": {
            "dtype": "F16",
            "shape": [config["vocab_size"], hidden_size],
        },
        "model.norm.weight": {"dtype": "F16", "shape": [hidden_size]},
        "lm_head.weight": {
            "dtype": "F16",
            "shape": [config["vocab_size"], hidden_size],
        },
    }
    for layer_id in range(layer_count):
        layer = f"model.layers.{layer_id}"
        specs[f"{layer}.input_layernorm.weight"] = {
            "dtype": "F16",
            "shape": [hidden_size],
        }
        specs[f"{layer}.post_attention_layernorm.weight"] = {
            "dtype": "F16",
            "shape": [hidden_size],
        }
        _add_projection_specs(
            specs,
            f"{layer}.self_attn.q_proj",
            hidden_size,
            hidden_size,
            group_size,
            True,
        )
        _add_projection_specs(
            specs,
            f"{layer}.self_attn.k_proj",
            hidden_size,
            kv_size,
            group_size,
            True,
        )
        _add_projection_specs(
            specs,
            f"{layer}.self_attn.v_proj",
            hidden_size,
            kv_size,
            group_size,
            True,
        )
        _add_projection_specs(
            specs,
            f"{layer}.self_attn.o_proj",
            hidden_size,
            hidden_size,
            group_size,
            False,
        )
        _add_projection_specs(
            specs,
            f"{layer}.mlp.gate_proj",
            hidden_size,
            intermediate_size,
            group_size,
            False,
        )
        _add_projection_specs(
            specs,
            f"{layer}.mlp.up_proj",
            hidden_size,
            intermediate_size,
            group_size,
            False,
        )
        _add_projection_specs(
            specs,
            f"{layer}.mlp.down_proj",
            intermediate_size,
            hidden_size,
            group_size,
            False,
        )
    require(len(specs) == 627, "independent tensor specification count mismatch")
    return specs


def _element_count(shape: list[int]) -> int:
    count = 1
    for extent in shape:
        require(
            type(extent) is int and extent > 0,
            f"invalid tensor extent: {extent}",
        )
        count *= extent
    return count


def _tensor_descriptor_core(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "name": record["name"],
        "dtype": record["dtype"],
        "shape": record["shape"],
        "data_offsets": record["data_offsets"],
        "absolute_file_offsets": record["absolute_file_offsets"],
        "byte_length": record["byte_length"],
    }


def expected_tensor_records(
    config: Mapping[str, Any] = OFFICIAL_CONFIG,
) -> list[dict[str, Any]]:
    specs = expected_tensor_specs(config)
    ordered_names = sorted(
        specs,
        key=lambda name: (DTYPE_STORAGE_ORDER[specs[name]["dtype"]], name),
    )
    cursor = 0
    records: list[dict[str, Any]] = []
    for name in ordered_names:
        spec = specs[name]
        byte_length = _element_count(spec["shape"]) * DTYPE_BYTES[spec["dtype"]]
        record = {
            "name": name,
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "data_offsets": [cursor, cursor + byte_length],
            "absolute_file_offsets": [
                SAFETENSORS_DATA_BASE + cursor,
                SAFETENSORS_DATA_BASE + cursor + byte_length,
            ],
            "byte_length": byte_length,
        }
        record["descriptor_sha256"] = canonical_sha256(
            "ace3-model24-tensor-descriptor-v1",
            _tensor_descriptor_core(record),
        )
        records.append(record)
        cursor += byte_length
    require(
        cursor == SAFETENSORS_DATA_LENGTH,
        "independent checkpoint data length mismatch",
    )
    require(
        SAFETENSORS_DATA_BASE + cursor == CHECKPOINT_SIZE,
        "independent checkpoint file length mismatch",
    )
    return records


def _layer_suffix(name: str) -> tuple[int, str] | None:
    components = name.split(".", 3)
    if len(components) != 4 or components[:2] != ["model", "layers"]:
        return None
    try:
        layer_id = int(components[2])
    except ValueError:
        return None
    return layer_id, components[3]


def expected_layer_bindings(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_layer: dict[int, list[dict[str, Any]]] = {
        layer_id: [] for layer_id in range(24)
    }
    for record in records:
        parsed = _layer_suffix(record["name"])
        if parsed is not None:
            layer_id, _ = parsed
            require(layer_id in by_layer, f"unexpected layer id {layer_id}")
            by_layer[layer_id].append(record)
    result: list[dict[str, Any]] = []
    for layer_id, layer_records in by_layer.items():
        require(
            len(layer_records) == 26,
            f"layer {layer_id} tensor count mismatch",
        )
        digest = canonical_sha256(
            "ace3-model24-layer-namespace-v1",
            {
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "layer_id": layer_id,
                "tensor_descriptors": [
                    record["descriptor_sha256"] for record in layer_records
                ],
            },
        )
        result.append(
            {
                "layer_id": layer_id,
                "namespace": f"model.layers.{layer_id}.",
                "tensor_count": len(layer_records),
                "descriptor_sha256": digest,
            }
        )
    return result


def expected_family_bindings(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_family: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for record in records:
        parsed = _layer_suffix(record["name"])
        if parsed is None:
            continue
        layer_id, suffix = parsed
        by_family.setdefault(suffix, []).append((layer_id, record))
    require(len(by_family) == 26, "layer tensor family count mismatch")
    result: list[dict[str, Any]] = []
    for suffix in sorted(by_family):
        members = sorted(by_family[suffix], key=lambda item: item[0])
        require(
            [layer_id for layer_id, _ in members] == list(range(24)),
            f"tensor family {suffix} does not span all layers",
        )
        first = members[0][1]
        require(
            all(
                member["dtype"] == first["dtype"]
                and member["shape"] == first["shape"]
                for _, member in members
            ),
            f"tensor family {suffix} geometry differs by layer",
        )
        digest = canonical_sha256(
            "ace3-model24-tensor-family-v1",
            {
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "suffix": suffix,
                "tensor_descriptors": [
                    member["descriptor_sha256"] for _, member in members
                ],
            },
        )
        result.append(
            {
                "suffix": suffix,
                "dtype": first["dtype"],
                "shape": first["shape"],
                "layer_count": 24,
                "descriptor_sha256": digest,
            }
        )
    return result


def inventory_sha256(records: list[dict[str, Any]]) -> str:
    return canonical_sha256(
        "ace3-model24-complete-inventory-v1",
        {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tensor_descriptors": [
                record["descriptor_sha256"] for record in records
            ],
        },
    )


def validate_safetensors_header(
    header_blob: bytes,
    config: Mapping[str, Any] = OFFICIAL_CONFIG,
) -> None:
    require(len(header_blob) >= 8, "safetensors header prefix is truncated")
    header_length = struct.unpack("<Q", header_blob[:8])[0]
    require(
        header_length == SAFETENSORS_HEADER_LENGTH,
        "safetensors header length mismatch",
    )
    require(
        len(header_blob) == 8 + header_length,
        "safetensors header payload is truncated or overlong",
    )
    header_payload = header_blob[8:]
    require(
        hashlib.sha256(header_payload).hexdigest()
        == SAFETENSORS_HEADER_SHA256,
        "safetensors header SHA256 mismatch",
    )
    header = _load_json_bytes(header_payload, "safetensors header")
    require(isinstance(header, dict), "safetensors header root must be an object")
    require(header.get("__metadata__") == {"format": "pt"}, "metadata mismatch")
    actual_entries = [
        (name, value)
        for name, value in header.items()
        if name != "__metadata__"
    ]
    expected_entries = [
        (
            record["name"],
            {
                "dtype": record["dtype"],
                "shape": record["shape"],
                "data_offsets": record["data_offsets"],
            },
        )
        for record in expected_tensor_records(config)
    ]
    require(
        actual_entries == expected_entries,
        "safetensors tensor names, geometry, order, or offsets mismatch",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _compare_tied_ranges(
    source: BinaryIO,
    target: BinaryIO,
    left_offset: int,
    right_offset: int,
    byte_length: int,
) -> str:
    source.seek(left_offset)
    target.seek(right_offset)
    digest = hashlib.sha256()
    remaining = byte_length
    while remaining:
        count = min(1024 * 1024, remaining)
        left = source.read(count)
        right = target.read(count)
        require(
            len(left) == count and len(right) == count,
            "tied tensor payload is truncated",
        )
        require(left == right, "tied tensor payload values differ")
        digest.update(left)
        remaining -= count
    return digest.hexdigest()


def authenticate_checkpoint(
    path: Path,
    config: Mapping[str, Any] = OFFICIAL_CONFIG,
) -> None:
    require(path.stat().st_size == CHECKPOINT_SIZE, "checkpoint size mismatch")
    actual_sha256 = _sha256_file(path)
    require(
        actual_sha256 == CHECKPOINT_SHA256,
        f"checkpoint SHA256 mismatch: expected {CHECKPOINT_SHA256}, got {actual_sha256}",
    )
    with path.open("rb") as checkpoint:
        header_blob = checkpoint.read(SAFETENSORS_DATA_BASE)
    validate_safetensors_header(header_blob, config)
    records = {record["name"]: record for record in expected_tensor_records(config)}
    lm_head = records["lm_head.weight"]
    embedding = records["model.embed_tokens.weight"]
    require(
        lm_head["byte_length"] == embedding["byte_length"],
        "tied tensor byte lengths differ",
    )
    with path.open("rb") as left, path.open("rb") as right:
        tied_sha256 = _compare_tied_ranges(
            left,
            right,
            lm_head["absolute_file_offsets"][0],
            embedding["absolute_file_offsets"][0],
            lm_head["byte_length"],
        )
    require(
        tied_sha256 == TIED_WEIGHT_SHA256,
        "tied tensor value SHA256 mismatch",
    )


def tensor_map_document(
    config: Mapping[str, Any] = OFFICIAL_CONFIG,
) -> dict[str, Any]:
    validate_config(config)
    records = expected_tensor_records(config)
    layers = expected_layer_bindings(records)
    families = expected_family_bindings(records)
    tied_record = next(record for record in records if record["name"] == "lm_head.weight")
    return {
        "schema_version": 1,
        "kind": "ace3_model24_official_tensor_address_map",
        "provenance": {
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "config": {
                "filename": "config.json",
                "sha256": CONFIG_SHA256,
            },
            "checkpoint": {
                "filename": "model.safetensors",
                "lfs_sha256": CHECKPOINT_SHA256,
                "file_size": CHECKPOINT_SIZE,
                "header_length": SAFETENSORS_HEADER_LENGTH,
                "header_sha256": SAFETENSORS_HEADER_SHA256,
                "data_base_offset": SAFETENSORS_DATA_BASE,
                "data_byte_length": SAFETENSORS_DATA_LENGTH,
            },
        },
        "offset_semantics": {
            "data_offsets": "zero-based half-open byte ranges relative to the safetensors data section",
            "absolute_file_offsets": "zero-based half-open byte ranges in model.safetensors",
            "storage_order": "I32 tensors then F16 tensors; names lexicographically ascending within each dtype",
            "coverage": "all ranges are contiguous, non-overlapping, in bounds, and end exactly at checkpoint EOF",
        },
        "model_geometry": {
            "layers": config["num_hidden_layers"],
            "hidden_size": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
            "vocab_size": config["vocab_size"],
            "query_heads": config["num_attention_heads"],
            "key_value_heads": config["num_key_value_heads"],
            "head_dim": config["hidden_size"] // config["num_attention_heads"],
            "max_position_embeddings": config["max_position_embeddings"],
            "rope_theta": config["rope_theta"],
            "rms_norm_eps": config["rms_norm_eps"],
        },
        "numeric_profile": {
            "weights": "native asymmetric AWQ packed INT4",
            "awq_bits": config["quantization_config"]["bits"],
            "awq_group_size": config["quantization_config"]["group_size"],
            "awq_version": config["quantization_config"]["version"],
            "awq_zero_point": config["quantization_config"]["zero_point"],
            "packed_nibble_order": [0, 4, 1, 5, 2, 6, 3, 7],
            "zero_interpretation": "native qweight minus qzero; no qzero plus-one adjustment",
            "scales": "FP16",
            "projection_biases": "FP16",
            "norm_weights": "FP16",
            "activations": "FP16",
            "kv_cache": "FP16",
        },
        "tied_output": {
            "embedding_tensor": "model.embed_tokens.weight",
            "lm_head_tensor": "lm_head.weight",
            "dtype": "F16",
            "shape": [config["vocab_size"], config["hidden_size"]],
            "byte_length": tied_record["byte_length"],
            "storage": "distinct non-overlapping checkpoint ranges",
            "binding": "authenticated byte-for-byte value equality",
            "value_sha256": TIED_WEIGHT_SHA256,
        },
        "inventory": {
            "tensor_count": len(records),
            "layer_tensor_count": 26,
            "global_tensor_count": 3,
            "layer_namespace_count": len(layers),
            "layer_family_count": len(families),
            "descriptor_sha256": inventory_sha256(records),
        },
        "layer_namespaces": layers,
        "tensor_families": families,
        "tensors": records,
        "verification_boundary": {
            "established": [
                "official config and checkpoint identity",
                "complete safetensors header inventory and address coverage",
                "24 distinct layer namespaces and 26 complete per-layer tensor families",
                "final RMSNorm and tied lm_head geometry",
                "authenticated embedding and lm_head value equality",
            ],
            "excluded": [
                "numerical decoder-layer or full-model execution",
                "dialogue",
                "latency or throughput",
                "synthesis, PPA, FPGA, or silicon",
            ],
        },
    }


def kv_address(
    cache_slot: int,
    layer_id: int,
    position: int,
    kv_head: int,
    dimension: int,
    kind: str,
) -> int:
    bounds = (
        ("cache_slot", cache_slot, KV_CACHE_SLOTS),
        ("layer_id", layer_id, OFFICIAL_CONFIG["num_hidden_layers"]),
        ("position", position, KV_MAX_TOKENS),
        ("kv_head", kv_head, OFFICIAL_CONFIG["num_key_value_heads"]),
        (
            "dimension",
            dimension,
            OFFICIAL_CONFIG["hidden_size"]
            // OFFICIAL_CONFIG["num_attention_heads"],
        ),
    )
    for name, value, limit in bounds:
        require(
            type(value) is int and 0 <= value < limit,
            f"{name} outside supported range [0, {limit})",
        )
    require(kind in ("K", "V"), "kind must be K or V")
    head_dim = OFFICIAL_CONFIG["hidden_size"] // OFFICIAL_CONFIG[
        "num_attention_heads"
    ]
    entries_per_layer = (
        KV_MAX_TOKENS
        * OFFICIAL_CONFIG["num_key_value_heads"]
        * head_dim
    )
    entries_per_slot = (
        OFFICIAL_CONFIG["num_hidden_layers"] * entries_per_layer
    )
    entries_per_bank = KV_CACHE_SLOTS * entries_per_slot
    bank_offset = 0 if kind == "K" else entries_per_bank
    return (
        bank_offset
        + cache_slot * entries_per_slot
        + layer_id * entries_per_layer
        + position * OFFICIAL_CONFIG["num_key_value_heads"] * head_dim
        + kv_head * head_dim
        + dimension
    )


def expected_operation_events(
    config: Mapping[str, Any] = OFFICIAL_CONFIG,
) -> list[dict[str, Any]]:
    validate_config(config)
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
            event["layer_namespace"] = f"model.layers.{layer_id}."
        if operation in ("kv_write", "kv_read"):
            event["kv_namespace"] = [
                "cache_slot",
                layer_id,
                "position",
                "kv_head",
                "dimension",
                "K_or_V",
            ]
        events.append(event)

    append("embedding_lookup", None, ["model.embed_tokens.weight"])
    for layer_id in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer_id}."
        for operation in PER_LAYER_OPERATIONS:
            tensor_names = [
                prefix + suffix
                for suffix in OPERATION_TENSOR_SUFFIXES.get(operation, ())
            ]
            append(operation, layer_id, tensor_names)
    append("final_rmsnorm", None, ["model.norm.weight"])
    append("lm_head", None, ["lm_head.weight"])
    require(len(events) == 483, "operation event count mismatch")
    return events


def control_document(
    config: Mapping[str, Any] = OFFICIAL_CONFIG,
) -> dict[str, Any]:
    validate_config(config)
    records = expected_tensor_records(config)
    layers = expected_layer_bindings(records)
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    entries_per_layer = (
        KV_MAX_TOKENS * config["num_key_value_heads"] * head_dim
    )
    entries_per_slot = config["num_hidden_layers"] * entries_per_layer
    entries_per_bank = KV_CACHE_SLOTS * entries_per_slot
    events = expected_operation_events(config)
    return {
        "schema_version": 1,
        "kind": "ace3_model24_layer_and_kv_control",
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "config_sha256": CONFIG_SHA256,
            "checkpoint_lfs_sha256": CHECKPOINT_SHA256,
            "tensor_inventory_sha256": inventory_sha256(records),
        },
        "numeric_profile": {
            "projection_weights": "asymmetric AWQ W4A16 G128 native GEMM",
            "qzero_adjustment": "none",
            "projection_scales": "FP16",
            "norms_biases_activations": "FP16",
            "kv": "FP16",
        },
        "layer_namespaces": {
            "count": config["num_hidden_layers"],
            "strict_order": list(range(config["num_hidden_layers"])),
            "pattern": "model.layers.{layer_id}.",
            "bindings": layers,
            "reuse_policy": "a layer may consume only tensors under its own namespace",
        },
        "schedule": {
            "initial_operation": "embedding_lookup",
            "per_layer_operation_order": list(PER_LAYER_OPERATIONS),
            "final_operations": ["final_rmsnorm", "lm_head"],
            "operation_count": len(events),
            "tensor_requirements": [
                {
                    "operation": operation,
                    "layer_tensor_suffixes": list(suffixes),
                }
                for operation, suffixes in OPERATION_TENSOR_SUFFIXES.items()
            ],
            "transition_policy": [
                "layer 0 starts after embedding lookup",
                "layer N+1 starts only after layer N mlp_residual_add",
                "final_rmsnorm starts only after layer 23 mlp_residual_add",
                "lm_head starts only after final_rmsnorm",
            ],
            "missing_tensor_policy": "reject before scheduling; no zero-fill, alias, fallback, or layer reuse",
        },
        "kv_cache": {
            "logical_key": [
                "cache_slot",
                "layer_id",
                "position",
                "kv_head",
                "dimension",
                "K_or_V",
            ],
            "format": "FP16",
            "key_value_semantics": {
                "K": "RoPE-rotated key",
                "V": "unrotated value",
            },
            "limits": {
                "cache_slots": KV_CACHE_SLOTS,
                "layers": config["num_hidden_layers"],
                "supported_positions": KV_MAX_TOKENS,
                "official_max_position_embeddings": config[
                    "max_position_embeddings"
                ],
                "kv_heads": config["num_key_value_heads"],
                "head_dim": head_dim,
            },
            "unified_element_address": {
                "formula": "bank(K=0,V=entries_per_bank) + cache_slot*entries_per_slot + layer_id*entries_per_layer + position*(kv_heads*head_dim) + kv_head*head_dim + dimension",
                "element_bytes": 2,
                "entries_per_layer": entries_per_layer,
                "entries_per_slot": entries_per_slot,
                "entries_per_bank": entries_per_bank,
                "total_entries": 2 * entries_per_bank,
                "total_bytes": 4 * entries_per_bank,
            },
            "ordering": [
                "q_rope and k_rope complete after Q/K projection",
                "kv_write stores current-position rotated K and unrotated V in the active layer namespace",
                "kv_read may read only the active layer and positions not greater than the current query position",
                "attention_qk begins only after required layer-local KV reads complete",
            ],
            "bounds_policy": "reject every out-of-range component; never mask, modulo-wrap, clamp, or alias an address",
            "clear_reset": "reset or clear invalidates validity metadata for every slot and every layer namespace",
        },
        "final_projection": {
            "rmsnorm": {
                "tensor": "model.norm.weight",
                "dtype": "F16",
                "shape": [config["hidden_size"]],
                "epsilon": config["rms_norm_eps"],
            },
            "lm_head": {
                "tensor": "lm_head.weight",
                "dtype": "F16",
                "shape": [config["vocab_size"], config["hidden_size"]],
                "input_features": config["hidden_size"],
                "output_logits": config["vocab_size"],
                "tied_to": "model.embed_tokens.weight",
                "storage": "separate checkpoint range",
                "value_equality_sha256": TIED_WEIGHT_SHA256,
            },
        },
        "verification_boundary": {
            "established": [
                "strict 24-layer control sequencing",
                "complete tensor dependency and address coverage",
                "layer-disjoint K/V namespaces and bounds rejection",
                "final RMSNorm and tied lm_head transition",
            ],
            "excluded": [
                "numerical full-model execution",
                "dialogue",
                "latency or throughput",
                "synthesis, PPA, FPGA, or silicon",
            ],
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
            difference = _first_difference(
                expected[key],
                actual[key],
                f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: expected {len(expected)} items, got {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual)
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


def validate_contract_documents(
    tensor_map: Any,
    control: Any,
    config: Mapping[str, Any] = OFFICIAL_CONFIG,
) -> dict[str, int]:
    require(isinstance(tensor_map, dict), "tensor map root must be an object")
    require(isinstance(control, dict), "control root must be an object")
    expected_map = tensor_map_document(config)
    expected_control = control_document(config)
    map_difference = _first_difference(expected_map, tensor_map)
    require(map_difference is None, f"tensor map mismatch: {map_difference}")
    control_difference = _first_difference(expected_control, control)
    require(
        control_difference is None,
        f"model control mismatch: {control_difference}",
    )

    records = tensor_map["tensors"]
    names = [record["name"] for record in records]
    require(len(names) == len(set(names)) == 627, "tensor names are not unique")
    cursor = 0
    touched_ranges: set[tuple[int, int]] = set()
    for record in records:
        start, end = record["data_offsets"]
        absolute_start, absolute_end = record["absolute_file_offsets"]
        require(start == cursor and end > start, f"{record['name']} range gap")
        require(
            (absolute_start, absolute_end)
            == (SAFETENSORS_DATA_BASE + start, SAFETENSORS_DATA_BASE + end),
            f"{record['name']} absolute range mismatch",
        )
        require(
            (absolute_start, absolute_end) not in touched_ranges,
            f"{record['name']} address aliases another tensor",
        )
        touched_ranges.add((absolute_start, absolute_end))
        require(
            record["descriptor_sha256"]
            == canonical_sha256(
                "ace3-model24-tensor-descriptor-v1",
                _tensor_descriptor_core(record),
            ),
            f"{record['name']} descriptor hash mismatch",
        )
        cursor = end
    require(cursor == SAFETENSORS_DATA_LENGTH, "tensor ranges do not cover data")

    events = expected_operation_events(config)
    scheduled_tensors = {
        tensor_name
        for event in events
        for tensor_name in event["tensor_names"]
    }
    require(
        scheduled_tensors == set(names),
        "operation schedule does not touch every tensor exactly by name",
    )
    return {
        "tensor_count": len(records),
        "touched_range_count": len(touched_ranges),
        "layer_namespace_count": len(tensor_map["layer_namespaces"]),
        "tensor_family_count": len(tensor_map["tensor_families"]),
        "operation_event_count": len(events),
    }
