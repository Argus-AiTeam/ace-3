#!/usr/bin/env python3
"""Decoder-independent oracle for the pinned model24 execution contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from awq_bit_oracle import q47_48_to_f16
from decoder_layer0_oracle import run_token as run_decoder_layer_token
from fp16_adaptation_oracle import decode_f16_q24, q24_to_f16, rmsnorm
from model24_oracle import (
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    ContractError as Model24OracleContractError,
    TIED_WEIGHT_SHA256 as OFFICIAL_TIED_WEIGHT_SHA256,
    authenticate_checkpoint,
    expected_tensor_records,
)
from projection_oracle import complete_projection_output
from qwen2_rope_oracle import qwen2_coefficient
from official_single_decoder_layer import (
    LayerExecutionError,
    official_single_decoder_layer_contract,
    official_single_decoder_layer_document,
)

PARENT_COMMIT = "3cf65b762d928e02e2b64fbba4389e294e1aa2c5"
MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
DEFAULT_OFFICIAL_CHECKPOINT = Path(
    os.environ.get(
        "ACE3_OFFICIAL_MODEL24_CHECKPOINT",
        Path(__file__).resolve().parents[2]
        / "model24_execution_vectors"
        / "model.safetensors",
    )
)
DEFAULT_OFFICIAL_TOKENIZER_DIR = (
    Path(
        os.environ.get(
            "ACE3_OFFICIAL_MODEL24_TOKENIZER_DIR",
            Path(__file__).resolve().parents[2]
            / "model24_execution_vectors"
            / "tokenizer",
        )
    )
)
TOKENIZER_SHA256 = (
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
)
TOKENIZER_CONFIG_SHA256 = (
    "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
)
FIXED_CHAT_MESSAGES = (
    ("system", "You are a concise assistant."),
    ("user", "Say hello in two words."),
)
FIXED_CHAT_SERIALIZATION = (
    "<|im_start|>system\n"
    "You are a concise assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Say hello in two words.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
FIXED_CHAT_TOKEN_IDS = (
    151644,
    8948,
    198,
    2610,
    525,
    264,
    63594,
    17847,
    13,
    151645,
    198,
    151644,
    872,
    198,
    45764,
    23811,
    304,
    1378,
    4244,
    13,
    151645,
    198,
    151644,
    77091,
    198,
)
REDUCED_RESPONSE_TOKEN_IDS = (9707, 1879)
EOS_TOKEN_ID = 151645
REDUCED_GENERATION_TOKEN_IDS = (*REDUCED_RESPONSE_TOKEN_IDS, EOS_TOKEN_ID)
REDUCED_EXECUTION_TOKEN_IDS = frozenset(
    (*FIXED_CHAT_TOKEN_IDS, *REDUCED_GENERATION_TOKEN_IDS)
)
REDUCED_LOGIT_FIXTURE = (
    (9, 3, 1),
    (1, 9, 3),
    (1, 2, 9),
)
TENSOR_MAP_SHA256 = (
    "11a03bed8049cd815ac2c37384a7ba15d71d2f69ee397110d1cd443193474624"
)
CONTROL_MAP_SHA256 = (
    "3364dc4c2c585f4687d8ad7943792ca4c44265b85463b91ab2a1c6866690b611"
)
DECODER_SOURCE_SHA256 = (
    "ea1ed72cec4f0b15b852265de55f886201eda15d8413b706d23896fc0712a0d6"
)
DECODER_INTERFACE_SHA256 = (
    "93445f03e9bb72c5fff5b18388703c8e734a5ab3a75e6e8c85992c183e39c2ab"
)
TIED_WEIGHT_SHA256 = (
    OFFICIAL_TIED_WEIGHT_SHA256
)
FINAL_NORM_SHA256 = (
    "1dd25d7720c68bc10838374200238c26626a624119cac0b45bff44bc43c354fe"
)
FIXED_TERMINAL_HIDDEN_SHA256 = (
    "fbbfb9fbf379ac045a31f5d9c0e1e0e5080f67c1bf764eff3466ef0f9a404fef"
)
FINAL_PROJECTION_TENSOR_NAMES = (
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
)
OFFICIAL_TOP_K = 10

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

LAYER0_VL15_FINAL_ROWS_SHA256 = (
    "22768ac6b337f920faac7de59b4eb43a203e1db45cdf688820fcbb35cdfe3446"
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


def indexed_layer_binding(layer_index: int) -> dict[str, Any]:
    require(
        type(layer_index) is int and 0 <= layer_index < len(LAYER_DESCRIPTOR_SHA256),
        "layer_index out of range",
    )
    return layer_bindings()[layer_index]


def indexed_layer_tensor_records(
    tensor_map: Mapping[str, Any],
    layer_index: int,
) -> list[dict[str, Any]]:
    binding = indexed_layer_binding(layer_index)
    namespaces = tensor_map.get("layer_namespaces")
    require(isinstance(namespaces, list), "tensor map layer_namespaces missing")
    matches = [
        item
        for item in namespaces
        if isinstance(item, dict) and item.get("layer_id") == layer_index
    ]
    require(len(matches) == 1, "selected layer namespace is not unique")
    require_document(
        {
            **binding,
            "tensor_count": 26,
        },
        {
            "layer_id": matches[0].get("layer_id"),
            "namespace": matches[0].get("namespace"),
            "descriptor_sha256": matches[0].get("descriptor_sha256"),
            "tensor_count": matches[0].get("tensor_count"),
        },
        "selected layer namespace",
    )
    tensors = tensor_map.get("tensors")
    require(isinstance(tensors, list), "tensor map tensors missing")
    records = [
        dict(item)
        for item in tensors
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and item["name"].startswith(binding["namespace"])
    ]
    require(len(records) == 26, "selected layer tensor count mismatch")
    require(
        all(item["name"].count(".") >= 4 for item in records),
        "selected layer tensor name mismatch",
    )
    return sorted(records, key=lambda item: item["name"])


def load_two_token_handoff(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[list[list[int]], dict[str, Any]]:
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    if expected_sha256 is not None:
        require(digest == expected_sha256, "two-token handoff SHA256 mismatch")
    require(payload.endswith(b"\n"), "two-token handoff must be LF terminated")
    lines = payload.splitlines()
    require(len(lines) == 2 * OFFICIAL_GEOMETRY.hidden_size, "two-token handoff row count")
    rows = [
        [0] * OFFICIAL_GEOMETRY.hidden_size
        for _ in range(2)
    ]
    for ordinal, line in enumerate(lines):
        require(len(line) == 10, f"two-token handoff row {ordinal} width")
        try:
            token = int(line[0:2], 16)
            index = int(line[2:6], 16)
            value = int(line[6:10], 16)
        except ValueError as error:
            raise ContractError(f"two-token handoff row {ordinal} is not hex") from error
        expected_token, expected_index = divmod(
            ordinal,
            OFFICIAL_GEOMETRY.hidden_size,
        )
        require(
            (token, index) == (expected_token, expected_index),
            f"two-token handoff row {ordinal} is out of sequence",
        )
        rows[token][index] = value
    return rows, {
        "sha256": digest,
        "rows": len(lines),
        "shape": [2, OFFICIAL_GEOMETRY.hidden_size],
        "dtype": "F16",
        "record_format": "token[7:0] index[15:0] f16[15:0]",
    }


def indexed_layer_input_handoff_binding(
    layer_index: int,
    handoff_binding: Mapping[str, Any],
) -> dict[str, Any]:
    indexed_layer_binding(layer_index)
    require(layer_index > 0, "indexed decoder layer requires a predecessor handoff")
    return {
        **handoff_binding,
        "source": f"authenticated decoder layer {layer_index - 1} raw final rows",
        "source_layer_index": layer_index - 1,
        "consumer_layer_index": layer_index,
        "byte_preserved_as": "inputs.hex",
    }


def indexed_capture_tensor_filenames(
    cpp_source: str,
    layer_index: int,
) -> tuple[str, ...]:
    indexed_layer_binding(layer_index)
    direct_tensors = re.findall(
        r'tensor\(dir,\s*"(layer(\d+)_[A-Za-z0-9_]+\.fp16le\.bin)"\)',
        cpp_source,
    )
    projections = re.findall(
        r'projection\(dir,\s*"(layer(\d+)_[A-Za-z0-9_]+)",\s*(true|false)\)',
        cpp_source,
    )
    require(
        len(direct_tensors) == 2 and len(projections) == 7,
        "capture tensor request set mismatch",
    )
    observed = {
        int(index)
        for _, index in (*direct_tensors, *((name, index) for name, index, _ in projections))
    }
    require(
        observed == {layer_index},
        "capture tensor layer-index binding mismatch",
    )
    filenames = [f"{name}.hex" for name, _ in direct_tensors]
    for prefix, _, has_bias in projections:
        filenames.extend(
            (
                f"{prefix}_qweight.i32le.bin.hex",
                f"{prefix}_qzeros.i32le.bin.hex",
                f"{prefix}_scales.fp16le.bin.hex",
            )
        )
        if has_bias == "true":
            filenames.append(f"{prefix}_bias.fp16le.bin.hex")
    require(
        len(filenames) == 26 and len(set(filenames)) == len(filenames),
        "capture tensor filename set mismatch",
    )
    return tuple(filenames)


def validate_indexed_capture_tensor_files(
    cpp_source: str,
    vector_dir: Path,
    layer_index: int,
) -> tuple[str, ...]:
    filenames = indexed_capture_tensor_filenames(cpp_source, layer_index)
    tensor_dir = vector_dir / "tensors"
    missing = [name for name in filenames if not (tensor_dir / name).is_file()]
    require(not missing, "capture tensor files missing: " + ", ".join(missing))
    return filenames


def validate_indexed_capture_sources(
    cpp_source: str,
    raw_evidence_header: str,
    layer_index: int,
) -> None:
    indexed_layer_binding(layer_index)
    expected_cpp = {
        f"if(layer_index != {layer_index})": 1,
        f"this sealed invocation requires layer index {layer_index}": 1,
        f"schema=ace3-layer{layer_index}-simulator-terminal-v1": 2,
        f"layer_index={layer_index}": 2,
        f"ACE3_LAYER{layer_index}_CAPTURE": 2,
    }
    expected_header = {
        f"schema=ace3-layer{layer_index}-raw-counts-v1": 1,
        f"layer_index={layer_index}": 1,
    }
    require(
        all(cpp_source.count(marker) == count for marker, count in expected_cpp.items())
        and all(
            raw_evidence_header.count(marker) == count
            for marker, count in expected_header.items()
        ),
        "capture source layer-index binding mismatch",
    )
    observed = {
        int(value)
        for pattern, source in (
            (r"ace3-layer(\d+)-simulator-terminal-v1", cpp_source),
            (r"ACE3_LAYER(\d+)_CAPTURE", cpp_source),
            (r"ace3-layer(\d+)-raw-counts-v1", raw_evidence_header),
        )
        for value in re.findall(pattern, source)
    }
    require(
        observed == {layer_index},
        "capture source contains a stale layer-index marker",
    )
    indexed_capture_tensor_filenames(cpp_source, layer_index)


def retarget_indexed_capture_sources(
    cpp_source: str,
    raw_evidence_header: str,
    *,
    source_layer_index: int,
    target_layer_index: int,
) -> tuple[str, str]:
    validate_indexed_capture_sources(
        cpp_source,
        raw_evidence_header,
        source_layer_index,
    )
    require(
        source_layer_index != target_layer_index,
        "capture source and target layer indexes must differ",
    )
    indexed_layer_binding(target_layer_index)
    cpp_replacements = (
        (
            f"if(layer_index != {source_layer_index})",
            f"if(layer_index != {target_layer_index})",
        ),
        (
            f"this sealed invocation requires layer index {source_layer_index}",
            f"this sealed invocation requires layer index {target_layer_index}",
        ),
        (
            f"schema=ace3-layer{source_layer_index}-simulator-terminal-v1",
            f"schema=ace3-layer{target_layer_index}-simulator-terminal-v1",
        ),
        (
            f"layer_index={source_layer_index}",
            f"layer_index={target_layer_index}",
        ),
        (
            f"ACE3_LAYER{source_layer_index}_CAPTURE",
            f"ACE3_LAYER{target_layer_index}_CAPTURE",
        ),
    )
    header_replacements = (
        (
            f"schema=ace3-layer{source_layer_index}-raw-counts-v1",
            f"schema=ace3-layer{target_layer_index}-raw-counts-v1",
        ),
        (
            f"layer_index={source_layer_index}",
            f"layer_index={target_layer_index}",
        ),
    )
    for old, new in cpp_replacements:
        cpp_source = cpp_source.replace(old, new)
    cpp_source = cpp_source.replace(
        f"layer{source_layer_index}_",
        f"layer{target_layer_index}_",
    )
    for old, new in header_replacements:
        raw_evidence_header = raw_evidence_header.replace(old, new)
    validate_indexed_capture_sources(
        cpp_source,
        raw_evidence_header,
        target_layer_index,
    )
    return cpp_source, raw_evidence_header


def _layer_tensor_payloads(
    checkpoint_path: Path,
    tensor_map_path: Path,
    layer_index: int,
) -> tuple[dict[str, bytes], list[dict[str, Any]], dict[str, Any]]:
    authenticate_checkpoint(checkpoint_path)
    tensor_map_payload = tensor_map_path.read_bytes()
    require(
        sha256_bytes(tensor_map_payload) == TENSOR_MAP_SHA256,
        "reviewed tensor map SHA256 mismatch",
    )
    tensor_map = load_json_bytes(tensor_map_payload, "tensor map")
    require(isinstance(tensor_map, dict), "tensor map root must be an object")
    records = indexed_layer_tensor_records(tensor_map, layer_index)
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ContractError("the safetensors package is required") from error
    payloads: dict[str, bytes] = {}
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        for record in records:
            name = record["name"]
            value = np.asarray(checkpoint.get_tensor(name))
            dtype = {"F16": "<f2", "I32": "<i4"}.get(record.get("dtype"))
            require(dtype is not None, f"{name} unsupported tensor dtype")
            require(list(value.shape) == record.get("shape"), f"{name} shape mismatch")
            raw = np.ascontiguousarray(value, dtype=dtype).tobytes()
            require(len(raw) == record.get("byte_length"), f"{name} byte length mismatch")
            payloads[name] = raw
    return payloads, records, indexed_layer_binding(layer_index)


def indexed_layer_tensor_value_hashes(
    checkpoint_path: Path,
    tensor_map_path: Path,
    layer_index: int,
) -> dict[str, str]:
    payloads, _, _ = _layer_tensor_payloads(
        checkpoint_path,
        tensor_map_path,
        layer_index,
    )
    return {
        name: sha256_bytes(payload)
        for name, payload in sorted(payloads.items())
    }


def indexed_layer_uses_accurate_silu(layer_index: int) -> bool:
    indexed_layer_binding(layer_index)
    return True


def sampled_indexed_q_projection_rows(
    checkpoint_path: Path,
    tensor_map_path: Path,
    handoff_path: Path,
    layer_index: int,
    channels: Sequence[int] = (0, 127, 895),
) -> list[dict[str, int]]:
    require(
        all(type(channel) is int and 0 <= channel < OFFICIAL_GEOMETRY.hidden_size
            for channel in channels),
        "sampled q_proj channel out of range",
    )
    payloads, _, binding = _layer_tensor_payloads(
        checkpoint_path,
        tensor_map_path,
        layer_index,
    )
    handoff, _ = load_two_token_handoff(handoff_path)
    prefix = binding["namespace"]

    def words(suffix: str, unit: int) -> list[int]:
        raw = payloads[prefix + suffix]
        dtype = "<u2" if unit == 2 else "<u4"
        return np.frombuffer(raw, dtype=dtype).astype(np.uint64).tolist()

    norm1 = rmsnorm(
        handoff[0],
        words("input_layernorm.weight", 2),
    )[0]
    require(not any(invalid for _, invalid, _ in norm1), "sampled input RMSNorm invalid")
    activation = [value for value, _, _ in norm1]
    qweight = words("self_attn.q_proj.qweight", 4)
    qzeros = words("self_attn.q_proj.qzeros", 4)
    scales = words("self_attn.q_proj.scales", 2)
    bias = words("self_attn.q_proj.bias", 2)
    groups = OFFICIAL_GEOMETRY.hidden_size // OFFICIAL_GEOMETRY.group_size
    packed_words = OFFICIAL_GEOMETRY.hidden_size // 8
    result: list[dict[str, int]] = []
    for channel in channels:
        packed, lane = divmod(channel, 8)
        _, value, invalid, _, _ = complete_projection_output(
            activation,
            [
                qweight[index * packed_words + packed]
                for index in range(OFFICIAL_GEOMETRY.hidden_size)
            ],
            [qzeros[group * packed_words + packed] for group in range(groups)],
            [scales[group * OFFICIAL_GEOMETRY.hidden_size + channel]
             for group in range(groups)],
            lane,
            bias[channel],
        )
        require(not invalid, f"sampled q_proj channel {channel} invalid")
        result.append({"channel": channel, "f16": value})
    return result


def materialize_indexed_decoder_vectors(
    checkpoint_path: Path,
    tensor_map_path: Path,
    handoff_path: Path,
    output_dir: Path,
    *,
    layer_index: int,
    expected_handoff_sha256: str | None = None,
    accurate_silu: bool | None = None,
) -> dict[str, Any]:
    indexed_layer_binding(layer_index)
    if expected_handoff_sha256 is None:
        require(
            layer_index == 1,
            "expected predecessor handoff SHA256 is required",
        )
        expected_handoff_sha256 = LAYER0_VL15_FINAL_ROWS_SHA256
    handoff, handoff_binding = load_two_token_handoff(
        handoff_path,
        expected_sha256=expected_handoff_sha256,
    )
    payloads, records, binding = _layer_tensor_payloads(
        checkpoint_path,
        tensor_map_path,
        layer_index,
    )
    require(not output_dir.exists(), "indexed decoder output directory already exists")
    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir(parents=True)
    prefix = binding["namespace"]
    values: dict[str, list[int]] = {}
    tensor_manifest: list[dict[str, Any]] = []
    for record in records:
        name = record["name"]
        suffix = name.removeprefix(prefix)
        dtype = record["dtype"]
        unit = 2 if dtype == "F16" else 4
        serialized = (
            f"layer{layer_index}_{suffix.replace('.', '_')}."
            f"{'fp16le.bin' if unit == 2 else 'i32le.bin'}"
        )
        raw = payloads[name]
        (tensor_dir / f"{serialized}.hex").write_text(
            "".join(
                f"{int.from_bytes(raw[index:index + unit], 'little'):0{unit * 2}x}\n"
                for index in range(0, len(raw), unit)
            ),
            encoding="ascii",
        )
        values[f"model.layers.0.{suffix}:"] = (
            np.frombuffer(raw, dtype="<u2" if unit == 2 else "<u4")
            .astype(np.uint64)
            .tolist()
        )
        tensor_manifest.append(
            {
                "checkpoint_name": name,
                "serialized_hex": f"tensors/{serialized}.hex",
                "dtype": dtype,
                "shape": record["shape"],
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )

    cache_k: list[list[int]] = []
    cache_v: list[list[int]] = []
    use_accurate_silu = (
        indexed_layer_uses_accurate_silu(layer_index)
        if accurate_silu is None
        else accurate_silu
    )
    all_trace: list[tuple[int, int, int, int, int]] = []
    final_rows: list[list[int]] = []
    for token, activation in enumerate(handoff):
        final, trace = run_decoder_layer_token(
            values,
            activation,
            token,
            cache_k,
            cache_v,
            use_accurate_silu,
        )
        final_rows.append(final)
        all_trace.extend(
            (token, stage, index, item, position)
            for stage, index, item, position in trace
        )

    inputs_payload = handoff_path.read_bytes()
    (output_dir / "inputs.hex").write_bytes(inputs_payload)
    (output_dir / "trace.hex").write_text(
        "".join(
            f"{token:02x}{position:04x}{stage:02x}{index:04x}{item:04x}\n"
            for token, stage, index, item, position in all_trace
        ),
        encoding="ascii",
    )
    (output_dir / "final.hex").write_text(
        "".join(
            f"{token:02x}{index:04x}{item:04x}\n"
            for token, row in enumerate(final_rows)
            for index, item in enumerate(row)
        ),
        encoding="ascii",
    )
    rope = []
    for position in range(2):
        for pair in range(32):
            cosine, sine = qwen2_coefficient(position, pair)
            rope.append(f"{position:04x}{pair:02x}{cosine:04x}{sine:04x}\n")
    (output_dir / "rope_coefficients.hex").write_text("".join(rope), encoding="ascii")
    stage_counts: dict[str, int] = {}
    for _, stage, _, _, _ in all_trace:
        stage_counts[str(stage)] = stage_counts.get(str(stage), 0) + 1
    manifest = {
        "schema_version": 1,
        "kind": "ace3_indexed_decoder_layer_two_token_trace",
        "layer_index": layer_index,
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tensor_map_sha256": TENSOR_MAP_SHA256,
        },
        "layer_binding": binding,
        "input_handoff": (
            indexed_layer_input_handoff_binding(layer_index, handoff_binding)
            if layer_index
            else {
                **handoff_binding,
                "source": "authenticated official token embedding rows",
                "source_layer_index": None,
                "consumer_layer_index": 0,
                "byte_preserved_as": "inputs.hex",
            }
        ),
        "positions": [0, 1],
        "cache_slot": 0,
        "trace_records": len(all_trace),
        "final_records": sum(len(row) for row in final_rows),
        "stage_counts": stage_counts,
        "consumed_tensors": tensor_manifest,
        "numeric_profile": {
            "projection": "native asymmetric packed INT4 AWQ W4A16 G128 GEMM",
            "qzero_adjustment": "none",
            "activations": "FP16",
            "kv": "FP16",
            "silu": (
                "exp range-reduced degree-7 Q24"
                if use_accurate_silu
                else "accepted rational Q24"
            ),
        },
    }
    (output_dir / "boundary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return manifest


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
            "checkpoint": {
                "filename": "model.safetensors",
                "sha256": CHECKPOINT_SHA256,
                "bytes": CHECKPOINT_SIZE,
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
        "tokenizer_host": {
            "tokenizer": {
                "artifact": "tokenizer.json",
                "sha256": TOKENIZER_SHA256,
                "config_artifact": "tokenizer_config.json",
                "config_sha256": TOKENIZER_CONFIG_SHA256,
                "chat_template_profile": "fixed system/user messages without tools",
            },
            "prompt": {
                "messages": [
                    {"role": role, "content": content}
                    for role, content in FIXED_CHAT_MESSAGES
                ],
                "serialization": FIXED_CHAT_SERIALIZATION,
                "token_ids": list(FIXED_CHAT_TOKEN_IDS),
            },
            "reduced_execution_vocabulary": {
                "token_ids": sorted(REDUCED_EXECUTION_TOKEN_IDS),
                "generated_candidates": list(REDUCED_GENERATION_TOKEN_IDS),
                "outside_policy": "reject and label as outside_reduced_execution_vocabulary",
                "boundary": (
                    "fixed structural token-flow profile only; token membership does "
                    "not authenticate embedding rows or official logits"
                ),
            },
            "generation": {
                "selection": "argmax over labeled reduced-logit fixture",
                "fixture_is_official_logits": False,
                "response_token_ids": list(REDUCED_RESPONSE_TOKEN_IDS),
                "eos_token_id": EOS_TOKEN_ID,
                "decoded_response": "Hello world",
                "default_max_new_tokens": 3,
            },
            "cache_slot_reuse": {
                "slot": 0,
                "position_rule": "advance once per prompt or generated token",
                "boundary": (
                    "host token-position lineage only; numerical FP16 KV contents "
                    "are not produced by this structural execution"
                ),
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
            "review_provenance": "layer0_decoder_token_boundary",
            "status": (
                "authenticated layer-0 RTL and Icarus width-boundary simulation; "
                "not 24-layer execution"
            ),
        },
        "numeric_profile": {
            "projection": "native asymmetric packed INT4 AWQ W4A16 G128 GEMM",
            "packed_nibble_order": [0, 4, 1, 5, 2, 6, 3, 7],
            "qzero_adjustment": "none",
            "scales": "FP16",
            "activations": "FP16",
            "kv": "FP16",
        },
        "official_single_decoder_layer": official_single_decoder_layer_contract(),
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
                "execution_stream": "FP16 tied linear projection",
                "storage_boundary": (
                    "official F16 checkpoint storage; no AWQ-packed lm_head claim"
                ),
            },
            "argmax": {
                "input_order": "token_id ascending from 0 through 151935",
                "selection": "greatest valid logit",
                "tie_break": "lowest token_id",
                "completion": "exactly 151936 logits accepted",
                "invalid": "fault; no token result",
            },
            "official_token_decision": {
                "consumed_checkpoint_tensors": [
                    {
                        "name": "lm_head.weight",
                        "sha256": TIED_WEIGHT_SHA256,
                        "purpose": "full-vocabulary FP16 linear projection",
                    },
                    {
                        "name": "model.embed_tokens.weight",
                        "sha256": TIED_WEIGHT_SHA256,
                        "purpose": "byte-for-byte tied-head provenance",
                    },
                    {
                        "name": "model.norm.weight",
                        "sha256": FINAL_NORM_SHA256,
                        "purpose": "final RMSNorm",
                    },
                ],
                "terminal_hidden_state": {
                    "source": "deterministic structural fixture",
                    "features": 896,
                    "dtype": "F16",
                    "algorithm": (
                        "q24_to_f16((((index * 73) % 513) - 256) << 14)"
                    ),
                    "sha256": FIXED_TERMINAL_HIDDEN_SHA256,
                    "engine_produced": False,
                },
                "arithmetic": {
                    "rmsnorm": (
                        "integer-only Q24/Q48 oracle with epsilon 1e-6 and "
                        "binary16 RNE"
                    ),
                    "lm_head": (
                        "exact FP16 products accumulated in signed Q47.48; "
                        "round once to binary16 RNE"
                    ),
                },
                "vocab_size": 151936,
                "top_k": OFFICIAL_TOP_K,
                "tie_break": "lowest token_id",
                "host_boundary": (
                    "authenticated tokenizer decodes the selected official "
                    "full-vocabulary token"
                ),
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
                "fixed parent, reviewed map, and authenticated layer-0 RTL/TB hashes",
                "Icarus layer-0 width, reset, clear, and fail-closed boundary",
                "authenticated tokenizer prompt serialization and deterministic decoding",
                "multi-token structural host stepping and cache-slot position lineage",
                (
                    "official checkpoint final RMSNorm and tied lm_head over a "
                    "fixed structural hidden-state fixture"
                ),
                (
                    "deterministic 151936-logit top-k and argmax token decision "
                    "with authenticated host decoding"
                ),
                (
                    "official layer-0 two-token numerical execution with authenticated "
                    "embeddings, norms, native-AWQ projections, RoPE, FP16 KV cache, "
                    "causal attention, residuals, and MLP"
                ),
                (
                    "independent PyTorch stage comparisons within explicit absolute "
                    "tolerances and 42 sampled exact projection bit-oracle checks"
                ),
            ],
            "excluded": [
                "layers 1 through 23 RTL execution",
                "layers 1 through 23 numerical execution",
                "full 24-layer numerical execution or dialogue",
                "official numerical layer-23 terminal hidden state",
                "readable multi-token official-checkpoint dialogue",
                "latency, throughput, synthesis, PPA, FPGA, or silicon",
                "full-model or dialogue acceptance",
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
            "algorithm": "model24-execution-v3",
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
            "tokenizer_sha256": TOKENIZER_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "lm_head_sha256": TIED_WEIGHT_SHA256,
            "embed_tokens_sha256": TIED_WEIGHT_SHA256,
            "final_norm_sha256": FINAL_NORM_SHA256,
            "terminal_hidden_state_sha256": FIXED_TERMINAL_HIDDEN_SHA256,
        },
        "artifact_set": [
            "host_generation.json",
            "manifest.json",
            "official_layer0_slice.json",
            "official_token_decision.json",
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


def validate_decoder_snapshot(repository_root: Path) -> None:
    for relative_path, expected_sha256 in (
        (
            "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
            DECODER_SOURCE_SHA256,
        ),
        (
            "ace3/tb/ace3_decoder_width_boundary_tb.sv",
            DECODER_INTERFACE_SHA256,
        ),
    ):
        path = repository_root / relative_path
        require(
            sha256_bytes(path.read_bytes()) == expected_sha256,
            f"{relative_path} SHA256 mismatch",
        )


def authenticate_tokenizer(tokenizer_dir: Path) -> Any:
    require(
        tokenizer_dir.is_dir(),
        (
            "official tokenizer directory is missing; pass "
            "--official-tokenizer-dir or set "
            "ACE3_OFFICIAL_MODEL24_TOKENIZER_DIR"
        ),
    )
    tokenizer_payload = (tokenizer_dir / "tokenizer.json").read_bytes()
    config_payload = (tokenizer_dir / "tokenizer_config.json").read_bytes()
    require(
        sha256_bytes(tokenizer_payload) == TOKENIZER_SHA256,
        "official tokenizer.json SHA256 mismatch",
    )
    require(
        sha256_bytes(config_payload) == TOKENIZER_CONFIG_SHA256,
        "official tokenizer_config.json SHA256 mismatch",
    )
    config = load_json_bytes(config_payload, "tokenizer_config.json")
    require(config.get("tokenizer_class") == "Qwen2Tokenizer", "tokenizer class mismatch")
    require(config.get("eos_token") == "<|im_end|>", "tokenizer EOS mismatch")
    require(
        config.get("added_tokens_decoder", {})
        .get(str(EOS_TOKEN_ID), {})
        .get("content")
        == "<|im_end|>",
        "tokenizer EOS token ID mismatch",
    )
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ContractError("the tokenizers package is required") from error
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    encoded = tokenizer.encode(FIXED_CHAT_SERIALIZATION, add_special_tokens=False).ids
    require(encoded == list(FIXED_CHAT_TOKEN_IDS), "fixed chat prompt tokenization mismatch")
    require(
        tokenizer.decode(encoded, skip_special_tokens=False) == FIXED_CHAT_SERIALIZATION,
        "fixed chat prompt decode mismatch",
    )
    require(
        tokenizer.decode(list(REDUCED_RESPONSE_TOKEN_IDS), skip_special_tokens=False)
        == "Hello world",
        "fixed reduced response decode mismatch",
    )
    return tokenizer


def serialize_chat_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    actual = [
        {"role": message.get("role"), "content": message.get("content")}
        for message in messages
    ]
    expected = [
        {"role": role, "content": content}
        for role, content in FIXED_CHAT_MESSAGES
    ]
    require_document(expected, actual, "fixed chat messages")
    return FIXED_CHAT_SERIALIZATION


def reduced_token_label(token_id: int) -> str:
    require(type(token_id) is int, "token_id must be an integer")
    require(
        token_id in REDUCED_EXECUTION_TOKEN_IDS,
        (
            f"token_id {token_id} is outside_reduced_execution_vocabulary; "
            "official embedding/logit execution is not available"
        ),
    )
    labels: list[str] = []
    if token_id in FIXED_CHAT_TOKEN_IDS:
        labels.append("prompt")
    if token_id in REDUCED_RESPONSE_TOKEN_IDS:
        labels.append("generated_content")
    if token_id == EOS_TOKEN_ID:
        labels.append("eos")
    return "+".join(labels)


class StructuralModel24Host:
    """Token-position host around the published structural execution machine."""

    def __init__(
        self,
        cache_slot: int = 0,
        geometry: Geometry = OFFICIAL_GEOMETRY,
    ) -> None:
        geometry.validate()
        require(
            type(cache_slot) is int and 0 <= cache_slot < geometry.cache_slots,
            "cache_slot outside supported range",
        )
        self.cache_slot = cache_slot
        self.geometry = geometry
        self.steps: list[dict[str, Any]] = []

    def step(self, token_id: int, phase: str) -> dict[str, Any]:
        label = reduced_token_label(token_id)
        require(phase in ("prompt", "generated"), "invalid host token phase")
        position = len(self.steps)
        require(position < self.geometry.positions, "host position capacity exceeded")
        schedule = expected_schedule(self.geometry)
        machine = ExecutionMachine(self.geometry)
        for event in schedule:
            machine.accept(event)
        require(machine.done, "structural Model24 token execution did not complete")
        record = {
            "ordinal": position,
            "position": position,
            "cache_slot": self.cache_slot,
            "cache_slot_reused": position > 0,
            "phase": phase,
            "token_id": token_id,
            "reduced_vocabulary_label": label,
            "schedule_event_count": machine.cursor,
            "first_operation": schedule[0]["operation"],
            "last_operation": schedule[-1]["operation"],
            "execution_boundary": "structural schedule only; no official logits",
        }
        self.steps.append(record)
        return record


def host_generation_document(
    tokenizer_dir: Path,
    *,
    max_new_tokens: int = 3,
    cache_slot: int = 0,
) -> dict[str, Any]:
    require(
        type(max_new_tokens) is int and max_new_tokens > 0,
        "max_new_tokens must be a positive integer",
    )
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    messages = [
        {"role": role, "content": content}
        for role, content in FIXED_CHAT_MESSAGES
    ]
    prompt = serialize_chat_prompt(messages)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    require(prompt_ids == list(FIXED_CHAT_TOKEN_IDS), "host prompt IDs mismatch")

    host = StructuralModel24Host(cache_slot)
    for token_id in prompt_ids:
        host.step(token_id, "prompt")

    generated_ids: list[int] = []
    selection_records: list[dict[str, Any]] = []
    stop_reason = "max_new_tokens"
    for fixture_ordinal, logits in enumerate(REDUCED_LOGIT_FIXTURE):
        if len(generated_ids) == max_new_tokens:
            break
        candidate_index = argmax_first(logits)
        token_id = REDUCED_GENERATION_TOKEN_IDS[candidate_index]
        host_step = host.step(token_id, "generated")
        generated_ids.append(token_id)
        selection_records.append(
            {
                "fixture_ordinal": fixture_ordinal,
                "candidate_token_ids": list(REDUCED_GENERATION_TOKEN_IDS),
                "reduced_fixture_logits": list(logits),
                "selected_token_id": token_id,
                "host_step_ordinal": host_step["ordinal"],
                "logit_boundary": "deterministic test fixture; not official logits",
            }
        )
        if token_id == EOS_TOKEN_ID:
            stop_reason = "eos_token"
            break

    require(
        stop_reason == "eos_token" or len(generated_ids) == max_new_tokens,
        "reduced generation fixture exhausted before a stop condition",
    )
    decoded_ids = [
        token_id for token_id in generated_ids if token_id != EOS_TOKEN_ID
    ]
    decoded_text = tokenizer.decode(decoded_ids, skip_special_tokens=False)
    output_payload = decoded_text.encode("utf-8")
    return {
        "schema_version": 1,
        "kind": "ace3_model24_reduced_tokenizer_host_generation",
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
        },
        "tokenizer_binding": {
            "tokenizer_artifact": "tokenizer.json",
            "tokenizer_sha256": TOKENIZER_SHA256,
            "config_artifact": "tokenizer_config.json",
            "config_sha256": TOKENIZER_CONFIG_SHA256,
        },
        "prompt": {
            "messages": messages,
            "serialization": prompt,
            "serialization_utf8_sha256": sha256_bytes(prompt.encode("utf-8")),
            "token_ids": prompt_ids,
            "decoded_roundtrip": tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
            ),
        },
        "reduced_execution_vocabulary": {
            "token_ids": sorted(REDUCED_EXECUTION_TOKEN_IDS),
            "outside_policy": "reject with outside_reduced_execution_vocabulary label",
            "boundary": (
                "fixed structural profile; membership does not authenticate "
                "official embedding rows or logits"
            ),
        },
        "token_flow": host.steps,
        "generation_selection": selection_records,
        "cache_slot_flow": {
            "cache_slot": cache_slot,
            "positions": [step["position"] for step in host.steps],
            "reuse_count": sum(
                1 for step in host.steps if step["cache_slot_reused"]
            ),
            "boundary": (
                "host token-position lineage only; numerical FP16 KV contents "
                "are unclaimed"
            ),
        },
        "stop": {
            "reason": stop_reason,
            "eos_token_id": EOS_TOKEN_ID,
            "eos_emitted": bool(generated_ids and generated_ids[-1] == EOS_TOKEN_ID),
            "max_new_tokens": max_new_tokens,
        },
        "output": {
            "generated_token_ids": generated_ids,
            "decoded_token_ids": decoded_ids,
            "decoded_text": decoded_text,
            "decoded_utf8_sha256": sha256_bytes(output_payload),
        },
        "claim_boundary": (
            "deterministic tokenizer and structural Model24 host evidence only; "
            "official-geometry logits and readable official-checkpoint dialogue "
            "remain unclaimed"
        ),
    }


def validate_vector_bindings(bindings: Any, contract_sha256: str) -> None:
    require(isinstance(bindings, dict), "vector bindings root must be an object")
    require_document(
        vector_bindings_document(contract_sha256),
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


def fixed_terminal_hidden_state_bits() -> list[int]:
    values: list[int] = []
    for index in range(OFFICIAL_GEOMETRY.hidden_size):
        bits, saturated = q24_to_f16((((index * 73) % 513) - 256) << 14)
        require(not saturated, "fixed terminal hidden-state fixture saturated")
        values.append(bits)
    return values


def _tensor_records() -> dict[str, dict[str, Any]]:
    return {
        record["name"]: record
        for record in expected_tensor_records()
        if record["name"] in FINAL_PROJECTION_TENSOR_NAMES
    }


def _sha256_range(path: Path, offset: int, byte_length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = byte_length
        while remaining:
            payload = stream.read(min(1024 * 1024, remaining))
            require(payload, "checkpoint tensor payload is truncated")
            digest.update(payload)
            remaining -= len(payload)
    return digest.hexdigest()


def authenticate_final_projection_tensors(
    checkpoint_path: Path,
) -> dict[str, dict[str, Any]]:
    try:
        authenticate_checkpoint(checkpoint_path)
    except Model24OracleContractError as error:
        raise ContractError(f"official checkpoint authentication failed: {error}") from error
    records = _tensor_records()
    require(
        set(records) == set(FINAL_PROJECTION_TENSOR_NAMES),
        "final projection tensor inventory mismatch",
    )
    expected_hashes = {
        "lm_head.weight": TIED_WEIGHT_SHA256,
        "model.embed_tokens.weight": TIED_WEIGHT_SHA256,
        "model.norm.weight": FINAL_NORM_SHA256,
    }
    bindings: dict[str, dict[str, Any]] = {}
    for name in FINAL_PROJECTION_TENSOR_NAMES:
        record = records[name]
        start, end = record["absolute_file_offsets"]
        actual_sha256 = _sha256_range(checkpoint_path, start, end - start)
        require(
            actual_sha256 == expected_hashes[name],
            f"{name} payload SHA256 mismatch",
        )
        bindings[name] = {
            "dtype": record["dtype"],
            "shape": record["shape"],
            "absolute_file_offsets": record["absolute_file_offsets"],
            "bytes": record["byte_length"],
            "sha256": actual_sha256,
        }
    require(
        bindings["lm_head.weight"]["absolute_file_offsets"]
        != bindings["model.embed_tokens.weight"]["absolute_file_offsets"],
        "tied checkpoint tensors must occupy distinct ranges",
    )
    require(
        bindings["lm_head.weight"]["sha256"]
        == bindings["model.embed_tokens.weight"]["sha256"],
        "tied checkpoint tensor hashes differ",
    )
    return bindings


def _read_f16_tensor_bits(
    checkpoint_path: Path,
    record: Mapping[str, Any],
) -> np.ndarray:
    start, end = record["absolute_file_offsets"]
    with checkpoint_path.open("rb") as stream:
        stream.seek(start)
        payload = stream.read(end - start)
    require(len(payload) == end - start, "checkpoint tensor payload is truncated")
    values = np.frombuffer(payload, dtype="<u2").copy()
    require(
        values.size == record["byte_length"] // 2,
        "checkpoint tensor element count mismatch",
    )
    return values.reshape(tuple(record["shape"]))


def _decode_f16_array_q24(bits: np.ndarray) -> np.ndarray:
    unsigned = bits.astype(np.uint16, copy=False)
    exponent = ((unsigned >> 10) & 0x1F).astype(np.int64)
    require(
        bool(np.all(exponent != 0x1F)),
        "official final projection tensor contains non-finite FP16",
    )
    fraction = (unsigned & 0x03FF).astype(np.int64)
    shifts = np.maximum(exponent - 1, 0)
    normal = np.left_shift(0x0400 | fraction, shifts)
    magnitude = np.where(exponent == 0, fraction, normal)
    negative = (unsigned & 0x8000) != 0
    return np.where(negative, -magnitude, magnitude).astype(np.int64)


def exact_tied_lm_head_logits(
    checkpoint_path: Path,
    normalized_hidden_f16: Sequence[int],
    *,
    rows_per_chunk: int = 512,
) -> list[int]:
    require(
        len(normalized_hidden_f16) == OFFICIAL_GEOMETRY.hidden_size,
        "lm_head hidden-state width mismatch",
    )
    require(
        type(rows_per_chunk) is int and rows_per_chunk > 0,
        "rows_per_chunk must be a positive integer",
    )
    activation_bits = np.asarray(normalized_hidden_f16, dtype="<u2")
    activation_q24 = _decode_f16_array_q24(activation_bits)
    sum_abs_activation = sum(abs(int(value)) for value in activation_q24)
    records = _tensor_records()
    lm_head = records["lm_head.weight"]
    start, _ = lm_head["absolute_file_offsets"]
    weight_bits = np.memmap(
        checkpoint_path,
        dtype="<u2",
        mode="r",
        offset=start,
        shape=tuple(lm_head["shape"]),
        order="C",
    )
    logits: list[int] = []
    int64_max = np.iinfo(np.int64).max
    for row_start in range(0, OFFICIAL_GEOMETRY.vocab_size, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, OFFICIAL_GEOMETRY.vocab_size)
        weight_q24 = _decode_f16_array_q24(weight_bits[row_start:row_end])
        max_abs_weight = int(np.max(np.abs(weight_q24)))
        require(
            max_abs_weight * sum_abs_activation <= int64_max,
            "exact lm_head Q47.48 accumulation exceeds signed int64 transport",
        )
        accumulators = np.sum(
            weight_q24 * activation_q24,
            axis=1,
            dtype=np.int64,
        )
        for accumulator in accumulators:
            bits, saturated = q47_48_to_f16(int(accumulator))
            require(not saturated, "official lm_head logit saturated")
            logits.append(bits)
    require(
        len(logits) == OFFICIAL_GEOMETRY.vocab_size,
        "official lm_head did not produce the full vocabulary",
    )
    return logits


def stable_top_k_f16(logits_f16: Sequence[int], top_k: int) -> list[int]:
    require(
        len(logits_f16) == OFFICIAL_GEOMETRY.vocab_size,
        "token decision requires exactly 151936 logits",
    )
    require(
        type(top_k) is int and 0 < top_k <= len(logits_f16),
        "top_k outside full-vocabulary bounds",
    )
    values: list[int] = []
    for bits in logits_f16:
        value, finite, _, _ = decode_f16_q24(bits)
        require(finite, "token decision received a non-finite FP16 logit")
        values.append(value)
    return sorted(
        range(len(values)),
        key=lambda token_id: (-values[token_id], token_id),
    )[:top_k]


class Model24TokenDecisionHost:
    """Authenticated host boundary for one full-vocabulary token decision."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def decide(
        self,
        logits_f16: Sequence[int],
        *,
        top_k: int = OFFICIAL_TOP_K,
    ) -> dict[str, Any]:
        token_ids = stable_top_k_f16(logits_f16, top_k)
        records = []
        for rank, token_id in enumerate(token_ids):
            value_q24, finite, _, _ = decode_f16_q24(logits_f16[token_id])
            require(finite, "selected token has a non-finite logit")
            records.append(
                {
                    "rank": rank,
                    "token_id": token_id,
                    "logit_f16_bits": logits_f16[token_id],
                    "logit_q24": value_q24,
                    "decoded_token": self.tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                    ),
                }
            )
        return {
            "vocab_size": len(logits_f16),
            "top_k": records,
            "argmax_token_id": token_ids[0],
            "argmax_decoded_token": records[0]["decoded_token"],
            "tie_break": "lowest token_id",
            "host_integration": (
                "selected token is decoded by the authenticated official tokenizer"
            ),
        }


def official_token_decision_document(
    checkpoint_path: Path,
    tokenizer_dir: Path,
) -> dict[str, Any]:
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    tensor_bindings = authenticate_final_projection_tensors(checkpoint_path)
    records = _tensor_records()
    hidden_bits = fixed_terminal_hidden_state_bits()
    hidden_payload = np.asarray(hidden_bits, dtype="<u2").tobytes()
    require(
        sha256_bytes(hidden_payload) == FIXED_TERMINAL_HIDDEN_SHA256,
        "fixed terminal hidden-state SHA256 mismatch",
    )
    norm_weight_bits = _read_f16_tensor_bits(
        checkpoint_path,
        records["model.norm.weight"],
    ).tolist()
    norm_outputs, mean_q48, rms_q24 = rmsnorm(hidden_bits, norm_weight_bits)
    require(
        all(not invalid and not saturated for _, invalid, saturated in norm_outputs),
        "official final RMSNorm produced an invalid or saturated value",
    )
    normalized_bits = [bits for bits, _, _ in norm_outputs]
    logits_bits = exact_tied_lm_head_logits(checkpoint_path, normalized_bits)
    logits_payload = np.asarray(logits_bits, dtype="<u2").tobytes()
    decision = Model24TokenDecisionHost(tokenizer).decide(logits_bits)
    return {
        "schema_version": 1,
        "kind": "ace3_model24_official_final_token_decision",
        "model_binding": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint": {
                "filename": "model.safetensors",
                "sha256": CHECKPOINT_SHA256,
                "bytes": CHECKPOINT_SIZE,
            },
            "consumed_tensors": tensor_bindings,
            "tied_head_provenance": {
                "lm_head_tensor": "lm_head.weight",
                "embedding_tensor": "model.embed_tokens.weight",
                "storage": "distinct non-overlapping checkpoint ranges",
                "binding": "authenticated byte-for-byte value equality",
                "value_sha256": TIED_WEIGHT_SHA256,
            },
        },
        "terminal_hidden_state": {
            "source": "deterministic structural fixture",
            "engine_produced": False,
            "dtype": "F16",
            "features": len(hidden_bits),
            "f16_bits": hidden_bits,
            "sha256": sha256_bytes(hidden_payload),
            "claim_boundary": (
                "not an official numerical layer-23 output; it exercises only "
                "the authenticated final token-decision slice"
            ),
        },
        "final_rmsnorm": {
            "tensor": "model.norm.weight",
            "epsilon_q48": 281_474_977,
            "mean_q48": mean_q48,
            "rms_q24": rms_q24,
            "output_f16_bits": normalized_bits,
            "output_sha256": sha256_bytes(
                np.asarray(normalized_bits, dtype="<u2").tobytes()
            ),
        },
        "lm_head": {
            "tensor": "lm_head.weight",
            "tied_to": "model.embed_tokens.weight",
            "input_features": OFFICIAL_GEOMETRY.hidden_size,
            "vocab_size": OFFICIAL_GEOMETRY.vocab_size,
            "logit_dtype": "F16",
            "accumulator": "exact signed Q47.48",
            "rounding": "round once to binary16 RNE",
            "logits_f16_bits": logits_bits,
            "logits_sha256": sha256_bytes(logits_payload),
        },
        "token_decision": decision,
        "claim_boundary": (
            "official checkpoint final RMSNorm, tied lm_head, full-vocabulary "
            "logits, top-k, argmax, and tokenizer decode for a fixed structural "
            "hidden-state fixture; no official 24-layer numerical execution, "
            "multi-token generation, or dialogue is claimed"
        ),
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
    tokenizer_dir: Path,
    checkpoint_path: Path = DEFAULT_OFFICIAL_CHECKPOINT,
) -> dict[str, bytes]:
    try:
        layer0_document = official_single_decoder_layer_document(checkpoint_path)
    except LayerExecutionError as error:
        raise ContractError(f"official layer-0 execution failed: {error}") from error
    documents = {
        "host_generation.json": host_generation_document(tokenizer_dir),
        "official_layer0_slice.json": layer0_document,
        "official_token_decision.json": official_token_decision_document(
            checkpoint_path,
            tokenizer_dir,
        ),
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
        "algorithm": "model24-execution-v3",
        "seed": 240483,
        "inputs": {
            "parent_commit": PARENT_COMMIT,
            "execution_contract_sha256": contract_sha256,
            "vector_bindings_sha256": bindings_sha256,
            "tensor_map_sha256": TENSOR_MAP_SHA256,
            "control_map_sha256": CONTROL_MAP_SHA256,
            "decoder_source_sha256": DECODER_SOURCE_SHA256,
            "decoder_interface_sha256": DECODER_INTERFACE_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "lm_head_sha256": TIED_WEIGHT_SHA256,
            "embed_tokens_sha256": TIED_WEIGHT_SHA256,
            "final_norm_sha256": FINAL_NORM_SHA256,
            "terminal_hidden_state_sha256": FIXED_TERMINAL_HIDDEN_SHA256,
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
            "official_layer0_tokens": 2,
            "official_layer0_stages": len(layer0_document["intermediates"]),
            "official_layer0_projection_bit_checks": layer0_document[
                "independent_reference"
            ]["sampled_projection_bit_oracle_checks"],
            "official_layer0_handoff_sha256": layer0_document[
                "final_token_decision_handoff"
            ]["sha256"],
            "small_geometry_cases": 3,
            "small_geometry_max_layers": 3,
            "host_prompt_tokens": len(FIXED_CHAT_TOKEN_IDS),
            "host_generated_tokens_including_eos": 3,
            "host_total_structural_token_steps": len(FIXED_CHAT_TOKEN_IDS) + 3,
            "official_vocab_size": OFFICIAL_GEOMETRY.vocab_size,
            "official_logits": OFFICIAL_GEOMETRY.vocab_size,
            "official_top_k": OFFICIAL_TOP_K,
            "official_argmax_token_id": documents[
                "official_token_decision.json"
            ]["token_decision"]["argmax_token_id"],
        },
        "verification_boundary": (
            "official numerical execution is established for layer 0 over two fixed "
            "tokens and produces an FP16 handoff compatible with the accepted final "
            "token-decision interface; that final token decision still consumes a "
            "fixed structural hidden-state fixture because layers 1 through 23, an "
            "official numerical terminal state, multi-token generation, and readable "
            "official-checkpoint dialogue remain unclaimed"
        ),
    }
    return {"manifest.json": canonical_json_bytes(manifest), **artifacts}
