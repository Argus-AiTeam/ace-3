#!/usr/bin/env python3
"""Extract and authenticate the fixed-revision Qwen2.5 AWQ layer-0 state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import zipfile
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
MODEL_SHA256 = "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
MODEL_BYTES = 730_652_248
MODEL_API_SHA256 = "9a4a3beea2283031c91d0de501fcb1a8613f9b5f5d6039111eac421833d5a768"
CONFIG_SHA256 = "bd20ae34a91eb38230b870d39f56677d1cda1e8b6688ad627e6efb6ca9f44090"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
TOKENIZER_CONFIG_SHA256 = (
    "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
)
TRANSFORMERS_VERSION = "4.41.1"
TRANSFORMERS_WHEEL_SHA256 = (
    "f0680e0b1a01067eccd11f62f0522409422c7d6f91d532fe0f50b136a406129d"
)
QWEN2_MODELING_SHA256 = (
    "90da2d6a23d53a4c62d1a16e9ab5c472c9145aac85689218161a54eef52b8105"
)
PROMPT = "Hello world"
TOKEN_IDS = [9707, 1879]
EMBEDDING_NAME = "model.embed_tokens.weight"
EMBEDDING_SHAPE = [151936, 896]

TENSORS: dict[str, tuple[str, list[int]]] = {
    "model.layers.0.input_layernorm.weight": ("float16", [896]),
    "model.layers.0.post_attention_layernorm.weight": ("float16", [896]),
}
for module, in_features, out_features in (
    ("self_attn.q_proj", 896, 896),
    ("self_attn.k_proj", 896, 128),
    ("self_attn.v_proj", 896, 128),
    ("self_attn.o_proj", 896, 896),
    ("mlp.gate_proj", 896, 4864),
    ("mlp.up_proj", 896, 4864),
    ("mlp.down_proj", 4864, 896),
):
    prefix = f"model.layers.0.{module}"
    TENSORS[f"{prefix}.qweight"] = ("int32", [in_features, out_features // 8])
    TENSORS[f"{prefix}.qzeros"] = ("int32", [in_features // 128, out_features // 8])
    TENSORS[f"{prefix}.scales"] = ("float16", [in_features // 128, out_features])
for module, out_features in (
    ("self_attn.q_proj", 896),
    ("self_attn.k_proj", 128),
    ("self_attn.v_proj", 128),
):
    TENSORS[f"model.layers.0.{module}.bias"] = ("float16", [out_features])

TORCH_DTYPES = {"float16": torch.float16, "int32": torch.int32}
SUFFIXES = {"float16": "fp16le.bin", "int32": "i32le.bin"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--official-tensor-dir", required=True, type=Path)
    parser.add_argument("--transformers-wheel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path, expected_hash: str) -> bytes:
    payload = path.read_bytes()
    actual = sha256_bytes(payload)
    if actual != expected_hash:
        raise RuntimeError(
            f"{path} SHA256 mismatch: expected {expected_hash}, got {actual}"
        )
    return payload


def tensor_filename(name: str, dtype: str) -> str:
    stem = name.removeprefix("model.layers.0.").replace(".", "_")
    return f"layer0_{stem}.{SUFFIXES[dtype]}"


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes(order="C")


def authenticate_sources(
    official_dir: Path, checkpoint: Path, wheel: Path
) -> dict[str, object]:
    if checkpoint.stat().st_size != MODEL_BYTES or sha256_file(checkpoint) != MODEL_SHA256:
        raise RuntimeError("fixed-revision model.safetensors identity mismatch")

    model_api_raw = checked_file(official_dir / "model-api.json", MODEL_API_SHA256)
    model_api = json.loads(model_api_raw)
    if (
        model_api.get("id") != MODEL_REPOSITORY
        or model_api.get("sha") != MODEL_REVISION
    ):
        raise RuntimeError("fixed-revision model API identity mismatch")
    siblings = {
        item.get("rfilename"): item
        for item in model_api.get("siblings", [])
        if isinstance(item, dict)
    }
    if (
        siblings.get("model.safetensors", {}).get("lfs", {}).get("sha256")
        != MODEL_SHA256
    ):
        raise RuntimeError("model API safetensors LFS identity mismatch")

    config_raw = checked_file(official_dir / "config.json", CONFIG_SHA256)
    config = json.loads(config_raw)
    expected_config = {
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32768,
        "rope_theta": 1_000_000.0,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError("fixed-revision Qwen2 geometry mismatch")
    quantization = config.get("quantization_config", {})
    if (
        quantization.get("bits") != 4
        or quantization.get("group_size") != 128
        or quantization.get("version") != "gemm"
        or quantization.get("zero_point") is not True
    ):
        raise RuntimeError("fixed-revision native AWQ profile mismatch")

    checked_file(official_dir / "tokenizer.json", TOKENIZER_SHA256)
    checked_file(official_dir / "tokenizer_config.json", TOKENIZER_CONFIG_SHA256)
    if sha256_file(wheel) != TRANSFORMERS_WHEEL_SHA256:
        raise RuntimeError("Transformers 4.41.1 wheel identity mismatch")
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(
            "transformers-4.41.1.dist-info/METADATA"
        ).decode("utf-8")
        if f"\nVersion: {TRANSFORMERS_VERSION}\n" not in f"\n{metadata}":
            raise RuntimeError("Transformers wheel version mismatch")
        qwen2_source = archive.read("transformers/models/qwen2/modeling_qwen2.py")
    if sha256_bytes(qwen2_source) != QWEN2_MODELING_SHA256:
        raise RuntimeError("Transformers 4.41.1 Qwen2 source identity mismatch")
    source_text = qwen2_source.decode("utf-8")
    for projection in ("q", "k", "v"):
        pattern = (
            rf"self\.{projection}_proj\s*=\s*nn\.Linear"
            rf"\([\s\S]{{0,240}}?bias=True[\s\S]{{0,20}}?\)"
        )
        if re.search(pattern, source_text) is None:
            raise RuntimeError(f"Qwen2 {projection}_proj bias semantics mismatch")

    return {
        "model_api_sha256": MODEL_API_SHA256,
        "config_sha256": CONFIG_SHA256,
        "model_safetensors_sha256": MODEL_SHA256,
        "model_safetensors_byte_count": MODEL_BYTES,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "transformers_version": TRANSFORMERS_VERSION,
        "transformers_wheel_sha256": TRANSFORMERS_WHEEL_SHA256,
        "qwen2_modeling_source_sha256": QWEN2_MODELING_SHA256,
    }


def main() -> None:
    args = parse_args()
    if sys.byteorder != "little":
        raise RuntimeError("serialized tensor contract requires little-endian storage")
    checkpoint = args.checkpoint.resolve(strict=True)
    official_dir = args.official_tensor_dir.resolve(strict=True)
    wheel = args.transformers_wheel.resolve(strict=True)
    sources = authenticate_sources(official_dir, checkpoint, wheel)

    tokenizer = Tokenizer.from_file(str(official_dir / "tokenizer.json"))
    encoding = tokenizer.encode(PROMPT, add_special_tokens=False)
    if encoding.ids != TOKEN_IDS:
        raise RuntimeError(
            f"authenticated tokenization mismatch: {encoding.ids} != {TOKEN_IDS}"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_records: list[dict[str, object]] = []
    embedding_records: list[dict[str, object]] = []
    with safe_open(checkpoint, framework="pt", device="cpu") as model:
        layer_keys = {key for key in model.keys() if key.startswith("model.layers.0.")}
        if layer_keys != set(TENSORS):
            missing = sorted(set(TENSORS) - layer_keys)
            unexpected = sorted(layer_keys - set(TENSORS))
            raise RuntimeError(
                f"layer-0 tensor inventory mismatch missing={missing} "
                f"unexpected={unexpected}"
            )
        embedding_slice = model.get_slice(EMBEDDING_NAME)
        if list(embedding_slice.get_shape()) != EMBEDDING_SHAPE:
            raise RuntimeError("embedding tensor shape mismatch")

        for name in sorted(TENSORS):
            dtype, shape = TENSORS[name]
            tensor = model.get_tensor(name)
            if tensor.dtype != TORCH_DTYPES[dtype] or list(tensor.shape) != shape:
                raise RuntimeError(
                    f"{name} metadata mismatch: dtype={tensor.dtype} "
                    f"shape={list(tensor.shape)}"
                )
            payload = tensor_bytes(tensor)
            filename = tensor_filename(name, dtype)
            (output_dir / filename).write_bytes(payload)
            tensor_records.append(
                {
                    "checkpoint_name": name,
                    "dtype": dtype,
                    "shape": shape,
                    "serialized_file": filename,
                    "byte_count": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            )

        for token_index, token_id in enumerate(TOKEN_IDS):
            row = embedding_slice[token_id : token_id + 1]
            if row.dtype != torch.float16 or list(row.shape) != [1, 896]:
                raise RuntimeError("embedding row metadata mismatch")
            payload = tensor_bytes(row)
            filename = f"token_{token_index}_id_{token_id}_embedding.fp16le.bin"
            (output_dir / filename).write_bytes(payload)
            embedding_records.append(
                {
                    "token_index": token_index,
                    "token_id": token_id,
                    "checkpoint_name": EMBEDDING_NAME,
                    "row_shape": [896],
                    "dtype": "float16",
                    "serialized_file": filename,
                    "byte_count": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            )

    manifest = {
        "schema_version": 1,
        "kind": "ace3_fixed_revision_decoder_layer0_tensors",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "sources": sources,
        "geometry": {
            "hidden_size": 896,
            "intermediate_size": 4864,
            "query_heads": 14,
            "key_value_heads": 2,
            "head_dim": 64,
            "group_size": 128,
            "max_position_embeddings": 32768,
            "rope_theta": 1_000_000.0,
        },
        "awq": {
            "bits": 4,
            "group_size": 128,
            "version": "gemm",
            "zero_point": True,
            "qzero_plus_one": False,
            "logical_lane_to_physical_nibble": [0, 4, 1, 5, 2, 6, 3, 7],
        },
        "transformers_qkv_bias_boundary": (
            "FP16 bias is added after each complete quantized projection output "
            "has rounded to FP16 and before Q/K RoPE or V cache insertion."
        ),
        "tokenization": {
            "prompt_utf8_hex": PROMPT.encode("utf-8").hex(),
            "add_special_tokens": False,
            "token_ids": TOKEN_IDS,
            "tokens": encoding.tokens,
        },
        "tensors": tensor_records,
        "token_embeddings": embedding_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "DECODER_LAYER0_EXTRACTION_PASS "
        f"tensors={len(tensor_records)} tokens={len(embedding_records)} "
        f"manifest_sha256={sha256_file(manifest_path)}"
    )


if __name__ == "__main__":
    main()
