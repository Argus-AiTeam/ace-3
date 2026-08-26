#!/usr/bin/env python3
"""Focused non-vacuous checks for layer-index selection and the vl15 handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--handoff", required=True, type=Path)
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    sys.path.insert(0, str(root / "ace3" / "model"))

    from fp16_adaptation_oracle import rmsnorm  # pylint: disable=import-outside-toplevel
    from model24_execution_oracle import (  # pylint: disable=import-outside-toplevel
        LAYER_DESCRIPTOR_SHA256,
        OFFICIAL_GEOMETRY,
        TENSOR_MAP_SHA256,
        layer_bindings,
    )
    from model24_oracle import authenticate_checkpoint  # pylint: disable=import-outside-toplevel
    from projection_oracle import complete_projection_output  # pylint: disable=import-outside-toplevel

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    def indexed_layer_binding(layer_index: int) -> dict[str, object]:
        require(
            type(layer_index) is int and 0 <= layer_index < len(LAYER_DESCRIPTOR_SHA256),
            "layer_index out of range",
        )
        return layer_bindings()[layer_index]

    def indexed_layer_tensor_records(
        tensor_map: dict[str, object],
        layer_index: int,
    ) -> list[dict[str, object]]:
        binding = indexed_layer_binding(layer_index)
        namespaces = tensor_map.get("layer_namespaces")
        require(isinstance(namespaces, list), "tensor map layer_namespaces missing")
        matches = [
            item
            for item in namespaces
            if isinstance(item, dict) and item.get("layer_id") == layer_index
        ]
        require(len(matches) == 1, "selected layer namespace is not unique")
        require(
            all(matches[0].get(key) == binding[key]
                for key in ("layer_id", "namespace", "descriptor_sha256")),
            "selected layer namespace binding mismatch",
        )
        require(matches[0].get("tensor_count") == 26, "selected layer tensor count mismatch")
        tensors = tensor_map.get("tensors")
        require(isinstance(tensors, list), "tensor map tensors missing")
        namespace = binding["namespace"]
        require(isinstance(namespace, str), "selected layer namespace type")
        records = [
            dict(item)
            for item in tensors
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and item["name"].startswith(namespace)
        ]
        require(len(records) == 26, "selected layer tensor count mismatch")
        return sorted(records, key=lambda item: str(item["name"]))

    def layer_payloads(
        tensor_map: dict[str, object],
        layer_index: int,
    ) -> tuple[dict[str, bytes], dict[str, object]]:
        binding = indexed_layer_binding(layer_index)
        records = indexed_layer_tensor_records(tensor_map, layer_index)
        from safetensors import safe_open  # pylint: disable=import-outside-toplevel

        payloads: dict[str, bytes] = {}
        with safe_open(args.checkpoint, framework="np") as checkpoint:
            for record in records:
                name = record["name"]
                require(isinstance(name, str), "tensor name type")
                value = np.asarray(checkpoint.get_tensor(name))
                dtype = {"F16": "<f2", "I32": "<i4"}.get(record.get("dtype"))
                require(dtype is not None, f"{name} unsupported tensor dtype")
                require(list(value.shape) == record.get("shape"), f"{name} shape mismatch")
                raw = np.ascontiguousarray(value, dtype=dtype).tobytes()
                require(len(raw) == record.get("byte_length"), f"{name} byte length mismatch")
                payloads[name] = raw
        return payloads, binding

    def load_two_token_handoff() -> tuple[list[list[int]], str]:
        payload = args.handoff.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        require(
            digest == "22768ac6b337f920faac7de59b4eb43a203e1db45cdf688820fcbb35cdfe3446",
            "vl15 two-token handoff SHA256 mismatch",
        )
        require(payload.endswith(b"\n"), "two-token handoff must be LF terminated")
        lines = payload.splitlines()
        require(len(lines) == 2 * OFFICIAL_GEOMETRY.hidden_size, "two-token handoff row count")
        rows = [[0] * OFFICIAL_GEOMETRY.hidden_size for _ in range(2)]
        for ordinal, line in enumerate(lines):
            require(len(line) == 10, f"two-token handoff row {ordinal} width")
            token = int(line[0:2], 16)
            index = int(line[2:6], 16)
            value = int(line[6:10], 16)
            require(
                (token, index) == divmod(ordinal, OFFICIAL_GEOMETRY.hidden_size),
                f"two-token handoff row {ordinal} is out of sequence",
            )
            rows[token][index] = value
        return rows, digest

    def sampled_indexed_q_projection_rows(
        tensor_map: dict[str, object],
        handoff: list[list[int]],
        layer_index: int,
    ) -> list[dict[str, int]]:
        payloads, binding = layer_payloads(tensor_map, layer_index)
        prefix = binding["namespace"]
        require(isinstance(prefix, str), "selected layer namespace type")

        def words(suffix: str, unit: int) -> list[int]:
            raw = payloads[prefix + suffix]
            dtype = "<u2" if unit == 2 else "<u4"
            return np.frombuffer(raw, dtype=dtype).astype(np.uint64).tolist()

        norm1 = rmsnorm(handoff[0], words("input_layernorm.weight", 2))[0]
        require(not any(invalid for _, invalid, _ in norm1), "sampled input RMSNorm invalid")
        activation = [value for value, _, _ in norm1]
        qweight = words("self_attn.q_proj.qweight", 4)
        qzeros = words("self_attn.q_proj.qzeros", 4)
        scales = words("self_attn.q_proj.scales", 2)
        bias = words("self_attn.q_proj.bias", 2)
        groups = OFFICIAL_GEOMETRY.hidden_size // OFFICIAL_GEOMETRY.group_size
        packed_words = OFFICIAL_GEOMETRY.hidden_size // 8
        result = []
        for channel in (0, 127, 895):
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

    tensor_map = json.loads(args.tensor_map.read_text(encoding="ascii"))
    assert hashlib.sha256(args.tensor_map.read_bytes()).hexdigest() == TENSOR_MAP_SHA256
    authenticate_checkpoint(args.checkpoint)
    layer0 = indexed_layer_binding(0)
    layer1 = indexed_layer_binding(1)
    assert layer0["namespace"] == "model.layers.0."
    assert layer1 == {
        "layer_id": 1,
        "namespace": "model.layers.1.",
        "descriptor_sha256":
            "c8a037c0043ededc764f02b14671781ceeb1fb5be3fa6b7f8e114d75a98ad8f4",
    }
    records0 = indexed_layer_tensor_records(tensor_map, 0)
    records1 = indexed_layer_tensor_records(tensor_map, 1)
    assert len(records0) == len(records1) == 26
    assert {item["name"].removeprefix(layer0["namespace"]) for item in records0} == {
        item["name"].removeprefix(layer1["namespace"]) for item in records1
    }
    payloads0, _ = layer_payloads(tensor_map, 0)
    payloads1, _ = layer_payloads(tensor_map, 1)
    hashes0 = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads0.items()}
    hashes1 = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads1.items()}
    suffix_hashes0 = {
        name.removeprefix(layer0["namespace"]): digest
        for name, digest in hashes0.items()
    }
    suffix_hashes1 = {
        name.removeprefix(layer1["namespace"]): digest
        for name, digest in hashes1.items()
    }
    changed_tensors = sorted(
        suffix for suffix in suffix_hashes0
        if suffix_hashes0[suffix] != suffix_hashes1[suffix]
    )
    assert changed_tensors

    handoff, handoff_sha256 = load_two_token_handoff()
    assert len(handoff) == 2 and all(len(row) == 896 for row in handoff)
    sample0 = sampled_indexed_q_projection_rows(
        tensor_map,
        handoff,
        0,
    )
    sample1 = sampled_indexed_q_projection_rows(
        tensor_map,
        handoff,
        1,
    )
    assert sample0 != sample1
    print(
        "MODEL24_LAYER_INDEXED_HANDOFF_PASS "
        f"layer1_descriptor={layer1['descriptor_sha256']} "
        f"changed_consumed_tensors={len(changed_tensors)} "
        f"handoff_shape=2x896 handoff_sha256={handoff_sha256} "
        f"layer0_q_rows={sample0} layer1_q_rows={sample1}"
    )


if __name__ == "__main__":
    main()
