#!/usr/bin/env python3
"""Validate exact decoder layer-0 vector inventory and independently rerun it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from decoder_layer0_oracle import load_authenticated_tensors, run_token
from qwen2_rope_oracle import qwen2_coefficient


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def files(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def raw_hex(raw: bytes, unit: int) -> str:
    return "".join(f"{int.from_bytes(raw[index:index + unit], 'little'):0{unit * 2}x}\n"
                   for index in range(0, len(raw), unit))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-tensor-dir", required=True, type=Path)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    args = parser.parse_args()
    source, generated = args.official_tensor_dir.resolve(strict=True), args.generated_dir.resolve(strict=True)
    binding = json.loads(args.bindings.read_text())
    require(binding.get("kind") == "ace3_decoder_layer0_vector_bindings", "binding kind")
    committed = {item["file"]: item for item in binding["serialized_artifacts"]}
    actual = files(generated)
    require(actual == set(committed), "artifact inventory")
    for name, item in committed.items():
        payload = (generated / name).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == item["sha256"], f"{name} SHA256")
        require(len(payload) == item["byte_count"] and payload.count(b"\n") == item["line_count"],
                f"{name} shape")
    tensor_values, manifest = load_authenticated_tensors(source)
    require(binding["expected_source_manifest_sha256"] ==
            hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(), "source manifest hash")
    cache_k: list[list[int]] = []
    cache_v: list[list[int]] = []
    trace: list[tuple[int, int, int, int, int]] = []
    final: list[list[int]] = []
    for token in range(2):
        vector, records = run_token(tensor_values, tensor_values[f"model.embed_tokens.weight:{token}"],
                                    token, cache_k, cache_v)
        final.append(vector)
        trace.extend((token, stage, index, item, position)
                     for stage, index, item, position in records)
    expected_trace = "".join(f"{token:02x}{position:04x}{stage:02x}{index:04x}{item:04x}\n"
                             for token, stage, index, item, position in trace)
    expected_final = "".join(f"{token:02x}{index:04x}{item:04x}\n"
                             for token, vector in enumerate(final)
                             for index, item in enumerate(vector))
    require((generated / "trace.hex").read_text() == expected_trace, "trace oracle")
    require((generated / "final.hex").read_text() == expected_final, "final oracle")
    expected_rope = "".join(
        f"{position:04x}{pair:02x}{qwen2_coefficient(position, pair)[0]:04x}"
        f"{qwen2_coefficient(position, pair)[1]:04x}\n"
        for position in range(2) for pair in range(32))
    require((generated / "rope_coefficients.hex").read_text() == expected_rope, "rope oracle")
    expected_inputs = "".join(
        f"{token:02x}{index:04x}{item:04x}\n"
        for token in range(2) for index, item in enumerate(
            tensor_values[f"model.embed_tokens.weight:{token}"]))
    require((generated / "inputs.hex").read_text() == expected_inputs, "input source serialization")
    for item in manifest["tensors"]:
        unit = 2 if item["dtype"] == "float16" else 4
        name = item["serialized_file"]
        require((generated / "tensors" / f"{name}.hex").read_text() ==
                raw_hex((source / name).read_bytes(), unit), f"tensor source serialization: {name}")
    boundary = json.loads((generated / "boundary_manifest.json").read_text())
    require(boundary["trace_records"] == len(trace) and boundary["final_records"] == 1792 and
            boundary["token_ids"] == [9707, 1879], "boundary manifest")
    print("DECODER_LAYER0_VECTOR_VALIDATION_PASS artifacts="
          f"{len(committed)} trace_records={len(trace)} final_records=1792 "
          "tokens=2 cache_reuse=pass sha256=pass source=pass oracle=pass")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError, ArithmeticError) as error:
        print(f"DECODER_LAYER0_VECTOR_VALIDATION_FAIL {error}", file=sys.stderr)
        raise SystemExit(1) from error
