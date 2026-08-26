#!/usr/bin/env python3
"""Materialize one authenticated, two-token decoder-layer-0 trace for SV."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from decoder_layer0_oracle import load_authenticated_tensors, run_token
from qwen2_rope_oracle import qwen2_coefficient


def _hex(raw: bytes, unit: int) -> str:
    return "".join(f"{int.from_bytes(raw[index:index + unit], 'little'):0{unit * 2}x}\n"
                   for index in range(0, len(raw), unit))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-tensor-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.official_tensor_dir.resolve(strict=True), args.output_dir.resolve()
    values, tensor_manifest = load_authenticated_tensors(source)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tensors").mkdir(exist_ok=True)
    cache_k: list[list[int]] = []
    cache_v: list[list[int]] = []
    all_trace: list[tuple[int, int, int, int, int]] = []
    finals: list[list[int]] = []
    for token in range(2):
        activation = values[f"model.embed_tokens.weight:{token}"]
        final, trace = run_token(values, activation, token, cache_k, cache_v)
        finals.append(final)
        all_trace.extend((token, stage, index, item, position)
                         for stage, index, item, position in trace)
    (output / "trace.hex").write_text(
        "".join(f"{token:02x}{position:04x}{stage:02x}{index:04x}{item:04x}\n"
                for token, stage, index, item, position in all_trace), encoding="ascii")
    (output / "final.hex").write_text(
        "".join(f"{token:02x}{index:04x}{item:04x}\n"
                for token, vector in enumerate(finals) for index, item in enumerate(vector)),
        encoding="ascii")
    (output / "inputs.hex").write_text(
        "".join(f"{token:02x}{index:04x}{item:04x}\n"
                for token in range(2)
                for index, item in enumerate(values[f"model.embed_tokens.weight:{token}"])),
        encoding="ascii")
    for name in ("layer0_input_layernorm_weight.fp16le.bin",
                 "layer0_post_attention_layernorm_weight.fp16le.bin"):
        (output / "tensors" / f"{name}.hex").write_text(
            _hex((source / name).read_bytes(), 2), encoding="ascii")
    for item in tensor_manifest["tensors"]:
        name = item["serialized_file"]
        unit = 2 if item["dtype"] == "float16" else 4
        relative = f"tensors/{name}.hex"
        (output / relative).write_text(_hex((source / name).read_bytes(), unit), encoding="ascii")
    rope = []
    for position in range(2):
        for pair in range(32):
            cos, sin = qwen2_coefficient(position, pair)
            rope.append(f"{position:04x}{pair:02x}{cos:04x}{sin:04x}\n")
    (output / "rope_coefficients.hex").write_text("".join(rope), encoding="ascii")
    counts: dict[str, int] = {}
    for _, stage, _, _, _ in all_trace:
        counts[str(stage)] = counts.get(str(stage), 0) + 1
    boundary = {
        "schema_version": 1,
        "kind": "ace3_decoder_layer0_two_token_trace",
        "model_repository": tensor_manifest["model_repository"],
        "model_revision": tensor_manifest["model_revision"],
        "source_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        "token_ids": tensor_manifest["tokenization"]["token_ids"],
        "positions": [0, 1],
        "cache_slot": 0,
        "trace_format": "token[7:0] position[14:0] stage[4:0] index[12:0] f16[15:0]",
        "trace_records": len(all_trace),
        "final_records": 1792,
        "stage_counts": counts,
        "projection_rounding": "AWQ Q53.48 round once, then Q/K/V FP16 bias",
        "rope": "Qwen2 half-split binary16 multiply/add",
        "attention": "14:2 GQA causal score/softmax/value over retained FP16 cache",
    }
    (output / "boundary_manifest.json").write_text(
        json.dumps(boundary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    (output / "decoder_layer0_params.svh").write_text(
        f"localparam integer DECODER_LAYER0_TRACE_RECORDS = {len(all_trace)};\n"
        "localparam integer DECODER_LAYER0_FINAL_RECORDS = 1792;\n"
        "localparam integer DECODER_LAYER0_TOKENS = 2;\n"
        "localparam integer DECODER_LAYER0_ROPE_COEFFICIENTS = 64;\n", encoding="ascii")
    print("DECODER_LAYER0_VECTOR_PASS tokens=2 positions=0,1 cache_slot=0 "
          f"trace_records={len(all_trace)} final_records=1792 tensor_hex={len(tensor_manifest['tensors'])} "
          "rope_coefficients=64 oracle=integer_only")


if __name__ == "__main__":
    main()
