#!/usr/bin/env python3
"""Materialize authenticated official layer-2 vectors from a layer-1 handoff."""

from __future__ import annotations

import argparse
from pathlib import Path

from model24_execution_oracle import materialize_indexed_decoder_vectors


LAYER1_FINAL_SHA256 = (
    "2324470c304f23a372378af6f9f65cc7a646fbaa614882c4ced44110b99dca85"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--layer1-handoff", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = materialize_indexed_decoder_vectors(
        args.checkpoint.resolve(strict=True),
        args.tensor_map.resolve(strict=True),
        args.layer1_handoff.resolve(strict=True),
        args.output_dir.resolve(),
        layer_index=2,
        expected_handoff_sha256=LAYER1_FINAL_SHA256,
    )
    print(
        "DECODER_LAYER2_VECTOR_PASS "
        f"input_sha256={manifest['input_handoff']['sha256']} "
        f"descriptor_sha256={manifest['layer_binding']['descriptor_sha256']} "
        f"consumed_tensors={len(manifest['consumed_tensors'])} "
        f"trace_records={manifest['trace_records']} "
        f"final_records={manifest['final_records']}"
    )


if __name__ == "__main__":
    main()
