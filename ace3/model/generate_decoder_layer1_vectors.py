#!/usr/bin/env python3
"""Materialize authenticated official layer-1 vectors from a layer-0 handoff."""

from __future__ import annotations

import argparse
from pathlib import Path

from model24_execution_oracle import materialize_indexed_decoder_vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--layer0-handoff", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = materialize_indexed_decoder_vectors(
        args.checkpoint.resolve(strict=True),
        args.tensor_map.resolve(strict=True),
        args.layer0_handoff.resolve(strict=True),
        args.output_dir.resolve(),
        layer_index=1,
    )
    print(
        "DECODER_LAYER1_VECTOR_PASS "
        f"input_sha256={manifest['input_handoff']['sha256']} "
        f"descriptor_sha256={manifest['layer_binding']['descriptor_sha256']} "
        f"consumed_tensors={len(manifest['consumed_tensors'])} "
        f"trace_records={manifest['trace_records']} "
        f"final_records={manifest['final_records']}"
    )


if __name__ == "__main__":
    main()
