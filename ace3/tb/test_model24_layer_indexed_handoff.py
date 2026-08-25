#!/usr/bin/env python3
"""Focused non-vacuous checks for layer-index selection and the vl15 handoff."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--handoff", required=True, type=Path)
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    sys.path.insert(0, str(root / "ace3" / "model"))

    from model24_execution_oracle import (  # pylint: disable=import-outside-toplevel
        ContractError,
        LAYER0_VL15_FINAL_ROWS_SHA256,
        indexed_layer_binding,
        indexed_layer_tensor_records,
        indexed_layer_tensor_value_hashes,
        load_two_token_handoff,
        materialize_indexed_decoder_vectors,
        sampled_indexed_q_projection_rows,
    )

    tensor_map = json.loads(args.tensor_map.read_text(encoding="ascii"))
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
    hashes0 = indexed_layer_tensor_value_hashes(
        args.checkpoint,
        args.tensor_map,
        0,
    )
    hashes1 = indexed_layer_tensor_value_hashes(
        args.checkpoint,
        args.tensor_map,
        1,
    )
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

    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        altered_handoff = temporary_path / "altered-vl15.rows"
        altered_payload = bytearray(args.handoff.read_bytes())
        altered_payload[-2] = ord("0") if altered_payload[-2] != ord("0") else ord("1")
        altered_handoff.write_bytes(altered_payload)
        rejected_handoffs = (
            ("omitted", temporary_path / "omitted-vl15.rows", FileNotFoundError),
            ("altered", altered_handoff, ContractError),
        )
        for name, handoff_path, expected_error in rejected_handoffs:
            output_dir = temporary_path / f"{name}-output"
            try:
                materialize_indexed_decoder_vectors(
                    args.checkpoint,
                    args.tensor_map,
                    handoff_path,
                    output_dir,
                    layer_index=1,
                )
            except expected_error:
                pass
            else:
                raise AssertionError(f"{name} vl15 handoff was accepted")
            assert not (output_dir / "boundary_manifest.json").exists()

    _, handoff = load_two_token_handoff(
        args.handoff,
        expected_sha256=LAYER0_VL15_FINAL_ROWS_SHA256,
    )
    assert handoff["shape"] == [2, 896]
    sample0 = sampled_indexed_q_projection_rows(
        args.checkpoint,
        args.tensor_map,
        args.handoff,
        0,
    )
    sample1 = sampled_indexed_q_projection_rows(
        args.checkpoint,
        args.tensor_map,
        args.handoff,
        1,
    )
    assert sample0 != sample1
    print(
        "MODEL24_LAYER_INDEXED_HANDOFF_PASS "
        f"layer1_descriptor={layer1['descriptor_sha256']} "
        f"changed_consumed_tensors={len(changed_tensors)} "
        f"handoff_shape=2x896 handoff_sha256={handoff['sha256']} "
        f"layer0_q_rows={sample0} layer1_q_rows={sample1}"
    )


if __name__ == "__main__":
    main()
