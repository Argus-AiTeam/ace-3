#!/usr/bin/env python3
"""Generate the independent checkpoint sequence for the Model24 RTL controller."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class ControllerVectorError(RuntimeError):
    """Raised when the controller contract cannot define the fixed trajectory."""


def _load_contract(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ControllerVectorError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    document = json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(document, dict):
        raise ControllerVectorError("controller contract must be a JSON object")
    return document


def expected_event_words(contract: dict[str, Any]) -> list[int]:
    geometry = contract.get("geometry")
    encoding = contract.get("checkpoint_encoding")
    if not isinstance(geometry, dict) or not isinstance(encoding, dict):
        raise ControllerVectorError("controller geometry or encoding is missing")
    if geometry.get("layer_count") != 24:
        raise ControllerVectorError("controller must cover exactly 24 layers")
    if geometry.get("strict_layer_order") != list(range(24)):
        raise ControllerVectorError("controller layer order must be 0 through 23")
    if geometry.get("cache_slots") != 2 or geometry.get("supported_positions") != 128:
        raise ControllerVectorError("controller cache geometry mismatch")
    if encoding != {
        "word_bits": 16,
        "completed_layer": [4, 0],
        "next_layer": [9, 5],
        "terminal": 10,
    }:
        raise ControllerVectorError("controller checkpoint encoding mismatch")
    return [
        layer | ((layer + 1) << 5) | ((1 if layer == 23 else 0) << 10)
        for layer in range(24)
    ]


def materialize(contract_path: Path, output_dir: Path) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    contract = _load_contract(contract_path)
    words = expected_event_words(contract)
    event_bytes = "".join(f"{word:04x}\n" for word in words).encode("ascii")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cascade_events.hex").write_bytes(event_bytes)
    manifest = {
        "schema_version": 1,
        "kind": "ace3_model24_layer_controller_vectors",
        "contract": {
            "path": "ace3/contracts/model24_layer_controller.json",
            "sha256": hashlib.sha256(contract_bytes).hexdigest(),
        },
        "scenario": {
            "cache_slot": 1,
            "position": 127,
            "layer_count": len(words),
        },
        "files": {
            "cascade_events.hex": {
                "bytes": len(event_bytes),
                "records": len(words),
                "sha256": hashlib.sha256(event_bytes).hexdigest(),
            }
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = materialize(
        args.contract.resolve(strict=True),
        args.output_dir.resolve(),
    )
    event = manifest["files"]["cascade_events.hex"]
    print(
        "MODEL24_LAYER_CONTROLLER_VECTOR_PASS "
        f"layers={event['records']} sha256={event['sha256']}"
    )


if __name__ == "__main__":
    main()
