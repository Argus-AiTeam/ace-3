#!/usr/bin/env python3
"""Authenticate Model24 controller vectors and recompute their expected sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from generate_model24_layer_controller_vectors import (
    _load_contract,
    expected_event_words,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--generated-dir", required=True, type=Path)
    args = parser.parse_args()
    contract_path = args.contract.resolve(strict=True)
    generated_dir = args.generated_dir.resolve(strict=True)
    event_path = generated_dir / "cascade_events.hex"
    manifest_path = generated_dir / "manifest.json"

    words = expected_event_words(_load_contract(contract_path))
    expected_bytes = "".join(f"{word:04x}\n" for word in words).encode("ascii")
    actual_bytes = event_path.read_bytes()
    if actual_bytes != expected_bytes:
        raise RuntimeError("cascade_events.hex does not match the contract")

    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    expected_digest = hashlib.sha256(actual_bytes).hexdigest()
    if manifest.get("contract") != {
        "path": "ace3/contracts/model24_layer_controller.json",
        "sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    }:
        raise RuntimeError("controller contract binding mismatch")
    if manifest.get("scenario") != {
        "cache_slot": 1,
        "position": 127,
        "layer_count": 24,
    }:
        raise RuntimeError("controller scenario mismatch")
    if manifest.get("files", {}).get("cascade_events.hex") != {
        "bytes": len(actual_bytes),
        "records": 24,
        "sha256": expected_digest,
    }:
        raise RuntimeError("controller event binding mismatch")
    print(
        "MODEL24_LAYER_CONTROLLER_VECTOR_VALIDATION_PASS "
        f"layers=24 sha256={expected_digest}"
    )


if __name__ == "__main__":
    main()
