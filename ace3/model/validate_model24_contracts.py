#!/usr/bin/env python3
"""Validate ACE-3 model24 contracts against the independent software oracle."""

from __future__ import annotations

import argparse
from pathlib import Path

from model24_oracle import (
    OFFICIAL_CONFIG,
    authenticate_checkpoint,
    load_authenticated_config,
    load_json_document,
    validate_contract_documents,
)


def parse_args() -> argparse.Namespace:
    ace3_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensor-map",
        type=Path,
        default=ace3_root / "contracts" / "model24_tensor_map.json",
    )
    parser.add_argument(
        "--control",
        type=Path,
        default=ace3_root / "contracts" / "model24_control.json",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.config is None) != (args.checkpoint is None):
        raise RuntimeError("--config and --checkpoint must be supplied together")
    config = OFFICIAL_CONFIG
    authenticated_checkpoint = False
    if args.config is not None and args.checkpoint is not None:
        config = load_authenticated_config(args.config.resolve(strict=True))
        authenticate_checkpoint(args.checkpoint.resolve(strict=True), config)
        authenticated_checkpoint = True
    summary = validate_contract_documents(
        load_json_document(args.tensor_map.resolve(strict=True)),
        load_json_document(args.control.resolve(strict=True)),
        config,
    )
    print(
        "MODEL24_CONTRACT_PASS "
        f"tensors={summary['tensor_count']} "
        f"ranges={summary['touched_range_count']} "
        f"layers={summary['layer_namespace_count']} "
        f"families={summary['tensor_family_count']} "
        f"events={summary['operation_event_count']} "
        f"checkpoint_reauthenticated={str(authenticated_checkpoint).lower()}"
    )


if __name__ == "__main__":
    main()
