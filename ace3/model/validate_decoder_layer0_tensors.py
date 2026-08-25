#!/usr/bin/env python3
"""Validate extracted layer-0 tensors against committed fixed hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    generated_dir = args.generated_dir.resolve(strict=True)
    manifest_path = generated_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
    if manifest != bindings["expected_manifest"]:
        raise RuntimeError("decoder layer-0 manifest does not match committed bindings")

    artifacts = manifest["tensors"] + manifest["token_embeddings"]
    expected_files = {"manifest.json"}
    for artifact in artifacts:
        path = generated_dir / artifact["serialized_file"]
        expected_files.add(path.name)
        if path.stat().st_size != artifact["byte_count"]:
            raise RuntimeError(f"{path.name} byte count mismatch")
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise RuntimeError(
                f"{path.name} SHA256 mismatch: expected {artifact['sha256']}, "
                f"got {actual}"
            )
    actual_files = {path.name for path in generated_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(
            f"decoder layer-0 artifact inventory mismatch: {sorted(actual_files)}"
        )
    print(
        "DECODER_LAYER0_VALIDATION_PASS "
        f"tensors={len(manifest['tensors'])} tokens={len(manifest['token_embeddings'])} "
        "source_sha256=pass serialized_sha256=pass"
    )


if __name__ == "__main__":
    main()
