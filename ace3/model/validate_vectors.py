#!/usr/bin/env python3
"""Validate standalone vectors against the frozen accepted CF01 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SEED = 0xACE3CF01
GROUP_SIZE = 128
OFFICIAL_CASES = 10
SYNTHETIC_CASES = 20
TOTAL_CASES = OFFICIAL_CASES + SYNTHETIC_CASES
TOTAL_PAIRS = TOTAL_CASES * GROUP_SIZE
FROZEN_MANIFEST_SHA256 = (
    "39140a1555ea07697b59256892ad59560a633513c338bcf42fdcb1fca365b363"
)
SERIALIZED_ARTIFACTS = {
    "manifest.json",
    "meta.hex",
    "pairs.hex",
    "cases.txt",
    "vector_params.svh",
}
OFFICIAL_HASHES = {
    "qweight_i32le": (
        "db4770023698611ff0115d220590fdb8232fbe5dcbd22fbe80e0bcdc838caf87"
    ),
    "qzeros_i32le": (
        "3cf7cd5712dd7523db3c7dd47c2b1d582e19545036f75b95ff0331c1fc0c596c"
    ),
    "scales_f16le": (
        "687adc7d7bcd6e45a065f914dd27a1284b7e48260491bb0d26ae1e13b78ac321"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--frozen-manifest", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--evidence-bindings", required=True, type=Path)
    parser.add_argument("--standalone-bindings", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def checked_hex_lines(path: Path, count: int, width: int) -> list[str]:
    lines = path.read_text(encoding="ascii").splitlines()
    require(len(lines) == count, f"{path}: expected {count} lines, got {len(lines)}")
    require(
        all(len(line) == width and all(char in "0123456789abcdef" for char in line)
            for line in lines),
        f"{path}: expected lowercase width-{width} hexadecimal records",
    )
    return lines


def authenticate_serialized_artifacts(
    generated_dir: Path, bindings: dict[str, Any]
) -> None:
    require(
        bindings.get("kind") == "ace3_standalone_serialized_vector_bindings",
        "unexpected standalone binding kind",
    )
    generation = bindings.get("generation")
    require(isinstance(generation, dict), "standalone generation binding missing")
    require(generation.get("seed") == SEED, "standalone binding seed mismatch")
    require(
        generation.get("group_size") == GROUP_SIZE,
        "standalone binding group size mismatch",
    )
    require(
        generation.get("case_count") == TOTAL_CASES,
        "standalone binding case count mismatch",
    )
    require(
        generation.get("pair_count") == TOTAL_PAIRS,
        "standalone binding pair count mismatch",
    )
    require(
        generation.get("exact_ulp_bound") == 0,
        "standalone binding ULP bound mismatch",
    )
    historical = bindings.get("historical_frozen_manifest")
    require(isinstance(historical, dict), "historical manifest binding missing")
    require(
        historical.get("sha256") == FROZEN_MANIFEST_SHA256,
        "standalone binding does not preserve the historical manifest hash",
    )

    artifacts = bindings.get("serialized_artifacts")
    require(isinstance(artifacts, list), "serialized artifact bindings missing")
    by_name: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        require(isinstance(artifact, dict), "serialized artifact binding is not an object")
        name = artifact.get("file")
        require(isinstance(name, str), "serialized artifact file name missing")
        require(name not in by_name, f"duplicate serialized artifact binding: {name}")
        by_name[name] = artifact
    require(
        set(by_name) == SERIALIZED_ARTIFACTS,
        "standalone bindings do not cover the exact simulator artifact set",
    )

    for name in sorted(SERIALIZED_ARTIFACTS):
        artifact = by_name[name]
        expected_hash = artifact.get("sha256")
        require(
            isinstance(expected_hash, str)
            and len(expected_hash) == 64
            and all(char in "0123456789abcdef" for char in expected_hash),
            f"{name} binding has an invalid SHA256",
        )
        payload = (generated_dir / name).read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        require(
            actual_hash == expected_hash,
            f"{name} SHA256 mismatch: expected {expected_hash}, got {actual_hash}",
        )
        require(
            len(payload) == artifact.get("byte_count"),
            f"{name} byte count mismatch",
        )
        require(
            payload.count(b"\n") == artifact.get("line_count"),
            f"{name} line count mismatch",
        )


def main() -> None:
    args = parse_args()
    generated_dir = args.generated_dir.resolve(strict=True)
    generated_path = generated_dir / "manifest.json"
    standalone_bindings = load_json(args.standalone_bindings)
    require(
        isinstance(standalone_bindings, dict),
        f"{args.standalone_bindings}: JSON root must be an object",
    )
    require(
        standalone_bindings.get("schema_version") == 1,
        f"{args.standalone_bindings}: schema_version != 1",
    )
    authenticate_serialized_artifacts(generated_dir, standalone_bindings)

    frozen_bytes = args.frozen_manifest.read_bytes()
    generated_bytes = generated_path.read_bytes()
    frozen_hash = hashlib.sha256(frozen_bytes).hexdigest()
    require(
        frozen_hash == FROZEN_MANIFEST_SHA256,
        f"frozen manifest hash mismatch: {frozen_hash}",
    )
    require(
        generated_bytes == frozen_bytes,
        "generated manifest is not byte-identical to the frozen accepted manifest",
    )

    generated = load_json(generated_path)
    frozen = load_json(args.frozen_manifest)
    contract = load_json(args.contract)
    evidence = load_json(args.evidence_bindings)
    for path, document in (
        (generated_path, generated),
        (args.frozen_manifest, frozen),
        (args.contract, contract),
        (args.evidence_bindings, evidence),
        (args.standalone_bindings, standalone_bindings),
    ):
        require(isinstance(document, dict), f"{path}: JSON root must be an object")
        require(document.get("schema_version") == 1, f"{path}: schema_version != 1")

    require(generated == frozen, "generated and frozen manifests differ semantically")
    require(generated.get("seed") == SEED, "deterministic seed mismatch")
    require(generated.get("group_size") == GROUP_SIZE, "group size mismatch")
    require(
        generated.get("official_case_count") == OFFICIAL_CASES,
        "official case count mismatch",
    )
    require(
        generated.get("synthetic_case_count") == SYNTHETIC_CASES,
        "synthetic case count mismatch",
    )
    require(generated.get("exact_ulp_bound") == 0, "ULP bound is not zero")
    cases = generated.get("cases")
    require(isinstance(cases, list) and len(cases) == TOTAL_CASES, "case count mismatch")
    require(
        sum(case.get("source") == "official_layer0_q_proj" for case in cases)
        == OFFICIAL_CASES,
        "official source binding count mismatch",
    )
    require(
        sum(case.get("source") == "contract_valid_synthetic" for case in cases)
        == SYNTHETIC_CASES,
        "synthetic source binding count mismatch",
    )

    profile = contract.get("profiles", {}).get("AWQ_W4A16", {})
    require(
        profile.get("native_awq", {}).get("group_size") == GROUP_SIZE,
        "contract group size mismatch",
    )
    require(
        profile.get("native_awq", {}).get("qzero_plus_one") is False,
        "contract unexpectedly applies qzero plus-one",
    )
    require(
        profile.get("numeric", {}).get("agreement")
        == "zero ULP against the integer bit-level oracle",
        "contract no longer requires zero-ULP agreement",
    )

    bindings = evidence.get("official_layer0_q_proj")
    require(isinstance(bindings, list) and len(bindings) == 3, "binding count mismatch")
    binding_hashes = {item.get("kind"): item.get("sha256") for item in bindings}
    require(binding_hashes == OFFICIAL_HASHES, "official tensor binding hashes mismatch")

    checked_hex_lines(generated_dir / "meta.hex", TOTAL_CASES, 42)
    checked_hex_lines(generated_dir / "pairs.hex", TOTAL_PAIRS, 12)
    case_lines = (generated_dir / "cases.txt").read_text(encoding="ascii").splitlines()
    require(len(case_lines) == TOTAL_CASES, "cases.txt case count mismatch")
    require(
        all(len(line.split()) == 8 for line in case_lines),
        "cases.txt record shape mismatch",
    )
    expected_params = (
        f"localparam integer VECTOR_CASES = {TOTAL_CASES};\n"
        f"localparam integer VECTOR_PAIRS = {TOTAL_PAIRS};\n"
    )
    require(
        (generated_dir / "vector_params.svh").read_text(encoding="ascii")
        == expected_params,
        "vector parameter include mismatch",
    )

    print(
        "JSON_VALIDATION_PASS "
        f"json_files=5 serialized_artifacts=5 sha256=pass "
        f"seed=0x{SEED:08x} cases={TOTAL_CASES} "
        f"pairs={TOTAL_PAIRS} group_size={GROUP_SIZE} ulp_bound=0 "
        f"official_bindings=3 frozen_manifest_sha256={frozen_hash}"
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"JSON_VALIDATION_FAIL {error}", file=sys.stderr)
        raise SystemExit(1) from error
