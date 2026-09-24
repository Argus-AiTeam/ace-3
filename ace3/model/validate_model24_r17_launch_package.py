#!/usr/bin/env python3
"""Fail-closed validation for the Model24 r17 binding-closure package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ADDITIONAL_NEGATIVE_CASES = (
    "missing-binding",
    "substituted-binding",
    "r16-stale-binding",
    "tensor-map-mutation",
)
ADDITIONAL_REVIEW_HASHES = (
    "bindings_sha256",
    "binding_closure_sha256",
    "r16_failed_ancestry_sha256",
)
R16_NONCE = "bbcf2022e008a933"
R16_FAILURE_HASHES = {
    "package.json": "7749ea8caa0c65e08c30ad7f6ba20771c56661b47295dc706a0c1125460c556b",
    "seal.json": "2a55e2e85c085bf196c165ed815611e60da3cea6f4e52b737dda2ac03df473b1",
    "review.json": "3a1faa11d16c45922c5e1abbabf9dd3e3d4876a18ea3a89b6d4d6787d019b4c8",
    "authority.json": "0ee4679e4969615cde6c33e49dc4b10df5313772075bb398ba04235afd3e4cbd",
    "authority-consumed.json": "0ee4679e4969615cde6c33e49dc4b10df5313772075bb398ba04235afd3e4cbd",
    "launch-terminal.json": "a7caf4d10590fd09671ea63cddabf5ec1eb9de9217409033c1ca5d942b9d7019",
    "terminal-manifest.json": "7a0baface178fab28b53585a8d0063c77dcc98ca2b7fe5b19b80fdab2b423061",
    "runner-receipt.json": "e0fc8326abb41070bd4f7be316e9ff7328d8f9654b4f6e38a6ad065b6256cc2a",
    "stdout.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "stderr.log": "b01e6217433862c60a51a287e58a5013dd24ed395984d8134e9938ba16ee59e3",
    "exit-code.txt": "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
    "bindings.json": "5d20b504efcf9500e03611c129b20bec26db39a52517c16dff0fd309a0221ad0",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def load_json(path: Path) -> dict[str, Any]:
    require(
        path.is_file() and not path.is_symlink(),
        f"regular JSON file required: {path}",
    )
    value = json.loads(path.read_text(encoding="ascii"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(
        spec is not None and spec.loader is not None,
        f"module loader unavailable: {path}",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PACKAGE = Path(__file__).resolve().parent
_base = _load_module(
    _PACKAGE / "validate-r16-package.py",
    "model24_r17_full_contract_base_validator",
)
_base_package_review_hashes = _base.package_review_hashes
REQUIRED_POSITIVE_FIXTURES = _base.REQUIRED_POSITIVE_FIXTURES
REQUIRED_NEGATIVE_CASES = (
    *_base.REQUIRED_NEGATIVE_CASES,
    *ADDITIONAL_NEGATIVE_CASES,
)
REQUIRED_REVIEW_HASHES = (
    *_base.REQUIRED_REVIEW_HASHES,
    *ADDITIONAL_REVIEW_HASHES,
)
_base.REQUIRED_NEGATIVE_CASES = REQUIRED_NEGATIVE_CASES
_base.REQUIRED_REVIEW_HASHES = REQUIRED_REVIEW_HASHES
_base._base.REQUIRED_NEGATIVE_CASES = REQUIRED_NEGATIVE_CASES
_base._base.REQUIRED_REVIEW_HASHES = REQUIRED_REVIEW_HASHES


def package_review_hashes(package: Path) -> dict[str, str]:
    hashes = _base_package_review_hashes(package)
    hashes.update(
        {
            "bindings_sha256": digest(package / "bindings.json"),
            "binding_closure_sha256": digest(
                package / "evidence/binding-closure/result.json"
            ),
            "r16_failed_ancestry_sha256": digest(
                package / "provenance/r16-failed/classification.json"
            ),
        }
    )
    return hashes


_base.package_review_hashes = package_review_hashes
_base._base.package_review_hashes = package_review_hashes


def load_binding_controller(package: Path) -> Any:
    source = package.parent / "source"
    model_root = source / "ace3/model"
    sys.path.insert(0, str(model_root))
    try:
        return _load_module(
            model_root / "controller_model24_cascade.py",
            "model24_r17_materialized_binding_controller",
        )
    finally:
        sys.path.pop(0)


def validate_binding_payload(
    package: Path,
    binding_payload: bytes | None = None,
    tensor_map_payload: bytes | None = None,
    binding_path: Path | None = None,
) -> dict[str, Any]:
    source = package.parent / "source"
    bindings_path = binding_path or package / "bindings.json"
    if binding_payload is None:
        require(
            bindings_path.is_file() and not bindings_path.is_symlink(),
            "final sealed binding is missing",
        )
        binding_payload = bindings_path.read_bytes()
    tensor_map_path = source / "ace3/contracts/model24_tensor_map.json"
    if tensor_map_payload is None:
        require(
            tensor_map_path.is_file() and not tensor_map_path.is_symlink(),
            "exact materialized tensor map is missing",
        )
        tensor_map_payload = tensor_map_path.read_bytes()
    controller = load_binding_controller(package)
    try:
        document = controller._load_json(binding_payload, "layer bindings")
        controller.validate_binding_document(document, source, tensor_map_payload)
    except controller.ControllerCascadeError as error:
        raise SystemExit(str(error)) from error
    return {
        "status": "PASS",
        "validator": "controller_model24_cascade.validate_binding_document",
        "bindings_sha256": hashlib.sha256(binding_payload).hexdigest(),
        "tensor_map_sha256": hashlib.sha256(tensor_map_payload).hexdigest(),
    }


def binding_differences(
    expected: Any,
    actual: Any,
    path: str = "$",
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}"
            if key not in expected:
                differences.append(
                    {"path": child, "expected": "<absent>", "r16": actual[key]}
                )
            elif key not in actual:
                differences.append(
                    {"path": child, "expected": expected[key], "r16": "<absent>"}
                )
            else:
                differences.extend(
                    binding_differences(expected[key], actual[key], child)
                )
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append(
                {
                    "path": f"{path}.length",
                    "expected": len(expected),
                    "r16": len(actual),
                }
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual)
        ):
            differences.extend(
                binding_differences(
                    expected_item,
                    actual_item,
                    f"{path}[{index}]",
                )
            )
    elif expected != actual:
        differences.append({"path": path, "expected": expected, "r16": actual})
    return differences


def validate_r16_failed_ancestry(package: Path) -> None:
    root = package / "provenance/r16-failed"
    require(
        {path.name for path in root.iterdir()}
        == set(R16_FAILURE_HASHES) | {"classification.json"},
        "r16 failed ancestry file set differs",
    )
    for name, expected in R16_FAILURE_HASHES.items():
        require(
            digest(root / name) == expected,
            f"r16 failed ancestry changed: {name}",
        )
    classification = load_json(root / "classification.json")
    require(
        classification
        == {
            "schema_version": 1,
            "kind": "ace3_model24_r16_consumed_failed_ancestry",
            "nonce": R16_NONCE,
            "classification": "LAWFUL_CONSUMED_FAILED",
            "sole_lawful_invocation": True,
            "historical_execution_invocations": 1,
            "authority_consumed": True,
            "natural_terminal": False,
            "exit_code": 1,
            "review_reusable": False,
            "authority_reusable": False,
            "execution_inputs_reusable": False,
            "copied_byte_for_byte": True,
            "hashes": R16_FAILURE_HASHES,
        },
        "r16 failed ancestry classification differs",
    )
    require(
        (root / "authority.json").read_bytes()
        == (root / "authority-consumed.json").read_bytes(),
        "r16 consumed authority differs from its original",
    )
    terminal = load_json(root / "launch-terminal.json")
    require(
        terminal["exit_code"] == 1
        and terminal["natural_terminal"] is False
        and terminal["invocation_count"] == 1,
        "r16 launch terminal does not prove one failed invocation",
    )
    receipt = load_json(root / "runner-receipt.json")
    require(
        receipt["state"] == "error" and receipt["exit_code"] == 1,
        "r16 runner receipt does not prove failure",
    )


def validate_binding_closure(
    package: Path,
    seal: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    observed = validate_binding_payload(package)
    result = load_json(package / "evidence/binding-closure/result.json")
    require(result["status"] == "PASS", "binding closure did not pass")
    require(
        result["validator"]
        == "controller_model24_cascade.validate_binding_document",
        "binding closure used a different validator",
    )
    require(
        result["bindings_sha256"] == observed["bindings_sha256"],
        "binding closure result differs from final binding",
    )
    require(
        result["tensor_map_sha256"] == observed["tensor_map_sha256"],
        "binding closure result differs from materialized tensor map",
    )
    differences = result.get("r16_expected_differences")
    require(
        isinstance(differences, list)
        and differences
        == binding_differences(
            load_json(package / "bindings.json"),
            load_json(package / "provenance/r16-failed/bindings.json"),
        ),
        "r16 expected-versus-actual binding diagnosis differs",
    )
    require(len(differences) == 6, "r16 binding diagnosis is not complete")
    require(
        result["r16_validation"]
        == {"status": "REJECT", "reason": "layer binding manifest mismatch"},
        "r16 stale binding rejection differs",
    )
    provenance = manifest["provenance"]["bindings"]
    require(
        provenance
        == {
            "generation": (
                "materialized controller build_binding_document over exact "
                "materialized tensor-map bytes"
            ),
            "sha256": observed["bindings_sha256"],
            "tensor_map_sha256": observed["tensor_map_sha256"],
            "binding_closure_sha256": digest(
                package / "evidence/binding-closure/result.json"
            ),
            "r16_stale_binding_sha256": R16_FAILURE_HASHES["bindings.json"],
        },
        "binding provenance differs",
    )
    require(
        seal["bindings_sha256"] == observed["bindings_sha256"],
        "final binding is not seal-bound",
    )
    require(
        seal["binding_closure_sha256"]
        == digest(package / "evidence/binding-closure/result.json"),
        "binding closure result is not seal-bound",
    )


def validate_package(
    package: Path,
    mode: str,
    *,
    review_path: Path | None = None,
    authority_path: Path | None = None,
) -> dict[str, Any]:
    package = package.resolve(strict=True)
    result = _base.validate_package(
        package,
        mode,
        review_path=review_path,
        authority_path=authority_path,
    )
    seal, manifest = _base.validate_package_seal(package)
    validate_binding_closure(package, seal, manifest)
    validate_r16_failed_ancestry(package)
    require(
        manifest["production_preflight"]["binding_validation"]
        == {
            "validator": (
                "controller_model24_cascade.validate_binding_document"
            ),
            "before_authority_consumption": True,
            "before_output_creation": True,
            "before_controller_simulation": True,
            "before_payload_execution": True,
        },
        "production binding preflight order differs",
    )
    result.update(
        {
            "binding_preflight": "PASS",
            "r16_stale_binding": "REJECT",
            "negative_case_count": len(REQUIRED_NEGATIVE_CASES),
        }
    )
    return result


def __getattr__(name: str) -> Any:
    return getattr(_base, name)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--mode",
        choices=("package", "review", "pre-submit", "launch"),
        required=True,
    )
    parser.add_argument("--review", type=Path)
    parser.add_argument("--authority", type=Path)
    args = parser.parse_args()
    result = validate_package(
        args.package,
        args.mode,
        review_path=args.review,
        authority_path=args.authority,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
