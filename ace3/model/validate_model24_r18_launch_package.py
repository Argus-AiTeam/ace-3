#!/usr/bin/env python3
"""Fail-closed validation for the Model24 r18 read-only build-output package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


CONTROLLER_PATH = "ace3/model/controller_model24_rtl_cascade.py"
R17_NONCE = "2edd86429a5c41f3"
R17_FAILURE_HASHES = {
    "package.json": "8538e88f0760a0d2e9fccbc827946ec0444187128dc0ba7b90dd12e951d1fd98",
    "seal.json": "5bddd342eef1b35f32c7995c418328257d02e8a926c1a51eac44a5435c6a9382",
    "review.json": "9ade35cb767dde28e0102d97941d0ac3b0ae34c0fbb0b4f2e157d769d818cadc",
    "authority.json": "fbaf49e2f6c30afe2d9d3200e5dca60d0f77246df19debb12f7f8a91a3d372b0",
    "authority-consumed.json": "fbaf49e2f6c30afe2d9d3200e5dca60d0f77246df19debb12f7f8a91a3d372b0",
    "launch-terminal.json": "b4f90cfe856b15a6854bcdac83998baf54e93276c04a697405cacb8ecff75b6a",
    "terminal-manifest.json": "fa30f7167f573dd798d36b5b2f9af306b96dbc1d2342a985d0585996e65f6605",
    "runner-receipt.json": "abd80ebbe9a7a0ba634eadf932b633da95bfc517c7a01f011f728a807fbe2f7f",
    "stdout.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "stderr.log": "e4d6d2ed8bb6b9a223f93d00b0c0858a412cdf1958eb66b2c89b5be42546f989",
    "exit-code.txt": "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865",
    "bindings.json": "18b5e79c8fab08872348275c34c5c132f21f1d02c5ac7d355ff096f6baffc28f",
    "compile.log": "da76ac02826fdecc0f16d114a8f019db70d1b4518bea0ea028882f875078ec06",
}
R17_CONTROLLER_SHA256 = (
    "a633a4f40089e0c486b9aa7247ae5884ba44cd4c4922d2b62ee28f49999459b6"
)
ADDITIONAL_NEGATIVE_CASES = (
    "omitted-build-root",
    "wrong-build-root",
    "outside-build-root",
    "symlinked-build-root",
    "source-write-attempt",
    "stale-build-output",
    "post-seal-source-mutation",
)
ADDITIONAL_REVIEW_HASHES = (
    "controller_overlay_sha256",
    "readonly_build_output_probe_sha256",
    "r17_failed_ancestry_sha256",
)


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
    _PACKAGE / "validate-r17-package.py",
    "model24_r18_full_contract_base_validator",
)
_r16 = _base._base
_base_package_review_hashes = _base.package_review_hashes
_r16_validate_v2_and_preflight = _r16.validate_v2_and_preflight
REQUIRED_POSITIVE_FIXTURES = _base.REQUIRED_POSITIVE_FIXTURES
REQUIRED_NEGATIVE_CASES = (
    *_base.REQUIRED_NEGATIVE_CASES,
    *ADDITIONAL_NEGATIVE_CASES,
)
REQUIRED_REVIEW_HASHES = (
    *_base.REQUIRED_REVIEW_HASHES,
    *ADDITIONAL_REVIEW_HASHES,
)


def _propagate_contract(module: Any) -> None:
    seen: set[int] = set()
    while id(module) not in seen:
        seen.add(id(module))
        module.REQUIRED_NEGATIVE_CASES = REQUIRED_NEGATIVE_CASES
        module.REQUIRED_REVIEW_HASHES = REQUIRED_REVIEW_HASHES
        if not hasattr(module, "_base"):
            break
        module = module._base


_propagate_contract(_base)


def package_review_hashes(package: Path) -> dict[str, str]:
    hashes = _base_package_review_hashes(package)
    hashes.update(
        {
            "controller_overlay_sha256": digest(
                package / "controller-overlay.json"
            ),
            "readonly_build_output_probe_sha256": digest(
                package / "evidence/readonly-build-output-probe/result.json"
            ),
            "r17_failed_ancestry_sha256": digest(
                package / "provenance/r17-failed/classification.json"
            ),
        }
    )
    return hashes


module = _base
while True:
    module.package_review_hashes = package_review_hashes
    if not hasattr(module, "_base"):
        break
    module = module._base


def load_rtl_controller(package: Path) -> Any:
    model_root = package.parent / "source/ace3/model"
    sys.path.insert(0, str(model_root))
    try:
        return _load_module(
            model_root / "controller_model24_rtl_cascade.py",
            "model24_r18_materialized_rtl_controller",
        )
    finally:
        sys.path.pop(0)


def validate_build_root(
    package: Path,
    build_root: Path | None,
    output_dir: Path,
) -> list[str]:
    controller = load_rtl_controller(package)
    try:
        return controller.build_layer_compile_argv(
            package.parent / "source",
            output_dir,
            0,
            build_root,
        )
    except controller.RtlCascadeError as error:
        raise SystemExit(str(error)) from error


def authenticate_probe_binary(
    package: Path,
    binary: Path,
    compile_started_ns: int,
) -> dict[str, Any]:
    controller = load_rtl_controller(package)
    try:
        return controller.authenticate_compiled_binary(
            binary,
            compile_started_ns,
        )
    except controller.RtlCascadeError as error:
        raise SystemExit(str(error)) from error


def reject_source_write_attempt(source: Path) -> None:
    probe = source / ".r18-source-write-attempt"
    try:
        descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        raise SystemExit(f"source write attempt rejected: {error}") from error
    else:
        os.close(descriptor)
        probe.unlink()
        raise SystemExit("sealed source was writable")


def require_source_records_match(
    expected: list[dict[str, Any]],
    observed: list[dict[str, Any]],
) -> None:
    require(expected == observed, "post-seal source mutation detected")


def _validate_materialization(package: Path) -> None:
    baseline = load_json(package / "provenance/accepted-source-tree.json")
    current = load_json(package / "source-tree.json")
    baseline_records = {
        record["path"]: record for record in baseline["records"]
    }
    current_records = {record["path"]: record for record in current["records"]}
    require(
        set(baseline_records) == set(current_records),
        "materialized source path set changed",
    )
    allowed = set(_r16.OVERLAY_PATHS) | {CONTROLLER_PATH}
    for path, current_record in current_records.items():
        if path in allowed and current_record["kind"] == "file":
            continue
        require(
            current_record == baseline_records[path],
            f"non-overlay accepted source entry changed: {path}",
        )


def _validate_v2_and_preflight(
    package: Path,
    seal: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    preflight = load_json(package / "evidence/production-preflight/result.json")
    previous = _r16.COMPILE_ARGV
    _r16.COMPILE_ARGV = preflight["compile_argv"]
    try:
        _r16_validate_v2_and_preflight(package, seal, manifest)
    finally:
        _r16.COMPILE_ARGV = previous


_r16.validate_materialization = _validate_materialization
_r16.validate_v2_and_preflight = _validate_v2_and_preflight


def validate_r17_failed_ancestry(package: Path) -> None:
    root = package / "provenance/r17-failed"
    require(
        {path.name for path in root.iterdir()}
        == set(R17_FAILURE_HASHES) | {"classification.json"},
        "r17 failed ancestry file set differs",
    )
    for name, expected in R17_FAILURE_HASHES.items():
        require(
            digest(root / name) == expected,
            f"r17 failed ancestry changed: {name}",
        )
    classification = load_json(root / "classification.json")
    require(
        classification
        == {
            "schema_version": 1,
            "kind": "ace3_model24_r17_consumed_failed_ancestry",
            "nonce": R17_NONCE,
            "classification": "LAWFUL_CONSUMED_FAILED",
            "sole_lawful_invocation": True,
            "historical_execution_invocations": 1,
            "authority_consumed": True,
            "natural_terminal": False,
            "exit_code": 1,
            "review_reusable": False,
            "authority_reusable": False,
            "execution_inputs_reusable": False,
            "predecessor_reusable": False,
            "copied_byte_for_byte": True,
            "terminal_manifest_sha256": R17_FAILURE_HASHES[
                "terminal-manifest.json"
            ],
            "compile_log_sha256": R17_FAILURE_HASHES["compile.log"],
            "hashes": R17_FAILURE_HASHES,
        },
        "r17 failed ancestry classification differs",
    )
    require(
        (root / "authority.json").read_bytes()
        == (root / "authority-consumed.json").read_bytes(),
        "r17 consumed authority differs from its original",
    )
    terminal = load_json(root / "launch-terminal.json")
    require(
        terminal["exit_code"] == 1
        and terminal["natural_terminal"] is False
        and terminal["invocation_count"] == 1,
        "r17 launch terminal does not prove one failed invocation",
    )
    receipt = load_json(root / "runner-receipt.json")
    require(
        receipt["state"] == "error" and receipt["exit_code"] == 1,
        "r17 runner receipt does not prove failure",
    )


def validate_controller_overlay(package: Path) -> None:
    overlay = load_json(package / "controller-overlay.json")
    controller = package.parent / "source" / CONTROLLER_PATH
    require(
        overlay
        == {
            "schema_version": 1,
            "kind": "ace3_model24_r18_controller_build_output_overlay",
            "path": CONTROLLER_PATH,
            "overlay_set_exact": True,
            "r17_source_sha256": R17_CONTROLLER_SHA256,
            "materialized_sha256": digest(controller),
            "production_make_target": "model24-rtl-layer-compile",
            "build_root_assignment": (
                "MODEL24_RTL_CASCADE_DIR=<bound payload output_dir>"
            ),
            "source_build_forbidden": True,
        },
        "r18 controller overlay binding differs",
    )
    source_manifest = load_json(package / "source-tree.json")
    records = {
        record["path"]: record for record in source_manifest["records"]
    }
    require(
        records[CONTROLLER_PATH]["sha256"] == digest(controller),
        "controller overlay differs from sealed source tree",
    )


def _source_is_recursively_readonly(source: Path) -> bool:
    for path in (source, *source.rglob("*")):
        mode = stat.S_IMODE(path.lstat().st_mode)
        if mode & 0o222:
            return False
    return True


def validate_readonly_build_output_probe(package: Path) -> None:
    evidence = package / "evidence/readonly-build-output-probe"
    result = load_json(evidence / "result.json")
    source = package.parent / "source"
    output = package.parent / "probe-output"
    require(result["status"] == "PASS", "read-only build-output probe failed")
    require(
        result["source"] == str(source)
        and result["output_root"] == str(output),
        "read-only build-output probe paths differ",
    )
    require(
        output.is_dir()
        and not output.is_symlink()
        and output.resolve(strict=True) == output,
        "probe output root is absent or symlinked",
    )
    require(
        source.resolve(strict=True) not in output.resolve(strict=True).parents,
        "probe output root is inside sealed source",
    )
    before = load_json(evidence / "source-before.json")
    after = load_json(evidence / "source-after.json")
    source_manifest = load_json(package / "source-tree.json")
    require_source_records_match(before["records"], after["records"])
    require_source_records_match(source_manifest["records"], after["records"])
    require(
        before["tree_sha256"] == after["tree_sha256"]
        == source_manifest["tree_sha256"],
        "post-seal source tree hash changed",
    )
    require(
        result["source_build_absent_before"] is True
        and result["source_build_absent_after"] is True
        and result["source_additions"] == 0
        and result["source_modifications"] == 0,
        "sealed source build-output closure is open",
    )
    require(
        _source_is_recursively_readonly(source),
        "sealed source is not recursively read-only",
    )
    controller = load_rtl_controller(package)
    for layer in (0, 23):
        layer_record = result["layers"][str(layer)]
        binary = (
            output
            / "compiled"
            / f"layer{layer}"
            / "obj_dir"
            / "Vace3_decoder_layer0_token_engine"
        )
        expected_command = controller.build_layer_compile_argv(
            source,
            output,
            layer,
            output,
        )
        require(
            layer_record["argv"] == expected_command,
            f"layer {layer} did not use production Make command construction",
        )
        require(
            expected_command[-1] == f"MODEL24_RTL_CASCADE_DIR={output}",
            f"layer {layer} build root is not the bound output root",
        )
        observed = authenticate_probe_binary(
            package,
            binary,
            layer_record["compile_started_ns"],
        )
        require(
            observed["sha256"] == layer_record["binary"]["sha256"]
            and observed["bytes"] == layer_record["binary"]["bytes"],
            f"layer {layer} probe binary authentication differs",
        )
        for suffix in ("stdout", "stderr"):
            name = f"compile-layer{layer}.{suffix}"
            require(
                digest(evidence / name)
                == layer_record["evidence_hashes"][name],
                f"layer {layer} compile evidence changed: {name}",
            )
    require(
        result["negative_cases"] == list(ADDITIONAL_NEGATIVE_CASES),
        "read-only build-output negative probe set differs",
    )
    require(
        result["model24_invocations"] == 0
        and result["controller_simulator_invocations"] == 0
        and result["rtl_binary_invocations"] == 0
        and result["authority_invocations"] == 0,
        "read-only build-output probe crossed the preparation boundary",
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
    validate_controller_overlay(package)
    validate_readonly_build_output_probe(package)
    validate_r17_failed_ancestry(package)
    require(
        seal["controller_overlay_sha256"]
        == digest(package / "controller-overlay.json"),
        "controller overlay is not seal-bound",
    )
    require(
        seal["readonly_build_output_probe_sha256"]
        == digest(
            package / "evidence/readonly-build-output-probe/result.json"
        ),
        "read-only build-output probe is not seal-bound",
    )
    require(
        manifest["provenance"]["r17_terminal_ancestry"]
        == {
            "classification": "LAWFUL_CONSUMED_FAILED",
            "classification_sha256": digest(
                package / "provenance/r17-failed/classification.json"
            ),
            "terminal_manifest_sha256": R17_FAILURE_HASHES[
                "terminal-manifest.json"
            ],
            "compile_log_sha256": R17_FAILURE_HASHES["compile.log"],
            "review_reusable": False,
            "authority_reusable": False,
            "execution_inputs_reusable": False,
        },
        "r17 failed ancestry manifest binding differs",
    )
    result.update(
        {
            "readonly_build_output_probe": "PASS",
            "r17_lawful_consumed_failed": True,
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
