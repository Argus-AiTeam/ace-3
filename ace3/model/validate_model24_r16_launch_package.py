#!/usr/bin/env python3
"""Fail-closed validation for a full Model24 r16 durable-launch package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
from typing import Any


ACCEPTED_COMMIT = "42c895ce1e5fea00525e9f2f7fef66f0fbb8e118"
REPOSITORY = Path("/home/argustest/ace3-argus")
V2_CANDIDATE = Path(
    "/home/argustest/ace3-model24-r15-successor-v2-20260829-rtl-compile-closure"
)
V2_HASHES = {
    "package.json": "349a07566e2a19ffef929c45162e62f95e14ea4b982c1a7d8283f9dd033ef418",
    "seal.json": "d8dbad9f1184bb03b0bc3f245ebcc7014252e4772457d07ea689297a18b1ece1",
    "review-request.json": "61405da89e8c03798213ebb9c3b0d4e4607c47e38b7c33a5242a2fd52bc252ad",
    "validate-package.py": "2d4f612d3a80b9f195557d7e9ff49ccb15fc6910a1733f77b818c8daa2a48778",
    "source-tree.json": "23a6a0caa6cfeb453110a5a51e59fefb1a0495dde8522b478c75a826d238a278",
    "layer0-compile-result.json": "fd74ae247267f7cf8842885c0bc9fd041ca0015b179cbf19df039340a54b492f",
}
V2_SOURCE_TREE_SHA256 = (
    "b5228db8d3513ab29059bc55d446186fa8440f0cfc2283da28c61e1fe9eb156c"
)
OVERLAY_PATHS = (
    "Makefile",
    "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
    "ace3/rtl/ace3_fp16_silu_gate_core.sv",
    "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp",
)
ADDITIONAL_NEGATIVE_CASES = (
    "external-wrapper",
    "wrong-overlay",
    "missing-make-target",
    "stale-or-substituted-compiled-binary",
    "predecessor-review-or-authority-reuse",
)
ADDITIONAL_REVIEW_HASHES = (
    "authority_schema_sha256",
    "source_overlays_sha256",
    "production_preflight_sha256",
    "v2_package_manifest_sha256",
    "v2_seal_sha256",
    "v2_review_request_sha256",
    "v2_validator_sha256",
    "v2_source_tree_manifest_sha256",
    "v2_layer0_compile_result_sha256",
    "r14_unauthorized_evidence_sha256",
    "r15_unauthorized_evidence_sha256",
)
COMPILE_ARGV = [
    "make",
    "--no-print-directory",
    "model24-rtl-layer-compile",
    "MODEL24_RTL_LAYER_INDEX=0",
    "MODEL24_RTL_ACCURATE_SILU=1",
]
EXPECTED_BINARY = "Vace3_decoder_layer0_token_engine"


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
    require(path.is_file() and not path.is_symlink(), f"regular JSON file required: {path}")
    value = json.loads(path.read_text(encoding="ascii"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _load_base(package: Path) -> Any:
    path = package / "validate-base-package.py"
    spec = importlib.util.spec_from_file_location("model24_r16_base_validator", path)
    require(spec is not None and spec.loader is not None, "base validator loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PACKAGE = Path(__file__).resolve().parent
_base = _load_base(_PACKAGE)
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


def package_review_hashes(package: Path) -> dict[str, str]:
    hashes = _base_package_review_hashes(package)
    v2_root = package / "provenance/v2-compile-source"
    hashes.update(
        {
            "authority_schema_sha256": digest(package / "authority-schema.json"),
            "source_overlays_sha256": digest(package / "source-overlays.json"),
            "production_preflight_sha256": digest(
                package / "evidence/production-preflight/result.json"
            ),
            "v2_package_manifest_sha256": digest(v2_root / "package.json"),
            "v2_seal_sha256": digest(v2_root / "seal.json"),
            "v2_review_request_sha256": digest(v2_root / "review-request.json"),
            "v2_validator_sha256": digest(v2_root / "validate-package.py"),
            "v2_source_tree_manifest_sha256": digest(v2_root / "source-tree.json"),
            "v2_layer0_compile_result_sha256": digest(
                v2_root / "layer0-compile-result.json"
            ),
            "r14_unauthorized_evidence_sha256": digest(
                package / "provenance/r14-unauthorized/classification.json"
            ),
            "r15_unauthorized_evidence_sha256": digest(
                package / "provenance/r15-unauthorized/classification.json"
            ),
        }
    )
    return hashes


_base.package_review_hashes = package_review_hashes


def validate_make_target_text(text: str) -> None:
    require(
        "model24-rtl-layer-compile:" in text,
        "sealed Makefile compile target is missing",
    )
    for fragment in (
        'MODEL24_RTL_LAYER_INDEX ?= 0',
        'MODEL24_RTL_ACCURATE_SILU ?= 1',
        '-GLAYER_INDEX="$(MODEL24_RTL_LAYER_INDEX)"',
        '-GACCURATE_SILU="$(MODEL24_RTL_ACCURATE_SILU)"',
        "--top-module ace3_decoder_layer0_token_engine",
        "$(DECODER_CPP_TB)",
    ):
        require(fragment in text, f"sealed Makefile compile closure is missing: {fragment}")


def validate_overlay_document(package: Path, overlays: dict[str, Any]) -> None:
    records = overlays.get("records")
    require(isinstance(records, list), "overlay records are missing")
    require(
        [record.get("path") for record in records if isinstance(record, dict)]
        == list(OVERLAY_PATHS),
        "overlay set or order differs from the exact r16 inventory",
    )
    source = package.parent / "source"
    for record in records:
        relative = record["path"]
        accepted = subprocess.run(
            ["git", "show", f"{ACCEPTED_COMMIT}:{relative}"],
            cwd=REPOSITORY,
            capture_output=True,
            check=False,
        )
        require(accepted.returncode == 0, f"accepted overlay baseline unavailable: {relative}")
        accepted_sha256 = hashlib.sha256(accepted.stdout).hexdigest()
        v2_source = V2_CANDIDATE / "source" / relative
        require(
            record
            == {
                "path": relative,
                "accepted_sha256": accepted_sha256,
                "v2_declared_sha256": record["v2_declared_sha256"],
                "v2_source_sha256": digest(v2_source),
                "materialized_sha256": digest(source / relative),
                "accepted_differs": accepted_sha256 != digest(v2_source),
                "v2_matches_materialized": digest(v2_source) == digest(source / relative),
                "comparison": "PASS",
            },
            f"overlay independent comparison differs: {relative}",
        )
        require(
            record["v2_declared_sha256"] == record["v2_source_sha256"],
            f"v2 overlay declaration differs from its sealed source: {relative}",
        )
        require(record["accepted_differs"] is True, f"overlay is not a change: {relative}")
        require(record["v2_matches_materialized"] is True, f"wrong overlay: {relative}")


def validate_materialization(package: Path) -> None:
    baseline = load_json(package / "provenance/accepted-source-tree.json")
    current = load_json(package / "source-tree.json")
    baseline_records = {
        record["path"]: record for record in baseline["records"]
    }
    current_records = {record["path"]: record for record in current["records"]}
    require(set(baseline_records) == set(current_records), "materialized source path set changed")
    for path, current_record in current_records.items():
        if path in OVERLAY_PATHS and current_record["kind"] == "file":
            continue
        require(
            current_record == baseline_records[path],
            f"non-overlay accepted source entry changed: {path}",
        )


def validate_compiled_binary(
    package: Path,
    preflight: dict[str, Any],
    binary: Path | None = None,
) -> None:
    binary = binary or package / "evidence/production-preflight" / EXPECTED_BINARY
    require(binary.is_file() and not binary.is_symlink(), "fresh compiled binary is missing")
    require(binary.stat().st_mode & stat.S_IXUSR, "fresh compiled binary is not executable")
    require(
        digest(binary) == preflight["binary"]["sha256"],
        "fresh compiled binary is stale or substituted",
    )
    require(
        binary.stat().st_size == preflight["binary"]["bytes"],
        "fresh compiled binary size differs",
    )


def validate_predecessor_reuse(review: str, authority: str) -> None:
    for value in (review, authority):
        require(
            not any(f"model24-r{revision}-" in value for revision in range(12, 16)),
            "predecessor review or authority reuse is forbidden",
        )


def validate_authority_schema(package: Path, manifest: dict[str, Any]) -> None:
    schema = load_json(package / "authority-schema.json")
    paths = _base.expected_paths(package, manifest)
    require(
        schema["kind"] == "ace3_model24_r16_exact_manager_authority_schema",
        "authority schema kind differs",
    )
    require(
        schema["document_kind"] == "ace3_model24_r16_manager_exactly_once_authority",
        "authority document kind differs",
    )
    require(schema["additional_fields_allowed"] is False, "authority schema permits extra fields")
    require(
        schema["required_exact_fields"]
        == [
            "schema_version",
            "kind",
            "nonce",
            "task_id",
            "review_path",
            "review_sha256",
            *_base.REQUIRED_REVIEW_HASHES,
            "submitter_cwd",
            "child_cwd",
            "submission_argv",
            "launch_argv",
            "durable_command",
            "terminal_root",
            "registry_root",
            "receipt_path",
            "stdout_log",
            "stderr_log",
            "terminal_manifest",
            "output_namespace",
            "simulation_namespace",
            "payload_output_namespace",
            "execution_cardinality",
            "manager_exactly_once_directive",
            "retry",
            "replay",
            "resume",
            "watcher",
        ],
        "authority exact field schema differs",
    )
    require(
        schema["canonical_paths"]
        == {
            "review_path": str(paths["review"] / "review.json"),
            "authority_path": str(paths["authority"]),
            "authority_consumed_path": str(paths["authority_consumed"]),
            "terminal_root": str(paths["terminal_root"]),
            "registry_root": str(paths["registry_root"]),
            "receipt_path": str(paths["receipt"]),
            "stdout_log": str(paths["stdout_log"]),
            "stderr_log": str(paths["stderr_log"]),
            "terminal_manifest": str(paths["terminal_manifest"]),
            "output_namespace": str(paths["output"]),
            "simulation_namespace": str(paths["simulation_dir"]),
            "payload_output_namespace": str(paths["payload_output_dir"]),
        },
        "authority canonical path schema differs",
    )
    require(
        schema["exact_policy"]
        == {
            "execution_cardinality": 1,
            "manager_exactly_once_directive": True,
            "retry": False,
            "replay": False,
            "resume": False,
            "watcher": False,
        },
        "authority exact policy differs",
    )


def reject_external_wrapper(value: bool) -> None:
    require(value is False, "external wrapper around v2 is forbidden")


def validate_v2_and_preflight(package: Path, seal: dict[str, Any], manifest: dict[str, Any]) -> None:
    v2_root = package / "provenance/v2-compile-source"
    for name, expected in V2_HASHES.items():
        require(digest(v2_root / name) == expected, f"bound v2 input changed: {name}")
    v2_source_manifest = load_json(v2_root / "source-tree.json")
    require(
        v2_source_manifest["tree_sha256"] == V2_SOURCE_TREE_SHA256,
        "v2 internal source-record tree hash changed",
    )
    v2_binding = manifest["provenance"]["v2_compile_source"]
    require(v2_binding["hashes"] == V2_HASHES, "manifest v2 hash binding differs")
    require(v2_binding["source_tree_sha256"] == V2_SOURCE_TREE_SHA256, "manifest v2 tree binding differs")
    require(v2_binding["interpretation"] == "compile_evidence_only", "v2 is not compile-only evidence")
    reject_external_wrapper(v2_binding["external_wrapper"])

    overlays = load_json(package / "source-overlays.json")
    validate_overlay_document(package, overlays)
    validate_materialization(package)
    validate_make_target_text((package.parent / "source/Makefile").read_text(encoding="utf-8"))

    preflight_root = package / "evidence/production-preflight"
    preflight = load_json(preflight_root / "result.json")
    require(preflight["status"] == "PASS", "production preflight did not pass")
    require(preflight["compile_argv"] == COMPILE_ARGV, "production compile argv differs")
    require(preflight["compile_exit_code"] == 0, "production layer-0 compile failed")
    require(
        preflight["cascade_layer0_argv"] == COMPILE_ARGV,
        "actual cascade layer-0 command differs from production compile",
    )
    require(
        preflight["source_tree_sha256"] == seal["source_tree_sha256"],
        "production preflight source tree differs from sealed source",
    )
    require(
        preflight["make_target_resolution"]
        == {"layer0_exit_code": 0, "layer23_exit_code": 0},
        "cascade Make target did not resolve at both bounds",
    )
    for name in ("compile.stdout", "compile.stderr", "make-layer0.stdout", "make-layer23.stdout"):
        require(
            digest(preflight_root / name) == preflight["evidence_hashes"][name],
            f"production preflight evidence changed: {name}",
        )
    validate_compiled_binary(package, preflight)
    require(
        seal["production_preflight_sha256"] == digest(preflight_root / "result.json"),
        "production preflight is not seal-bound",
    )
    require(
        seal["source_overlays_sha256"] == digest(package / "source-overlays.json"),
        "source overlays are not seal-bound",
    )


def validate_unauthorized_ancestry(package: Path, revision: int) -> None:
    root = package / f"provenance/r{revision}-unauthorized"
    classification = load_json(root / "classification.json")
    require(
        classification["classification"] == "UNAUTHORIZED_NO_MANAGER_DIRECTIVE",
        f"r{revision} ancestry is not classified as unauthorized",
    )
    require(classification["manager_directive_present"] is False, f"r{revision} Manager directive claim is open")
    require(classification["execution_authorized"] is False, f"r{revision} execution is treated as authorized")
    require(classification["copied_byte_for_byte"] is True, f"r{revision} ancestry was not byte-preserved")
    require(classification["review_reusable"] is False, f"r{revision} review is reusable")
    require(classification["authority_reusable"] is False, f"r{revision} authority is reusable")
    hashes = classification["hashes"]
    require(set(hashes) == {path.name for path in root.iterdir()} - {"classification.json"}, f"r{revision} ancestry file set differs")
    for name, expected in hashes.items():
        require(digest(root / name) == expected, f"r{revision} ancestry changed: {name}")


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
    validate_v2_and_preflight(package, seal, manifest)
    validate_authority_schema(package, manifest)
    validate_unauthorized_ancestry(package, 14)
    validate_unauthorized_ancestry(package, 15)
    paths = _base.expected_paths(package, manifest)
    validate_predecessor_reuse(
        str(paths["review"] / "review.json"),
        str(paths["authority"]),
    )
    require(
        manifest["review_policy"]["predecessor_review_reusable"] is False,
        "predecessor review reuse policy is open",
    )
    require(
        manifest["review_policy"]["predecessor_authority_reusable"] is False,
        "predecessor authority reuse policy is open",
    )
    result.update(
        {
            "production_preflight": "PASS",
            "negative_case_count": len(REQUIRED_NEGATIVE_CASES),
            "v2_interpretation": "compile_evidence_only",
            "r14_execution_authorized": False,
            "r15_execution_authorized": False,
        }
    )
    return result


def __getattr__(name: str) -> Any:
    return getattr(_base, name)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=("package", "review", "pre-submit", "launch"), required=True)
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
