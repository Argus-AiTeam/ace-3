#!/usr/bin/env python3
"""Fail-closed validation for the Model24 r19 reboot-crash successor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
from typing import Any


R18_NONCE = "33836f56ad76c8e0"
R18_TASK_ID = f"ace3-model24-r18-42c895c-{R18_NONCE}"
R18_RECEIPT_SHA256 = (
    "fd626aff158020a7db1a000c559e2db11ee4ad2a3ee9f1bf991ad8d471552a71"
)
R18_AUTHORITY_SHA256 = (
    "7f494c4e4fa1e56961f05ace96ccd3de454ce18b42dfaa77898ae42c504513f7"
)
REBOOT_CRASH_CASES = (
    "reboot-crash-positive",
    "fabricated-exit-code",
    "unexpected-exit-sidecar",
    "substituted-exit-sidecar",
    "non-crashed-receipt",
    "wrong-process-identity",
    "wrong-run-identity",
    "mutated-receipt",
    "mutated-authority",
    "partial-layer-tampering",
    "false-layer9-completion",
    "r18-identity-reuse",
    "duplicate-terminal-sealing",
)
ADDITIONAL_REVIEW_HASHES = (
    "lifecycle_overlay_sha256",
    "r18_crash_ancestry_sha256",
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
    _PACKAGE / "validate-r18-package.py",
    "model24_r19_full_contract_base_validator",
)
_base_package_review_hashes = _base.package_review_hashes
REQUIRED_POSITIVE_FIXTURES = _base.REQUIRED_POSITIVE_FIXTURES
REQUIRED_NEGATIVE_CASES = _base.REQUIRED_NEGATIVE_CASES
REQUIRED_REVIEW_HASHES = (
    *_base.REQUIRED_REVIEW_HASHES,
    *ADDITIONAL_REVIEW_HASHES,
)


def _propagate_contract(module: Any) -> None:
    seen: set[int] = set()
    while id(module) not in seen:
        seen.add(id(module))
        module.REQUIRED_REVIEW_HASHES = REQUIRED_REVIEW_HASHES
        if not hasattr(module, "_base"):
            break
        module = module._base


_propagate_contract(_base)


def package_review_hashes(package: Path) -> dict[str, str]:
    hashes = _base_package_review_hashes(package)
    hashes.update(
        {
            "lifecycle_overlay_sha256": digest(
                package / "lifecycle-overlay.json"
            ),
            "r18_crash_ancestry_sha256": digest(
                package / "provenance/r18-crashed/classification.json"
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


def tree_records(root: Path) -> list[dict[str, Any]]:
    require(root.is_dir() and not root.is_symlink(), f"directory required: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        require(not stat.S_ISLNK(metadata.st_mode), f"symlink forbidden: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            records.append({"path": relative, "kind": "directory", "mode": mode})
        else:
            require(stat.S_ISREG(metadata.st_mode), f"regular file required: {path}")
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": mode,
                    "bytes": metadata.st_size,
                    "sha256": digest(path),
                }
            )
    return records


def tree_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(records)).hexdigest()


def validate_partial_summary(summary: dict[str, Any]) -> None:
    require(
        summary.get("authenticated_complete_layers") == list(range(9)),
        "r18 authenticated partial layer set differs",
    )
    require(
        summary.get("incomplete_layer") == 9
        and summary.get("layer9_completion_record_present") is False,
        "r18 layer 9 is not classified as incomplete",
    )
    require(
        summary.get("resumable") is False
        and summary.get("execution_output_reusable") is False,
        "r18 partial execution was made reusable",
    )


def validate_partial_layers(output_root: Path, summary: dict[str, Any]) -> None:
    validate_partial_summary(summary)
    layers = output_root / "rtl-cascade/layers"
    require(
        {path.name for path in layers.iterdir() if path.is_dir()}
        == {f"layer{index:02d}" for index in range(10)},
        "r18 partial layer directory set differs",
    )
    records = summary["completion_records"]
    require(set(records) == {str(index) for index in range(9)}, "layer records differ")
    for index in range(9):
        layer = layers / f"layer{index:02d}"
        record_path = layer / "record.json"
        record = load_json(record_path)
        require(record["layer_index"] == index, f"layer {index} identity differs")
        require(
            record["comparison"]["within_tolerance"] is True
            and record["comparison"]["failure_count"] == 0,
            f"layer {index} completion is not authenticated",
        )
        require(
            records[str(index)] == {
                "bytes": record_path.stat().st_size,
                "sha256": digest(record_path),
                "output_raw_sha256": record["output_raw_sha256"],
            },
            f"layer {index} completion record changed",
        )
        for name, expected in record["artifacts"].items():
            artifact = layer / "raw" / name
            require(
                artifact.stat().st_size == expected["bytes"]
                and digest(artifact) == expected["sha256"],
                f"layer {index} authenticated artifact changed: {name}",
            )
    require(
        not (layers / "layer09/record.json").exists(),
        "r18 layer 9 falsely acquired a completion record",
    )


def validate_authority_pair(root: Path) -> None:
    original = root / "authority/original.json"
    consumed = root / "authority/consumed.json"
    require(
        digest(original) == R18_AUTHORITY_SHA256
        and digest(consumed) == R18_AUTHORITY_SHA256,
        "r18 authority binding changed",
    )
    require(
        original.read_bytes() == consumed.read_bytes(),
        "r18 consumed authority differs from original",
    )


def validate_r18_crash_ancestry(package: Path) -> None:
    root = package / "provenance/r18-crashed"
    classification = load_json(root / "classification.json")
    require(
        classification["schema_version"] == 1
        and classification["kind"] == "ace3_model24_r18_consumed_crash_ancestry"
        and classification["nonce"] == R18_NONCE
        and classification["task_id"] == R18_TASK_ID
        and classification["classification"] == "LAWFUL_CONSUMED_INFRASTRUCTURE_CRASH"
        and classification["authority_consumed"] is True
        and classification["exit_code_observed"] is False
        and classification["exit_code"] is None
        and classification["launch_terminal_present"] is False
        and classification["exit_sidecar_present"] is False
        and classification["copied_byte_for_byte"] is True,
        "r18 crash classification differs",
    )
    require(
        classification["receipt_sha256"] == R18_RECEIPT_SHA256
        and classification["authority_sha256"] == R18_AUTHORITY_SHA256,
        "r18 crash authority or receipt binding differs",
    )
    for reusable in (
        "nonce_reusable",
        "task_id_reusable",
        "review_reusable",
        "authority_reusable",
        "execution_output_reusable",
        "terminal_reusable",
        "predecessor_reusable",
    ):
        require(classification[reusable] is False, f"r18 {reusable} was enabled")
    for name, expected in classification["roots"].items():
        observed = tree_records(root / name)
        require(
            observed == expected["records"]
            and tree_sha256(observed) == expected["tree_sha256"],
            f"r18 copied ancestry changed: {name}",
        )
    receipt = load_json(
        root
        / "terminal/.argus_subagents"
        / f"{R18_TASK_ID}.json"
    )
    require(
        digest(
            root
            / "terminal/.argus_subagents"
            / f"{R18_TASK_ID}.json"
        )
        == R18_RECEIPT_SHA256
        and receipt["state"] == "crashed"
        and receipt["task_id"] == R18_TASK_ID
        and "exit_code" not in receipt,
        "r18 crashed receipt differs",
    )
    validate_authority_pair(root)
    validate_partial_layers(root / "output", classification["partial_layers"])


def validate_no_r18_reuse(values: list[str]) -> None:
    forbidden = (
        R18_NONCE,
        R18_TASK_ID,
        "ace3-model24-r18-review-20260829-33836f56ad76c8e0",
        "ace3-model24-r18-authority-20260829-33836f56ad76c8e0",
        "ace3-model24-r18-output-20260829-33836f56ad76c8e0",
        "ace3-model24-r18-terminal-20260829-33836f56ad76c8e0",
    )
    for value in values:
        require(
            not any(token in value for token in forbidden),
            "r18 identity or execution namespace reuse rejected",
        )


def validate_infrastructure_crash_layout(
    paths: dict[str, Path],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = paths["receipt"]
    receipt = load_json(receipt_path)
    require(receipt.get("state") == "crashed", "crash receipt state differs")
    require(
        receipt.get("task_id") == manifest["identity"]["task_id"],
        "crash receipt task identity differs",
    )
    run_id = receipt.get("run_id")
    require(
        isinstance(run_id, str)
        and run_id.startswith(f"{manifest['identity']['task_id']}-")
        and len(run_id) > len(manifest["identity"]["task_id"]) + 1,
        "crash receipt run identity differs",
    )
    require(
        isinstance(receipt.get("pid"), int)
        and receipt["pid"] > 0
        and isinstance(receipt.get("worker_pid"), int)
        and receipt["worker_pid"] > 0,
        "crash receipt process identity differs",
    )
    require(
        receipt.get("command") == manifest["execution"]["durable_command"]
        and receipt.get("cwd") == str(paths["child_cwd"])
        and receipt.get("mode") == "direct",
        "crash receipt launch identity differs",
    )
    require("exit_code" not in receipt, "crash receipt fabricated an exit code")
    expected_sidecar = (
        paths["log_dir"] / f"exit_code.{run_id}"
    )
    require(
        receipt.get("exit_status_path") == str(expected_sidecar),
        "crash receipt substituted the exit sidecar path",
    )
    require(not expected_sidecar.exists(), "crash receipt has an exit sidecar")
    require(
        receipt.get("stdout_log")
        == f".argus_subagents/{manifest['identity']['task_id']}_logs/stdout.log"
        and receipt.get("stderr_log")
        == f".argus_subagents/{manifest['identity']['task_id']}_logs/stderr.log",
        "crash receipt log identity differs",
    )
    for name in ("stdout_log", "stderr_log"):
        path = paths[name]
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_ISREG(path.stat().st_mode),
            f"crash {name} is absent or substituted",
        )
    require(
        not paths["terminal_manifest"].exists(),
        "crash terminal was already sealed",
    )
    launch_terminal = paths["output"] / "launch-terminal.json"
    require(
        not launch_terminal.exists(),
        "crash receipt conflicts with a launch terminal",
    )
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "stdout": paths["stdout_log"],
        "stderr": paths["stderr_log"],
        "exit_sidecar": expected_sidecar,
        "launch_terminal": launch_terminal,
    }


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
    validate_r18_crash_ancestry(package)
    overlay = load_json(package / "lifecycle-overlay.json")
    require(
        overlay["kind"] == "ace3_model24_r19_reboot_crash_lifecycle_overlay"
        and overlay["materialized_sha256"] == digest(package / "lifecycle.py")
        and overlay["exit_code_representation"] == "unobserved-null",
        "r19 lifecycle overlay differs",
    )
    require(
        seal["lifecycle_overlay_sha256"]
        == digest(package / "lifecycle-overlay.json")
        and seal["r18_crash_ancestry_sha256"]
        == digest(package / "provenance/r18-crashed/classification.json"),
        "r19 crash evidence is not seal-bound",
    )
    validate_no_r18_reuse(
        [
            manifest["identity"]["nonce"],
            manifest["identity"]["task_id"],
            *[str(value) for value in manifest["namespaces"].values()],
        ]
    )
    result.update(
        {
            "r18_lawful_consumed_infrastructure_crash": True,
            "r18_authenticated_complete_layers": list(range(9)),
            "r18_incomplete_layer": 9,
            "reboot_crash_case_count": len(REBOOT_CRASH_CASES),
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
