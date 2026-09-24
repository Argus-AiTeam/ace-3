#!/usr/bin/env python3
"""Fail-closed validation for the Model24 r20 review-gap repair."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
from typing import Any


R19_NONCE = "3ad61cc4058091ca"
R19_TASK_ID = f"ace3-model24-r19-42c895c-{R19_NONCE}"
R19_REVIEW_SHA256 = (
    "68d005fe808da68a0d3d94b48652d511b934c83d0c608e5acebdd5d14f9d5d9c"
)
ADDITIONAL_NEGATIVE_CASES = (
    "missing-reboot-crash-case",
    "substituted-reboot-crash-case",
)
ADDITIONAL_REVIEW_HASHES = (
    "r19_rejected_ancestry_sha256",
    "controller_entry_fixture_sha256",
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


def _resolve_package() -> Path:
    local_package = Path(__file__).resolve().parent
    if (local_package / "validate-r19-package.py").is_file():
        return local_package
    require("--package" in sys.argv, "repository validator requires --package")
    package_index = sys.argv.index("--package")
    require(
        package_index + 1 < len(sys.argv),
        "--package requires a path",
    )
    return Path(sys.argv[package_index + 1]).resolve()


_PACKAGE = _resolve_package()
_base = _load_module(
    _PACKAGE / "validate-r19-package.py",
    "model24_r20_inherited_r19_validator",
)
_base_package_review_hashes = _base.package_review_hashes
INHERITED_NEGATIVE_CASES = tuple(_base.REQUIRED_NEGATIVE_CASES)
REQUIRED_NEGATIVE_CASES = (
    *INHERITED_NEGATIVE_CASES,
    *ADDITIONAL_NEGATIVE_CASES,
)
REQUIRED_POSITIVE_FIXTURES = tuple(_base.REQUIRED_POSITIVE_FIXTURES)
REBOOT_CRASH_CASES = tuple(_base.REBOOT_CRASH_CASES)
REQUIRED_REVIEW_HASHES = (
    *_base.REQUIRED_REVIEW_HASHES,
    *ADDITIONAL_REVIEW_HASHES,
)


def passing_review_test_results() -> dict[str, Any]:
    return {
        "positive_fixtures": [
            {"name": name, "status": "PASS"}
            for name in REQUIRED_POSITIVE_FIXTURES
        ],
        "negative_cases": [
            {"name": name, "status": "PASS"}
            for name in REQUIRED_NEGATIVE_CASES
        ],
        "reboot_crash_cases": [
            {"name": name, "status": "PASS"}
            for name in REBOOT_CRASH_CASES
        ],
        "execution_invocations": 0,
        "launcher_invocations": 0,
        "model_invocations": 0,
        "simulator_invocations": 0,
        "fixture_controller_entry_invocations": 1,
        "fixture_artifact_authentications": 3,
        "real_controller_invocations": 0,
        "real_model24_invocations": 0,
        "real_rtl_simulator_invocations": 0,
    }


def _validate_named_results(
    value: dict[str, Any],
    field: str,
    required_names: tuple[str, ...],
    *,
    require_pass: bool,
) -> list[str]:
    results = value.get(field)
    require(isinstance(results, list), f"review {field} must be a list")
    require(
        all(
            isinstance(result, dict)
            and set(result) == {"name", "status"}
            and isinstance(result["name"], str)
            and result["status"] in {"PASS", "FAIL"}
            for result in results
        ),
        f"review {field} entries are malformed",
    )
    names = [result["name"] for result in results]
    if field == "reboot_crash_cases":
        require(
            len(names) == len(required_names)
            and len(set(names)) == len(names)
            and set(names) == set(required_names),
            "review reboot crash case names differ from required set",
        )
    else:
        require(
            names == list(required_names),
            f"review {field} names or order differ from required suite",
        )
    statuses = [result["status"] for result in results]
    if require_pass:
        require(
            all(status == "PASS" for status in statuses),
            f"review {field} contains a non-PASS entry",
        )
    return statuses


def validate_test_results(value: object, verdict: str) -> None:
    require(isinstance(value, dict), "review test results must be an object")
    expected_fields = {
        "positive_fixtures",
        "negative_cases",
        "reboot_crash_cases",
        "execution_invocations",
        "launcher_invocations",
        "model_invocations",
        "simulator_invocations",
        "fixture_controller_entry_invocations",
        "fixture_artifact_authentications",
        "real_controller_invocations",
        "real_model24_invocations",
        "real_rtl_simulator_invocations",
    }
    require(set(value) == expected_fields, "review test result fields differ")
    positive_statuses = _validate_named_results(
        value,
        "positive_fixtures",
        REQUIRED_POSITIVE_FIXTURES,
        require_pass=False,
    )
    negative_statuses = _validate_named_results(
        value,
        "negative_cases",
        REQUIRED_NEGATIVE_CASES,
        require_pass=False,
    )
    _validate_named_results(
        value,
        "reboot_crash_cases",
        REBOOT_CRASH_CASES,
        require_pass=True,
    )
    for field in (
        "execution_invocations",
        "launcher_invocations",
        "model_invocations",
        "simulator_invocations",
        "real_controller_invocations",
        "real_model24_invocations",
        "real_rtl_simulator_invocations",
    ):
        require(value[field] == 0, f"review reports nonzero {field}")
    require(
        value["fixture_controller_entry_invocations"] == 1
        and value["fixture_artifact_authentications"] == 3,
        "review fixture counters differ",
    )
    if verdict == "ACCEPT":
        require(
            all(
                status == "PASS"
                for status in positive_statuses + negative_statuses
            ),
            "ACCEPT review contains a failed inert result",
        )
    else:
        require(
            any(status == "FAIL" for status in positive_statuses + negative_statuses),
            "REJECT review must identify a failed result",
        )


def package_review_hashes(package: Path) -> dict[str, str]:
    hashes = _base_package_review_hashes(package)
    hashes.update(
        {
            "r19_rejected_ancestry_sha256": digest(
                package / "provenance/r19-rejected/classification.json"
            ),
            "controller_entry_fixture_sha256": digest(
                package / "controller-entry-fixture.json"
            ),
        }
    )
    return hashes


def _propagate_contract(module: Any) -> None:
    seen: set[int] = set()
    while id(module) not in seen:
        seen.add(id(module))
        module.REQUIRED_NEGATIVE_CASES = REQUIRED_NEGATIVE_CASES
        module.REQUIRED_REVIEW_HASHES = REQUIRED_REVIEW_HASHES
        module.passing_review_test_results = passing_review_test_results
        module.validate_test_results = validate_test_results
        module.package_review_hashes = package_review_hashes
        if not hasattr(module, "_base"):
            break
        module = module._base


_propagate_contract(_base)


def byte_tree_records(root: Path) -> list[dict[str, Any]]:
    require(root.is_dir() and not root.is_symlink(), f"directory required: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        require(not stat.S_ISLNK(metadata.st_mode), f"symlink forbidden: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            records.append({"path": relative, "kind": "directory"})
        else:
            require(stat.S_ISREG(metadata.st_mode), f"regular file required: {path}")
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "bytes": metadata.st_size,
                    "sha256": digest(path),
                }
            )
    return records


def validate_r19_rejected_ancestry(package: Path) -> None:
    root = package / "provenance/r19-rejected"
    classification = load_json(root / "classification.json")
    require(
        classification["schema_version"] == 1
        and classification["kind"]
        == "ace3_model24_r19_rejected_immutable_ancestry"
        and classification["nonce"] == R19_NONCE
        and classification["task_id"] == R19_TASK_ID
        and classification["review_sha256"] == R19_REVIEW_SHA256
        and classification["verdict"] == "REJECT"
        and classification["copied_byte_for_byte"] is True
        and classification["reusable"] is False,
        "r19 rejected ancestry classification differs",
    )
    for name in ("candidate", "review"):
        require(
            byte_tree_records(root / name)
            == classification["roots"][name]["records"],
            f"r19 rejected {name} bytes changed",
        )
    review_path = root / "review/review.json"
    review = load_json(review_path)
    require(
        digest(review_path) == R19_REVIEW_SHA256
        and review["nonce"] == R19_NONCE
        and review["task_id"] == R19_TASK_ID
        and review["verdict"] == "REJECT"
        and review["manager_may_issue_execution_authority"] is False,
        "r19 REJECT review binding differs",
    )
    require(
        review["decisive_findings"]
        == [
            "fresh-inert-contract-tests: fresh controller-entry probe was not inert",
            (
                "review-schema-reboot-results-enforcement: sealed validator "
                "accepts ACCEPT review with all declared reboot_crash_cases omitted"
            ),
        ],
        "r19 decisive findings differ",
    )


def validate_inert_fixture_contract(package: Path) -> None:
    fixture = load_json(package / "controller-entry-fixture.json")
    require(
        fixture["schema_version"] == 1
        and fixture["kind"] == "ace3_model24_r20_inert_controller_entry_fixture"
        and fixture["fixture_only"] is True
        and fixture["source_sha256"]
        == digest(package / "test-controller-entry.py")
        and fixture["assertions"]
        == [
            "exclusive-path-creation",
            "synthetic-artifact-authentication",
            "directory-before-fixture-entry-order",
        ]
        and fixture["expected_counters"]
        == {
            "fixture_artifact_authentications": 3,
            "fixture_artifact_writes": 3,
            "fixture_entry_calls": 1,
            "real_controller_invocations": 0,
            "real_model24_invocations": 0,
            "real_rtl_simulator_invocations": 0,
        },
        "controller-entry fixture contract differs",
    )


def validate_no_r19_reuse(values: list[str]) -> None:
    forbidden = (
        R19_NONCE,
        R19_TASK_ID,
        f"ace3-model24-r19-review-20260829-{R19_NONCE}",
        f"ace3-model24-r19-authority-20260829-{R19_NONCE}",
        f"ace3-model24-r19-output-20260829-{R19_NONCE}",
        f"ace3-model24-r19-terminal-20260829-{R19_NONCE}",
    )
    for value in values:
        require(
            not any(token in value for token in forbidden),
            "r19 identity or execution namespace reuse rejected",
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
    validate_r19_rejected_ancestry(package)
    validate_inert_fixture_contract(package)
    require(
        seal["r19_rejected_ancestry_sha256"]
        == digest(package / "provenance/r19-rejected/classification.json")
        and seal["controller_entry_fixture_sha256"]
        == digest(package / "controller-entry-fixture.json"),
        "r20 repair artifacts are not seal-bound",
    )
    contract = load_json(package / "launch-contract.json")
    require(
        contract["review"]["required_reboot_crash_cases"]
        == list(REBOOT_CRASH_CASES)
        and contract["review"]["required_negative_cases"]
        == list(REQUIRED_NEGATIVE_CASES),
        "r20 review suite contract differs",
    )
    validate_no_r19_reuse(
        [
            manifest["identity"]["nonce"],
            manifest["identity"]["task_id"],
            *[str(value) for value in manifest["namespaces"].values()],
        ]
    )
    result.update(
        {
            "r19_reject_bound": True,
            "inert_controller_fixture": "PASS",
            "negative_case_count": len(REQUIRED_NEGATIVE_CASES),
            "reboot_crash_case_count": len(REBOOT_CRASH_CASES),
            "real_controller_invocations": 0,
            "real_model24_invocations": 0,
            "real_rtl_simulator_invocations": 0,
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
