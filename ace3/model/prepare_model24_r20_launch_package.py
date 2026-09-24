#!/usr/bin/env python3
"""Prepare the Model24 r20 review-gap repair without executing Model24."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import time
from typing import Any


REPOSITORY = Path("/home/argustest/ace3-argus")
R20_VALIDATOR = (
    REPOSITORY / "ace3/model/validate_model24_r20_launch_package.py"
)
MODEL_PYTHON = Path("/home/argustest/miniconda3/bin/python3")
R19_NONCE = "3ad61cc4058091ca"
R19_TASK_ID = f"ace3-model24-r19-42c895c-{R19_NONCE}"
R19_BASE = Path(
    f"/home/argustest/ace3-model24-r19-prep-20260829-{R19_NONCE}"
)
R19_REVIEW = Path(
    f"/home/argustest/ace3-model24-r19-review-20260829-{R19_NONCE}"
)
R19_REVIEW_SHA256 = (
    "68d005fe808da68a0d3d94b48652d511b934c83d0c608e5acebdd5d14f9d5d9c"
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
    value = json.loads(path.read_text(encoding="ascii"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write(path: Path, payload: bytes, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(payload)
    path.chmod(mode)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = stat.S_IMODE(path.lstat().st_mode)
        require(not stat.S_ISLNK(path.lstat().st_mode), f"symlink forbidden: {path}")
        path.chmod(mode & ~0o222)
    root.chmod(stat.S_IMODE(root.lstat().st_mode) & ~0o222)


def writable_tree(root: Path) -> None:
    root.chmod(stat.S_IMODE(root.lstat().st_mode) | 0o700)
    for path in root.rglob("*"):
        if not path.is_symlink():
            path.chmod(stat.S_IMODE(path.lstat().st_mode) | 0o600)


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


def tree_records(root: Path) -> list[dict[str, Any]]:
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


def copy_tree_bytes(source: Path, target: Path) -> list[dict[str, Any]]:
    before = byte_tree_records(source)
    require(not target.exists(), f"copy target already exists: {target}")
    shutil.copytree(source, target)
    require(byte_tree_records(source) == before, f"copy changed source: {source}")
    require(byte_tree_records(target) == before, f"copy differs: {source}")
    return before


def transform_r20_text(payload: str, nonce: str) -> str:
    old_base = str(R19_BASE)
    new_base = f"/home/argustest/ace3-model24-r20-prep-20260829-{nonce}"
    replacements = (
        (old_base, new_base),
        (
            f"/home/argustest/ace3-model24-r19-review-20260829-{R19_NONCE}",
            f"/home/argustest/ace3-model24-r20-review-20260829-{nonce}",
        ),
        (
            f"/home/argustest/ace3-model24-r19-authority-20260829-{R19_NONCE}",
            f"/home/argustest/ace3-model24-r20-authority-20260829-{nonce}",
        ),
        (
            f"/home/argustest/ace3-model24-r19-output-20260829-{R19_NONCE}",
            f"/home/argustest/ace3-model24-r20-output-20260829-{nonce}",
        ),
        (
            f"/home/argustest/ace3-model24-r19-terminal-20260829-{R19_NONCE}",
            f"/home/argustest/ace3-model24-r20-terminal-20260829-{nonce}",
        ),
        (R19_TASK_ID, f"ace3-model24-r20-42c895c-{nonce}"),
        (R19_NONCE, nonce),
        ("R19", "R20"),
        ("r19", "r20"),
    )
    for old, new in replacements:
        payload = payload.replace(old, new)
    return payload


def inert_controller_fixture_source() -> bytes:
    return r'''#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,tempfile

FIXTURE_ARTIFACTS={
    "fixture-terminal.txt":b"fixture-terminal:PASS\n",
    "fixture-events.hex":b"f1000000\nf2000000\n",
    "fixture-manifest.json":b'{"fixture":true,"status":"PASS"}\n',
}
REQUIRED_NEGATIVE_CASES=(
    "absent-simulation-directory",
    "absent-payload-output-directory",
    "foreign-payload-directory",
    "pre-existing-payload-directory",
    "symlinked-payload-directory",
    "payload-directory-outside-output-root",
    "partial-payload-directory-creation",
    "foreign-output-root-entry",
    "absent-controller-simulation-artifact",
    "stale-controller-simulation-artifact",
    "wrong-parent-controller-simulation-artifact",
    "pre-existing-controller-simulation-artifact",
    "symlinked-controller-simulation-artifact",
    "substituted-controller-simulation-artifact",
)

class FixtureViolation(RuntimeError):
    pass

def require(condition,message,exception=FixtureViolation):
    if not condition:
        raise exception(message)

def create_paths(output,simulation,payload,events):
    require(not output.exists(),"fixture output collision")
    output.mkdir()
    events.append("create:output")
    simulation.mkdir()
    events.append("create:simulation")
    payload.mkdir()
    events.append("create:payload")

def validate_layout(output,simulation,payload):
    require(output.is_dir() and not output.is_symlink(),"fixture output is absent")
    require(
        simulation.name=="fixture-simulation"
        and payload.name=="fixture-payload"
        and simulation.parent==output
        and payload.parent==output,
        "fixture path is outside the canonical fixture output",
    )
    require(
        simulation.is_dir()
        and payload.is_dir()
        and not simulation.is_symlink()
        and not payload.is_symlink(),
        "fixture child path is absent or symlinked",
    )
    require(
        {path.name for path in output.iterdir()}
        =={"fixture-simulation","fixture-payload"},
        "fixture output contains absent or foreign entries",
    )

def fixture_entry(output,simulation,payload,events,counters):
    validate_layout(output,simulation,payload)
    require(not any(simulation.iterdir()),"fixture artifacts already exist")
    counters["fixture_entry_calls"]+=1
    events.append("fixture:entry")
    for name,content in FIXTURE_ARTIFACTS.items():
        path=simulation/name
        require(not path.exists(),"fixture artifact collision")
        path.write_bytes(content)
        counters["fixture_artifact_writes"]+=1

def authenticate(output,simulation,payload,events,counters):
    validate_layout(output,simulation,payload)
    require(
        {path.name for path in simulation.iterdir()}==set(FIXTURE_ARTIFACTS),
        "fixture artifact set differs",
    )
    records={}
    for name,content in FIXTURE_ARTIFACTS.items():
        path=simulation/name
        require(
            path.is_file() and not path.is_symlink(),
            "fixture artifact is absent or symlinked",
        )
        require(path.resolve().parent==simulation.resolve(),"fixture artifact parent differs")
        require(path.read_bytes()==content,"fixture artifact authentication failed")
        records[name]={
            "bytes":len(content),
            "sha256":hashlib.sha256(content).hexdigest(),
        }
    counters["fixture_artifact_authentications"]+=len(records)
    events.append("authenticate:fixture")
    return records

def validate_order(observed):
    require(
        observed==[
            "create:output","create:simulation","create:payload",
            "fixture:entry","authenticate:fixture",
        ],
        "fixture entry ordering differs",
    )

def direct_layout(root,name):
    output=root/name
    simulation=output/"fixture-simulation"
    payload=output/"fixture-payload"
    output.mkdir()
    simulation.mkdir()
    payload.mkdir()
    return output,simulation,payload

def populate_artifacts(simulation):
    for name,content in FIXTURE_ARTIFACTS.items():
        (simulation/name).write_bytes(content)

def run_fixture_probe():
    events=[]
    counters={
        "fixture_artifact_authentications":0,
        "fixture_artifact_writes":0,
        "fixture_entry_calls":0,
        "real_controller_invocations":0,
        "real_model24_invocations":0,
        "real_rtl_simulator_invocations":0,
    }
    negatives=[]

    def expect_rejection(name,callback,expected):
        try:
            callback()
        except FixtureViolation as error:
            require(
                expected in str(error),
                f"{name} rejected for wrong reason: {error}",
                SystemExit,
            )
        else:
            raise SystemExit(f"fixture unexpectedly accepted: {name}")
        negatives.append(name)

    with tempfile.TemporaryDirectory(prefix="ace3-r20-inert-entry-") as temporary:
        root=Path(temporary)
        output=root/"positive-output"
        simulation=output/"fixture-simulation"
        payload=output/"fixture-payload"
        create_paths(output,simulation,payload,events)
        fixture_entry(output,simulation,payload,events,counters)
        records=authenticate(output,simulation,payload,events,counters)
        validate_order(events)

        absent_simulation=root/"absent-simulation"
        absent_simulation.mkdir()
        (absent_simulation/"fixture-payload").mkdir()
        expect_rejection(
            "absent-simulation-directory",
            lambda:validate_layout(
                absent_simulation,
                absent_simulation/"fixture-simulation",
                absent_simulation/"fixture-payload",
            ),
            "absent or symlinked",
        )

        absent_payload=root/"absent-payload"
        absent_payload.mkdir()
        (absent_payload/"fixture-simulation").mkdir()
        expect_rejection(
            "absent-payload-output-directory",
            lambda:validate_layout(
                absent_payload,
                absent_payload/"fixture-simulation",
                absent_payload/"fixture-payload",
            ),
            "absent or symlinked",
        )

        foreign_output=root/"foreign-payload"
        foreign_output.mkdir()
        (foreign_output/"fixture-simulation").mkdir()
        (foreign_output/"foreign-payload").mkdir()
        expect_rejection(
            "foreign-payload-directory",
            lambda:validate_layout(
                foreign_output,
                foreign_output/"fixture-simulation",
                foreign_output/"foreign-payload",
            ),
            "outside the canonical fixture output",
        )

        collision=root/"pre-existing-payload"
        collision.mkdir()
        expect_rejection(
            "pre-existing-payload-directory",
            lambda:create_paths(
                collision,
                collision/"fixture-simulation",
                collision/"fixture-payload",
                [],
            ),
            "collision",
        )

        symlink_output=root/"symlinked-payload"
        symlink_output.mkdir()
        external=root/"external-simulation"
        external.mkdir()
        (symlink_output/"fixture-simulation").symlink_to(
            external,target_is_directory=True
        )
        (symlink_output/"fixture-payload").mkdir()
        expect_rejection(
            "symlinked-payload-directory",
            lambda:validate_layout(
                symlink_output,
                symlink_output/"fixture-simulation",
                symlink_output/"fixture-payload",
            ),
            "absent or symlinked",
        )

        outside_output=root/"outside-output"
        outside_output.mkdir()
        outside_simulation=root/"fixture-simulation"
        outside_simulation.mkdir()
        (outside_output/"fixture-payload").mkdir()
        expect_rejection(
            "payload-directory-outside-output-root",
            lambda:validate_layout(
                outside_output,
                outside_simulation,
                outside_output/"fixture-payload",
            ),
            "outside the canonical fixture output",
        )

        partial_output=root/"partial-output"
        partial_output.mkdir()
        (partial_output/"fixture-simulation").mkdir()
        expect_rejection(
            "partial-payload-directory-creation",
            lambda:validate_layout(
                partial_output,
                partial_output/"fixture-simulation",
                partial_output/"fixture-payload",
            ),
            "absent or symlinked",
        )

        extra_output,extra_simulation,extra_payload=direct_layout(
            root,"foreign-output-entry"
        )
        (extra_output/"foreign").write_bytes(b"")
        expect_rejection(
            "foreign-output-root-entry",
            lambda:validate_layout(extra_output,extra_simulation,extra_payload),
            "absent or foreign entries",
        )

        absent_output,absent_artifacts,absent_artifact_payload=direct_layout(
            root,"absent-artifact"
        )
        for name,content in list(FIXTURE_ARTIFACTS.items())[1:]:
            (absent_artifacts/name).write_bytes(content)
        expect_rejection(
            "absent-controller-simulation-artifact",
            lambda:authenticate(
                absent_output,
                absent_artifacts,
                absent_artifact_payload,
                [],
                counters,
            ),
            "artifact set differs",
        )

        stale_output,stale_simulation,stale_payload=direct_layout(
            root,"stale-artifact"
        )
        populate_artifacts(stale_simulation)
        (stale_simulation/"fixture-terminal.txt").write_bytes(b"stale\n")
        expect_rejection(
            "stale-controller-simulation-artifact",
            lambda:authenticate(
                stale_output,stale_simulation,stale_payload,[],counters
            ),
            "authentication failed",
        )

        wrong_parent_output,_,wrong_parent_payload=direct_layout(
            root,"wrong-artifact-parent"
        )
        other_output,other_simulation,_=direct_layout(
            root,"other-artifact-parent"
        )
        populate_artifacts(other_simulation)
        expect_rejection(
            "wrong-parent-controller-simulation-artifact",
            lambda:authenticate(
                wrong_parent_output,
                other_simulation,
                wrong_parent_payload,
                [],
                counters,
            ),
            "outside the canonical fixture output",
        )
        del other_output

        existing_output,existing_simulation,existing_payload=direct_layout(
            root,"pre-existing-artifact"
        )
        (existing_simulation/"fixture-terminal.txt").write_bytes(b"existing\n")
        expect_rejection(
            "pre-existing-controller-simulation-artifact",
            lambda:fixture_entry(
                existing_output,existing_simulation,existing_payload,[],counters
            ),
            "already exist",
        )

        link_output,link_simulation,link_payload=direct_layout(
            root,"symlinked-artifact"
        )
        populate_artifacts(link_simulation)
        external_terminal=root/"external-terminal.txt"
        external_terminal.write_bytes(FIXTURE_ARTIFACTS["fixture-terminal.txt"])
        (link_simulation/"fixture-terminal.txt").unlink()
        (link_simulation/"fixture-terminal.txt").symlink_to(external_terminal)
        expect_rejection(
            "symlinked-controller-simulation-artifact",
            lambda:authenticate(
                link_output,link_simulation,link_payload,[],counters
            ),
            "absent or symlinked",
        )

        substitute_output,substitute_simulation,substitute_payload=direct_layout(
            root,"substituted-artifact"
        )
        populate_artifacts(substitute_simulation)
        (substitute_simulation/"fixture-events.hex").write_bytes(b"substituted\n")
        expect_rejection(
            "substituted-controller-simulation-artifact",
            lambda:authenticate(
                substitute_output,
                substitute_simulation,
                substitute_payload,
                [],
                counters,
            ),
            "authentication failed",
        )

    require(
        negatives==list(REQUIRED_NEGATIVE_CASES),
        "fixture negative case order differs",
        SystemExit,
    )
    require(counters=={
        "fixture_artifact_authentications":3,
        "fixture_artifact_writes":3,
        "fixture_entry_calls":1,
        "real_controller_invocations":0,
        "real_model24_invocations":0,
        "real_rtl_simulator_invocations":0,
    },"fixture counters differ",SystemExit)
    return {
        "schema_version":1,
        "kind":"ace3_model24_r20_inert_controller_entry_probe",
        "status":"PASS",
        "fixture_only":True,
        "events":events,
        "artifacts":records,
        "negative_cases":negatives,
        "counters":counters,
    }

def main():
    print(json.dumps(
        run_fixture_probe(),sort_keys=True,separators=(",",":")
    ))

if __name__=="__main__":
    main()
'''.encode("ascii")


def patch_negative_suite(payload: str) -> str:
    legacy_probe = '''    path_order = run_controller_entry_integration(package)
    cases.extend(path_order["negative_cases"])
'''
    fixture_probe = '''    fixture_module = load_module(
        package / "test-controller-entry.py",
        "model24_r20_standalone_controller_fixture",
    )
    fixture_result = fixture_module.run_fixture_probe()
    require(
        fixture_result["status"] == "PASS"
        and fixture_result["fixture_only"] is True
        and fixture_result["negative_cases"]
        == list(fixture_module.REQUIRED_NEGATIVE_CASES)
        and fixture_result["counters"]["real_controller_invocations"] == 0
        and fixture_result["counters"]["real_model24_invocations"] == 0
        and fixture_result["counters"]["real_rtl_simulator_invocations"] == 0,
        "standalone controller-entry fixture result differs",
    )
    cases.extend(fixture_result["negative_cases"])
'''
    require(legacy_probe in payload, "r20 legacy controller probe seam missing")
    payload = payload.replace(legacy_probe, fixture_probe, 1)

    masked_result = '''        "simulator_invocations": (0 if os.environ.get("ACE3_R20_PREPARATION_ONLY") == "1" else 1),
        "canonical_artifact_creations": 0,
'''
    inert_result = '''        "simulator_invocations": 0,
        "fixture_controller_entry_invocations": fixture_result["counters"]["fixture_entry_calls"],
        "fixture_artifact_authentications": fixture_result["counters"]["fixture_artifact_authentications"],
        "real_controller_invocations": fixture_result["counters"]["real_controller_invocations"],
        "real_model24_invocations": fixture_result["counters"]["real_model24_invocations"],
        "real_rtl_simulator_invocations": fixture_result["counters"]["real_rtl_simulator_invocations"],
        "canonical_artifact_creations": 0,
'''
    require(masked_result in payload, "r20 masked negative result seam missing")
    payload = payload.replace(masked_result, inert_result, 1)

    integration_start = payload.index(
        "\ndef run_controller_entry_integration(package: Path)"
    )
    integration_end = payload.index("\n\ndef prepare(nonce: str)", integration_start)
    payload = (
        payload[:integration_start]
        + "\n\ndef run_controller_entry_integration(package: Path) -> dict[str, Any]:\n"
        + "    del package\n"
        + "    raise SystemExit(\n"
        + '        "production controller-entry integration is forbidden in r20 preparation"\n'
        + "    )\n"
        + payload[integration_end:]
    )

    marker = '''    cases.append("post-seal-source-mutation")

    require(
        cases == list(validator.REQUIRED_NEGATIVE_CASES),
'''
    replacement = '''    cases.append("post-seal-source-mutation")

    with tempfile.TemporaryDirectory(
        prefix="ace3-model24-r20-review-mutations-"
    ) as mutation_temporary:
        mutation_scratch = Path(mutation_temporary)
        for name in (
            "missing-reboot-crash-case",
            "substituted-reboot-crash-case",
        ):
            path = write_review(
                validator, package, manifest, mutation_scratch / name
            )
            review = validator.load_json(path)
            if name == "missing-reboot-crash-case":
                review["test_results"]["reboot_crash_cases"].pop()
            else:
                review["test_results"]["reboot_crash_cases"][-1]["name"] = (
                    "substituted-reboot-crash-case"
                )
            path.chmod(0o600)
            path.write_bytes(canonical_json(review))
            path.chmod(0o400)
            _expect_rejection(
                lambda path=path: validator.validate_review_document(
                    package, manifest, path
                ),
                "reboot crash case names differ from required set",
                name,
            )
            cases.append(name)

    require(
        cases == list(validator.REQUIRED_NEGATIVE_CASES),
'''
    require(marker in payload, "r20 negative-suite insertion seam missing")
    return payload.replace(marker, replacement, 1)


def update_package_documents(
    package: Path,
    nonce: str,
    ancestry: dict[str, Any],
    validator: Any,
) -> dict[str, Any]:
    lifecycle_overlay = load_json(package / "lifecycle-overlay.json")
    lifecycle_overlay["materialized_sha256"] = digest(
        package / "lifecycle.py"
    )
    write(
        package / "lifecycle-overlay.json",
        canonical_json(lifecycle_overlay),
    )

    fixture_contract = {
        "schema_version": 1,
        "kind": "ace3_model24_r20_inert_controller_entry_fixture",
        "fixture_only": True,
        "source_sha256": digest(package / "test-controller-entry.py"),
        "assertions": [
            "exclusive-path-creation",
            "synthetic-artifact-authentication",
            "directory-before-fixture-entry-order",
        ],
        "expected_counters": {
            "fixture_artifact_authentications": 3,
            "fixture_artifact_writes": 3,
            "fixture_entry_calls": 1,
            "real_controller_invocations": 0,
            "real_model24_invocations": 0,
            "real_rtl_simulator_invocations": 0,
        },
    }
    write(
        package / "controller-entry-fixture.json",
        canonical_json(fixture_contract),
    )

    contract = load_json(package / "launch-contract.json")
    contract["review"]["required_negative_cases"] = list(
        validator.REQUIRED_NEGATIVE_CASES
    )
    contract["review"]["required_positive_fixtures"] = list(
        validator.REQUIRED_POSITIVE_FIXTURES
    )
    contract["review"]["required_bound_hashes"] = list(
        validator.REQUIRED_REVIEW_HASHES
    )
    contract["review"]["required_reboot_crash_cases"] = list(
        validator.REBOOT_CRASH_CASES
    )
    contract["ancestry"]["r19_rejected"] = {
        "nonce": R19_NONCE,
        "task_id": R19_TASK_ID,
        "review_sha256": R19_REVIEW_SHA256,
        "verdict": "REJECT",
        "reusable": False,
        "classification_sha256": digest(
            package / "provenance/r19-rejected/classification.json"
        ),
    }
    contract["launch"]["controller_entry_preparation_probe"] = {
        "fixture_only": True,
        "real_controller_invocations": 0,
        "real_model24_invocations": 0,
        "real_rtl_simulator_invocations": 0,
    }
    write(package / "launch-contract.json", canonical_json(contract))

    manifest = load_json(package / "package.json")
    manifest["review_policy"]["preparer_identity"] = (
        f"role:engineer/session:r20-{nonce}"
    )
    manifest["provenance"]["r19_rejected_ancestry"] = {
        "classification_sha256": digest(
            package / "provenance/r19-rejected/classification.json"
        ),
        "review_sha256": R19_REVIEW_SHA256,
        "verdict": "REJECT",
        "copied_byte_for_byte": True,
        "reusable": False,
    }
    manifest["controller_entry_fixture"] = {
        "contract_sha256": digest(package / "controller-entry-fixture.json"),
        "fixture_only": True,
        "artifact_authentications": 3,
        "real_controller_invocations": 0,
        "real_model24_invocations": 0,
        "real_rtl_simulator_invocations": 0,
    }
    manifest["production_preflight"][
        "readonly_build_output_probe_sha256"
    ] = digest(
        package / "evidence/readonly-build-output-probe/result.json"
    )
    manifest["zero_state"].update(
        {
            "real_controller_invocations": 0,
            "real_model24_invocations": 0,
            "real_rtl_simulator_invocations": 0,
        }
    )
    write(package / "package.json", canonical_json(manifest))

    authority_schema = load_json(package / "authority-schema.json")
    fields = authority_schema["required_exact_fields"]
    for field in validator.ADDITIONAL_REVIEW_HASHES:
        if field not in fields:
            fields.insert(fields.index("submitter_cwd"), field)
    write(package / "authority-schema.json", canonical_json(authority_schema))

    review_request = load_json(package / "review-request.json")
    review_request["required_bound_hashes"] = list(
        validator.REQUIRED_REVIEW_HASHES
    )
    review_request["required_test_results"] = {
        "negative_cases": list(validator.REQUIRED_NEGATIVE_CASES),
        "positive_fixtures": list(validator.REQUIRED_POSITIVE_FIXTURES),
        "reboot_crash_cases": list(validator.REBOOT_CRASH_CASES),
        "fixture_controller_entry_invocations": 1,
        "fixture_artifact_authentications": 3,
        "real_controller_invocations": 0,
        "real_model24_invocations": 0,
        "real_rtl_simulator_invocations": 0,
    }
    review_request.update(
        {
            "package_manifest_sha256": digest(package / "package.json"),
            "validator_sha256": digest(package / "validate-package.py"),
            "r19_validator_sha256": digest(
                package / "validate-r19-package.py"
            ),
            "base_validator_sha256": digest(
                package / "validate-base-package.py"
            ),
            "r16_validator_sha256": digest(
                package / "validate-r16-package.py"
            ),
            "r17_validator_sha256": digest(
                package / "validate-r17-package.py"
            ),
            "r18_validator_sha256": digest(
                package / "validate-r18-package.py"
            ),
            "readonly_build_output_probe_sha256": digest(
                package / "evidence/readonly-build-output-probe/result.json"
            ),
            "lifecycle_sha256": digest(package / "lifecycle.py"),
            "lifecycle_overlay_sha256": digest(
                package / "lifecycle-overlay.json"
            ),
            "launch_contract_sha256": digest(
                package / "launch-contract.json"
            ),
            "source_tree_sha256": digest(package / "source-tree.json"),
            "authority_schema_sha256": digest(
                package / "authority-schema.json"
            ),
            "r19_rejected_ancestry_sha256": digest(
                package / "provenance/r19-rejected/classification.json"
            ),
            "controller_entry_fixture_sha256": digest(
                package / "controller-entry-fixture.json"
            ),
            "r19_reject_review_sha256": R19_REVIEW_SHA256,
            "r19_reject_bound": True,
            "canonical_execution_performed": False,
        }
    )
    write(package / "review-request.json", canonical_json(review_request))
    return manifest


def seal_package(
    package: Path,
    inherited_seal: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    readonly_tree(package)
    package.chmod(0o700)
    records = tree_records(package)
    seal = json.loads(
        transform_r20_text(json.dumps(inherited_seal), manifest["identity"]["nonce"])
    )
    seal.update(
        {
            "schema_version": 1,
            "kind": "ace3_model24_r20_package_seal",
            "nonce": manifest["identity"]["nonce"],
            "task_id": manifest["identity"]["task_id"],
            "package_manifest_sha256": digest(package / "package.json"),
            "review_request_sha256": digest(package / "review-request.json"),
            "source_tree_manifest_sha256": digest(
                package / "source-tree.json"
            ),
            "source_overlays_sha256": digest(
                package / "source-overlays.json"
            ),
            "controller_overlay_sha256": digest(
                package / "controller-overlay.json"
            ),
            "bindings_sha256": digest(package / "bindings.json"),
            "readonly_build_output_probe_sha256": digest(
                package / "evidence/readonly-build-output-probe/result.json"
            ),
            "lifecycle_overlay_sha256": digest(
                package / "lifecycle-overlay.json"
            ),
            "r18_crash_ancestry_sha256": digest(
                package / "provenance/r18-crashed/classification.json"
            ),
            "r19_rejected_ancestry_sha256": digest(
                package / "provenance/r19-rejected/classification.json"
            ),
            "controller_entry_fixture_sha256": digest(
                package / "controller-entry-fixture.json"
            ),
            "package_entries": records,
            "zero_state": True,
        }
    )
    write(package / "seal.json", canonical_json(seal))
    package.chmod(0o500)


def capture_command(
    validation: Path,
    name: str,
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        check=False,
    )
    write(validation / f"{name}.command.json", canonical_json(command))
    write(validation / f"{name}.stdout", completed.stdout)
    write(validation / f"{name}.stderr", completed.stderr)
    write(
        validation / f"{name}.status",
        f"{completed.returncode}\n".encode("ascii"),
    )
    require(
        completed.returncode == 0,
        f"{name} failed: {completed.stderr.decode(errors='replace')}",
    )
    return completed


def run_postseal_compile_probe(
    base: Path,
    validation: Path,
) -> dict[str, Any]:
    source = base / "source"
    output = base / "postseal-probe-output"
    require(not output.exists(), "post-seal output already exists")
    output.mkdir()
    before = tree_records(source)
    layers: dict[str, Any] = {}
    environment = dict(os.environ)
    for index in (0, 23):
        command = [
            "make",
            "--no-print-directory",
            "model24-rtl-layer-compile",
            f"MODEL24_RTL_LAYER_INDEX={index}",
            "MODEL24_RTL_ACCURATE_SILU=1",
            f"MODEL24_RTL_CASCADE_DIR={output}",
        ]
        started_ns = time.time_ns()
        completed = capture_command(
            validation,
            f"postseal-layer{index}",
            command,
            cwd=source,
            environment=environment,
        )
        binary = (
            output
            / f"compiled/layer{index}/obj_dir/Vace3_decoder_layer0_token_engine"
        )
        require(
            binary.is_file()
            and not binary.is_symlink()
            and binary.stat().st_mtime_ns >= started_ns,
            f"fresh layer {index} binary is absent or stale",
        )
        layers[str(index)] = {
            "argv": command,
            "exit_code": completed.returncode,
            "binary": {
                "path": str(binary),
                "bytes": binary.stat().st_size,
                "sha256": digest(binary),
            },
        }
    require(tree_records(source) == before, "post-seal compile changed source")
    require(not (source / "build").exists(), "post-seal compile wrote source/build")
    result = {
        "schema_version": 1,
        "kind": "ace3_model24_r20_postseal_readonly_compile_probe",
        "status": "PASS",
        "layers": layers,
        "source_unchanged": True,
        "source_build_absent": True,
        "real_controller_invocations": 0,
        "real_model24_invocations": 0,
        "real_rtl_simulator_invocations": 0,
        "rtl_binary_invocations": 0,
    }
    write(
        validation / "postseal-readonly-compile-probe.json",
        canonical_json(result),
    )
    readonly_tree(output)
    return result


def prepare(nonce: str) -> dict[str, Any]:
    require(
        len(nonce) == 16
        and all(character in "0123456789abcdef" for character in nonce),
        "nonce must be 16 lowercase hex characters",
    )
    require(nonce != R19_NONCE, "r20 nonce reuses rejected r19")
    require(
        REPOSITORY.resolve() == Path.cwd().resolve(),
        "run from the ACE-3 worktree",
    )
    require(R20_VALIDATOR.is_file(), "r20 validator is absent")
    require(
        R19_BASE.is_dir()
        and R19_REVIEW.is_dir()
        and digest(R19_REVIEW / "review.json") == R19_REVIEW_SHA256,
        "rejected r19 candidate or review binding differs",
    )
    rejected_review = load_json(R19_REVIEW / "review.json")
    require(
        rejected_review["nonce"] == R19_NONCE
        and rejected_review["task_id"] == R19_TASK_ID
        and rejected_review["verdict"] == "REJECT"
        and rejected_review["manager_may_issue_execution_authority"] is False,
        "r19 review is not the required REJECT",
    )

    base = Path(f"/home/argustest/ace3-model24-r20-prep-20260829-{nonce}")
    review = Path(f"/home/argustest/ace3-model24-r20-review-20260829-{nonce}")
    authority = Path(
        f"/home/argustest/ace3-model24-r20-authority-20260829-{nonce}.json"
    )
    output = Path(f"/home/argustest/ace3-model24-r20-output-20260829-{nonce}")
    terminal = Path(
        f"/home/argustest/ace3-model24-r20-terminal-20260829-{nonce}"
    )
    for path in (
        base,
        review,
        authority,
        Path(f"{authority}.consumed"),
        output,
        terminal,
    ):
        require(not path.exists(), f"fresh r20 namespace already exists: {path}")

    base.mkdir(mode=0o700)
    source = base / "source"
    package = base / "package"
    copy_tree_bytes(R19_BASE / "source", source)
    copy_tree_bytes(R19_BASE / "package", package)
    copy_tree_bytes(R19_BASE / "probe-output", base / "probe-output")
    writable_tree(package)
    inherited_seal = load_json(package / "seal.json")
    (package / "seal.json").unlink()

    rejected_root = package / "provenance/r19-rejected"
    candidate_records = copy_tree_bytes(R19_BASE, rejected_root / "candidate")
    review_records = copy_tree_bytes(R19_REVIEW, rejected_root / "review")
    classification = {
        "schema_version": 1,
        "kind": "ace3_model24_r19_rejected_immutable_ancestry",
        "nonce": R19_NONCE,
        "task_id": R19_TASK_ID,
        "review_sha256": R19_REVIEW_SHA256,
        "verdict": "REJECT",
        "decisive_findings": rejected_review["decisive_findings"],
        "copied_byte_for_byte": True,
        "reusable": False,
        "roots": {
            "candidate": {"records": candidate_records},
            "review": {"records": review_records},
        },
    }
    write(
        rejected_root / "classification.json",
        canonical_json(classification),
    )

    for path in package.iterdir():
        if path.is_file() and path.suffix in {".py", ".json"}:
            write(
                path,
                transform_r20_text(path.read_text(encoding="ascii"), nonce).encode(
                    "ascii"
                ),
                0o500 if path.suffix == ".py" else 0o400,
            )
    builder_path = package / "ancestry/repository-builder.py"
    builder_payload = transform_r20_text(
        builder_path.read_text(encoding="ascii"), nonce
    )
    write(
        builder_path,
        patch_negative_suite(builder_payload).encode("ascii"),
    )
    readonly_probe = (
        package / "evidence/readonly-build-output-probe/result.json"
    )
    write(
        readonly_probe,
        transform_r20_text(
            readonly_probe.read_text(encoding="ascii"), nonce
        ).encode("ascii"),
    )

    inherited_validator = package / "validate-package.py"
    write(
        package / "validate-r19-package.py",
        inherited_validator.read_bytes(),
        0o500,
    )
    write(package / "validate-package.py", R20_VALIDATOR.read_bytes(), 0o500)
    write(
        package / "test-controller-entry.py",
        inert_controller_fixture_source(),
        0o500,
    )
    validator = load_module(
        package / "validate-package.py",
        "model24_r20_contract_validator",
    )
    manifest = update_package_documents(
        package,
        nonce,
        classification,
        validator,
    )
    readonly_tree(source)
    readonly_tree(base / "probe-output")
    seal_package(package, inherited_seal, manifest)

    validation = base / "validation"
    preparation = base / "preparation"
    validation.mkdir()
    preparation.mkdir()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    commands = {
        "package-validation": [
            str(MODEL_PYTHON),
            "-B",
            str(package / "validate-package.py"),
            "--package",
            str(package),
            "--mode",
            "package",
        ],
        "negative-tests": [
            str(MODEL_PYTHON),
            "-B",
            str(package / "test-launch-contract.py"),
        ],
        "controller-entry-fixture": [
            str(MODEL_PYTHON),
            "-B",
            str(package / "test-controller-entry.py"),
        ],
        "production-binding-preflight": [
            str(MODEL_PYTHON),
            "-B",
            str(package / "test-binding-preflight.py"),
        ],
        "readonly-build-output-probe": [
            str(MODEL_PYTHON),
            "-B",
            str(package / "test-readonly-build-output.py"),
        ],
        "reboot-crash-tests": [
            str(MODEL_PYTHON),
            "-B",
            str(package / "test-reboot-crash.py"),
        ],
    }
    outcomes: dict[str, dict[str, Any]] = {}
    outputs: dict[str, subprocess.CompletedProcess[bytes]] = {}
    for name, command in commands.items():
        completed = capture_command(
            validation,
            name,
            command,
            cwd=base,
            environment=environment,
        )
        outputs[name] = completed
        outcomes[name] = {
            "argv": command,
            "exit_code": completed.returncode,
            "stdout_sha256": digest(validation / f"{name}.stdout"),
            "stderr_sha256": digest(validation / f"{name}.stderr"),
        }

    negative_result = json.loads(outputs["negative-tests"].stdout)
    require(
        negative_result["case_count"] == 61
        and negative_result["cases"]
        == list(validator.REQUIRED_NEGATIVE_CASES),
        "r20 negative suite did not retain 59 and append two",
    )
    fixture_result = json.loads(outputs["controller-entry-fixture"].stdout)
    require(
        fixture_result["status"] == "PASS"
        and fixture_result["fixture_only"] is True
        and fixture_result["counters"]
        == load_json(package / "controller-entry-fixture.json")[
            "expected_counters"
        ],
        "controller-entry fixture result differs",
    )
    reboot_result = json.loads(outputs["reboot-crash-tests"].stdout)
    require(
        reboot_result["cases"] == list(validator.REBOOT_CRASH_CASES),
        "13-case reboot crash suite differs",
    )
    compile_probe = run_postseal_compile_probe(base, validation)

    paths = validator.expected_paths(package, manifest)
    forbidden = (
        "review",
        "authority",
        "authority_consumed",
        "output",
        "simulation_dir",
        "payload_output_dir",
        "terminal_root",
        "receipt",
        "stdout_log",
        "stderr_log",
        "terminal_manifest",
    )
    zero_state = {
        f"{name}_absent": not paths[name].exists() for name in forbidden
    }
    require(all(zero_state.values()), "canonical r20 execution state is not zero")
    zero_state.update(
        {
            "fixture_controller_entry_invocations": 1,
            "fixture_artifact_authentications": 3,
            "real_controller_invocations": 0,
            "real_model24_invocations": 0,
            "real_rtl_simulator_invocations": 0,
            "rtl_binary_invocations": 0,
            "lifecycle_launch_invocations": 0,
            "durable_submissions": 0,
            "authority_invocations": 0,
            "output_creations": 0,
        }
    )
    write(validation / "zero-state.json", canonical_json(zero_state))
    readonly_tree(validation)
    validation.chmod(0o700)
    validation_records = tree_records(validation)
    validation_seal = {
        "schema_version": 1,
        "kind": "ace3_model24_r20_inert_validation_seal",
        "package_seal_sha256": digest(package / "seal.json"),
        "negative_case_count": len(validator.REQUIRED_NEGATIVE_CASES),
        "inherited_negative_case_count": len(
            validator.INHERITED_NEGATIVE_CASES
        ),
        "positive_fixture_count": len(
            validator.REQUIRED_POSITIVE_FIXTURES
        ),
        "reboot_crash_case_count": len(validator.REBOOT_CRASH_CASES),
        "outcomes": outcomes,
        "postseal_compile_probe": compile_probe,
        "records": validation_records,
        "fixture_controller_entry_invocations": 1,
        "fixture_artifact_authentications": 3,
        "real_controller_invocations": 0,
        "real_model24_invocations": 0,
        "real_rtl_simulator_invocations": 0,
        "rtl_binary_invocations": 0,
        "lifecycle_launch_invocations": 0,
        "durable_submissions": 0,
        "authority_invocations": 0,
    }
    write(validation / "seal.json", canonical_json(validation_seal))
    validation.chmod(0o500)

    result = {
        "status": "SEALED_AWAITING_INDEPENDENT_L2_REVIEW",
        "candidate": str(base),
        "package": str(package),
        "nonce": manifest["identity"]["nonce"],
        "task_id": manifest["identity"]["task_id"],
        "package_manifest_sha256": digest(package / "package.json"),
        "package_seal_sha256": digest(package / "seal.json"),
        "review_request_sha256": digest(package / "review-request.json"),
        "validation_seal_sha256": digest(validation / "seal.json"),
        "r19_reject_review_sha256": R19_REVIEW_SHA256,
        "r19_rejected_ancestry_bound": True,
        "negative_case_count": len(validator.REQUIRED_NEGATIVE_CASES),
        "inherited_negative_case_count": len(
            validator.INHERITED_NEGATIVE_CASES
        ),
        "positive_fixture_count": len(
            validator.REQUIRED_POSITIVE_FIXTURES
        ),
        "reboot_crash_case_count": len(validator.REBOOT_CRASH_CASES),
        "r18_authenticated_complete_layers": list(range(9)),
        "r18_incomplete_layer": 9,
        "r18_execution_output_reusable": False,
        "postseal_readonly_compile_layers": [0, 23],
        "canonical_zero_state": zero_state,
        "execution_authority_withheld": True,
        "review_obtained": False,
    }
    write(preparation / "result.json", canonical_json(result))
    readonly_tree(preparation)
    base.chmod(0o500)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", default=secrets.token_hex(8))
    args = parser.parse_args()
    print(canonical_json(prepare(args.nonce)).decode("ascii"), end="")


if __name__ == "__main__":
    main()
