#!/usr/bin/env python3
"""Build the zero-workload r20/V11 runtime-pass admission candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if __package__:
    from . import validate_model24_r20_v11_runtime_pass_admission as contract
else:
    import validate_model24_r20_v11_runtime_pass_admission as contract


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if os.path.lexists(path):
            path.chmod(0o600)
            path.unlink()
        raise


def build_candidate(output: Path) -> dict[str, object]:
    output = output.resolve()
    contract.require(
        not os.path.lexists(output),
        f"runtime-pass admission output already exists: {output}",
    )
    contract.require(
        output.parent.is_dir() and not output.parent.is_symlink(),
        f"runtime-pass admission parent directory required: {output.parent}",
    )
    payloads = contract.checked_source_payloads()
    contract.validate_reviewed_source_commit()

    external_records = {
        label: contract.source_record(path, payloads[label])
        for label, path in contract.EXTERNAL_SOURCES.items()
    }
    project = contract.load_object(
        contract.PROJECT_AUTHORITY,
        payloads["adopted_project_authority_preimage"],
    )
    manifest = contract.load_object(
        contract.R20_PACKAGE_MANIFEST,
        payloads["package_manifest"],
    )

    output.mkdir(mode=0o700)
    snapshots = output / "snapshots"
    snapshots.mkdir(mode=0o700)
    snapshot_records: dict[str, dict[str, object]] = {}
    for label, relative_path in contract.SNAPSHOT_PATHS.items():
        payload = payloads[label]
        path = output / relative_path
        write_exclusive(path, payload)
        snapshot_records[label] = contract.snapshot_record(
            output,
            relative_path,
            contract.SNAPSHOT_SOURCES[label],
            payload,
        )
    admission = {
        "accepted_r20_bindings": external_records,
        "activity_counters": project["activity_counters"],
        "authorization_state": {
            "admission_accepted": False,
            "authority_consumed": False,
            "authority_consumption_authorized": False,
            "durable_submission_authorized": False,
            "execution_authorized": False,
            "output_or_terminal_publication_authorized": False,
            "project_authority_updated": False,
            "workload_authorized": False,
        },
        "claim_boundary": (
            "This candidate prepares runtime-pass admission only. Independent "
            "admission acceptance and a later atomic project-authority update "
            "are both required before consumption, submission, workload, "
            "output, or terminal publication."
        ),
        "kind": "ace3_model24_r20_v11_runtime_pass_admission_candidate",
        "launch": {
            "argv": contract.LAUNCH_ARGV,
            "authority": str(contract.R20_AUTHORITY),
            "cwd": contract.LAUNCH_CWD,
            "review": str(contract.R20_REVIEW),
        },
        "manager_revision": "V11",
        "nonce": contract.R20_NONCE,
        "one_shot_controls": {
            "execution_cardinality": 1,
            "replay": False,
            "resume": False,
            "retry": False,
            "watcher": False,
        },
        "package_zero_state": manifest["zero_state"],
        "receipt_chain_validation": {
            "accepted": True,
            "review_task_id": contract.RECEIPT_CHAIN_TASK_ID,
            "reviewed_source_commit": contract.REVIEWED_SOURCE_COMMIT,
            "reviewed_source_tree": contract.REVIEWED_SOURCE_TREE,
        },
        "schema_version": 1,
        "snapshots": snapshot_records,
        "status": "AWAITING_INDEPENDENT_REVIEW",
        "task_id": contract.R20_TASK_ID,
    }
    write_exclusive(output / "admission.json", contract.canonical_json(admission))
    fsync_directory(snapshots)
    fsync_directory(output)

    result = contract.validate_candidate(output)
    validation_lines = [
        "ACE3_R20_V11_RUNTIME_PASS_ADMISSION_CANDIDATE PASS",
        "admission_status=AWAITING_INDEPENDENT_REVIEW",
        "manager_v11_directive=exact_bytes",
        "preflight=exact",
        "receipt_chain_validation=accepted_and_source_bound",
        "launch_tuple=exact",
        "one_shot_controls=exact",
        "authority_consumed=false",
        "execution_authorized=false",
        "project_authority_updated=false",
        "workload_counters=all_zero",
        "runtime_namespaces=absent",
    ]
    write_exclusive(
        output / "validation.log",
        ("\n".join(validation_lines) + "\n").encode("ascii"),
    )
    fsync_directory(output)
    for path in snapshots.iterdir():
        path.chmod(0o400)
    snapshots.chmod(0o500)
    (output / "admission.json").chmod(0o400)
    (output / "validation.log").chmod(0o400)
    output.chmod(0o500)
    fsync_directory(output.parent)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    contract.require(
        output == contract.CANONICAL_OUTPUT,
        "noncanonical runtime-pass admission output path",
    )
    result = build_candidate(output)
    print(
        "ACE3_R20_V11_RUNTIME_PASS_ADMISSION_CANDIDATE_PASS "
        f"artifact={output / 'admission.json'} "
        f"status={result['admission_status']} "
        "authority_consumed=false execution_authorized=false "
        "workload_counters=all_zero namespaces=absent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
