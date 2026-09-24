#!/usr/bin/env python3
"""Run one source-bound v10 position-2 traversal through the canonical validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/argustest/ace3-argus")
PACKAGE_ID = "ace3-position2-fresh-v7-20260831t084959z"
PACKAGE_ROOT = (
    ROOT / "build/model24_selected_token_position2_packages" / PACKAGE_ID
)
PACKAGE_MANIFEST = PACKAGE_ROOT / "package-manifest.json"
PACKAGE_SEAL = PACKAGE_ROOT / "package-seal.json"
PACKAGE_REVIEW = (
    ROOT
    / "build/model24_selected_token_position2_package_reviews"
    / f"{PACKAGE_ID}.json"
)
SOURCE_REVIEW = (
    ROOT
    / "build/model24_selected_token_position2_source_reviews"
    / "ace3-position2-v7-runtime-repair-source-review-20260831t084959z.json"
)
SOURCE_ARCHIVE = (
    ROOT
    / "build/model24_selected_token_position2_source_archives"
    / f"{PACKAGE_ID}.tar"
)
CHECKPOINT = (
    ROOT / "build/model24_rtl_cascade/checkpoint/model.safetensors"
)
RUN_PARENT = ROOT / "build/model24_selected_token_position2_runs"
CANONICAL_EVIDENCE = (
    ROOT / "build/model24_selected_token_position2/evidence.json"
)
VALIDATOR_RELATIVE = Path(
    "ace3/model/validate_selected_token_position2_traversal.py"
)
RUN_ID_PATTERN = re.compile(
    r"ace3-position2-fresh-v10-[a-z0-9][a-z0-9-]{7,63}"
)
EXPECTED_HASHES = {
    PACKAGE_MANIFEST: (
        8244,
        "e7e79d2dc2c5654a4b10039f2ddd665e1745bffb5d1e4c9de44ac5d6032c5bfe",
    ),
    PACKAGE_SEAL: (
        1530,
        "7e36b2fb4ef49915e1df0d82b89270802e2e9eae5d7119dc77be998082e03d58",
    ),
    PACKAGE_REVIEW: (
        5142,
        "66e68f8e96ad5da38b8a66985130b83c9693d78bef5955c03bc16c410a0a9ba9",
    ),
    SOURCE_REVIEW: (
        9831,
        "1bd390f3190faea257526797682df504fe23a3889e8dc7986315f2d23334aa02",
    ),
    SOURCE_ARCHIVE: (
        768000,
        "b0aeb44ef1cb33d250864016c88297297cf0630187440d617cc1afee554f79f0",
    ),
    CHECKPOINT: (
        730652248,
        "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b",
    ),
}


class RuntimeErrorV10(RuntimeError):
    """Raised when a v10 runtime gate fails."""


class TerminalSignal(RuntimeError):
    """Raised when the durable child receives a terminal signal."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = signum


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeErrorV10(message)


def now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def canonical_json(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(8 * 1024 * 1024):
            digest.update(payload)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(document, dict), f"{path} must contain a JSON object")
    return document


def file_record(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_expected_file(path: Path) -> dict[str, Any]:
    expected_bytes, expected_hash = EXPECTED_HASHES[path]
    record = file_record(path)
    require(
        record["bytes"] == expected_bytes
        and record["sha256"] == expected_hash,
        f"accepted input binding changed: {path}",
    )
    return record


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive(path: Path, document: object) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(canonical_json(document))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def runtime_paths(run_id: str) -> dict[str, Path]:
    require(
        RUN_ID_PATTERN.fullmatch(run_id) is not None,
        "run ID is outside the fresh v10 namespace",
    )
    root = RUN_PARENT / run_id
    return {
        "root": root,
        "source": root / "source",
        "logs": root / "logs",
        "evidence": root / "evidence.json",
        "summary": root / "v10-runtime-submission.json",
        "terminal": root / "terminal.json",
        "runner_receipt": ROOT / ".argus_subagents" / f"{run_id}.json",
        "runner_logs": ROOT / ".argus_subagents" / f"{run_id}_logs",
    }


def tool_record(name: str) -> dict[str, Any]:
    selected = shutil.which(name)
    require(selected is not None, f"required tool is unavailable: {name}")
    return file_record(Path(selected).resolve(strict=True))


def verify_static_inputs() -> dict[str, Any]:
    require(Path.cwd().resolve() == ROOT, "runtime cwd is not the project root")
    package_manifest = load_json(PACKAGE_MANIFEST)
    package_review = load_json(PACKAGE_REVIEW)
    source_review = load_json(SOURCE_REVIEW)
    require(
        package_manifest.get("package_id") == PACKAGE_ID
        and package_manifest.get("status")
        == "SEALED_AWAITING_INDEPENDENT_PACKAGE_REVIEW",
        "accepted v7 package identity mismatch",
    )
    require(
        package_review.get("package_id") == PACKAGE_ID
        and package_review.get("verdict") == "PASS",
        "accepted v7 package review mismatch",
    )
    require(
        source_review.get("kind")
        == "ace3_position2_fresh_post_repair_source_review"
        and source_review.get("verdict") == "PASS",
        "accepted v7 source review mismatch",
    )
    accepted = source_review.get("accepted_source_set")
    files = accepted.get("files") if isinstance(accepted, dict) else None
    require(
        isinstance(files, list)
        and len(files) == 30
        and all(
            isinstance(item, dict)
            and set(item) == {"bytes", "path", "sha256"}
            for item in files
        ),
        "accepted v7 source set is malformed",
    )
    records = {
        path.name: verify_expected_file(path)
        for path in EXPECTED_HASHES
    }
    return {
        "accepted_package_id": PACKAGE_ID,
        "accepted_package_manifest": records[PACKAGE_MANIFEST.name],
        "accepted_package_seal": records[PACKAGE_SEAL.name],
        "accepted_package_review": records[PACKAGE_REVIEW.name],
        "accepted_source_review": records[SOURCE_REVIEW.name],
        "accepted_source_archive": records[SOURCE_ARCHIVE.name],
        "checkpoint": records[CHECKPOINT.name],
        "accepted_source_set": {
            "file_count": len(files),
            "canonical_sha256": accepted["canonical_digest"]["sha256"],
            "files": files,
        },
        "interpreter": {
            **file_record(Path(sys.executable).resolve(strict=True)),
            "version": sys.version,
        },
        "toolchain": {
            name: tool_record(name)
            for name in ("g++", "make", "verilator")
        },
    }


def safe_extract_verified(source_root: Path, bindings: Mapping[str, Any]) -> None:
    expected = {
        item["path"]: item
        for item in bindings["accepted_source_set"]["files"]
    }
    with tarfile.open(SOURCE_ARCHIVE, "r") as archive:
        members = archive.getmembers()
        require(
            sorted(member.name for member in members) == sorted(expected),
            "source archive member set differs from the accepted source set",
        )
        for member in members:
            require(
                member.isfile() and not member.issym() and not member.islnk(),
                f"source archive member is not regular: {member.name}",
            )
            target = (source_root / member.name).resolve()
            require(
                target.is_relative_to(source_root),
                f"source archive member escapes extraction root: {member.name}",
            )
        archive.extractall(source_root, filter="data")
    for relative, accepted in expected.items():
        extracted = file_record(source_root / relative)
        require(
            extracted["bytes"] == accepted["bytes"]
            and extracted["sha256"] == accepted["sha256"],
            f"extracted accepted source changed: {relative}",
        )


def verify_preflight(run_id: str) -> dict[str, Any]:
    paths = runtime_paths(run_id)
    for label in ("root", "runner_receipt", "runner_logs"):
        require(
            not os.path.lexists(paths[label]),
            f"fresh v10 {label} namespace already exists: {paths[label]}",
        )
    require(
        not os.path.lexists(CANONICAL_EVIDENCE),
        "canonical position-2 evidence already exists",
    )
    bindings = verify_static_inputs()
    return {
        "schema_version": 1,
        "kind": "ace3_position2_v10_canonical_runtime_preflight",
        "run_id": run_id,
        "status": "PASS_READY_FOR_ONE_DURABLE_SUBMISSION",
        "canonical_validator": (
            bindings["accepted_source_set"]["files"][
                [
                    item["path"]
                    for item in bindings["accepted_source_set"]["files"]
                ].index(VALIDATOR_RELATIVE.as_posix())
            ]
        ),
        "accepted_source_archive": bindings["accepted_source_archive"],
        "checkpoint": bindings["checkpoint"],
        "runtime_action_counts": {
            "durable_runner_submissions": 0,
            "canonical_validator_invocations": 0,
        },
    }


def run_step(
    name: str,
    argv: list[str],
    cwd: Path,
    logs: Path,
    counters: dict[str, int],
) -> int:
    write_json_exclusive(
        logs / f"{name}-invocation.json",
        {
            "schema_version": 1,
            "kind": "ace3_position2_v10_canonical_validator_invocation",
            "name": name,
            "argv": argv,
            "cwd": str(cwd),
            "started_at_utc": now(),
        },
    )
    counters[f"{name}_invocations"] += 1
    counters["canonical_validator_invocations"] += 1
    with (logs / f"{name}.stdout.log").open("xb") as stdout:
        with (logs / f"{name}.stderr.log").open("xb") as stderr:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
    write_json_exclusive(
        logs / f"{name}-terminal.json",
        {
            "schema_version": 1,
            "kind": "ace3_position2_v10_canonical_validator_terminal",
            "name": name,
            "exit_code": completed.returncode,
            "ended_at_utc": now(),
        },
    )
    return completed.returncode


def accepted_traversal(document: Mapping[str, Any]) -> dict[str, Any]:
    require(
        document.get("schema_version") == 3
        and document.get("kind")
        == "ace3_selected_token_position2_fresh_traversal_evidence"
        and document.get("status") == "COMPLETE",
        "canonical validator evidence identity mismatch",
    )
    traversal = document.get("current_continuation_attempt")
    require(isinstance(traversal, dict), "canonical traversal record is missing")
    layers = traversal.get("layers")
    require(
        traversal.get("position") == 2
        and traversal.get("selected_token_id") == 271
        and traversal.get("layer_order") == list(range(24))
        and traversal.get("natural_terminal_layers") == 24
        and isinstance(layers, list)
        and len(layers) == 24,
        "canonical position-2 24-layer traversal was not accepted",
    )
    for index, layer in enumerate(layers):
        require(
            layer.get("layer_index") == index
            and layer.get("position") == 2
            and layer.get("independent_oracle_comparison", {}).get(
                "rtl_matches_exact_integer_oracle"
            )
            is True
            and layer.get("independent_oracle_comparison", {}).get(
                "within_tolerance"
            )
            is True,
            f"canonical layer {index} acceptance record is incomplete",
        )
    return {
        "status": "CANONICAL_VALIDATOR_PASS_AWAITING_INDEPENDENT_REVIEW",
        "selected_token_id": 271,
        "position": 2,
        "layer_count": len(layers),
        "layer_order": "0..23",
        "natural_terminal_layers": traversal["natural_terminal_layers"],
        "post_layer23": traversal["post_layer23"],
    }


def install_signal_handlers() -> None:
    def handle(signum: int, _frame: Any) -> None:
        raise TerminalSignal(signum)

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)


def execute(run_id: str) -> int:
    paths = runtime_paths(run_id)
    require(
        not os.path.lexists(paths["root"]),
        "fresh v10 runtime namespace already exists",
    )
    require(
        paths["runner_receipt"].is_file()
        and not paths["runner_receipt"].is_symlink(),
        "active durable-runner receipt is missing",
    )
    RUN_PARENT.mkdir(parents=True, exist_ok=True)
    paths["root"].mkdir(mode=0o700)
    paths["logs"].mkdir(mode=0o700)
    install_signal_handlers()
    counters = {
        "driver_invocations": 1,
        "durable_runner_submissions": 1,
        "source_archive_extractions": 0,
        "generate_invocations": 0,
        "validate_invocations": 0,
        "canonical_validator_invocations": 0,
        "accepted_24_layer_traversals": 0,
        "terminal_manifests": 0,
        "retries": 0,
        "replays": 0,
        "resumes": 0,
        "relaunches": 0,
    }
    status = "RUNTIME_FAIL_NON_RETRYABLE"
    detail = "v10 canonical runtime did not reach a classified terminal"
    exit_code = 2
    try:
        bindings = verify_static_inputs()
        require(
            not os.path.lexists(CANONICAL_EVIDENCE),
            "canonical evidence appeared before v10 runtime review",
        )
        paths["source"].mkdir(mode=0o700)
        safe_extract_verified(paths["source"], bindings)
        counters["source_archive_extractions"] = 1
        checkpoint_link = (
            paths["source"]
            / "build/model24_rtl_cascade/checkpoint/model.safetensors"
        )
        checkpoint_link.parent.mkdir(parents=True)
        checkpoint_link.symlink_to(CHECKPOINT)
        validator = paths["source"] / VALIDATOR_RELATIVE
        interpreter = bindings["interpreter"]["path"]
        generate_argv = [
            interpreter,
            "-B",
            str(validator),
            "generate",
            "--output",
            str(paths["evidence"]),
        ]
        validate_argv = [
            interpreter,
            "-B",
            str(validator),
            "validate",
            "--output",
            str(paths["evidence"]),
        ]
        generate_exit = run_step(
            "generate",
            generate_argv,
            paths["source"],
            paths["logs"],
            counters,
        )
        validate_exit = (
            run_step(
                "validate",
                validate_argv,
                paths["source"],
                paths["logs"],
                counters,
            )
            if generate_exit == 0
            else None
        )
        require(
            generate_exit == 0 and validate_exit == 0,
            "canonical validator generate/validate did not both pass",
        )
        evidence = load_json(paths["evidence"])
        traversal = accepted_traversal(evidence)
        counters["accepted_24_layer_traversals"] = 1
        source_records = evidence.get("consumed_sources")
        require(
            isinstance(source_records, dict)
            and len(source_records) == 30,
            "canonical evidence consumed-source closure is incomplete",
        )
        summary = {
            "schema_version": 1,
            "kind": "ace3_position2_v10_canonical_runtime_submission",
            "run_id": run_id,
            "status": "RUNTIME_PASS_AWAITING_INDEPENDENT_REVIEW",
            "durable_runner": {
                "receipt": file_record(paths["runner_receipt"]),
                "submission_count": 1,
            },
            "runtime_harness": file_record(Path(__file__).resolve()),
            "accepted_source_path": {
                "package_id": PACKAGE_ID,
                "package_manifest": bindings["accepted_package_manifest"],
                "package_seal": bindings["accepted_package_seal"],
                "package_review": bindings["accepted_package_review"],
                "source_review": bindings["accepted_source_review"],
                "source_archive": bindings["accepted_source_archive"],
                "source_set": bindings["accepted_source_set"],
                "extracted_validator": file_record(validator),
                "canonical_evidence_consumed_sources": source_records,
            },
            "fixture_checkpoint_bindings": {
                "checkpoint": bindings["checkpoint"],
                "official_tied_embedding": evidence[
                    "official_tied_embedding"
                ],
                "fresh_token_inputs": evidence[
                    "current_continuation_attempt"
                ]["fresh_token_inputs"],
                "runtime_vector_and_tensor_hashes": (
                    "complete per-layer records are sealed in validator evidence"
                ),
            },
            "canonical_validator": {
                "generate_argv": generate_argv,
                "validate_argv": validate_argv,
                "generate_exit_code": generate_exit,
                "validate_exit_code": validate_exit,
                "invocation_counts": {
                    "generate": counters["generate_invocations"],
                    "validate": counters["validate_invocations"],
                    "total": counters[
                        "canonical_validator_invocations"
                    ],
                },
                "evidence": file_record(paths["evidence"]),
                "verdict": "PASS",
            },
            "accepted_traversal": traversal,
            "v9_boundary": {
                "identity": "ace3-position2-fresh-v9-20260831t100202z",
                "result": "REHEARSAL_ONLY",
                "driver_invocations": 1,
                "verilator_transactions": 1,
                "canonical_validator_invocations": 0,
                "canonical_runtime_pass": False,
                "distinction": (
                    "v10 invokes the extracted accepted canonical validator "
                    "for generate and validate across all 24 causal layers"
                ),
            },
            "claim_boundary": {
                "position2_runtime": (
                    "canonical validator PASS; independent acceptance pending"
                ),
                "canonical_publication": (
                    "withheld; canonical evidence path remains absent"
                ),
                "position3": "blocked until this v10 PASS is accepted",
                "lm_head": "blocked until this v10 PASS is accepted",
                "dialogue": "blocked until this v10 PASS is accepted",
                "synthesis_ppa_fpga": "not run or measured",
                "latency_throughput": "not measured",
            },
            "counters": counters,
            "completed_at_utc": now(),
        }
        write_json_exclusive(paths["summary"], summary)
        status = "RUNTIME_PASS_AWAITING_INDEPENDENT_REVIEW"
        detail = (
            "one canonical-validator position-2 24-layer traversal passed"
        )
        exit_code = 0
        return 0
    except (
        RuntimeErrorV10,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        tarfile.TarError,
    ) as error:
        detail = str(error)
        return 2
    except TerminalSignal as error:
        detail = str(error)
        exit_code = 128 + error.signum
        return exit_code
    finally:
        if not paths["terminal"].exists():
            counters["terminal_manifests"] = 1
            write_json_exclusive(
                paths["terminal"],
                {
                    "schema_version": 1,
                    "kind": "ace3_position2_v10_canonical_runtime_terminal",
                    "run_id": run_id,
                    "status": status,
                    "detail": detail,
                    "exit_code": exit_code,
                    "retryable": False,
                    "counters": counters,
                    "ended_at_utc": now(),
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("preflight", "execute"))
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    try:
        if arguments.operation == "preflight":
            print(json.dumps(verify_preflight(arguments.run_id), sort_keys=True))
        else:
            raise SystemExit(execute(arguments.run_id))
    except (RuntimeErrorV10, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"POSITION2_V10_CANONICAL_RUNTIME_FAIL: {error}") from error


if __name__ == "__main__":
    main()
