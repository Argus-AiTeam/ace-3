#!/usr/bin/env python3
"""Rehearse and seal the position-2 durable-runner lifecycle boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Callable, Mapping


ROOT = Path("/home/argustest/ace3-argus")
V8_ID = "ace3-position2-fresh-v8-20260831t090000z"
ACCEPTED_PACKAGE_ID = "ace3-position2-fresh-v7-20260831t084959z"
LIFECYCLE_BINDINGS_KIND = "ace3_position2_successor_lifecycle_bindings"
AUTHORITY_KIND = "ace3_position2_successor_lifecycle_authority"
DISPOSABLE_AUTHORITY_MODE = "disposable_rehearsal"
FORMAL_AUTHORITY_MODE = "formal_one_shot"
V8_AUTHORITY = (
    ROOT
    / "build/model24_selected_token_position2_authorities"
    / f"{V8_ID}.json"
)
FORMAL_STATE_PARENTS = (
    ROOT / "build/model24_selected_token_position2_authorities",
    ROOT / "build/model24_selected_token_position2_authority_reviews",
    ROOT / "build/model24_selected_token_position2_authority_payloads",
    ROOT / "build/model24_selected_token_position2_authority_consumed",
    ROOT / "build/model24_selected_token_position2_submissions",
    ROOT / "build/model24_selected_token_position2_runs",
    ROOT / "build/model24_selected_token_position2_runtime_evidence",
    ROOT / "build/model24_selected_token_position2_execution_logs",
)
REHEARSAL_PARENT = ROOT / "ace3/build/position2_runtime_wrapper_lifecycle"
FORMAL_SUCCESSOR_PARENT = REHEARSAL_PARENT / "formal-successors"
V6_VECTOR_ROOT = (
    ROOT
    / "build/model24_selected_token_position2_runs"
    / "ace3-position2-fresh-v6-20260831t081500z"
    / "traversal/layer00/vectors"
)
V6_POSITION0_VECTOR_ROOT = (
    ROOT
    / "build/model24_selected_token_position2_runs"
    / "ace3-position2-fresh-v6-20260831t081500z"
    / "traversal/layer00/position000/vectors"
)
DECODER_RTL_NAMES = (
    "ace3_fp16_fixed.sv",
    "ace3_q47_48_to_f16_rne.sv",
    "ace3_awq_w4a16_g128_dot_lane.sv",
    "ace3_awq_w4a16_projection_engine.sv",
    "ace3_fp16_rmsnorm_core.sv",
    "ace3_fp16_residual_add_core.sv",
    "ace3_fp16_silu_gate_core.sv",
    "ace3_qwen2_rope_pair.sv",
    "ace3_fp16_kv_cache.sv",
    "ace3_attention_score_core.sv",
    "ace3_attention_softmax_core.sv",
    "ace3_attention_value_core.sv",
    "ace3_decoder_qzeros_address.sv",
    "ace3_decoder_layer0_token_engine.sv",
)
ROUTE_METADATA = {
    "node_key": "close-runtime-wrapper-lifecycle-gate",
    "plan_id": "plan-46010fa616da",
    "planner_scope": "bounded",
    "route_kind": "runtime_pass",
    "route_position": 2,
    "stage": "rtl",
    "vertical": "chip_design",
}
TERMINAL_STATES = frozenset(
    {"done", "error", "timeout", "crashed", "cancelled", "discussing"}
)


class LifecycleError(RuntimeError):
    """Raised when the wrapper lifecycle fails closed."""


class TerminalSignal(RuntimeError):
    """Classify a signal as a non-retryable child terminal."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = signum


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LifecycleError(message)


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


def compact_json(document: object) -> str:
    return json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LifecycleError(f"invalid JSON document: {path}: {error}") from error
    require(isinstance(document, dict), f"JSON object required: {path}")
    return document


def file_record(path: Path, *, executable: bool = False) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    resolved = path.resolve(strict=True)
    require(resolved == path, f"canonical non-symlink path required: {path}")
    if executable:
        require(os.access(path, os.X_OK), f"executable required: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_record(
    binding: Mapping[str, Any],
    path: Path,
    label: str,
    *,
    executable: bool = False,
) -> None:
    actual = file_record(path, executable=executable)
    require(
        all(binding.get(key) == value for key, value in actual.items()),
        f"{label} record mismatch",
    )
    if "mode" in binding:
        require(
            binding["mode"] == f"{stat.S_IMODE(path.stat().st_mode):04o}",
            f"{label} mode mismatch",
        )


def write_json_exclusive(path: Path, document: object, mode: int = 0o444) -> None:
    payload = canonical_json(document)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, mode)
    fsync_directory(path.parent)


def write_bytes_exclusive(path: Path, payload: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, mode)
    fsync_directory(path.parent)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def path_absent(path: Path) -> bool:
    return not os.path.lexists(path)


def require_absent(path: Path, label: str) -> None:
    require(path_absent(path), f"{label} already exists: {path}")


def pid_alive(value: object) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def resolve_tool(name: str) -> dict[str, Any]:
    discovered = shutil.which(name)
    require(discovered is not None, f"required tool is unavailable: {name}")
    path = Path(discovered).resolve(strict=True)
    return file_record(path, executable=True)


def verify_tool_record(binding: Mapping[str, Any], label: str) -> Path:
    path = Path(str(binding.get("path", "")))
    verify_record(binding, path, label, executable=True)
    return path


def formal_state_snapshot() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for parent in FORMAL_STATE_PARENTS:
        if not parent.is_dir():
            continue
        for path in sorted(parent.glob("ace3-position2-fresh-v[89]*")):
            if path.is_file() and not path.is_symlink():
                records.append(file_record(path))
            elif path.is_dir() and not path.is_symlink():
                for child in sorted(path.rglob("*")):
                    if child.is_file() and not child.is_symlink():
                        records.append(file_record(child))
    return records


def framework_records() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    interpreter = Path(os.environ.get("ARGUS_SKILL_PYTHON", sys.executable)).resolve(
        strict=True
    )
    probe = (
        "import importlib,json;"
        "names=['argus_skill.tools.subagent','argus_skill.tools.subagent._cli',"
        "'argus_skill.tools.subagent._direct_run',"
        "'argus_skill.tools.subagent._registry'];"
        "print(json.dumps([importlib.import_module(n).__file__ for n in names]))"
    )
    completed = subprocess.run(
        [str(interpreter), "-B", "-c", probe],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        completed.returncode == 0,
        f"active framework source probe failed: {completed.stderr.strip()}",
    )
    paths = json.loads(completed.stdout)
    require(
        isinstance(paths, list) and len(paths) == 4,
        "active framework source probe returned the wrong file set",
    )
    return file_record(interpreter, executable=True), [
        file_record(Path(path).resolve(strict=True)) for path in paths
    ]


def verify_framework(bindings: Mapping[str, Any]) -> None:
    interpreter, sources = framework_records()
    require(
        interpreter == bindings["toolchain"]["argus_python"],
        "active Argus interpreter changed",
    )
    require(sources == bindings["active_framework"], "active framework source changed")


def verify_accepted_package(
    package: Mapping[str, Any],
    source_archive: Mapping[str, Any],
) -> None:
    require(
        package.get("package_id") == ACCEPTED_PACKAGE_ID
        and package.get("package_review_verdict") == "PASS"
        and package.get("source_archive") == source_archive,
        "accepted v7 package binding mismatch",
    )
    for name in (
        "package_manifest",
        "package_review",
        "package_seal",
        "source_archive",
        "source_review",
    ):
        record = package[name]
        verify_record(record, Path(record["path"]), f"accepted package {name}")


def validate_authority_document(
    bindings: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    required_keys = {
        "accepted_package",
        "accepted_source_archive",
        "claim_boundary",
        "formal_execution_authority",
        "identity",
        "kind",
        "lifecycle_gate",
        "mode",
        "one_shot",
        "reusable",
        "route_metadata",
        "runtime_consumable",
        "runtime_driver",
        "schema_version",
        "status",
        "task_id",
    }
    require(
        authority.get("schema_version") == 1
        and authority.get("kind") == AUTHORITY_KIND
        and set(authority) == required_keys
        and authority.get("mode")
        in {DISPOSABLE_AUTHORITY_MODE, FORMAL_AUTHORITY_MODE}
        and authority.get("reusable") is False,
        "successor lifecycle authority identity mismatch",
    )
    require(
        authority.get("task_id") == bindings["task_id"]
        and authority.get("identity") == bindings["identity"]
        and authority.get("mode") == bindings["mode"]
        and authority.get("status") == bindings["status"]
        and authority.get("route_metadata") == ROUTE_METADATA,
        "successor lifecycle authority binding mismatch",
    )
    require(
        authority.get("runtime_driver") == bindings["runtime_driver"]
        and authority.get("accepted_package") == bindings["accepted_package"]
        and authority.get("accepted_source_archive")
        == bindings["accepted_source_archive"],
        "successor lifecycle authority artifact binding mismatch",
    )
    require(
        authority.get("one_shot")
        == {
            "cardinality": 1,
            "current_authority_consumption_count": 0,
            "current_durable_submission_count": 0,
            "current_payload_invocation_count": 0,
            "current_runtime_invocation_count": 0,
            "automatic_retry": False,
            "retry": False,
            "replay": False,
            "resume": False,
        },
        "successor lifecycle authority is not one-shot and unconsumed",
    )
    mode_contract = {
        DISPOSABLE_AUTHORITY_MODE: {
            "status": "DISPOSABLE_REHEARSAL_UNCONSUMED",
            "formal_execution_authority": False,
            "runtime_consumable": False,
            "claim_boundary": "DISPOSABLE_REHEARSAL_ONLY_NOT_FORMAL_AUTHORITY",
        },
        FORMAL_AUTHORITY_MODE: {
            "status": "AUTHORIZED_ONCE_UNCONSUMED",
            "formal_execution_authority": True,
            "runtime_consumable": True,
            "claim_boundary": "FORMAL_ONE_SHOT_RUNTIME_AUTHORITY",
        },
    }[authority["mode"]]
    require(
        all(authority.get(key) == value for key, value in mode_contract.items()),
        "successor lifecycle authority mode/claim mismatch",
    )
    lifecycle_gate = authority.get("lifecycle_gate")
    require(
        isinstance(lifecycle_gate, dict)
        and set(lifecycle_gate)
        == {"closure_packet", "independent_l2_review", "required_for_formal_mode"}
        and lifecycle_gate.get("required_for_formal_mode") is True,
        "successor lifecycle L2 gate shape mismatch",
    )
    if authority["mode"] == DISPOSABLE_AUTHORITY_MODE:
        require(
            lifecycle_gate["closure_packet"] is None
            and lifecycle_gate["independent_l2_review"] is None,
            "disposable authority cannot carry formal L2 claims",
        )
        return
    closure_record = lifecycle_gate["closure_packet"]
    review_record = lifecycle_gate["independent_l2_review"]
    require(
        isinstance(closure_record, dict) and isinstance(review_record, dict),
        "formal mode requires independent L2 PASS",
    )
    verify_record(
        closure_record,
        Path(closure_record["path"]),
        "lifecycle closure packet",
    )
    verify_record(
        review_record,
        Path(review_record["path"]),
        "independent lifecycle L2 review",
    )
    review = load_json(Path(review_record["path"]))
    require(
        review.get("kind")
        == "ace3_position2_runtime_wrapper_independent_l2_review_binding"
        and review.get("binding_producer_role") == "engineer"
        and review.get("verdict_producer_role") == "reviewer"
        and review.get("independent") is True
        and review.get("reviewer_level") == "L2"
        and review.get("verdict") == "PASS"
        and review.get("status") == "L2_PASS"
        and review.get("closure_binding") == closure_record,
        "formal mode requires independent L2 PASS",
    )
    review_source_record = review.get("review_source")
    require(
        isinstance(review_source_record, dict),
        "formal mode requires sealed independent review source",
    )
    verify_record(
        review_source_record,
        Path(review_source_record["path"]),
        "independent lifecycle L2 review source",
    )
    review_source = load_json(Path(review_source_record["path"]))
    review_decision = review_source.get("review")
    require(
        review_source.get("kind") == "round_reviewed_handoff"
        and review_source.get("producer_role") == "reviewer"
        and isinstance(review_decision, dict)
        and review_decision.get("status") == "continue"
        and str(review_decision.get("reason", "")).startswith(
            "Independent L2 PASS:"
        ),
        "formal mode requires sealed independent review source",
    )


def authenticate_static(
    bindings: Mapping[str, Any],
    authority_path: Path,
    driver_path: Path,
) -> dict[str, Any]:
    require(
        bindings.get("schema_version") == 1
        and bindings.get("kind") == LIFECYCLE_BINDINGS_KIND,
        "wrapper binding identity mismatch",
    )
    require(
        bindings.get("route_metadata") == ROUTE_METADATA,
        "route metadata binding mismatch",
    )
    verify_record(bindings["runtime_driver"], driver_path, "runtime driver")
    authority_record = bindings["identity_authority"]
    verify_record(authority_record, authority_path, "identity authority")
    authority = load_json(authority_path)
    verify_accepted_package(
        bindings["accepted_package"],
        bindings["accepted_source_archive"],
    )
    validate_authority_document(bindings, authority)
    verify_record(
        bindings["accepted_source_archive"],
        Path(bindings["accepted_source_archive"]["path"]),
        "accepted source archive",
    )
    vector_sources = bindings["transaction_input_sources"]
    verify_record(
        vector_sources["inputs"],
        Path(vector_sources["inputs"]["path"]),
        "transaction inputs",
    )
    verify_record(
        vector_sources["rope_coefficients"],
        Path(vector_sources["rope_coefficients"]["path"]),
        "transaction RoPE coefficients",
    )
    for index, record in enumerate(vector_sources["tensors"]):
        verify_record(
            record,
            Path(record["path"]),
            f"transaction tensor {index}",
        )
    for label, record in bindings["toolchain"].items():
        verify_tool_record(record, label)
    verify_framework(bindings)
    return authority


def validate_cwd(expected: Path, actual: Path) -> None:
    require(actual.resolve(strict=True) == expected, "driver cwd is not project root")


def task_paths(
    bindings: Mapping[str, Any],
    registry_root: Path | None = None,
) -> tuple[Path, Path]:
    root = registry_root if registry_root is not None else ROOT / ".argus_subagents"
    task_id = bindings["task_id"]
    return root / f"{task_id}.json", root / f"{task_id}_logs"


def authenticate_active_wrapper(
    bindings: Mapping[str, Any],
    *,
    registry_root: Path | None = None,
) -> dict[str, Any]:
    receipt_path, log_dir = task_paths(bindings, registry_root)
    receipt = load_json(receipt_path)
    task_id = bindings["task_id"]
    require(receipt.get("task_id") == task_id, "active receipt task mismatch")
    require(receipt.get("state") in {"starting", "running"}, "prior terminal receipt")
    require(receipt.get("mode") == "direct", "active receipt mode mismatch")
    require(receipt.get("cwd") == str(ROOT), "active receipt cwd mismatch")
    require(
        receipt.get("command") == bindings["submission"]["child_command"],
        "active receipt command mismatch",
    )
    try:
        description = json.loads(str(receipt.get("description", "")))
    except json.JSONDecodeError as error:
        raise LifecycleError("active receipt route metadata is not structured") from error
    require(description == ROUTE_METADATA, "active receipt route metadata mismatch")
    run_id = receipt.get("run_id")
    require(
        isinstance(run_id, str)
        and run_id.startswith(task_id + "-")
        and run_id[len(task_id) + 1 :].isdigit(),
        "active receipt run ID mismatch",
    )
    require(
        not any(key in receipt for key in ("completed_at", "exit_code", "error")),
        "active receipt contains terminal state",
    )
    require(
        any(
            pid_alive(receipt.get(key))
            for key in ("worker_pid", "pid", "submitter_pid")
        ),
        "active receipt has no live wrapper process",
    )
    require(log_dir.is_dir() and not log_dir.is_symlink(), "active log directory missing")
    require(
        {path.name for path in log_dir.iterdir()} == {"stdout.log", "stderr.log"},
        "active log namespace is not the exact pre-child namespace",
    )
    for name in ("stdout.log", "stderr.log"):
        path = log_dir / name
        require(path.is_file() and not path.is_symlink(), f"active log missing: {name}")
    require_absent(log_dir / f"exit_code.{run_id}", "active exit sidecar")
    return receipt


def runtime_paths(bindings: Mapping[str, Any]) -> dict[str, Path]:
    return {
        name: Path(value)
        for name, value in bindings["runtime_paths"].items()
    }


def verify_replay_absence(bindings: Mapping[str, Any]) -> None:
    paths = runtime_paths(bindings)
    for name in (
        "consumption_sentinel",
        "payload_invocation",
        "output_root",
        "terminal",
        "transaction_boundary",
    ):
        require_absent(paths[name], name.replace("_", " "))


def outer_pre_submit(bindings_path: Path, authority_path: Path) -> dict[str, Any]:
    bindings = load_json(bindings_path)
    driver = Path(bindings["runtime_driver"]["path"])
    authenticate_static(bindings, authority_path, driver)
    validate_cwd(ROOT, Path.cwd())
    receipt, logs = task_paths(bindings)
    require_absent(receipt, "durable runner receipt")
    require_absent(logs, "durable runner log namespace")
    verify_replay_absence(bindings)
    paths = runtime_paths(bindings)
    review = {
        "schema_version": 1,
        "kind": "ace3_position2_wrapper_outer_pre_submit_review",
        "task_id": bindings["task_id"],
        "route_metadata": ROUTE_METADATA,
        "receipt_absent": True,
        "runner_logs_absent": True,
        "replay_state_absent": True,
        "runtime_driver": bindings["runtime_driver"],
        "identity_authority": bindings["identity_authority"],
        "active_framework": bindings["active_framework"],
        "reviewed_at_utc": now(),
    }
    write_json_exclusive(paths["pre_submit_review"], review)
    return review


def run_logged_step(
    name: str,
    argv: list[str],
    cwd: Path,
    step_root: Path,
    *,
    environment: Mapping[str, str] | None = None,
    expected_exit_codes: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    invocation = step_root / f"{name}-invocation.json"
    stdout_path = step_root / f"{name}.stdout.log"
    stderr_path = step_root / f"{name}.stderr.log"
    terminal_path = step_root / f"{name}-terminal.json"
    write_json_exclusive(
        invocation,
        {
            "schema_version": 1,
            "kind": "ace3_position2_wrapper_rehearsal_step_invocation",
            "name": name,
            "argv": argv,
            "cwd": str(cwd),
            "started_at_utc": now(),
        },
    )
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=None if environment is None else dict(environment),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    stdout_path.chmod(0o444)
    stderr_path.chmod(0o444)
    write_json_exclusive(
        terminal_path,
        {
            "schema_version": 1,
            "kind": "ace3_position2_wrapper_rehearsal_step_terminal",
            "name": name,
            "exit_code": completed.returncode,
            "ended_at_utc": now(),
        },
    )
    require(
        completed.returncode in expected_exit_codes,
        f"{name} failed with exit {completed.returncode}",
    )
    return {
        "invocation": file_record(invocation),
        "stdout": file_record(stdout_path),
        "stderr": file_record(stderr_path),
        "terminal": file_record(terminal_path),
    }


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with tarfile.open(archive_path, "r") as archive:
        members = archive.getmembers()
        require(members, "accepted source archive is empty")
        for member in members:
            target = (destination / member.name).resolve()
            require(
                target == destination_root or destination_root in target.parents,
                "accepted source archive member escapes destination",
            )
            require(
                (member.isfile() or member.isdir())
                and not member.issym()
                and not member.islnk(),
                "accepted source archive contains a special or linked member",
            )
        archive.extractall(destination, filter="data")


def install_signal_handlers() -> None:
    def handle(signum: int, _frame: object) -> None:
        raise TerminalSignal(signum)

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)


def execute_rehearsal(bindings_path: Path, authority_path: Path) -> int:
    bindings = load_json(bindings_path)
    paths = runtime_paths(bindings)
    paths["runtime_root"].mkdir(mode=0o700)
    fsync_directory(paths["runtime_root"].parent)
    install_signal_handlers()
    counters = {
        "driver_invocations": 1,
        "authority_consumptions": 0,
        "payload_invocations": 0,
        "source_archive_extractions": 0,
        "verilator_transactions": 0,
        "terminal_manifests": 0,
        "retries": 0,
        "replays": 0,
        "resumes": 0,
    }
    status = "FAIL_NON_RETRYABLE"
    detail = "runtime driver did not reach a classified terminal"
    exit_code = 2
    receipt: dict[str, Any] | None = None
    try:
        authenticate_static(bindings, authority_path, Path(__file__).resolve())
        validate_cwd(ROOT, Path.cwd())
        receipt = authenticate_active_wrapper(bindings)
        verify_replay_absence(bindings)
        write_json_exclusive(
            paths["consumption_sentinel"],
            {
                "schema_version": 1,
                "kind": "ace3_position2_disposable_rehearsal_consumed",
                "task_id": bindings["task_id"],
                "route_metadata": ROUTE_METADATA,
                "identity_authority": bindings["identity_authority"],
                "cardinality": 1,
                "retry": False,
                "replay": False,
                "resume": False,
                "consumed_at_utc": now(),
            },
        )
        counters["authority_consumptions"] = 1
        write_json_exclusive(
            paths["payload_invocation"],
            {
                "schema_version": 1,
                "kind": "ace3_position2_disposable_rehearsal_payload_invocation",
                "task_id": bindings["task_id"],
                "route_metadata": ROUTE_METADATA,
                "run_id": receipt["run_id"],
                "cardinality": 1,
                "invoked_at_utc": now(),
            },
        )
        counters["payload_invocations"] = 1

        output_root = paths["output_root"]
        output_root.mkdir(mode=0o700)
        fsync_directory(output_root.parent)
        source_root = output_root / "source"
        source_root.mkdir(mode=0o700)
        archive = Path(bindings["accepted_source_archive"]["path"])
        safe_extract(archive, source_root)
        counters["source_archive_extractions"] = 1

        vectors = output_root / "vectors"
        raw = output_root / "raw"
        obj_dir = output_root / "obj_dir"
        steps = output_root / "steps"
        for directory in (vectors, raw, steps):
            directory.mkdir(mode=0o700)
        vector_sources = bindings["transaction_input_sources"]
        copied_inputs = {
            "inputs": (
                Path(vector_sources["inputs"]["path"]),
                vectors / "inputs.hex",
            ),
            "rope_coefficients": (
                Path(vector_sources["rope_coefficients"]["path"]),
                vectors / "rope_coefficients.hex",
            ),
        }
        tensor_root = vectors / "tensors"
        tensor_root.mkdir(mode=0o700)
        copied_records: list[dict[str, Any]] = []
        for label, (source, destination) in copied_inputs.items():
            verify_record(vector_sources[label], source, label)
            write_bytes_exclusive(destination, source.read_bytes(), 0o444)
            copied_records.append(file_record(destination))
        for record in vector_sources["tensors"]:
            source = Path(record["path"])
            verify_record(record, source, "transaction tensor")
            destination = tensor_root / source.name
            write_bytes_exclusive(destination, source.read_bytes(), 0o444)
            copied_records.append(file_record(destination))
        write_bytes_exclusive(
            vectors / "trace.hex",
            b"0000000000000000\n",
            0o444,
        )
        write_bytes_exclusive(
            vectors / "final.hex",
            b"0000000000\n" * 896,
            0o444,
        )
        write_json_exclusive(
            vectors / "boundary_manifest.json",
            {
                "schema_version": 1,
                "kind": "ace3_disposable_post_vector_open_probe",
                "trace_records": 1,
                "final_records": 896,
                "comparison_claim": False,
                "expected_stop": "injected failure after first raw trace",
            },
        )
        model_python = verify_tool_record(
            bindings["toolchain"]["model_python"], "model_python"
        )
        compiler = verify_tool_record(bindings["toolchain"]["compiler"], "compiler")
        verilator = verify_tool_record(bindings["toolchain"]["verilator"], "verilator")
        make = verify_tool_record(bindings["toolchain"]["make"], "make")
        environment = {
            **os.environ,
            "CXX": str(compiler),
            "MAKE": str(make),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        step_records = {}
        step_records["interpreter_probe"] = run_logged_step(
            "interpreter-probe",
            [
                str(model_python),
                "-B",
                "-c",
                (
                    "from pathlib import Path;"
                    f"p=Path({str(vectors / 'inputs.hex')!r});"
                    "assert p.is_file() and len(p.read_text().splitlines())==896"
                ),
            ],
            source_root,
            steps,
            environment=environment,
        )
        rtl_paths = [
            source_root / "ace3/rtl" / name for name in DECODER_RTL_NAMES
        ]
        require(
            all(path.is_file() and not path.is_symlink() for path in rtl_paths),
            "accepted archive decoder RTL closure is incomplete",
        )
        step_records["verilator"] = run_logged_step(
            "verilator",
            [
                str(verilator),
                "--cc",
                "--exe",
                "--savable",
                "-Wall",
                "-Wno-fatal",
                "--top-module",
                "ace3_decoder_layer0_token_engine",
                "--Mdir",
                str(obj_dir),
                *[str(path) for path in rtl_paths],
                str(
                    source_root
                    / "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp"
                ),
            ],
            source_root,
            steps,
            environment=environment,
        )
        step_records["make"] = run_logged_step(
            "make",
            [
                str(make),
                "-C",
                str(obj_dir),
                "-f",
                "Vace3_decoder_layer0_token_engine.mk",
                "-j1",
                f"CXX={compiler}",
            ],
            source_root,
            steps,
            environment=environment,
        )
        simulator = obj_dir / "Vace3_decoder_layer0_token_engine"
        require(
            simulator.is_file()
            and not simulator.is_symlink()
            and os.access(simulator, os.X_OK),
            "fresh Verilator simulator is unavailable",
        )
        step_records["simulate"] = run_logged_step(
            "simulate",
            [
                str(simulator),
                "--layer-index",
                "0",
                "--vector-dir",
                str(vectors),
                "--tensor-dir",
                str(vectors),
                "--raw-dir",
                str(raw),
                "--transaction-position",
                "0",
                "--state-out",
                str(output_root / "position001.state"),
                "--fail-after-raw",
                "--progress-interval",
                "1000000",
            ],
            source_root,
            steps,
            environment=environment,
            expected_exit_codes=(2,),
        )
        simulator_terminal = raw / "terminal.txt"
        raw_trace = raw / "trace.hex"
        require(
            simulator_terminal.read_text(encoding="ascii")
            == (
                "schema=ace3_decoder_token_transaction_v1 layer_index=0 "
                "position=0 natural_terminal=0 exit_code=2 trace_count=1 "
                "final_count=0 done_count=0\n"
            ),
            "real Verilator injected-stop terminal mismatch",
        )
        trace_rows = raw_trace.read_text(encoding="ascii").splitlines()
        require(len(trace_rows) == 1, "real Verilator raw trace cardinality mismatch")
        simulate_stderr = (
            steps / "simulate.stderr.log"
        ).read_text(encoding="utf-8")
        require(
            "injected failure after raw trace" in simulate_stderr,
            "real Verilator did not stop after the raw transaction boundary",
        )
        counters["verilator_transactions"] = 1
        extracted_records = [
            file_record(
                source_root / "ace3/rtl/ace3_decoder_layer0_token_engine.sv"
            ),
            file_record(
                source_root
                / "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp"
            ),
        ]
        boundary = {
            "schema_version": 1,
            "kind": "ace3_position2_wrapper_real_verilator_transaction_boundary",
            "task_id": bindings["task_id"],
            "route_metadata": ROUTE_METADATA,
            "accepted_source_archive": bindings["accepted_source_archive"],
            "source_archive_extracted": True,
            "extracted_source_records": extracted_records,
            "authenticated_transaction_inputs": {
                "source_records": vector_sources,
                "copied_records": copied_records,
                "probe_trace": file_record(vectors / "trace.hex"),
                "probe_final": file_record(vectors / "final.hex"),
                "probe_manifest": file_record(
                    vectors / "boundary_manifest.json"
                ),
                "oracle_comparison_claimed": False,
            },
            "resolved_toolchain": bindings["toolchain"],
            "step_records": step_records,
            "simulator": file_record(simulator),
            "simulator_terminal": file_record(simulator_terminal),
            "first_raw_trace": file_record(raw_trace),
            "post_vector_open_real_verilator_boundary": True,
            "accepted_transaction_events": len(trace_rows),
            "expected_injected_stop_observed": True,
            "rtl_correctness_claimed": False,
        }
        write_json_exclusive(paths["transaction_boundary"], boundary)
        status = "REHEARSAL_PASS_AWAITING_INDEPENDENT_L2"
        detail = "fresh wrapper reached the post-vector-open real Verilator boundary"
        exit_code = 0
        return 0
    except LifecycleError as error:
        detail = str(error)
        return 2
    except TerminalSignal as error:
        detail = str(error)
        exit_code = 128 + error.signum
        return exit_code
    finally:
        if path_absent(paths["terminal"]):
            counters["terminal_manifests"] = 1
            write_json_exclusive(
                paths["terminal"],
                {
                    "schema_version": 1,
                    "kind": "ace3_position2_disposable_rehearsal_terminal",
                    "task_id": bindings["task_id"],
                    "route_metadata": ROUTE_METADATA,
                    "run_id": None if receipt is None else receipt.get("run_id"),
                    "status": status,
                    "detail": detail,
                    "exit_code": exit_code,
                    "retryable": False,
                    "consumed": counters["authority_consumptions"] == 1,
                    "counters": counters,
                    "ended_at_utc": now(),
                },
            )


def classify_crash_state(
    *, child_started: bool, consumed: bool, terminal: bool
) -> dict[str, Any]:
    retryable = not child_started and not consumed and not terminal
    return {
        "child_started": child_started,
        "consumed": consumed,
        "terminal": terminal,
        "retryable": retryable,
        "invariant_pass": not (retryable and (consumed or terminal)),
    }


def expect_failure(
    results: list[dict[str, str]],
    name: str,
    operation: Callable[[], object],
    expected: str,
) -> None:
    try:
        operation()
    except LifecycleError as error:
        require(expected in str(error), f"{name} failed at the wrong boundary: {error}")
        results.append({"name": name, "status": "PASS", "detail": str(error)})
        return
    raise LifecycleError(f"{name} did not reject")


def run_negative_controls(
    bindings_path: Path,
    authority_path: Path,
) -> dict[str, Any]:
    bindings = load_json(bindings_path)
    authority = load_json(authority_path)
    paths = runtime_paths(bindings)
    results: list[dict[str, str]] = []
    scratch_parent = paths["scratch_root"]
    scratch_parent.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory(dir=scratch_parent) as temporary:
        scratch = Path(temporary)
        receipt_root = scratch / "registry"
        logs = receipt_root / f"{bindings['task_id']}_logs"
        logs.mkdir(parents=True)
        for name in ("stdout.log", "stderr.log"):
            (logs / name).write_bytes(b"")
        write_json_exclusive(
            receipt_root / f"{bindings['task_id']}.json",
            {
                "state": "done",
                "task_id": bindings["task_id"],
                "run_id": f"{bindings['task_id']}-123",
                "mode": "direct",
                "cwd": str(ROOT),
                "command": bindings["submission"]["child_command"],
                "description": compact_json(ROUTE_METADATA),
                "worker_pid": os.getpid(),
            },
            mode=0o600,
        )
        expect_failure(
            results,
            "pre-existing-terminal-receipt",
            lambda: authenticate_active_wrapper(
                bindings, registry_root=receipt_root
            ),
            "prior terminal receipt",
        )

        existing = scratch / "consumed.json"
        existing.write_text("{}\n", encoding="ascii")
        changed = json.loads(json.dumps(bindings))
        changed["runtime_paths"]["consumption_sentinel"] = str(existing)
        expect_failure(
            results,
            "pre-existing-consumption-sentinel",
            lambda: verify_replay_absence(changed),
            "consumption sentinel already exists",
        )

        changed = json.loads(json.dumps(bindings))
        changed["runtime_driver"]["sha256"] = "0" * 64
        expect_failure(
            results,
            "altered-driver-hash",
            lambda: authenticate_static(
                changed,
                authority_path,
                Path(bindings["runtime_driver"]["path"]),
            ),
            "runtime driver record mismatch",
        )

        changed = json.loads(json.dumps(bindings))
        changed["identity_authority"]["sha256"] = "0" * 64
        expect_failure(
            results,
            "altered-authority-hash",
            lambda: authenticate_static(
                changed,
                authority_path,
                Path(bindings["runtime_driver"]["path"]),
            ),
            "identity authority record mismatch",
        )

        changed_authority = json.loads(json.dumps(authority))
        changed_authority["mode"] = FORMAL_AUTHORITY_MODE
        changed_bindings = json.loads(json.dumps(bindings))
        changed_bindings["mode"] = FORMAL_AUTHORITY_MODE
        confused_path = scratch / "disposable-mode-confusion.json"
        write_json_exclusive(confused_path, changed_authority)
        changed_bindings["identity_authority"] = file_record(confused_path)
        expect_failure(
            results,
            "disposable-cannot-become-formal",
            lambda: authenticate_static(
                changed_bindings,
                confused_path,
                Path(bindings["runtime_driver"]["path"]),
            ),
            "authority mode/claim mismatch",
        )

        bypass_authority = json.loads(json.dumps(authority))
        bypass_authority.update(
            {
                "mode": FORMAL_AUTHORITY_MODE,
                "status": "AUTHORIZED_ONCE_UNCONSUMED",
                "formal_execution_authority": True,
                "runtime_consumable": True,
                "claim_boundary": "FORMAL_ONE_SHOT_RUNTIME_AUTHORITY",
            }
        )
        bypass_bindings = json.loads(json.dumps(bindings))
        bypass_bindings.update(
            {
                "mode": FORMAL_AUTHORITY_MODE,
                "status": "AUTHORIZED_ONCE_UNCONSUMED",
            }
        )
        bypass_path = scratch / "formal-without-independent-l2.json"
        write_json_exclusive(bypass_path, bypass_authority)
        bypass_bindings["identity_authority"] = file_record(bypass_path)
        expect_failure(
            results,
            "formal-cannot-bypass-independent-l2",
            lambda: authenticate_static(
                bypass_bindings,
                bypass_path,
                Path(bindings["runtime_driver"]["path"]),
            ),
            "formal mode requires independent L2 PASS",
        )

        wrong = scratch.resolve()
        expect_failure(
            results,
            "wrong-cwd",
            lambda: validate_cwd(ROOT, wrong),
            "driver cwd is not project root",
        )

        tool = Path(bindings["toolchain"]["compiler"]["path"])
        symlink = scratch / "compiler-link"
        symlink.symlink_to(tool)
        symlink_binding = {
            **bindings["toolchain"]["compiler"],
            "path": str(symlink),
        }
        expect_failure(
            results,
            "symlinked-tool-path",
            lambda: verify_tool_record(symlink_binding, "compiler"),
            "regular file required",
        )
        unbound = scratch / "missing-tool"
        unbound_binding = {
            "path": str(unbound),
            "bytes": 0,
            "sha256": "0" * 64,
        }
        expect_failure(
            results,
            "unbound-tool-path",
            lambda: verify_tool_record(unbound_binding, "compiler"),
            "regular file required",
        )

    scratch_parent.rmdir()
    crash_states = [
        classify_crash_state(
            child_started=child_started,
            consumed=consumed,
            terminal=terminal,
        )
        for child_started, consumed, terminal in (
            (False, False, False),
            (True, False, True),
            (True, True, True),
            (False, False, True),
        )
    ]
    require(
        all(state["invariant_pass"] for state in crash_states),
        "crash retry/consumption invariant failed",
    )
    report = {
        "schema_version": 1,
        "kind": "ace3_position2_wrapper_negative_controls",
        "task_id": bindings["task_id"],
        "results": results,
        "crash_boundary_classification": crash_states,
        "no_state_retryable_and_consumed_or_terminal": True,
    }
    write_json_exclusive(paths["negative_controls"], report)
    return report


def record_duplicate_control(bindings_path: Path, exit_code: int) -> dict[str, Any]:
    bindings = load_json(bindings_path)
    paths = runtime_paths(bindings)
    stdout = paths["duplicate_stdout"].read_text(encoding="utf-8")
    stderr = paths["duplicate_stderr"].read_text(encoding="utf-8")
    paths["duplicate_stdout"].chmod(0o444)
    paths["duplicate_stderr"].chmod(0o444)
    require(exit_code == 1, "duplicate submit did not return rejection status")
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise LifecycleError("duplicate submit did not return JSON") from error
    require(
        "already" in str(response.get("error", ""))
        and bindings["task_id"] in str(response["error"]),
        "duplicate submit rejected at the wrong boundary",
    )
    receipt_path, _logs = task_paths(bindings)
    receipt = load_json(receipt_path)
    report = {
        "schema_version": 1,
        "kind": "ace3_position2_wrapper_duplicate_submit_control",
        "task_id": bindings["task_id"],
        "status": "PASS",
        "exit_code": exit_code,
        "response": response,
        "preserved_run_id": receipt["run_id"],
        "stdout": file_record(paths["duplicate_stdout"]),
        "stderr": file_record(paths["duplicate_stderr"]),
    }
    write_json_exclusive(paths["duplicate_control"], report)
    return report


def copied_evidence(source: Path, destination: Path) -> dict[str, Any]:
    require(source.is_file() and not source.is_symlink(), f"evidence file missing: {source}")
    write_bytes_exclusive(destination, source.read_bytes(), 0o444)
    return file_record(destination)


def seal_closure(bindings_path: Path, authority_path: Path) -> dict[str, Any]:
    bindings = load_json(bindings_path)
    paths = runtime_paths(bindings)
    authenticate_static(
        bindings,
        authority_path,
        Path(bindings["runtime_driver"]["path"]),
    )
    require(
        formal_state_snapshot() == bindings["formal_state_before"],
        "formal v8/v9 state changed during disposable rehearsal",
    )
    receipt_path, runner_logs = task_paths(bindings)
    receipt = load_json(receipt_path)
    terminal = load_json(paths["terminal"])
    boundary = load_json(paths["transaction_boundary"])
    pre_submit = load_json(paths["pre_submit_review"])
    negatives = load_json(paths["negative_controls"])
    duplicate = load_json(paths["duplicate_control"])
    invocation = load_json(paths["payload_invocation"])
    consumption = load_json(paths["consumption_sentinel"])
    require(
        receipt.get("state") == "done" and receipt.get("exit_code") == 0,
        "durable runner did not reach a successful terminal",
    )
    require(
        json.loads(receipt["description"]) == ROUTE_METADATA
        and terminal["route_metadata"] == ROUTE_METADATA
        and invocation["route_metadata"] == ROUTE_METADATA
        and consumption["route_metadata"] == ROUTE_METADATA,
        "route metadata was not preserved end to end",
    )
    require(
        terminal.get("status") == "REHEARSAL_PASS_AWAITING_INDEPENDENT_L2"
        and terminal.get("retryable") is False
        and terminal.get("consumed") is True,
        "child terminal classification mismatch",
    )
    counters = terminal["counters"]
    require(
        counters["driver_invocations"] == 1
        and counters["authority_consumptions"] == 1
        and counters["payload_invocations"] == 1
        and counters["source_archive_extractions"] == 1
        and counters["verilator_transactions"] == 1
        and counters["terminal_manifests"] == 1
        and all(
            counters[name] == 0 for name in ("retries", "replays", "resumes")
        ),
        "positive lifecycle counters mismatch",
    )
    require(
        boundary.get("post_vector_open_real_verilator_boundary") is True,
        "real Verilator transaction boundary was not reached",
    )
    require(
        pre_submit.get("receipt_absent") is True
        and pre_submit.get("runner_logs_absent") is True,
        "outer pre-submit absence ownership is unproven",
    )
    negative_names = {result["name"] for result in negatives["results"]}
    require(
        negative_names
        == {
            "pre-existing-terminal-receipt",
            "pre-existing-consumption-sentinel",
            "altered-driver-hash",
            "altered-authority-hash",
            "wrong-cwd",
            "symlinked-tool-path",
            "unbound-tool-path",
            "disposable-cannot-become-formal",
            "formal-cannot-bypass-independent-l2",
        }
        and all(result["status"] == "PASS" for result in negatives["results"]),
        "pre-submit negative controls are incomplete",
    )
    require(duplicate.get("status") == "PASS", "duplicate submit control failed")
    comparison = load_json(paths["driver_comparison"])
    require(
        comparison.get("driver_bytes_identical") is True
        and comparison["rehearsed_driver"]["sha256"]
        == comparison["production_successor_driver"]["sha256"],
        "production/rehearsal driver byte comparison failed",
    )
    require(
        comparison.get("shared_binding_kind") == LIFECYCLE_BINDINGS_KIND
        and comparison.get("shared_authority_kind") == AUTHORITY_KIND
        and comparison.get("shared_authentication_entrypoint")
        == "authenticate_static"
        and comparison.get("production_l2_requirement")
        == "INDEPENDENT_L2_PASS_REQUIRED"
        and comparison.get("formal_successor_authority_minted") is False,
        "production shared-authority interface comparison failed",
    )

    copied_root = paths["copied_evidence_root"]
    copied_root.mkdir(mode=0o700)
    copied = {
        "runner_receipt": copied_evidence(
            receipt_path, copied_root / "runner-receipt.json"
        ),
        "runner_stdout": copied_evidence(
            runner_logs / "stdout.log", copied_root / "runner-stdout.log"
        ),
        "runner_stderr": copied_evidence(
            runner_logs / "stderr.log", copied_root / "runner-stderr.log"
        ),
    }
    exit_sidecar = runner_logs / f"exit_code.{receipt['run_id']}"
    copied["runner_exit_sidecar"] = copied_evidence(
        exit_sidecar, copied_root / "runner-exit-sidecar.txt"
    )
    packet = {
        "schema_version": 1,
        "kind": "ace3_position2_runtime_wrapper_engineer_closure_packet",
        "status": "ENGINEER_PASS_AWAITING_INDEPENDENT_L2",
        "task_id": bindings["task_id"],
        "route_metadata": ROUTE_METADATA,
        "positive_control": {
            "structured_task_count": 1,
            "durable_runner": copied,
            "receipt_run_id": receipt["run_id"],
            "consumption_sentinel": file_record(paths["consumption_sentinel"]),
            "payload_invocation": file_record(paths["payload_invocation"]),
            "child_terminal": file_record(paths["terminal"]),
            "transaction_boundary": file_record(paths["transaction_boundary"]),
            "accepted_source_archive": bindings["accepted_source_archive"],
            "resolved_toolchain": bindings["toolchain"],
        },
        "negative_controls": {
            "duplicate_submit": file_record(paths["duplicate_control"]),
            "isolated_controls": file_record(paths["negative_controls"]),
            "all_required_controls_pass": True,
        },
        "crash_boundary": {
            "classification": negatives["crash_boundary_classification"],
            "no_state_retryable_and_consumed_or_terminal": True,
        },
        "outer_pre_submit_review": file_record(paths["pre_submit_review"]),
        "production_successor_comparison": file_record(
            paths["driver_comparison"]
        ),
        "active_framework": bindings["active_framework"],
        "formal_successor_authority_minted": False,
        "formal_v8_v9_state_unchanged": True,
        "claim_boundary": (
            "disposable controller transaction rehearsal only; no position-2 "
            "runtime PASS, formal successor authority, full-model, dialogue, "
            "synthesis, PPA, FPGA, or silicon claim"
        ),
        "sealed_at_utc": now(),
    }
    write_json_exclusive(paths["closure_packet"], packet)
    seal = {
        "schema_version": 1,
        "kind": "ace3_position2_runtime_wrapper_engineer_closure_seal",
        "task_id": bindings["task_id"],
        "closure_packet": file_record(paths["closure_packet"]),
        "bound_evidence": [
            file_record(path)
            for path in (
                paths["pre_submit_review"],
                paths["negative_controls"],
                paths["duplicate_control"],
                paths["consumption_sentinel"],
                paths["payload_invocation"],
                paths["terminal"],
                paths["transaction_boundary"],
                paths["driver_comparison"],
            )
        ],
        "copied_runner_evidence": copied,
        "independent_l2_review": "REQUIRED_NOT_ENGINEER_AUTHORED",
    }
    write_json_exclusive(paths["closure_seal"], seal)
    review_request = {
        "schema_version": 1,
        "kind": "ace3_position2_runtime_wrapper_independent_l2_review_request",
        "task_id": bindings["task_id"],
        "closure_packet": file_record(paths["closure_packet"]),
        "closure_seal": file_record(paths["closure_seal"]),
        "required_checks": [
            "all seven baseline negative gate classes plus both mode-confusion controls",
            "positive active-wrapper lifecycle and exact task cardinality",
            "crash counter retry/consumed/terminal invariant",
            "active framework source bindings",
            "rehearsed versus production successor driver bytes and argv shape",
            "shared authority schema and authentication path for disposable and formal modes",
            "production formal-authority interface accepted by the rehearsed byte-identical driver",
            "formal mode requires the independent lifecycle L2 PASS and cannot bypass it",
            "no formal v8/v9 successor authority or runtime claim",
        ],
        "review_output": str(paths["l2_review"]),
        "requested_reviewer_level": "independent L2",
    }
    write_json_exclusive(paths["l2_review_request"], review_request)
    return packet


def prepare(rehearsal_id: str) -> dict[str, Any]:
    require(
        rehearsal_id.startswith("ace3-position2-wrapper-rehearsal-"),
        "rehearsal ID is outside the disposable namespace",
    )
    root = REHEARSAL_PARENT / rehearsal_id
    require_absent(root, "rehearsal root")
    REHEARSAL_PARENT.mkdir(parents=True, exist_ok=True)
    immutable = root / "immutable"
    outer = root / "outer"
    evidence = root / "evidence"
    review = root / "review"
    for directory in (root, immutable, outer, evidence, review):
        directory.mkdir(mode=0o700)
        fsync_directory(directory.parent)

    task_id = rehearsal_id
    rehearsal_driver = immutable / "rehearsed-runtime-driver.py"
    production_driver = immutable / "production-successor-runtime-driver.py"
    driver_bytes = Path(__file__).read_bytes()
    write_bytes_exclusive(rehearsal_driver, driver_bytes, 0o555)
    write_bytes_exclusive(production_driver, driver_bytes, 0o555)
    driver_record = file_record(rehearsal_driver, executable=True)
    production_record = file_record(production_driver, executable=True)
    require(
        driver_record["sha256"] == production_record["sha256"]
        and driver_record["bytes"] == production_record["bytes"],
        "staged driver bytes differ",
    )

    v8_authority = load_json(V8_AUTHORITY)
    accepted_package = v8_authority["accepted_package"]
    archive_binding = v8_authority["accepted_package"]["source_archive"]
    archive_path = Path(archive_binding["path"])
    verify_record(archive_binding, archive_path, "accepted source archive")
    argus_python, active_framework = framework_records()
    model_python = file_record(
        Path(v8_authority["exact_execution"]["interpreter"]["path"]).resolve(
            strict=True
        ),
        executable=True,
    )
    toolchain = {
        "argus_python": argus_python,
        "model_python": model_python,
        "compiler": resolve_tool("g++"),
        "verilator": resolve_tool("verilator"),
        "make": resolve_tool("make"),
    }
    tensor_records = [
        file_record(path.resolve(strict=True))
        for path in sorted((V6_VECTOR_ROOT / "tensors").iterdir())
        if path.is_file()
    ]
    require(tensor_records, "authenticated v6 tensor inputs are unavailable")
    transaction_input_sources = {
        "inputs": file_record(
            (V6_POSITION0_VECTOR_ROOT / "inputs.hex").resolve(strict=True)
        ),
        "rope_coefficients": file_record(
            (
                V6_POSITION0_VECTOR_ROOT / "rope_coefficients.hex"
            ).resolve(strict=True)
        ),
        "tensors": tensor_records,
        "provenance": (
            "hash-bound v6 position-0 inputs and layer-0 tensors; disposable "
            "probe expected trace/final rows are non-oracle injected-stop data"
        ),
    }
    authority_path = immutable / "disposable-lifecycle-authority.json"
    authority = {
        "schema_version": 1,
        "kind": AUTHORITY_KIND,
        "mode": DISPOSABLE_AUTHORITY_MODE,
        "status": "DISPOSABLE_REHEARSAL_UNCONSUMED",
        "task_id": task_id,
        "identity": task_id,
        "route_metadata": ROUTE_METADATA,
        "runtime_driver": driver_record,
        "accepted_package": accepted_package,
        "accepted_source_archive": archive_binding,
        "formal_execution_authority": False,
        "runtime_consumable": False,
        "reusable": False,
        "one_shot": {
            "cardinality": 1,
            "current_authority_consumption_count": 0,
            "current_durable_submission_count": 0,
            "current_payload_invocation_count": 0,
            "current_runtime_invocation_count": 0,
            "automatic_retry": False,
            "retry": False,
            "replay": False,
            "resume": False,
        },
        "lifecycle_gate": {
            "required_for_formal_mode": True,
            "closure_packet": None,
            "independent_l2_review": None,
        },
        "claim_boundary": "DISPOSABLE_REHEARSAL_ONLY_NOT_FORMAL_AUTHORITY",
    }
    write_json_exclusive(authority_path, authority)

    bindings_path = immutable / "rehearsal-bindings.json"
    child_command = shlex.join(
        [
            argus_python["path"],
            "-B",
            str(rehearsal_driver),
            "execute",
            "--bindings",
            str(bindings_path),
            "--authority",
            str(authority_path),
        ]
    )
    runtime_root = root / "runtime"
    bindings = {
        "schema_version": 1,
        "kind": LIFECYCLE_BINDINGS_KIND,
        "mode": DISPOSABLE_AUTHORITY_MODE,
        "status": "DISPOSABLE_REHEARSAL_UNCONSUMED",
        "task_id": task_id,
        "identity": task_id,
        "route_metadata": ROUTE_METADATA,
        "runtime_driver": driver_record,
        "identity_authority": file_record(authority_path),
        "accepted_package": accepted_package,
        "accepted_source_archive": archive_binding,
        "transaction_input_sources": transaction_input_sources,
        "toolchain": toolchain,
        "active_framework": active_framework,
        "formal_state_before": formal_state_snapshot(),
        "submission": {
            "cwd": str(ROOT),
            "description": compact_json(ROUTE_METADATA),
            "child_command": child_command,
            "mode": "direct",
            "timeout_seconds": 600,
        },
        "runtime_paths": {
            "runtime_root": str(runtime_root),
            "consumption_sentinel": str(runtime_root / "consumption.json"),
            "payload_invocation": str(runtime_root / "payload-invocation.json"),
            "output_root": str(runtime_root / "output"),
            "terminal": str(runtime_root / "terminal.json"),
            "transaction_boundary": str(runtime_root / "transaction-boundary.json"),
            "scratch_root": str(root / "scratch"),
            "pre_submit_review": str(outer / "pre-submit-review.json"),
            "negative_controls": str(evidence / "negative-controls.json"),
            "duplicate_stdout": str(outer / "duplicate-submit.stdout"),
            "duplicate_stderr": str(outer / "duplicate-submit.stderr"),
            "duplicate_control": str(evidence / "duplicate-submit-control.json"),
            "driver_comparison": str(evidence / "driver-comparison.json"),
            "copied_evidence_root": str(evidence / "runner"),
            "closure_packet": str(evidence / "closure-packet.json"),
            "closure_seal": str(evidence / "closure-seal.json"),
            "l2_review_request": str(review / "l2-review-request.json"),
            "l2_review": str(review / "l2-review.json"),
        },
    }
    write_json_exclusive(bindings_path, bindings)

    production_bindings = immutable / "production-successor-bindings.template.json"
    write_json_exclusive(
        production_bindings,
        {
            "schema_version": 1,
            "kind": "ace3_position2_production_successor_binding_template",
            "status": "UNAUTHORIZED_TEMPLATE_DO_NOT_SUBMIT",
            "required_runtime_binding_kind": LIFECYCLE_BINDINGS_KIND,
            "required_mode": FORMAL_AUTHORITY_MODE,
            "required_status": "AUTHORIZED_ONCE_UNCONSUMED",
            "identity_authority": {
                "requirement": "REQUIRED_AFTER_L2_ACCEPTANCE",
                "accepted_kind": AUTHORITY_KIND,
                "required_mode": FORMAL_AUTHORITY_MODE,
                "required_status": "AUTHORIZED_ONCE_UNCONSUMED",
                "formal_execution_authority": True,
                "runtime_consumable": True,
                "reusable": False,
                "cardinality": 1,
                "required_fields": sorted(authority),
                "lifecycle_gate": {
                    "closure_packet": "REQUIRED_HASH_BOUND_RECORD",
                    "independent_l2_review": "REQUIRED_HASH_BOUND_L2_PASS",
                    "required_for_formal_mode": True,
                },
            },
            "identity": "<fresh-successor-identity>",
            "route_metadata": ROUTE_METADATA,
            "runtime_driver": production_record,
            "accepted_package": accepted_package,
            "accepted_source_archive": archive_binding,
            "bound_runtime_records": "<bind-after-independent-L2>",
        },
    )
    production_argv = [
        argus_python["path"],
        "-B",
        str(production_driver),
        "execute",
        "--bindings",
        "<formal-successor-bindings-path-required>",
        "--authority",
        "<formal-authority-path-required>",
    ]
    comparison_path = Path(bindings["runtime_paths"]["driver_comparison"])
    write_json_exclusive(
        comparison_path,
        {
            "schema_version": 1,
            "kind": "ace3_position2_rehearsed_production_driver_comparison",
            "rehearsed_driver": driver_record,
            "production_successor_driver": production_record,
            "driver_bytes_identical": True,
            "rehearsed_argv": shlex.split(child_command),
            "production_successor_argv": production_argv,
            "production_successor_binding_template": file_record(
                production_bindings
            ),
            "shared_binding_kind": LIFECYCLE_BINDINGS_KIND,
            "shared_authority_kind": AUTHORITY_KIND,
            "shared_authority_fields": sorted(authority),
            "shared_authentication_entrypoint": "authenticate_static",
            "rehearsal_authority_mode": DISPOSABLE_AUTHORITY_MODE,
            "production_authority_mode": FORMAL_AUTHORITY_MODE,
            "production_l2_requirement": "INDEPENDENT_L2_PASS_REQUIRED",
            "accepted_package_id": ACCEPTED_PACKAGE_ID,
            "argv_shape_identical": [
                "-B",
                "<driver>",
                "execute",
                "--bindings",
                "<bound-records>",
                "--authority",
                "<identity-authority-record>",
            ],
            "allowed_differences": [
                "identity",
                "mode and claim values within the shared authority schema",
                "bound artifact records",
                "driver path naming with byte-identical contents",
            ],
            "formal_successor_authority_minted": False,
        },
    )

    submit_argv = [
        argus_python["path"],
        "-m",
        "argus_skill.tools.subagent",
        "submit",
        "--task-id",
        task_id,
        "--description",
        compact_json(ROUTE_METADATA),
        "--mode",
        "direct",
        "--timeout",
        "600",
        "--cwd",
        str(ROOT),
        "--command",
        child_command,
    ]
    submit_script = immutable / "submit-rehearsal.sh"
    duplicate_stdout = Path(bindings["runtime_paths"]["duplicate_stdout"])
    duplicate_stderr = Path(bindings["runtime_paths"]["duplicate_stderr"])
    script = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(ROOT))}
{shlex.join([argus_python["path"], "-B", str(rehearsal_driver), "pre-submit", "--bindings", str(bindings_path), "--authority", str(authority_path)])}
{shlex.join(submit_argv)} > {shlex.quote(str(outer / "submit-receipt.json"))}
set +e
{shlex.join(submit_argv)} > {shlex.quote(str(duplicate_stdout))} 2> {shlex.quote(str(duplicate_stderr))}
duplicate_status=$?
set -e
{shlex.join([argus_python["path"], "-B", str(rehearsal_driver), "record-duplicate", "--bindings", str(bindings_path), "--exit-code"])} "$duplicate_status"
test "$duplicate_status" -eq 1
cat {shlex.quote(str(outer / "submit-receipt.json"))}
"""
    write_bytes_exclusive(submit_script, script.encode("utf-8"), 0o555)
    run_negative_controls(bindings_path, authority_path)
    result = {
        "schema_version": 1,
        "kind": "ace3_position2_wrapper_rehearsal_preparation",
        "task_id": task_id,
        "root": str(root),
        "bindings": file_record(bindings_path),
        "identity_authority": file_record(authority_path),
        "submit_script": file_record(submit_script, executable=True),
        "runtime_driver": driver_record,
        "production_successor_driver": production_record,
        "driver_bytes_identical": True,
        "formal_successor_authority_minted": False,
    }
    write_json_exclusive(outer / "preparation.json", result)
    return result


def bind_l2_and_mint(
    rehearsal_root: Path,
    review_source: Path,
    successor_id: str,
) -> dict[str, Any]:
    require(
        rehearsal_root.parent == REHEARSAL_PARENT
        and rehearsal_root.name.startswith(
            "ace3-position2-wrapper-rehearsal-"
        ),
        "rehearsal root is outside the sealed lifecycle namespace",
    )
    require(
        successor_id.startswith("ace3-position2-fresh-v9-"),
        "successor identity is outside the fresh v9 namespace",
    )
    bindings_path = rehearsal_root / "immutable/rehearsal-bindings.json"
    bindings = load_json(bindings_path)
    require(
        bindings.get("task_id") == rehearsal_root.name
        and bindings.get("mode") == DISPOSABLE_AUTHORITY_MODE,
        "sealed rehearsal binding mismatch",
    )
    paths = runtime_paths(bindings)
    closure_path = paths["closure_packet"]
    closure_seal_path = paths["closure_seal"]
    l2_request_path = paths["l2_review_request"]
    l2_review_path = paths["l2_review"]
    l2_seal_path = rehearsal_root / "evidence/l2-closure-seal.json"
    successor_root = FORMAL_SUCCESSOR_PARENT / successor_id
    for path, label in (
        (l2_review_path, "lifecycle L2 review binding"),
        (l2_seal_path, "lifecycle L2 closure seal"),
        (successor_root, "formal successor root"),
    ):
        require_absent(path, label)

    closure = load_json(closure_path)
    closure_record = file_record(closure_path)
    closure_seal_record = file_record(closure_seal_path)
    l2_request_record = file_record(l2_request_path)
    require(
        closure.get("task_id") == rehearsal_root.name
        and closure.get("status")
        == "ENGINEER_PASS_AWAITING_INDEPENDENT_L2"
        and closure.get("formal_successor_authority_minted") is False,
        "sealed rehearsal closure is not eligible for L2 binding",
    )
    closure_seal = load_json(closure_seal_path)
    require(
        closure_seal.get("closure_packet") == closure_record
        and closure_seal.get("independent_l2_review")
        == "REQUIRED_NOT_ENGINEER_AUTHORED",
        "sealed rehearsal closure seal mismatch",
    )
    l2_request = load_json(l2_request_path)
    require(
        l2_request.get("task_id") == rehearsal_root.name
        and l2_request.get("closure_packet") == closure_record
        and l2_request.get("closure_seal") == closure_seal_record
        and l2_request.get("review_output") == str(l2_review_path),
        "lifecycle L2 review request mismatch",
    )
    review_source_record = file_record(review_source)
    source = load_json(review_source)
    source_decision = source.get("review")
    require(
        source.get("kind") == "round_reviewed_handoff"
        and source.get("producer_role") == "reviewer"
        and source.get("mission_id") == "72e5ec8d2fac"
        and isinstance(source_decision, dict)
        and source_decision.get("status") == "continue"
        and str(source_decision.get("reason", "")).startswith(
            "Independent L2 PASS:"
        )
        and "sealed r5 rehearsal proves"
        in str(source_decision.get("reason", "")),
        "review source does not record the independent r5 L2 PASS",
    )

    template_path = (
        rehearsal_root
        / "immutable/production-successor-bindings.template.json"
    )
    template = load_json(template_path)
    production_driver = Path(template["runtime_driver"]["path"])
    verify_record(
        template["runtime_driver"],
        production_driver,
        "repaired production successor driver",
        executable=True,
    )
    verify_accepted_package(
        template["accepted_package"],
        template["accepted_source_archive"],
    )
    verify_framework(bindings)
    for label, record in bindings["toolchain"].items():
        verify_tool_record(record, label)

    l2_review = {
        "schema_version": 1,
        "kind": (
            "ace3_position2_runtime_wrapper_independent_l2_review_binding"
        ),
        "task_id": rehearsal_root.name,
        "binding_producer_role": "engineer",
        "verdict_producer_role": "reviewer",
        "independent": True,
        "reviewer_level": "L2",
        "verdict": "PASS",
        "status": "L2_PASS",
        "closure_binding": closure_record,
        "closure_seal": closure_seal_record,
        "review_request": l2_request_record,
        "review_source": review_source_record,
        "checks": [
            "active argus_skill.tools.subagent source paths",
            "front-door route metadata preservation",
            "positive post-vector-open Verilator transaction boundary",
            "required negative and mode-confusion controls",
            "crash-boundary counters and invariant",
            "rehearsed/production driver bytes and argv equivalence",
            "shared formal authority authentication path",
        ],
        "claim_boundary": (
            "lifecycle L2 PASS only; no runtime submission, consumption, "
            "retry, replay, resume, or position-2 runtime claim"
        ),
    }
    write_json_exclusive(l2_review_path, l2_review)
    l2_review_record = file_record(l2_review_path)
    write_json_exclusive(
        l2_seal_path,
        {
            "schema_version": 1,
            "kind": "ace3_position2_runtime_wrapper_l2_closure_seal",
            "task_id": rehearsal_root.name,
            "status": "L2_PASS_SUCCESSOR_MINT_PERMITTED",
            "closure_packet": closure_record,
            "closure_seal": closure_seal_record,
            "review_request": l2_request_record,
            "independent_l2_review_binding": l2_review_record,
            "independent_review_source": review_source_record,
            "runtime_action_counts": {
                "submissions": 0,
                "authority_consumptions": 0,
                "retries": 0,
                "replays": 0,
                "resumes": 0,
            },
        },
    )

    FORMAL_SUCCESSOR_PARENT.mkdir(parents=True, exist_ok=True)
    successor_root.mkdir(mode=0o700)
    immutable = successor_root / "immutable"
    outer = successor_root / "outer"
    review = successor_root / "review"
    for directory in (immutable, outer, review):
        directory.mkdir(mode=0o700)
        fsync_directory(directory.parent)

    authority_path = immutable / "successor-authority.json"
    successor_bindings_path = immutable / "successor-bindings.json"
    runtime_root = successor_root / "runtime"
    authority_review_path = review / "authority-l2-review.json"
    zero_state_path = review / "zero-state.json"
    authority_review_request_path = review / "authority-l2-review-request.json"
    successor_seal_path = successor_root / "successor-seal.json"
    task_id = successor_id
    child_command = shlex.join(
        [
            bindings["toolchain"]["argus_python"]["path"],
            "-B",
            str(production_driver),
            "execute",
            "--bindings",
            str(successor_bindings_path),
            "--authority",
            str(authority_path),
        ]
    )
    successor_runtime_paths = {
        "runtime_root": str(runtime_root),
        "consumption_sentinel": str(runtime_root / "consumption.json"),
        "payload_invocation": str(runtime_root / "payload-invocation.json"),
        "output_root": str(runtime_root / "output"),
        "terminal": str(runtime_root / "terminal.json"),
        "transaction_boundary": str(runtime_root / "transaction-boundary.json"),
        "scratch_root": str(successor_root / "scratch"),
        "pre_submit_review": str(outer / "pre-submit-review.json"),
        "negative_controls": str(successor_root / "evidence/negative-controls.json"),
        "duplicate_stdout": str(outer / "duplicate-submit.stdout"),
        "duplicate_stderr": str(outer / "duplicate-submit.stderr"),
        "duplicate_control": str(successor_root / "evidence/duplicate-control.json"),
        "driver_comparison": str(
            rehearsal_root / "evidence/driver-comparison.json"
        ),
        "copied_evidence_root": str(successor_root / "evidence/runner"),
        "closure_packet": str(closure_path),
        "closure_seal": str(l2_seal_path),
        "l2_review_request": str(authority_review_request_path),
        "l2_review": str(authority_review_path),
    }
    authority = {
        "schema_version": 1,
        "kind": AUTHORITY_KIND,
        "mode": FORMAL_AUTHORITY_MODE,
        "status": "AUTHORIZED_ONCE_UNCONSUMED",
        "task_id": task_id,
        "identity": task_id,
        "route_metadata": ROUTE_METADATA,
        "runtime_driver": template["runtime_driver"],
        "accepted_package": template["accepted_package"],
        "accepted_source_archive": template["accepted_source_archive"],
        "formal_execution_authority": True,
        "runtime_consumable": True,
        "reusable": False,
        "one_shot": {
            "cardinality": 1,
            "current_authority_consumption_count": 0,
            "current_durable_submission_count": 0,
            "current_payload_invocation_count": 0,
            "current_runtime_invocation_count": 0,
            "automatic_retry": False,
            "retry": False,
            "replay": False,
            "resume": False,
        },
        "lifecycle_gate": {
            "required_for_formal_mode": True,
            "closure_packet": closure_record,
            "independent_l2_review": l2_review_record,
        },
        "claim_boundary": "FORMAL_ONE_SHOT_RUNTIME_AUTHORITY",
    }
    write_json_exclusive(authority_path, authority)
    successor_bindings = {
        "schema_version": 1,
        "kind": LIFECYCLE_BINDINGS_KIND,
        "mode": FORMAL_AUTHORITY_MODE,
        "status": "AUTHORIZED_ONCE_UNCONSUMED",
        "task_id": task_id,
        "identity": task_id,
        "route_metadata": ROUTE_METADATA,
        "runtime_driver": template["runtime_driver"],
        "identity_authority": file_record(authority_path),
        "accepted_package": template["accepted_package"],
        "accepted_source_archive": template["accepted_source_archive"],
        "transaction_input_sources": bindings["transaction_input_sources"],
        "toolchain": bindings["toolchain"],
        "active_framework": bindings["active_framework"],
        "formal_state_before": formal_state_snapshot(),
        "submission": {
            "cwd": str(ROOT),
            "description": compact_json(ROUTE_METADATA),
            "child_command": child_command,
            "mode": "direct",
            "timeout_seconds": 7200,
        },
        "runtime_paths": successor_runtime_paths,
    }
    write_json_exclusive(successor_bindings_path, successor_bindings)
    authenticate_static(
        successor_bindings,
        authority_path,
        production_driver,
    )

    registry_receipt, registry_logs = task_paths(successor_bindings)
    zero_namespaces = {
        "authority_review": authority_review_path,
        "durable_runner_receipt": registry_receipt,
        "durable_runner_logs": registry_logs,
        "pre_submit_review": Path(successor_runtime_paths["pre_submit_review"]),
        "runtime_root": runtime_root,
        "consumption_sentinel": Path(
            successor_runtime_paths["consumption_sentinel"]
        ),
        "payload_invocation": Path(successor_runtime_paths["payload_invocation"]),
        "terminal": Path(successor_runtime_paths["terminal"]),
        "transaction_boundary": Path(
            successor_runtime_paths["transaction_boundary"]
        ),
    }
    for label, path in zero_namespaces.items():
        require_absent(path, label.replace("_", " "))
    zero_state = {
        "schema_version": 1,
        "kind": "ace3_position2_formal_successor_authority_zero_state",
        "task_id": task_id,
        "counts": {
            "authority_consumptions": 0,
            "durable_runner_submissions": 0,
            "payload_invocations": 0,
            "runtime_invocations": 0,
            "terminal_manifests": 0,
        },
        "namespaces": {
            label: {"path": str(path), "state": "ABSENT"}
            for label, path in zero_namespaces.items()
        },
    }
    write_json_exclusive(zero_state_path, zero_state)
    write_json_exclusive(
        authority_review_request_path,
        {
            "schema_version": 1,
            "kind": (
                "ace3_position2_formal_successor_authority_review_request"
            ),
            "task_id": task_id,
            "execution_during_review": "PROHIBITED",
            "required_verdict": (
                "L2_PASS_UNCONSUMED_READY_FOR_SEPARATE_RUNTIME_PASS_TASK"
            ),
            "review_path": str(authority_review_path),
            "authority": file_record(authority_path),
            "bindings": file_record(successor_bindings_path),
            "runtime_driver": template["runtime_driver"],
            "accepted_package": template["accepted_package"],
            "lifecycle_l2_closure_seal": file_record(l2_seal_path),
            "zero_state": file_record(zero_state_path),
            "required_checks": [
                "fresh identity and exact one-shot cardinality",
                "accepted v7 package and source archive bindings",
                "repaired production driver binding",
                "sealed independent lifecycle L2 PASS binding",
                "exact active framework and route metadata",
                "zero submission, consumption, payload, runtime, and terminal state",
            ],
        },
    )
    minted = [
        path
        for path in FORMAL_SUCCESSOR_PARENT.iterdir()
        if path.is_dir() and path.name.startswith("ace3-position2-fresh-v9-")
    ]
    require(
        minted == [successor_root],
        "formal successor namespace does not contain exactly one v9 identity",
    )
    write_json_exclusive(
        successor_seal_path,
        {
            "schema_version": 1,
            "kind": "ace3_position2_formal_successor_authority_seal",
            "task_id": task_id,
            "status": "AWAITING_INDEPENDENT_AUTHORITY_L2_REVIEW",
            "fresh_successor_identity_count": 1,
            "authority": file_record(authority_path),
            "bindings": file_record(successor_bindings_path),
            "runtime_driver": template["runtime_driver"],
            "accepted_package": template["accepted_package"],
            "lifecycle_l2_closure_seal": file_record(l2_seal_path),
            "zero_state": file_record(zero_state_path),
            "authority_review_request": file_record(
                authority_review_request_path
            ),
            "authority_review_output": str(authority_review_path),
            "static_authentication": "PASS",
            "runtime_action_counts": zero_state["counts"],
            "claim_boundary": (
                "formal authority minted and review routed only; no submit, "
                "consume, retry, replay, resume, or runtime PASS"
            ),
        },
    )
    return {
        "task_id": task_id,
        "lifecycle_l2_closure_seal": file_record(l2_seal_path),
        "successor_authority": file_record(authority_path),
        "successor_bindings": file_record(successor_bindings_path),
        "successor_seal": file_record(successor_seal_path),
        "authority_review_request": file_record(
            authority_review_request_path
        ),
        "authority_review_output": str(authority_review_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--rehearsal-id", required=True)
    for operation in ("pre-submit", "execute", "seal"):
        subparser = subparsers.add_parser(operation)
        subparser.add_argument("--bindings", required=True, type=Path)
        subparser.add_argument("--authority", required=True, type=Path)
    negative_parser = subparsers.add_parser("negative-controls")
    negative_parser.add_argument("--bindings", required=True, type=Path)
    negative_parser.add_argument("--authority", required=True, type=Path)
    duplicate_parser = subparsers.add_parser("record-duplicate")
    duplicate_parser.add_argument("--bindings", required=True, type=Path)
    duplicate_parser.add_argument("--exit-code", required=True, type=int)
    mint_parser = subparsers.add_parser("bind-l2-and-mint")
    mint_parser.add_argument("--rehearsal-root", required=True, type=Path)
    mint_parser.add_argument("--review-source", required=True, type=Path)
    mint_parser.add_argument("--successor-id", required=True)
    arguments = parser.parse_args()
    try:
        if arguments.operation == "prepare":
            result = prepare(arguments.rehearsal_id)
        elif arguments.operation == "pre-submit":
            result = outer_pre_submit(arguments.bindings, arguments.authority)
        elif arguments.operation == "execute":
            raise SystemExit(
                execute_rehearsal(arguments.bindings, arguments.authority)
            )
        elif arguments.operation == "negative-controls":
            result = run_negative_controls(
                arguments.bindings,
                arguments.authority,
            )
        elif arguments.operation == "record-duplicate":
            result = record_duplicate_control(
                arguments.bindings, arguments.exit_code
            )
        elif arguments.operation == "bind-l2-and-mint":
            result = bind_l2_and_mint(
                arguments.rehearsal_root,
                arguments.review_source,
                arguments.successor_id,
            )
        else:
            result = seal_closure(arguments.bindings, arguments.authority)
        print(compact_json(result))
    except (LifecycleError, OSError) as error:
        raise SystemExit(
            f"POSITION2_WRAPPER_LIFECYCLE_{arguments.operation.upper()}_FAIL: {error}"
        ) from error


if __name__ == "__main__":
    main()
