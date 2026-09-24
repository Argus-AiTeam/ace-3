#!/usr/bin/env python3
"""Run and adjudicate a fresh indexed RTL layer from its sealed predecessor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY = Path("/home/argustest/ace3-argus")
PREDECESSOR = REPOSITORY / "build/model24_layer21_attempt002"
MISSION_CONTEXT = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
    "e6938f43bfda/latest.json"
)
LAYER_INDEX = 22
PREDECESSOR_LAYER_INDEX = 21
OFFICIAL_ATTEMPT_CANDIDATE = 1
TOP_MODULE = "ace3_decoder_layer0_token_engine"
TASK_ID = "ace3-layer22-from-layer21-diagnostic"
PYTHON = Path("/home/argustest/miniconda3/bin/python3")
EXPECTED_TERMINAL = (
    "schema=ace3_decoder_layer_raw_v1 layer_index=22 "
    "natural_terminal=1 exit_code=0 trace_count=46676 "
    "final_count=1792 done_count=2\n"
)
POLICY_SOURCES = (
    REPOSITORY / "ace3/model/layer3_token0_diagnostic.py",
    REPOSITORY / "ace3/model/controller_model24_rtl_cascade.py",
    REPOSITORY / "ace3/model/official_model24_next_token.py",
)
MISSION_NODE_KEY = "layer22-runtime-pass-from-layer21"
MISSION_OBJECTIVE_REQUIRED = (
    "sealed build/model24_layer21_attempt002 as the only predecessor"
)
OUTPUT_DIR = REPOSITORY / "build/diagnose_layer22_from_layer21"


def configure_layer(layer_index: int) -> None:
    global EXPECTED_TERMINAL
    global LAYER_INDEX
    global MISSION_CONTEXT
    global MISSION_NODE_KEY
    global MISSION_OBJECTIVE_REQUIRED
    global OFFICIAL_ATTEMPT_CANDIDATE
    global OUTPUT_DIR
    global PREDECESSOR
    global PREDECESSOR_LAYER_INDEX
    global TASK_ID

    if layer_index == 22:
        return
    if layer_index != 23:
        raise LayerRunError(
            "scope_failure", f"unsupported layer index: {layer_index}"
        )
    LAYER_INDEX = 23
    PREDECESSOR_LAYER_INDEX = 22
    OFFICIAL_ATTEMPT_CANDIDATE = 1
    PREDECESSOR = REPOSITORY / "build/diagnose_layer22_from_layer21"
    MISSION_CONTEXT = Path(
        "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
        "584e7ea085a7/latest.json"
    )
    MISSION_NODE_KEY = "layer23-runtime-pass-from-layer22-continuation"
    MISSION_OBJECTIVE_REQUIRED = (
        "accepted layer22 continuation evidence as predecessor"
    )
    OUTPUT_DIR = REPOSITORY / "build/model24_layer23_attempt001"
    TASK_ID = "ace3-model24-layer23-attempt001"
    EXPECTED_TERMINAL = (
        "schema=ace3_decoder_layer_raw_v1 layer_index=23 "
        "natural_terminal=1 exit_code=0 trace_count=46676 "
        "final_count=1792 done_count=2\n"
    )


def layer_name() -> str:
    return f"layer{LAYER_INDEX}"


def predecessor_name() -> str:
    return f"layer{PREDECESSOR_LAYER_INDEX}"


def evidence_kind(suffix: str) -> str:
    return (
        f"ace3_layer{LAYER_INDEX}_from_layer{PREDECESSOR_LAYER_INDEX}_{suffix}"
    )


class LayerRunError(RuntimeError):
    def __init__(
        self,
        classification: str,
        message: str,
        *,
        mismatch: dict[str, Any] | None = None,
        next_diagnostic_point: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.classification = classification
        self.mismatch = mismatch
        self.next_diagnostic_point = next_diagnostic_point


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise LayerRunError("malformed_evidence", f"JSON object required: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": sha256(path),
        "mode": f"{stat.st_mode & 0o777:04o}",
    }


def require(condition: bool, classification: str, message: str) -> None:
    if not condition:
        raise LayerRunError(classification, message)


def predecessor_snapshot() -> list[dict[str, Any]]:
    return [
        {
            "relative_path": str(path.relative_to(PREDECESSOR)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "mode": f"{path.stat().st_mode & 0o777:04o}",
        }
        for path in sorted(PREDECESSOR.rglob("*"))
        if path.is_file()
    ]


def snapshot_digest(snapshot: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(snapshot)).hexdigest()


def authenticate_predecessor() -> dict[str, Any]:
    require(
        PREDECESSOR.is_dir(),
        "predecessor_authentication_failure",
        f"sealed {predecessor_name()} predecessor directory is missing",
    )
    seal_path = PREDECESSOR / "sealed_output_manifest.json"
    seal = load_json(seal_path)
    status = load_json(PREDECESSOR / "status.json")
    attempt = load_json(PREDECESSOR / "attempt.json")
    if PREDECESSOR_LAYER_INDEX == 21:
        require(
            seal.get("kind")
            == "ace3_model24_layer21_attempt002_sealed_manifest"
            and seal.get("layer_index") == PREDECESSOR_LAYER_INDEX
            and seal.get("attempt") == 2
            and seal.get("status") == "PASS",
            "predecessor_authentication_failure",
            "layer21 attempt002 seal does not declare PASS",
        )
    else:
        require(
            seal.get("kind")
            == "ace3_layer22_from_layer21_sealed_diagnostic"
            and seal.get("layer_index") == PREDECESSOR_LAYER_INDEX
            and seal.get("official_attempt_candidate") == 1
            and seal.get("official_attempt001_authorized") is True
            and seal.get("status") == "PASS",
            "predecessor_authentication_failure",
            "layer22 continuation seal does not declare an accepted PASS",
        )
    artifacts = {
        row["relative_path"]: row
        for row in seal.get("artifacts", [])
        if isinstance(row, dict) and isinstance(row.get("relative_path"), str)
    }
    require(
        len(artifacts) == seal.get("artifact_count"),
        "predecessor_authentication_failure",
        f"{predecessor_name()} seal has duplicate or malformed artifacts",
    )
    for relative, row in artifacts.items():
        path = PREDECESSOR / relative
        require(
            path.is_file()
            and path.stat().st_size == row.get("bytes")
            and sha256(path) == row.get("sha256"),
            "predecessor_authentication_failure",
            f"{predecessor_name()} sealed artifact changed: {relative}",
        )
    if PREDECESSOR_LAYER_INDEX == 21:
        require(
            status.get("status") == "PASS"
            and status.get("natural_terminal") is True
            and status.get("integer_oracle_bit_exact") is True
            and status.get("float_reference_within_tolerance") is True,
            "predecessor_authentication_failure",
            "layer21 attempt002 status is not an accepted numerical PASS",
        )
        require(
            attempt.get("layer_index") == PREDECESSOR_LAYER_INDEX
            and attempt.get("attempt") == 2,
            "predecessor_authentication_failure",
            "layer21 attempt002 identity is malformed",
        )
    else:
        terminal = load_json(PREDECESSOR / "natural_terminal_gate.json")
        integer = load_json(PREDECESSOR / "integer_comparison.json")
        comparison = load_json(PREDECESSOR / "comparison.json")
        adjudication = load_json(PREDECESSOR / "adjudication.json")
        require(
            status.get("status") == "PASS"
            and terminal.get("natural_terminal") is True
            and terminal.get("actual_exit_code") == 0
            and integer.get("bit_exact") is True
            and comparison.get("status") == "PASS"
            and comparison.get("failure_count") == 0
            and adjudication.get("status") == "PASS"
            and adjudication.get("official_attempt001_authorized") is True,
            "predecessor_authentication_failure",
            "layer22 continuation is not an accepted numerical PASS",
        )
        require(
            attempt.get("layer_index") == PREDECESSOR_LAYER_INDEX
            and attempt.get("official_attempt_candidate") == 1,
            "predecessor_authentication_failure",
            "layer22 continuation identity is malformed",
        )
    final_path = PREDECESSOR / "raw/final.hex"
    final_record = artifacts.get("raw/final.hex")
    require(
        final_record is not None
        and sha256(final_path) == final_record.get("sha256"),
        "predecessor_authentication_failure",
        f"{predecessor_name()} sealed output handoff is inconsistent",
    )
    if PREDECESSOR_LAYER_INDEX == 21:
        require(
            status.get("output_raw_sha256") == final_record.get("sha256"),
            "predecessor_authentication_failure",
            "layer21 status does not bind its sealed output handoff",
        )
    else:
        require(
            integer["final"]["produced"]["sha256"]
            == final_record.get("sha256"),
            "predecessor_authentication_failure",
            "layer22 integer comparison does not bind its sealed output",
        )
    return {
        "seal": file_record(seal_path),
        "seal_identity": {
            "kind": seal["kind"],
            "layer_index": seal["layer_index"],
            "attempt": seal.get("attempt"),
            "official_attempt_candidate": seal.get(
                "official_attempt_candidate"
            ),
            "status": seal["status"],
            "sealed_at_utc": seal["sealed_at_utc"],
            "artifact_count": seal["artifact_count"],
        },
        "status": file_record(PREDECESSOR / "status.json"),
        "attempt": file_record(PREDECESSOR / "attempt.json"),
        "input_handoff": {
            **file_record(final_path),
            "source_layer_index": PREDECESSOR_LAYER_INDEX,
            "consumer_layer_index": LAYER_INDEX,
            "complete_rows": 1792,
        },
        "source_attempt": attempt,
    }


def source_records(source_attempt: dict[str, Any]) -> dict[str, Any]:
    predecessor_source = source_attempt["source"]
    source_root = Path(
        predecessor_source.get(
            "immutable_source_root", predecessor_source.get("root")
        )
    )
    model_hashes = predecessor_source.get(
        "immutable_model_hashes", predecessor_source.get("model_hashes")
    )
    rtl_hashes = predecessor_source.get(
        "immutable_rtl_and_testbench_hashes",
        predecessor_source.get("rtl_and_testbench_hashes"),
    )
    require(
        isinstance(model_hashes, dict) and isinstance(rtl_hashes, dict),
        "predecessor_authentication_failure",
        "predecessor source hash bindings are missing",
    )
    immutable_model = {
        relative: sha256(source_root / relative)
        for relative in model_hashes
    }
    immutable_rtl = {
        relative: sha256(source_root / relative)
        for relative in rtl_hashes
    }
    return {
        "runner": file_record(Path(__file__).resolve()),
        "policy_oracle": {
            str(path): file_record(path) for path in POLICY_SOURCES
        },
        "immutable_source_root": str(source_root),
        "immutable_model_hashes": immutable_model,
        "immutable_rtl_and_testbench_hashes": immutable_rtl,
        "makefile": file_record(source_root / "Makefile"),
    }


def version(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.splitlines()[0]


def prepare(output_dir: Path) -> None:
    require(
        not output_dir.exists(),
        "freshness_failure",
        f"diagnostic output already exists: {output_dir}",
    )
    output_dir.mkdir(parents=True, mode=0o700)
    (output_dir / "tmp").mkdir(mode=0o700)
    predecessor = authenticate_predecessor()
    source_attempt = predecessor.pop("source_attempt")
    source = source_records(source_attempt)
    source_root = Path(source["immutable_source_root"])
    checkpoint = Path(source_attempt["official_model"]["checkpoint"]["path"])
    tensor_map = Path(source_attempt["oracle"]["tensor_map"]["path"])
    bindings = Path(source_attempt["oracle"]["bindings"]["path"])
    snapshot = predecessor_snapshot()
    write_json(
        output_dir / "predecessor_before.json",
        {
            "schema_version": 1,
            "root": str(PREDECESSOR),
            "file_count": len(snapshot),
            "ordered_file_set_sha256": snapshot_digest(snapshot),
            "files": snapshot,
        },
    )
    mission = load_json(MISSION_CONTEXT)
    require(
        mission.get("kind") == "mission_context"
        and mission.get("node_key") == MISSION_NODE_KEY
        and MISSION_OBJECTIVE_REQUIRED in (mission.get("objective") or ""),
        "mission_authentication_failure",
        f"live mission does not authorize this {layer_name()} runtime attempt",
    )
    compile_argv = [
        "make",
        "--no-print-directory",
        "model24-rtl-layer-compile",
        f"MODEL24_RTL_LAYER_INDEX={LAYER_INDEX}",
        "MODEL24_RTL_ACCURATE_SILU=1",
        f"MODEL24_RTL_CASCADE_DIR={output_dir}",
    ]
    binary = (
        output_dir
        / f"compiled/layer{LAYER_INDEX}/obj_dir/V{TOP_MODULE}"
    )
    simulator_argv = [
        str(binary),
        "--layer-index",
        str(LAYER_INDEX),
        "--vector-dir",
        str(output_dir / "vectors"),
        "--raw-dir",
        str(output_dir / "raw"),
        "--progress-interval",
        "1000000",
    ]
    meminfo = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            meminfo[key] = value.strip()
    write_json(
        output_dir / "attempt.json",
        {
            "schema_version": 1,
            "kind": evidence_kind("runtime_attempt"),
            "status": "PREPARED_NOT_EXECUTED",
            "layer_index": LAYER_INDEX,
            "official_attempt_candidate": OFFICIAL_ATTEMPT_CANDIDATE,
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "mission_context": file_record(MISSION_CONTEXT),
            "consumed_predecessor": predecessor,
            "official_model": source_attempt["official_model"],
            "numeric_profile": source_attempt["official_model"]["profile"],
            "reference_policy": source_attempt.get(
                "reference_policy",
                source_attempt["oracle"].get("reference_policy"),
            ),
            "oracle": {
                "integer": (
                    "fresh model24_execution_oracle."
                    "materialize_indexed_decoder_vectors"
                ),
                "float": (
                    "independent PyTorch CPU float64 dequantized-AWQ with "
                    "round-to-nearest-even FP16 after each implemented stage"
                ),
                "checkpoint": file_record(checkpoint),
                "tensor_map": file_record(tensor_map),
                "bindings": file_record(bindings),
                "absolute_tolerance": 0.125,
                "relative_tolerance": 0.001,
                "max_ulp_distance": 1,
            },
            "source": source,
            "interface": {
                "top_module": TOP_MODULE,
                "parameters": {"LAYER_INDEX": LAYER_INDEX, "ACCURATE_SILU": 1},
                "cycle_origin": 0,
                "compatibility_aliases": False,
            },
            "execution": {
                "compile_argv": compile_argv,
                "simulator_argv": simulator_argv,
                "simulator_timeout_seconds": 2400,
                "task_id": TASK_ID,
            },
            "freshness": {
                "output_directory_was_absent": True,
                "fresh_vector_generation_required": True,
                "fresh_rtl_compile_required": True,
                "fresh_cycle_zero_simulation_required": True,
                f"official_{layer_name()}_attempt_outputs_consumed": False,
            },
            "runtime": {
                "python": version([str(PYTHON), "--version"]),
                "verilator": version(["verilator", "--version"]),
                "make": version(["make", "--version"]),
                "g++": version(["g++", "--version"]),
                "kernel": platform.platform(),
                "hostname": socket.gethostname(),
                "logical_cpus": os.cpu_count(),
                "memory": meminfo,
                "thread_limits": {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
            },
            "claim_boundary": {
                "scope": (
                    f"computer-local {layer_name()} RTL simulation attempt"
                ),
                "official_attempt": "not run",
                "full_model": "not claimed",
                "dialogue": "not run",
                "synthesis": "not run",
                "ppa": "not measured",
                "fpga": "not run",
            },
        },
    )
    print(
        f"LAYER{LAYER_INDEX}_RUNTIME_PREPARED "
        f"predecessor_seal_sha256={predecessor['seal']['sha256']} "
        f"input_sha256={predecessor['input_handoff']['sha256']}"
    )


def verify_preflight(output_dir: Path, attempt: dict[str, Any]) -> None:
    require(
        attempt.get("kind") == evidence_kind("runtime_attempt")
        and attempt.get("status") == "PREPARED_NOT_EXECUTED"
        and attempt.get("layer_index") == LAYER_INDEX
        and attempt.get("official_attempt_candidate")
        == OFFICIAL_ATTEMPT_CANDIDATE,
        "prelaunch_contract_failure",
        f"prepared {layer_name()} runtime identity is invalid",
    )
    current = authenticate_predecessor()
    require(
        current["seal"]["sha256"]
        == attempt["consumed_predecessor"]["seal"]["sha256"]
        and current["input_handoff"]["sha256"]
        == attempt["consumed_predecessor"]["input_handoff"]["sha256"],
        "predecessor_authentication_failure",
        f"sealed {predecessor_name()} predecessor changed after preparation",
    )
    baseline = load_json(output_dir / "predecessor_before.json")
    snapshot = predecessor_snapshot()
    require(
        snapshot == baseline.get("files")
        and snapshot_digest(snapshot)
        == baseline.get("ordered_file_set_sha256"),
        "predecessor_preservation_failure",
        f"sealed {predecessor_name()} predecessor changed after baseline capture",
    )
    require(
        sha256(Path(__file__).resolve())
        == attempt["source"]["runner"]["sha256"],
        "source_authentication_failure",
        f"{layer_name()} runtime runner changed after preparation",
    )
    for path, record in attempt["source"]["policy_oracle"].items():
        require(
            sha256(Path(path)) == record["sha256"],
            "source_authentication_failure",
            f"FP16 policy source changed after preparation: {path}",
        )
    source_root = Path(attempt["source"]["immutable_source_root"])
    for relative, digest in attempt["source"]["immutable_model_hashes"].items():
        require(
            sha256(source_root / relative) == digest,
            "source_authentication_failure",
            f"immutable model source changed: {relative}",
        )
    for relative, digest in attempt["source"][
        "immutable_rtl_and_testbench_hashes"
    ].items():
        require(
            sha256(source_root / relative) == digest,
            "source_authentication_failure",
            f"immutable RTL source changed: {relative}",
        )
    for key in ("checkpoint", "tensor_map", "bindings"):
        record = attempt["oracle"][key]
        require(
            sha256(Path(record["path"])) == record["sha256"],
            "input_authentication_failure",
            f"authenticated oracle input changed: {key}",
        )


def phase_record(
    output_dir: Path,
    name: str,
    started_ns: int,
    *,
    status: str,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ended_ns = time.time_ns()
    record = {
        "phase": name,
        "status": status,
        "started_unix_ns": started_ns,
        "ended_unix_ns": ended_ns,
        "wall_seconds": (ended_ns - started_ns) / 1_000_000_000,
    }
    if detail:
        record.update(detail)
    write_json(output_dir / f"{name}.json", record)
    return record


def run_process(
    output_dir: Path,
    name: str,
    argv: list[str],
    cwd: Path,
    timeout: int,
) -> tuple[subprocess.CompletedProcess[bytes] | None, dict[str, Any]]:
    started_ns = time.time_ns()
    stdout_path = output_dir / f"{name}.stdout"
    stderr_path = output_dir / f"{name}.stderr"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, phase_record(
                output_dir,
                name,
                started_ns,
                status="TIMEOUT",
                detail={
                    "argv": argv,
                    "cwd": str(cwd),
                    "timeout_seconds": timeout,
                    "exit_code": None,
                    "stdout": file_record(stdout_path),
                    "stderr": file_record(stderr_path),
                },
            )
    return completed, phase_record(
        output_dir,
        name,
        started_ns,
        status="PASS" if completed.returncode == 0 else "FAIL",
        detail={
            "argv": argv,
            "cwd": str(cwd),
            "timeout_seconds": timeout,
            "exit_code": completed.returncode,
            "stdout": file_record(stdout_path),
            "stderr": file_record(stderr_path),
        },
    )


def compare_hex(produced_path: Path, oracle_path: Path) -> dict[str, Any]:
    produced = produced_path.read_bytes()
    oracle = oracle_path.read_bytes()
    produced_lines = produced.splitlines()
    oracle_lines = oracle.splitlines()
    first_mismatch = None
    for index, (actual, expected) in enumerate(
        zip(produced_lines, oracle_lines, strict=False)
    ):
        if actual != expected:
            first_mismatch = {
                "line_index": index,
                "produced": actual.decode("ascii", "replace"),
                "oracle": expected.decode("ascii", "replace"),
            }
            break
    if first_mismatch is None and len(produced_lines) != len(oracle_lines):
        first_mismatch = {
            "line_index": min(len(produced_lines), len(oracle_lines)),
            "produced": (
                produced_lines[len(oracle_lines)].decode("ascii", "replace")
                if len(produced_lines) > len(oracle_lines)
                else None
            ),
            "oracle": (
                oracle_lines[len(produced_lines)].decode("ascii", "replace")
                if len(oracle_lines) > len(produced_lines)
                else None
            ),
        }
    return {
        "bit_exact": produced == oracle,
        "produced": file_record(produced_path),
        "oracle": file_record(oracle_path),
        "produced_line_count": len(produced_lines),
        "oracle_line_count": len(oracle_lines),
        "first_mismatch": first_mismatch,
    }


def trace_diagnostic_point(mismatch: dict[str, Any]) -> dict[str, Any]:
    value = mismatch.get("produced") or mismatch.get("oracle")
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{16}", value) is None:
        return {"trace_line_index": mismatch["line_index"]}
    return {
        "trace_line_index": mismatch["line_index"],
        "token_index": int(value[0:2], 16),
        "position": int(value[2:6], 16),
        "stage_index": int(value[6:8], 16),
        "element_index": int(value[8:12], 16),
    }


def gate_natural_terminal(output_dir: Path) -> dict[str, Any]:
    terminal_path = output_dir / "raw/terminal.txt"
    simulation_path = output_dir / "simulation.json"
    require(
        terminal_path.is_file() and simulation_path.is_file(),
        "simulator_terminal_failure",
        "simulator terminal or process record is missing",
    )
    terminal = terminal_path.read_text(encoding="ascii")
    require(
        terminal == EXPECTED_TERMINAL
        and re.fullmatch(re.escape(EXPECTED_TERMINAL), terminal) is not None,
        "simulator_terminal_failure",
        (
            "simulator did not publish the exact unambiguous "
            f"{layer_name()} terminal"
        ),
    )
    simulation = load_json(simulation_path)
    require(
        simulation.get("status") == "PASS"
        and simulation.get("exit_code") == 0,
        "simulator_terminal_failure",
        "simulator process record is not a zero-exit PASS",
    )
    return {
        "terminal": file_record(terminal_path),
        "simulation": file_record(simulation_path),
        "actual_exit_code": simulation["exit_code"],
        "natural_terminal": True,
    }


def ordered_fp16_bits(bits: np.ndarray) -> np.ndarray:
    signed = bits.astype(np.int64)
    return np.where(
        (signed & 0x8000) != 0,
        0x8000 - (signed & 0x7FFF),
        0x8000 + signed,
    )


def compare_stage(
    diagnostic: Any,
    produced_bits: np.ndarray,
    reference: np.ndarray,
    *,
    token: int,
    stage: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_ulp_distance: int,
) -> dict[str, Any]:
    bits = np.asarray(produced_bits, dtype="<u2").reshape(-1)
    produced = diagnostic.f16_values(bits)
    expected = np.asarray(reference, dtype=np.float64).reshape(-1)
    require(
        produced.shape == expected.shape,
        "float_reference_shape_failure",
        "FP16 stage comparison shape mismatch",
    )
    require(
        np.all(np.isfinite(produced)) and np.all(np.isfinite(expected)),
        "float_reference_nonfinite",
        "FP16 stage comparison contains non-finite values",
    )
    rounded = expected.astype("<f2")
    require(
        np.all(np.isfinite(rounded)),
        "float_reference_nonfinite",
        "independent reference is outside finite FP16",
    )
    reference_bits = rounded.view("<u2")
    difference = np.abs(produced - expected)
    relative = difference / np.maximum(
        np.abs(expected), float(np.finfo(np.float16).tiny)
    )
    ulp = np.abs(
        ordered_fp16_bits(bits) - ordered_fp16_bits(reference_bits)
    )
    accepted = (difference <= absolute_tolerance) | (
        (relative < relative_tolerance) & (ulp <= max_ulp_distance)
    )
    failed = np.flatnonzero(~accepted)
    first_failure = None
    if failed.size:
        index = int(failed[0])
        first_failure = {
            "token_index": token,
            "stage_index": stage,
            "stage": diagnostic.STAGES[stage],
            "element_index": index,
            "produced_bits": f"{int(bits[index]):04x}",
            "produced_value": float(produced[index]),
            "reference_value": float(expected[index]),
            "absolute_error": float(difference[index]),
            "relative_error": float(relative[index]),
            "ulp_distance": int(ulp[index]),
        }
    return {
        "token_index": token,
        "stage_index": stage,
        "stage": diagnostic.STAGES[stage],
        "records": int(bits.size),
        "failure_count": int(failed.size),
        "within_tolerance": bool(accepted.all()),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "max_relative_error": float(relative.max()),
        "max_ulp_distance": int(ulp.max()),
        "first_failure": first_failure,
    }


def compare_float_policy(output_dir: Path) -> int:
    attempt = load_json(output_dir / "attempt.json")
    verify_preflight(output_dir, attempt)
    gate = gate_natural_terminal(output_dir)
    integer = load_json(output_dir / "integer_comparison.json")
    require(
        integer.get("bit_exact") is True,
        "integer_oracle_mismatch",
        "integer oracle must pass before the FP16-policy oracle opens",
    )

    sys.path.insert(0, str(REPOSITORY / "ace3/model"))
    import torch
    import layer3_token0_diagnostic as diagnostic

    handoff_path = Path(
        attempt["consumed_predecessor"]["input_handoff"]["path"]
    )
    expected_handoff_sha256 = attempt["consumed_predecessor"][
        "input_handoff"
    ]["sha256"]
    handoff, handoff_binding = diagnostic.authenticate_predecessor_handoff(
        handoff_path,
        layer_index=LAYER_INDEX,
        expected_predecessor_layer=PREDECESSOR_LAYER_INDEX,
        expected_handoff_sha256=expected_handoff_sha256,
    )
    primary = diagnostic.load_trace(output_dir / "raw/trace.hex")
    final, _ = diagnostic.load_two_token_handoff(
        output_dir / "raw/final.hex",
        expected_sha256=sha256(output_dir / "raw/final.hex"),
    )
    for token in range(2):
        require(
            primary[token][18].tolist() == final[token],
            "trace_final_consistency_failure",
            f"token {token} final trace and final output differ",
        )
    tensors, layer_binding, tensor_hashes = diagnostic.load_tensors(
        Path(attempt["oracle"]["checkpoint"]["path"]),
        Path(attempt["oracle"]["tensor_map"]["path"]),
        LAYER_INDEX,
    )
    torch.set_num_threads(1)
    reference = diagnostic.independent_reference(
        torch.from_numpy(
            diagnostic.f16_values(np.asarray(handoff, dtype="<u2"))
        ),
        tensors,
        LAYER_INDEX,
        reference_policy="w4a16_fp16_interstage",
    )
    comparisons = [
        compare_stage(
            diagnostic,
            primary[token][stage],
            reference[token][stage],
            token=token,
            stage=stage,
            absolute_tolerance=attempt["oracle"]["absolute_tolerance"],
            relative_tolerance=attempt["oracle"]["relative_tolerance"],
            max_ulp_distance=attempt["oracle"]["max_ulp_distance"],
        )
        for token in range(2)
        for stage in range(len(diagnostic.STAGES))
    ]
    first_failure = next(
        (
            row["first_failure"]
            for row in comparisons
            if row["first_failure"] is not None
        ),
        None,
    )
    failure_count = sum(row["failure_count"] for row in comparisons)
    report = {
        "schema_version": 1,
        "kind": evidence_kind("fp16_interstage_policy_comparison"),
        "layer_index": LAYER_INDEX,
        "official_attempt_candidate": OFFICIAL_ATTEMPT_CANDIDATE,
        "status": "PASS" if failure_count == 0 else "FAIL",
        "within_tolerance": failure_count == 0,
        "failure_count": failure_count,
        "first_material_mismatch": first_failure,
        "next_diagnostic_point": (
            None
            if first_failure is None
            else {
                "token_index": first_failure["token_index"],
                "stage_index": first_failure["stage_index"],
                "stage": first_failure["stage"],
                "element_index": first_failure["element_index"],
            }
        ),
        "max_abs_error": max(row["max_abs_error"] for row in comparisons),
        "mean_abs_error": float(
            np.mean([row["mean_abs_error"] for row in comparisons])
        ),
        "max_relative_error": max(
            row["max_relative_error"] for row in comparisons
        ),
        "max_ulp_distance": max(
            row["max_ulp_distance"] for row in comparisons
        ),
        "reference_policy": {
            "name": "w4a16_fp16_interstage",
            "activation_boundaries": (
                "round-to-nearest-even FP16 after every implemented stage"
            ),
            "absolute_tolerance": attempt["oracle"]["absolute_tolerance"],
            "relative_tolerance": attempt["oracle"]["relative_tolerance"],
            "max_ulp_distance": attempt["oracle"]["max_ulp_distance"],
        },
        "natural_terminal_gate": gate,
        "input_handoff": handoff_binding,
        "checkpoint": attempt["oracle"]["checkpoint"],
        "tensor_map": attempt["oracle"]["tensor_map"],
        "layer_binding": layer_binding,
        "consumed_tensor_count": len(tensor_hashes),
        "consumed_tensors": tensor_hashes,
        "stage_comparisons": comparisons,
        "final_hidden_comparisons": [
            row for row in comparisons if row["stage"] == "mlp_residual"
        ],
        "source_bindings": attempt["source"]["policy_oracle"],
    }
    write_json(output_dir / "comparison.json", report)
    print(
        f"LAYER{LAYER_INDEX}_FP16_INTERSTAGE_{report['status']} "
        f"failure_count={failure_count}"
    )
    return 0 if failure_count == 0 else 1


def preservation_record(output_dir: Path) -> dict[str, Any]:
    baseline = load_json(output_dir / "predecessor_before.json")
    current = predecessor_snapshot()
    identical = (
        current == baseline.get("files")
        and snapshot_digest(current) == baseline.get("ordered_file_set_sha256")
    )
    return {
        "schema_version": 1,
        "status": "PASS" if identical else "FAIL",
        "predecessor": str(PREDECESSOR),
        "file_count": len(current),
        "before_after_identical": identical,
        "before_ordered_file_set_sha256": baseline.get(
            "ordered_file_set_sha256"
        ),
        "after_ordered_file_set_sha256": snapshot_digest(current),
        "baseline": file_record(output_dir / "predecessor_before.json"),
    }


def write_output_manifest(output_dir: Path) -> None:
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if (
            not path.is_file()
            or path.name == "output_manifest.json"
            or ".argus_subagents" in path.parts
        ):
            continue
        record = file_record(path)
        record["relative_path"] = str(path.relative_to(output_dir))
        record.pop("path")
        artifacts.append(record)
    write_json(
        output_dir / "output_manifest.json",
        {
            "schema_version": 1,
            "kind": evidence_kind("output_manifest"),
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "excluded_live_runner_namespace": ".argus_subagents",
        },
    )


def finish(
    output_dir: Path,
    started_ns: int,
    *,
    status: str,
    classification: str | None,
    diagnosis: str,
    mismatch: dict[str, Any] | None = None,
    next_diagnostic_point: dict[str, Any] | None = None,
) -> int:
    preservation = preservation_record(output_dir)
    if preservation["status"] != "PASS":
        status = "FAIL"
        classification = "predecessor_preservation_failure"
        diagnosis = (
            f"sealed {predecessor_name()} predecessor changed during "
            f"{layer_name()} execution"
        )
    write_json(output_dir / "preservation.json", preservation)
    ended_ns = time.time_ns()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write_json(
        output_dir / "status.json",
        {
            "schema_version": 1,
            "kind": evidence_kind("runtime_status"),
            "status": status,
            "layer_index": LAYER_INDEX,
            "official_attempt_candidate": OFFICIAL_ATTEMPT_CANDIDATE,
            "official_attempt001_authorized": status == "PASS",
            "failure_classification": classification,
            "diagnosis": diagnosis,
            "first_material_mismatch": mismatch,
            "next_diagnostic_point": next_diagnostic_point,
            "started_unix_ns": started_ns,
            "ended_unix_ns": ended_ns,
            "wall_seconds": (ended_ns - started_ns) / 1_000_000_000,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "max_rss_kib": usage.ru_maxrss,
            "claim_boundary": {
                "demonstrated": (
                    f"fresh {layer_name()} RTL execution with independent "
                    "integer and FP16-policy comparisons"
                ),
                "official_attempt": "not run",
                "full_model": "not claimed",
                "dialogue": "not run",
                "synthesis": "not run",
                "ppa": "not measured",
                "fpga": "not run",
            },
        },
    )
    write_output_manifest(output_dir)
    print(
        f"LAYER{LAYER_INDEX}_RUNTIME_{status} "
        f"classification={classification or 'none'} diagnosis={diagnosis}"
    )
    return 0 if status == "PASS" else 1


def execute(output_dir: Path) -> int:
    started_ns = time.time_ns()
    try:
        attempt = load_json(output_dir / "attempt.json")
        require(
            not (output_dir / "execution_started.json").exists(),
            "freshness_failure",
            f"{layer_name()} runtime execution already started",
        )
        verify_preflight(output_dir, attempt)
        write_json(
            output_dir / "execution_started.json",
            {
                "status": "RUNNING",
                "attempt_sha256": sha256(output_dir / "attempt.json"),
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "layer_index": LAYER_INDEX,
                "cycle_origin": 0,
            },
        )
        source_root = Path(attempt["source"]["immutable_source_root"])
        sys.path.insert(0, str(source_root / "ace3/model"))
        from controller_model24_rtl_cascade import (
            authenticate_compiled_binary,
            build_layer_compile_argv,
        )
        from model24_execution_oracle import (
            materialize_indexed_decoder_vectors,
        )

        vector_started_ns = time.time_ns()
        vector_manifest = materialize_indexed_decoder_vectors(
            Path(attempt["oracle"]["checkpoint"]["path"]),
            Path(attempt["oracle"]["tensor_map"]["path"]),
            Path(attempt["consumed_predecessor"]["input_handoff"]["path"]),
            output_dir / "vectors",
            layer_index=LAYER_INDEX,
            expected_handoff_sha256=attempt["consumed_predecessor"][
                "input_handoff"
            ]["sha256"],
            accurate_silu=True,
        )
        require(
            vector_manifest.get("layer_index") == LAYER_INDEX
            and vector_manifest.get("layer_binding", {}).get("layer_id")
            == LAYER_INDEX,
            "oracle_materialization_failure",
            f"fresh vectors are not bound to numeric layer {LAYER_INDEX}",
        )
        phase_record(
            output_dir,
            "vector_generation",
            vector_started_ns,
            status="PASS",
            detail={
                "layer_binding": vector_manifest["layer_binding"],
                "boundary_manifest": file_record(
                    output_dir / "vectors/boundary_manifest.json"
                ),
                "fresh_input": file_record(output_dir / "vectors/inputs.hex"),
                "fresh_integer_oracle_trace": file_record(
                    output_dir / "vectors/trace.hex"
                ),
                "fresh_integer_oracle_final": file_record(
                    output_dir / "vectors/final.hex"
                ),
            },
        )

        compile_argv = build_layer_compile_argv(
            source_root, output_dir, LAYER_INDEX, output_dir
        )
        require(
            compile_argv == attempt["execution"]["compile_argv"],
            "prelaunch_contract_failure",
            (
                "runtime compile command differs from the prepared "
                f"{layer_name()} command"
            ),
        )
        compile_started_ns = time.time_ns()
        compile_result, _ = run_process(
            output_dir, "compile", compile_argv, source_root, 600
        )
        if compile_result is None:
            return finish(
                output_dir,
                started_ns,
                status="TIMEOUT",
                classification="compile_timeout",
                diagnosis=(
                    f"fresh {layer_name()} Verilator compilation exceeded "
                    "600 seconds"
                ),
                next_diagnostic_point={"phase": "compile"},
            )
        require(
            compile_result.returncode == 0,
            "compile_failure",
            f"fresh {layer_name()} compile exited {compile_result.returncode}",
        )
        binary = Path(attempt["execution"]["simulator_argv"][0])
        compiled_binary = authenticate_compiled_binary(
            binary, compile_started_ns
        )
        write_json(
            output_dir / "compiled_binary.json",
            {
                "top_module": TOP_MODULE,
                "parameters": {"LAYER_INDEX": LAYER_INDEX, "ACCURATE_SILU": 1},
                **compiled_binary,
            },
        )

        raw_dir = output_dir / "raw"
        raw_dir.mkdir(mode=0o700)
        simulation_result, _ = run_process(
            output_dir,
            "simulation",
            attempt["execution"]["simulator_argv"],
            source_root,
            attempt["execution"]["simulator_timeout_seconds"],
        )
        if simulation_result is None:
            return finish(
                output_dir,
                started_ns,
                status="TIMEOUT",
                classification="simulator_timeout",
                diagnosis=(
                    f"fresh {layer_name()} simulator did not reach a natural "
                    "terminal "
                    "within 2400 seconds; the performance cause is unclassified"
                ),
                next_diagnostic_point={"phase": "simulation_phase_timing"},
            )
        require(
            simulation_result.returncode == 0,
            "simulator_runtime_failure",
            (
                f"fresh {layer_name()} simulator exited "
                f"{simulation_result.returncode}"
            ),
        )
        gate = gate_natural_terminal(output_dir)
        write_json(output_dir / "natural_terminal_gate.json", gate)

        trace_comparison = compare_hex(
            raw_dir / "trace.hex", output_dir / "vectors/trace.hex"
        )
        final_comparison = compare_hex(
            raw_dir / "final.hex", output_dir / "vectors/final.hex"
        )
        integer = {
            "schema_version": 1,
            "kind": evidence_kind("integer_oracle_comparison"),
            "layer_index": LAYER_INDEX,
            "independent_oracle": (
                "fresh model24_execution_oracle materialization from the "
                f"authenticated sealed {predecessor_name()} final handoff"
            ),
            "trace": trace_comparison,
            "final": final_comparison,
            "bit_exact": (
                trace_comparison["bit_exact"] and final_comparison["bit_exact"]
            ),
        }
        write_json(output_dir / "integer_comparison.json", integer)
        if not integer["bit_exact"]:
            mismatch = (
                trace_comparison["first_mismatch"]
                if not trace_comparison["bit_exact"]
                else final_comparison["first_mismatch"]
            )
            next_point = (
                trace_diagnostic_point(mismatch)
                if not trace_comparison["bit_exact"]
                else {
                    "token_index": mismatch["line_index"] // 896,
                    "stage": "mlp_residual",
                    "element_index": mismatch["line_index"] % 896,
                }
            )
            raise LayerRunError(
                "rtl_integer_oracle_mismatch",
                f"{layer_name()} RTL differs from the fresh integer oracle",
                mismatch=mismatch,
                next_diagnostic_point=next_point,
            )

        compare_result, _ = run_process(
            output_dir,
            "float_policy",
            [
                str(PYTHON),
                "-B",
                str(Path(__file__).resolve()),
                "compare",
                "--layer-index",
                str(LAYER_INDEX),
                "--output-dir",
                str(output_dir),
            ],
            REPOSITORY,
            600,
        )
        if compare_result is None:
            return finish(
                output_dir,
                started_ns,
                status="TIMEOUT",
                classification="float_policy_timeout",
                diagnosis=(
                    f"independent {layer_name()} FP16-policy oracle exceeded "
                    "600 seconds"
                ),
                next_diagnostic_point={"phase": "float_policy"},
            )
        comparison = (
            load_json(output_dir / "comparison.json")
            if (output_dir / "comparison.json").is_file()
            else None
        )
        if compare_result.returncode != 0 or comparison is None:
            raise LayerRunError(
                "float_reference_tolerance_failure",
                f"{layer_name()} independent FP16-policy comparison failed",
                mismatch=(
                    comparison.get("first_material_mismatch")
                    if comparison
                    else None
                ),
                next_diagnostic_point=(
                    comparison.get("next_diagnostic_point")
                    if comparison
                    else {"phase": "float_policy"}
                ),
            )
        require(
            comparison.get("status") == "PASS"
            and comparison.get("within_tolerance") is True
            and comparison.get("failure_count") == 0,
            "float_reference_tolerance_failure",
            f"{layer_name()} independent FP16-policy report is not PASS",
        )
        write_json(
            output_dir / "record.json",
            {
                "schema_version": 1,
                "kind": evidence_kind("runtime_record"),
                "layer_index": LAYER_INDEX,
                "official_attempt_candidate": OFFICIAL_ATTEMPT_CANDIDATE,
                "predecessor_seal": attempt["consumed_predecessor"]["seal"],
                "input_handoff": attempt["consumed_predecessor"][
                    "input_handoff"
                ],
                "fresh_input": file_record(output_dir / "vectors/inputs.hex"),
                "fresh_integer_oracle_trace": file_record(
                    output_dir / "vectors/trace.hex"
                ),
                "fresh_integer_oracle_final": file_record(
                    output_dir / "vectors/final.hex"
                ),
                "rtl_trace": file_record(raw_dir / "trace.hex"),
                "rtl_final": file_record(raw_dir / "final.hex"),
                "compiled_binary": compiled_binary,
                "integer_comparison": integer,
                "fp16_policy_comparison": comparison,
                "numeric_profile": vector_manifest["numeric_profile"],
                "source": attempt["source"],
            },
        )
        return finish(
            output_dir,
            started_ns,
            status="PASS",
            classification=None,
            diagnosis=(
                f"fresh cycle-zero {layer_name()} RTL reached its natural "
                "terminal, "
                "matched the fresh integer oracle bit-for-bit, and passed the "
                "independent FP16-interstage policy comparison"
            ),
        )
    except LayerRunError as error:
        return finish(
            output_dir,
            started_ns,
            status="FAIL",
            classification=error.classification,
            diagnosis=str(error),
            mismatch=error.mismatch,
            next_diagnostic_point=error.next_diagnostic_point,
        )
    except Exception as error:
        if not (output_dir / "status.json").exists():
            finish(
                output_dir,
                started_ns,
                status="FAIL",
                classification="harness_exception",
                diagnosis=f"{type(error).__name__}: {error}",
                next_diagnostic_point={"phase": "harness"},
            )
        traceback.print_exc()
        return 1


def adjudicate(output_dir: Path) -> None:
    attempt = load_json(output_dir / "attempt.json")
    status = load_json(output_dir / "status.json")
    preservation = preservation_record(output_dir)
    require(
        preservation["status"] == "PASS"
        and load_json(output_dir / "preservation.json").get(
            "before_after_identical"
        )
        is True,
        "predecessor_preservation_failure",
        "sealed predecessor preservation did not pass",
    )
    receipt_path = output_dir / f".argus_subagents/{TASK_ID}.json"
    receipt = load_json(receipt_path)
    expected_exit = 0 if status.get("status") == "PASS" else 1
    require(
        receipt.get("state") in {"done", "failed"}
        and receipt.get("exit_code") == expected_exit,
        "runner_receipt_failure",
        "durable runner receipt is inconsistent with diagnostic status",
    )
    integer = load_json(output_dir / "integer_comparison.json")
    comparison = load_json(output_dir / "comparison.json")
    if status.get("status") == "PASS":
        require(
            integer.get("bit_exact") is True
            and comparison.get("status") == "PASS"
            and comparison.get("failure_count") == 0
            and status.get("official_attempt001_authorized") is True,
            "adjudication_failure",
            "PASS status lacks required integer or FP16-policy evidence",
        )
    write_json(
        output_dir / "adjudication.json",
        {
            "schema_version": 1,
            "kind": evidence_kind("adjudication"),
            "status": status["status"],
            "layer_index": LAYER_INDEX,
            "official_attempt_candidate": OFFICIAL_ATTEMPT_CANDIDATE,
            "official_attempt001_authorized": (
                status["status"] == "PASS"
                and integer.get("bit_exact") is True
                and comparison.get("status") == "PASS"
            ),
            "first_material_mismatch": status.get(
                "first_material_mismatch"
            ),
            "next_diagnostic_point": status.get("next_diagnostic_point"),
            "consumed_predecessor_seal": attempt["consumed_predecessor"][
                "seal"
            ],
            f"fresh_{layer_name()}_inputs": {
                "inputs": file_record(output_dir / "vectors/inputs.hex"),
                "boundary_manifest": file_record(
                    output_dir / "vectors/boundary_manifest.json"
                ),
            },
            f"fresh_{layer_name()}_oracles": {
                "integer_trace": file_record(
                    output_dir / "vectors/trace.hex"
                ),
                "integer_final": file_record(
                    output_dir / "vectors/final.hex"
                ),
                "integer_comparison": file_record(
                    output_dir / "integer_comparison.json"
                ),
                "fp16_policy_comparison": file_record(
                    output_dir / "comparison.json"
                ),
            },
            "source": attempt["source"],
            "predecessor_preservation": preservation,
            "durable_runner": file_record(receipt_path),
            "phase_timing_seconds": {
                name: load_json(output_dir / f"{name}.json")["wall_seconds"]
                for name in (
                    "vector_generation",
                    "compile",
                    "simulation",
                    "float_policy",
                )
            },
            "claim_boundary": status["claim_boundary"],
        },
    )
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "sealed_output_manifest.json":
            continue
        record = file_record(path)
        record["relative_path"] = str(path.relative_to(output_dir))
        record.pop("path")
        artifacts.append(record)
    write_json(
        output_dir / "sealed_output_manifest.json",
        {
            "schema_version": 1,
            "kind": evidence_kind("sealed_runtime_attempt"),
            "status": status["status"],
            "layer_index": LAYER_INDEX,
            "official_attempt_candidate": OFFICIAL_ATTEMPT_CANDIDATE,
            "official_attempt001_authorized": status["status"] == "PASS",
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        },
    )
    for path in sorted(output_dir.rglob("*"), reverse=True):
        path.chmod(0o500 if path.is_dir() else 0o400)
    output_dir.chmod(0o500)
    print(
        f"LAYER{LAYER_INDEX}_RUNTIME_ADJUDICATION_{status['status']} "
        f"official_attempt001_authorized="
        f"{str(status['status'] == 'PASS').lower()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "execute", "compare", "adjudicate"))
    parser.add_argument("--layer-index", type=int, choices=(22, 23), default=22)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_layer(args.layer_index)
    output_dir = args.output_dir.resolve()
    require(
        output_dir == OUTPUT_DIR,
        "scope_failure",
        f"output must be {OUTPUT_DIR}",
    )
    if args.operation == "prepare":
        prepare(output_dir)
        return 0
    if args.operation == "execute":
        return execute(output_dir)
    if args.operation == "compare":
        return compare_float_policy(output_dir)
    adjudicate(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
