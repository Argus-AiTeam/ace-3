#!/usr/bin/env python3
"""Run the official final RMSNorm RTL from the sealed layer-23 residual."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path("/home/argustest/ace3-argus")
PREDECESSOR = ROOT / "build/model24_layer23_attempt001"
OUTPUT = ROOT / "build/model24_final_rmsnorm_attempt001"
MISSION = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
    "2ba4205c8938/latest.json"
)
HIDDEN_SIZE = 896
TOKEN_COUNT = 2
EXPECTED_ROWS = HIDDEN_SIZE * TOKEN_COUNT
TOP_MODULE = "ace3_fp16_rmsnorm_core"
MODEL_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MODEL_REVISION = "db09cd27ead7fee40cdee309693cf83601b9c899"
ABSOLUTE_TOLERANCE = 0.125
RELATIVE_TOLERANCE = 0.001
MAX_ULP_DISTANCE = 1
TERMINAL_RE = re.compile(
    rb"schema=ace3_final_rmsnorm_raw_v1 natural_terminal=1 exit_code=0 "
    rb"input_count=1792 output_count=1792 transaction_count=2 "
    rb"cycles=([1-9][0-9]*) rms_q24_token0=([1-9][0-9]*) "
    rb"rms_q24_token1=([1-9][0-9]*)\n"
)


class AttemptError(RuntimeError):
    """Raised when an evidence or execution boundary fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AttemptError(message)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    document = json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicate_keys,
    )
    require(isinstance(document, dict), f"JSON object required: {path}")
    return document


def canonical_json(document: object) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def write_json(path: Path, document: object) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json(document))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": sha256(path),
        "mode": f"{stat.st_mode & 0o777:04o}",
    }
    if root is not None:
        record["relative_path"] = str(path.relative_to(root))
    return record


def snapshot(root: Path) -> list[dict[str, Any]]:
    return [
        file_record(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def snapshot_digest(records: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json(records))


def authenticate_predecessor() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(PREDECESSOR.is_dir(), "accepted layer-23 predecessor is missing")
    before = snapshot(PREDECESSOR)
    seal = load_json(PREDECESSOR / "sealed_output_manifest.json")
    require(
        seal.get("kind") == "ace3_layer23_from_layer22_sealed_runtime_attempt"
        and seal.get("status") == "PASS"
        and seal.get("layer_index") == 23
        and seal.get("official_attempt_candidate") == 1,
        "layer-23 seal identity is not an accepted PASS",
    )
    artifacts = {
        row.get("relative_path"): row
        for row in seal.get("artifacts", [])
        if isinstance(row, dict) and isinstance(row.get("relative_path"), str)
    }
    require(
        len(artifacts) == seal.get("artifact_count"),
        "layer-23 seal artifact set is malformed or duplicated",
    )
    for relative, expected in artifacts.items():
        path = PREDECESSOR / relative
        require(path.is_file(), f"sealed layer-23 artifact is missing: {relative}")
        require(
            path.stat().st_size == expected.get("bytes")
            and sha256(path) == expected.get("sha256"),
            f"sealed layer-23 artifact changed: {relative}",
        )

    status = load_json(PREDECESSOR / "status.json")
    terminal = load_json(PREDECESSOR / "natural_terminal_gate.json")
    integer = load_json(PREDECESSOR / "integer_comparison.json")
    comparison = load_json(PREDECESSOR / "comparison.json")
    adjudication = load_json(PREDECESSOR / "adjudication.json")
    preservation = load_json(PREDECESSOR / "preservation.json")
    require(
        status.get("status") == "PASS"
        and terminal.get("natural_terminal") is True
        and terminal.get("actual_exit_code") == 0
        and integer.get("bit_exact") is True
        and comparison.get("status") == "PASS"
        and comparison.get("failure_count") == 0
        and comparison.get("first_material_mismatch") is None
        and adjudication.get("status") == "PASS"
        and preservation.get("status") == "PASS",
        "layer-23 predecessor is not a complete accepted numerical PASS",
    )
    final_path = PREDECESSOR / "raw/final.hex"
    final_seal = artifacts.get("raw/final.hex")
    require(
        final_seal is not None
        and final_path.stat().st_size == final_seal.get("bytes")
        and sha256(final_path) == final_seal.get("sha256"),
        "layer-23 residual is not bound by its seal",
    )
    attempt = load_json(PREDECESSOR / "attempt.json")
    require(
        attempt.get("layer_index") == 23
        and attempt.get("official_attempt_candidate") == 1,
        "layer-23 attempt identity is malformed",
    )
    return (
        {
            "root": str(PREDECESSOR),
            "seal": file_record(PREDECESSOR / "sealed_output_manifest.json"),
            "seal_identity": {
                "kind": seal["kind"],
                "status": seal["status"],
                "layer_index": seal["layer_index"],
                "official_attempt_candidate": seal[
                    "official_attempt_candidate"
                ],
                "artifact_count": seal["artifact_count"],
            },
            "status": file_record(PREDECESSOR / "status.json"),
            "adjudication": file_record(PREDECESSOR / "adjudication.json"),
            "input_handoff": {
                **file_record(final_path),
                "source_stage": "layer23_mlp_residual",
                "consumer_stage": "final_rmsnorm",
                "dtype": "F16",
                "shape": [TOKEN_COUNT, HIDDEN_SIZE],
                "record_format": "token[7:0] index[15:0] f16[15:0]",
                "rows": EXPECTED_ROWS,
            },
            "source_attempt": attempt,
        },
        before,
    )


def parse_layer23_residual(path: Path) -> np.ndarray:
    rows = path.read_bytes().splitlines()
    require(len(rows) == EXPECTED_ROWS, "layer-23 residual row count mismatch")
    values = np.empty((TOKEN_COUNT, HIDDEN_SIZE), dtype="<u2")
    for row_index, row in enumerate(rows):
        require(
            len(row) == 10 and re.fullmatch(rb"[0-9a-f]{10}", row) is not None,
            f"malformed layer-23 residual row {row_index}",
        )
        token = int(row[0:2], 16)
        index = int(row[2:6], 16)
        require(
            token == row_index // HIDDEN_SIZE
            and index == row_index % HIDDEN_SIZE,
            f"out-of-order layer-23 residual row {row_index}",
        )
        values[token, index] = int(row[6:10], 16)
    return values


def source_and_tensor_bindings(
    source_attempt: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, Path, Path, Path]:
    source = source_attempt.get("source")
    require(isinstance(source, dict), "layer-23 source binding is missing")
    source_root = Path(str(source.get("immutable_source_root")))
    model_hashes = source.get("immutable_model_hashes")
    rtl_hashes = source.get("immutable_rtl_and_testbench_hashes")
    require(
        source_root.is_dir()
        and isinstance(model_hashes, dict)
        and isinstance(rtl_hashes, dict),
        "layer-23 immutable source closure is malformed",
    )
    fixed_relative = "ace3/rtl/ace3_fp16_fixed.sv"
    rms_relative = "ace3/rtl/ace3_fp16_rmsnorm_core.sv"
    oracle_relative = "ace3/model/fp16_adaptation_oracle.py"
    fixed = source_root / fixed_relative
    rms = source_root / rms_relative
    oracle = source_root / oracle_relative
    for relative, path, expected in (
        (fixed_relative, fixed, rtl_hashes.get(fixed_relative)),
        (rms_relative, rms, rtl_hashes.get(rms_relative)),
        (oracle_relative, oracle, model_hashes.get(oracle_relative)),
    ):
        require(
            path.is_file() and sha256(path) == expected,
            f"immutable source hash mismatch: {relative}",
        )

    official = source_attempt.get("official_model")
    oracle_binding = source_attempt.get("oracle")
    require(
        isinstance(official, dict)
        and official.get("repository") == MODEL_REPOSITORY
        and official.get("revision") == MODEL_REVISION
        and isinstance(oracle_binding, dict),
        "official model identity mismatch",
    )
    checkpoint_record = official.get("checkpoint")
    tensor_map_record = oracle_binding.get("tensor_map")
    require(
        isinstance(checkpoint_record, dict)
        and isinstance(tensor_map_record, dict),
        "checkpoint or tensor-map binding is missing",
    )
    checkpoint = Path(str(checkpoint_record.get("path")))
    tensor_map = Path(str(tensor_map_record.get("path")))
    require(
        checkpoint.is_file()
        and checkpoint.stat().st_size == checkpoint_record.get("bytes")
        and sha256(checkpoint) == checkpoint_record.get("sha256"),
        "official checkpoint binding mismatch",
    )
    require(
        tensor_map.is_file()
        and tensor_map.stat().st_size == tensor_map_record.get("bytes")
        and sha256(tensor_map) == tensor_map_record.get("sha256"),
        "authenticated tensor-map binding mismatch",
    )
    tensor_document = load_json(tensor_map)
    tensors = tensor_document.get("tensors")
    require(isinstance(tensors, list), "tensor map has no tensor inventory")
    matches = [
        row
        for row in tensors
        if isinstance(row, dict) and row.get("name") == "model.norm.weight"
    ]
    require(len(matches) == 1, "model.norm.weight tensor binding is ambiguous")
    descriptor = matches[0]
    require(
        descriptor.get("dtype") == "F16"
        and descriptor.get("shape") == [HIDDEN_SIZE]
        and descriptor.get("byte_length") == HIDDEN_SIZE * 2
        and descriptor.get("absolute_file_offsets") == [730650456, 730652248]
        and descriptor.get("descriptor_sha256")
        == "e26d80c9eb6e2bab95106bdb480ae43232b17d77ebdecbae7b94209bdd635561",
        "model.norm.weight descriptor mismatch",
    )
    begin, end = descriptor["absolute_file_offsets"]
    with checkpoint.open("rb") as stream:
        stream.seek(begin)
        weight_payload = stream.read(end - begin)
    require(
        len(weight_payload) == HIDDEN_SIZE * 2,
        "model.norm.weight payload is truncated",
    )
    weights = np.frombuffer(weight_payload, dtype="<u2").copy()
    bindings = {
        "immutable_source_root": str(source_root),
        "rtl": {
            "fp16_fixed": file_record(fixed),
            "rmsnorm_core": file_record(rms),
        },
        "integer_oracle_source": file_record(oracle),
        "official_model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint": file_record(checkpoint),
        },
        "tensor_map": file_record(tensor_map),
        "model_norm_weight": {
            "name": "model.norm.weight",
            "dtype": "F16",
            "shape": [HIDDEN_SIZE],
            "absolute_file_offsets": [begin, end],
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "bytes": len(weight_payload),
            "sha256": sha256_bytes(weight_payload),
        },
    }
    return bindings, weights, fixed, rms, oracle


def write_hex(path: Path, values: np.ndarray) -> None:
    with path.open("x", encoding="ascii", newline="\n") as stream:
        for value in values.flat:
            stream.write(f"{int(value):04x}\n")


def harness_source() -> str:
    return r'''#include "Vace3_fp16_rmsnorm_core.h"
#include "verilated.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static uint64_t cycles = 0;

static void fail(const std::string& message) {
    std::cerr << message << "\n";
    std::exit(1);
}

static std::vector<uint16_t> read_hex(const std::string& path) {
    std::ifstream input(path);
    if (!input)
        fail("cannot open " + path);
    std::vector<uint16_t> values;
    std::string line;
    while (std::getline(input, line)) {
        if (line.size() != 4)
            fail("malformed input row in " + path);
        values.push_back(static_cast<uint16_t>(
            std::stoul(line, nullptr, 16)));
    }
    return values;
}

static void tick(Vace3_fp16_rmsnorm_core* top) {
    top->clk_i = 0;
    top->eval();
    top->clk_i = 1;
    top->eval();
    ++cycles;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    std::string vector_dir;
    std::string raw_dir;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--vector-dir" && index + 1 < argc)
            vector_dir = argv[++index];
        else if (argument == "--raw-dir" && index + 1 < argc)
            raw_dir = argv[++index];
        else
            fail("unexpected or incomplete argument: " + argument);
    }
    if (vector_dir.empty() || raw_dir.empty())
        fail("usage: final-rmsnorm --vector-dir PATH --raw-dir PATH");

    const auto inputs = read_hex(vector_dir + "/layer23_residual.hex");
    const auto weights = read_hex(vector_dir + "/model_norm_weight.hex");
    if (inputs.size() != 1792 || weights.size() != 896)
        fail("authenticated simulator input count mismatch");
    std::ofstream output(raw_dir + "/final.hex", std::ios::out | std::ios::trunc);
    if (!output)
        fail("cannot create raw final output");

    auto* top = new Vace3_fp16_rmsnorm_core;
    top->clear_i = 0;
    top->start_valid_i = 0;
    top->element_count_i = 896;
    top->in_valid_i = 0;
    top->activation_f16_i = 0;
    top->weight_f16_i = 0;
    top->out_ready_i = 0;
    top->rst_ni = 0;
    tick(top);
    tick(top);
    top->rst_ni = 1;
    tick(top);

    uint64_t roots[2] = {0, 0};
    for (unsigned token = 0; token < 2; ++token) {
        top->start_valid_i = 1;
        top->eval();
        if (!top->start_ready_o)
            fail("RMSNorm legal start was not ready");
        tick(top);
        top->start_valid_i = 0;

        for (unsigned index = 0; index < 896; ++index) {
            top->activation_f16_i = inputs.at(token * 896 + index);
            top->weight_f16_i = weights.at(index);
            top->in_valid_i = 1;
            top->eval();
            if (!top->in_ready_o)
                fail("RMSNorm input was not ready");
            tick(top);
        }
        top->in_valid_i = 0;
        unsigned sqrt_cycles = 0;
        while (!top->out_valid_o && sqrt_cycles <= 46) {
            tick(top);
            ++sqrt_cycles;
        }
        if (sqrt_cycles != 46)
            fail("RMSNorm square-root latency mismatch");
        roots[token] = top->rms_q24_o;
        if (roots[token] == 0)
            fail("RMSNorm produced a zero root");

        for (unsigned index = 0; index < 896; ++index) {
            top->eval();
            if (!top->out_valid_o || top->out_index_o != index ||
                top->out_last_o != (index == 895) ||
                top->invalid_operand_o || top->saturation_o)
                fail("RMSNorm output protocol or status mismatch");
            output << std::hex << std::nouppercase << std::setfill('0')
                   << std::setw(2) << token
                   << std::setw(4) << index
                   << std::setw(4) << static_cast<unsigned>(top->out_f16_o)
                   << "\n";
            top->out_ready_i = 1;
            tick(top);
            top->out_ready_i = 0;
        }
    }
    output.close();
    if (!output)
        fail("raw final output write failed");

    std::ofstream terminal(raw_dir + "/terminal.txt",
                           std::ios::out | std::ios::trunc);
    if (!terminal)
        fail("cannot create raw terminal record");
    terminal
        << "schema=ace3_final_rmsnorm_raw_v1 natural_terminal=1 exit_code=0 "
        << "input_count=1792 output_count=1792 transaction_count=2 "
        << "cycles=" << std::dec << cycles
        << " rms_q24_token0=" << roots[0]
        << " rms_q24_token1=" << roots[1] << "\n";
    terminal.close();
    if (!terminal)
        fail("raw terminal record write failed");

    std::cout
        << "ACE3_FINAL_RMSNORM_RTL_PASS inputs=1792 outputs=1792 "
        << "transactions=2 cycles=" << cycles
        << " synthesis=not_run ppa=not_measured fpga=not_run\n";
    top->final();
    delete top;
    return 0;
}
'''


def run_command(
    argv: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
) -> tuple[subprocess.CompletedProcess[bytes], int, int]:
    started = time.time_ns()
    completed = subprocess.run(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    ended = time.time_ns()
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    return completed, started, ended


def compile_rtl(
    fixed: Path, rms: Path, source_path: Path
) -> tuple[Path, dict[str, Any]]:
    obj_dir = OUTPUT / "tmp/obj_dir"
    obj_dir.mkdir(parents=True)
    argv = [
        "verilator",
        "--cc",
        "--exe",
        "--build",
        "-Wno-fatal",
        "--top-module",
        TOP_MODULE,
        f"-GHIDDEN_SIZE={HIDDEN_SIZE}",
        "--Mdir",
        str(obj_dir),
        str(fixed),
        str(rms),
        str(source_path),
    ]
    completed, started, ended = run_command(
        argv,
        ROOT,
        OUTPUT / "compile.stdout",
        OUTPUT / "compile.stderr",
        120,
    )
    built = obj_dir / f"V{TOP_MODULE}"
    require(
        completed.returncode == 0 and built.is_file(),
        "Verilator compile/elaboration failed",
    )
    compiled_dir = OUTPUT / "compiled"
    compiled_dir.mkdir()
    binary = compiled_dir / built.name
    shutil.copy2(built, binary)
    binary.chmod(0o500)
    record = {
        "phase": "compile_and_elaboration",
        "status": "PASS",
        "argv": argv,
        "cwd": str(ROOT),
        "started_unix_ns": started,
        "ended_unix_ns": ended,
        "wall_seconds": (ended - started) / 1_000_000_000,
        "exit_code": completed.returncode,
        "top_module": TOP_MODULE,
        "parameters": {"HIDDEN_SIZE": HIDDEN_SIZE},
        "stdout": file_record(OUTPUT / "compile.stdout"),
        "stderr": file_record(OUTPUT / "compile.stderr"),
        "binary": file_record(binary),
    }
    write_json(OUTPUT / "compile.json", record)
    shutil.rmtree(OUTPUT / "tmp")
    return binary, record


def execute_rtl(binary: Path) -> dict[str, Any]:
    argv = [
        str(binary),
        "--vector-dir",
        str(OUTPUT / "vectors"),
        "--raw-dir",
        str(OUTPUT / "raw"),
    ]
    write_json(
        OUTPUT / "execution_started.json",
        {
            "phase": "simulation",
            "argv": argv,
            "cwd": str(ROOT),
            "started_unix_ns": time.time_ns(),
        },
    )
    completed, started, ended = run_command(
        argv,
        ROOT,
        OUTPUT / "simulation.stdout",
        OUTPUT / "simulation.stderr",
        120,
    )
    record = {
        "phase": "simulation",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "argv": argv,
        "cwd": str(ROOT),
        "started_unix_ns": started,
        "ended_unix_ns": ended,
        "wall_seconds": (ended - started) / 1_000_000_000,
        "exit_code": completed.returncode,
        "stdout": file_record(OUTPUT / "simulation.stdout"),
        "stderr": file_record(OUTPUT / "simulation.stderr"),
    }
    write_json(OUTPUT / "simulation.json", record)
    require(completed.returncode == 0, "final-RMSNorm RTL simulation failed")
    return record


def natural_terminal_gate(simulation: Mapping[str, Any]) -> dict[str, Any]:
    terminal_path = OUTPUT / "raw/terminal.txt"
    final_path = OUTPUT / "raw/final.hex"
    require(
        simulation.get("status") == "PASS"
        and simulation.get("exit_code") == 0
        and terminal_path.is_file()
        and final_path.is_file(),
        "simulation did not reach the terminal gate",
    )
    terminal_payload = terminal_path.read_bytes()
    require(
        terminal_payload.isascii(),
        "terminal record contains non-ASCII bytes",
    )
    match = TERMINAL_RE.fullmatch(terminal_payload)
    require(match is not None, "terminal record is missing, malformed, or ambiguous")
    produced = parse_layer23_residual(final_path)
    gate = {
        "schema_version": 1,
        "kind": "ace3_final_rmsnorm_natural_terminal_gate",
        "natural_terminal": True,
        "actual_exit_code": simulation["exit_code"],
        "recorded_exit_code": 0,
        "input_count": EXPECTED_ROWS,
        "output_count": int(produced.size),
        "transaction_count": TOKEN_COUNT,
        "reported_cycles": int(match.group(1)),
        "rms_q24": [int(match.group(2)), int(match.group(3))],
        "terminal": file_record(terminal_path),
        "raw_output": file_record(final_path),
        "simulation": file_record(OUTPUT / "simulation.json"),
    }
    write_json(OUTPUT / "natural_terminal_gate.json", gate)
    return gate


def load_integer_oracle(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "ace3_final_rmsnorm_integer_oracle", path
    )
    require(spec is not None and spec.loader is not None, "cannot load integer oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def integer_comparison(
    inputs: np.ndarray,
    weights: np.ndarray,
    actual: np.ndarray,
    oracle_path: Path,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        gate.get("natural_terminal") is True,
        "integer comparison attempted before natural terminal",
    )
    oracle = load_integer_oracle(oracle_path)
    expected = np.empty_like(actual)
    roots: list[int] = []
    mean_q48: list[int] = []
    for token in range(TOKEN_COUNT):
        outputs, mean, root = oracle.rmsnorm(
            inputs[token].tolist(), weights.tolist()
        )
        require(
            all(not invalid and not saturated for _, invalid, saturated in outputs),
            f"integer oracle rejected token {token}",
        )
        expected[token] = np.asarray(
            [bits for bits, _, _ in outputs], dtype="<u2"
        )
        mean_q48.append(int(mean))
        roots.append(int(root))
    write_hex(OUTPUT / "oracle/integer_expected.hex", expected)
    mismatch_indices = np.argwhere(actual != expected)
    first = None
    if mismatch_indices.size:
        token, index = map(int, mismatch_indices[0])
        first = {
            "token_index": token,
            "element_index": index,
            "actual_f16_bits": f"{int(actual[token, index]):04x}",
            "expected_f16_bits": f"{int(expected[token, index]):04x}",
        }
    result = {
        "schema_version": 1,
        "kind": "ace3_final_rmsnorm_integer_oracle_comparison",
        "status": "PASS" if not mismatch_indices.size else "FAIL",
        "records": EXPECTED_ROWS,
        "failure_count": int(mismatch_indices.shape[0]),
        "first_material_mismatch": first,
        "bit_exact": not mismatch_indices.size,
        "mean_q48": mean_q48,
        "expected_rms_q24": roots,
        "reported_rms_q24": gate["rms_q24"],
        "root_bit_exact": roots == gate["rms_q24"],
        "input": file_record(OUTPUT / "vectors/layer23_residual.hex"),
        "weight": file_record(OUTPUT / "vectors/model_norm_weight.hex"),
        "expected": file_record(OUTPUT / "oracle/integer_expected.hex"),
        "produced": file_record(OUTPUT / "raw/final.hex"),
        "oracle_source": file_record(oracle_path),
    }
    if roots != gate["rms_q24"]:
        result["status"] = "FAIL"
        result["bit_exact"] = False
        result["failure_count"] = max(1, result["failure_count"])
        if result["first_material_mismatch"] is None:
            result["first_material_mismatch"] = {
                "field": "rms_q24",
                "actual": gate["rms_q24"],
                "expected": roots,
            }
    write_json(OUTPUT / "integer_comparison.json", result)
    return result


def ordered_f16(raw: int) -> int:
    return 0x8000 - (raw & 0x7FFF) if raw & 0x8000 else 0x8000 + raw


def fp16_interstage_expected(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    require(
        inputs.dtype == weights.dtype == np.dtype("<u2")
        and inputs.ndim == 2 and inputs.shape[1] == HIDDEN_SIZE
        and weights.shape == (HIDDEN_SIZE,),
        "invalid final-RMSNorm FP16 input shape or encoding",
    )
    expected = np.empty_like(inputs)
    input_f16 = inputs.view("<f2")
    weight_f16 = weights.view("<f2")
    require(np.isfinite(input_f16).all() and np.isfinite(weight_f16).all(),
            "nonfinite final-RMSNorm reference operand")
    for token in range(len(inputs)):
        values_f32 = input_f16[token].astype(np.float32)
        variance_f32 = np.mean(values_f32 * values_f32, dtype=np.float32)
        inverse_rms_f32 = np.float32(1.0) / np.sqrt(
            np.float32(variance_f32 + np.float32(1.0e-6))
        )
        normalized_f16 = np.asarray(
            values_f32 * inverse_rms_f32, dtype="<f2"
        )
        expected[token] = np.asarray(
            normalized_f16 * weight_f16, dtype="<f2"
        ).view("<u2")
    require(np.isfinite(expected.view("<f2")).all(),
            "nonfinite final-RMSNorm reference output")
    return expected


def fp16_policy_comparison(
    inputs: np.ndarray,
    weights: np.ndarray,
    actual: np.ndarray,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        gate.get("natural_terminal") is True,
        "FP16-policy comparison attempted before natural terminal",
    )
    expected = fp16_interstage_expected(inputs, weights)
    write_hex(OUTPUT / "oracle/fp16_policy_expected.hex", expected)

    actual_values = actual.view("<f2").astype(np.float64)
    expected_values = expected.view("<f2").astype(np.float64)
    require(
        np.isfinite(actual_values).all() and np.isfinite(expected_values).all(),
        "nonfinite final-RMSNorm comparison operand",
    )
    absolute = np.abs(actual_values - expected_values)
    relative = absolute / np.maximum(np.abs(expected_values), 2.0**-14)
    ulp = np.empty(actual.shape, dtype=np.int64)
    failures: list[tuple[int, int]] = []
    for token in range(TOKEN_COUNT):
        for index in range(HIDDEN_SIZE):
            distance = abs(
                ordered_f16(int(actual[token, index]))
                - ordered_f16(int(expected[token, index]))
            )
            ulp[token, index] = distance
            if (
                absolute[token, index] > ABSOLUTE_TOLERANCE
                and (
                    relative[token, index] >= RELATIVE_TOLERANCE
                    or distance > MAX_ULP_DISTANCE
                )
            ):
                failures.append((token, index))
    first = None
    if failures:
        token, index = failures[0]
        first = {
            "token_index": token,
            "element_index": index,
            "actual_f16_bits": f"{int(actual[token, index]):04x}",
            "expected_f16_bits": f"{int(expected[token, index]):04x}",
            "actual": float(actual_values[token, index]),
            "expected": float(expected_values[token, index]),
            "absolute_error": float(absolute[token, index]),
            "relative_error": float(relative[token, index]),
            "ulp_distance": int(ulp[token, index]),
        }
    result = {
        "schema_version": 1,
        "kind": "ace3_final_rmsnorm_fp16_policy_comparison",
        "status": "PASS" if not failures else "FAIL",
        "within_tolerance": not failures,
        "records": EXPECTED_ROWS,
        "failure_count": len(failures),
        "first_material_mismatch": first,
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "max_relative_error": float(relative.max()),
        "max_ulp_distance": int(ulp.max()),
        "reference_policy": {
            "name": "qwen2_fp16_interstage_final_rmsnorm",
            "operation_order": (
                "FP16 input decode; float32 mean-square and rsqrt with "
                "epsilon=1e-6; normalized activation rounded to FP16; "
                "model.norm.weight multiply rounded to FP16"
            ),
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "relative_denominator_floor": 2.0**-14,
            "max_ulp_distance": MAX_ULP_DISTANCE,
            "material_failure_rule": (
                "absolute_error > 0.125 AND "
                "(relative_error >= 0.001 OR ordered_FP16_ULP > 1)"
            ),
        },
        "expected": file_record(OUTPUT / "oracle/fp16_policy_expected.hex"),
        "produced": file_record(OUTPUT / "raw/final.hex"),
        "implementation": {
            "library": "NumPy",
            "version": np.__version__,
            "independent_from_integer_oracle": True,
        },
    }
    write_json(OUTPUT / "fp16_policy_comparison.json", result)
    return result


def seal_attempt(
    predecessor_before: list[dict[str, Any]],
    integer: Mapping[str, Any],
    fp16_policy: Mapping[str, Any],
    simulation: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> None:
    predecessor_after = snapshot(PREDECESSOR)
    preserved = predecessor_before == predecessor_after
    preservation = {
        "schema_version": 1,
        "kind": "ace3_final_rmsnorm_predecessor_preservation",
        "status": "PASS" if preserved else "FAIL",
        "predecessor": str(PREDECESSOR),
        "before_file_count": len(predecessor_before),
        "after_file_count": len(predecessor_after),
        "before_ordered_file_set_sha256": snapshot_digest(predecessor_before),
        "after_ordered_file_set_sha256": snapshot_digest(predecessor_after),
        "predecessor_preserved": preserved,
    }
    write_json(OUTPUT / "preservation.json", preservation)
    require(preserved, "accepted layer-23 predecessor changed during the attempt")
    passed = (
        integer.get("status") == "PASS"
        and integer.get("failure_count") == 0
        and integer.get("first_material_mismatch") is None
        and integer.get("root_bit_exact") is True
        and fp16_policy.get("status") == "PASS"
        and fp16_policy.get("failure_count") == 0
        and fp16_policy.get("first_material_mismatch") is None
        and gate.get("natural_terminal") is True
        and simulation.get("exit_code") == 0
    )
    status = {
        "schema_version": 1,
        "kind": "ace3_final_rmsnorm_runtime_status",
        "status": "PASS" if passed else "FAIL",
        "official_attempt": 1,
        "source_stage": "accepted_layer23_mlp_residual",
        "operation": "final_rmsnorm",
        "natural_terminal": gate["natural_terminal"],
        "simulation_exit_code": simulation["exit_code"],
        "simulation_wall_seconds": simulation["wall_seconds"],
        "reported_cycles": gate["reported_cycles"],
        "input_rows": EXPECTED_ROWS,
        "output_rows": EXPECTED_ROWS,
        "integer_oracle_bit_exact": integer["bit_exact"],
        "fp16_policy_within_tolerance": fp16_policy["within_tolerance"],
        "failure_count": integer["failure_count"]
        + fp16_policy["failure_count"],
        "first_material_mismatch": integer["first_material_mismatch"]
        or fp16_policy["first_material_mismatch"],
        "predecessor_preserved": preservation["predecessor_preserved"],
        "review_status": "PENDING_INDEPENDENT_REVIEW",
        "advance_to_tied_lm_head_topk": False,
        "claim_boundary": {
            "demonstrated": (
                "computer-local final-RMSNorm RTL simulation from the "
                "authenticated accepted layer-23 residual"
            ),
            "not_demonstrated": [
                "independent reviewer PASS",
                "tied lm_head or top-k",
                "tokenizer or host integration",
                "persistent multi-token K/V",
                "readable dialogue",
                "synthesis, PPA, bitstream, FPGA, or silicon",
            ],
        },
    }
    write_json(OUTPUT / "status.json", status)
    require(passed, "final-RMSNorm numerical adjudication failed")
    write_json(
        OUTPUT / "adjudication.json",
        {
            "schema_version": 1,
            "kind": "ace3_final_rmsnorm_engineer_adjudication",
            "status": "ENGINEERING_PASS_REVIEW_REQUIRED",
            "official_attempt": 1,
            "failure_count": 0,
            "first_material_mismatch": None,
            "natural_terminal": True,
            "predecessor_preserved": True,
            "review_requirement": {
                "required": True,
                "status": "PENDING",
                "independent_reviewer_must_authenticate": [
                    "accepted layer-23 predecessor",
                    "official model.norm.weight binding",
                    "fresh source, input, and tensor hashes",
                    "natural simulator terminal and raw output",
                    "integer bit-exact comparison",
                    "FP16-policy tolerance comparison",
                    "predecessor preservation",
                ],
            },
            "advance_to_tied_lm_head_topk": False,
        },
    )
    excluded = {"sealed_output_manifest.json"}
    artifacts = [
        file_record(path, OUTPUT)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]
    write_json(
        OUTPUT / "sealed_output_manifest.json",
        {
            "schema_version": 1,
            "kind": "ace3_final_rmsnorm_attempt001_sealed_manifest",
            "status": "ENGINEERING_PASS_REVIEW_REQUIRED",
            "official_attempt": 1,
            "operation": "final_rmsnorm",
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        },
    )
    for path in sorted(OUTPUT.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o400)
        elif path.is_dir():
            path.chmod(0o500)
    OUTPUT.chmod(0o500)


def main() -> int:
    require(Path.cwd().resolve() == ROOT, f"run from {ROOT}")
    require(not OUTPUT.exists(), f"fresh attempt already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, mode=0o700)
    try:
        (OUTPUT / "vectors").mkdir()
        (OUTPUT / "source").mkdir()
        (OUTPUT / "raw").mkdir()
        (OUTPUT / "oracle").mkdir()
        (OUTPUT / "tmp").mkdir()
        predecessor, predecessor_before = authenticate_predecessor()
        mission = load_json(MISSION)
        require(
            mission.get("kind") == "mission_context"
            and mission.get("node_key")
            == "final-rmsnorm-runtime-pass-from-accepted-layer23"
            and "build/model24_layer23_attempt001/raw/final.hex"
            in str(mission.get("objective")),
            "live mission does not authorize this final-RMSNorm attempt",
        )
        bindings, weights, fixed, rms, oracle = source_and_tensor_bindings(
            predecessor["source_attempt"]
        )
        inputs = parse_layer23_residual(PREDECESSOR / "raw/final.hex")
        write_hex(OUTPUT / "vectors/layer23_residual.hex", inputs)
        write_hex(OUTPUT / "vectors/model_norm_weight.hex", weights)
        source_path = OUTPUT / "source/final_rmsnorm_main.cpp"
        source_path.write_text(harness_source(), encoding="ascii", newline="\n")
        write_json(
            OUTPUT / "predecessor_before.json",
            {
                "schema_version": 1,
                "root": str(PREDECESSOR),
                "file_count": len(predecessor_before),
                "ordered_file_set_sha256": snapshot_digest(
                    predecessor_before
                ),
                "files": predecessor_before,
            },
        )
        input_manifest = {
            "schema_version": 1,
            "kind": "ace3_final_rmsnorm_authenticated_inputs",
            "predecessor": {
                key: value
                for key, value in predecessor.items()
                if key != "source_attempt"
            },
            "source_and_official_tensor_bindings": bindings,
            "fresh_vectors": {
                "layer23_residual": file_record(
                    OUTPUT / "vectors/layer23_residual.hex"
                ),
                "model_norm_weight": file_record(
                    OUTPUT / "vectors/model_norm_weight.hex"
                ),
            },
            "generated_simulator_harness": file_record(source_path),
            "no_layer_replay": True,
            "predecessor_layers_executed": [],
        }
        write_json(OUTPUT / "input_manifest.json", input_manifest)
        write_json(
            OUTPUT / "attempt.json",
            {
                "schema_version": 1,
                "kind": "ace3_final_rmsnorm_attempt",
                "official_attempt": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "mission": file_record(MISSION),
                "host": {
                    "hostname": platform.node(),
                    "platform": platform.platform(),
                    "python": sys.version.splitlines()[0],
                    "numpy": np.__version__,
                    "cpu_count": os.cpu_count(),
                },
                "profile": {
                    "model": MODEL_REPOSITORY,
                    "revision": MODEL_REVISION,
                    "activations": "FP16",
                    "final_norm_weight": "official FP16 model.norm.weight",
                    "rtl_arithmetic": "ACE-3 exact Q24/Q48 RMSNorm",
                    "epsilon": "1e-6",
                },
                "input_manifest": file_record(OUTPUT / "input_manifest.json"),
            },
        )
        binary, _ = compile_rtl(fixed, rms, source_path)
        simulation = execute_rtl(binary)
        gate = natural_terminal_gate(simulation)
        actual = parse_layer23_residual(OUTPUT / "raw/final.hex")
        integer = integer_comparison(inputs, weights, actual, oracle, gate)
        fp16_policy = fp16_policy_comparison(
            inputs, weights, actual, gate
        )
        seal_attempt(
            predecessor_before,
            integer,
            fp16_policy,
            simulation,
            gate,
        )
        print(
            "ACE3_FINAL_RMSNORM_ATTEMPT001_ENGINEERING_PASS "
            "natural_terminal=1 integer_failures=0 fp16_policy_failures=0 "
            "review=pending tied_lm_head_topk=not_started"
        )
        return 0
    except Exception as error:
        if OUTPUT.is_dir():
            failure = {
                "schema_version": 1,
                "kind": "ace3_final_rmsnorm_attempt_failure",
                "status": "FAIL",
                "error_type": type(error).__name__,
                "message": str(error),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            failure_path = OUTPUT / "failure.json"
            if not failure_path.exists():
                write_json(failure_path, failure)
        print(f"FINAL_RMSNORM_ATTEMPT_FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
