#!/usr/bin/env python3
"""Generate authenticated Model24 software and RTL-simulation timing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from controller_model24_cascade import (
    _load_json,
    parse_controller_events,
    parse_simulation_terminal,
)
from model24_execution_oracle import (
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_SHA256,
    authenticate_tokenizer,
)
from model24_oracle import (
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    authenticate_checkpoint,
)
from official_model24_dialogue import (
    ARTIFACT_NAME as DIALOGUE_ARTIFACT_NAME,
    MANIFEST_NAME as DIALOGUE_MANIFEST_NAME,
    validate_document as validate_dialogue_document,
)

KIND = "ace3_model24_latency_diagnostics"
MANIFEST_KIND = "ace3_model24_latency_diagnostics_manifest"
MEASUREMENT_NAME = "measurement.json"
MANIFEST_NAME = "manifest.json"
LAYER_COUNT = 24
SOURCE_PATHS = (
    "ace3/contracts/model24_execution_vector_bindings.json",
    "ace3/contracts/model24_layer_controller.json",
    "ace3/contracts/model24_tensor_map.json",
    "ace3/model/attention_oracle.py",
    "ace3/model/awq_bit_oracle.py",
    "ace3/model/controller_model24_cascade.py",
    "ace3/model/controller_model24_rtl_cascade.py",
    "ace3/model/decoder_layer0_oracle.py",
    "ace3/model/fp16_adaptation_oracle.py",
    "ace3/model/measure_model24_latency.py",
    "ace3/model/model24_execution_oracle.py",
    "ace3/model/model24_oracle.py",
    "ace3/model/official_model24_dialogue.py",
    "ace3/model/official_model24_next_token.py",
    "ace3/model/official_single_decoder_layer.py",
    "ace3/model/projection_oracle.py",
    "ace3/model/qwen2_rope_oracle.py",
    "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
    "ace3/rtl/ace3_fp16_silu_gate_core.sv",
    "ace3/rtl/ace3_model24_layer_controller.sv",
    "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp",
    "ace3/tb/ace3_model24_layer_controller_main.cpp",
)
CLAIM_BOUNDARY = {
    "software_wall": (
        "single-run host process wall time for the fixed official four-token "
        "software dialogue continuation; diagnostic only"
    ),
    "rtl_wall": (
        "single-run Verilator controller and decoder process wall time; "
        "diagnostic simulation evidence only"
    ),
    "rtl_cycles": (
        "Verilator harness tick counts from 24 separately indexed two-token "
        "decoder simulations plus the abstract controller scheduler; not "
        "hardware clock cycles or end-to-end dialogue cycles"
    ),
    "synthesis": "not run",
    "timing_closure": "not run",
    "ppa": "not measured",
    "fpga": "not run",
    "hardware_latency": "not measured",
    "hardware_throughput": "not measured",
    "bottleneck": "not attributed by this measurement",
}
CONTROLLER_PASS = re.compile(
    r"^MODEL24_LAYER_CONTROLLER_VERILATOR_PASS .* "
    r"cycles=(?P<cycles>[0-9]+) numerical_rtl=not_claimed$"
)
LAYER_PASS = re.compile(
    r"^DECODER_LAYER_TOKEN_ENGINE_VERILATOR_PASS "
    r"layer=(?P<layer>[0-9]+) trace_count=(?P<trace>[0-9]+) "
    r"final_count=(?P<final>[0-9]+) cycles=(?P<cycles>[0-9]+) "
    r"stalls=(?P<stalls>[0-9]+) token0_cycles=(?P<token0>[0-9]+) "
    r"token1_cycles=(?P<token1>[0-9]+) phase_p_run=(?P<p_run>[0-9]+) "
    r"phase_final=(?P<final_phase>[0-9]+) "
    r"phase_cycles=(?P<phases>[0-9]+(?:,[0-9]+){36}) "
    r"reset=pass clear=pass slot_isolation=pass$"
)


class MeasurementError(RuntimeError):
    """Raised when latency evidence is incomplete or ambiguous."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MeasurementError(message)


def canonical_json(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path, repository_root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as error:
        raise MeasurementError(
            f"path is outside the ACE-3 worktree: {resolved}"
        ) from error
    payload = resolved.read_bytes()
    return {
        "path": relative.as_posix(),
        "bytes": len(payload),
        "sha256": sha256(payload),
    }


def load_json(path: Path, label: str) -> dict[str, Any]:
    return _load_json(path.read_bytes(), label)


def parse_controller_cycles(text: str) -> int:
    matches = [
        line
        for line in text.splitlines()
        if line.startswith("MODEL24_LAYER_CONTROLLER_VERILATOR_PASS")
    ]
    require(len(matches) == 1, "ambiguous controller PASS")
    match = CONTROLLER_PASS.fullmatch(matches[0])
    require(match is not None, "ambiguous controller PASS")
    cycles = int(match.group("cycles"))
    require(cycles > 0, "controller cycle count must be positive")
    return cycles


def parse_layer_cycles(text: str, expected_layer: int) -> dict[str, Any]:
    matches = [
        line
        for line in text.splitlines()
        if line.startswith("DECODER_LAYER_TOKEN_ENGINE_VERILATOR_PASS")
    ]
    require(
        len(matches) == 1,
        f"layer {expected_layer} has an ambiguous PASS",
    )
    match = LAYER_PASS.fullmatch(matches[0])
    require(
        match is not None,
        f"layer {expected_layer} has an ambiguous PASS",
    )
    record = {
        "layer_index": int(match.group("layer")),
        "cycles": int(match.group("cycles")),
        "stall_cycles": int(match.group("stalls")),
        "token0_cycles": int(match.group("token0")),
        "token1_cycles": int(match.group("token1")),
        "trace_count": int(match.group("trace")),
        "final_count": int(match.group("final")),
        "phase_p_run_cycles": int(match.group("p_run")),
        "phase_final_cycles": int(match.group("final_phase")),
        "phase_cycles": [int(value) for value in match.group("phases").split(",")],
    }
    require(
        record["layer_index"] == expected_layer,
        f"layer PASS index mismatch: {expected_layer}",
    )
    require(
        all(
            record[name] > 0
            for name in ("cycles", "token0_cycles", "token1_cycles")
        ),
        f"layer {expected_layer} cycle evidence is vacuous",
    )
    require(
        len(record["phase_cycles"]) == 37
        and all(value > 0 for value in record["phase_cycles"]),
        f"layer {expected_layer} cycle evidence is vacuous",
    )
    require(
        record["trace_count"] == 46676 and record["final_count"] == 1792,
        f"layer {expected_layer} output counts mismatch",
    )
    return record


def run_timed_command(
    command: Sequence[str],
    display_command: Sequence[str],
    repository_root: Path,
    log_path: Path,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("wb") as log:
        log.write(("$ " + shlex.join(display_command) + "\n").encode("utf-8"))
        log.flush()
        completed = subprocess.run(
            list(command),
            cwd=repository_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        wall_seconds = time.perf_counter() - start
        log.write(
            (
                f"MEASUREMENT_COMMAND_TERMINAL exit_code={completed.returncode} "
                f"wall_seconds_diagnostic={wall_seconds:.9f}\n"
            ).encode("ascii")
        )
    require(
        completed.returncode == 0,
        f"command failed with exit {completed.returncode}: "
        f"{shlex.join(display_command)}",
    )
    return wall_seconds


def runtime_state() -> dict[str, Any]:
    try:
        cpu_model = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="ascii").splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        cpu_model = platform.processor() or "unavailable"
    affinity = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None
    )
    verilator = subprocess.run(
        ["verilator", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    require(verilator.returncode == 0, "unable to query Verilator version")
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "verilator": verilator.stdout.strip(),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": affinity,
        "load_average": (
            list(os.getloadavg()) if hasattr(os, "getloadavg") else None
        ),
        "timing_clock": "time.perf_counter",
    }


def source_records(repository_root: Path) -> list[dict[str, Any]]:
    return [file_record(repository_root / path, repository_root) for path in SOURCE_PATHS]


def software_records(
    software_dir: Path,
    repository_root: Path,
    tokenizer_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_path = software_dir / DIALOGUE_ARTIFACT_NAME
    manifest_path = software_dir / DIALOGUE_MANIFEST_NAME
    evidence = load_json(evidence_path, "dialogue evidence")
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    summary = validate_dialogue_document(evidence, tokenizer)
    require(summary["steps"] == 4, "software dialogue did not execute four steps")
    return (
        {
            "evidence": file_record(evidence_path, repository_root),
            "manifest": file_record(manifest_path, repository_root),
        },
        summary,
    )


def layer_cycle_records(
    rtl_output_dir: Path,
    repository_root: Path,
) -> list[dict[str, Any]]:
    records = []
    for layer in range(LAYER_COUNT):
        layer_dir = rtl_output_dir / "layers" / f"layer{layer:02d}"
        record = parse_layer_cycles(
            (layer_dir / "simulation.log").read_text(encoding="utf-8"),
            layer,
        )
        record.update(
            {
                "simulation_log": file_record(
                    layer_dir / "simulation.log", repository_root
                ),
                "compile_log": file_record(
                    layer_dir / "compile.log", repository_root
                ),
                "record": file_record(layer_dir / "record.json", repository_root),
                "simulator_binary": file_record(
                    rtl_output_dir
                    / "compiled"
                    / f"layer{layer}"
                    / "obj_dir"
                    / "Vace3_decoder_layer0_token_engine",
                    repository_root,
                ),
            }
        )
        records.append(record)
    return records


def validate_measurement_document(document: Mapping[str, Any]) -> None:
    require(document.get("schema_version") == 1, "measurement schema mismatch")
    require(document.get("kind") == KIND, "measurement kind mismatch")
    measurements = document.get("measurements")
    require(isinstance(measurements, Mapping), "measurement records are missing")
    software = measurements.get("software_dialogue_continuation")
    rtl = measurements.get("rtl_controller_cascade")
    require(isinstance(software, Mapping), "software measurement is missing")
    require(isinstance(rtl, Mapping), "RTL measurement is missing")
    for label, value in (
        ("software wall", software.get("wall_seconds_diagnostic")),
        ("RTL wall", rtl.get("wall_seconds_diagnostic")),
        ("controller wall", rtl.get("controller_wall_seconds_diagnostic")),
        ("decoder wall", rtl.get("decoder_wall_seconds_diagnostic")),
    ):
        require(
            isinstance(value, float) and math.isfinite(value) and value > 0.0,
            f"{label} must be a positive finite float",
        )
    require(
        math.isclose(
            rtl["wall_seconds_diagnostic"],
            rtl["controller_wall_seconds_diagnostic"]
            + rtl["decoder_wall_seconds_diagnostic"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ),
        "RTL wall-time aggregate mismatch",
    )
    layers = rtl.get("layers")
    require(
        isinstance(layers, list)
        and [record.get("layer_index") for record in layers]
        == list(range(LAYER_COUNT)),
        "RTL layer cycle coverage mismatch",
    )
    require(
        isinstance(rtl.get("controller_scheduler_cycles"), int)
        and rtl["controller_scheduler_cycles"] > 0,
        "controller cycle evidence is vacuous",
    )
    for key, field in (
        ("aggregate_decoder_harness_cycles", "cycles"),
        ("aggregate_token0_cycles", "token0_cycles"),
        ("aggregate_token1_cycles", "token1_cycles"),
        ("aggregate_stall_cycles", "stall_cycles"),
    ):
        require(
            rtl.get(key) == sum(record[field] for record in layers),
            f"{key} aggregate mismatch",
        )
    require(
        document.get("claim_boundary") == CLAIM_BOUNDARY,
        "measurement claim boundary mismatch",
    )


def _measurement_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    return (
        args.repository_root.resolve(strict=True),
        args.checkpoint.resolve(strict=True),
        args.tokenizer_dir.resolve(strict=True),
        args.tensor_map.resolve(strict=True),
        args.bindings.resolve(strict=True),
        args.controller_vector_dir.resolve(strict=True),
        args.controller_binary.resolve(strict=True),
        args.output_dir.resolve(),
        args.rtl_output_dir.resolve(),
    )


def generate(args: argparse.Namespace) -> dict[str, Any]:
    (
        repository_root,
        checkpoint,
        tokenizer_dir,
        tensor_map,
        bindings,
        controller_vector_dir,
        controller_binary,
        output_dir,
        rtl_output_dir,
    ) = _measurement_paths(args)
    for path in (output_dir, rtl_output_dir):
        try:
            path.relative_to(repository_root)
        except ValueError as error:
            raise MeasurementError(
                f"measurement path is outside the ACE-3 worktree: {path}"
            ) from error
    require(
        not output_dir.exists() or not any(output_dir.iterdir()),
        "measurement output directory is not empty",
    )
    require(not rtl_output_dir.exists(), "RTL output directory already exists")
    authenticate_checkpoint(checkpoint)
    authenticate_tokenizer(tokenizer_dir)
    before = runtime_state()

    logs_dir = output_dir / "logs"
    software_dir = output_dir / "software-dialogue"
    controller_raw_dir = output_dir / "controller-raw"
    logs_dir.mkdir(parents=True)
    software_wall = run_timed_command(
        [
            sys.executable,
            str(repository_root / "ace3/model/official_model24_dialogue.py"),
            "generate",
            "--output-dir",
            str(software_dir),
            "--official-checkpoint",
            str(checkpoint),
            "--official-tokenizer-dir",
            str(tokenizer_dir),
        ],
        [
            "python3",
            "ace3/model/official_model24_dialogue.py",
            "generate",
            "--output-dir",
            "build/model24_latency/software-dialogue",
            "--official-checkpoint",
            checkpoint.relative_to(repository_root).as_posix(),
            "--official-tokenizer-dir",
            tokenizer_dir.relative_to(repository_root).as_posix(),
        ],
        repository_root,
        logs_dir / "software-dialogue.log",
    )
    software_artifacts, software_summary = software_records(
        software_dir,
        repository_root,
        tokenizer_dir,
    )

    controller_raw_dir.mkdir(parents=True)
    controller_wall = run_timed_command(
        [
            str(controller_binary),
            f"+VECTOR_DIR={controller_vector_dir}",
            f"+RAW_DIR={controller_raw_dir}",
        ],
        [
            controller_binary.relative_to(repository_root).as_posix(),
            "+VECTOR_DIR=build/model24_layer_controller",
            "+RAW_DIR=build/model24_latency/controller-raw",
        ],
        repository_root,
        logs_dir / "rtl-controller.log",
    )
    controller_log = (logs_dir / "rtl-controller.log").read_text(encoding="utf-8")
    controller_cycles = parse_controller_cycles(controller_log)
    parse_simulation_terminal(controller_raw_dir / "terminal.txt")
    require(
        parse_controller_events(controller_raw_dir / "controller_events.hex")
        == list(range(LAYER_COUNT)),
        "measured controller launch order mismatch",
    )

    rtl_wall = run_timed_command(
        [
            sys.executable,
            str(repository_root / "ace3/model/controller_model24_rtl_cascade.py"),
            "--repository-root",
            str(repository_root),
            "--checkpoint",
            str(checkpoint),
            "--tensor-map",
            str(tensor_map),
            "--bindings",
            str(bindings),
            "--simulation-dir",
            str(controller_raw_dir),
            "--output-dir",
            str(rtl_output_dir),
        ],
        [
            "python3",
            "ace3/model/controller_model24_rtl_cascade.py",
            "--repository-root",
            ".",
            "--checkpoint",
            checkpoint.relative_to(repository_root).as_posix(),
            "--tensor-map",
            tensor_map.relative_to(repository_root).as_posix(),
            "--bindings",
            bindings.relative_to(repository_root).as_posix(),
            "--simulation-dir",
            "build/model24_latency/controller-raw",
            "--output-dir",
            rtl_output_dir.relative_to(repository_root).as_posix(),
        ],
        repository_root,
        logs_dir / "rtl-cascade.log",
    )
    rtl_execution = load_json(rtl_output_dir / "execution.json", "RTL execution")
    require(
        rtl_execution["post_layer23"]["within_tolerance"]
        and len(rtl_execution["layers"]) == LAYER_COUNT,
        "RTL cascade execution did not complete within tolerance",
    )
    layers = layer_cycle_records(rtl_output_dir, repository_root)
    after = runtime_state()

    rtl_artifacts = {
        "execution": file_record(
            rtl_output_dir / "execution.json", repository_root
        ),
        "manifest": file_record(rtl_output_dir / "manifest.json", repository_root),
        "terminal": file_record(rtl_output_dir / "terminal.txt", repository_root),
        "layers": [
            {
                name: record[name]
                for name in (
                    "simulation_log",
                    "compile_log",
                    "record",
                    "simulator_binary",
                )
            }
            for record in layers
        ],
    }
    document = {
        "schema_version": 1,
        "kind": KIND,
        "profile": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            "model_revision": "db09cd27ead7fee40cdee309693cf83601b9c899",
            "software": (
                "one fresh fixed-prompt four-token greedy dialogue continuation "
                "through 24 software/oracle layers and the tied head"
            ),
            "rtl": (
                "one fresh controller launch trace followed by 24 separately "
                "indexed two-token decoder Verilator simulations"
            ),
            "runs": 1,
            "warmup_runs": 0,
            "wall_clock": "time.perf_counter around each child process",
        },
        "runtime_before": before,
        "runtime_after": after,
        "bindings": {
            "checkpoint": file_record(checkpoint, repository_root),
            "tokenizer": {
                "tokenizer.json": {
                    **file_record(
                        tokenizer_dir / "tokenizer.json", repository_root
                    ),
                    "accepted_sha256": TOKENIZER_SHA256,
                },
                "tokenizer_config.json": {
                    **file_record(
                        tokenizer_dir / "tokenizer_config.json", repository_root
                    ),
                    "accepted_sha256": TOKENIZER_CONFIG_SHA256,
                },
            },
            "tensor_map": file_record(tensor_map, repository_root),
            "controller_bindings": file_record(bindings, repository_root),
            "controller_binary": file_record(controller_binary, repository_root),
            "sources": source_records(repository_root),
            "software_artifacts": software_artifacts,
            "controller_artifacts": {
                "events": file_record(
                    controller_raw_dir / "controller_events.hex",
                    repository_root,
                ),
                "terminal": file_record(
                    controller_raw_dir / "terminal.txt", repository_root
                ),
            },
            "rtl_artifacts": rtl_artifacts,
        },
        "measurements": {
            "software_dialogue_continuation": {
                "wall_seconds_diagnostic": software_wall,
                "generated_tokens": software_summary["steps"],
                "generated_token_ids": software_summary["generated_token_ids"],
                "decoded_text": software_summary["decoded_text"],
                "stop_reason": software_summary["stop_reason"],
                "cycle_evidence": "not applicable to the software host process",
            },
            "rtl_controller_cascade": {
                "wall_seconds_diagnostic": controller_wall + rtl_wall,
                "controller_wall_seconds_diagnostic": controller_wall,
                "decoder_wall_seconds_diagnostic": rtl_wall,
                "controller_scheduler_cycles": controller_cycles,
                "aggregate_decoder_harness_cycles": sum(
                    record["cycles"] for record in layers
                ),
                "aggregate_token0_cycles": sum(
                    record["token0_cycles"] for record in layers
                ),
                "aggregate_token1_cycles": sum(
                    record["token1_cycles"] for record in layers
                ),
                "aggregate_stall_cycles": sum(
                    record["stall_cycles"] for record in layers
                ),
                "layers": [
                    {
                        key: value
                        for key, value in record.items()
                        if key
                        not in {
                            "simulation_log",
                            "compile_log",
                            "record",
                            "simulator_binary",
                        }
                    }
                    for record in layers
                ],
            },
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    validate_measurement_document(document)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / MEASUREMENT_NAME).write_bytes(canonical_json(document))

    internal_artifacts = {
        path.relative_to(output_dir).as_posix(): {
            key: value
            for key, value in file_record(path, repository_root).items()
            if key in {"bytes", "sha256"}
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME
    }
    external_records = [
        rtl_artifacts[name] for name in ("execution", "manifest", "terminal")
    ]
    for layer in rtl_artifacts["layers"]:
        external_records.extend(layer.values())
    external_artifacts = {
        record["path"]: {
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        for record in external_records
    }
    manifest = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "artifacts": internal_artifacts,
        "external_rtl_artifacts": external_artifacts,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / MANIFEST_NAME).write_bytes(canonical_json(manifest))
    return document


def validate(args: argparse.Namespace) -> dict[str, Any]:
    (
        repository_root,
        checkpoint,
        tokenizer_dir,
        tensor_map,
        bindings,
        _controller_vector_dir,
        controller_binary,
        output_dir,
        rtl_output_dir,
    ) = _measurement_paths(args)
    authenticate_checkpoint(checkpoint)
    tokenizer = authenticate_tokenizer(tokenizer_dir)
    document = load_json(output_dir / MEASUREMENT_NAME, "latency measurement")
    validate_measurement_document(document)
    dialogue = load_json(
        output_dir / "software-dialogue" / DIALOGUE_ARTIFACT_NAME,
        "dialogue evidence",
    )
    validate_dialogue_document(dialogue, tokenizer)
    manifest = load_json(output_dir / MANIFEST_NAME, "latency manifest")
    require(
        manifest.get("schema_version") == 1
        and manifest.get("kind") == MANIFEST_KIND
        and manifest.get("claim_boundary") == CLAIM_BOUNDARY,
        "latency manifest identity mismatch",
    )

    internal = {
        path.relative_to(output_dir).as_posix(): {
            key: value
            for key, value in file_record(path, repository_root).items()
            if key in {"bytes", "sha256"}
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME
    }
    require(
        manifest.get("artifacts") == internal,
        "latency artifact coverage mismatch",
    )
    binding = document["bindings"]
    require(
        binding["checkpoint"]
        == {
            "path": checkpoint.relative_to(repository_root).as_posix(),
            "bytes": CHECKPOINT_SIZE,
            "sha256": CHECKPOINT_SHA256,
        },
        "checkpoint measurement binding mismatch",
    )
    require(
        binding["sources"] == source_records(repository_root),
        "measurement source binding mismatch",
    )
    expected_tokenizer = {
        "tokenizer.json": {
            **file_record(tokenizer_dir / "tokenizer.json", repository_root),
            "accepted_sha256": TOKENIZER_SHA256,
        },
        "tokenizer_config.json": {
            **file_record(
                tokenizer_dir / "tokenizer_config.json", repository_root
            ),
            "accepted_sha256": TOKENIZER_CONFIG_SHA256,
        },
    }
    for name, expected in (
        ("tokenizer", expected_tokenizer),
        ("tensor_map", file_record(tensor_map, repository_root)),
        ("controller_bindings", file_record(bindings, repository_root)),
        ("controller_binary", file_record(controller_binary, repository_root)),
    ):
        require(binding[name] == expected, f"{name} measurement binding mismatch")
    expected_software, _summary = software_records(
        output_dir / "software-dialogue",
        repository_root,
        tokenizer_dir,
    )
    require(
        binding["software_artifacts"] == expected_software,
        "software artifact measurement binding mismatch",
    )
    parse_simulation_terminal(output_dir / "controller-raw" / "terminal.txt")
    require(
        parse_controller_events(
            output_dir / "controller-raw" / "controller_events.hex"
        )
        == list(range(LAYER_COUNT)),
        "validated controller launch order mismatch",
    )
    layers = layer_cycle_records(rtl_output_dir, repository_root)
    measured_layers = document["measurements"]["rtl_controller_cascade"]["layers"]
    for expected, measured in zip(layers, measured_layers, strict=True):
        for key, value in measured.items():
            require(expected[key] == value, f"layer {expected['layer_index']} {key} changed")

    rtl_artifacts = binding["rtl_artifacts"]
    external_records = [
        rtl_artifacts[name] for name in ("execution", "manifest", "terminal")
    ]
    for layer in rtl_artifacts["layers"]:
        external_records.extend(layer.values())
    expected_external = {
        record["path"]: {
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        for record in external_records
    }
    require(
        manifest.get("external_rtl_artifacts") == expected_external,
        "external RTL artifact hash mismatch",
    )
    for path, expected in expected_external.items():
        actual = file_record(repository_root / path, repository_root)
        require(
            {
                "bytes": actual["bytes"],
                "sha256": actual["sha256"],
            }
            == expected,
            f"latency artifact hash mismatch: {path}",
        )
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("generate", "validate"))
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--controller-vector-dir", required=True, type=Path)
    parser.add_argument("--controller-binary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rtl-output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        document = generate(args) if args.operation == "generate" else validate(args)
    except (KeyError, OSError, ValueError, MeasurementError) as error:
        raise SystemExit(
            f"MODEL24_LATENCY_{args.operation.upper()}_FAIL {error}"
        ) from error
    software = document["measurements"]["software_dialogue_continuation"]
    rtl = document["measurements"]["rtl_controller_cascade"]
    print(
        f"MODEL24_LATENCY_{args.operation.upper()}_PASS "
        f"software_wall_seconds={software['wall_seconds_diagnostic']:.6f} "
        f"rtl_wall_seconds={rtl['wall_seconds_diagnostic']:.6f} "
        f"controller_cycles={rtl['controller_scheduler_cycles']} "
        f"decoder_harness_cycles={rtl['aggregate_decoder_harness_cycles']} "
        "evidence=diagnostic_only synthesis=not_run fpga=not_run "
        "hardware_latency=not_measured hardware_throughput=not_measured"
    )


if __name__ == "__main__":
    main()
