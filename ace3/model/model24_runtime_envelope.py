#!/usr/bin/env python3
"""Capture a read-only runtime envelope from an existing partial RTL cascade."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from measure_model24_latency import LAYER_PASS

KIND = "ace3_model24_cpu_contention_runtime_envelope"
LAYER_COUNT = 24
MINIMUM_COMPLETED_LAYERS = 3


class RuntimeEnvelopeError(RuntimeError):
    """Raised when partial runtime evidence is incomplete or ambiguous."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeEnvelopeError(message)


def canonical_json(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def file_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def pressure_record(path: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        records[fields[0]] = {
            key: (int(value) if key == "total" else float(value))
            for key, value in (field.split("=", 1) for field in fields[1:])
        }
    return records


def parse_completed_layer_cycles(text: str, expected_layer: int) -> int:
    matches = [
        line
        for line in text.splitlines()
        if line.startswith("DECODER_LAYER_TOKEN_ENGINE_VERILATOR_PASS")
    ]
    require(len(matches) == 1, f"layer {expected_layer} has an ambiguous PASS")
    match = LAYER_PASS.fullmatch(matches[0])
    require(match is not None, f"layer {expected_layer} has an ambiguous PASS")
    require(
        int(match.group("layer")) == expected_layer,
        f"layer PASS index mismatch: {expected_layer}",
    )
    require(
        int(match.group("trace")) == 46676
        and int(match.group("final")) == 1792,
        f"layer {expected_layer} output counts mismatch",
    )
    cycles = int(match.group("cycles"))
    require(cycles > 0, f"layer {expected_layer} cycle evidence is vacuous")
    return cycles


def runtime_snapshot() -> dict[str, Any]:
    load_1m, load_5m, load_15m = os.getloadavg()
    states: dict[str, int] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            state = stat_path.read_text(encoding="ascii").rsplit(") ", 1)[1][0]
        except (OSError, IndexError):
            continue
        states[state] = states.get(state, 0) + 1
    return {
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "load_average": [load_1m, load_5m, load_15m],
        "runnable_processes": states.get("R", 0),
        "uninterruptible_processes": states.get("D", 0),
        "pressure": {
            resource: pressure_record(Path(f"/proc/pressure/{resource}"))
            for resource in ("cpu", "io", "memory")
        },
    }


def _utc_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def validate_runtime_envelope(document: Mapping[str, Any]) -> None:
    require(document.get("schema_version") == 1, "envelope schema mismatch")
    require(document.get("kind") == KIND, "envelope kind mismatch")
    layers = document.get("completed_layer_timings")
    require(
        isinstance(layers, list)
        and len(layers) >= MINIMUM_COMPLETED_LAYERS
        and [record.get("layer_index") for record in layers]
        == list(range(len(layers))),
        "completed layer timing coverage is insufficient",
    )
    measured = [
        record["inter_record_seconds"]
        for record in layers[1:]
    ]
    require(
        measured and all(value > 0.0 and math.isfinite(value) for value in measured),
        "inter-record timing evidence is vacuous",
    )
    summary = document.get("summary")
    require(isinstance(summary, Mapping), "runtime summary is missing")
    require(
        math.isclose(summary["median_layer_seconds"], statistics.median(measured))
        and math.isclose(summary["maximum_layer_seconds"], max(measured)),
        "runtime summary statistics mismatch",
    )
    require(
        summary["authorized_timeout_seconds"]
        >= summary["projected_24_layer_upper_seconds"] * 1.25,
        "authorized timeout lacks the declared margin",
    )
    require(
        document.get("claim_boundary", {}).get("full_24_layer_run")
        == "not launched by this capture",
        "full-run claim boundary mismatch",
    )


def build_runtime_envelope(
    rtl_output_dir: Path,
    *,
    snapshot: Mapping[str, Any] | None = None,
    captured_at_utc: str | None = None,
) -> dict[str, Any]:
    rtl_output_dir = rtl_output_dir.resolve(strict=True)
    initial_path = rtl_output_dir / "initial_embeddings.hex"
    require(initial_path.is_file(), "initial embedding handoff is missing")
    layers_dir = rtl_output_dir / "layers"
    records: list[dict[str, Any]] = []
    previous_end = initial_path.stat().st_mtime
    for layer_id in range(LAYER_COUNT):
        layer_dir = layers_dir / f"layer{layer_id:02d}"
        record_path = layer_dir / "record.json"
        if not record_path.is_file():
            break
        compile_path = layer_dir / "compile.log"
        simulation_path = layer_dir / "simulation.log"
        require(
            compile_path.is_file() and simulation_path.is_file(),
            f"layer {layer_id} phase logs are missing",
        )
        record = json.loads(record_path.read_text(encoding="ascii"))
        require(record.get("layer_index") == layer_id, "layer record index mismatch")
        simulator_cycles = parse_completed_layer_cycles(
            simulation_path.read_text(encoding="utf-8"),
            layer_id,
        )
        compile_end = compile_path.stat().st_mtime
        simulation_end = simulation_path.stat().st_mtime
        record_end = record_path.stat().st_mtime
        require(
            previous_end <= compile_end <= simulation_end <= record_end,
            f"layer {layer_id} phase timestamps are out of order",
        )
        total_seconds = record_end - previous_end
        simulation_seconds = simulation_end - compile_end
        records.append(
            {
                "layer_index": layer_id,
                "start_utc": _utc_from_timestamp(previous_end),
                "record_utc": _utc_from_timestamp(record_end),
                "inter_record_seconds": total_seconds,
                "pre_simulation_seconds": compile_end - previous_end,
                "simulation_seconds": simulation_seconds,
                "post_simulation_seconds": record_end - simulation_end,
                "simulation_share": simulation_seconds / total_seconds,
                "simulator_cycles": simulator_cycles,
                "legacy_absolute_comparison_within_tolerance": (
                    record.get("comparison", {}).get("within_tolerance") is True
                ),
                "artifacts": {
                    "compile_log": file_record(compile_path),
                    "simulation_log": file_record(simulation_path),
                    "record": file_record(record_path),
                },
            }
        )
        previous_end = record_end
    require(
        len(records) >= MINIMUM_COMPLETED_LAYERS,
        "at least three contiguous completed layers are required",
    )
    require(
        not any(
            (layers_dir / f"layer{layer_id:02d}" / "record.json").exists()
            for layer_id in range(len(records) + 1, LAYER_COUNT)
        ),
        "completed layer records are not contiguous",
    )
    measured = [record["inter_record_seconds"] for record in records[1:]]
    median_seconds = statistics.median(measured)
    maximum_seconds = max(measured)
    first_layer_seconds = records[0]["inter_record_seconds"]
    projected_lower = first_layer_seconds + median_seconds * (LAYER_COUNT - 1)
    projected_upper = first_layer_seconds + maximum_seconds * (LAYER_COUNT - 1)
    authorized_timeout = math.ceil(projected_upper * 1.25 / 3600.0) * 3600
    stall_review = math.ceil(maximum_seconds * 2.0 / 300.0) * 300
    incomplete_layer = None
    next_layer = len(records)
    next_dir = layers_dir / f"layer{next_layer:02d}"
    next_simulation = next_dir / "simulation.log"
    if next_layer < LAYER_COUNT and next_simulation.is_file():
        incomplete_layer = {
            "layer_index": next_layer,
            "last_progress_utc": _utc_from_timestamp(
                next_simulation.stat().st_mtime
            ),
            "observed_elapsed_since_predecessor_seconds": (
                next_simulation.stat().st_mtime - previous_end
            ),
            "natural_terminal": False,
            "cause": "not attributed by this capture",
            "simulation_log": file_record(next_simulation),
        }
    document = {
        "schema_version": 1,
        "kind": KIND,
        "capture_mode": "read-only existing partial cascade artifacts",
        "captured_at_utc": captured_at_utc
        or datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_window": {
            "start_utc": records[0]["start_utc"],
            "last_completed_record_utc": records[-1]["record_utc"],
            "runtime_snapshot_relation": (
                "capture-time contention snapshot; not a historical load trace"
            ),
        },
        "runtime_snapshot": dict(snapshot or runtime_snapshot()),
        "initial_handoff": file_record(initial_path),
        "completed_layer_timings": records,
        "incomplete_layer_observation": incomplete_layer,
        "summary": {
            "completed_layers": len(records),
            "median_layer_seconds": median_seconds,
            "maximum_layer_seconds": maximum_seconds,
            "aggregate_simulation_share": sum(
                record["simulation_seconds"] for record in records[1:]
            )
            / sum(measured),
            "projected_24_layer_lower_seconds": projected_lower,
            "projected_24_layer_upper_seconds": projected_upper,
            "authorized_timeout_seconds": authorized_timeout,
            "per_layer_stall_review_seconds": stall_review,
        },
        "authorization": {
            "later_run": "authorized for supervised scheduling after L2 review",
            "durable_runner_timeout_seconds": authorized_timeout,
            "review_if_no_layer_terminal_seconds": stall_review,
            "require_scale_aware_per_layer_fail_fast": True,
        },
        "claim_boundary": {
            "full_24_layer_run": "not launched by this capture",
            "incomplete_layer_cause": "not attributed",
            "hardware_latency": "not measured",
            "throughput": "not measured",
        },
    }
    validate_runtime_envelope(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtl-output-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = build_runtime_envelope(args.rtl_output_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(document))
    summary = document["summary"]
    print(
        "MODEL24_RUNTIME_ENVELOPE_PASS "
        f"completed_layers={summary['completed_layers']} "
        f"median_layer_seconds={summary['median_layer_seconds']:.3f} "
        f"maximum_layer_seconds={summary['maximum_layer_seconds']:.3f} "
        f"simulation_share={summary['aggregate_simulation_share']:.6f} "
        f"projected_upper_seconds={summary['projected_24_layer_upper_seconds']:.3f} "
        f"authorized_timeout_seconds={summary['authorized_timeout_seconds']} "
        "full_24_layer_run=not_launched"
    )


if __name__ == "__main__":
    main()
