#!/usr/bin/env python3
"""Run the checkpoint controller's 24 launches through compiled decoder RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors import safe_open

from controller_model24_cascade import (
    _canonical_json,
    _check_tensor_against_binding,
    _hidden_comparison,
    _load_json,
    _load_raw_hidden,
    _sha256,
    parse_controller_events,
    parse_simulation_terminal,
    validate_binding_document,
)
from model24_execution_oracle import materialize_indexed_decoder_vectors
from model24_oracle import authenticate_checkpoint
from official_model24_next_token import (
    HIDDEN_SIZE,
    LAYER_COUNT,
    TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
    _bits_to_f16,
    _canonical_bytes,
    _f16_to_bits,
    _load_embeddings,
    _reference_layer,
)

KIND = "ace3_controller_model24_rtl_cascade"
TERMINAL_SCHEMA = "ace3_controller_model24_rtl_cascade_v1"


class RtlCascadeError(RuntimeError):
    """Raised when resumable RTL evidence fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RtlCascadeError(message)


def hash_file(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": _sha256(payload)}


def raw_hidden_payload(hidden: np.ndarray) -> bytes:
    return "".join(
        f"{token:02x}{index:04x}{int(value):04x}\n"
        for token, row in enumerate(hidden)
        for index, value in enumerate(row)
    ).encode("ascii")


def natural_terminal(layer_id: int) -> str:
    if layer_id == 0:
        prefix = "schema=ace3_decoder_layer0_raw_v1"
    else:
        prefix = f"schema=ace3_decoder_layer_raw_v1 layer_index={layer_id}"
    return (
        f"{prefix} natural_terminal=1 exit_code=0 trace_count=46676 "
        "final_count=1792 done_count=2\n"
    )


def source_hashes(repository_root: Path) -> dict[str, str]:
    paths = (
        "ace3/model/controller_model24_rtl_cascade.py",
        "ace3/model/model24_execution_oracle.py",
        "ace3/model/decoder_layer0_oracle.py",
        "ace3/model/fp16_adaptation_oracle.py",
        "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
        "ace3/rtl/ace3_fp16_silu_gate_core.sv",
        "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp",
    )
    return {
        path: _sha256((repository_root / path).read_bytes())
        for path in paths
    }


def validate_completed_layer(
    layer_id: int,
    layer_dir: Path,
    expected_input_sha256: str,
    binding: Mapping[str, Any],
    sources: Mapping[str, str],
) -> dict[str, Any] | None:
    record_path = layer_dir / "record.json"
    if not record_path.exists():
        return None
    try:
        record = _load_json(record_path.read_bytes(), f"layer {layer_id} record")
        require(record.get("layer_index") == layer_id, "layer record index mismatch")
        require(
            record.get("descriptor_sha256") == binding["descriptor_sha256"],
            "layer record descriptor mismatch",
        )
        require(
            record.get("input_raw_sha256") == expected_input_sha256,
            "layer record input lineage mismatch",
        )
        require(record.get("sources") == sources, "layer record source mismatch")
        for name in ("trace.hex", "final.hex", "terminal.txt", "comparison.json"):
            require(
                hash_file(layer_dir / "raw" / name) == record["artifacts"][name],
                f"layer {layer_id} {name} hash mismatch",
            )
        require(
            (layer_dir / "raw" / "terminal.txt").read_text(encoding="ascii")
            == natural_terminal(layer_id),
            f"layer {layer_id} did not reach a natural terminal",
        )
        return record
    except (OSError, KeyError, ValueError, RtlCascadeError):
        return None


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    require(
        completed.returncode == 0,
        f"command failed with exit {completed.returncode}: {' '.join(command)}",
    )


def execute(
    repository_root: Path,
    checkpoint_path: Path,
    tensor_map_path: Path,
    bindings_path: Path,
    simulation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    parse_simulation_terminal(simulation_dir / "terminal.txt")
    layer_order = parse_controller_events(simulation_dir / "controller_events.hex")
    require(layer_order == list(range(LAYER_COUNT)), "controller launch order mismatch")
    bindings_payload = bindings_path.read_bytes()
    bindings = _load_json(bindings_payload, "layer bindings")
    validate_binding_document(bindings, repository_root, tensor_map_path.read_bytes())
    authenticate_checkpoint(checkpoint_path)
    sources = source_hashes(repository_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers_dir = output_dir / "layers"
    layers_dir.mkdir(exist_ok=True)
    initial_path = output_dir / "initial_embeddings.hex"

    torch.set_num_threads(1)
    with safe_open(checkpoint_path, framework="np") as checkpoint:
        embeddings, embedding_record = _load_embeddings(checkpoint)
        initial_bits = _f16_to_bits(embeddings)
        initial_payload = raw_hidden_payload(initial_bits)
        if initial_path.exists():
            require(
                initial_path.read_bytes() == initial_payload,
                "initial embedding handoff changed",
            )
        else:
            initial_path.write_bytes(initial_payload)
        reference_hidden = torch.from_numpy(embeddings.astype(np.float64))
        predecessor_path = initial_path
        predecessor_sha256 = _sha256(initial_payload)
        records = []

        for layer_id in layer_order:
            binding = bindings["layers"][layer_id]
            tensors = {
                record["name"]: np.asarray(checkpoint.get_tensor(record["name"]))
                for record in binding["tensors"]
            }
            consumed = [
                _check_tensor_against_binding(record, tensors[record["name"]])
                for record in binding["tensors"]
            ]
            reference_hidden = _reference_layer(layer_id, tensors, reference_hidden)
            layer_dir = layers_dir / f"layer{layer_id:02d}"
            completed = validate_completed_layer(
                layer_id,
                layer_dir,
                predecessor_sha256,
                binding,
                sources,
            )
            if completed is None:
                if layer_dir.exists():
                    shutil.rmtree(layer_dir)
                vector_dir = layer_dir / "vectors"
                raw_dir = layer_dir / "raw"
                raw_dir.mkdir(parents=True)
                manifest = materialize_indexed_decoder_vectors(
                    checkpoint_path,
                    tensor_map_path,
                    predecessor_path,
                    vector_dir,
                    layer_index=layer_id,
                    expected_handoff_sha256=predecessor_sha256,
                )
                require(
                    manifest["layer_binding"] == {
                        key: binding[key]
                        for key in ("layer_id", "namespace", "descriptor_sha256")
                    },
                    f"layer {layer_id} materialized binding mismatch",
                )
                compile_log = layer_dir / "compile.log"
                run_command(
                    [
                        "make",
                        "--no-print-directory",
                        "model24-rtl-layer-compile",
                        f"MODEL24_RTL_LAYER_INDEX={layer_id}",
                        "MODEL24_RTL_ACCURATE_SILU="
                        + ("1" if layer_id >= 3 else "0"),
                    ],
                    repository_root,
                    compile_log,
                )
                binary = (
                    output_dir
                    / "compiled"
                    / f"layer{layer_id}"
                    / "obj_dir"
                    / "Vace3_decoder_layer0_token_engine"
                )
                require(binary.is_file(), f"layer {layer_id} RTL binary missing")
                run_command(
                    [
                        str(binary),
                        "--layer-index",
                        str(layer_id),
                        "--vector-dir",
                        str(vector_dir),
                        "--raw-dir",
                        str(raw_dir),
                        "--progress-interval",
                        "1000000",
                    ],
                    repository_root,
                    layer_dir / "simulation.log",
                )
                require(
                    (raw_dir / "terminal.txt").read_text(encoding="ascii")
                    == natural_terminal(layer_id),
                    f"layer {layer_id} did not reach a natural terminal",
                )
                require(
                    (raw_dir / "trace.hex").read_bytes()
                    == (vector_dir / "trace.hex").read_bytes(),
                    f"layer {layer_id} RTL trace differs from independent integer oracle",
                )
                require(
                    (raw_dir / "final.hex").read_bytes()
                    == (vector_dir / "final.hex").read_bytes(),
                    f"layer {layer_id} RTL final differs from independent integer oracle",
                )
                produced = _load_raw_hidden((raw_dir / "final.hex").read_bytes())
                reference_values = reference_hidden.detach().cpu().numpy()
                comparison = {
                    "schema_version": 1,
                    "layer_index": layer_id,
                    "independent_oracle": "PyTorch CPU float64 dequantized-AWQ",
                    **_hidden_comparison(produced, reference_values),
                }
                (raw_dir / "comparison.json").write_bytes(
                    _canonical_json(comparison)
                )
                raw_sha256 = _sha256((raw_dir / "final.hex").read_bytes())
                completed = {
                    "schema_version": 1,
                    "layer_index": layer_id,
                    "namespace": binding["namespace"],
                    "descriptor_sha256": binding["descriptor_sha256"],
                    "input_raw_sha256": predecessor_sha256,
                    "output_raw_sha256": raw_sha256,
                    "consumed_tensors": consumed,
                    "numeric_profile": manifest["numeric_profile"],
                    "comparison": comparison,
                    "sources": sources,
                    "artifacts": {
                        name: hash_file(raw_dir / name)
                        for name in (
                            "trace.hex",
                            "final.hex",
                            "terminal.txt",
                            "comparison.json",
                        )
                    },
                }
                (layer_dir / "record.json").write_bytes(
                    _canonical_json(completed)
                )
            else:
                produced = _load_raw_hidden(
                    (layer_dir / "raw" / "final.hex").read_bytes()
                )
                reference_values = reference_hidden.detach().cpu().numpy()
                comparison = _hidden_comparison(produced, reference_values)
                require(
                    completed["comparison"] == {
                        "schema_version": 1,
                        "layer_index": layer_id,
                        "independent_oracle": (
                            "PyTorch CPU float64 dequantized-AWQ"
                        ),
                        **comparison,
                    },
                    f"layer {layer_id} resumed comparison mismatch",
                )
            records.append(completed)
            predecessor_path = layer_dir / "raw" / "final.hex"
            predecessor_sha256 = completed["output_raw_sha256"]
            del tensors

    terminal_comparison = records[-1]["comparison"]
    require(
        terminal_comparison["within_tolerance"],
        "post-layer-23 hidden tolerance exceeded: "
        + ", ".join(
            f"token {token['token_index']}={token['max_abs_error']}"
            for token in terminal_comparison["tokens"]
        ),
    )
    document = {
        "schema_version": 1,
        "kind": KIND,
        "model_binding": bindings["model_binding"],
        "controller": {
            "bindings_sha256": _sha256(bindings_payload),
            "events": layer_order,
            "events_sha256": hash_file(
                simulation_dir / "controller_events.hex"
            )["sha256"],
            "terminal_sha256": hash_file(simulation_dir / "terminal.txt")[
                "sha256"
            ],
        },
        "input": embedding_record,
        "layers": records,
        "post_layer23": {
            "raw_sha256": records[-1]["output_raw_sha256"],
            "max_abs_error": terminal_comparison["max_abs_error"],
            "decision_token_max_abs_error": terminal_comparison["tokens"][1][
                "max_abs_error"
            ],
            "tokens": terminal_comparison["tokens"],
            "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
            "within_tolerance": True,
        },
        "claim_boundary": {
            "demonstrated": "compiled Verilator decoder RTL layers 0 through 23",
            "legacy_layer012": (
                "preserved by the separate default-profile regression"
            ),
            "tokenizer_dialogue": "not produced",
            "tied_lm_head": "not executed",
            "synthesis": "not run",
            "ppa": "not measured",
            "fpga": "not run",
            "latency": "not measured",
            "throughput": "not measured",
        },
    }
    execution_payload = _canonical_json(document)
    (output_dir / "execution.json").write_bytes(execution_payload)
    manifest = {
        "schema_version": 1,
        "kind": "ace3_controller_model24_rtl_cascade_manifest",
        "execution": {
            "bytes": len(execution_payload),
            "sha256": _sha256(execution_payload),
        },
        "completed_layers": LAYER_COUNT,
        "consumed_tensors": sum(
            len(record["consumed_tensors"]) for record in records
        ),
        "terminal_layer": 23,
        "max_abs_error": terminal_comparison["max_abs_error"],
        "decision_token_max_abs_error": terminal_comparison["tokens"][1][
            "max_abs_error"
        ],
        "tokens": terminal_comparison["tokens"],
        "absolute_tolerance": TERMINAL_HIDDEN_ABSOLUTE_TOLERANCE,
        "within_tolerance": True,
    }
    (output_dir / "manifest.json").write_bytes(_canonical_json(manifest))
    (output_dir / "terminal.txt").write_text(
        f"schema={TERMINAL_SCHEMA} natural_terminal=1 exit_code=0 "
        f"completed_layers={LAYER_COUNT} terminal_layer=23\n",
        encoding="ascii",
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tensor-map", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--simulation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    document = execute(
        args.repository_root.resolve(strict=True),
        args.checkpoint.resolve(strict=True),
        args.tensor_map.resolve(strict=True),
        args.bindings.resolve(strict=True),
        args.simulation_dir.resolve(strict=True),
        args.output_dir.resolve(),
    )
    print(
        "MODEL24_CONTROLLER_RTL_CASCADE_PASS "
        f"layers={len(document['layers'])} terminal_layer=23 "
        f"max_abs_error={document['post_layer23']['max_abs_error']} "
        f"token0_max_abs_error="
        f"{document['post_layer23']['tokens'][0]['max_abs_error']} "
        f"token1_max_abs_error="
        f"{document['post_layer23']['tokens'][1]['max_abs_error']} "
        "tokenizer_dialogue=not_produced tied_lm_head=not_executed "
        "synthesis=not_run ppa=not_measured fpga=not_run "
        "latency=not_measured throughput=not_measured"
    )


if __name__ == "__main__":
    main()
