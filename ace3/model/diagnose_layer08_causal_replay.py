#!/usr/bin/env python3
"""Controlled upstream substitutions, gated by an exact preserved-RTL replay."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import decoder_layer0_oracle as decoder
from awq_bit_oracle import AWQ_REVERSE_ORDER
from diagnose_layer01_position2_stage08 import authenticated_array, score_matrix
from diagnose_layer08_position2_stage08 import (
    ATTEMPT, REFERENCE, ROOT, detail, file_record, load_hex, load_json, require,
    sha256_bytes, write_json,
)
from run_corrected_q_selected_token_frontier import seal
from run_repaired_q_selected_token_frontier import trace_stage, verify_file_record
from validate_selected_token_position2_traversal import load_hidden_bits


def run(output: Path) -> dict:
    require(output.parent == ROOT / "build", "output must be a fresh build attempt")
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    sources = [file_record(Path(__file__).resolve())]
    sources.extend(file_record(ROOT / "ace3/model" / name) for name in (
        "decoder_layer0_oracle.py", "accepted_awq_projection.py", "awq_bit_oracle.py",
        "projection_oracle.py", "attention_oracle.py", "fp16_adaptation_oracle.py",
        "qwen2_rope_oracle.py", "diagnose_layer01_position2_stage08.py",
        "diagnose_layer08_position2_stage08.py", "run_corrected_q_selected_token_frontier.py",
        "run_repaired_q_selected_token_frontier.py",
        "validate_selected_token_position2_traversal.py",
    ))
    parent = load_json(ATTEMPT / "result.json")
    execution = load_json(ATTEMPT / "execution.json")
    frozen = load_json(verify_file_record(execution["frozen"]))
    bound_refs = {item["path"]: item for item in frozen["independent_references"]}
    inputs = [file_record(ATTEMPT / "result.json"), file_record(ATTEMPT / "execution.json"),
              file_record(verify_file_record(execution["frozen"]))]

    def authenticate(record: dict) -> Path:
        path = verify_file_record(record)
        inputs.append(file_record(path))
        return path

    def reference(layer: int, stage: int) -> np.ndarray:
        path = REFERENCE / f"layer{layer:02d}_generation1.json"
        record = load_json(authenticate(bound_refs[str(path)]))
        suffix = f"/layer{layer:02d}/position002/stage{stage:02d}.npy"
        matches = [item for item in record["stages"] if item["path"].endswith(suffix)]
        require(len(matches) == 1, f"ambiguous independent reference: {suffix}")
        result = authenticated_array(matches[0]).reshape(-1)
        inputs.append(file_record(Path(matches[0]["path"])))
        return result

    tensors, actual, hidden_inputs, expected_final = {}, {}, {}, {}
    for layer in range(9):
        record = load_json(authenticate(parent["layers"][layer]))
        require(record["layer_index"] == layer and record["actual_output_fed_rtl_chain"],
                "parent is not an ordered actual-output-fed chain")
        require([p["position"] for p in record["positions"]] == [0, 1, 2],
                "parent position history mismatch")
        tensors[layer] = {}
        for tensor_record in record["positions"][2]["vectors"]["tensors"]:
            tensor = tensor_record["checkpoint_tensor"]
            path = authenticate(tensor_record["serialized"])
            value = load_hex(path, 4 if tensor["dtype"] == "F16" else 8)
            require(value.nbytes == tensor["bytes"]
                    and sha256_bytes(value.tobytes()) == tensor["sha256"],
                    f"checkpoint tensor serialization mismatch: {tensor['name']}")
            name = tensor["name"].replace(f"model.layers.{layer}.", "model.layers.0.") + ":"
            tensors[layer][name] = (
                value.tolist() if name.endswith("layernorm.weight:") else value
            )
        actual[layer], hidden_inputs[layer] = {}, []
        for position, transaction in enumerate(record["positions"]):
            terminal = authenticate(transaction["raw"]["terminal"]).read_text().split()
            fields = dict(item.split("=", 1) for item in terminal)
            require(len(fields) == len(terminal)
                    and fields["natural_terminal"] == "1" and fields["exit_code"] == "0"
                    and fields["done_count"] == "1" and fields["final_count"] == "896"
                    and fields["layer_index"] == str(layer)
                    and fields["position"] == str(position), "invalid parent terminal")
            trace = authenticate(transaction["raw"]["trace"]).read_bytes()
            require(len(trace.splitlines()) == transaction["raw"]["trace_count"],
                    "parent trace length mismatch")
            stages = {}
            for stage in range(19):
                count = 4864 if stage in (14, 15, 16) else (
                    128 if stage in (2, 3, 5, 6, 7) else (
                        14 * (position + 1) if stage in (8, 9) else 896))
                stages[stage] = trace_stage(
                    trace, position, stage, count,
                    position + 1 if stage in (8, 9) else None)
            actual[layer][position] = stages
            incoming = load_hidden_bits(authenticate(transaction["vectors"]["input"]))
            require(sha256_bytes(incoming.tobytes()) == transaction["input"]["sha256"],
                    "input semantic hash mismatch")
            if layer:
                require(np.array_equal(incoming, actual[layer - 1][position][18]),
                        "broken inherited hidden-state boundary")
            hidden_inputs[layer].append(incoming)
        if layer < 8:
            expected_final[layer] = reference(layer, 18)
    expected_scores = reference(8, 8)
    write_json(output / "frozen.json", {
        "argv": [sys.executable, "-B", *sys.argv], "sources": sources, "inputs": inputs,
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "parent_policy": frozen["policy"],
        "comparison_policy": load_json(REFERENCE / "POLICY.json")["comparison"],
        "experiments": [
            "exact all-stage replay of layers00-07 and positions0-2",
            "replace selected-token incoming hidden at each boundary, replay suffix",
            "single-round K/V together, separately, and at individual layer operators",
        ],
        "endpoint": "layer08 selected-token Q/score with preserved layer08 K held fixed",
        "boundary": "Numerical counterfactuals only; no new RTL execution or repair",
    })
    # Cached binary64 weights avoid the scalar oracle's per-output packed-word traversal.
    weights = {}
    current_layer = 0
    intervention: tuple[int | None, str] | None = None
    active_position = 0

    def module(values, prefix, activation, count, bias=None, accepted_q_projection=True):
        key = (current_layer, prefix)
        if key not in weights:
            namespace = f"model.layers.0.{prefix}"
            width = len(activation)
            groups, packed = width // 128, count // 8
            qw = values[namespace + ".qweight:"].reshape(width, packed)
            qz = values[namespace + ".qzeros:"].reshape(groups, packed)
            shifts = np.asarray(AWQ_REVERSE_ORDER, dtype=np.uint32) * 4
            unpacked = ((qw[..., None] >> shifts) & 15).reshape(width, count)
            zeros = ((qz[..., None] >> shifts) & 15).reshape(groups, count)
            scales = values[namespace + ".scales:"].view("<f2").astype(np.float64)
            weights[key] = torch.from_numpy(
                (unpacked.astype(np.float64) - np.repeat(zeros, 128, axis=0))
                * np.repeat(scales.reshape(groups, count), 128, axis=0))
        activation = np.asarray(activation, dtype="<u2").view("<f2").astype(np.float64)
        result = (torch.from_numpy(activation) @ weights[key]).numpy()
        single = prefix == "self_attn.q_proj" and accepted_q_projection
        if intervention is not None:
            target_layer, kind = intervention
            single |= (target_layer is None or target_layer == current_layer) and (
                prefix == f"self_attn.{kind}_proj"
                or (kind == "kv" and prefix in ("self_attn.k_proj", "self_attn.v_proj")))
        if bias is not None:
            if not single:
                result = result.astype("<f2").astype(np.float64)
            result = result + np.asarray(bias, dtype="<u2").view("<f2").astype(np.float64)
        require(np.isfinite(result).all(),
                f"nonfinite diagnostic projection: {current_layer}/{active_position}/{prefix}")
        return result.astype("<f2").view("<u2").tolist()

    original_module = decoder._module
    decoder._module = module
    rows = []

    def replay(name: str, start: int, boundary: bool = False,
               change: tuple[int | None, str] | None = None, exact: bool = False) -> dict:
        nonlocal current_layer, intervention, active_position
        intervention = change
        hidden = [value.copy() for value in hidden_inputs[start]]
        if boundary:
            require(start > 0, "no predecessor reference before layer00")
            hidden[2] = expected_final[start - 1].copy()
        began = time.monotonic()
        boundaries = []
        for layer in range(start, 8):
            current_layer = layer
            cache_k, cache_v = [], []
            for position in range(3):
                active_position = position
                final, trace = decoder.run_token(
                    tensors[layer], hidden[position].tolist(), position, cache_k, cache_v,
                    accurate_silu=True, accepted_q_projection=True)
                if exact:
                    for stage in range(19):
                        values = np.asarray([value for s, _, value, _ in trace if s == stage],
                                            dtype="<u2")
                        # RoPE emits split-half pairs, unlike the normalized trace arrays.
                        if stage in (4, 5):
                            indexed = [(index, value) for s, index, value, _ in trace if s == stage]
                            values = np.asarray([value for _, value in sorted(indexed)], dtype="<u2")
                        compared = detail(values, actual[layer][position][stage])
                        if compared["different_count"]:
                            write_json(output / "replay_failure.json", {
                                "layer": layer, "position": position, "stage": stage,
                                "comparison": compared, "taxonomy": "diagnostic_replay_mismatch",
                                "hypothesis": "accelerated numerical replay differs from the preserved RTL policy",
                                "regression": "exact same-input all-stage replay before any counterfactual",
                            })
                            require(False, f"replay differs at {layer}/{position}/{stage}")
                hidden[position] = np.asarray(final, dtype="<u2")
            boundaries.append({"layer": layer, "final": detail(hidden[2], expected_final[layer])})
        current_layer, active_position = 8, 2
        norm = decoder._expect_finite(decoder.rmsnorm(
            hidden[2].tolist(), tensors[8]["model.layers.0.input_layernorm.weight:"])[0], "norm")
        q = module(tensors[8], "self_attn.q_proj", norm, 896,
                   tensors[8]["model.layers.0.self_attn.q_proj.bias:"])
        rotated = np.asarray(decoder._rope(q, 14, 2), dtype="<u2")
        fixed_k = np.stack([actual[8][p][6].reshape(2, 64) for p in range(3)])
        scores = score_matrix(rotated, fixed_k)
        if exact:
            require(np.array_equal(scores, actual[8][2][8]), "baseline endpoint replay differs")
        row = {"name": name, "start_layer": start, "boundaries": boundaries,
               "layer08_fixed_k_scores": detail(scores, expected_scores),
               "score_bits": [f"{int(value):04x}" for value in scores],
               "elapsed_seconds": time.monotonic() - began}
        write_json(output / f"{name}.json", row)
        rows.append(row)
        print(f"{name}: score_failures={row['layer08_fixed_k_scores']['failure_count']}",
              flush=True)
        return row

    try:
        replay("baseline", 0, exact=True)
        for start in range(7, 0, -1):
            replay(f"reference_boundary{start:02d}", start, boundary=True)
        for kind in ("kv", "k", "v"):
            replay(f"single_round_{kind}_all", 0, change=(None, kind))
        for layer in range(8):
            for kind in ("k", "v"):
                replay(f"single_round_{kind}_layer{layer:02d}", layer, change=(layer, kind))
    finally:
        decoder._module = original_module
    for record in sources + inputs:
        verify_file_record(record)
    return {
        "status": "CONTROLLED_NUMERICAL_REPLAY_COMPLETE", "experiments": rows,
        "exact_baseline_layers": list(range(8)), "exact_baseline_positions": [0, 1, 2],
        "elapsed_seconds": time.monotonic() - started,
        "actual_output_fed_rtl_chain": False, "fresh_rtl_executed": False,
        "repair_applied": False, "independent_review": "pending Host Reviewer",
        "boundary": (
            "Counterfactual operator policies are diagnostic, not accepted repairs. "
            "The endpoint holds layer08 K fixed; only a fresh full RTL traversal can close acceptance."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "attempt already exists")
    try:
        result = run(output)
    except (RuntimeError, ValueError, OSError, AssertionError) as error:
        if output.is_dir():
            seal(output, {
                "status": "DIAGNOSTIC_FAILED", "error": str(error),
                "taxonomy": "diagnostic_execution_or_replay_failure",
                "hypothesis": "bound inputs or numerical replay do not meet the exact baseline gate",
                "regression": "inspect preserved failure before a fresh diagnostic attempt",
                "rtl_correctness_conclusion": "none",
            })
        raise
    seal(output, result)
