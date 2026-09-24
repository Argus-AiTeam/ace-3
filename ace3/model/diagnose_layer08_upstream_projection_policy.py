#!/usr/bin/env python3
"""Distinguish upstream Q/K/V rounding drift from layer08-local defects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from accepted_awq_projection import project_fp16_stage
from diagnose_layer08_position2_stage08 import (
    ATTEMPT, ROOT, detail, file_record, load_hex, load_json, require,
    sha256_bytes, write_json,
)
from run_repaired_q_selected_token_frontier import trace_stage, verify_file_record


def run(output: Path) -> dict:
    require(output.parent == ROOT / "build", "output must be a fresh build attempt")
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    parent = load_json(ATTEMPT / "result.json")
    frozen = load_json(verify_file_record(load_json(ATTEMPT / "execution.json")["frozen"]))
    sources = [
        file_record(Path(__file__).resolve()),
        *[file_record(ROOT / "ace3/model" / name) for name in (
            "accepted_awq_projection.py",
            "diagnose_layer08_position2_stage08.py",
            "diagnose_layer01_position2_stage08.py",
            "run_repaired_q_selected_token_frontier.py",
            "replay_layer00_q_projection_repair.py",
        )],
    ]
    write_json(output / "frozen.json", {
        "argv": [sys.executable, "-B", *sys.argv],
        "sources": sources,
        "parent": file_record(ATTEMPT / "result.json"),
        "parent_frozen": file_record(ATTEMPT / "frozen.json"),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "layers": list(range(8)),
        "positions": [0, 1, 2],
        "policies": {
            "single_round": "exact native AWQ dot plus FP16 bias, then FP16 RNE",
            "double_round": "FP16 RNE dot, add FP16 bias, then FP16 RNE",
        },
        "parent_policy": frozen["policy"],
        "boundary": "Numerical same-operand diagnostic; no fresh RTL traversal",
    })
    rows = []
    inputs = []
    for layer in range(8):
        layer_path = verify_file_record(parent["layers"][layer])
        result = load_json(layer_path)
        require(result["layer_index"] == layer, "parent layer ordering mismatch")
        require(result["actual_output_fed_rtl_chain"] is True, "parent chain flag missing")
        inputs.append(file_record(layer_path))
        tensors = {}
        for record in result["positions"][2]["vectors"]["tensors"]:
            tensor = record["checkpoint_tensor"]
            if not any(f"self_attn.{name}_proj." in tensor["name"] for name in "qkv"):
                continue
            path = verify_file_record(record["serialized"])
            value = load_hex(path, 4 if tensor["dtype"] == "F16" else 8)
            require(value.nbytes == tensor["bytes"]
                    and sha256_bytes(value.tobytes()) == tensor["sha256"],
                    f"checkpoint serialization mismatch: {tensor['name']}")
            tensors[tensor["name"]] = value
            inputs.append(file_record(path))
        for position, transaction in enumerate(result["positions"]):
            require(transaction["position"] == position, "parent position ordering mismatch")
            trace_path = verify_file_record(transaction["raw"]["trace"])
            inputs.append(file_record(trace_path))
            payload = trace_path.read_bytes()
            norm = trace_stage(payload, position, 0, 896)
            for stage, name, count in ((1, "q", 896), (2, "k", 128), (3, "v", 128)):
                prefix = f"model.layers.{layer}.self_attn.{name}_proj."
                operands = [norm, tensors[prefix + "qweight"], tensors[prefix + "qzeros"],
                            tensors[prefix + "scales"], count]
                bias = tensors[prefix + "bias"]
                single = project_fp16_stage(*operands, bias)
                dot = project_fp16_stage(*operands)
                double = (
                    dot.view("<f2").astype(np.float64)
                    + bias.view("<f2").astype(np.float64)
                ).astype("<f2").view("<u2")
                actual = trace_stage(payload, position, stage, count)
                rows.append({
                    "layer": layer, "position": position, "stage": stage,
                    "projection": name,
                    "single_round_vs_rtl": detail(actual, single),
                    "double_round_vs_rtl": detail(actual, double),
                })
    for record in sources + inputs:
        verify_file_record(record)
    result = {
        "status": "NUMERICAL_DIAGNOSIS_COMPLETE",
        "comparisons": rows,
        "authenticated_inputs": inputs,
        "q_single_round_exact": all(
            row["single_round_vs_rtl"]["different_count"] == 0
            for row in rows if row["projection"] == "q"
        ),
        "kv_double_round_exact": all(
            row["double_round_vs_rtl"]["different_count"] == 0
            for row in rows if row["projection"] in ("k", "v")
        ),
        "kv_single_round_different_elements": sum(
            row["single_round_vs_rtl"]["different_count"]
            for row in rows if row["projection"] in ("k", "v")
        ),
        "boundary": (
            "Same-operand local rounding attribution only. Does not establish how much "
            "K/V rounding contributes to layer08 stage08 or justify changing its policy. "
            "No RTL repair, new simulation, passing traversal, or layer09 advance."
        ),
    }
    write_json(output / "result.json", result)
    write_json(output / "seal.json", {
        "files": [file_record(path) for path in sorted(output.iterdir())],
    })
    for path in output.iterdir():
        path.chmod(0o444)
    output.chmod(0o555)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output.resolve())
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("comparisons", "authenticated_inputs")}))
