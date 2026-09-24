"""Immutable lineage-B earliest-output gate and retained residual-cone diagnosis."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
ADMISSION = ROOT / "build/lineage_b_binary64_admission_4ebb09f7e6e4_attempt001"
ARCHIVE = Path("/home/argustest/argustest2/ace3-continuous-generation-20260906")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from ace3.model.candidates import binary64_fp16_excess_v1 as profile


def load(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def half(bits):
    return struct.unpack(">e", int(bits).to_bytes(2, "big"))[0]


def rne(value):
    return int.from_bytes(struct.pack(">e", value), "big")


def exact(value):
    return Fraction.from_float(float(value))


def independent_scalar(bits, reference):
    actual_error = abs(exact(half(bits)) - exact(reference))
    nearest = rne(reference)
    q = abs(exact(half(nearest)) - exact(reference))
    return {
        "nearest_fp16_bits": f"{nearest:04x}",
        "actual_error": str(actual_error),
        "q": str(q),
        "excess_error": str(actual_error - q),
        "accepted": actual_error - q <= Fraction(1, 8),
    }


def residual_diagnosis(first, outputs, frozen, backend):
    import numpy as np
    import torch

    require(first["position"] == 0, "diagnostic supports the position-zero residual cone only")
    index = first["index"]
    rows = []
    total_rounding = Fraction()
    for layer in range(first["layer"] + 1):
        item = outputs[layer, 0]
        receipt = load(item["receipt"]["path"])
        trace_path = Path(receipt["transaction"]["raw"]["trace"]["path"])
        stages = {}
        for text in trace_path.read_text().splitlines():
            word = int(text, 16)
            position, stage, channel = (word >> 40) & 0xFFFF, (word >> 32) & 0xFF, (word >> 16) & 0xFFFF
            require(position == 0, "foreign trace position")
            if stage in (11, 12, 17, 18):
                require((stage, channel) not in stages, "duplicate residual trace coordinate")
                stages[stage, channel] = word & 0xFFFF
        require(set(stages) == {(s, i) for s in (11, 12, 17, 18) for i in range(896)},
                "incomplete residual trace")
        parent = (frozen["embeddings"][0]["input"] if layer == 0
                  else outputs[layer - 1, 0]["actual"])
        actual_input = backend.engine.load_hidden_bits(Path(parent["path"]))
        reference_input = (actual_input.view("<f2").astype(np.float64) if layer == 0
                           else np.load(outputs[layer - 1, 0]["reference"]["path"], allow_pickle=False))
        reference_final = np.load(item["reference"]["path"], allow_pickle=False)
        state = backend.reference_layer(Path(frozen["checkpoint"]["path"]), layer, None)
        captured = {}
        original = backend.dialogue._reference_projection

        def capture(activation, projection):
            result = original(activation, projection)
            name = next(name for name, value in state.projections.items() if value is projection)
            captured[name] = result.detach().numpy().reshape(-1).copy()
            return result

        # Observe the independent recurrence without substituting any operands.
        backend.dialogue._reference_projection = capture
        try:
            with torch.no_grad():
                reproduced = backend.dialogue._reference_layer_step(
                    state, torch.from_numpy(reference_input.reshape(1, 896)), 0
                ).numpy().reshape(-1)
        finally:
            backend.dialogue._reference_projection = original
        require(np.array_equal(reproduced, reference_final),
                f"independent_reference_reproduction_failed: layer{layer}")
        actual_final = backend.engine.load_hidden_bits(Path(item["actual"]["path"]))
        for channel in range(896):
            require(rne(half(actual_input[channel]) + half(stages[11, channel])) == stages[12, channel],
                    f"residual1_RNE_defect: {layer}/{channel}")
            require(rne(half(stages[12, channel]) + half(stages[17, channel])) == stages[18, channel],
                    f"residual2_RNE_defect: {layer}/{channel}")
            require(stages[18, channel] == int(actual_final[channel]), "final/trace mismatch")
        a = half(actual_input[index])
        o, res1, down, final = [half(stages[s, index]) for s in (11, 12, 17, 18)]
        ref_in, ref_o, ref_down, ref_final = [
            float(v[index]) for v in (reference_input, captured["o"], captured["down"], reference_final)
        ]
        terms = {
            "inherited_hidden_error": exact(a) - exact(ref_in),
            "attention_projection_trajectory_error": exact(o) - exact(ref_o),
            "residual1_rounding": exact(res1) - exact(a) - exact(o),
            "down_projection_trajectory_error": exact(down) - exact(ref_down),
            "residual2_rounding": exact(final) - exact(res1) - exact(down),
            "binary64_addition_rounding_correction": exact(ref_in) + exact(ref_o) + exact(ref_down) - exact(ref_final),
        }
        require(sum(terms.values()) == exact(final) - exact(ref_final), "error decomposition does not close")
        total_rounding += terms["residual1_rounding"] + terms["residual2_rounding"]
        counterfactual = rne(a + o + down)
        rows.append({
            "layer": layer, "position": 0, "index": index,
            "actual": {"input": a, "attention_projection": o, "residual1": res1,
                       "down_projection": down, "final": final},
            "reference": {"input": ref_in, "attention_projection": ref_o,
                          "down_projection": ref_down, "final": ref_final},
            "exact_error_terms": {k: str(v) for k, v in terms.items()},
            "decimal_error_terms_display_only": {k: float(v) for k, v in terms.items()},
            "cumulative_residual_rounding": str(total_rounding),
            "own_operand_residual_RNE_agreement_count": 1792,
            "independent_reference_bit_exact_elements": 896,
            "same_operand_single_round_counterfactual": {
                "actual_fp16_bits": f"{counterfactual:04x}",
                **independent_scalar(counterfactual, ref_final),
                "scope": "Conditional software diagnostic only; down operands held fixed. "
                         "Not a changed-norm2 rollout, RTL repair or trajectory admission.",
            },
        })
    return {
        "scope": "Position-zero retained residual cone through the earliest failing output",
        "rows": rows,
        "local_residual_arithmetic": "Both additions agree with independent IEEE binary16 RNE",
        "root_cause_boundary": "Exact decomposition separates inherited hidden error, projection "
                               "trajectory differences and the two residual roundings. Projection "
                               "producer differences are not uniquely localized by this diagnostic.",
        "rtl_repair_justified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "build"), "evidence must remain under build")
    output.mkdir()
    started = time.monotonic()
    old_result = load(ADMISSION / "result.json")
    old = load(ADMISSION / "freeze.json")
    require(record(ADMISSION / "freeze.json") == old_result["freeze"], "prior admission freeze drift")
    require(old_result["status"] == "RETAINED_OUTPUT_BINARY64_V1_FAIL", "unexpected prior result")
    bindings = {item["path"]: item for item in old["sources_and_inputs"]}
    for item in bindings.values():
        current = record(item["path"])
        require(all(current[k] == item[k] for k in ("path", "bytes", "sha256")),
                f"bound input/source drift: {item['path']}")
    sys.path.insert(0, str(ARCHIVE / "ace3/model"))
    import continuous_kv_rtl_backend as backend
    import numpy as np
    import torch

    imported = []
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(ARCHIVE):
            path = str(Path(filename).resolve())
            require(path in bindings, f"unbound independent reference module: {path}")
            imported.append(bindings[path])
    frozen = load(old["lineage_B_freeze"]["path"])
    require(frozen["token_history"] == [9707, 1879, 0], "unexpected lineage-B roots")
    outputs = {(item["layer"], item["position"]): item for item in old["outputs"]}
    require(list(outputs) == [(l, p) for l in range(9) for p in range(3)], "ordered scope mismatch")
    write(output / "freeze.json", {
        "attempt": output.name, "mission": "f6846b6185e4",
        "operation": "Retained-output earliest-failure gate and residual-cone diagnosis; no RTL replay",
        "parent_admission_freeze": old_result["freeze"],
        "parent_admission_result": record(ADMISSION / "result.json"),
        "source": record(__file__), "profile_sources": [record(ROOT / p) for p in profile.SOURCE_PATHS],
        "profile_id": profile.PROFILE_ID, "reference_policy": profile.REFERENCE_POLICY,
        "score_policy": "finite valid operands; abs(reference)<=65504; exact excess_error<=1/8",
        "order": "layer, position, scalar index ascending; STOP at the first rejected scalar",
        "scope": "layers0-8/positions0-2; independent residual diagnostics only through first failure",
        "inputs": old["outputs"], "lineage_B_freeze": old["lineage_B_freeze"],
        "public_interfaces": frozen["public_interfaces"], "parameters": frozen["parameters"],
        "retained_compile_commands": frozen["compile_commands"], "retained_tools": frozen["tools"],
        "reference_modules": imported, "python": sys.version, "numpy": np.__version__,
        "torch": torch.__version__, "command": sys.argv,
        "new_RTL_simulations": 0, "lineage_A_actual_or_state_imported": False,
        "position3_executed": False,
    })
    first = None
    count = 0
    complete = 0
    with (output / "scalar_results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "layer", "position", "index", "actual_fp16_bits", "reference_binary64_hex",
            "nearest_fp16_bits", "actual_error", "q", "excess_error", "accepted",
        ])
        writer.writeheader()
        for (layer, position), item in outputs.items():
            actual = backend.engine.load_hidden_bits(Path(item["actual"]["path"]))
            reference = np.load(item["reference"]["path"], allow_pickle=False)
            require(actual.shape == reference.shape == (896,) and reference.dtype == np.float64,
                    "invalid output geometry or reference dtype")
            for index, (bits, ref) in enumerate(zip(actual, reference, strict=True)):
                measured = profile.evaluate_layer_final_output(
                    actual_fp16_bits=int(bits), reference_binary64_hex=float(ref).hex())
                oracle = independent_scalar(int(bits), float(ref))
                require(all(measured[k] == v for k, v in oracle.items()),
                        f"independent scalar oracle disagreement: {layer}/{position}/{index}")
                row = {"layer": layer, "position": position, "index": index,
                       "actual_fp16_bits": f"{int(bits):04x}",
                       "reference_binary64_hex": float(ref).hex(), **oracle}
                writer.writerow(row)
                count += 1
                if not measured["accepted"]:
                    first = row
                    break
            if first:
                break
            complete += 1
    diagnosis = residual_diagnosis(first, outputs, frozen, backend) if first else None
    if diagnosis:
        write(output / "residual_diagnosis.json", diagnosis)
        require(count == (first["layer"] * 3 + first["position"]) * 896 + first["index"] + 1,
                "earliest-failure stop regression")
    result = {
        "status": "RETAINED_OUTPUT_BINARY64_V1_FAIL" if first else "RETAINED_OUTPUT_BINARY64_V1_PASS",
        "admitted": first is None, "freeze": record(output / "freeze.json"),
        "first_failure": first, "complete_passing_outputs": complete,
        "scalar_count": count, "independent_scalar_oracle_agreement_count": count,
        "scalar_results": record(output / "scalar_results.csv"),
        "diagnosis": record(output / "residual_diagnosis.json") if diagnosis else None,
        "failure_taxonomy": "retained_layer_final_excess_error" if first else None,
        "root_cause_hypothesis": "Accumulated hidden/projection trajectory error and FP16 residual "
                                 "rounding, not an incorrect own-operand residual addition.",
        "regression": "Retained exact residual RNE, reproduced independent binary64 arrays, "
                      "exact error decomposition, independent scalar oracle and first-failure stop.",
        "harness_repair": "Stop at the earliest rejected scalar instead of evaluating the remaining scope.",
        "rtl_arithmetic_changed": False, "new_RTL_simulations": 0,
        "legacy_failures_preserved": True, "normal_host_review": "PENDING",
        "lineage_A_actual_or_state_imported": False, "position3_executed": False,
        "tail_executed": False, "full_driver_admitted": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    write(output / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 1 if first else 0


if __name__ == "__main__":
    raise SystemExit(main())
