#!/usr/bin/env python3
"""Retained A-lineage RMSNorm and residual-component diagnosis; no RTL replay."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

from diagnose_lineage_a_position3_boundary import (
    BUILD, HISTORY, POLICY, POSITIVE, PREFIX, Inputs, accepts, decode_trace,
    require, units, write,
)
from fp16_adaptation_oracle import EPSILON_Q48, rmsnorm
from trace_lineage_a_layer07_downproj import framed_hidden, hex_rows, residual, round_q48
from trace_lineage_a_layer07_gate_up import snapshot
from trace_lineage_a_layer07_silu import source_records

ROOT = BUILD.parent
PARENT = (BUILD / "model24_selected_token_position3_continuations"
          / "lineage_a_layer07_gate_up_09d54a9cb531_attempt001")
PARENT_SHA256 = "3b63423bc8c79c547efc8565035ee84a13c5d2db928ae880562ed77150ee916f"


def floor_sqrt(value):
    require(value >= 0, "negative square root")
    low, high = 0, 1 << ((value.bit_length() + 1) // 2)
    while low < high:
        middle = (low + high + 1) // 2
        if middle * middle <= value:
            low = middle
        else:
            high = middle - 1
    return low


def normalize(activations, weights, root):
    require(len(activations) == len(weights) and root > 0, "invalid normalization geometry")
    outputs = []
    for activation, weight in zip(activations, weights, strict=True):
        product = units(activation) * units(weight)
        q24 = round(Fraction(product, root))
        bits = round_q48(q24 << 24)
        if bits & 0x7fff == 0:
            bits |= (activation ^ weight) & 0x8000
        outputs.append(bits)
    return outputs


def reconstruct(activations, weights):
    require(activations and len(activations) == len(weights), "invalid RMSNorm geometry")
    squares = [units(bits) ** 2 for bits in activations]
    numerator = sum(squares) + EPSILON_Q48 * len(activations)
    require(numerator < 1 << 92, "RMSNorm reduction overflow")
    mean = round(Fraction(numerator, len(activations)))
    root = floor_sqrt(mean)
    outputs = normalize(activations, weights, root)
    return outputs, {"sumsq_q48": sum(squares), "mean_q48": mean, "rms_q24": root}


def mathematical_outputs(activations, weights):
    """Direct mathematical RMSNorm RNE via exact squared midpoint comparisons.

    This removes intermediate Q48 mean/Q24 root/quotient rounding only as a
    diagnostic. It does not replace the declared recurrence or either gate.
    """
    require(activations and len(activations) == len(weights), "invalid RMSNorm geometry")
    n = len(activations)
    denominator = sum(units(x) ** 2 for x in activations) + EPSILON_Q48 * n
    outputs = []
    for activation, weight in zip(activations, weights, strict=True):
        target = (units(activation) * units(weight)) ** 2 * n
        require(target <= POSITIVE[-1] ** 2 * denominator, "mathematical FP16 overflow")
        low, high = 0, len(POSITIVE) - 1
        while low < high:
            middle = (low + high) // 2
            if POSITIVE[middle] ** 2 * denominator < target:
                low = middle + 1
            else:
                high = middle
        upper, lower = low, max(0, low - 1)
        midpoint_square = (POSITIVE[lower] + POSITIVE[upper]) ** 2 * denominator
        bits = (upper if 4 * target > midpoint_square else lower)
        if 4 * target == midpoint_square:
            bits = lower if lower % 2 == 0 else upper
        outputs.append(bits | ((activation ^ weight) & 0x8000))
    return outputs


def difference(actual, reference):
    return [i for i, (a, r) in enumerate(zip(actual, reference, strict=True)) if a != r]


def delta(actual, reference):
    return str(Fraction(units(actual) - units(reference), 1 << 24))


def run(out):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    inputs = Inputs()
    inputs.load(PARENT / "seal.json", root=True)
    require(inputs.records[str(PARENT / "seal.json")]["sha256"] == PARENT_SHA256,
            "reviewed 09d54a9cb531 seal changed")
    parent = inputs.load(PARENT / "frozen.json")
    prior = inputs.load(PARENT / "result.json")
    projections = inputs.load(PARENT / "projections.json")
    weighted = inputs.read(PARENT / "weighted_stage13_terms.csv")
    require(parent["token_history"] == prior["token_history"] == HISTORY
            and parent["comparison_policy"] == POLICY, "foreign history or policy")
    require(not prior["actual_local_bit_differences"]
            and prior["stage16_actual_reconstructed"] == 4864,
            "downstream predecessor was not locally cleared")
    inputs.load(PREFIX / "seal.json")
    prefix = inputs.load(PREFIX / "frozen.json")
    original = inputs.load(prefix["original_frozen"]["path"])
    policy = inputs.load(original["independent_policy"]["path"])
    require(original["comparison_policy"] == policy["comparison"] == POLICY,
            "independent comparison changed")

    prepared, results, actual, reference = {}, {}, {}, {}
    for layer in (6, 7, 8):
        inputs.load(PREFIX / f"layer{layer:02d}/seal.json")
        p = prepared[layer] = inputs.load(PREFIX / f"layer{layer:02d}/prepared.json")
        results[layer] = inputs.load(PREFIX / f"layer{layer:02d}/result.json")
        require(p["vectors"]["layer_index"] == layer and p["vectors"]["position"] == 3,
                "foreign layer or position")
        kv = p["kv_parent"]
        require(p["binary"] == kv["live_binary"] and kv["layer_index"] == layer
                and kv["valid_positions"] == [0, 1, 2]
                and "/model24_persistent_kv_selected_token_corrected_q_attempt005/"
                in kv["state"]["path"], "incompatible A KV or binary ABI")
        for record in [p["binary"], kv["state"], p["input_hidden"], *kv["actual_kv_traces"]]:
            inputs.read(record["path"])
        history = inputs.load(p["independent_parent"]["path"])
        require(history["history"] == HISTORY[:3] and history["layer"] == layer,
                "foreign independent history")
        if layer == 6:
            reference[layer] = {18: hex_rows(inputs.read(p["independent_stages"]["18"]["path"]))}
        else:
            trace = Path(results[layer]["output_hidden"]["path"]).with_name("trace.hex")
            actual[layer] = decode_trace(inputs.read(trace), 3)
            reference[layer] = {int(stage): hex_rows(inputs.read(record["path"]))
                                for stage, record in p["independent_stages"].items()}
    for layer in (7, 8):
        require(all(prepared[layer]["input_hidden"][k] == results[layer - 1]["output_hidden"][k]
                    for k in ("path", "bytes", "sha256")), "broken actual hidden edge")
    selected = inputs.load(BUILD / "position2_tail_runtime_attempt001/selected_token.json")
    require(selected["actual_token_id"] == selected["expected_token_id"] == 358
            and selected["matches"] is True, "A tail token changed")
    a, r = actual[7], reference[7]
    incoming_a = framed_hidden(inputs.read(prepared[7]["input_hidden"]["path"]))
    incoming_r = reference[6][18]
    require(incoming_a == framed_hidden(inputs.read(prepared[7]["vectors"]["input"]["path"])),
            "stage12 actual residual input not the consumed vector")
    require(a[18] == framed_hidden(inputs.read(results[7]["output_hidden"]["path"]))
            == framed_hidden(inputs.read(prepared[8]["vectors"]["input"]["path"])),
            "layer08 did not consume actual layer07 stage18")
    require(all(len(v) == 896 for v in [incoming_a, incoming_r, a[11], r[11],
                                       a[12], r[12], a[13], r[13]]),
            "public hidden geometry mismatch")
    norm_records = prior["next_upstream_boundary"]["official_norm_tensor_bindings"]
    require(len(norm_records) == 1
            and norm_records[0] in prepared[7]["vectors"]["tensors"], "norm tensor binding")
    norm = norm_records[0]
    meta = norm["checkpoint_tensor"]
    require(meta["name"] == "model.layers.7.post_attention_layernorm.weight"
            and meta["dtype"] == "F16" and meta["shape"] == [896], "foreign norm tensor")
    weights = hex_rows(inputs.read(norm["serialized"]["path"]))
    raw = b"".join(w.to_bytes(2, "little") for w in weights)
    require(len(raw) == meta["bytes"] == 1792
            and hashlib.sha256(raw).hexdigest() == meta["sha256"],
            "serialized norm differs from official tensor")

    sources = []
    for name in ("ace3_fp16_fixed.sv", "ace3_fp16_rmsnorm_core.sv",
                 "ace3_decoder_layer0_token_engine.sv", "ace3_fp16_residual_add_core.sv"):
        path = ROOT / "ace3/rtl" / name
        record = snapshot(out, path)
        old = [s for s in source_records(original) if s["path"] == str(path)]
        require(old and all(s["sha256"] == record["sha256"] for s in old),
                f"historical RTL source changed: {name}")
        sources.append(record)
    for name in (Path(__file__).name, "diagnose_lineage_a_position3_boundary.py",
                 "trace_lineage_a_layer07_downproj.py", "trace_lineage_a_layer07_gate_up.py",
                 "trace_lineage_a_layer07_silu.py", "fp16_adaptation_oracle.py",
                 "projection_oracle.py", "awq_bit_oracle.py"):
        record = snapshot(out, Path(__file__).with_name(name))
        old = [s for s in parent["sources"] if s["path"] == record["path"]]
        require(all(s["sha256"] == record["sha256"] for s in old),
                f"reviewed helper changed: {name}")
        sources.append(record)
    sources.append(snapshot(out, Path(__file__).parent / "tests/test_trace_lineage_a_layer07_rmsnorm.py"))
    top = "ace3_fp16_rmsnorm_core"
    command = ["iverilog", "-g2012", "-s", top, f"-P{top}.HIDDEN_SIZE=896",
               f"-P{top}.EPSILON_Q48={EPSILON_Q48}", "-o", str(out / "public_contract.vvp"),
               str(out / "ace3_fp16_fixed.sv"), str(out / "ace3_fp16_rmsnorm_core.sv")]
    tests = ["env", "PYTHONPATH=ace3/model", "PYTHONDONTWRITEBYTECODE=1",
             sys.executable, "-m", "unittest", "discover", "-s", "ace3/model/tests",
             "-p", "test_trace_lineage_a_layer07_rmsnorm.py"]
    version = subprocess.run(["iverilog", "-V"], capture_output=True, text=True, check=True)
    prioritized = prior["next_upstream_boundary"]["prioritized_output_coordinates"]
    write(out / "frozen.json", {
        "attempt_id": out.name, "scope": "Retained A-only numerical diagnosis, not RTL execution",
        "parent_seal_sha256": PARENT_SHA256, "authenticated_inputs": list(inputs.records.values()),
        "sources": sources, "token_history": HISTORY, "profile": original["profile"],
        "comparison_policy": POLICY, "independent_reference_policy": policy,
        "norm_tensor": norm, "public_top": top,
        "public_declaration": (out / "ace3_fp16_rmsnorm_core.sv").read_text().split(");", 1)[0] + ");",
        "parameters": {"HIDDEN_SIZE": 896, "EPSILON_Q48": EPSILON_Q48},
        "python": sys.version, "iverilog_version": version.stdout + version.stderr,
        "compile_command": command, "regression_command": tests,
        "measurement_command": [sys.executable, str(Path(__file__).resolve()), "--out", str(out)],
        "cases": {"coordinates": list(range(896)), "prioritized": prioritized,
                  "operand_sets": ["actual", "independent_reference"],
                  "factorial": "Numerator/denominator and residual-component software diagnostics only",
                  "ranking": "descending absolute exact reduction delta; coordinate tie-break"},
        "arithmetic": "Exact Q24 decode; RNE mean Q48; floor sqrt Q24; RNE quotient Q24; FP16 RNE.",
        "diagnostic_only": "Direct mathematical RNE retains epsilon_q48; changes no policy or RTL input.",
    })
    for name, cmd in (("compile", command), ("regression", tests)):
        with (out / f"{name}.log").open("x") as stream:
            subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

    predictions, reductions, ideal, oracle_differences = {}, {}, {}, {}
    for label, values in (("actual", a), ("reference", r)):
        prediction, reduction = reconstruct(values[12], weights)
        independent, mean, root = rmsnorm(values[12], weights)
        require(all(not invalid and not saturated for _, invalid, saturated in independent),
                "oracle invalid or saturated")
        oracle_bits = [bits for bits, _, _ in independent]
        oracle_differences[label] = difference(prediction, oracle_bits)
        require(not oracle_differences[label]
                and (mean, root) == (reduction["mean_q48"], reduction["rms_q24"]),
                "independent recurrence reconstruction disagrees with integer oracle")
        predictions[label], reductions[label] = prediction, reduction
        ideal[label] = mathematical_outputs(values[12], weights)
    cross = normalize(r[12], weights, reductions["actual"]["rms_q24"])
    local = difference(a[13], predictions["actual"])
    components, rows = [], []
    for i in range(896):
        aa, rr = a[13][i], r[13][i]
        pa, pr, middle = predictions["actual"][i], predictions["reference"][i], cross[i]
        parts = [units(aa) - units(pa), units(pa) - units(middle),
                 units(middle) - units(pr), units(pr) - units(rr)]
        require(sum(parts) == units(aa) - units(rr), "normalization attribution does not close")
        components.append(parts)
        rows.append({
            "index": i, "prioritized": i in prioritized,
            "actual_stage12": f"{a[12][i]:04x}", "reference_stage12": f"{r[12][i]:04x}",
            "weight": f"{weights[i]:04x}", "actual_stage13": f"{aa:04x}",
            "reference_stage13": f"{rr:04x}", "own_input_reconstruction": f"{pa:04x}",
            "reference_input_reconstruction": f"{pr:04x}",
            "reference_numerator_actual_root": f"{middle:04x}",
            "actual_mathematical_diagnostic": f"{ideal['actual'][i]:04x}",
            "reference_mathematical_diagnostic": f"{ideal['reference'][i]:04x}",
            "local_delta_q24": parts[0], "numerator_delta_q24": parts[1],
            "root_delta_q24": parts[2], "reference_recurrence_delta_q24": parts[3],
            "reduction_delta_q48": units(a[12][i]) ** 2 - units(r[12][i]) ** 2,
            "within_unchanged_trajectory_gate": accepts(aa, rr),
        })
    write(out / "rmsnorm_coordinates.json", rows)

    residual_rows = []
    for i in range(896):
        predicted_a = residual(incoming_a[i], a[11][i])
        predicted_r = residual(incoming_r[i], r[11][i])
        residual_rows.append({
            "index": i, "incoming_actual": f"{incoming_a[i]:04x}",
            "incoming_reference": f"{incoming_r[i]:04x}",
            "stage11_actual": f"{a[11][i]:04x}", "stage11_reference": f"{r[11][i]:04x}",
            "stage12_actual": f"{a[12][i]:04x}", "stage12_reference": f"{r[12][i]:04x}",
            "predicted_actual": f"{predicted_a:04x}", "predicted_reference": f"{predicted_r:04x}",
            "incoming_delta": delta(incoming_a[i], incoming_r[i]),
            "stage11_delta": delta(a[11][i], r[11][i]),
            "residual_rounding_delta": str(Fraction(
                units(a[12][i]) - units(r[12][i]) - units(incoming_a[i])
                + units(incoming_r[i]) - units(a[11][i]) + units(r[11][i]), 1 << 24)),
            "stage12_delta": delta(a[12][i], r[12][i]),
        })
    write(out / "residual_components.json", residual_rows)
    residual_local = [row["index"] for row in residual_rows
                      if row["stage12_actual"] != row["predicted_actual"]]
    residual_reference = [row["index"] for row in residual_rows
                          if row["stage12_reference"] != row["predicted_reference"]]
    residual_factorial = []
    for use_reference_hidden, use_reference_o in ((False, False), (False, True),
                                                 (True, False), (True, True)):
        hidden = incoming_r if use_reference_hidden else incoming_a
        o = r[11] if use_reference_o else a[11]
        res = [residual(x, y) for x, y in zip(hidden, o, strict=True)]
        norm_bits, reduction = reconstruct(res, weights)
        residual_factorial.append({
            "reference_incoming": use_reference_hidden, "reference_stage11": use_reference_o,
            "stage12_bit_difference_count": len(difference(res, r[12])),
            "stage13_bit_difference_count": len(difference(norm_bits, r[13])),
            "stage13_material_failure_count": sum(not accepts(x, y) for x, y in zip(norm_bits, r[13], strict=True)),
            "reduction": reduction, "channel5_stage12": f"{res[5]:04x}",
            "scope": "Conditional residual/normalization sensitivity only; no MLP, Q or score execution.",
        })

    weighted_parts = {}
    for term in csv.DictReader(io.StringIO(weighted.decode("ascii"))):
        stage, channel, i = int(term["stage"]), int(term["channel"]), int(term["index"])
        require(term["actual_stage13_bits"] == f"{a[13][i]:04x}"
                and term["reference_stage13_bits"] == f"{r[13][i]:04x}",
                "reviewed weighted term uses different normalization inputs")
        coefficient = (int(term["weight"]) - int(term["zero"])) * units(int(term["scale_bits"], 16))
        parts = [part * coefficient for part in components[i]]
        require(sum(parts) == int(term["delta_q48"]), "weighted propagation does not close")
        totals = weighted_parts.setdefault((stage, channel), [0, 0, 0, 0])
        for j, value in enumerate(parts):
            totals[j] += value
    projection_rows = []
    for row in projections:
        key = row["stage"], row["channel"]
        require(row["actual"]["bits"] == f"{a[key[0]][key[1]]:04x}"
                and row["reference"]["bits"] == f"{r[key[0]][key[1]]:04x}",
                "reviewed projection output changed")
        if key in weighted_parts:
            parts = weighted_parts[key]
            require(sum(parts) == row["actual_q48"] - row["reference_q48"],
                    "projection decomposition does not close")
            projection_rows.append({
                "stage": key[0], "channel": key[1], "component_deltas_q48": parts,
                "actual": row["actual"], "reference": row["reference"],
                "rounding_and_local_delta": row["rounding_and_local_delta"],
            })
    require(len(projection_rows) == 32, "missing sealed gate/up propagation channels")
    write(out / "weighted_propagation.json", {
        "component_order": ["local_norm", "numerator", "root", "reference_recurrence"],
        "gate_up": projection_rows,
        "reviewed_stage16": {"actual_reconstructed": prior["stage16_actual_reconstructed"],
                            "scope": "Authenticated predecessor measurement, not rerun"},
        "reviewed_stage17_channel5": prior["stage17_channel5"],
        "reviewed_stage18_channel5": prior["stage18_channel5"],
        "reviewed_layer08_incoming": prior["layer08_incoming_hidden"],
        "parent_result": inputs.records[str(PARENT / "result.json")],
    })
    mean_rank = sorted(rows, key=lambda row: (-abs(row["reduction_delta_q48"]), row["index"]))[:16]
    next_coordinates = sorted(set(prioritized) | {5} | {row["index"] for row in mean_rank})
    failed_scores = [i for i, (x, y) in enumerate(zip(actual[8][8], reference[8][8], strict=True))
                     if not accepts(x, y)]
    require(failed_scores == [13, 15], "original layer08 failures changed")
    result = {
        "status": "RETAINED_RMSNORM_RECONSTRUCTED_NOT_REPAIRED" if not local and not residual_local
                  else "LOCAL_RECONSTRUCTION_DISCREPANCY_NOT_REPAIRED",
        "token_history": HISTORY, "reconstructed_actual_outputs": 896,
        "independent_integer_oracle_outputs": 1792, "oracle_disagreements": oracle_differences,
        "actual_rmsnorm_local_bit_differences": local,
        "reference_recurrence_bit_differences": difference(predictions["reference"], r[13]),
        "mathematical_diagnostic_differences": {
            label: difference(predictions[label], ideal[label]) for label in predictions},
        "reductions": reductions, "largest_reduction_contributions": mean_rank,
        "trajectory_bit_difference_counts": {
            "incoming": len(difference(incoming_a, incoming_r)),
            **{f"stage{s}": len(difference(a[s], r[s])) for s in (11, 12, 13, 18)}},
        "stage13_material_failure_indices": [i for i in range(896) if not accepts(a[13][i], r[13][i])],
        "residual_local_bit_differences": residual_local,
        "residual_reference_bit_differences": residual_reference,
        "channel5_residual_components": residual_rows[5],
        "residual_factorial_software_only": residual_factorial,
        "layer08_original_failures": prior["layer08_incoming_hidden"]["original_stage08_failures"],
        "layer08_actual_input_bit_exact_stage18": True,
        "next_upstream_boundary": {
            "layer": 7, "position": 3, "producer": "stage10 attention output -> stage11 O projection",
            "coordinates": next_coordinates, "required_input_coordinates": list(range(896)),
            "separate_incoming_branch": prepared[7]["input_hidden"],
            "independent_incoming_branch": prepared[6]["independent_stages"]["18"],
            "component_records": str(out / "residual_components.json"),
            "question": "Separate same-input native O-projection error from inherited stage10 drift; "
                        "retain layer06 hidden contribution and stage12 residual rounding separately.",
            "scope": "A next discriminating boundary, not a uniquely proved historical producer.",
        } if not local and not residual_local else None,
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "root_cause_hypothesis": "Inherited stage12 components, not an additional RMSNorm implementation "
                                 "error if own-input reconstruction is exact; no unique producer proven.",
        "regression": "All 896 own-input outputs and residual components; exact weighted decomposition; "
                      "retain original layer08 failures13/15 and both unchanged numerical gates.",
        "repair": None, "RTL_simulations": 0, "contract_compiles": 1,
        "binary64_evaluated": False, "lineage_B_imports": 0, "third_token_selected": False,
        "independent_review": "pending normal Host Reviewer",
        "elapsed_seconds": time.monotonic() - started,
    }
    write(out / "result.json", result)
    for path in out.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps({k: result[k] for k in (
        "status", "actual_rmsnorm_local_bit_differences", "reference_recurrence_bit_differences",
        "reductions", "trajectory_bit_difference_counts", "residual_local_bit_differences",
        "residual_reference_bit_differences", "channel5_residual_components",
        "layer08_original_failures", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out.resolve())
