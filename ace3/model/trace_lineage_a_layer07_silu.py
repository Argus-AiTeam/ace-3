#!/usr/bin/env python3
"""Source-bound retained SiLU diagnosis; never a trajectory admission."""

from __future__ import annotations

import argparse
import bisect
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from diagnose_lineage_a_position3_boundary import (
    BUILD, HISTORY, POLICY, POSITIVE, PREFIX, Inputs, accepts, decode_trace,
    require, units, write,
)
from fp16_adaptation_oracle import silu_gate_exp
from trace_lineage_a_layer07_downproj import (
    framed_hidden, hex_rows, projection_terms, residual, round_q48, scalar,
)

PARENT = (BUILD / "model24_selected_token_position3_continuations"
          / "lineage_a_layer07_downproj_afcbb5dd88d5_attempt002")
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
              "handoffs/afcbb5dd88d5/round-0001.json")
ROOT = BUILD.parent
ONE = 1 << 24


def rne_ratio(numerator, denominator):
    require(denominator > 0, "nonpositive rounding denominator")
    quotient, remainder = divmod(abs(numerator), denominator)
    rounded = quotient + int(2 * remainder > denominator
                             or (2 * remainder == denominator and quotient & 1))
    return -rounded if numerator < 0 else rounded


def accurate_silu(gate_bits, up_bits):
    """Independent integer reconstruction of the source-bound Q24 data path."""
    gate, up = units(gate_bits), units(up_bits)
    exponent, remainder = divmod(abs(gate), 11629080)
    polynomial = -3329
    for coefficient in (23302, -139810, 699051, -2796203,
                        8388608, -16777216, ONE):
        polynomial = coefficient + rne_ratio(remainder * polynomial, ONE)
    exponential = 0 if exponent >= 63 else rne_ratio(polynomial, 1 << exponent)
    negative_sigmoid = rne_ratio(exponential * ONE, ONE + exponential)
    sigmoid = negative_sigmoid if gate < 0 else ONE - negative_sigmoid
    product = gate * up * sigmoid
    q24 = rne_ratio(product, 1 << 48)
    bits = round_q48(q24 << 24)
    if bits & 0x7fff == 0:
        bits = (gate_bits ^ up_bits) & 0x8000
    return {"bits": bits, "sigmoid_q24": sigmoid, "product_q72": product,
            "rounded_q24": q24}


def decimal_fraction(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def mathematical_silu(gate_bits, up_bits):
    gate, up = Decimal(units(gate_bits)) / ONE, Decimal(units(up_bits)) / ONE
    exponential = (-abs(gate)).exp()
    sigmoid = (exponential if gate < 0 else Decimal(1)) / (1 + exponential)
    return gate * up * sigmoid


def nearest_decimal(value, zero_sign=0):
    require(value.is_finite(), "nonfinite mathematical output")
    magnitude = abs(value) * ONE
    require(magnitude <= POSITIVE[-1], "mathematical output outside finite FP16")
    right = bisect.bisect_left(POSITIVE, magnitude)
    bits = min({right, max(0, right - 1)},
               key=lambda b: (abs(Decimal(POSITIVE[b]) - magnitude), b & 1))
    return bits | (0x8000 if value < 0 or (value == 0 and zero_sign) else 0)


def decompose(gate, up, actual, reconstructed, ideal):
    product = Decimal(reconstructed["product_q72"]) / (1 << 72)
    q24 = Decimal(reconstructed["rounded_q24"]) / ONE
    output = Decimal(units(reconstructed["bits"])) / ONE
    retained = Decimal(units(actual)) / ONE
    components = {
        "sigmoid_approximation": product - ideal,
        "product_q24_rounding": q24 - product,
        "output_fp16_rounding": output - q24,
        "retained_minus_reconstruction": retained - output,
    }
    require(abs(sum(components.values()) - (retained - ideal)) < Decimal("1e-65"),
            "local error decomposition does not close")
    return {key: str(value) for key, value in components.items()}


def source_records(value):
    if isinstance(value, dict):
        if {"path", "sha256"} <= value.keys():
            yield value
        for child in value.values():
            yield from source_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from source_records(child)


def classification(differences):
    # The independent trajectory specifies mathematical SiLU, not the RTL approximation.
    defect = bool(differences["actual"])
    return {
        "status": ("LOCAL_SILU_DISCREPANCY_NOT_REPAIRED" if defect
                   else "RETAINED_SILU_RECONSTRUCTED_NOT_REPAIRED"),
        "causal_local_implementation_defect_proven": defect,
        "reference_comparison_scope": "Reference operands evaluated with the source-faithful "
                                      "Q24 approximation; differences from mathematical "
                                      "trajectory outputs are not RTL implementation defects.",
    }


def upstream_boundary(coordinates, tensor_bindings):
    return {
        "layer": 7, "position": 3, "input": "authenticated actual stage13[0:896]",
        "producers": ["stage14 mlp.gate_proj", "stage15 mlp.up_proj"],
        "output_coordinates": coordinates, "groups": 7, "group_size": 128,
        "output_size": 4864, "official_tensor_bindings": tensor_bindings,
        "experiment": "Reconstruct native-GEMM AWQ exact products and group/cross-group "
                      "sums from actual stage13 at these output channels, comparing "
                      "own-input local rounding against propagated stage13 drift. "
                      "Stage12 channel5 remains a separate inherited residual branch.",
    }


def classify_retained(out, measurement):
    inputs = Inputs()
    inputs.load(measurement / "seal.json", root=True)
    frozen = inputs.load(measurement / "frozen.json")
    result = inputs.load(measurement / "result.json")
    coordinates = inputs.load(measurement / "coordinates.json")
    original = inputs.load(BUILD / "token358_position3_full_continuation_attempt001/frozen.json")
    policy = inputs.load(original["independent_policy"]["path"])
    require(policy["silu"] == "binary64 silu(gate)*up then one FP16 boundary, as accepted helper",
            "independent SiLU reference policy differs")
    require(frozen["token_history"] == result["token_history"] == HISTORY
            and frozen["comparison_policy"] == POLICY, "foreign retained measurement")
    sources = []
    for path in (Path(__file__), Path(__file__).with_name("tests")
                 / "test_trace_lineage_a_layer07_silu.py"):
        data = path.read_bytes()
        with (out / path.name).open("xb") as stream:
            stream.write(data)
        sources.append({"path": str(path), "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    write(out / "frozen.json", {
        "attempt_id": out.name, "kind": "retained_stage16_classification_correction_v1",
        "measurement_attempt": str(measurement), "sources": sources,
        "authenticated_inputs": list(inputs.records.values()),
        "comparison_policy": POLICY, "token_history": HISTORY,
        "command": (out / "run.command.sh").read_text(),
        "correction": "Only actual-output/source-operator disagreement can indicate a local "
                      "RTL implementation discrepancy. Mathematical-reference/source-operator "
                      "differences must remain separately reported.",
        "arithmetic_reexecutions": 0, "new_RTL_invocations": 0,
    })
    result["attempt_id"] = out.name
    result["measurement_attempt"] = str(measurement)
    result["classification_only"] = True
    result["arithmetic_reexecutions"] = 0
    result["new_contract_compiles"] = 0
    result.update(classification(result["same_input_bit_difference_indices"]))
    result["next_upstream_boundary"] = (
        None if result["causal_local_implementation_defect_proven"] else
        upstream_boundary(frozen["cases"]["weighted_coordinates"],
                          frozen["checkpoint_tensor_bindings"])
    )
    result["sealed_coordinate_reference_source_differences"] = [
        row["index"] for row in coordinates if row["reference_reconstruction"]["bits"]
        != int(row["weighted_term"]["reference_stage16_bits"], 16)
    ]
    result["sealed_coordinate_mathematical_reference_differences"] = [
        row["index"] for row in coordinates if row["mathematical_rne_fp16"]["rr"]["bits"]
        != row["weighted_term"]["reference_stage16_bits"]
    ]
    result["independent_reference_policy"] = original["independent_policy"]
    result["preserved_classification_failure"] = {
        "path": str(measurement / "result.json"),
        "taxonomy": "evaluator_scope_classification",
        "root_cause_hypothesis": "The original any(differences.values()) conflated "
                                 "reference mathematical-vs-Q24 differences with actual "
                                 "RTL own-input disagreement.",
        "regression": "Reference-only disagreement must not suppress an upstream boundary; "
                      "actual disagreement must still suppress it.",
    }
    write(out / "result.json", result)
    artifacts = []
    for path in sorted(out.iterdir()):
        if path.name in ("execution.log", "exit_status.txt"):
            continue
        data = path.read_bytes()
        artifacts.append({"path": str(path), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
        path.chmod(0o444)
    write(out / "seal.json", {"artifacts": artifacts})
    print(json.dumps({key: result[key] for key in (
        "status", "causal_local_implementation_defect_proven", "classification_only",
        "sealed_coordinate_reference_source_differences",
        "sealed_coordinate_mathematical_reference_differences",
        "next_upstream_boundary") if key != "next_upstream_boundary"}, indent=2))
    print("Next boundary: actual stage13 -> stage14/15 at the sealed 16 output coordinates.")


def run(out):
    started = time.monotonic()
    inputs = Inputs()
    inputs.load(PARENT / "seal.json", root=True)
    frozen = inputs.load(PARENT / "frozen.json")
    prior = inputs.load(PARENT / "result.json")
    review = inputs.load(REVIEW, root=True)
    require(review["producer_role"] == "reviewer"
            and review["mission_id"] == "afcbb5dd88d5"
            and review["review"]["status"] == "done", "wrong parent review")
    require(frozen["token_history"] == prior["token_history"] == HISTORY
            and frozen["comparison_policy"] == POLICY, "foreign lineage or gate")
    inputs.load(PREFIX / "seal.json")
    prefix = inputs.load(PREFIX / "frozen.json")
    original = inputs.load(prefix["original_frozen"]["path"])
    require(original["comparison_policy"] == POLICY, "original policy mismatch")
    prepared, results, actual, reference = {}, {}, {}, {}
    for layer in (7, 8):
        inputs.load(PREFIX / f"layer{layer:02d}/seal.json")
        prepared[layer] = inputs.load(PREFIX / f"layer{layer:02d}/prepared.json")
        results[layer] = inputs.load(PREFIX / f"layer{layer:02d}/result.json")
        p = prepared[layer]
        require(p["vectors"]["layer_index"] == layer and p["vectors"]["position"] == 3,
                "foreign transaction")
        kv = p["kv_parent"]
        require(p["binary"] == kv["live_binary"]
                and kv["layer_index"] == layer and kv["valid_positions"] == [0, 1, 2]
                and "/model24_persistent_kv_selected_token_corrected_q_attempt005/"
                in kv["state"]["path"], "incompatible historical KV or binary")
        inputs.read(p["binary"]["path"])
        inputs.read(kv["state"]["path"])
        for record in kv["actual_kv_traces"]:
            inputs.read(record["path"])
        history = inputs.load(p["independent_parent"]["path"])
        require(history["history"] == HISTORY[:3] and history["layer"] == layer,
                "foreign independent history")
        inputs.read(p["input_hidden"]["path"])
        trace_path = Path(results[layer]["output_hidden"]["path"]).with_name("trace.hex")
        actual[layer] = decode_trace(inputs.read(trace_path), 3)
        reference[layer] = {
            int(stage): hex_rows(inputs.read(record["path"]))
            for stage, record in p["independent_stages"].items()
        }
    selected = inputs.load(BUILD / "position2_tail_runtime_attempt001/selected_token.json")
    require(selected["actual_token_id"] == selected["expected_token_id"] == 358
            and selected["matches"] is True, "foreign tail-selected input")
    require(all(prepared[8]["input_hidden"][key] == results[7]["output_hidden"][key]
                for key in ("path", "sha256", "bytes")), "hidden parent mismatch")
    a, r = actual[7], reference[7]
    require(all(len(a[s]) == len(r[s]) == (4864 if s in (14, 15, 16) else 896)
                for s in (12, 13, 14, 15, 16, 17, 18)), "public stage geometry mismatch")
    require(a[18] == framed_hidden(inputs.read(results[7]["output_hidden"]["path"]))
            == framed_hidden(inputs.read(prepared[8]["vectors"]["input"]["path"])),
            "layer08 incoming hidden is not actual layer07 stage18")

    tensors, tensor_bindings = {}, []
    for record in prepared[7]["vectors"]["tensors"]:
        meta = record["checkpoint_tensor"]
        if not any(f"mlp.{name}_proj." in meta["name"] for name in ("down", "gate", "up")):
            continue
        words = hex_rows(inputs.read(record["serialized"]["path"]))
        width = 2 if meta["dtype"] == "F16" else 4
        raw = b"".join(word.to_bytes(width, "little") for word in words)
        require(len(raw) == meta["bytes"]
                and hashlib.sha256(raw).hexdigest() == meta["sha256"],
                "tensor differs from official checkpoint binding")
        tensor_bindings.append(record)
        if "down_proj" in meta["name"]:
            tensors[meta["name"].rsplit(".", 1)[1]] = words
    require(len(tensor_bindings) == 9 and len(tensors) == 3, "missing MLP tensors")

    core = ROOT / "ace3/rtl/ace3_fp16_silu_gate_core.sv"
    fixed = ROOT / "ace3/rtl/ace3_fp16_fixed.sv"
    core_bytes = core.read_bytes()
    core_bindings = [rec for rec in source_records(original) if rec["path"] == str(core)]
    require(core_bindings and all(rec["sha256"] == hashlib.sha256(core_bytes).hexdigest()
                                  for rec in core_bindings), "historical SiLU source changed")
    decoder = frozen["public_interfaces"]["decoder"]
    require("parameter integer ACCURATE_SILU = (LAYER_INDEX >= 3)" in decoder,
            "historical accurate SiLU selection changed")
    snapshots = []
    source_paths = [Path(__file__), core, fixed] + [
        Path(__file__).with_name(name) for name in
        ("diagnose_lineage_a_position3_boundary.py", "trace_lineage_a_layer07_downproj.py",
         "fp16_adaptation_oracle.py", "projection_oracle.py", "awq_bit_oracle.py")
    ]
    for path in source_paths:
        data = path.read_bytes()
        previous = [rec for rec in frozen["sources"] if rec["path"] == str(path)]
        require(all(rec["sha256"] == hashlib.sha256(data).hexdigest() for rec in previous),
                "parent numerical helper changed")
        with (out / path.name).open("xb") as stream:
            stream.write(data)
        snapshots.append({"path": str(path), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest(),
                          "snapshot": str(out / path.name)})
    coordinates = prior["next_discriminating_boundary"]["coordinates"]
    require(len(coordinates) == len(set(coordinates)) == 16, "unsealed coordinate set")
    compile_command = [
        "iverilog", "-g2012", "-s", "ace3_fp16_silu_gate_core",
        "-Pace3_fp16_silu_gate_core.INTERMEDIATE_SIZE=4864",
        "-Pace3_fp16_silu_gate_core.ACCURATE_SIGMOID=1",
        "-o", str(out / "public_contract.vvp"),
        str(out / fixed.name), str(out / core.name),
    ]
    version = subprocess.run(["iverilog", "-V"], check=True, capture_output=True, text=True)
    write(out / "frozen.json", {
        "attempt_id": out.name, "kind": "lineage_A_layer07_stage16_retained_diagnostic_v1",
        "token_history": HISTORY, "profile": frozen["profile"], "comparison_policy": POLICY,
        "parent": str(PARENT), "review": str(REVIEW),
        "public_interfaces": frozen["public_interfaces"],
        "silu_public_declaration": core_bytes.decode("ascii").split(");", 1)[0] + ");",
        "silu_parameters": {"INTERMEDIATE_SIZE": 4864, "ACCURATE_SIGMOID": 1},
        "sources": snapshots, "authenticated_inputs": list(inputs.records.values()),
        "checkpoint_tensor_bindings": tensor_bindings,
        "cases": {"layer": 7, "position": 3, "weighted_coordinates": coordinates,
                  "own_input_reconstruction": "all 4864 actual and reference operands",
                  "mathematical_factorial": "sealed 16 only; 80-digit Decimal stable logistic",
                  "rounding_confirmation": "110-digit Decimal, same nearest-even binary16",
                  "residual_branch": "stage12 channel5, never folded into SiLU"},
        "compile_command": compile_command, "iverilog_version": version.stdout.splitlines()[0],
        "python": sys.version, "command": (out / "run.command.sh").read_text(),
        "boundaries": "No new RTL simulation, production change, KV import, reference "
                      "re-anchoring, tolerance change, binary64 evaluation or generation.",
    })
    with (out / "compile.log").open("x", encoding="ascii") as log:
        subprocess.run(compile_command, check=True, stdout=log, stderr=subprocess.STDOUT)

    reconstructed = {
        label: [accurate_silu(g, u) for g, u in zip(values[14], values[15], strict=True)]
        for label, values in (("actual", a), ("reference", r))
    }
    differences = {}
    for label, values in (("actual", a), ("reference", r)):
        differences[label] = [i for i, item in enumerate(reconstructed[label])
                              if item["bits"] != values[16][i]]
        require(all(silu_gate_exp(g, u) == (item["bits"], False, False)
                    for g, u, item in zip(values[14], values[15],
                                          reconstructed[label], strict=True)),
                "independent integer reconstruction disagrees with existing oracle")
    terms = projection_terms(a[16], r[16], tensors["qweight"], tensors["qzeros"],
                             tensors["scales"], 5)
    require(coordinates == [t["index"] for t in sorted(
        terms, key=lambda t: (-abs(t["delta_q48"]), t["index"]))[:16]],
        "parent weighted coordinate ranking changed")
    rows, exact_actual_patch = [], list(a[16])
    with localcontext() as ctx:
        ctx.prec = 80
        for i in coordinates:
            term = terms[i]
            parent_term = next(t for t in prior["largest_stage16_contributors"]
                               if t["index"] == i)
            require(all(term[key] == parent_term[key] for key in term),
                    "sealed weighted operands changed")
            values = {
                "aa": mathematical_silu(a[14][i], a[15][i]),
                "ar": mathematical_silu(a[14][i], r[15][i]),
                "ra": mathematical_silu(r[14][i], a[15][i]),
                "rr": mathematical_silu(r[14][i], r[15][i]),
            }
            rounded = {}
            for key, (g, u) in {
                "aa": (a[14][i], a[15][i]), "ar": (a[14][i], r[15][i]),
                "ra": (r[14][i], a[15][i]), "rr": (r[14][i], r[15][i]),
            }.items():
                rounded[key] = nearest_decimal(values[key], (g ^ u) & 0x8000)
                with localcontext() as high:
                    high.prec = 110
                    require(rounded[key] == nearest_decimal(
                        mathematical_silu(g, u), (g ^ u) & 0x8000),
                        "high-precision rounding unresolved")
            exact_actual_patch[i] = rounded["aa"]
            aa, ar, ra, rr = (values[key] for key in ("aa", "ar", "ra", "rr"))
            coefficient = Decimal(term["signed_delta"] * units(int(term["scale_bits"], 16))) / ONE
            gate_effect = ((aa - ra) + (ar - rr)) / 2
            up_effect = ((aa - ar) + (ra - rr)) / 2
            local = Decimal(units(a[16][i]) - units(r[16][i])) / ONE - (aa - rr)
            weighted = {
                "gate_drift_symmetric": coefficient * gate_effect,
                "up_drift_symmetric": coefficient * up_effect,
                "local_approximation_and_rounding_difference": coefficient * local,
            }
            retained_delta = Decimal(term["delta_q48"]) / (1 << 48)
            require(abs(sum(weighted.values()) - retained_delta) < Decimal("1e-65"),
                    "weighted gate/up/local decomposition does not close")
            rows.append({
                "index": i, "weighted_term": term,
                "stage14_actual": scalar(a[14][i]), "stage14_reference": scalar(r[14][i]),
                "stage15_actual": scalar(a[15][i]), "stage15_reference": scalar(r[15][i]),
                "actual_reconstruction": reconstructed["actual"][i],
                "reference_reconstruction": reconstructed["reference"][i],
                "mathematical_factorial": {key: str(value) for key, value in values.items()},
                "mathematical_rne_fp16": {key: scalar(value) for key, value in rounded.items()},
                "actual_local_errors": decompose(a[14][i], a[15][i], a[16][i],
                                                 reconstructed["actual"][i], aa),
                "reference_local_errors": decompose(r[14][i], r[15][i], r[16][i],
                                                    reconstructed["reference"][i], rr),
                "weighted_decomposition": {key: str(value) for key, value in weighted.items()},
                "weighted_retained_delta": str(retained_delta),
            })
        aggregate = {
            key: str(sum(Decimal(row["weighted_decomposition"][key]) for row in rows))
            for key in rows[0]["weighted_decomposition"]
        }
    write(out / "coordinates.json", rows)
    write(out / "actual_inputs.json", {
        "layer": 7, "position": 3,
        "actual": {str(s): [f"{b:04x}" for b in a[s]] for s in (12, 13, 14, 15, 16)},
        "independent": {str(s): [f"{b:04x}" for b in r[s]] for s in (12, 13, 14, 15, 16)},
        "provenance": "Decoded only from authenticated retained trace/reference bytes; "
                      "diagnostic materialization, not execution input or a new reference.",
    })
    totals = [sum(t[key] for t in terms)
              for key in ("actual_term_q48", "reference_term_q48")]
    require(round_q48(totals[0]) == a[17][5] and round_q48(totals[1]) == r[17][5],
            "downstream stage17 reconstruction changed")
    require(all(residual(values[12][i], values[17][i]) == values[18][i]
                for values in (a, r) for i in range(896)), "stage18 reconstruction changed")
    failed = [i for i, (x, y) in enumerate(zip(actual[8][8], reference[8][8], strict=True))
              if not accepts(x, y)]
    require(failed == prior["regression"]["preserved_original_layer08_stage08_failures"]
            == [13, 15], "original stage08 failures changed")
    patched_total = totals[0] + sum(
        (units(exact_actual_patch[i]) - units(a[16][i]))
        * terms[i]["signed_delta"] * units(int(terms[i]["scale_bits"], 16))
        for i in coordinates
    )
    patched17 = round_q48(patched_total)
    result = {
        "attempt_id": out.name, "status": "RETAINED_SILU_RECONSTRUCTED_NOT_REPAIRED",
        "token_history": HISTORY, "same_input_bit_difference_indices": differences,
        "existing_integer_oracle_agreement_count": 9728,
        "sealed_coordinate_count": len(rows), "weighted_decomposition_sum": aggregate,
        "selected_weighted_retained_delta": str(Fraction(
            sum(terms[i]["delta_q48"] for i in coordinates), 1 << 48)),
        "all_stage16_weighted_retained_delta": str(Fraction(totals[0] - totals[1], 1 << 48)),
        "stage17": prior["stage17"], "stage18_separate_residual": prior["stage18"],
        "layer08_incoming_hidden": prior["layer08_incoming_hidden"],
        "conditional_software_only": {
            "intervention": "Mathematical SiLU RNE on actual operands at sealed 16 only; "
                            "unselected stage16 and actual stage12 kept fixed.",
            "changed_stage16_indices": [i for i in coordinates if exact_actual_patch[i] != a[16][i]],
            "channel5_stage17": scalar(patched17),
            "channel5_stage18": scalar(residual(a[12][5], patched17)),
            "actual_channel5_stage17": scalar(a[17][5]), "actual_channel5_stage18": scalar(a[18][5]),
            "downstream_layer08_reexecution": False,
        },
        "regression": {"original_layer08_stage08_failures": failed,
                       "all_actual_and_reference_residuals_reconstructed": 1792,
                       "actual_layer08_input_bit_exact_stage18": True},
        "next_upstream_boundary": upstream_boundary(coordinates, tensor_bindings),
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "root_cause_hypothesis": "Stage14/15 operand drift, not an established local SiLU "
                                 "implementation defect; weighted factorial is conditional "
                                 "and does not identify a unique global failure producer.",
        "repair": None, "RTL_simulations": 0, "contract_compiles": 1,
        "binary64_evaluated": False, "lineage_B_imports": 0, "third_token_selected": False,
        "independent_review": "pending normal Host Reviewer",
        "elapsed_seconds": time.monotonic() - started,
    }
    result.update(classification(differences))
    if result["causal_local_implementation_defect_proven"]:
        result["next_upstream_boundary"] = None
    write(out / "result.json", result)
    artifacts = []
    for path in sorted(out.iterdir()):
        if path.name in ("execution.log", "exit_status.txt"):
            continue
        data = path.read_bytes()
        artifacts.append({"path": str(path), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
        path.chmod(0o444)
    write(out / "seal.json", {"artifacts": artifacts})
    print(json.dumps({key: result[key] for key in (
        "status", "same_input_bit_difference_indices", "weighted_decomposition_sum",
        "conditional_software_only", "regression", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--classify-retained", type=Path)
    args = parser.parse_args()
    require(args.out.is_dir()
            and {p.name for p in args.out.iterdir()} <= {"run.command.sh", "execution.log"},
            "attempt directory must be fresh except command and active log")
    if args.classify_retained:
        classify_retained(args.out.resolve(), args.classify_retained.resolve())
    else:
        run(args.out.resolve())
