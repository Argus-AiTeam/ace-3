#!/usr/bin/env python3
"""Independent retained-input AWQ reconstruction, not a new RTL trajectory."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from diagnose_lineage_a_position3_boundary import (
    BUILD, HISTORY, POLICY, PREFIX, Inputs, accepts, decode_trace, require,
    units, write,
)
from projection_oracle import complete_projection_output
from trace_lineage_a_layer07_downproj import (
    framed_hidden, hex_rows, projection_terms, residual, round_q48, scalar,
)
from trace_lineage_a_layer07_silu import accurate_silu, source_records

ROOT = BUILD.parent
PARENT = (BUILD / "model24_selected_token_position3_continuations"
          / "lineage_a_layer07_silu_24788b40db9f_attempt002")
PARENT_SEAL_SHA256 = "4a6571dbe264868f92befd08590411d6a289336f364ce9567c757a3bd8a1c436"


def native_projection_pair(actual, reference, tensors, channel):
    """Exact signed Q48 products; independently unpack native GEMM output lanes."""
    require(len(actual) == len(reference) and len(actual) % 128 == 0,
            "projection input geometry")
    groups = len(actual) // 128
    require(groups > 0 and len(tensors["scales"]) % groups == 0, "scale geometry")
    outputs = len(tensors["scales"]) // groups
    require(outputs % 8 == 0 and 0 <= channel < outputs, "output geometry")
    words = outputs // 8
    require(len(tensors["qweight"]) == len(actual) * words
            and len(tensors["qzeros"]) == groups * words, "packed geometry")
    shift = (0, 16, 4, 20, 8, 24, 12, 28)[channel % 8]
    word = channel // 8
    group_actual, group_reference, terms = [0] * groups, [0] * groups, []
    for index, (a, r) in enumerate(zip(actual, reference, strict=True)):
        group = index // 128
        qw = tensors["qweight"][index * words + word]
        qz = tensors["qzeros"][group * words + word]
        weight, zero = (qw >> shift) & 15, (qz >> shift) & 15
        scale = tensors["scales"][group * outputs + channel]
        coefficient = (weight - zero) * units(scale)
        at, rt = units(a) * coefficient, units(r) * coefficient
        group_actual[group] += at
        group_reference[group] += rt
        terms.append({
            "index": index, "group": group, "nibble_shift": shift,
            "qweight_word": f"{qw:08x}", "qzeros_word": f"{qz:08x}",
            "weight": weight, "zero": zero, "scale_bits": f"{scale:04x}",
            "actual_stage13_bits": f"{a:04x}", "reference_stage13_bits": f"{r:04x}",
            "actual_term_q48": at, "reference_term_q48": rt, "delta_q48": at - rt,
        })
    require(all(-(1 << 95) <= n < 1 << 95
                for n in group_actual + group_reference), "group accumulator overflow")
    totals = [sum(group_actual), sum(group_reference)]
    require(all(-(1 << 101) <= n < 1 << 101 for n in totals),
            "cross-group accumulator overflow")
    return {
        "actual_bits": round_q48(totals[0]), "reference_bits": round_q48(totals[1]),
        "actual_q48": totals[0], "reference_q48": totals[1],
        "actual_groups_q48": group_actual, "reference_groups_q48": group_reference,
        "terms": terms,
    }


def oracle_pair(actual, reference, tensors, channel):
    groups = len(actual) // 128
    outputs = len(tensors["scales"]) // groups
    words, word = outputs // 8, channel // 8
    weights = tensors["qweight"][word::words]
    zeros = tensors["qzeros"][word::words]
    scales = tensors["scales"][channel::outputs]
    return [complete_projection_output(x, weights, zeros, scales, channel % 8)
            for x in (actual, reference)]


def snapshot(out, path):
    data = path.read_bytes()
    target = out / path.name
    with target.open("xb") as stream:
        stream.write(data)
    target.chmod(0o444)
    return {"path": str(path), "snapshot": str(target), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def run(out):
    started = time.monotonic()
    inputs = Inputs()
    inputs.load(PARENT / "seal.json", root=True)
    require(inputs.records[str(PARENT / "seal.json")]["sha256"] == PARENT_SEAL_SHA256,
            "sealed 24788b40db9f root changed")
    parent = inputs.load(PARENT / "frozen.json")
    prior = inputs.load(PARENT / "result.json")
    measurement = Path(parent["measurement_attempt"])
    inputs.load(measurement / "seal.json")
    measured = inputs.load(measurement / "frozen.json")
    coordinates = prior["next_upstream_boundary"]["output_coordinates"]
    require(coordinates == measured["cases"]["weighted_coordinates"]
            and len(coordinates) == len(set(coordinates)) == 16, "changed weighted coordinates")
    require(parent["token_history"] == prior["token_history"] == HISTORY
            and parent["comparison_policy"] == POLICY, "foreign lineage or gate")
    inputs.load(PREFIX / "seal.json")
    prefix = inputs.load(PREFIX / "frozen.json")
    original = inputs.load(prefix["original_frozen"]["path"])
    policy = inputs.load(original["independent_policy"]["path"])
    require(original["comparison_policy"] == policy["comparison"] == POLICY,
            "independent gate changed")
    prepared, results, actual, reference = {}, {}, {}, {}
    for layer in (7, 8):
        inputs.load(PREFIX / f"layer{layer:02d}/seal.json")
        p = prepared[layer] = inputs.load(PREFIX / f"layer{layer:02d}/prepared.json")
        results[layer] = inputs.load(PREFIX / f"layer{layer:02d}/result.json")
        require(p["vectors"]["layer_index"] == layer and p["vectors"]["position"] == 3,
                "foreign layer/position")
        kv = p["kv_parent"]
        require(p["binary"] == kv["live_binary"] and kv["layer_index"] == layer
                and kv["valid_positions"] == [0, 1, 2]
                and "/model24_persistent_kv_selected_token_corrected_q_attempt005/"
                in kv["state"]["path"], "foreign KV lineage or binary ABI")
        for record in [p["binary"], kv["state"], p["input_hidden"], *kv["actual_kv_traces"]]:
            inputs.read(record["path"])
        history = inputs.load(p["independent_parent"]["path"])
        require(history["history"] == HISTORY[:3] and history["layer"] == layer,
                "independent history changed")
        trace = Path(results[layer]["output_hidden"]["path"]).with_name("trace.hex")
        actual[layer] = decode_trace(inputs.read(trace), 3)
        reference[layer] = {int(stage): hex_rows(inputs.read(record["path"]))
                            for stage, record in p["independent_stages"].items()}
    selected = inputs.load(BUILD / "position2_tail_runtime_attempt001/selected_token.json")
    require(selected["actual_token_id"] == selected["expected_token_id"] == 358
            and selected["matches"] is True, "A tail token changed")
    require(all(prepared[8]["input_hidden"][k] == results[7]["output_hidden"][k]
                for k in ("path", "sha256", "bytes")), "hidden parent binding mismatch")
    a, r = actual[7], reference[7]
    require(all(len(a[s]) == len(r[s]) == (4864 if s in (14, 15, 16) else 896)
                for s in range(12, 19)), "public stage geometry mismatch")
    require(a[18] == framed_hidden(inputs.read(results[7]["output_hidden"]["path"]))
            == framed_hidden(inputs.read(prepared[8]["vectors"]["input"]["path"])),
            "layer07 final/actual layer08 incoming-hidden mismatch")
    tensors, tensor_bindings, norm_bindings = {}, [], []
    for record in prepared[7]["vectors"]["tensors"]:
        meta = record["checkpoint_tensor"]
        if "post_attention_layernorm" in meta["name"]:
            norm_bindings.append(record)
        if not any(f"mlp.{name}_proj." in meta["name"] for name in ("gate", "up", "down")):
            continue
        words = hex_rows(inputs.read(record["serialized"]["path"]))
        raw = b"".join(w.to_bytes(2 if meta["dtype"] == "F16" else 4, "little") for w in words)
        require(len(raw) == meta["bytes"] and hashlib.sha256(raw).hexdigest() == meta["sha256"],
                "serialized MLP tensor differs from frozen official checkpoint")
        name, kind = meta["name"].split(".")[-2:]
        tensors.setdefault(name, {})[kind] = words
        tensor_bindings.append(record)
    require(len(tensor_bindings) == 9, "missing official MLP tensors")

    sources = []
    rtl_names = ("ace3_fp16_fixed.sv", "ace3_awq_w4a16_g128_dot_lane.sv",
                 "ace3_q47_48_to_f16_rne.sv", "ace3_awq_w4a16_projection_engine.sv")
    for name in (*rtl_names, "ace3_decoder_layer0_token_engine.sv"):
        path = ROOT / "ace3/rtl" / name
        records = [rec for rec in source_records(original) if rec["path"] == str(path)]
        record = snapshot(out, path)
        require(records and all(rec["sha256"] == record["sha256"] for rec in records),
                f"historical source changed: {name}")
        sources.append(record)
    model_names = (Path(__file__).name, "diagnose_lineage_a_position3_boundary.py",
                   "trace_lineage_a_layer07_downproj.py", "trace_lineage_a_layer07_silu.py",
                   "projection_oracle.py", "awq_bit_oracle.py", "fp16_adaptation_oracle.py")
    for name in model_names:
        record = snapshot(out, Path(__file__).with_name(name))
        source_parent = parent if name == "trace_lineage_a_layer07_silu.py" else measured
        old = [rec for rec in source_parent["sources"] if rec["path"] == record["path"]]
        require(all(rec["sha256"] == record["sha256"] for rec in old),
                f"parent numerical helper changed: {name}")
        sources.append(record)
    test_path = Path(__file__).parent / "tests/test_trace_lineage_a_layer07_gate_up.py"
    sources.append(snapshot(out, test_path))
    channels = sorted({8 * (c // 8) + lane for c in coordinates for lane in range(8)})
    top = "ace3_awq_w4a16_projection_engine"
    compile_command = [
        "iverilog", "-g2012", "-s", top, f"-P{top}.IN_FEATURES=896",
        f"-P{top}.OUT_FEATURES=4864", f"-P{top}.BIAS_ENABLE=0",
        f"-P{top}.SINGLE_ROUND_BIAS=0", "-o", str(out / "public_contract.vvp"),
        *(str(out / name) for name in rtl_names),
    ]
    test_command = [sys.executable, "-m", "unittest", "discover", "-s", "ace3/model/tests",
                    "-p", "test_trace_lineage_a_layer07*.py"]
    for name, command in (("compile", compile_command), ("regression", test_command)):
        with (out / f"{name}.command.sh").open("x", encoding="ascii") as stream:
            stream.write(shlex.join(command) + "\n")
    with (out / "version.command.sh").open("x", encoding="ascii") as stream:
        stream.write("iverilog -V\n")
    version = subprocess.run(["iverilog", "-V"], capture_output=True, text=True, check=True)
    declaration = (out / f"{top}.sv").read_text().split(");", 1)[0] + ");"
    write(out / "frozen.json", {
        "attempt_id": out.name, "kind": "retained_A_stage13_gate_up_reconstruction_v1",
        "parent_seal_sha256": PARENT_SEAL_SHA256, "token_history": HISTORY,
        "profile": original["profile"], "comparison_policy": POLICY,
        "independent_reference_policy": policy, "sources": sources,
        "authenticated_inputs": list(inputs.records.values()),
        "checkpoint_tensor_bindings": tensor_bindings,
        "command": (out / "run.command.sh").read_text(),
        "compile_command": compile_command, "regression_command": test_command,
        "iverilog_version": version.stdout + version.stderr, "python": sys.version,
        "public_top": top, "public_declaration": declaration,
        "parameters": {"IN_FEATURES": 896, "OUT_FEATURES": 4864,
                       "BIAS_ENABLE": 0, "SINGLE_ROUND_BIAS": 0},
        "cases": {"weighted_coordinates": coordinates, "packed_word_surroundings": channels,
                  "input_coordinates": list(range(896)), "group_size": 128, "groups": 7,
                  "term_ranking": "descending absolute signed Q48 drift; index tie-break",
                  "conditional_factorial": "sealed 16 only; all other actual stage16 held fixed"},
        "arithmetic": "Exact Q24 activation * signed INT4 delta * Q24 scale; Q48 group "
                      "and cross-group sums; one nearest-even FP16 rounding; no qzero +1.",
        "scope": "Retained numerical diagnosis. Compile is interface evidence only. "
                 "No RTL run, trajectory substitution, changed numerical gate, or B import.",
    })
    for name, command in (("compile", compile_command), ("regression", test_command)):
        with (out / f"{name}.log").open("x") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

    rows, mismatches, ref_differences, influential = [], [], [], set()
    predictions = {}
    with (out / "weighted_stage13_terms.csv").open("x", newline="") as stream:
        writer = None
        for stage, name in ((14, "gate_proj"), (15, "up_proj")):
            for channel in channels:
                pair = native_projection_pair(a[13], r[13], tensors[name], channel)
                oracle = oracle_pair(a[13], r[13], tensors[name], channel)
                for label, check in zip(("actual", "reference"), oracle, strict=True):
                    require(not check[2] and not check[3]
                            and check[0] == pair[f"{label}_q48"]
                            and check[1] == pair[f"{label}_bits"]
                            and check[4] == pair[f"{label}_groups_q48"],
                            "independent reconstruction disagrees with existing integer oracle")
                predictions[stage, channel] = pair["actual_bits"], pair["reference_bits"]
                if pair["actual_bits"] != a[stage][channel]:
                    mismatches.append([stage, channel])
                if pair["reference_bits"] != r[stage][channel]:
                    ref_differences.append([stage, channel])
                terms = pair.pop("terms")
                drift = pair["actual_q48"] - pair["reference_q48"]
                require(sum(t["delta_q48"] for t in terms) == drift, "term drift does not close")
                rounded_delta = units(a[stage][channel]) - units(r[stage][channel])
                pair.update({
                    "stage": stage, "channel": channel, "sealed_weighted": channel in coordinates,
                    "actual": scalar(a[stage][channel]), "reference": scalar(r[stage][channel]),
                    "exact_input_drift": str(Fraction(drift, 1 << 48)),
                    "retained_output_delta": str(Fraction(rounded_delta, 1 << 24)),
                    "rounding_and_local_delta": str(Fraction((rounded_delta << 24) - drift, 1 << 48)),
                    "actual_local_delta": str(Fraction(
                        units(a[stage][channel]) - units(pair["actual_bits"]), 1 << 24)),
                })
                if channel in coordinates:
                    ranked = sorted(terms, key=lambda t: (-abs(t["delta_q48"]), t["index"]))
                    pair["largest_input_terms"] = ranked[:16]
                    influential.update(t["index"] for t in ranked[:4])
                    for term in terms:
                        row = {"stage": stage, "channel": channel, **term}
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                rows.append(pair)
    write(out / "projections.json", rows)
    write(out / "retained_stages.json", {
        "actual_layer07": {s: a[s] for s in range(12, 19)},
        "independent_layer07": {s: r[s] for s in range(12, 19)},
    })
    silu_actual_differences = [
        i for i in range(4864) if accurate_silu(a[14][i], a[15][i])["bits"] != a[16][i]
    ]
    require(not silu_actual_differences, "previously cleared own-input stage16 changed")
    down = tensors["down_proj"]
    terms = projection_terms(a[16], r[16], down["qweight"], down["qzeros"], down["scales"], 5)
    totals = [sum(t[f"{label}_term_q48"] for t in terms) for label in ("actual", "reference")]
    require([round_q48(t) for t in totals] == [a[17][5], r[17][5]],
            "previously cleared stage17 channel5 changed")
    require(all(residual(v[12][i], v[17][i]) == v[18][i]
                for v in (a, r) for i in range(896)), "previously cleared stage18 changed")
    factorial = {}
    for label, gate_operand, up_operand in (("aa", 0, 0), ("ra", 1, 0),
                                           ("ar", 0, 1), ("rr", 1, 1)):
        patched = list(a[16])
        for channel in coordinates:
            patched[channel] = accurate_silu(
                predictions[14, channel][gate_operand],
                predictions[15, channel][up_operand],
            )["bits"]
        patch_terms = projection_terms(patched, r[16], down["qweight"], down["qzeros"],
                                       down["scales"], 5)
        total = sum(t["actual_term_q48"] for t in patch_terms)
        stage17 = round_q48(total)
        factorial[label] = {
            "stage16_changed_indices": [i for i in coordinates if patched[i] != a[16][i]],
            "channel5_exact_down_sum": str(Fraction(total, 1 << 48)),
            "channel5_stage17": scalar(stage17),
            "channel5_stage18_actual_stage12_fixed": scalar(residual(a[12][5], stage17)),
        }
    if not mismatches:
        require(not factorial["aa"]["stage16_changed_indices"],
                "exact gate/up reconstruction did not propagate exactly through stage16")
    failed_scores = [i for i, (av, rv) in enumerate(zip(actual[8][8], reference[8][8], strict=True))
                     if not accepts(av, rv)]
    require(failed_scores == [13, 15], "original layer08 failures changed")
    input_differences = [i for i in range(896) if a[13][i] != r[13][i]]
    result = {
        "attempt_id": out.name,
        "status": ("LOCAL_PROJECTION_DISCREPANCY_NOT_REPAIRED" if mismatches
                   else "RETAINED_GATE_UP_RECONSTRUCTED_NOT_REPAIRED"),
        "token_history": HISTORY, "lineage_B_imports": 0, "RTL_simulations": 0,
        "contract_compiles": 1, "third_token_selected": False,
        "binary64_evaluated": False, "repair": None,
        "weighted_coordinates": coordinates, "surrounding_channel_count": len(channels),
        "reconstructed_actual_outputs": len(rows),
        "existing_integer_oracle_comparisons": 2 * len(rows),
        "actual_local_bit_differences": mismatches,
        "reference_exact_projection_bit_differences": ref_differences,
        "causal_local_implementation_defect_proven": bool(mismatches),
        "stage13_bit_difference_indices": input_differences,
        "projection_trajectory_bit_differences": [
            [row["stage"], row["channel"]] for row in rows
            if row["actual"]["bits"] != row["reference"]["bits"]],
        "stage16_actual_reconstructed": 4864,
        "stage17_channel5": {
            "actual": scalar(a[17][5]), "reference": scalar(r[17][5]),
            "actual_sum": str(Fraction(totals[0], 1 << 48)),
            "reference_sum": str(Fraction(totals[1], 1 << 48)),
            "input_drift_sum": str(Fraction(totals[0] - totals[1], 1 << 48)),
        },
        "stage18_channel5": {
            "actual": scalar(a[18][5]), "reference": scalar(r[18][5]),
            "actual_stage12": scalar(a[12][5]), "reference_stage12": scalar(r[12][5]),
            "stage12_delta": str(Fraction(units(a[12][5]) - units(r[12][5]), 1 << 24)),
            "stage17_delta": str(Fraction(units(a[17][5]) - units(r[17][5]), 1 << 24)),
            "stage18_delta": str(Fraction(units(a[18][5]) - units(r[18][5]), 1 << 24)),
            "all_actual_and_reference_residuals_reconstructed": 1792,
        },
        "conditional_software_only": {
            "scope": "Reconstruct gate/up from actual/reference stage13 at sealed 16 channels; "
                     "all unselected stage16 and actual stage12 fixed. Not an RTL intervention "
                     "or layer08 score repair; projection rounding retained.",
            "factorial": factorial,
        },
        "layer08_incoming_hidden": {
            "actual_input_bit_exact_layer07_stage18": True,
            "trajectory_bit_difference_count": sum(x != y for x, y in zip(a[18], r[18], strict=True)),
            "inherited_Q_channels": prior["layer08_incoming_hidden"]["prior_Q_channels"],
            "original_stage08_failures": [
                {"index": i, "actual": scalar(actual[8][8][i]), "reference": scalar(reference[8][8][i])}
                for i in failed_scores],
        },
        "next_upstream_boundary": None if mismatches else {
            "layer": 7, "position": 3, "producer": "stage12 -> stage13 post-attention RMSNorm",
            "actual_input": "authenticated actual stage12[0:896]",
            "actual_output": "authenticated actual stage13[0:896]",
            "prioritized_output_coordinates": sorted(influential),
            "required_surrounding_coordinates": list(range(896)),
            "reason": "RMSNorm reduction couples all 896 stage12 coordinates. Reconstruct its "
                      "own-input operation and FP16 boundary using official norm weights; "
                      "separate normalization error from inherited stage12 drift.",
            "official_norm_tensor_bindings": norm_bindings,
            "separate_branch": "Stage12 channel5 directly contributes to stage18 and must remain separate.",
        },
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "root_cause_hypothesis": "Inherited stage13 drift propagates through exact native AWQ "
                                 "gate/up arithmetic at inspected channels; stage12->13 RMSNorm "
                                 "versus its inherited input remains unlocalized. Not a unique root cause.",
        "regression": "Require exact integer group sums and RNE agreement for both operands and "
                      "all packed-word neighbors; preserve layer08 stage08 failures13/15 and "
                      "separate same-input agreement from whole-trajectory acceptance.",
        "independent_review": "pending normal Host Reviewer",
        "elapsed_seconds": time.monotonic() - started,
    }
    write(out / "result.json", result)
    artifacts = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "run.log":
            data = path.read_bytes()
            artifacts.append({"path": str(path), "bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest()})
            path.chmod(0o444)
    write(out / "seal.json", {"artifacts": artifacts})
    print(json.dumps({
        key: result[key] for key in ("status", "reconstructed_actual_outputs",
                                    "existing_integer_oracle_comparisons",
                                    "actual_local_bit_differences",
                                    "reference_exact_projection_bit_differences",
                                    "stage17_channel5", "stage18_channel5",
                                    "layer08_incoming_hidden", "elapsed_seconds")
    }, indent=2))
    print(f"Stage13 input bit differences: {len(input_differences)}")
    print("Next boundary:", result["next_upstream_boundary"]["producer"]
          if result["next_upstream_boundary"] else "local projection discrepancy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(args.out.is_dir(), "pre-create fresh attempt with exact run.command.sh")
    run(args.out.resolve())
