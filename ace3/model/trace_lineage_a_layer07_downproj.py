#!/usr/bin/env python3
"""Exact retained-input diagnostic, not a simulator or trajectory admission."""

from __future__ import annotations

import argparse
import bisect
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

from diagnose_lineage_a_position3_boundary import (
    BUILD, HISTORY, POLICY, PREFIX, Inputs, POSITIVE, accepts, decode_trace,
    require, units, write,
)
from projection_oracle import complete_projection_output

PARENT = (BUILD / "model24_selected_token_position3_continuations"
          / "lineage_a_q_producer_968f6ec61614_attempt002")


def round_q48(total):
    """Nearest binary16 using exact distances; ties select an even significand."""
    magnitude = abs(total)
    require(magnitude <= POSITIVE[-1] << 24, "outside finite FP16 diagnostic range")
    right = bisect.bisect_left(POSITIVE, Fraction(magnitude, 1 << 24))
    bits = min({right, max(0, right - 1)},
               key=lambda b: (abs((POSITIVE[b] << 24) - magnitude), b & 1))
    # The frozen projection/residual implementation canonicalizes rounded zero.
    return bits | (0x8000 if total < 0 and bits else 0)


def scalar(bits):
    return {"bits": f"{bits:04x}", "value": units(bits) / 2**24}


def rational(total, power=48):
    return str(Fraction(total, 1 << power))


def residual(a, b):
    return round_q48((units(a) + units(b)) << 24)


def projection_terms(actual, reference, weights, zeros, scales, channel):
    require(len(actual) == len(reference) == 4864, "stage16 width")
    require(len(weights) == 4864 * 112 and len(zeros) == 38 * 112
            and len(scales) == 38 * 896, "down-projection tensor geometry")
    require(0 <= channel < 896, "channel outside public geometry")
    # Native GEMM maps logical lanes 0,1,... to packed lanes 0,4,1,5,2,6,3,7.
    lane, word = channel % 8, channel // 8
    shift = 4 * ((lane % 2) * 4 + lane // 2)
    terms = []
    for i, (a, r) in enumerate(zip(actual, reference, strict=True)):
        group = i // 128
        qw, qz = weights[i * 112 + word], zeros[group * 112 + word]
        w, z = (qw >> shift) & 15, (qz >> shift) & 15
        scale = scales[group * 896 + channel]
        coefficient = (w - z) * units(scale)
        at, rt = units(a) * coefficient, units(r) * coefficient
        terms.append({
            "index": i, "group": group, "packed_word_index": word,
            "nibble_shift": shift, "qweight_word": f"{qw:08x}",
            "qzeros_word": f"{qz:08x}", "weight": w, "zero": z,
            "signed_delta": w - z, "scale_bits": f"{scale:04x}",
            "actual_stage16_bits": f"{a:04x}", "reference_stage16_bits": f"{r:04x}",
            "actual_term_q48": at, "reference_term_q48": rt,
            "delta_q48": at - rt,
        })
    return terms


def hex_rows(data):
    return [int(line, 16) for line in data.splitlines()]


def framed_hidden(data):
    rows = data.decode("ascii").splitlines()
    require(len(rows) == 896 and all(
        len(row) == 10 and int(row[:6], 16) == i for i, row in enumerate(rows)
    ), "hidden framing mismatch")
    return [int(row[6:], 16) for row in rows]


def run(out):
    started = time.monotonic()
    inputs = Inputs()
    inputs.load(PARENT / "seal.json", root=True)
    parent = inputs.load(PARENT / "frozen.json")
    prior = inputs.load(PARENT / "result.json")
    require(parent["token_history"] == prior["token_history"] == HISTORY,
            "foreign token lineage")
    require(parent["comparison_policy"] == POLICY, "changed numerical gate")
    inputs.load(PREFIX / "seal.json")
    frozen = inputs.load(PREFIX / "frozen.json")
    original = inputs.load(frozen["original_frozen"]["path"])
    require(frozen["comparison_policy"] == original["comparison_policy"] == POLICY,
            "original numerical gate mismatch")
    prepared = {}
    results = {}
    for layer in (7, 8):
        inputs.load(PREFIX / f"layer{layer:02d}/seal.json")
        prepared[layer] = inputs.load(PREFIX / f"layer{layer:02d}/prepared.json")
        results[layer] = inputs.load(PREFIX / f"layer{layer:02d}/result.json")
        p = prepared[layer]
        require(p["vectors"]["layer_index"] == layer and p["vectors"]["position"] == 3,
                "layer/position mismatch")
        require(p["binary"] == p["kv_parent"]["live_binary"], "binary/state ABI mismatch")
        inputs.read(p["binary"]["path"])
        kv = p["kv_parent"]
        require(kv["layer_index"] == layer and kv["valid_positions"] == [0, 1, 2]
                and "/model24_persistent_kv_selected_token_corrected_q_attempt005/"
                in kv["state"]["path"], "foreign KV lineage")
        inputs.read(kv["state"]["path"])
        for trace in kv["actual_kv_traces"]:
            inputs.read(trace["path"])
        inputs.read(p["input_hidden"]["path"])
        history = inputs.load(p["independent_parent"]["path"])
        require(history["history"] == HISTORY[:3] and history["layer"] == layer,
                "independent history mismatch")
    selected = inputs.load(BUILD / "position2_tail_runtime_attempt001/selected_token.json")
    require(selected["actual_token_id"] == selected["expected_token_id"] == 358
            and selected["matches"] is True, "tail token mismatch")
    require(all(prepared[8]["input_hidden"][key] == results[7]["output_hidden"][key]
                for key in ("path", "sha256", "bytes")),
            "layer7-to-layer8 hidden parent mismatch")
    trace_paths = {
        layer: Path(result["output_hidden"]["path"]).with_name("trace.hex")
        for layer, result in results.items()
    }
    trace_bytes = {layer: inputs.read(path) for layer, path in trace_paths.items()}
    reference_bytes = {
        layer: {int(stage): inputs.read(rec["path"])
                for stage, rec in p["independent_stages"].items()}
        for layer, p in prepared.items()
    }
    actual_hidden_bytes = inputs.read(results[7]["output_hidden"]["path"])
    vector_input_bytes = inputs.read(prepared[8]["vectors"]["input"]["path"])
    tensors = {}
    tensor_bindings = []
    for record in prepared[7]["vectors"]["tensors"]:
        meta = record["checkpoint_tensor"]
        if "mlp.down_proj." not in meta["name"]:
            continue
        data = inputs.read(record["serialized"]["path"])
        words = hex_rows(data)
        width = 2 if meta["dtype"] == "F16" else 4
        raw = b"".join(word.to_bytes(width, "little") for word in words)
        require(len(raw) == meta["bytes"]
                and hashlib.sha256(raw).hexdigest() == meta["sha256"],
                "serialized tensor differs from frozen official checkpoint tensor")
        tensors[meta["name"].rsplit(".", 1)[1]] = words
        tensor_bindings.append(record)
    require(set(tensors) == {"qweight", "qzeros", "scales"}, "missing AWQ tensors")

    sources = []
    for name in (Path(__file__).name, "diagnose_lineage_a_position3_boundary.py",
                 "projection_oracle.py", "awq_bit_oracle.py", "fp16_adaptation_oracle.py"):
        path = Path(__file__).with_name(name)
        data = path.read_bytes()
        with (out / name).open("xb") as stream:
            stream.write(data)
        (out / name).chmod(0o444)
        sources.append({"path": str(path), "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    command = (out / "run.command.sh").read_bytes()
    write(out / "frozen.json", {
        "attempt_id": out.name, "kind": "lineage_A_layer07_downproj_retained_diagnostic_v1",
        "token_history": HISTORY, "profile": frozen["profile"],
        "public_interfaces": frozen["public_interfaces"],
        "comparison_policy": POLICY, "python": sys.version,
        "command": command.decode("ascii"), "sources": sources,
        "authenticated_inputs": list(inputs.records.values()),
        "checkpoint_tensor_bindings": tensor_bindings,
        "cases": {"layer": 7, "position": 3, "projection_channel": 5,
                  "projection_terms": 4864, "groups": 38, "residual_channels": 896,
                  "upstream_selection": "16 largest absolute signed term deltas, index tiebreak"},
        "arithmetic": "Exact Q48 integer products; no dequantized-weight FP16 rounding; "
                      "G128 sum then cross-group sum, one RNE; residual exact add then RNE.",
        "boundaries": "Retained actual stage16/12; independent trajectory remains unchanged. "
                      "No new RTL compile/run, source change, state import or admission.",
    })

    traces = {layer: decode_trace(data, 3) for layer, data in trace_bytes.items()}
    refs = {layer: {stage: hex_rows(data) for stage, data in stages.items()}
            for layer, stages in reference_bytes.items()}
    a, r = traces[7], refs[7]
    require(all(len(a[s]) == len(r[s]) == (4864 if s in (14, 15, 16) else 896)
                for s in (12, 14, 15, 16, 17, 18)), "stage width mismatch")
    actual_hidden = framed_hidden(actual_hidden_bytes)
    require(actual_hidden == a[18] == framed_hidden(vector_input_bytes),
            "stage18/final/actual layer08 input not bit-exact")
    terms = projection_terms(a[16], r[16], tensors["qweight"],
                             tensors["qzeros"], tensors["scales"], 5)
    with (out / "channel5_terms.csv").open("x", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(terms[0]))
        writer.writeheader()
        writer.writerows(terms)
    groups = [{
        "group": g,
        "actual_q48": sum(t["actual_term_q48"] for t in terms[g * 128:(g + 1) * 128]),
        "reference_q48": sum(t["reference_term_q48"] for t in terms[g * 128:(g + 1) * 128]),
    } for g in range(38)]
    require(all(-(1 << 95) <= g[key] < (1 << 95)
                for g in groups for key in ("actual_q48", "reference_q48")),
            "G128 accumulator overflow")
    totals = [sum(g[key] for g in groups) for key in ("actual_q48", "reference_q48")]
    require(all(-(1 << 101) <= total < (1 << 101) for total in totals),
            "cross-group accumulator overflow")
    projected = [round_q48(total) for total in totals]
    weights = tensors["qweight"][0::112]
    zeros = tensors["qzeros"][0::112]
    scales = tensors["scales"][5::896]
    local = complete_projection_output(a[16], weights, zeros, scales, 5)
    require(local[0] == totals[0] and local[1] == projected[0]
            and not local[2] and not local[3]
            and local[4] == [g["actual_q48"] for g in groups],
            "independent arithmetic and existing integer oracle disagree")
    residual_differences = [
        i for i in range(896) if residual(a[12][i], a[17][i]) != a[18][i]
    ]
    reference_residual_differences = [
        i for i in range(896) if residual(r[12][i], r[17][i]) != r[18][i]
    ]
    failed = [i for i, (actual, ref) in enumerate(zip(traces[8][8], refs[8][8], strict=True))
              if not accepts(actual, ref)]
    require(failed == prior["original_stage08_failure_indices"] == [13, 15],
            "original failing witnesses changed")
    previous = [item["largest_hidden_contributors"][0]
                for item in prior["channels"]]
    for item in previous:
        require(item["input_index"] == 5, "prior channel5 attribution changed")
        for stage in (12, 17, 18):
            for label, values in (("actual", a), ("reference", r)):
                require(item[f"layer07_stage{stage}_{label}"] == scalar(values[stage][5]),
                        "prior incoming-hidden witness mismatch")
    top = sorted(terms, key=lambda t: (-abs(t["delta_q48"]), t["index"]))[:16]
    for term in top:
        i = term["index"]
        term["stage14_actual"] = scalar(a[14][i])
        term["stage14_reference"] = scalar(r[14][i])
        term["stage15_actual"] = scalar(a[15][i])
        term["stage15_reference"] = scalar(r[15][i])
    stage12_delta = units(a[12][5]) - units(r[12][5])
    stage17_delta = units(a[17][5]) - units(r[17][5])
    stage18_delta = units(a[18][5]) - units(r[18][5])
    projection_defect = projected[0] != a[17][5]
    result = {
        "attempt_id": out.name, "status": "RETAINED_BOUNDARY_RECONSTRUCTED_NOT_REPAIRED",
        "token_history": HISTORY, "groups": groups,
        "stage17": {
            "actual": scalar(a[17][5]), "reference": scalar(r[17][5]),
            "independent_actual_input": scalar(projected[0]),
            "independent_reference_input": scalar(projected[1]),
            "actual_exact_sum": rational(totals[0]),
            "reference_exact_sum": rational(totals[1]),
            "input_drift_exact_sum": rational(totals[0] - totals[1]),
            "local_actual_bit_exact": not projection_defect,
            "reference_reconstruction_bit_exact": projected[1] == r[17][5],
            "existing_integer_oracle_exact": True,
        },
        "stage18": {
            "actual_stage12": scalar(a[12][5]), "reference_stage12": scalar(r[12][5]),
            "actual": scalar(a[18][5]), "reference": scalar(r[18][5]),
            "independent_actual_input": scalar(residual(a[12][5], a[17][5])),
            "independent_reference_input": scalar(residual(r[12][5], r[17][5])),
            "actual_all_channel_bit_difference_indices": residual_differences,
            "reference_all_channel_bit_difference_indices": reference_residual_differences,
            "stage12_signed_delta": rational(stage12_delta, 24),
            "stage17_signed_delta": rational(stage17_delta, 24),
            "stage18_signed_delta": rational(stage18_delta, 24),
            "rounding_delta": rational(stage18_delta - stage12_delta - stage17_delta, 24),
            "conditional_software_only": {
                "actual12_reference17": scalar(residual(a[12][5], r[17][5])),
                "reference12_actual17": scalar(residual(r[12][5], a[17][5])),
            },
        },
        "layer08_incoming_hidden": {
            "actual_input_bit_exact_layer07_stage18": True,
            "trajectory_bit_differences": sum(x != y for x, y in zip(a[18], r[18], strict=True)),
            "channel5_matches_prior_Q_pair_attribution": True,
            "prior_Q_channels": [row["channel"] for row in prior["channels"]],
            "scope": "Channel5 is a weighted contributor, not proof of unique cause "
                     "or sufficient removal of downstream Q/score failures.",
        },
        "largest_stage16_contributors": top,
        "next_discriminating_boundary": {
            "producer": "layer07 position3 stage16 SiLU(gate stage14) times up stage15",
            "coordinates": [term["index"] for term in top],
            "experiment": "Independently reconstruct the frozen accurate-SiLU operator "
                          "from actual stage14/15 at these signed-weighted coordinates; "
                          "separate own-input approximation/rounding from propagated "
                          "gate/up drift, retaining stage12 channel5 as a separate residual branch.",
        },
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "root_cause_hypothesis": "The measured stage16 weighted input drift and inherited "
                                 "stage12 offset may explain channel5 without a local "
                                 "down-projection/residual implementation defect.",
        "regression": {"preserved_original_layer08_stage08_failures": failed,
                       "channel5_within_unchanged_stage17_gate": accepts(a[17][5], r[17][5]),
                       "channel5_within_unchanged_stage18_gate": accepts(a[18][5], r[18][5])},
        "causal_local_implementation_defect_proven": projection_defect or bool(residual_differences),
        "repair": None, "RTL_invocations": 0, "lineage_B_imports": 0,
        "binary64_evaluated": False, "third_token_selected": False,
        "independent_review": "pending normal Host Reviewer",
        "elapsed_seconds": time.monotonic() - started,
    }
    write(out / "result.json", result)
    artifacts = []
    for path in sorted(out.iterdir()):
        if path.name in ("execution.log", "exit_status.txt"):
            continue
        data = path.read_bytes()
        artifacts.append({"path": str(path.resolve()), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
        path.chmod(0o444)
    write(out / "seal.json", {"artifacts": artifacts})
    print(json.dumps({k: result[k] for k in ("attempt_id", "status", "stage17", "stage18",
                                           "next_discriminating_boundary",
                                           "causal_local_implementation_defect_proven")},
                     indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    require(args.out.is_dir()
            and {p.name for p in args.out.iterdir()} <= {"run.command.sh", "execution.log"},
            "attempt directory must be fresh except command and active log")
    run(args.out.resolve())
