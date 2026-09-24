#!/usr/bin/env python3
"""Discriminate retained A-lineage score, softmax and V producer drift."""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from attention_oracle import attention_score, attention_softmax, exp_approx_q24
from diagnose_lineage_a_position3_boundary import (
    BUILD, HISTORY, POLICY, PREFIX, Inputs, accepts, decode_trace, require, units, write,
)
from projection_oracle import complete_projection_output
from trace_lineage_a_layer07_attention_value import nearest_f16
from trace_lineage_a_layer07_downproj import hex_rows, round_q48
from trace_lineage_a_layer07_gate_up import native_projection_pair, snapshot
from trace_lineage_a_layer07_o_projection import recorded_command
from trace_lineage_a_layer07_silu import rne_ratio, source_records


ROOT = BUILD.parent
DESTINATION = BUILD / "model24_selected_token_position3_continuations"
PARENT = DESTINATION / "lineage_a_layer07_attention_value_81f9befac236_attempt002"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
              "81f9befac236/round-0001.json")
HEADS = (4, 6, 12)
AV_INDICES = (291, 419, 436, 782, 258)
PROBABILITY_PARTS = (
    "local_softmax", "q24_normalization_rounding", "exponential_approximation",
    "score_trajectory", "reference_recurrence",
)


def decimal_bits(value):
    numerator, denominator = value.as_integer_ratio()
    return nearest_f16(numerator << 24, denominator)


def softmax_components(scores, reference_scores, actual, reference):
    """Exact telescoping split, not an attribution by largest raw delta."""
    def accurate(row):
        with localcontext() as context:
            context.prec = 80
            values = [Decimal(units(b)) / (1 << 24) for b in row]
            maximum = max(values)
            exponentials = [(v - maximum).exp() for v in values]
            denominator = sum(exponentials)
            decimal = [decimal_bits(v / denominator) for v in exponentials]
        floats = [math.exp(float(v - maximum)) for v in values]
        float_bits = [decimal_bits(Decimal.from_float(v / math.fsum(floats)))
                      for v in floats]
        require(decimal == float_bits, "independent exponential rounding disagreement")
        return decimal

    count = len(scores)
    require(count > 0 and all(len(row) == count
                            for row in (reference_scores, actual, reference)),
            "softmax operand geometry")
    oracle = attention_softmax(scores, list(range(count)), [True] * count,
                               [False] * count, [False] * count, count - 1)
    require(not (oracle.invalid or oracle.cache_miss or oracle.row_error),
            "invalid retained softmax operands")
    maximum = max(map(units, scores))
    exponentials = [exp_approx_q24(maximum - units(b)) for b in scores]
    denominator = sum(exponentials)
    direct = [nearest_f16(e << 24, denominator) for e in exponentials]
    local = [nearest_f16(rne_ratio(e << 24, denominator), 1) for e in exponentials]
    require(local == list(oracle.probabilities_f16), "softmax integer oracle disagreement")
    exact_a, exact_r = accurate(scores), accurate(reference_scores)
    rows = []
    for i in range(count):
        chain = [actual[i], local[i], direct[i], exact_a[i], exact_r[i], reference[i]]
        parts = {key: units(chain[j]) - units(chain[j + 1])
                 for j, key in enumerate(PROBABILITY_PARTS)}
        require(sum(parts.values()) == units(actual[i]) - units(reference[i]),
                "probability decomposition does not close")
        rows.append({
            "key_position": i, "actual": f"{actual[i]:04x}",
            "reference": f"{reference[i]:04x}", "local_reconstructed": f"{local[i]:04x}",
            "accurate_on_actual_scores": f"{exact_a[i]:04x}",
            "accurate_on_reference_scores": f"{exact_r[i]:04x}",
            "approximate_exp_q24": exponentials[i], "denominator_q24": denominator,
            "components_q24": parts, "within_gate": accepts(actual[i], reference[i]),
        })
    return rows


def score_bits(q, k):
    require(len(q) == len(k) == 64, "score public head geometry")
    total = sum(units(a) * units(b) for a, b in zip(q, k, strict=True))
    bits = nearest_f16(rne_ratio(total, 1 << 27), 1)
    oracle = attention_score(q, k, [True] * 64, 3, 0)
    require(not (oracle.invalid or oracle.cache_miss or oracle.saturation)
            and bits == oracle.score_f16, "independent score oracle disagreement")
    return total, bits


def v_projection(activation, tensors, channel):
    measured = native_projection_pair(activation, activation, tensors, channel)
    bias = tensors["bias"][channel]
    two_round = round_q48((units(measured["actual_bits"]) + units(bias)) << 24)
    single_round = round_q48(measured["actual_q48"] + (units(bias) << 24))
    words = len(tensors["bias"]) // 8
    accumulator, oracle, invalid, saturation, _ = complete_projection_output(
        activation, tensors["qweight"][channel // 8::words],
        tensors["qzeros"][channel // 8::words],
        tensors["scales"][channel::len(tensors["bias"])], channel % 8, bias)
    require(not (invalid or saturation) and oracle == two_round
            and accumulator == measured["actual_q48"], "independent V oracle disagreement")
    return two_round, single_round, measured


def array(inputs, record):
    inputs.register({"path": record["path"], "sha256": record["file_sha256"]})
    value = np.load(io.BytesIO(inputs.read(record["path"])), allow_pickle=False)
    require(list(value.shape) == record["shape"] and value.dtype == np.dtype("<u2")
            and hashlib.sha256(value.tobytes()).hexdigest() == record["semantic_sha256"],
            "independent array semantic binding")
    return value


def run(out):
    require(out.is_relative_to(DESTINATION.resolve()), "output outside mission build scope")
    out.mkdir(parents=True, exist_ok=False)
    inputs = Inputs()
    for filename, digest in (
        ("frozen.json", "234a165e397f83e9a5f95d9f77a2f723e3ea76e59354a39c249897531c6fecda"),
        ("artifacts.json", "7dc176368ed47c948baa5521847148b5696de2ee5d7e7fa61066fd4d072654be"),
    ):
        inputs.load(PARENT / filename, root=True)
        require(inputs.records[str(PARENT / filename)]["sha256"] == digest,
                "reviewed parent changed")
    parent = inputs.load(PARENT / "result.json")
    av = inputs.load(PARENT / "av_coordinates.json")
    inputs.read(PARENT / "propagation.json")
    review = inputs.load(REVIEW, root=True)
    require(review["kind"] == "round_reviewed_handoff"
            and review["producer_role"] == "reviewer"
            and review["mission_id"] == "81f9befac236"
            and review["review"]["status"] == "done", "parent review not accepted")
    prepared = inputs.load(PREFIX / "layer07/prepared.json")
    result = inputs.load(PREFIX / "layer07/result.json")
    prefix = inputs.load(PREFIX / "frozen.json")
    original = inputs.load(prefix["original_frozen"]["path"])
    policy = inputs.load(original["independent_policy"]["path"])
    require(original["comparison_policy"] == policy["comparison"] == POLICY,
            "numerical policy changed")
    kv = prepared["kv_parent"]
    require(kv["valid_positions"] == [0, 1, 2] and kv["layer_index"] == 7
            and prepared["binary"] == kv["live_binary"]
            and "/model24_persistent_kv_selected_token_corrected_q_attempt005/"
            in kv["state"]["path"], "foreign state lineage or ABI")
    inputs.read(kv["state"]["path"])
    inputs.read(prepared["binary"]["path"])
    history = inputs.load(prepared["independent_parent"]["path"])
    require(history["history"] == HISTORY[:3] and history["layer"] == 7
            and history["own_cache"]["axes"] == ["position", "kv_head", "head_dim"],
            "foreign independent history")
    inputs.load(kv["layer_result"]["path"])
    actual = [decode_trace(inputs.read(r["path"]), p)
              for p, r in enumerate(kv["actual_kv_traces"])]
    actual.append(decode_trace(inputs.read(Path(result["output_hidden"]["path"])
                                          .with_name("trace.hex")), 3))
    reference = {s: hex_rows(inputs.read(prepared["independent_stages"][str(s)]["path"]))
                 for s in (0, 3, 4, 6, 7, 8, 9)}
    kr = array(inputs, history["own_cache"]["k"]).reshape(3, 128).tolist() + [reference[6]]
    vr = array(inputs, history["own_cache"]["v"]).reshape(3, 128).tolist() + [reference[3]]
    reference_norms = {3: reference[0]}
    # Only consume historical normalized inputs carrying pre-existing bindings.
    for record in history["stages"]:
        if Path(record["path"]).name == "stage00.npy":
            position = int(Path(record["path"]).parent.name.removeprefix("position"))
            reference_norms[position] = array(inputs, record).reshape(-1).tolist()
    require(len(actual) == 4 and all(len(a[0]) == 896 and len(a[3]) == 128
                                    and a[3] == a[7] for a in actual)
            and reference[3] == reference[7], "V projection/cache-write geometry or identity")
    require(len(actual[3][4]) == len(reference[4]) == 896
            and len(actual[3][8]) == len(reference[8]) == 56
            and len(actual[3][9]) == len(reference[9]) == 56, "score/probability geometry")
    tensors = {}
    for record in prepared["vectors"]["tensors"]:
        meta = record["checkpoint_tensor"]
        if ".self_attn.v_proj." not in meta["name"]:
            continue
        values = hex_rows(inputs.read(record["serialized"]["path"]))
        raw = b"".join(v.to_bytes(2 if meta["dtype"] == "F16" else 4, "little")
                       for v in values)
        require(len(raw) == meta["bytes"] and hashlib.sha256(raw).hexdigest() == meta["sha256"],
                "V serialized tensor differs from official checkpoint")
        require(any(r["name"] == meta["name"] and r["sha256"] == meta["sha256"]
                    for r in history["checkpoint_tensor_hashes"]), "foreign independent V tensor")
        tensors[meta["name"].rsplit(".", 1)[1]] = values
    require(set(tensors) == {"qweight", "qzeros", "scales", "bias"}, "missing V tensors")
    for index in AV_INDICES:
        row = av[index]
        head, dim = divmod(index, 64)
        lane = head // 7 * 64 + dim
        require(row["index"] == index
                and row["probability_actual"] == [f"{b:04x}" for b in actual[3][9][head*4:head*4+4]]
                and row["probability_reference"] == [f"{b:04x}" for b in reference[9][head*4:head*4+4]]
                and row["v_actual"] == [f"{a[3][lane]:04x}" for a in actual]
                and row["v_reference"] == [f"{v[lane]:04x}" for v in vr],
                "reviewed AV producer binding changed")

    rtl_names = (
        "ace3_fp16_fixed.sv", "ace3_attention_score_core.sv", "ace3_attention_softmax_core.sv",
        "ace3_awq_w4a16_g128_dot_lane.sv", "ace3_q47_48_to_f16_rne.sv",
        "ace3_awq_w4a16_projection_engine.sv", "ace3_qkv_projection_cluster.sv",
        "ace3_decoder_layer0_token_engine.sv",
    )
    sources = []
    for name in rtl_names:
        path = ROOT / "ace3/rtl" / name
        record = snapshot(out, path)
        bound = [r for r in source_records(original) if r["path"] == str(path)]
        require(bound and all(r["sha256"] == record["sha256"] for r in bound),
                f"historical RTL source changed: {name}")
        sources.append(record)
    for name in (
        Path(__file__).name, "attention_oracle.py", "fp16_adaptation_oracle.py",
        "projection_oracle.py", "awq_bit_oracle.py", "diagnose_lineage_a_position3_boundary.py",
        "trace_lineage_a_layer07_attention_value.py", "trace_lineage_a_layer07_downproj.py",
        "trace_lineage_a_layer07_gate_up.py", "trace_lineage_a_layer07_silu.py",
        "trace_lineage_a_layer07_o_projection.py",
    ):
        sources.append(snapshot(out, Path(__file__).with_name(name)))
    sources.append(snapshot(out, Path(__file__).parent / "tests"
                            / "test_trace_lineage_a_layer07_score_softmax_v.py"))
    tops = ("ace3_attention_score_core", "ace3_attention_softmax_core",
            "ace3_awq_w4a16_projection_engine")
    compile_commands = [
        ["iverilog", "-g2012", "-s", top, "-o", str(out / f"{top}.vvp")]
        + [str(out / name) for name in rtl_names[:6]] for top in tops
    ]
    test_command = ["env", f"PYTHONPATH={ROOT / 'ace3/model'}",
                    sys.executable, "-B", "-m", "unittest", "discover",
                    "-s", str(ROOT / "ace3/model/tests"),
                    "-p", "test_trace_lineage_a_layer07_score_softmax_v.py"]
    write(out / "frozen.json", {
        "scope": "Retained A layer07 position3 producer discrimination; no model/RTL replay",
        "token_history": HISTORY, "comparison_policy": POLICY,
        "authenticated_inputs": list(inputs.records.values()), "sources": sources,
        "review": inputs.records[str(REVIEW)], "heads": HEADS, "av_indices": AV_INDICES,
        "v_positions": [0, 1, 2, 3], "reference_stage00_positions": sorted(reference_norms),
        "public_declarations": {top: (out / f"{top}.sv").read_text().split(");", 1)[0] + ");"
                                for top in tops},
        "working_directory": str(ROOT),
        "compile_commands": compile_commands, "test_command": test_command,
        "python": sys.version, "numpy": np.__version__,
        "iverilog": subprocess.run(["iverilog", "-V"], capture_output=True,
                                   text=True, check=True).stdout,
        "normalization_oracle": "80-digit Decimal exp and binary64 math.exp, separate FP16 RNE",
        "score_arithmetic": "exact FP16 Q48 dot /8, RNE Q24, RNE FP16",
        "v_arithmetic": "native G128 GEMM nibble order, no zero+1, FP16 scales; "
                        "RNE FP16 projection then FP16 bias addition; direct biased dot diagnostic",
        "boundary": "Source-derived reads from authenticated trace writes and serialized state, "
                    "not a captured cache-read bus. Conditional arithmetic is not trajectory admission.",
        "exclusions": ["layer06 incoming-hidden investigation", "B lineage", "binary64 admission",
                       "software injection", "third token", "new arithmetic adoption"],
    })
    for top, command in zip(tops, compile_commands, strict=True):
        recorded_command(out, f"{top}_contract_compile", command)
    recorded_command(out, "unit_tests", test_command)

    probability_rows, scores = {}, []
    for head in HEADS:
        begin = head * 4
        probability_rows[head] = softmax_components(
            actual[3][8][begin:begin+4], reference[8][begin:begin+4],
            actual[3][9][begin:begin+4], reference[9][begin:begin+4])
        qa, qr = actual[3][4][head*64:head*64+64], reference[4][head*64:head*64+64]
        lane = head // 7 * 64
        for position in range(4):
            ka, rk = actual[position][6][lane:lane+64], kr[position][lane:lane+64]
            total, local = score_bits(qa, ka)
            reference_total, _ = score_bits(qr, rk)
            reference_direct = nearest_f16(reference_total, 1 << 27)
            q_delta = sum((units(a)-units(r))*units(k)
                          for a, r, k in zip(qa, qr, rk, strict=True))
            k_delta = sum(units(q)*(units(a)-units(r))
                          for q, a, r in zip(qr, ka, rk, strict=True))
            interaction = total - reference_total - q_delta - k_delta
            a, r = actual[3][8][begin+position], reference[8][begin+position]
            scores.append({
                "head": head, "key_position": position, "actual": f"{a:04x}",
                "reference": f"{r:04x}", "local_reconstructed": f"{local:04x}",
                "reference_direct_reconstructed": f"{reference_direct:04x}",
                "local_discrepancy_q24": units(a)-units(local),
                "reference_recurrence_q24": units(reference_direct)-units(r),
                "q_trajectory_dot_q48": q_delta, "k_trajectory_dot_q48": k_delta,
                "interaction_dot_q48": interaction, "actual_dot_q48": total,
                "reference_dot_q48": reference_total, "within_gate": accepts(a, r),
            })

    v_rows, av_rows = [], []
    for index in AV_INDICES:
        head, dim = divmod(index, 64)
        lane = head // 7 * 64 + dim
        contributions = dict.fromkeys(PROBABILITY_PARTS, 0)
        v_contributions = []
        for position in range(4):
            two, single, measured = v_projection(actual[position][0], tensors, lane)
            a, r = actual[position][3][lane], vr[position][lane]
            parts = {
                "local_v": units(a)-units(two),
                "projection_bias_rounding": units(two)-units(single),
                "incoming_or_reference_recurrence": units(single)-units(r),
            }
            ref_single = None
            if position in reference_norms:
                _, ref_single, _ = v_projection(reference_norms[position], tensors, lane)
                parts.pop("incoming_or_reference_recurrence")
                parts["stage00_trajectory"] = units(single)-units(ref_single)
                parts["reference_recurrence"] = units(ref_single)-units(r)
            require(sum(parts.values()) == units(a)-units(r), "V producer closure")
            reference_probability = units(reference[9][head * 4 + position])
            v_contributions.append({key: value * reference_probability
                                    for key, value in parts.items()})
            terms = sorted(measured["terms"], key=lambda t: abs(t["actual_term_q48"]),
                           reverse=True)[:8]
            v_rows.append({
                "av_index": index, "lane": lane, "position": position,
                "actual": f"{a:04x}", "reference": f"{r:04x}",
                "two_round": f"{two:04x}", "single_round": f"{single:04x}",
                "reference_single_round": None if ref_single is None else f"{ref_single:04x}",
                "components_q24": parts, "within_gate": accepts(a, r),
                "actual_dot_q48": measured["actual_q48"],
                "largest_actual_dot_terms": terms,
            })
            for key, value in probability_rows[head][position]["components_q24"].items():
                contributions[key] += value * units(r)
        require(sum(contributions.values()) == av[index]["components_q48"]["probability"],
                "probability producer contributions do not close at retained AV")
        require(sum(sum(row.values()) for row in v_contributions[:3])
                == av[index]["components_q48"]["historical_v"]
                and sum(v_contributions[3].values()) == av[index]["components_q48"]["current_v"],
                "V producer contributions do not close at retained AV")
        av_rows.append({"av_index": index, "probability_components_q48": contributions,
                        "v_components_by_position_q48": v_contributions,
                        "retained_av_components_q48": av[index]["components_q48"]})
    write(out / "measurements.json", {
        "scores": scores, "probabilities": probability_rows, "v_producers": v_rows,
        "av_propagation": av_rows,
    })
    summary = {
        "status": "RETAINED_PRODUCER_DISCRIMINATION_NOT_REPAIRED",
        "score_coordinates": len(scores), "probability_coordinates": sum(map(len, probability_rows.values())),
        "v_producer_coordinates": len(v_rows),
        "score_local_discrepancies": [r for r in scores if r["local_discrepancy_q24"]],
        "score_reference_recurrence_discrepancies": [r for r in scores if r["reference_recurrence_q24"]],
        "probability_component_nonzero_counts": {
            key: sum(bool(r["components_q24"][key]) for rows in probability_rows.values()
                     for r in rows) for key in PROBABILITY_PARTS},
        "probability_component_l1_q24_by_head": {
            head: {key: sum(abs(r["components_q24"][key]) for r in rows)
                   for key in PROBABILITY_PARTS}
            for head, rows in probability_rows.items()},
        "v_component_l1_q24_by_position": {
            position: {
                key: sum(abs(r["components_q24"].get(key, 0)) for r in v_rows
                         if r["position"] == position)
                for key in sorted({key for r in v_rows if r["position"] == position
                                   for key in r["components_q24"]})}
            for position in range(4)},
        "v_local_discrepancies": [r for r in v_rows if r["components_q24"]["local_v"]],
        "v_reference_recurrence_discrepancies": [
            r for r in v_rows if r["components_q24"].get("reference_recurrence", 0)],
        "v_reference_stage00_unavailable_positions": sorted(set(range(4))-reference_norms.keys()),
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "hypothesis": "Measured telescoping terms distinguish score trajectory, native exponential "
                      "approximation, normalization rounding, and V projection/bias rounding. "
                      "Unbound historical normalized references are not reconstructed or guessed.",
        "next_boundary": "Rank the measured score-trajectory versus exponential terms per selected "
                         "head. Trace dominant Q/K terms to layer07 stage04/06 producers. For V, "
                         "trace stage00 input drift at positions2/3 and authenticate historical "
                         "position0/1 stage00 before splitting their input/reference remainder. "
                         "Keep the independent layer06 hidden branch separate.",
        "regression": "12 own-input score identities; 12 probability telescoping identities; "
                      "20 native biased V identities; five AV probability and V contribution closures. "
                      "Inherited layer08 stage08 failures13/15 and prior evidence remain unchanged.",
        "repair": None, "RTL_simulations": 0, "contract_compiles": len(tops),
        "binary64_evaluated": False, "review_status": "PENDING_NORMAL_HOST_REVIEWER",
        "parent_next_boundary": parent["next_boundary"],
    }
    write(out / "result.json", summary)
    write(out / "artifacts.json", [
        {"path": str(p), "bytes": p.stat().st_size,
         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(out.iterdir()) if p.is_file()
    ])
    for path in out.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out.resolve())
