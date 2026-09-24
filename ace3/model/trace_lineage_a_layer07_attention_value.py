#!/usr/bin/env python3
"""Reconstruct retained A-lineage AV operands without replaying a model."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from attention_oracle import attention_value
from diagnose_lineage_a_position3_boundary import (
    BUILD, HISTORY, POLICY, POSITIVE, PREFIX, Inputs, accepts, decode_trace,
    require, units, write,
)
from trace_lineage_a_layer07_downproj import hex_rows
from trace_lineage_a_layer07_gate_up import snapshot
from trace_lineage_a_layer07_o_projection import recorded_command
from trace_lineage_a_layer07_silu import rne_ratio, source_records


ROOT = BUILD.parent
PARENT = (BUILD / "model24_selected_token_position3_continuations"
          / "lineage_a_layer07_o_projection_cf782c98a5c5_attempt001")
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
              "cf782c98a5c5/round-0001.json")
PARTS = (
    "probability", "historical_v", "current_v",
    "historical_interaction", "current_interaction",
    "av_rounding", "local_av", "reference_recurrence",
)


def nearest_f16(numerator, denominator):
    """Independent nearest-finite grid search, with ties to even."""
    require(denominator > 0, "invalid denominator")
    magnitude = abs(numerator)
    require(magnitude <= POSITIVE[-1] * denominator, "FP16 overflow")
    index = bisect.bisect_left(POSITIVE, magnitude // denominator)
    candidates = range(max(0, index - 1), min(len(POSITIVE), index + 2))
    bits = min(candidates, key=lambda b: (
        abs(POSITIVE[b] * denominator - magnitude), b & 1))
    return bits | (0x8000 if numerator < 0 else 0)


def compose(probabilities, values):
    require(len(probabilities) == len(values) > 0, "AV row geometry")
    require(all(units(p) >= 0 for p in probabilities), "negative probability")
    total = sum(units(p) * units(v)
                for p, v in zip(probabilities, values, strict=True))
    q24 = rne_ratio(total, 1 << 24)
    # The RTL first rounds to Q24, then to FP16, normalizing exact zero.
    bits = nearest_f16(q24, 1)
    oracle = attention_value(probabilities, values, [True] * len(values),
                             [False] * len(values))
    require(not (oracle.row_error or oracle.cache_miss or oracle.invalid
                 or oracle.saturation) and oracle.value_f16 == bits,
            "independent AV reconstruction disagrees with integer oracle")
    return total, bits


def decompose(pa, pr, va, vr, actual, reference):
    require(len(pa) == len(pr) == len(va) == len(vr) > 0, "paired AV geometry")
    sums, rounded = {}, {}
    for name, p, v in (("aa", pa, va), ("ar", pa, vr),
                       ("ra", pr, va), ("rr", pr, vr)):
        sums[name], rounded[name] = compose(p, v)
    parts = dict.fromkeys(PARTS, 0)
    terms = []
    for position, (ap, rp, av, rv) in enumerate(zip(pa, pr, va, vr, strict=True)):
        ap, rp, av, rv = map(units, (ap, rp, av, rv))
        dp, dv = ap - rp, av - rv
        branch = "current" if position == len(pa) - 1 else "historical"
        pieces = {"probability": dp * rv, f"{branch}_v": rp * dv,
                  f"{branch}_interaction": dp * dv}
        require(sum(pieces.values()) == ap * av - rp * rv, "AV term closure")
        for key, value in pieces.items():
            parts[key] += value
        terms.append({"position": position, "actual_product_q48": ap * av,
                      "reference_product_q48": rp * rv, **pieces})
    delta = sums["aa"] - sums["rr"]
    require(sum(parts.values()) == delta, "AV sum closure")
    parts["av_rounding"] = ((units(rounded["aa"]) - units(rounded["rr"])) << 24) - delta
    parts["local_av"] = (units(actual) - units(rounded["aa"])) << 24
    parts["reference_recurrence"] = (units(rounded["rr"]) - units(reference)) << 24
    require(sum(parts.values()) == (units(actual) - units(reference)) << 24,
            "AV output decomposition closure")
    return {
        "sums_q48": sums, "rounded": {k: f"{v:04x}" for k, v in rounded.items()},
        "direct_reference_rne": f"{nearest_f16(sums['rr'], 1 << 24):04x}",
        "components_q48": parts, "terms": terms,
        "probability_only_within_gate": accepts(rounded["ar"], reference),
        "v_only_within_gate": accepts(rounded["ra"], reference),
    }


def run(out):
    started = time.monotonic()
    require(out.resolve().is_relative_to(BUILD.resolve()), "output must be under ignored build")
    out.mkdir(parents=True, exist_ok=False)
    inputs = Inputs()
    frozen = inputs.load(PARENT / "frozen.json", root=True)
    require(inputs.records[str(PARENT / "frozen.json")]["sha256"]
            == "9b65053f622adcf4127a7ad5db6c0f80c02d2c8d886ccad49de672d752652df9",
            "reviewed parent freeze changed")
    inputs.load(PARENT / "artifacts.json", root=True)
    require(inputs.records[str(PARENT / "artifacts.json")]["sha256"]
            == "96ca6a6293ba6990797a608a4cff8ecc5cb37a47ca28dd06ff957eb4410ede7f",
            "reviewed parent artifacts changed")
    prior = inputs.load(PARENT / "result.json")
    parent_rows = inputs.load(PARENT / "projection_coordinates.json")
    propagation = inputs.load(PARENT / "propagation.json")
    weights = inputs.read(PARENT / "witness_terms.csv")
    review = inputs.load(REVIEW, root=True)
    require(review["kind"] == "round_reviewed_handoff"
            and review["producer_role"] == "reviewer"
            and review["mission_id"] == "cf782c98a5c5"
            and review["review"]["status"] == "done", "parent review missing")
    require(frozen["token_history"] == HISTORY and frozen["comparison_policy"] == POLICY,
            "foreign lineage/policy")
    p = inputs.load(PREFIX / "layer07/prepared.json")
    result = inputs.load(PREFIX / "layer07/result.json")
    kv = p["kv_parent"]
    require(kv["layer_index"] == 7 and kv["valid_positions"] == [0, 1, 2]
            and p["binary"] == kv["live_binary"]
            and "/model24_persistent_kv_selected_token_corrected_q_attempt005/"
            in kv["state"]["path"], "foreign KV lineage/ABI")
    inputs.read(p["binary"]["path"])
    inputs.read(kv["state"]["path"])
    actual = decode_trace(inputs.read(Path(result["output_hidden"]["path"])
                                     .with_name("trace.hex")), 3)
    reference = {s: hex_rows(inputs.read(p["independent_stages"][str(s)]["path"]))
                 for s in (3, 7, 9, 10)}
    va = []
    for position, record in enumerate(kv["actual_kv_traces"]):
        trace = decode_trace(inputs.read(record["path"]), position)
        require(len(trace[7]) == 128 and trace[3] == trace[7],
                "historical V projection/cache write mismatch")
        va.append(trace[7])
    require(len(va) == 3 and actual[3] == actual[7] and len(actual[7]) == 128,
            "current V projection/cache write mismatch")
    va.append(actual[7])
    history = inputs.load(p["independent_parent"]["path"])
    require(history["history"] == HISTORY[:3] and history["layer"] == 7
            and history["own_cache"]["axes"] == ["position", "kv_head", "head_dim"],
            "foreign independent V history")
    record = history["own_cache"]["v"]
    inputs.register({"path": record["path"], "sha256": record["file_sha256"]})
    vr_array = np.load(io.BytesIO(inputs.read(record["path"])), allow_pickle=False)
    require(vr_array.shape == (3, 2, 64) and vr_array.dtype == np.dtype("<u2")
            and hashlib.sha256(vr_array.tobytes()).hexdigest() == record["semantic_sha256"],
            "independent V semantic binding")
    vr = vr_array.reshape(3, 128).tolist() + [reference[7]]
    require(reference[3] == reference[7] and len(reference[7]) == 128
            and len(actual[9]) == len(reference[9]) == 56
            and len(actual[10]) == len(reference[10]) == len(parent_rows) == 896,
            "AV boundary geometry")
    for vector in va + vr + [actual[9], reference[9], actual[10], reference[10]]:
        for bits in vector:
            units(bits)
    for index, row in enumerate(parent_rows):
        require(row["index"] == index and int(row["stage10_actual"], 16) == actual[10][index]
                and int(row["stage10_reference"], 16) == reference[10][index],
                "reviewed stage10 boundary changed")

    prefix = inputs.load(PREFIX / "frozen.json")
    original = inputs.load(prefix["original_frozen"]["path"])
    sources = []
    for name in ("ace3_attention_value_core.sv", "ace3_fp16_fixed.sv",
                 "ace3_decoder_layer0_token_engine.sv", "ace3_fp16_kv_cache.sv"):
        path = ROOT / "ace3/rtl" / name
        bound = [r for r in source_records(original) if r["path"] == str(path)]
        source = snapshot(out, path)
        require(bound and all(r["sha256"] == source["sha256"] for r in bound),
                f"historical source mismatch: {name}")
        sources.append(source)
    controller = (out / "ace3_decoder_layer0_token_engine.sv").read_text()
    wiring = [
        ".probability_f16_i(probability_mem[key_position_q]),.value_f16_i(cache_v_w)",
        ".read_position_i({8'd0,key_position_q}),.read_head_i(mapped_kv_head_w)",
        ".write_v_f16_i(v_mem[kv_flat_index_w])",
        "probability_mem[context_index_q]<=sm_out_w",
        "attention_mem[q_flat_index_w]<=av_out_w",
        "(head_q<4'd7)?4'd0:4'd1",
    ]
    require(all(text in controller for text in wiring), "AV controller wiring changed")
    for name in (Path(__file__).name, "attention_oracle.py", "fp16_adaptation_oracle.py",
                 "diagnose_lineage_a_position3_boundary.py",
                 "trace_lineage_a_layer07_silu.py", "trace_lineage_a_layer07_gate_up.py",
                 "trace_lineage_a_layer07_downproj.py", "trace_lineage_a_layer07_o_projection.py"):
        sources.append(snapshot(out, Path(__file__).with_name(name)))
    sources.append(snapshot(out, Path(__file__).parent / "tests"
                            / "test_trace_lineage_a_layer07_attention_value.py"))
    top = "ace3_attention_value_core"
    command = ["iverilog", "-g2012", "-s", top, "-o", str(out / "public_contract.vvp"),
               str(out / "ace3_fp16_fixed.sv"), str(out / f"{top}.sv")]
    write(out / "frozen.json", {
        "scope": "Retained A-only layer07/position3 AV diagnosis; no RTL replay",
        "token_history": HISTORY, "comparison_policy": POLICY,
        "parent": inputs.records[str(PARENT / "frozen.json")],
        "review": inputs.records[str(REVIEW)], "authenticated_inputs": list(inputs.records.values()),
        "sources": sources, "public_top": top,
        "public_declaration": (out / f"{top}.sv").read_text().split(");", 1)[0] + ");",
        "compile_command": command, "parameters": "Exact declaration defaults; no overrides",
        "python": sys.version, "iverilog_version": subprocess.run(
            ["iverilog", "-V"], capture_output=True, text=True, check=True).stdout,
        "coordinates": list(range(896)), "projection_witnesses": prior["witnesses"],
        "component_order": PARTS, "arithmetic": "exact Q48 dot -> RNE Q24 -> RNE FP16",
        "input_boundary": "Source-derived probability/cache read operands from authenticated "
        "writes and bound serialized state; no independent cache-read bus capture.",
        "interventions": "ar=actual P/reference V; ra=reference P/actual V. Software only; "
        "not injected into execution, not a downstream counterfactual or unique-cause proof.",
    })
    recorded_command(out, "public_contract_compile", command)
    rows = []
    for index in range(896):
        head, dimension = divmod(index, 64)
        lane = (head // 7) * 64 + dimension
        pa, pr = actual[9][head * 4:head * 4 + 4], reference[9][head * 4:head * 4 + 4]
        values_a, values_r = [v[lane] for v in va], [v[lane] for v in vr]
        measured = decompose(pa, pr, values_a, values_r, actual[10][index], reference[10][index])
        rows.append({
            "index": index, "head": head, "kv_head": head // 7, "dimension": dimension,
            "probability_actual": [f"{b:04x}" for b in pa],
            "probability_reference": [f"{b:04x}" for b in pr],
            "v_actual": [f"{b:04x}" for b in values_a],
            "v_reference": [f"{b:04x}" for b in values_r],
            "actual": f"{actual[10][index]:04x}", "reference": f"{reference[10][index]:04x}",
            "within_gate": accepts(actual[10][index], reference[10][index]), **measured,
        })
    write(out / "av_coordinates.json", rows)
    weighted = {c: dict.fromkeys(PARTS, 0) for c in prior["witnesses"]}
    seen = set()
    for term in csv.DictReader(io.StringIO(weights.decode("ascii"))):
        channel, index = int(term["channel"]), int(term["index"])
        require(channel in weighted and 0 <= index < 896 and (channel, index) not in seen,
                "duplicate/foreign projection contributor")
        seen.add((channel, index))
        row = rows[index]
        require(term["actual_stage10_bits"] == row["actual"]
                and term["reference_stage10_bits"] == row["reference"],
                "weighted stage10 operand binding")
        weight = (int(term["weight"]) - int(term["zero"])) * units(int(term["scale_bits"], 16))
        contributions = {key: row["components_q48"][key] * weight for key in PARTS}
        require(sum(contributions.values()) == int(term["delta_q48"]) << 24,
                "weighted AV term does not close")
        for key in PARTS:
            weighted[channel][key] += contributions[key]
    require(len(seen) == 72 * 896, "incomplete reviewed witness contributors")
    for channel, parts in weighted.items():
        require(sum(parts.values()) == parent_rows[channel]["attention_input_delta_q48"] << 24,
                "O projection witness decomposition does not close")
    write(out / "propagation.json", {
        "av_components_through_o_q72": weighted,
        "reviewed_downstream_unchanged": propagation,
        "scope": "All 64512 retained O terms close; inherited layer06 hidden remains separate. "
        "Existing nonlinear downstream accounting is authenticated, not recomputed or re-anchored.",
    })
    local = [r["index"] for r in rows if r["components_q48"]["local_av"]]
    recurrence = [r["index"] for r in rows if r["components_q48"]["reference_recurrence"]]
    summary = {
        "status": "LOCAL_AV_DISCREPANCY_NOT_REPAIRED" if local
        else "RETAINED_AV_RECONSTRUCTED_NOT_REPAIRED",
        "all_stage10_outputs": len(rows), "exact_av_terms": len(rows) * 4,
        "independent_oracle_comparisons": len(rows) * 4,
        "local_discrepancies": local, "reference_recurrence_discrepancies": recurrence,
        "stage10_material_failures": [r["index"] for r in rows if not r["within_gate"]],
        "stage10_bit_differences": sum(a != b for a, b in zip(actual[10], reference[10], strict=True)),
        "probability_bit_differences": sum(a != b for a, b in zip(actual[9], reference[9], strict=True)),
        "v_bit_differences_by_position": [
            sum(a != b for a, b in zip(x, y, strict=True)) for x, y in zip(va, vr, strict=True)],
        "probability_only_stage10_material_failures": [r["index"] for r in rows
                                                     if not r["probability_only_within_gate"]],
        "v_only_stage10_material_failures": [r["index"] for r in rows
                                           if not r["v_only_within_gate"]],
        "channel5_weighted_components_q72": weighted[5],
        "reviewed_ranked_contributors": prior["next_upstream_boundary"]["prioritized_input_contributors"],
        "next_boundary": "Layer07 stage08 score -> stage09 probability operator, together with "
        "stage00 -> stage03 V projection at positions0-3; distinguish softmax approximation/"
        "score trajectory from V input trajectory. The layer06 hidden branch is independent.",
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "hypothesis": "Operand drift propagates through AV and O; component magnitudes are "
        "not a unique producer or whole-trajectory repair claim.",
        "regression": "All 896 AV identities, 3584 term identities, 64512 weighted identities; "
        "original layer08 stage08 failures13/15 retained via reviewed downstream evidence.",
        "RTL_simulations": 0, "contract_compiles": 1, "binary64_evaluated": False,
        "repair": None, "elapsed_seconds": time.monotonic() - started,
        "review_status": "PENDING_NORMAL_HOST_REVIEWER",
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
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "reviewed_ranked_contributors"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out.resolve())
