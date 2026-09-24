#!/usr/bin/env python3
"""Retained layer06 residual factorial; diagnostic arrays never enter RTL."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

from diagnose_lineage_a_position3_boundary import (
    HISTORY, POLICY, PREFIX, Inputs, accepts, decode_trace, require, units, write,
)
from trace_lineage_a_layer07_downproj import framed_hidden, hex_rows, residual, round_q48
from trace_lineage_a_layer07_gate_up import native_projection_pair, snapshot
from trace_lineage_a_layer07_qk_producers import projection, rotations
from trace_lineage_a_layer07_score_softmax_v import DESTINATION, ROOT, array
from trace_lineage_a_layer07_stage00_inputs import (
    PARTS, PRIORITIES, command, differences, reference_norm,
)


PARENT = DESTINATION / "lineage_a_layer07_stage00_9b2bcb33c886_attempt001"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
              "9b2bcb33c886/round-0001.json")
ROOT_HASHES = {
    "frozen.json": "725956281e1fa5df929ca46d242bd183778549e4fb357bc28e07e9f8f74d6b30",
    "measurements.json": "d78c872119ad03ffad573c0f982075e0ef7aedb21c1e169fcf760370f4d3da63",
    "result.json": "3af653efc04c5a954db748ace03e80f884fd5f3420c733d781460aaa9eb4de61",
}
ORDERS = {
    "stage12_first": (
        ("local_stage18", "A", "AA"), ("stage17_after_stage12", "AA", "AR"),
        ("stage12_at_reference_stage17", "AR", "RR"),
        ("reference_stage18_recurrence", "RR", "R")),
    "stage17_first": (
        ("local_stage18", "A", "AA"), ("stage12_after_stage17", "AA", "RA"),
        ("stage17_at_reference_stage12", "RA", "RR"),
        ("reference_stage18_recurrence", "RR", "R")),
}


def values(bits):
    for bit in bits:
        units(bit)
    return np.asarray(bits, dtype="<u2").view("<f2").astype(np.float64)


def reference_add(a, b):
    require(len(a) == len(b) and len(a) > 0, "residual geometry")
    summed = values(a) + values(b)
    require(np.all(np.abs(summed) <= 65504), "residual outside finite FP16 range")
    result = summed.astype("<f2").view("<u2").tolist()
    native = [0x8000 if x == y == 0x8000 else residual(x, y)
              for x, y in zip(a, b, strict=True)]
    require(result == native, "independent exact-integer/NumPy residual disagreement")
    return result


def deltas(cases, order, index):
    result = {name: units(cases[a][index]) - units(cases[b][index])
              for name, a, b in ORDERS[order]}
    require(sum(result.values()) == units(cases["A"][index]) - units(cases["R"][index]),
            "ordered residual component closure")
    return result


def hidden_factorial(actual, reference, weight):
    require(all(len(stages[s]) == len(weight)
                for stages in (actual, reference) for s in (12, 17, 18)),
            "stage12/17/18 geometry")
    hidden = {"A": actual[18], "R": reference[18]}
    for name, left, right in (
            ("AA", actual, actual), ("AR", actual, reference),
            ("RA", reference, actual), ("RR", reference, reference)):
        hidden[name] = reference_add(left[12], right[17])
    norms = {name: reference_norm(h, weight) for name, h in hidden.items()}
    reductions = {}
    for name, h in hidden.items():
        f = values(h)
        mean = float(np.mean(f * f))
        reductions[name] = {
            "sum_squares_q48": sum(units(x)**2 for x in h),
            "mean_binary64_hex": mean.hex(),
            "inverse_rms_binary64_hex": float(1 / np.sqrt(mean + 1e-6)).hex(),
        }
    rows = []
    for i in range(len(weight)):
        a12, r12 = units(actual[12][i]), units(reference[12][i])
        a17, r17 = units(actual[17][i]), units(reference[17][i])
        parts = {
            "stage12": a12 - r12, "stage17": a17 - r17,
            "sum_rounding_difference": (
                units(hidden["AA"][i]) - a12 - a17
                - units(hidden["RR"][i]) + r12 + r17),
            "local_stage18": units(hidden["A"][i]) - units(hidden["AA"][i]),
            "reference_stage18_recurrence": units(hidden["RR"][i]) - units(hidden["R"][i]),
        }
        require(sum(parts.values()) == units(actual[18][i]) - units(reference[18][i]),
                "unrounded residual contribution closure")
        rows.append({
            "index": i, "actual_stage12": f"{actual[12][i]:04x}",
            "reference_stage12": f"{reference[12][i]:04x}",
            "actual_stage17": f"{actual[17][i]:04x}",
            "reference_stage17": f"{reference[17][i]:04x}",
            "hidden_bits": {name: f"{h[i]:04x}" for name, h in hidden.items()},
            "hidden_components_q24": parts,
            "norm_bits": {name: f"{h[i]:04x}" for name, h in norms.items()},
            "norm_components_q24": {order: deltas(norms, order, i) for order in ORDERS},
            "norm_interaction_q24": (
                units(norms["AA"][i]) - units(norms["AR"][i])
                - units(norms["RA"][i]) + units(norms["RR"][i])),
        })
    return hidden, norms, reductions, rows


def denominator_split(hidden, norms, reductions, weight, index):
    result = {}
    w = units(weight[index]) / 2**24
    for order, transitions in ORDERS.items():
        result[order] = {}
        for name, a, b in transitions:
            ha, hb = (units(hidden[k][index]) / 2**24 for k in (a, b))
            ia, ib = (float.fromhex(reductions[k]["inverse_rms_binary64_hex"])
                      for k in (a, b))
            # Full-vector denominators; the ordered scalar identity is exact
            # over the recorded binary64 factors, not a fixed-denominator Jacobian.
            numerator = (Fraction(ha) - Fraction(hb)) * Fraction(w) * Fraction(ia)
            denominator = Fraction(hb) * Fraction(w) * (Fraction(ia) - Fraction(ib))
            rounded = Fraction(units(norms[a][index]) - units(norms[b][index]), 2**24)
            result[order][name] = {
                "numerator_change": str(numerator), "denominator_change": str(denominator),
                "rounding_and_binary64_operation_remainder": str(rounded - numerator - denominator),
                "rounded_total": str(rounded),
                "sum_squares_delta_q48": (
                    reductions[a]["sum_squares_q48"] - reductions[b]["sum_squares_q48"]),
            }
    return result


def project_cases(norms, tensors, channel):
    measured = native_projection_pair(norms["A"], norms["R"], tensors, channel)
    coefficients = [(term["weight"] - term["zero"]) * units(int(term["scale_bits"], 16))
                    for term in measured["terms"]]
    totals = {name: sum(units(x) * c for x, c in zip(h, coefficients, strict=True))
              for name, h in norms.items()}
    require(totals["A"] == measured["actual_q48"]
            and totals["R"] == measured["reference_q48"], "native G128 weighted identity")
    raw = {}
    for name, total in totals.items():
        biased = total + (units(tensors["bias"][channel]) << 24)
        raw[name] = round_q48(biased)
        if biased < 0 and raw[name] == 0:
            raw[name] = 0x8000
    require(projection(norms["A"], tensors, channel, True)[0] == raw["A"],
            "independent native GEMM projection disagreement")
    return raw, coefficients, totals


def run(out):
    require(out.parent == DESTINATION.resolve()
            and out.name.startswith("lineage_a_layer06_stage18_"), "output scope")
    out.mkdir(exist_ok=False)
    inputs = Inputs()
    roots = {}
    for name, digest in ROOT_HASHES.items():
        roots[name] = inputs.load(PARENT / name, root=True)
        require(inputs.records[str(PARENT / name)]["sha256"] == digest,
                f"reviewed parent changed: {name}")
    freeze, parent = roots["frozen.json"], roots["measurements.json"]
    review = inputs.load(REVIEW, root=True)
    require(review["kind"] == "round_reviewed_handoff"
            and review["producer_role"] == "reviewer"
            and review["mission_id"] == "9b2bcb33c886"
            and review["review"]["status"] == "done", "missing genuine predecessor review")
    require(freeze["token_history"] == HISTORY and freeze["comparison_policy"] == POLICY,
            "foreign lineage or numerical policy")
    inputs.load(PREFIX / "seal.json")
    p6 = inputs.load(PREFIX / "layer06/prepared.json")
    r6 = inputs.load(PREFIX / "layer06/result.json")
    p7 = inputs.load(PREFIX / "layer07/prepared.json")
    r7 = inputs.load(PREFIX / "layer07/result.json")
    require(r6["layer"] == 6 and r6["position"] == 3
            and r6["output_hidden"] == p7["input_hidden"], "layer06-to-layer07 identity")
    for prepared in (p6, p7):
        require(prepared["binary"] == prepared["kv_parent"]["live_binary"],
                "retained binary/KV ABI mismatch")
        inputs.read(prepared["binary"]["path"])
        inputs.read(prepared["kv_parent"]["state"]["path"])
    a = {3: decode_trace(inputs.read(Path(r6["output_hidden"]["path"]).with_name("trace.hex")), 3)}
    r = {3: {s: hex_rows(inputs.read(p6["independent_stages"][str(s)]["path"]))
             for s in (12, 17, 18)}}
    require(a[3][18] == framed_hidden(inputs.read(r6["output_hidden"]["path"]))
            == framed_hidden(inputs.read(p7["vectors"]["input"]["path"])),
            "trace/final/canonical consumed vector identity")
    require(r[3][18] == hex_rows(inputs.read(r6["independent_hidden"]["path"])),
            "reference final identity")
    historical_path = Path(p7["kv_parent"]["layer_result"]["path"]).parent.parent / "layer06/result.json"
    historical = inputs.load(historical_path)
    entry = historical["positions"][2]
    require(entry["position"] == 2 and entry["layer_index"] == 6, "historical layer identity")
    a[2] = decode_trace(inputs.read(Path(entry["output"]["path"]).with_name("trace.hex")), 2)
    require(a[2][18] == framed_hidden(inputs.read(entry["output"]["path"])),
            "historical trace/final identity")
    ref_history = inputs.load(p6["independent_parent"]["path"])
    require(ref_history["layer"] == 6 and ref_history["history"] == HISTORY[:3]
            and ref_history["positions"] == [2], "historical reference identity")
    r[2] = {}
    for stage in (12, 17, 18):
        records = [x for x in ref_history["stages"] if x["path"].endswith(f"/stage{stage:02}.npy")]
        require(len(records) == 1, "historical stage ambiguity")
        r[2][stage] = array(inputs, records[0]).reshape(-1).tolist()
    qk_path = next(Path(x["path"]) for x in freeze["authenticated_inputs"]
                   if x["path"].endswith("qk_bd2d1b4dffa3_attempt002/measurements.json"))
    qk = inputs.load(qk_path)
    producers = {(x["kind"], x["position"], x["channel"]): x for x in qk["producers"]}
    tensors = {"q": {}, "k": {}}
    weight = None
    checkpoint_history = inputs.load(p7["independent_parent"]["path"])
    for record in p7["vectors"]["tensors"]:
        meta = record["checkpoint_tensor"]
        name = meta["name"]
        kind = next((k for k in tensors if f".{k}_proj." in name), None)
        if kind is None and name != "model.layers.7.input_layernorm.weight":
            continue
        data = hex_rows(inputs.read(record["serialized"]["path"]))
        raw = b"".join(x.to_bytes(2 if meta["dtype"] == "F16" else 4, "little") for x in data)
        require(len(raw) == meta["bytes"] and hashlib.sha256(raw).hexdigest() == meta["sha256"]
                and any(x["name"] == name and x["sha256"] == meta["sha256"]
                        for x in checkpoint_history["checkpoint_tensor_hashes"]),
                "serialized official tensor binding")
        if kind is None:
            weight = data
        else:
            tensors[kind][name.rsplit(".", 1)[1]] = data
    require(weight is not None and len(weight) == 896
            and all(set(t) == {"qweight", "qzeros", "scales", "bias"} for t in tensors.values()),
            "normalization/projection parameter geometry")
    # Bind the retained layer07 inputs, not just an independently reconstructed match.
    history7 = inputs.load(p7["kv_parent"]["layer_result"]["path"])
    require(a[2][18] == framed_hidden(inputs.read(history7["positions"][2]["vectors"]["input"]["path"])),
            "historical layer07 consumed input identity")
    for position in (2, 3):
        rows = parent["norm_coordinates"][str(position)]
        require([f"{x:04x}" for x in a[position][18]] == [x["actual_hidden"] for x in rows]
                and [f"{x:04x}" for x in r[position][18]] == [x["reference_hidden"] for x in rows],
                "reviewed normalized-input hidden identity")
    paths = {Path(module.__file__).resolve() for module in tuple(sys.modules.values())
             if getattr(module, "__file__", None)
             and Path(module.__file__).resolve().is_relative_to(ROOT / "ace3/model")
             and Path(module.__file__).suffix == ".py"}
    paths.update((Path(__file__).resolve(),
                  ROOT / "ace3/model/tests/test_trace_lineage_a_layer06_stage18_hidden.py"))
    paths.update(ROOT / "ace3/rtl" / name for name in (
        "ace3_fp16_residual_add_core.sv", "ace3_fp16_fixed.sv",
        "ace3_fp16_rmsnorm_core.sv", "ace3_decoder_layer0_token_engine.sv"))
    previous_sources = {x["path"]: x for x in freeze["sources"]}
    sources = []
    for path in sorted(paths):
        record = snapshot(out, path)
        if str(path) in previous_sources:
            require(record["sha256"] == previous_sources[str(path)]["sha256"],
                    f"reviewed helper/producer changed: {path}")
        sources.append(record)
    compile_commands = [
        ["iverilog", "-g2012", "-s", top, "-o", str(out / f"{top}.vvp"),
         str(out / f"{top}.sv"), str(out / "ace3_fp16_fixed.sv")]
        for top in ("ace3_fp16_residual_add_core", "ace3_fp16_rmsnorm_core")]
    tests = [sys.executable, "-B", "-m", "unittest", "discover",
             "-s", str(ROOT / "ace3/model/tests"),
             "-p", "test_trace_lineage_a_layer06_stage18_hidden.py"]
    command(out, "iverilog_version", ["iverilog", "-V"])
    write(out / "frozen.json", {
        "scope": "retained A-lineage software factorial, no injected activations or RTL replay",
        "token_history": HISTORY, "comparison_policy": POLICY,
        "parent_root_hashes": ROOT_HASHES, "authenticated_inputs": list(inputs.records.values()),
        "sources": sources, "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "commands": [*compile_commands, tests], "priority_scores": PRIORITIES, "orders": ORDERS,
        "public_declarations": {
            top: (out / f"{top}.sv").read_text().split(");", 1)[0] + ");"
            for top in ("ace3_fp16_residual_add_core", "ace3_fp16_rmsnorm_core")},
        "score_denominator": 2**51, "epsilon_binary64_hex": float(1e-6).hex(),
        "position2_upstream_provenance": "conditional; content-bound retained manifests, not unique producer proof",
        "parent_position2_reference_previously_bound": freeze["position2_reference_hidden_previously_bound"],
        "historical_positions01": freeze["historical_positions01"],
        "attribution": "two ordered full-vector residual factorials; interaction retained, no unique-cause assertion",
        "limits": ["No general defect inferred from drift", "No policy change or binary64 evaluation",
                   "No new token, trajectory admission, simulator result, timing, or hardware claim"],
    })
    for i, argv in enumerate(compile_commands):
        command(out, f"public_contract_compile_{i}", argv)
    command(out, "unit_tests", tests)
    cases, norms, reductions, coordinates = {}, {}, {}, {}
    for position in (2, 3):
        cases[position], norms[position], reductions[position], coordinates[position] = hidden_factorial(
            a[position], r[position], weight)
        old = parent["norm_coordinates"][str(position)]
        require(norms[position]["A"] == [int(x["chain_bits"][2], 16) for x in old]
                and norms[position]["R"] == [int(x["chain_bits"][3], 16) for x in old],
                "full-denominator reference recurrence disagrees with reviewed endpoints")
        for label, column in (("retained", 0), ("native", 1), ("reference", 4)):
            norms[position][label] = [int(x["chain_bits"][column], 16) for x in old]
    selected_scores = [next(x for x in qk["scores"] if (x["head"], x["key_position"]) == key)
                       for key in PRIORITIES]
    selected = set()
    for score in selected_scores:
        for row in score["all_dimensions"]:
            for kind in ("q", "k"):
                position = 3 if kind == "q" else score["key_position"]
                if position >= 2:
                    ch = row[f"{kind}_channel"]
                    selected.add((kind, position, ch // 64, ch % 32))
    ranked = {(x["kind"], x["position"], x["channel"]): x
              for x in parent["ranked_projection_inputs"]}
    require(("q", 3, 442) in ranked and ("q", 3, 410) in ranked, "priority Q identities missing")
    post, projection_rows = {}, []
    for kind, position, head, dim in sorted(selected):
        channels = (head * 64 + dim, head * 64 + dim + 32)
        raw = {}
        for channel in channels:
            raw[channel], coeffs, totals = project_cases(norms[position], tensors[kind], channel)
            old = ranked.get((kind, position, channel))
            if old is not None:
                terms = []
                for old_term in old["top_weighted_inputs"]:
                    i = old_term["input_index"]
                    require((units(norms[position]["retained"][i])
                             - units(norms[position]["reference"][i])) * coeffs[i]
                            == old_term["delta_q48"], "ranked input weighted identity")
                    terms.append({
                        "input_index": i, "coefficient_q24": coeffs[i],
                        "retained_parent_term": old_term, "layer06": coordinates[position][i],
                        "full_denominator_accounting": denominator_split(
                            cases[position], norms[position], reductions[position], weight, i),
                        "components_q48": {
                            order: {name: value * coeffs[i]
                                    for name, value in deltas(norms[position], order, i).items()}
                            for order in ORDERS},
                    })
                projection_rows.append({
                    "kind": kind, "position": position, "channel": channel,
                    "raw_bits": {name: f"{bits:04x}" for name, bits in raw[channel].items()},
                    "dot_totals_q48": totals, "ranked_inputs": terms,
                    "dot_components_q48": {
                        order: {name: totals[left] - totals[right]
                                for name, left, right in transitions}
                        for order, transitions in ORDERS.items()},
                })
        producer = producers[kind, position, channels[0]]
        coeff = [int(x, 16) for x in producer["cos_sin_bits"]]
        rotated = {name: rotations([raw[ch][name] for ch in channels], coeff, position, dim)[2]
                   for name in norms[position]}
        for side, channel in enumerate(channels):
            post[kind, position, channel] = {name: bits[side] for name, bits in rotated.items()}
    score_rows = []
    old_transitions = list(zip(PARTS, ("retained", "native", "A", "R"),
                               ("native", "A", "R", "reference"), strict=True))
    for score, previous in zip(selected_scores, parent["scores"], strict=True):
        require((score["head"], score["key_position"]) ==
                (previous["head"], previous["key_position"]), "priority score ordering")
        totals = {order: {kind: {name: 0 for name, _, _ in transitions} for kind in ("q", "k")}
                  for order, transitions in ORDERS.items()}
        inherited = {kind: dict.fromkeys(PARTS, 0) for kind in ("q", "k")}
        for row in score["all_dimensions"]:
            for kind in ("q", "k"):
                position = 3 if kind == "q" else score["key_position"]
                if position < 2:
                    continue
                p = post[kind, position, row[f"{kind}_channel"]]
                other = (("k", score["key_position"], row["k_channel"]) if kind == "q"
                         else ("q", 3, row["q_channel"]))
                coefficient = units(int(producers[other]["reference"], 16))
                for name, left, right in old_transitions:
                    inherited[kind][name] += (units(p[left]) - units(p[right])) * coefficient
                for order, transitions in ORDERS.items():
                    for name, left, right in transitions:
                        totals[order][kind][name] += (units(p[left]) - units(p[right])) * coefficient
        require(inherited == previous["stage00_parts_dot_q48"], "reviewed score decomposition changed")
        for order in ORDERS:
            for kind in ("q", "k"):
                require(sum(totals[order][kind].values()) ==
                        inherited[kind]["layer06_hidden_under_reference_rmsnorm"],
                        "hidden-to-score exact closure")
        score_rows.append({
            "head": score["head"], "key_position": score["key_position"],
            "hidden_split_dot_q48": totals,
            "hidden_split_scaled": {
                order: {kind: {name: str(Fraction(value, 2**51)) for name, value in parts.items()}
                        for kind, parts in kinds.items()} for order, kinds in totals.items()},
            "stage12_order_interaction_dot_q48": {
                kind: totals["stage17_first"][kind]["stage12_after_stage17"]
                - totals["stage12_first"][kind]["stage12_at_reference_stage17"]
                for kind in ("q", "k")},
            "retained_parent_score": previous,
            "position2_upstream_provenance_conditional": score["key_position"] == 2,
        })
    discrepancies = {
        p: {"local_stage18": differences(cases[p]["A"], cases[p]["AA"]),
            "reference_stage18": differences(cases[p]["RR"], cases[p]["R"]),
            "stage12_drift": differences(a[p][12], r[p][12]),
            "stage17_drift": differences(a[p][17], r[p][17]),
            "stage18_drift": differences(a[p][18], r[p][18]),
            "stage18_material": [i for i in range(896) if not accepts(a[p][18][i], r[p][18][i])]}
        for p in (2, 3)}
    priority_inputs = sorted({term["input_index"] for row in projection_rows
                              if row["kind"] == "q" and row["position"] == 3
                              and row["channel"] in (410, 442) for term in row["ranked_inputs"]})
    write(out / "measurements.json", {
        "hidden_coordinates": coordinates, "reductions": reductions,
        "ranked_projections": projection_rows, "scores": score_rows,
        "discrepancies": discrepancies, "historical_v_unsplit": parent["historical_v_unsplit"],
    })
    summary = {
        "status": "LAYER06_STAGE18_BOUNDARY_LOCALIZED_NOT_REPAIRED",
        "counts": {p: {name: len(indices) for name, indices in d.items()} for p, d in discrepancies.items()},
        "normalization_coordinates": 1792, "ranked_projection_channels": len(projection_rows),
        "score_rows": len(score_rows), "priority_q410_q442_input_indices": priority_inputs,
        "next_boundary": (
            "Layer06 position3 stage12 = incoming hidden + stage11 O projection, and "
            "stage17 = down projection of stage16 SiLU-times-up. Use both full-denominator "
            "ordered score splits and retained ranked input coordinates to select the "
            "affected upstream cone; stage18 sum rounding and nonlinear interaction are "
            "separate, not unique producer attribution. Position2 provenance remains "
            "conditional; positions0/1 K/V remain unsplit."),
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "hypothesis": "Stage12 and stage17 inherited differences combine with residual rounding "
                      "and global RMSNorm reduction; correct local addition need not repair trajectory drift.",
        "regression": "Exact integer/NumPy residual agreement, full-vector normalization, both "
                      "substitution orders, native AWQ coefficients and six retained score closures.",
        "general_defect_proven": False, "repair": None, "RTL_simulations": 0,
        "contract_compiles": 2, "binary64_evaluated": False,
        "review_status": "PENDING_NORMAL_HOST_REVIEWER",
    }
    if any(d["local_stage18"] or d["reference_stage18"] for d in discrepancies.values()):
        summary["status"] = "STAGE18_RECONSTRUCTION_DISCREPANCY_NOT_REPAIRED"
        summary["next_boundary"] = "Resolve listed local/reference stage18 reconstruction differences first."
    write(out / "result.json", summary)
    for path in out.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out.resolve())
