#!/usr/bin/env python3
"""Retained layer06 producer interventions; never feed diagnostic arrays to RTL."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch

from diagnose_lineage_a_position3_boundary import (
    HISTORY, POLICY, POSITIVE, PREFIX, Inputs, decode_trace, require, units, write,
)
from trace_lineage_a_layer06_stage18_hidden import (
    ORDERS, project_cases, reference_add, values,
)
from official_single_decoder_layer import _torch_rmsnorm
from trace_lineage_a_layer07_downproj import framed_hidden, hex_rows, round_q48
from trace_lineage_a_layer07_gate_up import native_projection_pair, oracle_pair, snapshot
from trace_lineage_a_layer07_qk_producers import rotations
from trace_lineage_a_layer07_score_softmax_v import DESTINATION, ROOT
from trace_lineage_a_layer07_stage00_inputs import (
    PRIORITIES, command, differences, reference_norm,
)


PARENT = DESTINATION / "lineage_a_layer06_stage18_22f4b93c3e45_attempt001"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
              "22f4b93c3e45/round-0001.json")
ROOT_HASHES = {
    "frozen.json": "9b7cc6fde2a7a5c91b73e69e41ee3c6de69d17f3c6f0598a9d49d17262d637d4",
    "measurements.json": "a23be8278ac10877f542572fa5409d1dcdd8eaf2340d036b8f17f17bcec2688b",
    "result.json": "303aefc5fa7fac08a628ce38fdbe52d76ab7442062acb070851ba8b68016f60a",
}
INNER_ORDERS = {
    "incoming_first": (
        ("local_stage12", "A", "AA"), ("O_after_incoming", "AA", "AR"),
        ("incoming_at_reference_O", "AR", "RR"),
        ("reference_stage12_recurrence", "RR", "R")),
    "O_first": (
        ("local_stage12", "A", "AA"), ("incoming_after_O", "AA", "RA"),
        ("O_at_reference_incoming", "RA", "RR"),
        ("reference_stage12_recurrence", "RR", "R")),
}
DOWN_CHAIN = (
    ("local_down_projection", "A", "DA"),
    ("stage16_activation", "DA", "DR"),
    ("reference_down_recurrence", "DR", "R"),
)
FAILED_ATTEMPT = DESTINATION / "lineage_a_layer06_stage12_stage17_1bf9d4ba6988_attempt001"


def exact_norm_round(numerator, variance):
    """Round numerator/sqrt(variance) by exact squared binary16 midpoints."""
    require(variance > 0, "nonpositive RMSNorm variance")
    magnitude = numerator * numerator
    require(magnitude <= Fraction(POSITIVE[-1], 2**24)**2 * variance,
            "exact norm outside finite FP16")
    lo, hi = 0, len(POSITIVE) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if Fraction(POSITIVE[mid], 2**24)**2 * variance < magnitude:
            lo = mid + 1
        else:
            hi = mid
    upper, lower = lo, max(0, lo - 1)
    midpoint = Fraction(POSITIVE[upper] + POSITIVE[lower], 2**25)
    boundary = midpoint * midpoint * variance
    bits = lower if magnitude < boundary else upper
    if magnitude == boundary:
        bits = lower if lower % 2 == 0 else upper
    return bits | (0x8000 if numerator < 0 else 0)


def norm_boundary_measurement(hidden, weight):
    """Keep the retained Torch recurrence; expose, never waive, oracle differences."""
    x, w = values(hidden), values(weight)
    primary = _torch_rmsnorm(torch.from_numpy(x), w).to(torch.float16).numpy().view("<u2").tolist()
    independent = (x * (1 / np.sqrt(np.mean(x*x) + 1e-6)) * w).astype("<f2").view("<u2").tolist()
    sum_squares = sum(units(b)**2 for b in hidden)
    variance = Fraction(sum_squares, len(hidden) * 2**48) + Fraction(1e-6)
    rows = []
    for i in differences(primary, independent):
        numerator = Fraction(units(hidden[i]) * units(weight[i]), 2**48)
        exact = exact_norm_round(numerator, variance)
        require(exact in (primary[i], independent[i]), "neither RMSNorm implementation matches exact oracle")
        rows.append({
            "index": i, "torch_bits": f"{primary[i]:04x}",
            "numpy_bits": f"{independent[i]:04x}", "exact_mathematical_bits": f"{exact:04x}",
            "numerator": str(numerator), "variance": str(variance),
            "torch_matches_exact": primary[i] == exact, "numpy_matches_exact": independent[i] == exact,
        })
    torch_mean = torch.from_numpy(x).pow(2).mean()
    inverse = float(torch.rsqrt(torch_mean + 1e-6))
    return primary, rows, {
        "sum_squares_q48": sum_squares, "mean_binary64_hex": float(torch_mean).hex(),
        "inverse_rms_binary64_hex": inverse.hex(),
    }


def residual_cases(actual_input, reference_input, actual_o, reference_o, actual, reference):
    return {
        "A": actual, "AA": reference_add(actual_input, actual_o),
        "AR": reference_add(actual_input, reference_o),
        "RA": reference_add(reference_input, actual_o),
        "RR": reference_add(reference_input, reference_o), "R": reference,
    }


def project_vectors(actual, reference, tensors):
    """Independent matrix unpack/reduction, guarded against host int64 overflow."""
    require(len(actual) == len(reference) and len(actual) > 0
            and len(actual) % 128 == 0, "projection input geometry")
    n, groups = len(actual), len(actual) // 128
    require(set(tensors) == {"qweight", "qzeros", "scales"}, "unbiased projection contract")
    require(len(tensors["scales"]) % groups == 0, "scale geometry")
    outputs = len(tensors["scales"]) // groups
    require(outputs > 0 and outputs % 8 == 0, "projection output geometry")
    require(len(tensors["qweight"]) == n * outputs // 8
            and len(tensors["qzeros"]) == groups * outputs // 8, "packed geometry")
    for name in ("qweight", "qzeros"):
        require(all(isinstance(x, int) and 0 <= x < 2**32 for x in tensors[name]),
                "invalid packed INT32 word")
    shifts = np.array([0, 16, 4, 20, 8, 24, 12, 28], dtype=np.uint32)
    qw = np.asarray(tensors["qweight"], dtype=np.uint32).reshape(n, outputs // 8)
    qz = np.asarray(tensors["qzeros"], dtype=np.uint32).reshape(groups, outputs // 8)
    w = ((qw[:, :, None] >> shifts) & 15).reshape(n, outputs).astype(np.int64)
    z = ((qz[:, :, None] >> shifts) & 15).reshape(groups, outputs).astype(np.int64)
    scales = np.array([units(x) for x in tensors["scales"]], dtype=np.int64)
    coefficients = (w - np.repeat(z, 128, axis=0)) * np.repeat(
        scales.reshape(groups, outputs), 128, axis=0)
    maximum = max(abs(int(coefficients.min())), abs(int(coefficients.max())))
    totals, bits = {}, {}
    for name, activation in (("A", actual), ("R", reference)):
        x = [units(b) for b in activation]
        require(max(map(abs, x)) * maximum * n < 2**63, "int64 reduction bound exceeded")
        totals[name] = (np.asarray(x, dtype=np.int64) @ coefficients).tolist()
        bits[name] = [round_q48(total) for total in totals[name]]
    return bits, totals


def lifted_cases(stage12, stage17, actual_hidden, reference_hidden):
    """Refine each parent edge, retaining both outer and inner attribution orders."""
    hidden = {"A": actual_hidden, "R": reference_hidden}
    for label, left, right in (("AA", "A", "A"), ("AR", "A", "R"),
                               ("RA", "R", "A"), ("RR", "R", "R")):
        hidden[label] = reference_add(stage12[left], stage17[right])
    chains = {}
    for outer, transitions in ORDERS.items():
        for inner, inner_transitions in INNER_ORDERS.items():
            edges = []
            for part, start, end in transitions:
                if part.startswith("stage12_"):
                    fixed = "A" if start == "AA" else "R"
                    mapping = {"A": start, "R": end}
                    for label in ("AA", "AR", "RA", "RR"):
                        key = f"{outer}/{inner}/stage12/{label}"
                        hidden[key] = reference_add(stage12[label], stage17[fixed])
                        mapping[label] = key
                    edges.extend((part + "/" + name, mapping[a], mapping[b])
                                 for name, a, b in inner_transitions)
                elif part.startswith("stage17_"):
                    fixed = "A" if start == "AA" else "R"
                    mapping = {"A": start, "R": end}
                    for label in ("DA", "DR"):
                        key = f"{outer}/{inner}/stage17/{label}"
                        hidden[key] = reference_add(stage12[fixed], stage17[label])
                        mapping[label] = key
                    edges.extend((part + "/" + name, mapping[a], mapping[b])
                                 for name, a, b in DOWN_CHAIN)
                else:
                    edges.append((part, start, end))
            require(edges[0][1] == "A" and edges[-1][2] == "R"
                    and all(x[2] == y[1] for x, y in zip(edges, edges[1:])),
                    "disconnected attribution chain")
            chains[outer + "/" + inner] = edges
    return hidden, chains


def components(cases, chains, index):
    result = {}
    for order, edges in chains.items():
        parts = {name: units(cases[a][index]) - units(cases[b][index])
                 for name, a, b in edges}
        require(sum(parts.values()) == units(cases["A"][index]) - units(cases["R"][index]),
                "ordered component closure")
        result[order] = parts
    return result


def denominator_accounting(hidden, norms, reductions, weight, chains, index):
    result = {}
    w = Fraction(units(weight[index]), 2**24)
    for order, edges in chains.items():
        result[order] = {}
        for name, a, b in edges:
            ha, hb = (Fraction(units(hidden[k][index]), 2**24) for k in (a, b))
            ia, ib = (Fraction(float.fromhex(reductions[k]["inverse_rms_binary64_hex"]))
                      for k in (a, b))
            numerator = (ha - hb) * w * ia
            denominator = hb * w * (ia - ib)
            rounded = Fraction(units(norms[a][index]) - units(norms[b][index]), 2**24)
            result[order][name] = {
                "numerator_change": str(numerator), "denominator_change": str(denominator),
                "rounding_and_binary64_operation_remainder": str(rounded - numerator - denominator),
                "rounded_total": str(rounded),
                "sum_squares_delta_q48": reductions[a]["sum_squares_q48"]
                - reductions[b]["sum_squares_q48"],
            }
    return result


def tensors_for(inputs, prepared, layer, names):
    history = inputs.load(prepared["independent_parent"]["path"])
    require(history["layer"] == layer and history["history"] == HISTORY[:3],
            "independent tensor history identity")
    result = {name: {} for name in names}
    norm = None
    for record in prepared["vectors"]["tensors"]:
        meta = record["checkpoint_tensor"]
        name = meta["name"]
        kind = next((k for k in names if f".{k}_proj." in name), None)
        if kind is None and name != f"model.layers.{layer}.input_layernorm.weight":
            continue
        data = hex_rows(inputs.read(record["serialized"]["path"]))
        raw = b"".join(x.to_bytes(2 if meta["dtype"] == "F16" else 4, "little") for x in data)
        require(len(raw) == meta["bytes"] and hashlib.sha256(raw).hexdigest() == meta["sha256"]
                and any(x["name"] == name and x["sha256"] == meta["sha256"]
                        for x in history["checkpoint_tensor_hashes"]),
                "serialized official tensor mismatch")
        if kind is None:
            norm = data
        else:
            result[kind][name.rsplit(".", 1)[1]] = data
    return result, norm


def run(out):
    require(out.parent == DESTINATION.resolve()
            and out.name.startswith("lineage_a_layer06_stage12_stage17_"), "output scope")
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    inputs = Inputs()
    roots = {}
    for name, digest in ROOT_HASHES.items():
        roots[name] = inputs.load(PARENT / name, root=True)
        require(inputs.records[str(PARENT / name)]["sha256"] == digest,
                f"reviewed parent changed: {name}")
    freeze, parent = roots["frozen.json"], roots["measurements.json"]
    review = inputs.load(REVIEW, root=True)
    require(review["kind"] == "round_reviewed_handoff" and review["producer_role"] == "reviewer"
            and review["mission_id"] == "22f4b93c3e45" and review["review"]["status"] == "done",
            "missing genuine predecessor review")
    require(freeze["token_history"] == HISTORY and freeze["comparison_policy"] == POLICY,
            "foreign lineage or policy")
    inputs.load(FAILED_ATTEMPT / "frozen.json", root=True)
    for name in ("unit_tests.log", "public_contract_compile_0.log", "public_contract_compile_1.log",
                 "ace3_fp16_residual_add_core.vvp", "ace3_fp16_rmsnorm_core.vvp"):
        inputs.read(FAILED_ATTEMPT / name, root=True)
    inputs.read(FAILED_ATTEMPT.with_suffix(".command.log"), root=True)
    inputs.load(PREFIX / "seal.json")
    prepared = {layer: inputs.load(PREFIX / f"layer{layer:02}/prepared.json") for layer in (6, 7)}
    results = {layer: inputs.load(PREFIX / f"layer{layer:02}/result.json") for layer in (5, 6)}
    p6, p7 = prepared[6], prepared[7]
    for layer, p in prepared.items():
        require(p["vectors"]["layer_index"] == layer and p["vectors"]["position"] == 3
                and p["binary"] == p["kv_parent"]["live_binary"],
                "retained layer/position/binary ABI mismatch")
        inputs.read(p["binary"]["path"])
        inputs.read(p["kv_parent"]["state"]["path"])
    require(results[5]["output_hidden"] == p6["input_hidden"]
            and results[6]["output_hidden"] == p7["input_hidden"], "actual hidden parent mismatch")
    actual_input = framed_hidden(inputs.read(p6["input_hidden"]["path"]))
    require(actual_input == framed_hidden(inputs.read(p6["vectors"]["input"]["path"])),
            "canonical simulator input identity")
    reference_input = hex_rows(inputs.read(results[5]["independent_hidden"]["path"]))
    actual = decode_trace(inputs.read(
        Path(results[6]["output_hidden"]["path"]).with_name("trace.hex")), 3)
    reference = {s: hex_rows(inputs.read(p6["independent_stages"][str(s)]["path"]))
                 for s in (10, 11, 12, 16, 17, 18)}
    require(actual[18] == framed_hidden(inputs.read(results[6]["output_hidden"]["path"]))
            == framed_hidden(inputs.read(p7["vectors"]["input"]["path"])), "trace/final identity")
    require(all(len(actual[s]) == len(reference[s]) == (4864 if s == 16 else 896)
                for s in reference) and len(actual_input) == len(reference_input) == 896,
            "public producer geometry")
    old = parent["hidden_coordinates"]["3"]
    for stage in (12, 17):
        require([f"{b:04x}" for b in actual[stage]] == [r[f"actual_stage{stage}"] for r in old]
                and [f"{b:04x}" for b in reference[stage]]
                == [r[f"reference_stage{stage}"] for r in old], "parent stage identity")
    require([f"{b:04x}" for b in actual[18]] == [r["hidden_bits"]["A"] for r in old]
            and [f"{b:04x}" for b in reference[18]] == [r["hidden_bits"]["R"] for r in old],
            "parent final identity")
    tensors, _ = tensors_for(inputs, p6, 6, ("o", "down"))
    qk_tensors, weight = tensors_for(inputs, p7, 7, ("q", "k"))
    require(weight is not None and len(weight) == 896, "RMSNorm public geometry")
    qk_path = next(Path(x["path"]) for x in freeze["authenticated_inputs"]
                   if x["path"].endswith("qk_bd2d1b4dffa3_attempt002/measurements.json"))
    qk = inputs.load(qk_path)
    producers = {(x["kind"], x["position"], x["channel"]): x for x in qk["producers"]}
    priority_inputs = sorted({483, 494, 694} | set(
        roots["result.json"]["priority_q410_q442_input_indices"]))
    selected_scores = [next(x for x in qk["scores"] if (x["head"], x["key_position"]) == key)
                       for key in PRIORITIES]
    paths = {Path(module.__file__).resolve() for module in tuple(sys.modules.values())
             if getattr(module, "__file__", None)
             and Path(module.__file__).resolve().is_relative_to(ROOT / "ace3/model")
             and Path(module.__file__).suffix == ".py"}
    paths.update((Path(__file__).resolve(), ROOT / "ace3/model/tests" /
                  "test_trace_lineage_a_layer06_stage12_stage17_producers.py"))
    paths.update(ROOT / "ace3/rtl" / name for name in (
        "ace3_fp16_fixed.sv", "ace3_fp16_residual_add_core.sv",
        "ace3_fp16_rmsnorm_core.sv", "ace3_awq_w4a16_projection_engine.sv",
        "ace3_awq_w4a16_g128_dot_lane.sv", "ace3_q47_48_to_f16_rne.sv",
        "ace3_decoder_layer0_token_engine.sv"))
    previous_sources = {r["path"]: r for r in freeze["sources"]}
    sources = []
    for path in sorted(paths):
        record = snapshot(out, path)
        if str(path) in previous_sources:
            require(record["sha256"] == previous_sources[str(path)]["sha256"],
                    f"reviewed source changed: {path}")
        sources.append(record)
    tops = ("ace3_fp16_residual_add_core", "ace3_fp16_rmsnorm_core")
    compiles = []
    tests = [sys.executable, "-B", "-m", "unittest", "discover", "-s",
             str(ROOT / "ace3/model/tests"), "-p",
             "test_trace_lineage_a_layer06_stage12_stage17_producers.py",
             "-k", "NormBoundaryTests"]
    command(out, "iverilog_version", ["iverilog", "-V"])
    write(out / "frozen.json", {
        "scope": "retained A-lineage position3 producer accounting, no RTL execution or injection",
        "parent_root_hashes": ROOT_HASHES, "token_history": HISTORY, "comparison_policy": POLICY,
        "authenticated_inputs": list(inputs.records.values()), "sources": sources,
        "commands": [*compiles, tests], "python": sys.version, "numpy": np.__version__,
        "torch": torch.__version__, "outer_orders": ORDERS, "inner_orders": INNER_ORDERS,
        "stage17_chain": DOWN_CHAIN, "priority_inputs": priority_inputs,
        "priority_scores": PRIORITIES, "score_denominator": 2**51,
        "epsilon_binary64_hex": float(1e-6).hex(),
        "public_declarations": {
            path.name: (out / path.name).read_text().split(");", 1)[0] + ");"
            for path in paths if path.suffix == ".sv" and path.name != "ace3_fp16_fixed.sv"},
        "position2": "parent decomposition retained unchanged and conditional; not resplit",
        "positions01": "parent historical K/V remain unsplit",
        "score_scope": "refines parent Q main effect with reference K; Q/K interaction retained in parent",
        "measurement_kind": "software causal substitutions with all 896 RMSNorm denominator coordinates",
        "previous_attempt": str(FAILED_ATTEMPT),
        "previous_failure_taxonomy": "independent_oracle_rounding_disagreement",
        "diagnostic_change": "Retain Torch recurrence and explicitly record every NumPy disagreement; "
                             "use exact squared-midpoint RNE as a third mathematical oracle. "
                             "No reference, production arithmetic or acceptance policy change.",
        "reused_public_contract_compiles": list(tops),
    })
    for i, argv in enumerate(compiles):
        command(out, f"public_contract_compile_{i}", argv)
    command(out, "unit_tests", tests)
    phase_times = {"authentication_freeze_compile_tests": time.monotonic() - started}
    phase = time.monotonic()
    projection_bits, projection_totals, projection_rows = {}, {}, {}
    for name, source_stage, target_stage in (("o", 10, 11), ("down", 16, 17)):
        bits, totals = project_vectors(actual[source_stage], reference[source_stage], tensors[name])
        projection_bits[name], projection_totals[name] = bits, totals
        rows = []
        for channel in priority_inputs:
            measured = native_projection_pair(
                actual[source_stage], reference[source_stage], tensors[name], channel)
            oracle = oracle_pair(actual[source_stage], reference[source_stage], tensors[name], channel)
            for label, key, result in zip(("A", "R"), ("actual", "reference"), oracle, strict=True):
                total, output, invalid, overflow, _ = result
                require(not (invalid or overflow) and total == totals[label][channel]
                        == measured[f"{key}_q48"] and output == bits[label][channel]
                        == measured[f"{key}_bits"], "independent projection reconstruction disagreement")
            parts = {
                "local_projection": (units(actual[target_stage][channel])
                                     - units(bits["A"][channel])) << 24,
                "input_activation": totals["A"][channel] - totals["R"][channel],
                "projection_rounding_difference": (
                    (units(bits["A"][channel]) - units(bits["R"][channel])) << 24)
                    - totals["A"][channel] + totals["R"][channel],
                "reference_recurrence": (units(bits["R"][channel])
                                         - units(reference[target_stage][channel])) << 24,
            }
            require(sum(parts.values()) == (
                units(actual[target_stage][channel]) - units(reference[target_stage][channel])) << 24,
                "projection producer closure")
            terms = measured["terms"]
            for term in terms:
                term[f"actual_stage{source_stage}_bits"] = term.pop("actual_stage13_bits")
                term[f"reference_stage{source_stage}_bits"] = term.pop("reference_stage13_bits")
            ranked = sorted(terms, key=lambda t: (-abs(t["delta_q48"]), t["index"]))
            rows.append({
                "channel": channel, "source_stage": source_stage, "target_stage": target_stage,
                "actual_bits": f"{actual[target_stage][channel]:04x}",
                "reference_bits": f"{reference[target_stage][channel]:04x}",
                "same_input_bits": {k: f"{v[channel]:04x}" for k, v in bits.items()},
                "components_q48": parts, "top_input_terms": ranked[:24],
                "all_input_deltas_q48": [t["delta_q48"] for t in terms],
                "group_actual_q48": measured["actual_groups_q48"],
                "group_reference_q48": measured["reference_groups_q48"],
            })
        projection_rows[name] = rows
    phase_times["producer_projections"] = time.monotonic() - phase
    write(out / "projection_measurements.json", {
        "projection_producers": projection_rows, "projection_totals_q48": projection_totals,
        "projection_bits": projection_bits, "phase_seconds": phase_times,
    })
    phase = time.monotonic()
    stage12 = residual_cases(actual_input, reference_input, actual[11], reference[11],
                             actual[12], reference[12])
    stage17 = {"A": actual[17], "DA": projection_bits["down"]["A"],
               "DR": projection_bits["down"]["R"], "R": reference[17]}
    hidden, chains = lifted_cases(stage12, stage17, actual[18], reference[18])
    norms, reductions, norm_disagreements = {}, {}, {}
    for name, h in hidden.items():
        norms[name], norm_disagreements[name], reductions[name] = norm_boundary_measurement(h, weight)
    for label in ("A", "AA", "AR", "RA", "RR", "R"):
        require(norms[label] == [int(r["norm_bits"][label], 16) for r in old],
                "reviewed full-denominator endpoint changed")
    coordinates = []
    for i in range(896):
        row = {
            "index": i, "actual_input": f"{actual_input[i]:04x}",
            "reference_input": f"{reference_input[i]:04x}",
            "actual_O": f"{actual[11][i]:04x}", "reference_O": f"{reference[11][i]:04x}",
            "stage12_bits": {k: f"{v[i]:04x}" for k, v in stage12.items()},
            "stage12_components_q24": components(stage12, INNER_ORDERS, i),
            "stage17_bits": {k: f"{v[i]:04x}" for k, v in stage17.items()},
            "stage17_components_q24": components(stage17, {"producer_chain": DOWN_CHAIN}, i),
            "hidden_bits": {k: f"{v[i]:04x}" for k, v in hidden.items()},
            "norm_bits": {k: f"{v[i]:04x}" for k, v in norms.items()},
            "norm_components_q24": components(norms, chains, i),
        }
        if i in priority_inputs:
            row["denominator_accounting"] = denominator_accounting(
                hidden, norms, reductions, weight, chains, i)
        coordinates.append(row)
    selected_channels = set()
    for score in selected_scores:
        for row in score["all_dimensions"]:
            for kind in ("q", "k"):
                position = 3 if kind == "q" else score["key_position"]
                if position == 3:
                    ch = row[f"{kind}_channel"]
                    selected_channels.add((kind, ch // 64, ch % 32))
    ranked_parent = {(r["kind"], r["position"], r["channel"]): r
                     for r in parent["ranked_projections"]}
    post, ranked_projections = {}, []
    for kind, head, dim in sorted(selected_channels):
        pair = (head * 64 + dim, head * 64 + dim + 32)
        raw = {}
        for channel in pair:
            raw[channel], coeffs, totals = project_cases(norms, qk_tensors[kind], channel)
            key = (kind, 3, channel)
            if key in ranked_parent:
                previous = ranked_parent[key]
                require(all(totals[k] == previous["dot_totals_q48"][k]
                            for k in ("A", "AA", "AR", "RA", "RR", "R")),
                        "reviewed weighted projection endpoint changed")
                selected_inputs = sorted(set(priority_inputs) |
                                         {r["input_index"] for r in previous["ranked_inputs"]})
                ranked_projections.append({
                    "kind": kind, "position": 3, "channel": channel,
                    "raw_bits": {k: f"{v:04x}" for k, v in raw[channel].items()},
                    "dot_totals_q48": totals,
                    "dot_components_q48": {
                        order: {name: totals[a] - totals[b] for name, a, b in edges}
                        for order, edges in chains.items()},
                    "weighted_inputs": [{
                        "input_index": i, "coefficient_q24": coeffs[i],
                        "components_q48": {order: {name: value * coeffs[i]
                                                  for name, value in parts.items()}
                                           for order, parts in components(norms, chains, i).items()},
                    } for i in selected_inputs],
                })
        producer = producers[kind, 3, pair[0]]
        coeff = [int(x, 16) for x in producer["cos_sin_bits"]]
        rotated = {name: rotations([raw[ch][name] for ch in pair], coeff, 3, dim)[2]
                   for name in norms}
        for side, channel in enumerate(pair):
            post[kind, channel] = {name: bits[side] for name, bits in rotated.items()}
    score_rows = []
    for score, previous in zip(selected_scores, parent["scores"], strict=True):
        require((score["head"], score["key_position"]) ==
                (previous["head"], previous["key_position"]), "score ordering")
        totals = {order: {kind: {name: 0 for name, _, _ in edges} for kind in ("q", "k")}
                  for order, edges in chains.items()}
        for row in score["all_dimensions"]:
            for kind in ("q", "k"):
                position = 3 if kind == "q" else score["key_position"]
                if position != 3:
                    continue
                p = post[kind, row[f"{kind}_channel"]]
                other = (("k", score["key_position"], row["k_channel"]) if kind == "q"
                         else ("q", 3, row["q_channel"]))
                coefficient = units(int(producers[other]["reference"], 16))
                for order, edges in chains.items():
                    for name, a, b in edges:
                        totals[order][kind][name] += (units(p[a]) - units(p[b])) * coefficient
        for order, kinds in totals.items():
            outer = order.split("/", 1)[0]
            for kind, parts in kinds.items():
                if kind == "k" and score["key_position"] != 3:
                    require(sum(parts.values()) == 0, "historical K accidentally resplit")
                    continue
                for name, _, _ in ORDERS[outer]:
                    refined = sum(v for k, v in parts.items() if k == name or k.startswith(name + "/"))
                    require(refined == previous["hidden_split_dot_q48"][outer][kind][name],
                            "six-score parent edge closure")
        score_rows.append({
            "head": score["head"], "key_position": score["key_position"],
            "position3_producer_parts_dot_q48": totals,
            "position3_producer_parts_scaled": {
                order: {kind: {name: str(Fraction(v, 2**51)) for name, v in parts.items()}
                        for kind, parts in kinds.items()} for order, kinds in totals.items()},
            "retained_parent_score": previous,
            "historical_K": "retained unsplit here; position2 parent attribution remains conditional",
        })
    require({410, 442} <= {r["channel"] for r in ranked_projections if r["kind"] == "q"},
            "priority Q channels missing")
    phase_times["full_denominator_and_score_accounting"] = time.monotonic() - phase
    discrepancies = {
        "local_stage12": differences(stage12["A"], stage12["AA"]),
        "reference_stage12": differences(stage12["RR"], stage12["R"]),
        "local_O": differences(actual[11], projection_bits["o"]["A"]),
        "reference_O": differences(projection_bits["o"]["R"], reference[11]),
        "local_down": differences(actual[17], projection_bits["down"]["A"]),
        "reference_down": differences(projection_bits["down"]["R"], reference[17]),
        "incoming_drift": differences(actual_input, reference_input),
        "stage10_drift": differences(actual[10], reference[10]),
        "stage16_drift": differences(actual[16], reference[16]),
    }
    for record in sources:
        require(hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() == record["sha256"],
                "source changed during measurement")
    write(out / "measurements.json", {
        "coordinates": coordinates, "projection_producers": projection_rows,
        "projection_totals_q48": projection_totals, "projection_bits": projection_bits,
        "reductions": reductions, "chains": chains, "ranked_projections": ranked_projections,
        "scores": score_rows, "discrepancies": discrepancies, "phase_seconds": phase_times,
        "preserved_parent_discrepancies": parent["discrepancies"],
        "historical_v_unsplit": parent["historical_v_unsplit"],
        "counterfactual_norm_oracle_disagreements": norm_disagreements,
    })
    local = any(discrepancies[k] for k in ("local_stage12", "local_O", "local_down"))
    summary = {
        "status": "LAYER06_PRODUCER_BOUNDARY_LOCALIZED_NOT_REPAIRED",
        "counts": {k: len(v) for k, v in discrepancies.items()}, "position": 3,
        "coordinates": 896, "priority_inputs": priority_inputs, "score_rows": len(score_rows),
        "ranked_projection_channels": len(ranked_projections), "attribution_orders": len(chains),
        "general_defect_proven": False, "repair": None, "RTL_simulations": 0,
        "contract_compiles": len(compiles), "binary64_evaluated": False,
        "reused_public_contract_compiles": 2,
        "counterfactual_norm_oracle_disagreements": sum(map(len, norm_disagreements.values())),
        "counterfactual_torch_vs_exact_differences": sum(
            not row["torch_matches_exact"] for rows in norm_disagreements.values() for row in rows),
        "priority_producer_inputs": {
            name: {str(row["channel"]): [t["index"] for t in row["top_input_terms"][:8]]
                   for row in rows if row["channel"] in (483, 494, 694)}
            for name, rows in projection_rows.items()},
        "priority_hidden_effects_q24": {
            str(i): {"stage12": coordinates[i]["stage12_components_q24"],
                     "stage17": coordinates[i]["stage17_components_q24"]}
            for i in (483, 494, 694)},
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "hypothesis": "Inherited incoming hidden, stage10 AV and stage16 activation differences "
                      "propagate through exact packed projections, residual RNE and full RMSNorm coupling.",
        "next_boundary": "Layer06 position3 stage16 = SiLU(stage14 gate) times stage15 up, "
                         "including stage13 post-attention RMSNorm denominator; in parallel causally "
                         "account for stage10 attention-value and layer05 stage18 incoming hidden. "
                         "Use retained priority output483/494/694 and complete Q410/Q442 weighted "
                         "terms in both orders; no unique producer or repaired trajectory is established.",
        "regression": "Native packed-lane scalar oracle versus bounded int64 matrix projection, "
                      "residual ties, both nested orders, full denominator and six original score edge closures.",
        "review_status": "PENDING_NORMAL_HOST_REVIEWER",
        "limits": ["position2 attribution remains conditional", "positions0/1 K/V remain unsplit",
                   "no changed arithmetic, RTL replay, gate waiver, binary64 admission or third token"],
    }
    if local:
        summary["status"] = "LOCAL_PRODUCER_RECONSTRUCTION_DISCREPANCY_NOT_REPAIRED"
        summary["next_boundary"] = (
            "Resolve listed local stage12/O/down reconstruction discrepancies against the retained "
            "binary-bound arithmetic before interpreting upstream input effects as a repair.")
    summary["diagnostic_oracle_boundary"] = (
        "Original attempt001 failed the strict Torch/NumPy norm cross-check. This diagnostic "
        "retains both differing encodings and the exact mathematical rounding classification. "
        "Only the original Torch recurrence is used for additive attribution; all six parent "
        "endpoints are unchanged. Intermediate library disagreement is not an RTL defect "
        "or permission to change the independently propagated reference.")
    write(out / "result.json", summary)
    for path in out.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out.resolve())
