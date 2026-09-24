#!/usr/bin/env python3
"""Refine retained lineage-A activation effects; no software arrays enter RTL."""

from __future__ import annotations

import argparse
from decimal import localcontext
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from diagnose_lineage_a_position3_boundary import (
    HISTORY, POLICY, PREFIX, Inputs, decode_trace, require, units, write,
)
from fp16_adaptation_oracle import rmsnorm, silu_gate_exp
from trace_lineage_a_layer06_stage12_stage17_producers import (
    INNER_ORDERS, components, norm_boundary_measurement, project_vectors, tensors_for,
)
from trace_lineage_a_layer06_stage18_hidden import project_cases, reference_add, values
from trace_lineage_a_layer07_downproj import framed_hidden, hex_rows, round_q48
from trace_lineage_a_layer07_gate_up import native_projection_pair, oracle_pair, snapshot
from trace_lineage_a_layer07_qk_producers import rotations
from trace_lineage_a_layer07_score_softmax_v import DESTINATION, ROOT
from trace_lineage_a_layer07_silu import (
    accurate_silu, mathematical_silu, nearest_decimal, rne_ratio, source_records,
)
from trace_lineage_a_layer07_stage00_inputs import PRIORITIES, command, differences


PARENT = DESTINATION / "lineage_a_layer06_stage12_stage17_1bf9d4ba6988_attempt003"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
              "1bf9d4ba6988/round-0001.json")
ROOT_HASHES = {
    "frozen.json": "896033d438e1e93244208c6845d8e394e6907b73316dbfdedded2de78b85f672",
    "measurements.json": "32c83bf13d4628b49a6c419c19a804f7e34c1ba10f96c2d8d77811b27840d939",
    "result.json": "1f421d84e7f7ac2675a22d3074838a607cfb5d64d9b5610b255df2c09d1baf9b",
}
PRIORITY_CHANNELS = (2321, 2990, 3288, 3856, 4653)


def integer_norm(hidden, weight):
    """Separate scalar reconstruction of the Q24/isqrt/epsilon RTL recurrence."""
    require(len(hidden) == len(weight) and len(hidden) > 0, "norm geometry")
    mean = rne_ratio(sum(units(x)**2 for x in hidden) + 281474977 * len(hidden),
                     len(hidden))
    root = math.isqrt(mean)
    require(root > 0, "zero norm denominator")
    result = []
    for x, w in zip(hidden, weight, strict=True):
        q24 = rne_ratio(units(x) * units(w), root)
        bits = round_q48(q24 << 24)
        if bits & 0x7fff == 0:
            bits = (x ^ w) & 0x8000
        result.append(bits)
    oracle, oracle_mean, oracle_root = rmsnorm(hidden, weight)
    require(mean == oracle_mean and root == oracle_root
            and all(not (invalid or overflow) and b == reconstructed
                    for (b, invalid, overflow), reconstructed in zip(oracle, result, strict=True)),
            "independent integer norm disagreement")
    return result


def true_silu(gate, up):
    require(len(gate) == len(up), "SiLU geometry")
    g, u = values(gate), values(up)
    primary = (torch.from_numpy(g) * torch.sigmoid(torch.from_numpy(g))
               * torch.from_numpy(u)).numpy().astype("<f2").view("<u2").tolist()
    exponential = np.exp(-np.abs(g))
    sigmoid = np.where(g < 0, exponential, 1) / (1 + exponential)
    secondary = (g * sigmoid * u).astype("<f2").view("<u2").tolist()
    disagreements = []
    with localcontext() as context:
        context.prec = 90
        for i in differences(primary, secondary):
            exact = nearest_decimal(mathematical_silu(gate[i], up[i]),
                                    (gate[i] ^ up[i]) & 0x8000)
            disagreements.append({
                "index": i, "torch_bits": primary[i], "numpy_bits": secondary[i],
                "decimal90_bits": exact,
            })
    for b in primary + secondary:
        units(b)
    return primary, disagreements


def projection_chain(norm_edges):
    return [("local_projection", "A", "P:A"),
            *((name, "P:" + a, "P:" + b) for name, a, b in norm_edges),
            ("reference_projection_recurrence", "P:R", "R")]


def norm_producer_chains():
    return {
        order: [("local_norm", "A", "own"), ("norm_arithmetic", "own", "n:A"),
                *((name, "n:" + left, "n:" + right) for name, left, right in edges),
                ("reference_norm_recurrence", "n:R", "R")]
        for order, edges in INNER_ORDERS.items()}


def activation_chain(projection_edges, gate_first):
    edges = [("local_silu", "A", "own"), ("silu_arithmetic", "own", "G:A/U:A")]
    for kind, fixed in (("gate", "A"), ("up", "R")) if gate_first else (
            ("up", "A"), ("gate", "R")):
        def key(value):
            return f"G:{value}/U:{fixed}" if kind == "gate" else f"G:{fixed}/U:{value}"
        edges.extend((kind + "/" + name, key(a), key(b))
                     for name, a, b in projection_edges)
    edges.append(("reference_silu_recurrence", "G:R/U:R", "R"))
    require(all(a[2] == b[1] for a, b in zip(edges, edges[1:])),
            "disconnected activation attribution")
    return edges


def qk_cases(norms, tensors, channels):
    """Batch exact bounded integer dots; retain independent scalar priority checks."""
    names = list(norms)
    x = np.asarray([[units(b) for b in norms[name]] for name in names], dtype=np.int64)
    coefficients = []
    for channel in channels:
        pair = native_projection_pair(norms[names[0]], norms[names[-1]], tensors, channel)
        coefficients.append([(term["weight"] - term["zero"])
                             * units(int(term["scale_bits"], 16)) for term in pair["terms"]])
    c = np.asarray(coefficients, dtype=np.int64).T
    require(int(np.max(np.abs(x))) * int(np.max(np.abs(c))) * x.shape[1] < 2**63,
            "Q/K int64 dot bound")
    products = x @ c
    result, totals = {}, {}
    for j, channel in enumerate(channels):
        totals[channel] = {name: int(products[i, j]) for i, name in enumerate(names)}
        result[channel] = {}
        for name, total in totals[channel].items():
            biased = total + (units(tensors["bias"][channel]) << 24)
            b = round_q48(biased)
            result[channel][name] = 0x8000 if biased < 0 and b == 0 else b
        if channel in (410, 442):
            scalar_norms = {"A": norms[names[0]], "R": norms[names[-1]]}
            scalar, _, scalar_totals = project_cases(scalar_norms, tensors, channel)
            require(all(scalar[k] == result[channel][name]
                        and scalar_totals[k] == totals[channel][name]
                        for k, name in (("A", names[0]), ("R", names[-1]))),
                    "independent priority Q projection disagreement")
    return result, totals


def run(out):
    require(out.parent == DESTINATION.resolve()
            and out.name.startswith("lineage_a_layer06_stage16_"), "output scope")
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
    require(review["producer_role"] == "reviewer" and review["review"]["status"] == "done"
            and review["mission_id"] == "1bf9d4ba6988"
            and review["kind"] == "round_reviewed_handoff", "missing independent parent review")
    require(freeze["token_history"] == HISTORY and freeze["comparison_policy"] == POLICY,
            "foreign lineage or policy")
    inputs.load(PREFIX / "seal.json")
    prefix = inputs.load(PREFIX / "frozen.json")
    original = inputs.load(prefix["original_frozen"]["path"])
    policy = inputs.load(original["independent_policy"]["path"])
    require(policy["comparison"] == original["comparison_policy"] == POLICY, "changed gate")
    prepared = {l: inputs.load(PREFIX / f"layer{l:02}/prepared.json") for l in (6, 7)}
    results = {l: inputs.load(PREFIX / f"layer{l:02}/result.json") for l in (5, 6)}
    p6, p7 = prepared[6], prepared[7]
    for layer, p in prepared.items():
        require(p["vectors"]["layer_index"] == layer and p["vectors"]["position"] == 3
                and p["binary"] == p["kv_parent"]["live_binary"]
                and p["kv_parent"]["valid_positions"] == [0, 1, 2],
                "layer/position/binary/KV contract")
        for record in (p["binary"], p["kv_parent"]["state"], p["input_hidden"],
                       *p["kv_parent"]["actual_kv_traces"]):
            inputs.read(record["path"])
    require(results[5]["output_hidden"] == p6["input_hidden"]
            and results[6]["output_hidden"] == p7["input_hidden"], "hidden lineage")
    a = decode_trace(inputs.read(Path(results[6]["output_hidden"]["path"])
                                .with_name("trace.hex")), 3)
    r = {s: hex_rows(inputs.read(p6["independent_stages"][str(s)]["path"]))
         for s in range(10, 19)}
    require(all(len(a[s]) == len(r[s]) == (4864 if s in (14, 15, 16) else 896)
                for s in r), "stage geometry")
    incoming = framed_hidden(inputs.read(p6["vectors"]["input"]["path"]))
    require(incoming == framed_hidden(inputs.read(p6["input_hidden"]["path"]))
            and a[18] == framed_hidden(inputs.read(p7["vectors"]["input"]["path"])),
            "authenticated canonical simulator input mismatch")
    incoming_reference = hex_rows(inputs.read(results[5]["independent_hidden"]["path"]))
    tensors, _ = tensors_for(inputs, p6, 6, ("gate", "up", "down"))
    qk_tensors, norm7_weight = tensors_for(inputs, p7, 7, ("q", "k"))
    history = inputs.load(p6["independent_parent"]["path"])
    norm_record = next(rec for rec in p6["vectors"]["tensors"] if
                       rec["checkpoint_tensor"]["name"] ==
                       "model.layers.6.post_attention_layernorm.weight")
    meta = norm_record["checkpoint_tensor"]
    weight = hex_rows(inputs.read(norm_record["serialized"]["path"]))
    raw = b"".join(b.to_bytes(2, "little") for b in weight)
    require(len(weight) == 896 and len(raw) == meta["bytes"] and meta["dtype"] == "F16"
            and hashlib.sha256(raw).hexdigest() == meta["sha256"]
            and any(rec["name"] == meta["name"] and rec["sha256"] == meta["sha256"]
                    for rec in history["checkpoint_tensor_hashes"]), "official post-norm tensor")
    qk_path = next(rec["path"] for rec in freeze["authenticated_inputs"]
                   if rec["path"].endswith("qk_bd2d1b4dffa3_attempt002/measurements.json"))
    qk = inputs.load(qk_path)
    producers = {(x["kind"], x["position"], x["channel"]): x for x in qk["producers"]}
    selected_scores = [next(s for s in qk["scores"] if
                            (s["head"], s["key_position"]) == key) for key in PRIORITIES]
    selected = sorted(set(PRIORITY_CHANNELS) | {
        t["index"] for row in parent["projection_producers"]["down"]
        for t in row["top_input_terms"]})
    paths = {Path(module.__file__).resolve() for module in tuple(sys.modules.values())
             if getattr(module, "__file__", None)
             and Path(module.__file__).resolve().is_relative_to(ROOT / "ace3/model")
             and Path(module.__file__).suffix == ".py"}
    paths.update((Path(__file__).resolve(), ROOT / "ace3/model/tests" /
                  "test_trace_lineage_a_layer06_stage16_activation.py"))
    rtl_names = ("ace3_fp16_fixed.sv", "ace3_fp16_silu_gate_core.sv",
                 "ace3_fp16_rmsnorm_core.sv", "ace3_awq_w4a16_g128_dot_lane.sv",
                 "ace3_q47_48_to_f16_rne.sv", "ace3_awq_w4a16_projection_engine.sv",
                 "ace3_decoder_layer0_token_engine.sv")
    paths.update(ROOT / "ace3/rtl" / name for name in rtl_names)
    previous_sources = {rec["path"]: rec for rec in freeze["sources"]}
    original_sources = {rec["path"]: rec for rec in source_records(original)}
    sources = []
    for path in sorted(paths):
        record = snapshot(out, path)
        expected = previous_sources.get(str(path))
        if path.suffix == ".sv":
            expected = original_sources.get(str(path))
            require(expected is not None, f"missing binary-bound source: {path}")
        if expected:
            require(expected["sha256"] == record["sha256"], f"source changed: {path}")
        sources.append(record)
    compiles = []
    for top, parameters, dependencies in (
            ("ace3_fp16_silu_gate_core", {"INTERMEDIATE_SIZE": 4864, "ACCURATE_SIGMOID": 1},
             ("ace3_fp16_fixed.sv", "ace3_fp16_silu_gate_core.sv")),
            ("ace3_fp16_rmsnorm_core", {"HIDDEN_SIZE": 896},
             ("ace3_fp16_fixed.sv", "ace3_fp16_rmsnorm_core.sv")),
            ("ace3_awq_w4a16_projection_engine",
             {"IN_FEATURES": 896, "OUT_FEATURES": 4864, "BIAS_ENABLE": 0,
              "SINGLE_ROUND_BIAS": 0},
             ("ace3_fp16_fixed.sv", "ace3_awq_w4a16_g128_dot_lane.sv",
              "ace3_q47_48_to_f16_rne.sv", "ace3_awq_w4a16_projection_engine.sv"))):
        compiles.append(["iverilog", "-g2012", "-s", top,
                         *(f"-P{top}.{key}={value}" for key, value in parameters.items()),
                         "-o", str(out / (top + ".vvp")),
                         *(str(out / name) for name in dependencies)])
    tests = [sys.executable, "-B", "-m", "unittest", "discover", "-s",
             str(ROOT / "ace3/model/tests"), "-p",
             "test_trace_lineage_a_layer06_stage16_activation.py"]
    command(out, "iverilog_version", ["iverilog", "-V"])
    write(out / "frozen.json", {
        "scope": "retained A position3 software causal accounting, not a repaired RTL attempt",
        "parent": str(PARENT), "parent_hashes": ROOT_HASHES, "token_history": HISTORY,
        "comparison_policy": POLICY, "independent_policy": policy, "sources": sources,
        "authenticated_inputs": list(inputs.records.values()),
        "priority_channels": PRIORITY_CHANNELS, "weighted_channels": selected,
        "priority_residuals": [483, 494, 694], "priority_scores": PRIORITIES,
        "orders": {"upstream": INNER_ORDERS, "activation": ["gate_then_up", "up_then_gate"],
                   "downstream": ["actual_stage12", "reference_stage12"]},
        "public_declarations": {name: (out / name).read_text().split(");", 1)[0] + ");"
                                for name in rtl_names if name != "ace3_fp16_fixed.sv"},
        "commands": [*compiles, tests], "python": sys.version,
        "numpy": np.__version__, "torch": torch.__version__,
        "score_scope": "Refine only the parent's position3 stage16 activation main effect. "
                       "Other parent effects, historical K and Q/K interaction remain separate.",
        "counterfactual_policy": "Torch binary64 recurrence rounded to FP16; NumPy and Decimal "
                                "disagreements are recorded, never substituted into the trajectory.",
    })
    for i, argv in enumerate(compiles):
        command(out, f"public_contract_compile_{i}", argv)
    command(out, "unit_tests", tests)
    phase_times = {"authentication_freeze_compile_tests": time.monotonic() - started}
    phase = time.monotonic()

    stage12 = {name: [int(row["stage12_bits"][name], 16) for row in parent["coordinates"]]
               for name in ("A", "AA", "AR", "RA", "RR", "R")}
    require(stage12["A"] == a[12] and stage12["R"] == r[12]
            and reference_add(incoming, a[11]) == stage12["AA"]
            and reference_add(incoming_reference, r[11]) == stage12["RR"],
            "parent residual endpoint mismatch")
    norms = {"A": a[13], "own": integer_norm(a[12], weight), "R": r[13]}
    norm_disagreements, reductions = {}, {}
    for name, hidden in stage12.items():
        key = "n:" + name
        norms[key], norm_disagreements[key], reductions[key] = norm_boundary_measurement(hidden, weight)
    norm_chains = norm_producer_chains()
    projected, projection_totals, weighted = {}, {}, []
    for kind, stage in (("gate", 14), ("up", 15)):
        projected[kind] = {"A": a[stage], "R": r[stage]}
        projection_totals[kind] = {}
        names = list(norms)
        for offset in range(0, len(names), 2):
            left, right = names[offset], names[min(offset + 1, len(names) - 1)]
            bits, totals = project_vectors(norms[left], norms[right], tensors[kind])
            for label, name in (("A", left), ("R", right)):
                projected[kind]["P:" + name] = bits[label]
                projection_totals[kind][name] = totals[label]
        for channel in selected:
            pair = native_projection_pair(a[13], r[13], tensors[kind], channel)
            oracle = oracle_pair(a[13], r[13], tensors[kind], channel)
            for label, name, check in zip(("actual", "reference"), ("A", "R"), oracle, strict=True):
                require(not (check[2] or check[3]) and check[0] == pair[label + "_q48"]
                        == projection_totals[kind][name][channel]
                        and check[1] == pair[label + "_bits"]
                        == projected[kind]["P:" + name][channel],
                        "independent native-GEMM reconstruction disagreement")
            terms = sorted(pair["terms"], key=lambda t: (-abs(t["delta_q48"]), t["index"]))
            weighted.append({
                "kind": kind, "channel": channel, "actual_q48": pair["actual_q48"],
                "reference_q48": pair["reference_q48"],
                "top_stage13_terms": [{
                    **term, "norm_components_q24": components(norms, norm_chains, term["index"]),
                } for term in terms[:16]],
                "all_stage13_deltas_q48": [t["delta_q48"] for t in pair["terms"]],
                "projection_components_q24": components(
                    projected[kind], {k: projection_chain(v) for k, v in norm_chains.items()}, channel),
            })
    stage16 = {"A": a[16], "R": r[16], "own": []}
    for gate, up in zip(a[14], a[15], strict=True):
        measured, oracle = accurate_silu(gate, up), silu_gate_exp(gate, up)
        require(not (oracle[1] or oracle[2]) and oracle[0] == measured["bits"],
                "independent Q24 SiLU reconstruction disagreement")
        stage16["own"].append(measured["bits"])
    silu_disagreements = {}
    for name in projected["gate"]:
        for fixed in ("A", "R"):
            for gate_key, up_key in ((name, fixed), (fixed, name)):
                key = f"G:{gate_key}/U:{up_key}"
                if key not in stage16:
                    stage16[key], silu_disagreements[key] = true_silu(
                        projected["gate"][gate_key], projected["up"][up_key])
    chains = {order + "/" + activation: activation_chain(projection_chain(edges), first)
              for order, edges in norm_chains.items()
              for activation, first in (("gate_then_up", True), ("up_then_gate", False))}
    activation_rows = []
    with localcontext() as context:
        context.prec = 90
        for channel in selected:
            ideal = nearest_decimal(mathematical_silu(a[14][channel], a[15][channel]),
                                    (a[14][channel] ^ a[15][channel]) & 0x8000)
            activation_rows.append({
                "channel": channel, "bits": {k: v[channel] for k, v in stage16.items()},
                "actual_gate_bits": a[14][channel], "reference_gate_bits": r[14][channel],
                "actual_up_bits": a[15][channel], "reference_up_bits": r[15][channel],
                "decimal90_same_input_bits": ideal,
                "components_q24": components(stage16, chains, channel),
            })
    phase_times["stage13_gate_up_silu"] = time.monotonic() - phase
    phase = time.monotonic()
    down = {}
    keys = list(stage16)
    for offset in range(0, len(keys), 2):
        left, right = keys[offset], keys[min(offset + 1, len(keys) - 1)]
        bits, _ = project_vectors(stage16[left], stage16[right], tensors["down"])
        down[left], down[right] = bits["A"], bits["R"]
    require(down["A"] == parent["projection_bits"]["down"]["A"]
            and down["R"] == parent["projection_bits"]["down"]["R"], "parent down endpoints")
    weighted_down = []
    for row in parent["projection_producers"]["down"]:
        channel = row["channel"]
        pair = native_projection_pair(a[16], r[16], tensors["down"], channel)
        terms = {term["index"]: term for term in pair["terms"]}
        contributions = []
        for index in selected:
            term = terms[index]
            coefficient = (term["weight"] - term["zero"]) * units(int(term["scale_bits"], 16))
            contributions.append({
                "stage16_channel": index, "coefficient_q24": coefficient,
                "delta_q48": term["delta_q48"],
                "components_q48": {order: {name: value * coefficient for name, value in parts.items()}
                                   for order, parts in components(stage16, chains, index).items()},
            })
        selected_sum = sum(t["delta_q48"] for t in contributions)
        total = pair["actual_q48"] - pair["reference_q48"]
        require(total == sum(row["all_input_deltas_q48"]), "parent weighted down identity")
        weighted_down.append({
            "output": channel, "selected_terms": contributions, "total_delta_q48": total,
            "unselected_delta_q48": total - selected_sum,
            "rounded_components_q24": components(down, chains, channel),
        })
    downstream_norms, downstream_disagreements = {}, {}
    for fixed in ("A", "R"):
        for name, vector in down.items():
            key = fixed + "|" + name
            hidden = reference_add(stage12[fixed], vector)
            downstream_norms[key], downstream_disagreements[key], _ = norm_boundary_measurement(
                hidden, norm7_weight)
    phase_times["down_residual_full_norm"] = time.monotonic() - phase
    phase = time.monotonic()
    selected_channels = {"q": set(), "k": set()}
    for score in selected_scores:
        for row in score["all_dimensions"]:
            for kind in ("q", "k"):
                if kind == "q" or score["key_position"] == 3:
                    ch = row[kind + "_channel"]
                    base = ch // 64 * 64 + ch % 32
                    selected_channels[kind].update((base, base + 32))
    post, priority_q = {}, {}
    for kind, channels in selected_channels.items():
        raw, totals = qk_cases(downstream_norms, qk_tensors[kind], sorted(channels))
        for ch in (410, 442) if kind == "q" else ():
            priority_q[str(ch)] = {
                "raw_bits": raw[ch], "dot_totals_q48": totals[ch],
                "components_q48": {
                    fixed + "/" + order: {name: totals[ch][fixed + "|" + left]
                                          - totals[ch][fixed + "|" + right]
                                          for name, left, right in edges}
                    for fixed in ("A", "R") for order, edges in chains.items()},
            }
        for ch in sorted(channels):
            if ch % 64 >= 32:
                continue
            coeff = [int(x, 16) for x in producers[kind, 3, ch]["cos_sin_bits"]]
            rotated = {name: rotations([raw[ch][name], raw[ch + 32][name]],
                                       coeff, 3, ch % 32)[2] for name in downstream_norms}
            for side, channel in enumerate((ch, ch + 32)):
                post[kind, channel] = {name: bits[side] for name, bits in rotated.items()}
    scores = []
    for score, previous in zip(selected_scores, parent["scores"], strict=True):
        require((score["head"], score["key_position"]) ==
                (previous["head"], previous["key_position"]), "score row identity")
        refined = {}
        for fixed, outer in (("A", "stage12_first"), ("R", "stage17_first")):
            for order, edges in chains.items():
                parts = {kind: {name: 0 for name, _, _ in edges} for kind in ("q", "k")}
                for row in score["all_dimensions"]:
                    for kind in ("q", "k"):
                        if kind == "k" and score["key_position"] != 3:
                            continue
                        other = (("k", score["key_position"], row["k_channel"]) if kind == "q"
                                 else ("q", 3, row["q_channel"]))
                        coefficient = units(int(producers[other]["reference"], 16))
                        p = post[kind, row[kind + "_channel"]]
                        for name, left, right in edges:
                            parts[kind][name] += (units(p[fixed + "|" + left])
                                                  - units(p[fixed + "|" + right])) * coefficient
                parent_parts = previous["position3_producer_parts_dot_q48"][
                    outer + "/" + order.split("/", 1)[0]]
                for kind in ("q", "k"):
                    expected = sum(v for k, v in parent_parts[kind].items()
                                   if k.endswith("/stage16_activation"))
                    require(sum(parts[kind].values()) == expected,
                            "six-score stage16 parent edge closure")
                refined[outer + "/" + order] = parts
        scores.append({
            "head": score["head"], "key_position": score["key_position"],
            "stage16_refinement_dot_q48": refined, "score_scale_denominator": 2**51,
            "retained_parent_score": previous,
        })
    phase_times["six_score_refinement"] = time.monotonic() - phase
    discrepancy = {
        "local_stage13": differences(a[13], norms["own"]),
        "stage13_arithmetic_vs_torch_same_input": differences(norms["own"], norms["n:A"]),
        "reference_stage13": differences(norms["n:R"], r[13]),
        "local_gate": differences(a[14], projected["gate"]["P:A"]),
        "local_up": differences(a[15], projected["up"]["P:A"]),
        "reference_gate": differences(projected["gate"]["P:R"], r[14]),
        "reference_up": differences(projected["up"]["P:R"], r[15]),
        "local_stage16": differences(a[16], stage16["own"]),
        "stage16_arithmetic_vs_true_same_input": differences(stage16["own"], stage16["G:A/U:A"]),
        "reference_stage16": differences(stage16["G:R/U:R"], r[16]),
    }
    for record in sources:
        require(hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() == record["sha256"],
                "source changed during measurement")
    write(out / "measurements.json", {
        "stage13_bits": norms, "norm_chains": norm_chains, "stage13_reductions": reductions,
        "weighted_gate_up": weighted, "stage16_chains": chains, "stage16": activation_rows,
        "weighted_down": weighted_down, "priority_q": priority_q, "scores": scores,
        "discrepancies": discrepancy, "phase_seconds": phase_times,
        "counterfactual_norm_disagreements": norm_disagreements,
        "counterfactual_downstream_norm_disagreements": downstream_disagreements,
        "counterfactual_silu_disagreements": silu_disagreements,
        "preserved_parent_norm_disagreements": parent["counterfactual_norm_oracle_disagreements"],
        "preserved_parent_discrepancies": parent["discrepancies"],
        "connected_stage12": [{
            "index": row["index"], "incoming_actual": row["actual_input"],
            "incoming_reference": row["reference_input"], "O_actual": row["actual_O"],
            "O_reference": row["reference_O"],
            "both_orders": row["stage12_components_q24"],
        } for row in parent["coordinates"]],
    })
    local = any(discrepancy[k] for k in ("local_stage13", "local_gate", "local_up", "local_stage16"))
    boundary = {
        "stage13_inputs": {
            f"{row['kind']}/{row['channel']}": [term["index"] for term in row["top_stage13_terms"][:8]]
            for row in weighted if row["channel"] in PRIORITY_CHANNELS},
        "incoming_hidden": "layer05 stage18 -> layer06 stage12, all 896 coordinates in the denominator",
        "attention_value": {
            str(row["channel"]): [t["index"] for t in row["top_input_terms"][:8]]
            for row in parent["projection_producers"]["o"] if row["channel"] in (483, 494, 694)},
        "intervention": "Discriminate layer06 stage10 AV operands (stage09 probabilities versus "
                        "historical/current V) at the retained weighted channels, separately from "
                        "layer05 stage18 incoming-hidden producers at the ranked stage13 inputs. "
                        "Both paths enter stage12 directly and feed stage13's full denominator. "
                        "Keep both gate/up and incoming/O orders; do not presume a unique cause.",
    }
    if local:
        boundary["intervention"] = (
            "Resolve the listed own-input stage13/gate/up/stage16 reconstruction discrepancies "
            "against the frozen retained binary before proposing an arithmetic repair.")
    result = {
        "status": "STAGE16_PRODUCER_BOUNDARY_LOCALIZED_NOT_REPAIRED" if not local else
                  "LOCAL_ACTIVATION_RECONSTRUCTION_DISCREPANCY_NOT_REPAIRED",
        "counts": {k: len(v) for k, v in discrepancy.items()}, "weighted_channels": len(selected),
        "score_rows": len(scores), "score_attribution_orders": 8,
        "priority_channels": PRIORITY_CHANNELS, "next_upstream_boundary": boundary,
        "general_implementation_defect_proven": False, "repair": None,
        "RTL_simulations": 0, "contract_compiles": len(compiles), "binary64_evaluated": False,
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "hypothesis": "Inherited stage12 inputs and separately measured RMSNorm/SiLU arithmetic "
                      "effects propagate through exact packed projections and rounded residuals; "
                      "none is established as the unique cause.",
        "regression": "Independent integer norm/projection/SiLU checks, exact additive closures, "
                      "both causal orders and all six retained score stage16 edges.",
        "original_failures": "Preserved through the authenticated parent and nested score records.",
        "review_status": "PENDING_NORMAL_HOST_REVIEWER",
        "limits": ["software counterfactuals only; not injected into actual execution",
                   "no trajectory PASS, new token, K/V splice, changed gate or source promotion",
                   "historical K and Q/K interaction remain separately retained",
                   "counterfactual RMSNorm library disagreements are not an RTL defect"],
    }
    write(out / "result.json", result)
    for path in out.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out.resolve())
