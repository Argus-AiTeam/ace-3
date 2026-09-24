#!/usr/bin/env python3
"""Retained stage00 producer accounting; no trajectory execution or injection."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import torch

from diagnose_lineage_a_position3_boundary import (
    HISTORY, POLICY, PREFIX, Inputs, accepts, decode_trace, require, units, write,
)
from fp16_adaptation_oracle import rmsnorm
from official_single_decoder_layer import _torch_rmsnorm
from trace_lineage_a_layer07_downproj import framed_hidden, hex_rows, round_q48
from trace_lineage_a_layer07_gate_up import native_projection_pair, snapshot
from trace_lineage_a_layer07_qk_producers import projection, rotations, stage_arrays
from trace_lineage_a_layer07_rmsnorm import reconstruct
from trace_lineage_a_layer07_score_softmax_v import DESTINATION, ROOT, array


PARENT = DESTINATION / "lineage_a_layer07_qk_bd2d1b4dffa3_attempt002"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
              "bd2d1b4dffa3/round-0001.json")
ROOT_HASHES = {
    "frozen.json": "8cedf702d2074dea75251f424e840c747fce91a8e626826f629b62b6b7405a5e",
    "measurements.json": "6f10ff9abf032d1c903b55d0a0ab32a76c94a0d14186eb31a64f246a8a7b2725",
    "result.json": "1c7b6b338722ab522ce880657aecd215bcb3bf444e08efd28102a8ff09c68f3f",
}
PRIORITIES = ((6, 1), (6, 2), (6, 3), (4, 3), (12, 2), (12, 3))
PARTS = ("local_rmsnorm", "rmsnorm_policy_on_actual_hidden",
         "layer06_hidden_under_reference_rmsnorm", "reference_recurrence")


def reference_norm(activation, weight):
    require(activation and len(activation) == len(weight), "reference norm geometry")
    for value in (*activation, *weight):
        units(value)
    a = np.asarray(activation, dtype="<u2").view("<f2").astype(np.float64)
    w = np.asarray(weight, dtype="<u2").view("<f2")
    got = _torch_rmsnorm(torch.from_numpy(a), w).to(torch.float16).numpy().view("<u2")
    # Independent reduction/rsqrt implementation, retaining the specified epsilon.
    independent = (a * (1.0 / np.sqrt(np.mean(a * a) + 1e-6))
                   * w.astype(np.float64)).astype("<f2").view("<u2")
    require(np.array_equal(got, independent), "independent RMSNorm rounding disagreement")
    result = got.tolist()
    for value in result:
        units(value)
    return result


def differences(a, b):
    require(len(a) == len(b), "difference geometry")
    return [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]


def split_bits(chain):
    require(len(chain) == len(PARTS) + 1, "component chain geometry")
    parts = {name: units(a) - units(b)
             for name, a, b in zip(PARTS, chain, chain[1:], strict=False)}
    require(sum(parts.values()) == units(chain[0]) - units(chain[-1]),
            "component closure")
    return parts


def norm_measurement(hidden, reference_hidden, weight, retained, reference):
    native, reduction = reconstruct(hidden, weight)
    oracle, mean, root = rmsnorm(hidden, weight)
    require(all(not invalid and not sat for _, invalid, sat in oracle)
            and native == [b for b, _, _ in oracle]
            and (mean, root) == (reduction["mean_q48"], reduction["rms_q24"]),
            "independent integer RMSNorm disagreement")
    accepted = reference_norm(hidden, weight)
    propagated = reference_norm(reference_hidden, weight)
    native_reference, reference_reduction = reconstruct(reference_hidden, weight)
    chains = [retained, native, accepted, propagated, reference]
    require(all(len(x) == 896 for x in chains), "stage00 geometry")
    rows = []
    for i in range(896):
        parts = split_bits([x[i] for x in chains])
        policy_on_reference = units(native_reference[i]) - units(propagated[i])
        rows.append({
            "index": i, "actual_hidden": f"{hidden[i]:04x}",
            "reference_hidden": f"{reference_hidden[i]:04x}",
            "weight": f"{weight[i]:04x}",
            "chain_bits": [f"{x[i]:04x}" for x in chains],
            "components_q24": parts,
            "policy_on_reference_hidden_q24": policy_on_reference,
            "policy_hidden_interaction_q24": (
                parts["rmsnorm_policy_on_actual_hidden"] - policy_on_reference),
        })
    return chains, rows, {
        "actual_reduction": reduction, "reference_reduction": reference_reduction,
        "hidden_bit_differences": differences(hidden, reference_hidden),
        "local_bit_differences": differences(retained, native),
        "policy_on_actual_bit_differences": differences(native, accepted),
        "policy_on_reference_bit_differences": differences(native_reference, propagated),
        "reference_recurrence_bit_differences": differences(propagated, reference),
        "stage00_bit_differences": differences(retained, reference),
        "stage00_material_differences": [
            i for i in range(896) if not accepts(retained[i], reference[i])],
        "component_l1_q24": {
            name: sum(abs(row["components_q24"][name]) for row in rows) for name in PARTS},
    }


def weighted_channel(chains, tensors, channel):
    measured = native_projection_pair(chains[0], chains[-1], tensors, channel)
    coefficients = [(t["weight"] - t["zero"]) * units(int(t["scale_bits"], 16))
                    for t in measured["terms"]]
    totals = [sum(units(x) * c for x, c in zip(row, coefficients, strict=True))
              for row in chains]
    require(totals[0] == measured["actual_q48"]
            and totals[-1] == measured["reference_q48"], "AWQ dot identity")
    rows = []
    for i, (term, coefficient) in enumerate(zip(measured["terms"], coefficients, strict=True)):
        parts = {name: value * coefficient
                 for name, value in split_bits([x[i] for x in chains]).items()}
        require(sum(parts.values()) == term["delta_q48"], "weighted input closure")
        rows.append({
            "input_index": i, "group": term["group"], "nibble_shift": term["nibble_shift"],
            "weight": term["weight"], "zero": term["zero"], "scale_bits": term["scale_bits"],
            "actual_stage00": term["actual_stage13_bits"],
            "reference_stage00": term["reference_stage13_bits"],
            "delta_q48": term["delta_q48"], "components_q48": parts,
        })
    parts = {name: sum(row["components_q48"][name] for row in rows) for name in PARTS}
    require(sum(parts.values()) == totals[0] - totals[-1], "weighted dot closure")
    bias = units(tensors["bias"][channel]) << 24
    rounded = []
    for total in totals:
        bits = round_q48(total + bias)
        if bits == 0 and total + bias < 0:
            bits = 0x8000
        rounded.append(bits)
    for activation, bits in ((chains[0], rounded[0]), (chains[-1], rounded[-1])):
        require(projection(activation, tensors, channel, True)[0] == bits,
                "independent native GEMM projection disagreement")
    rows.sort(key=lambda r: (-abs(r["delta_q48"]), r["input_index"]))
    return rounded, rows, parts


def command(out, label, argv):
    with (out / f"{label}.command.sh").open("x", encoding="ascii") as stream:
        stream.write(shlex.join(argv) + "\n")
    with (out / f"{label}.log").open("x", encoding="ascii") as stream:
        completed = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    require(completed.returncode == 0,
            f"{label}: exit {completed.returncode}; original command and log retained")


def run(out):
    require(out.parent == DESTINATION.resolve()
            and out.name.startswith("lineage_a_layer07_stage00_"), "output scope")
    out.mkdir(exist_ok=False)
    inputs = Inputs()
    roots = {}
    for name, expected in ROOT_HASHES.items():
        roots[name] = inputs.load(PARENT / name, root=True)
        require(inputs.records[str(PARENT / name)]["sha256"] == expected,
                f"reviewed parent changed: {name}")
    freeze, parent = roots["frozen.json"], roots["measurements.json"]
    review = inputs.load(REVIEW, root=True)
    require(review["kind"] == "round_reviewed_handoff"
            and review["producer_role"] == "reviewer"
            and review["mission_id"] == "bd2d1b4dffa3"
            and review["review"]["status"] == "done", "missing genuine predecessor review")
    require(freeze["token_history"] == HISTORY and freeze["comparison_policy"] == POLICY,
            "foreign lineage or numerical policy")
    prepared = inputs.load(PREFIX / "layer07/prepared.json")
    result = inputs.load(PREFIX / "layer07/result.json")
    prefix = inputs.load(PREFIX / "frozen.json")
    original = inputs.load(prefix["original_frozen"]["path"])
    policy = inputs.load(original["independent_policy"]["path"])
    require(policy["comparison"] == POLICY, "changed independent policy")
    # These are additive roots, not retroactive claims about the predecessor freeze.
    inputs.load(PREFIX / "seal.json", root=True)
    layer6 = inputs.load(PREFIX / "layer06/result.json")
    require(layer6["status"] == "ZERO_MATERIAL_MISMATCHES"
            and layer6["output_hidden"] == prepared["input_hidden"],
            "position3 incoming hidden is not the retained layer06 output")
    hidden = {3: framed_hidden(inputs.read(prepared["input_hidden"]["path"]))}
    require(hidden[3] == framed_hidden(inputs.read(prepared["vectors"]["input"]["path"])),
            "consumed position3 input differs from hidden parent")
    reference_hidden = {3: hex_rows(inputs.read(layer6["independent_hidden"]["path"]))}
    kv = prepared["kv_parent"]
    require(kv["layer_index"] == 7 and kv["valid_positions"] == [0, 1, 2]
            and prepared["binary"] == kv["live_binary"]
            and "/model24_persistent_kv_selected_token_corrected_q_attempt005/"
            in kv["state"]["path"], "incompatible A-lineage K/V or ABI")
    inputs.read(kv["state"]["path"])
    inputs.read(prepared["binary"]["path"])
    previous = inputs.load(kv["layer_result"]["path"])
    entry = previous["positions"][2]
    require(entry["position"] == 2 and entry["layer_index"] == 7, "historical input identity")
    hidden[2] = framed_hidden(inputs.read(entry["vectors"]["input"]["path"]))
    require(hashlib.sha256(b"".join(x.to_bytes(2, "little") for x in hidden[2])).hexdigest()
            == entry["input"]["sha256"], "historical consumed hidden semantic hash")
    historical6_path = Path(kv["layer_result"]["path"]).parent.parent / "layer06/result.json"
    historical6 = inputs.load(historical6_path, root=True)
    hidden6 = historical6["positions"][2]["output"]
    require(hidden6["sha256"] == entry["vectors"]["input"]["sha256"]
            and hidden6["semantic_sha256"] == entry["input"]["sha256"]
            and framed_hidden(inputs.read(hidden6["path"])) == hidden[2],
            "historical layer06-to-layer07 hidden identity")
    history = inputs.load(prepared["independent_parent"]["path"])
    require(history["history"] == HISTORY[:3] and history["layer"] == 7,
            "foreign historical reference")
    refs = stage_arrays(inputs, history)
    refs[3] = {int(stage): hex_rows(inputs.read(record["path"]))
               for stage, record in prepared["independent_stages"].items()
               if int(stage) in (0, 1, 2, 4, 6)}
    historical_ref_path = Path(prepared["independent_parent"]["path"]).with_name(
        "layer06_generation1.json")
    prior_reference_bound = str(historical_ref_path) in inputs.bindings
    historical_ref = inputs.load(historical_ref_path, root=True)
    require(historical_ref["history"] == HISTORY[:3]
            and historical_ref["layer"] == 6 and historical_ref["positions"] == [2],
            "historical reference hidden identity")
    records = [r for r in historical_ref["stages"] if r["path"].endswith("/stage18.npy")]
    require(len(records) == 1, "historical reference hidden geometry")
    reference_hidden[2] = array(inputs, records[0]).reshape(-1).tolist()
    actual = {2: decode_trace(inputs.read(kv["actual_kv_traces"][2]["path"]), 2),
              3: decode_trace(inputs.read(Path(result["output_hidden"]["path"])
                                         .with_name("trace.hex")), 3)}
    tensors = {"q": {}, "k": {}}
    weight = None
    for record in prepared["vectors"]["tensors"]:
        meta = record["checkpoint_tensor"]
        name = meta["name"]
        selected = next((kind for kind in tensors if f".{kind}_proj." in name), None)
        if selected is None and name != "model.layers.7.input_layernorm.weight":
            continue
        values = hex_rows(inputs.read(record["serialized"]["path"]))
        raw = b"".join(x.to_bytes(2 if meta["dtype"] == "F16" else 4, "little") for x in values)
        require(len(raw) == meta["bytes"] and hashlib.sha256(raw).hexdigest() == meta["sha256"]
                and any(r["name"] == name and r["sha256"] == meta["sha256"]
                        for r in history["checkpoint_tensor_hashes"]),
                "serialized parameter differs from official independent tensor")
        if selected is None:
            weight = values
        else:
            tensors[selected][name.rsplit(".", 1)[1]] = values
    require(weight is not None and len(weight) == 896
            and all(set(t) == {"qweight", "qzeros", "scales", "bias"} for t in tensors.values()),
            "missing normalization/projection parameters")
    sources = []
    source_paths = {Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                    if getattr(module, "__file__", None)
                    and Path(module.__file__).resolve().is_relative_to(ROOT / "ace3/model")
                    and Path(module.__file__).suffix == ".py"}
    source_paths.add(Path(__file__).resolve())
    source_paths.add(ROOT / "ace3/model/tests/test_trace_lineage_a_layer07_stage00_inputs.py")
    source_paths.update(ROOT / "ace3/rtl" / name for name in (
        "ace3_fp16_rmsnorm_core.sv", "ace3_fp16_fixed.sv",
        "ace3_decoder_layer0_token_engine.sv"))
    previous_sources = {r["path"]: r for r in freeze["sources"]}
    for path in sorted(source_paths):
        record = snapshot(out, path)
        if str(path) in previous_sources:
            require(record["sha256"] == previous_sources[str(path)]["sha256"],
                    f"retained producer/helper source changed: {path}")
        sources.append(record)
    top = "ace3_fp16_rmsnorm_core"
    compile_command = ["iverilog", "-g2012", "-s", top, "-o", str(out / f"{top}.vvp"),
                       str(out / f"{top}.sv"), str(out / "ace3_fp16_fixed.sv")]
    test_command = [sys.executable, "-B", "-m", "unittest", "discover",
                    "-s", str(ROOT / "ace3/model/tests"),
                    "-p", "test_trace_lineage_a_layer07_stage00_inputs.py"]
    version_command = ["iverilog", "-V"]
    command(out, "iverilog_version", version_command)
    write(out / "frozen.json", {
        "scope": "A-only retained layer07 stage00; software accounting, no RTL execution",
        "token_history": HISTORY, "comparison_policy": POLICY,
        "priority_scores": PRIORITIES, "positions_with_hidden_reconstruction": [2, 3],
        "projection": "Native G128 asymmetric GEMM INT4, no zero+1, FP16 scales, Q/K single-round bias",
        "components": PARTS, "score_denominator": 2**51,
        "public_declaration": (out / f"{top}.sv").read_text().split(");", 1)[0] + ");",
        "commands": [version_command, compile_command, test_command],
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "authenticated_inputs": list(inputs.records.values()), "sources": sources,
        "new_retained_roots": [str(PREFIX / "seal.json"), str(historical6_path),
                               str(historical_ref_path)],
        "position2_reference_hidden_previously_bound": prior_reference_bound,
        "position2_reference_hidden_limit": (
            "If not previously bound, the new layer06 manifest is only content-bound. "
            "Its reconstruction of an authenticated stage00 vector is not unique provenance; "
            "the RMSNorm/upstream split remains conditional."),
        "historical_positions01": parent["historical_reference_boundary"],
        "limits": ["No reference injection", "No binary64 evaluation or policy change",
                   "No arithmetic repair or fresh RTL traversal", "No performance claim",
                   "Historical position0/1 K and V remainders remain unsplit"],
    })
    command(out, "public_contract_compile", compile_command)
    command(out, "unit_tests", test_command)
    norms, norm_rows, norm_summary = {}, {}, {}
    for position in (2, 3):
        norms[position], norm_rows[position], norm_summary[position] = norm_measurement(
            hidden[position], reference_hidden[position], weight,
            actual[position][0], refs[position][0])
    producers = {(r["kind"], r["position"], r["channel"]): r for r in parent["producers"]}
    selected_scores = [next(r for r in parent["scores"]
                            if (r["head"], r["key_position"]) == key) for key in PRIORITIES]
    selected = set()
    ranked = set()
    for score in selected_scores:
        for kind in ("q", "k"):
            position = 3 if kind == "q" else score["key_position"]
            if position < 2:
                continue
            for row in score["all_dimensions"]:
                selected.add((kind, position, row[f"{kind}_channel"] // 64,
                              row[f"{kind}_channel"] % 32))
            for row in score[f"ranked_{kind}_dimensions"]:
                ranked.add((kind, position, row[f"{kind}_channel"]))
    post_parts, projection_rows = {}, []
    for kind, position, head, dim in sorted(selected):
        channels = (head * 64 + dim, head * 64 + dim + 32)
        raw_chains = []
        for channel in channels:
            raw_chain, terms, dot_parts = weighted_channel(norms[position], tensors[kind], channel)
            previous_producer = producers[kind, position, channel]
            require(raw_chain[0] == int(previous_producer["raw_actual"], 16)
                    and raw_chain[-1] == int(previous_producer["raw_reference"], 16),
                    "retained raw Q/K reconstruction")
            old_terms = previous_producer["largest_stage00_dot_delta_terms"]
            require(all({k: row[k] for k in old} == old
                        for row, old in zip(terms[:8], old_terms, strict=True)),
                    "ranked normalized-input identity changed")
            raw_chains.append(raw_chain)
            if (kind, position, channel) in ranked:
                projection_rows.append({
                    "kind": kind, "position": position, "channel": channel,
                    "components_dot_q48": dot_parts, "top_weighted_inputs": terms[:16],
                    "all_input_delta_dot_q48": sum(dot_parts.values()),
                })
        previous_producer = producers[kind, position, channels[0]]
        coeff = [int(x, 16) for x in previous_producer["cos_sin_bits"]]
        post = [rotations([a, b], coeff, position, dim)[2]
                for a, b in zip(*raw_chains, strict=True)]
        for side, channel in enumerate(channels):
            previous_producer = producers[kind, position, channel]
            require(post[0][side] == int(previous_producer["chain_bits"][5], 16)
                    and post[-1][side] == int(previous_producer["chain_bits"][6], 16),
                    "reviewed stage00-to-post-RoPE identity")
            post_parts[kind, position, channel] = split_bits([p[side] for p in post])
    scores = []
    for score in selected_scores:
        totals = {"q": dict.fromkeys(PARTS, 0), "k": dict.fromkeys(PARTS, 0)}
        unsplit = {"q": 0, "k": 0}
        for row in score["all_dimensions"]:
            for kind in ("q", "k"):
                position = 3 if kind == "q" else score["key_position"]
                if position < 2:
                    unsplit[kind] += row[f"{kind}_producer_parts_dot_q48"].get(
                        "incoming_or_reference_remainder", 0)
                    continue
                channel = row[f"{kind}_channel"]
                other = (("k", score["key_position"], row["k_channel"]) if kind == "q"
                         else ("q", 3, row["q_channel"]))
                coefficient = units(int(producers[other]["reference"], 16))
                parts = post_parts[kind, position, channel]
                weighted = {name: value * coefficient for name, value in parts.items()}
                require(sum(weighted.values())
                        == row[f"{kind}_producer_parts_dot_q48"]["stage00_trajectory"],
                        "reviewed weighted score component closure")
                for name, value in weighted.items():
                    totals[kind][name] += value
        scores.append({
            "head": score["head"], "key_position": score["key_position"],
            "stage00_parts_dot_q48": totals,
            "stage00_parts_scaled": {
                kind: {name: str(Fraction(value, 2**51)) for name, value in parts.items()}
                for kind, parts in totals.items()},
            "historical_unsplit_dot_q48": unsplit,
            "unchanged_parent_producer_parts_dot_q48": score["producer_parts_dot_q48"],
            "unchanged_parent_totals_dot_q48": score["totals_dot_q48"],
            "position2_k_upstream_split_conditional": (
                score["key_position"] == 2 and not prior_reference_bound),
        })
    local = {p: s["local_bit_differences"] for p, s in norm_summary.items()}
    reference_errors = {p: s["reference_recurrence_bit_differences"]
                        for p, s in norm_summary.items()}
    write(out / "measurements.json", {
        "norm_coordinates": norm_rows, "norm_summary": norm_summary,
        "ranked_projection_inputs": projection_rows, "scores": scores,
        "historical_v_unsplit": [
            {"position": r["position"], "av_index": r["av_index"],
             "components_q24": r["components_q24"]}
            for r in parent["v_producers"] if r["position"] < 2],
    })
    summary = {
        "status": "STAGE00_BOUNDARY_LOCALIZED_NOT_REPAIRED",
        "normalization_coordinates": 1792, "ranked_projection_channels": len(projection_rows),
        "score_rows": len(scores), "local_discrepancies": local,
        "reference_recurrence_discrepancies": reference_errors,
        "counts": {p: {k: len(v) for k, v in s.items() if isinstance(v, list)}
                   for p, s in norm_summary.items()},
        "next_boundary": (
            "Layer06 position3 stage18 residual output feeding layer07 stage00, using the "
            "ranked input indices and exact weighted Q/K contributions in measurements.json. "
            "Decompose layer06 stage12 residual versus stage17 down projection on this same "
            "authenticated hidden boundary, retaining the separately measured RMSNorm policy "
            "and hidden/policy interaction. Position2's layer06 reference manifest is an "
            "additional root, so its upstream split is conditional unless the freeze records "
            "a pre-existing binding. Position0/1 K and V remainders stay unsplit."),
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "hypothesis": "Incoming hidden differences and declared RMSNorm numerical boundaries "
                      "can contribute separately; an algebraic split is not a unique defect.",
        "regression": "Independent RMSNorm reconstruction, ranked native AWQ input identities "
                      "and exact closure back to six retained Q/K score rows.",
        "general_defect_proven": False, "repair": None, "RTL_simulations": 0,
        "contract_compiles": 1, "binary64_evaluated": False,
        "review_status": "PENDING_NORMAL_HOST_REVIEWER",
    }
    if any(local.values()) or any(reference_errors.values()):
        summary["status"] = "STAGE00_RECONSTRUCTION_DISCREPANCY_NOT_REPAIRED"
        summary["next_boundary"] = "Resolve the explicit reconstruction discrepancies before upstream attribution."
    write(out / "result.json", summary)
    for path in out.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out.resolve())
