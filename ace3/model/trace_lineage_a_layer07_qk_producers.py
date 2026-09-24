#!/usr/bin/env python3
"""Source-bound, retained Q/K producer accounting; never execute a trajectory."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from diagnose_lineage_a_position3_boundary import (
    HISTORY, POLICY, PREFIX, Inputs, accepts, decode_trace, require, units, write,
)
from projection_oracle import complete_projection_output
from qwen2_rope_oracle import f16_bits, rotate_pair
from trace_lineage_a_layer07_downproj import hex_rows, round_q48
from trace_lineage_a_layer07_gate_up import native_projection_pair, snapshot
from trace_lineage_a_layer07_o_projection import recorded_command
from trace_lineage_a_layer07_score_softmax_v import (
    AV_INDICES, DESTINATION, HEADS, ROOT, array, score_bits, v_projection,
)
from trace_lineage_a_layer07_silu import source_records


PARENT = DESTINATION / "lineage_a_layer07_score_softmax_v_e1de6b4303c9_attempt004"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
              "e1de6b4303c9/round-0002.json")
PARTS = (
    "local_rope", "rope_product_rounding", "rope_coefficient_quantization",
    "local_projection", "projection_bias_rounding", "stage00_trajectory",
    "reference_recurrence",
)
DECODER_SINGLE_ROUND_PRODUCERS = ("q", "k")


def split(chain, names=PARTS):
    require(len(chain) == len(names) + 1, "producer chain geometry")
    result = {name: units(a) - units(b)
              for name, a, b in zip(names, chain, chain[1:], strict=False)}
    require(sum(result.values()) == units(chain[0]) - units(chain[-1]),
            "producer components do not close")
    return result


def projection(activation, tensors, channel, single_round):
    measured = native_projection_pair(activation, activation, tensors, channel)
    bias = tensors["bias"][channel]
    direct = round_q48(measured["actual_q48"] + (units(bias) << 24))
    if direct == 0 and measured["actual_q48"] + (units(bias) << 24) < 0:
        direct = 0x8000
    local = direct if single_round else round_q48(
        (units(measured["actual_bits"]) + units(bias)) << 24)
    count = len(tensors["bias"])
    words = count // 8
    total, oracle, invalid, overflow, _ = complete_projection_output(
        activation, tensors["qweight"][channel // 8::words],
        tensors["qzeros"][channel // 8::words], tensors["scales"][channel::count],
        channel % 8, bias, single_round_bias=single_round)
    require(not (invalid or overflow) and total == measured["actual_q48"]
            and oracle == local, "independent native projection disagreement")
    return local, direct


def rotations(pair, coefficients, position, dim):
    x, y = pair
    c, s = coefficients

    def mul(a, b):
        product = units(a) * units(b)
        return round_q48(product) if product else (a ^ b) & 0x8000

    local = [
        round_q48((units(mul(x, c)) - units(mul(y, s))) << 24),
        round_q48((units(mul(y, c)) + units(mul(x, s))) << 24),
    ]
    oracle = rotate_pair(x, y, c, s)
    require(not any(oracle[2:]) and local == list(oracle[:2]),
            "independent multiply/add RoPE disagreement")
    quantized = [
        round_q48(units(x)*units(c) - units(y)*units(s)),
        round_q48(units(y)*units(c) + units(x)*units(s)),
    ]
    angle = position / (1_000_000.0 ** (2 * dim / 64))
    cosine, sine = math.cos(angle), math.sin(angle)
    xf, yf = units(x) / 2**24, units(y) / 2**24
    mathematical = [f16_bits(xf*cosine-yf*sine), f16_bits(yf*cosine+xf*sine)]
    return local, quantized, mathematical


def score_terms(q, qr, k, kr):
    require(len(q) == len(qr) == len(k) == len(kr) == 64, "score geometry")
    rows = []
    for dim, (a, r, b, s) in enumerate(zip(q, qr, k, kr, strict=True)):
        dq, dk = units(a)-units(r), units(b)-units(s)
        rows.append({
            "dimension": dim, "q_dot_q48": dq*units(s),
            "k_dot_q48": units(r)*dk, "interaction_dot_q48": dq*dk,
        })
    total = sum(units(a)*units(b)-units(r)*units(s)
                for a, r, b, s in zip(q, qr, k, kr, strict=True))
    require(sum(sum(row[key] for key in ("q_dot_q48", "k_dot_q48",
                                        "interaction_dot_q48")) for row in rows) == total,
            "score decomposition does not close")
    return rows


def stage_arrays(inputs, manifest):
    stages = {}
    for record in manifest["stages"]:
        path = Path(record["path"])
        stage = int(path.stem.removeprefix("stage"))
        if stage not in (0, 1, 2, 3, 6):
            continue
        position = int(path.parent.name.removeprefix("position"))
        stages.setdefault(position, {})[stage] = array(inputs, record).reshape(-1).tolist()
    return stages


def run(out):
    require(out.is_relative_to(DESTINATION.resolve()), "output outside writable build scope")
    out.mkdir(parents=True, exist_ok=False)
    inputs = Inputs()
    for filename, expected in (
        ("frozen.json", "5bd55c2da9cd374bcbb61dc79f7ccaa1906074cad4904b770e86c422a2f7bea5"),
        ("artifacts.json", "eef2fd1642fcddf64f4395c1b5433e8ab526cb3d7bc218721dbe5f900f4c826d"),
    ):
        inputs.load(PARENT / filename, root=True)
        require(inputs.records[str(PARENT / filename)]["sha256"] == expected,
                "reviewed attempt004 root changed")
    parent = inputs.load(PARENT / "measurements.json")
    review = inputs.load(REVIEW, root=True)
    require(review["kind"] == "round_reviewed_handoff"
            and review["producer_role"] == "reviewer"
            and review["mission_id"] == "e1de6b4303c9"
            and review["review"]["status"] == "done", "missing genuine parent review")
    prepared = inputs.load(PREFIX / "layer07/prepared.json")
    result = inputs.load(PREFIX / "layer07/result.json")
    prefix = inputs.load(PREFIX / "frozen.json")
    original = inputs.load(prefix["original_frozen"]["path"])
    policy = inputs.load(original["independent_policy"]["path"])
    require(original["comparison_policy"] == policy["comparison"] == POLICY,
            "numerical policy changed")
    superseded = DESTINATION / "lineage_a_layer07_qk_bd2d1b4dffa3_attempt001"
    inputs.load(superseded / "frozen.json", root=True)
    inputs.load(superseded / "result.json", root=True)
    kv = prepared["kv_parent"]
    require(kv["valid_positions"] == [0, 1, 2] and kv["layer_index"] == 7
            and prepared["binary"] == kv["live_binary"]
            and "/model24_persistent_kv_selected_token_corrected_q_attempt005/"
            in kv["state"]["path"], "foreign state lineage or ABI")
    inputs.read(kv["state"]["path"])
    inputs.read(prepared["binary"]["path"])
    inputs.load(kv["layer_result"]["path"])
    history = inputs.load(prepared["independent_parent"]["path"])
    require(history["history"] == HISTORY[:3] and history["layer"] == 7,
            "foreign independent reference history")
    actual = [decode_trace(inputs.read(record["path"]), pos)
              for pos, record in enumerate(kv["actual_kv_traces"])]
    actual.append(decode_trace(inputs.read(Path(result["output_hidden"]["path"])
                                          .with_name("trace.hex")), 3))
    reference = {int(stage): hex_rows(inputs.read(record["path"]))
                 for stage, record in prepared["independent_stages"].items()
                 if int(stage) in (0, 1, 2, 3, 4, 6, 8)}
    kr = array(inputs, history["own_cache"]["k"]).reshape(3, 128).tolist() + [reference[6]]
    vr = array(inputs, history["own_cache"]["v"]).reshape(3, 128).tolist() + [reference[3]]
    refs = stage_arrays(inputs, history)
    refs[3] = reference

    # A newly observed manifest is not retroactively bound by the older freeze.
    historical_path = Path(prepared["independent_parent"]["path"]).with_name(
        "layer07_generation0.json")
    historical = inputs.load(historical_path, root=True)
    require(historical["layer"] == 7 and historical["history"] == HISTORY[:2]
            and historical["positions"] == [0, 1]
            and historical["checkpoint_tensor_hashes"] == history["checkpoint_tensor_hashes"],
            "historical manifest identity mismatch")
    for name, retained in (("k", kr), ("v", vr)):
        historical_cache = array(inputs, historical["own_cache"][name]).reshape(2, 128).tolist()
        require(historical_cache == retained[:2], "historical cache prefix mismatch")
    refs.update(stage_arrays(inputs, historical))
    require(set(refs) == {0, 1, 2, 3}
            and all(len(refs[p][0]) == len(actual[p][0]) == 896 for p in refs),
            "normalized input geometry")
    historical_boundary = {
        "manifest": inputs.records[str(historical_path)],
        "prior_freeze_binds_manifest": False,
        "cache_prefix_equality": "full K and V, two positions, 512 FP16 words",
        "scope": "Newly content-bound generation0 manifest and arrays; exact consumed-cache "
                 "prefix equality is not unique authentication of its normalized inputs. "
                 "Historical stage00 splits are conditional; authoritative remainder stays unsplit.",
    }
    tensors = {kind: {} for kind in ("q", "k", "v")}
    for record in prepared["vectors"]["tensors"]:
        meta = record["checkpoint_tensor"]
        for kind in tensors:
            if f".self_attn.{kind}_proj." not in meta["name"]:
                continue
            values = hex_rows(inputs.read(record["serialized"]["path"]))
            raw = b"".join(v.to_bytes(2 if meta["dtype"] == "F16" else 4, "little")
                           for v in values)
            require(len(raw) == meta["bytes"]
                    and hashlib.sha256(raw).hexdigest() == meta["sha256"]
                    and any(r["name"] == meta["name"] and r["sha256"] == meta["sha256"]
                            for r in history["checkpoint_tensor_hashes"]),
                    "serialized tensor is not the independent official AWQ tensor")
            tensors[kind][meta["name"].rsplit(".", 1)[1]] = values
    require(all(set(t) == {"qweight", "qzeros", "scales", "bias"}
                for t in tensors.values()), "missing producer tensors")
    coefficient_rows = inputs.read(prepared["vectors"]["rope_coefficients"]["path"]).splitlines()
    require(len(coefficient_rows) == 128 and all(
        len(row) == 14 and int(row[:4], 16) == i // 32
        and int(row[4:6], 16) == i % 32 for i, row in enumerate(coefficient_rows)),
        "RoPE coefficient framing")
    coefficients = [[tuple(int(row[i:i+4], 16) for i in (6, 10))
                     for row in coefficient_rows[p*32:(p+1)*32]] for p in range(4)]
    rtl_names = (
        "ace3_fp16_fixed.sv", "ace3_awq_w4a16_g128_dot_lane.sv",
        "ace3_q47_48_to_f16_rne.sv", "ace3_awq_w4a16_projection_engine.sv",
        "ace3_qkv_projection_cluster.sv", "ace3_qwen2_rope_pair.sv",
        "ace3_attention_score_core.sv",
    )
    rtl_names = tuple(dict.fromkeys((*rtl_names, *sorted({
        Path(record["path"]).name for record in source_records(original)
        if Path(record["path"]).parent == ROOT / "ace3/rtl"
        and Path(record["path"]).suffix == ".sv"
    }))))
    require("ace3_decoder_layer0_token_engine.sv" in rtl_names,
            "actual decoder source is not historically bound")
    sources = []
    for name in rtl_names:
        path = ROOT / "ace3/rtl" / name
        record = snapshot(out, path)
        bound = [r for r in source_records(original) if r["path"] == str(path)]
        require(bound and all(r["sha256"] == record["sha256"] for r in bound),
                f"retained RTL source differs: {name}")
        sources.append(record)
    for name in (
        Path(__file__).name, "diagnose_lineage_a_position3_boundary.py",
        "projection_oracle.py", "awq_bit_oracle.py", "fp16_adaptation_oracle.py",
        "qwen2_rope_oracle.py", "attention_oracle.py",
        "trace_lineage_a_layer07_score_softmax_v.py",
        "trace_lineage_a_layer07_downproj.py", "trace_lineage_a_layer07_gate_up.py",
        "trace_lineage_a_layer07_silu.py", "trace_lineage_a_layer07_o_projection.py",
        "trace_lineage_a_layer07_attention_value.py",
    ):
        sources.append(snapshot(out, Path(__file__).with_name(name)))
    sources.append(snapshot(out, Path(__file__).parent / "tests"
                            / "test_trace_lineage_a_layer07_qk_producers.py"))
    tops = ("ace3_qkv_projection_cluster", "ace3_qwen2_rope_pair", "ace3_attention_score_core",
            "ace3_decoder_layer0_token_engine")
    commands = [["iverilog", "-g2012", "-s", top, "-o", str(out / f"{top}.vvp")]
                + [str(out / name) for name in rtl_names] for top in tops]
    test = ["env", f"PYTHONPATH={ROOT / 'ace3/model'}", sys.executable, "-B",
            "-m", "unittest", "discover", "-s", str(ROOT / "ace3/model/tests"),
            "-p", "test_trace_lineage_a_layer07_qk_producers.py"]
    write(out / "frozen.json", {
        "scope": "A-only layer07 retained Q/K and historical V diagnosis; no RTL replay",
        "token_history": HISTORY, "comparison_policy": POLICY, "heads": HEADS,
        "key_positions": [0, 1, 2, 3], "q_channels": [h*64+d for h in HEADS for d in range(64)],
        "k_channels": list(range(128)), "v_av_indices": AV_INDICES,
        "authenticated_inputs": list(inputs.records.values()), "sources": sources,
        "historical_reference_boundary": historical_boundary,
        "public_declarations": {top: (out / f"{top}.sv").read_text().split(");", 1)[0] + ");"
                                for top in tops},
        "compile_commands": commands, "test_command": test, "working_directory": str(ROOT),
        "python": sys.version, "numpy": np.__version__,
        "iverilog": subprocess.run(["iverilog", "-V"], capture_output=True,
                                   text=True, check=True).stdout,
        "producer_chain_order": PARTS,
        "projection": "Native G128 asymmetric GEMM nibble order, no zero+1, FP16 scales; "
                      "actual decoder Q/K single-round biased dot; V dot-round then bias",
        "decoder_single_round_producers": DECODER_SINGLE_ROUND_PRODUCERS,
        "superseded_diagnostic": {
            "path": str(superseded),
            "failure_taxonomy": "diagnostic_operator_boundary_misselection",
            "root_cause": "Attempt001 selected the standalone QKV cluster's legacy K mode "
                          "instead of the executed decoder's Q/K single-round bias mode. "
                          "Its 188 apparent K raw mismatches are not RTL defect evidence.",
            "regression": "Bind and compile the actual decoder and reconstruct all 512 K "
                          "outputs with its single-round bias mode; keep V two-rounding.",
        },
        "reference_recurrence": "Exact biased dot to FP16, binary64 mathematical half-split RoPE",
        "score_parts": "Q delta times reference K; reference Q times K delta; interaction. "
                       "Q48 dot numerators divided by 2**51 give exact scaled score terms.",
        "limits": ["No historical cache-read bus capture", "No incoming layer06 attribution",
                   "No binary64 admission", "No new precision or arithmetic adoption",
                   "No simulation, software activation injection, or repaired trajectory claim"],
    })
    for top, command in zip(tops, commands, strict=True):
        recorded_command(out, top + "_contract_compile", command)
    recorded_command(out, "unit_tests", test)

    producers = {}
    for kind, positions, heads, raw_stage, post_stage in (
        ("q", (3,), HEADS, 1, 4), ("k", range(4), range(2), 2, 6),
    ):
        t = tensors[kind]
        for position in positions:
            for head in heads:
                for dim in range(32):
                    channels = (head*64+dim, head*64+dim+32)
                    raw = [actual[position][raw_stage][c] for c in channels]
                    native, direct, ref_direct, rankings = [], [], [], []
                    for channel in channels:
                        local, single = projection(
                            actual[position][0], t, channel,
                            kind in DECODER_SINGLE_ROUND_PRODUCERS)
                        _, independent = projection(refs[position][0], t, channel, True)
                        native.append(local)
                        direct.append(single)
                        ref_direct.append(independent)
                        measured = native_projection_pair(actual[position][0], refs[position][0],
                                                          t, channel)
                        require(sum(row["delta_q48"] for row in measured["terms"])
                                == measured["actual_q48"]-measured["reference_q48"],
                                "stage00 projection contribution closure")
                        rankings.append([{
                            "input_index": row["index"], "group": row["group"],
                            "nibble_shift": row["nibble_shift"], "weight": row["weight"],
                            "zero": row["zero"], "scale_bits": row["scale_bits"],
                            "actual_stage00": row["actual_stage13_bits"],
                            "reference_stage00": row["reference_stage13_bits"],
                            "delta_q48": row["delta_q48"],
                        } for row in sorted(measured["terms"],
                                            key=lambda r: (-abs(r["delta_q48"]), r["index"]))[:8]])
                    coeff = coefficients[position][dim]
                    local_rope, quantized, mathematical = rotations(raw, coeff, position, dim)
                    native_rope = rotations(native, coeff, position, dim)[2]
                    direct_rope = rotations(direct, coeff, position, dim)[2]
                    ref_rope = rotations(ref_direct, coeff, position, dim)[2]
                    for side, channel in enumerate(channels):
                        a = actual[position][post_stage][channel]
                        r = reference[4][channel] if kind == "q" else kr[position][channel]
                        chain = [a, local_rope[side], quantized[side], mathematical[side],
                                 native_rope[side], direct_rope[side], ref_rope[side], r]
                        components = split(chain)
                        authoritative = dict(components)
                        if position < 2:
                            authoritative["incoming_or_reference_remainder"] = (
                                authoritative.pop("stage00_trajectory")
                                + authoritative.pop("reference_recurrence"))
                        producers[kind, position, channel] = {
                            "kind": kind, "position": position, "channel": channel,
                            "pair_channels": channels, "cos_sin_bits": [f"{b:04x}" for b in coeff],
                            "raw_actual": f"{raw[side]:04x}", "raw_local": f"{native[side]:04x}",
                            "raw_reference": f"{refs[position][raw_stage][channel]:04x}",
                            "raw_reference_reconstructed": f"{ref_direct[side]:04x}",
                            "actual": f"{a:04x}", "reference": f"{r:04x}",
                            "chain_bits": [f"{b:04x}" for b in chain],
                            "components_q24": authoritative,
                            "conditional_historical_components_q24": components if position < 2 else None,
                            "local_projection_delta_q24": units(raw[side])-units(native[side]),
                            "raw_reference_delta_q24": (
                                units(ref_direct[side])-units(refs[position][raw_stage][channel])),
                            "largest_stage00_dot_delta_terms": rankings[side],
                            "within_gate": accepts(a, r),
                        }

    scores = []
    for previous in parent["scores"]:
        head, position = previous["head"], previous["key_position"]
        qbegin, kbegin = head*64, head//7*64
        q, qr = actual[3][4][qbegin:qbegin+64], reference[4][qbegin:qbegin+64]
        k, rk = actual[position][6][kbegin:kbegin+64], kr[position][kbegin:kbegin+64]
        rows = score_terms(q, qr, k, rk)
        for row in rows:
            dim = row["dimension"]
            row["q_channel"], row["k_channel"] = qbegin+dim, kbegin+dim
            row["q_producer_parts_dot_q48"] = {
                name: value*units(rk[dim]) for name, value in
                producers["q", 3, qbegin+dim]["components_q24"].items()}
            row["k_producer_parts_dot_q48"] = {
                name: value*units(qr[dim]) for name, value in
                producers["k", position, kbegin+dim]["components_q24"].items()}
        totals = {name: sum(row[name] for row in rows)
                  for name in ("q_dot_q48", "k_dot_q48", "interaction_dot_q48")}
        require(totals["q_dot_q48"] == previous["q_trajectory_dot_q48"]
                and totals["k_dot_q48"] == previous["k_trajectory_dot_q48"]
                and totals["interaction_dot_q48"] == previous["interaction_dot_q48"],
                "ranked contributions differ from reviewed attempt004")
        total, local = score_bits(q, k)
        require(f"{local:04x}" == previous["actual"]
                and total == previous["actual_dot_q48"], "retained stage08 identity changed")
        aggregate = {}
        for kind in ("q", "k"):
            names = rows[0][kind + "_producer_parts_dot_q48"]
            aggregate[kind] = {name: sum(row[kind+"_producer_parts_dot_q48"][name] for row in rows)
                               for name in names}
            require(sum(aggregate[kind].values()) == totals[kind+"_dot_q48"],
                    "weighted producer parts do not close")
        scores.append({
            "head": head, "key_position": position, "totals_dot_q48": totals,
            "exact_scaled_totals": {key: str(Fraction(value, 1 << 51))
                                    for key, value in totals.items()},
            "producer_parts_dot_q48": aggregate,
            "producer_parts_scaled": {
                kind: {name: str(Fraction(value, 1 << 51)) for name, value in parts.items()}
                for kind, parts in aggregate.items()},
            "ranked_q_dimensions": sorted(rows, key=lambda r: (-abs(r["q_dot_q48"]), r["dimension"]))[:8],
            "ranked_k_dimensions": sorted(rows, key=lambda r: (-abs(r["k_dot_q48"]), r["dimension"]))[:8],
            "all_dimensions": rows,
        })

    vrows = []
    for previous in parent["v_producers"]:
        position, channel = previous["position"], previous["lane"]
        two, single, _ = v_projection(actual[position][0], tensors["v"], channel)
        _, independent, _ = v_projection(refs[position][0], tensors["v"], channel)
        a, r = actual[position][3][channel], vr[position][channel]
        parts = split([a, two, single, independent, r], (
            "local_v", "projection_bias_rounding", "stage00_trajectory", "reference_recurrence"))
        require(f"{a:04x}" == previous["actual"] and f"{r:04x}" == previous["reference"],
                "retained V producer changed")
        authoritative = dict(parts)
        if position < 2:
            authoritative["incoming_or_reference_recurrence"] = (
                authoritative.pop("stage00_trajectory") + authoritative.pop("reference_recurrence"))
        require(authoritative == previous["components_q24"], "V accounting changed")
        vrows.append({
            "av_index": previous["av_index"], "position": position, "lane": channel,
            "components_q24": authoritative,
            "conditional_historical_components_q24": parts if position < 2 else None,
            "reference_reconstructed": f"{independent:04x}",
        })
    rows = list(producers.values())
    local_errors = [r for r in rows if r["local_projection_delta_q24"]
                    or r["components_q24"]["local_rope"]]
    reference_errors = [r for r in rows if r["raw_reference_delta_q24"]
                        or (r["conditional_historical_components_q24"]
                            or r["components_q24"])["reference_recurrence"]]
    write(out / "measurements.json", {
        "producers": rows, "scores": scores, "v_producers": vrows,
        "historical_reference_boundary": historical_boundary,
    })
    summary = {
        "status": "RETAINED_QK_PRODUCER_BOUNDARY_NOT_REPAIRED",
        "q_coordinates": sum(r["kind"] == "q" for r in rows),
        "k_coordinates": sum(r["kind"] == "k" for r in rows),
        "scores": len(scores), "v_coordinates": len(vrows),
        "local_producer_discrepancies": local_errors,
        "reference_recurrence_discrepancies": reference_errors,
        "historical_reference_boundary": historical_boundary,
        "component_l1_q24": {
            kind: {name: sum(abs(r["components_q24"].get(name, 0)) for r in rows if r["kind"] == kind)
                   for name in (*PARTS, "incoming_or_reference_remainder")}
            for kind in ("q", "k")},
        "score_producer_parts_scaled": [
            {"head": row["head"], "key_position": row["key_position"],
             "parts": row["producer_parts_scaled"],
             "top_q_channels": [r["q_channel"] for r in row["ranked_q_dimensions"]],
             "top_k_channels": [r["k_channel"] for r in row["ranked_k_dimensions"]]}
            for row in scores],
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "hypothesis": "Declared projection/RoPE boundaries and incoming stage00 drift are "
                      "separated algebraically, not uniquely assigned to an upstream defect.",
        "regression": "704 native projection/RoPE identities, 12 exact parent score "
                      "decompositions, and 20 retained V component identities.",
        "next_boundary": "Use ranked Q/K channels and their exact weighted components to select "
                         "a general operator intervention or stage00 producer investigation. "
                         "Do not reopen cleared stage08/09 or AV/O arithmetic. Historical "
                         "positions0/1 still need a pre-existing source/manifest binding before "
                         "their conditional normalized-input decomposition is an authenticated split. "
                         "Incoming layer06 hidden attribution remains a separate branch.",
        "repair": None, "RTL_simulations": 0, "contract_compiles": len(tops),
        "binary64_evaluated": False, "review_status": "PENDING_NORMAL_HOST_REVIEWER",
    }
    write(out / "result.json", summary)
    for path in out.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps({key: summary[key] for key in
                      ("status", "q_coordinates", "k_coordinates", "scores", "v_coordinates")}))
    print(f"local_discrepancies={len(local_errors)} reference_discrepancies={len(reference_errors)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve())
