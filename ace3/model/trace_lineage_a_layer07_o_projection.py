#!/usr/bin/env python3
"""Authenticate and reconstruct the retained A-lineage attention/O boundary."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from diagnose_lineage_a_position3_boundary import (
    BUILD, HISTORY, POLICY, PREFIX, Inputs, accepts, decode_trace, require, units, write,
)
from trace_lineage_a_layer07_downproj import framed_hidden, hex_rows, residual
from trace_lineage_a_layer07_gate_up import native_projection_pair, oracle_pair, snapshot
from trace_lineage_a_layer07_silu import source_records

ROOT = BUILD.parent
BASE = BUILD / "model24_selected_token_position3_continuations"
PARENT = BASE / "lineage_a_layer07_rmsnorm_115829738bd1_attempt001"
REVIEWED = BASE / "lineage_a_layer07_rmsnorm_115829738bd1_attempt002"
PARENT_FREEZE = "a64d95bdde1665f3178cc842a7f685e9d2ea7f289ac08050f18c94d4c7615e20"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
              "115829738bd1/round-0002.json")


def projection_components(measured, actual, reference):
    """Exact additive accounting; a reference recurrence difference is not RTL error."""
    scale = 1 << 24
    inherited = measured["actual_q48"] - measured["reference_q48"]
    rounding = ((units(measured["actual_bits"]) - units(measured["reference_bits"]))
                * scale - inherited)
    local = (units(actual) - units(measured["actual_bits"])) * scale
    reference_recurrence = (units(measured["reference_bits"]) - units(reference)) * scale
    parts = {
        "attention_input_delta_q48": inherited,
        "projection_rounding_delta_q48": rounding,
        "local_projection_delta_q48": local,
        "reference_recurrence_delta_q48": reference_recurrence,
    }
    require(sum(parts.values()) == (units(actual) - units(reference)) * scale,
            "projection decomposition does not close")
    return parts


def residual_components(incoming_a, incoming_r, o_a, o_r, sum_a, sum_r):
    require(residual(incoming_a, o_a) == sum_a
            and residual(incoming_r, o_r) == sum_r, "residual reconstruction mismatch")
    incoming = units(incoming_a) - units(incoming_r)
    projection = units(o_a) - units(o_r)
    return {
        "incoming_hidden_delta_q24": incoming,
        "projection_delta_q24": projection,
        "residual_rounding_delta_q24": units(sum_a) - units(sum_r) - incoming - projection,
        "stage12_delta_q24": units(sum_a) - units(sum_r),
    }


def recorded_command(out, label, command):
    with (out / f"{label}.log").open("x", encoding="ascii") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f"{label} failed, exit {result.returncode}; retained log")


def run(out):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    inputs = Inputs()
    inputs.load(REVIEWED / "frozen.json", root=True)
    require(inputs.records[str(REVIEWED / "frozen.json")]["sha256"] == PARENT_FREEZE,
            "reviewed RMSNorm freeze changed")
    review = inputs.load(REVIEW, root=True)
    require(review["producer_role"] == "reviewer" and review["review"]["status"] == "done"
            and review["mission_id"] == "115829738bd1", "missing genuine parent review")
    parent = inputs.load(PARENT / "frozen.json")
    prior = inputs.load(PARENT / "result.json")
    residual_rows = inputs.load(PARENT / "residual_components.json")
    norm_rows = inputs.load(PARENT / "rmsnorm_coordinates.json")
    propagation = inputs.load(PARENT / "weighted_propagation.json")
    require(parent["token_history"] == prior["token_history"] == HISTORY
            and parent["comparison_policy"] == POLICY, "foreign lineage or policy")
    witnesses = prior["next_upstream_boundary"]["coordinates"]
    require(len(witnesses) == len(set(witnesses)) == 72
            and prior["next_upstream_boundary"]["required_input_coordinates"] == list(range(896)),
            "witness contract changed")
    gate_up = Path(propagation["parent_result"]["path"]).parent
    inputs.load(gate_up / "seal.json")
    gate_freeze = inputs.load(gate_up / "frozen.json")
    inputs.load(PREFIX / "seal.json")
    prefix = inputs.load(PREFIX / "frozen.json")
    original = inputs.load(prefix["original_frozen"]["path"])
    require(original["comparison_policy"] == POLICY, "original policy changed")

    prepared, results = {}, {}
    for layer in (6, 7, 8):
        inputs.load(PREFIX / f"layer{layer:02d}/seal.json")
        p = prepared[layer] = inputs.load(PREFIX / f"layer{layer:02d}/prepared.json")
        results[layer] = inputs.load(PREFIX / f"layer{layer:02d}/result.json")
        require(p["vectors"]["layer_index"] == layer and p["vectors"]["position"] == 3,
                "foreign layer/position")
        kv = p["kv_parent"]
        require(p["binary"] == kv["live_binary"] and kv["layer_index"] == layer
                and kv["valid_positions"] == [0, 1, 2]
                and "/model24_persistent_kv_selected_token_corrected_q_attempt005/"
                in kv["state"]["path"], "incompatible A-lineage KV/binary")
        # Reauthenticate the bounded binary/state ancestry, not an unrelated output census.
        for record in [p["binary"], kv["state"], *kv["actual_kv_traces"]]:
            inputs.read(record["path"])
        history = inputs.load(p["independent_parent"]["path"])
        require(history["history"] == HISTORY[:3] and history["layer"] == layer,
                "foreign independent history")
    for layer in (7, 8):
        require(all(prepared[layer]["input_hidden"][key]
                    == results[layer - 1]["output_hidden"][key]
                    for key in ("path", "bytes", "sha256")), "broken incoming-hidden lineage")
    p = prepared[7]
    a = decode_trace(inputs.read(Path(results[7]["output_hidden"]["path"])
                                .with_name("trace.hex")), 3)
    r = {int(stage): hex_rows(inputs.read(record["path"]))
         for stage, record in p["independent_stages"].items()}
    incoming_a = framed_hidden(inputs.read(p["input_hidden"]["path"]))
    incoming_r = hex_rows(inputs.read(prepared[6]["independent_stages"]["18"]["path"]))
    require(incoming_a == framed_hidden(inputs.read(p["vectors"]["input"]["path"])),
            "consumed layer07 hidden mismatch")
    require(a[18] == framed_hidden(inputs.read(results[7]["output_hidden"]["path"]))
            == framed_hidden(inputs.read(prepared[8]["vectors"]["input"]["path"])),
            "layer08 consumed-hidden mismatch")
    for vector in (incoming_a, incoming_r, a[10], r[10], a[11], r[11], a[12], r[12]):
        require(len(vector) == 896, "hidden/projection geometry mismatch")
        for bits in vector:
            units(bits)
    require(len(residual_rows) == len(norm_rows) == 896, "incomplete parent reduction records")
    for index in range(896):
        old = residual_rows[index]
        require(old["index"] == norm_rows[index]["index"] == index
                and [int(old[key], 16) for key in (
                    "incoming_actual", "incoming_reference", "stage11_actual", "stage11_reference",
                    "stage12_actual", "stage12_reference")]
                == [incoming_a[index], incoming_r[index], a[11][index], r[11][index],
                    a[12][index], r[12][index]], "parent residual coordinate mismatch")
        require(int(norm_rows[index]["actual_stage12"], 16) == a[12][index]
                and int(norm_rows[index]["reference_stage12"], 16) == r[12][index],
                "parent norm coordinate mismatch")

    tensors, tensor_bindings = {}, []
    geometry = {"qweight": ("I32", [896, 112]), "qzeros": ("I32", [7, 112]),
                "scales": ("F16", [7, 896])}
    for record in p["vectors"]["tensors"]:
        meta = record["checkpoint_tensor"]
        if not meta["name"].startswith("model.layers.7.self_attn.o_proj."):
            continue
        kind = meta["name"].rsplit(".", 1)[1]
        require(kind in geometry and kind not in tensors
                and (meta["dtype"], meta["shape"]) == geometry[kind], "O tensor contract mismatch")
        words = hex_rows(inputs.read(record["serialized"]["path"]))
        raw = b"".join(w.to_bytes(2 if kind == "scales" else 4, "little") for w in words)
        require(len(raw) == meta["bytes"] and hashlib.sha256(raw).hexdigest() == meta["sha256"],
                "serialized O tensor differs from official checkpoint")
        tensors[kind] = words
        tensor_bindings.append(record)
    require(set(tensors) == set(geometry), "missing official O tensors")

    rtl_names = [Path(arg).name for arg in gate_freeze["compile_command"] if arg.endswith(".sv")]
    rtl_names = list(dict.fromkeys(rtl_names + ["ace3_decoder_layer0_token_engine.sv"]))
    sources = []
    for name in rtl_names:
        path = ROOT / "ace3/rtl" / name
        record = snapshot(out, path)
        old = [s for s in source_records(original) if s["path"] == str(path)]
        require(old and all(s["sha256"] == record["sha256"] for s in old),
                f"historical RTL source changed: {name}")
        sources.append(record)
    helper_names = [
        Path(__file__).name, "diagnose_lineage_a_position3_boundary.py",
        "trace_lineage_a_layer07_downproj.py", "trace_lineage_a_layer07_gate_up.py",
        "trace_lineage_a_layer07_silu.py", "projection_oracle.py", "awq_bit_oracle.py",
        "fp16_adaptation_oracle.py",
    ]
    for name in helper_names:
        record = snapshot(out, Path(__file__).with_name(name))
        old = [s for s in parent["sources"] if s["path"] == record["path"]]
        require(all(s["sha256"] == record["sha256"] for s in old), "reviewed helper changed")
        sources.append(record)
    sources.append(snapshot(out, Path(__file__).parent / "tests"
                            / "test_trace_lineage_a_layer07_o_projection.py"))
    controller = (out / "ace3_decoder_layer0_token_engine.sv").read_text()
    wiring = [
        "TRACE_AV     = 5'd10, TRACE_O      = 5'd11",
        "o_pair_address_valid_w ? attention_mem[o_pair_in_w[9:0]] : 16'd0",
        "attention_mem[q_flat_index_w]<=av_out_w",
        "trace_stage_q<=TRACE_AV;trace_index_q<={3'd0,q_flat_index_w};trace_f16_q<=av_out_w",
        "o_mem[o_out_ch_w[9:0]]<=o_out_f16_w;trace_stage_q<=TRACE_O",
        ".IN_FEATURES(896),.OUT_FEATURES(896),.BIAS_ENABLE(0)) p_out",
        ".activation_f16_i(o_activation_w),.qweight_i(projection_qweight_i)",
    ]
    require(all(text in controller for text in wiring), "historical stage10/O wiring changed")
    top = "ace3_awq_w4a16_projection_engine"
    parameters = {"IN_FEATURES": 896, "OUT_FEATURES": 896, "BIAS_ENABLE": 0,
                  "SINGLE_ROUND_BIAS": 0}
    compile_command = ["iverilog", "-g2012", "-s", top,
                       *(f"-P{top}.{key}={value}" for key, value in parameters.items()),
                       "-o", str(out / "public_contract.vvp"),
                       *(str(out / name) for name in rtl_names
                         if name != "ace3_decoder_layer0_token_engine.sv")]
    regression = ["env", "PYTHONPATH=ace3/model", "PYTHONDONTWRITEBYTECODE=1", sys.executable,
                  "-m", "unittest", "discover", "-s", "ace3/model/tests",
                  "-p", "test_trace_lineage_a_layer07_o_projection.py", "-v"]
    version = subprocess.run(["iverilog", "-V"], capture_output=True, text=True, check=True)
    write(out / "frozen.json", {
        "attempt_id": out.name, "scope": "Retained A-only numerical reconstruction; no RTL replay",
        "reviewed_parent_freeze_sha256": PARENT_FREEZE, "review": inputs.records[str(REVIEW)],
        "authenticated_inputs": list(inputs.records.values()), "sources": sources,
        "token_history": HISTORY, "profile": original["profile"], "comparison_policy": POLICY,
        "independent_reference_policy": parent["independent_reference_policy"],
        "tensor_bindings": tensor_bindings, "public_top": top, "parameters": parameters,
        "public_declaration": (out / f"{top}.sv").read_text().split(");", 1)[0] + ");",
        "input_wiring": wiring,
        "input_evidence_boundary": "Stage10 trace is the source-bound attention_mem write; "
                                   "O inputs are derived from that memory wiring, not a new bus capture.",
        "coordinates": list(range(896)), "witnesses": witnesses,
        "arithmetic": "Native G128 GEMM nibble order; no qzero+1 or intermediate dequant rounding; "
                      "exact signed Q48 group/cross-group accumulation, final FP16 RNE.",
        "decomposition_order": ["attention_input", "projection_rounding", "local_projection",
                                "reference_recurrence", "incoming_hidden", "residual_rounding"],
        "compile_command": compile_command, "regression_command": regression,
        "measurement_command": [sys.executable, str(Path(__file__).resolve()), "--out", str(out)],
        "working_directory": str(ROOT), "python": sys.version,
        "iverilog_version": version.stdout + version.stderr,
    })
    recorded_command(out, "public_contract_compile", compile_command)
    recorded_command(out, "regression", regression)
    rows, local, recurrence, contributors = [], [], [], {}
    with (out / "witness_terms.csv").open("x", newline="", encoding="ascii") as stream:
        writer = None
        for channel in range(896):
            measured = native_projection_pair(a[10], r[10], tensors, channel)
            oracle = oracle_pair(a[10], r[10], tensors, channel)
            for label, check in zip(("actual", "reference"), oracle, strict=True):
                require(check == (measured[f"{label}_q48"], measured[f"{label}_bits"],
                                  False, False, measured[f"{label}_groups_q48"]),
                        f"independent projection oracle disagreement: {channel}/{label}")
            components = projection_components(measured, a[11][channel], r[11][channel])
            residual_parts = residual_components(
                incoming_a[channel], incoming_r[channel], a[11][channel], r[11][channel],
                a[12][channel], r[12][channel])
            if measured["actual_bits"] != a[11][channel]:
                local.append(channel)
            if measured["reference_bits"] != r[11][channel]:
                recurrence.append(channel)
            row = {
                "index": channel, "witness": channel in witnesses,
                "stage10_actual": f"{a[10][channel]:04x}",
                "stage10_reference": f"{r[10][channel]:04x}",
                "stage11_input_actual": f"{a[10][channel]:04x}",
                "stage11_input_reference": f"{r[10][channel]:04x}",
                "stage11_actual": f"{a[11][channel]:04x}",
                "stage11_reference": f"{r[11][channel]:04x}",
                "own_input_reconstruction": f"{measured['actual_bits']:04x}",
                "reference_input_reconstruction": f"{measured['reference_bits']:04x}",
                **{key: value for key, value in measured.items() if key.endswith("_q48")},
                **components, **residual_parts,
                "parent_residual": residual_rows[channel],
                "squared_residual_delta_q48": units(a[12][channel]) ** 2
                                             - units(r[12][channel]) ** 2,
                "stage10_within_gate": accepts(a[10][channel], r[10][channel]),
                "stage11_within_gate": accepts(a[11][channel], r[11][channel]),
            }
            require(row["squared_residual_delta_q48"]
                    == norm_rows[channel]["reduction_delta_q48"], "RMS reduction propagation mismatch")
            rows.append(row)
            if channel in witnesses:
                terms = measured["terms"]
                require(sum(t["delta_q48"] for t in terms)
                        == components["attention_input_delta_q48"], "weighted terms do not close")
                for term in terms:
                    term = dict(term)
                    term["actual_stage10_bits"] = term.pop("actual_stage13_bits")
                    term["reference_stage10_bits"] = term.pop("reference_stage13_bits")
                    record = {"channel": channel, **term}
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(record))
                        writer.writeheader()
                    writer.writerow(record)
                contributors[str(channel)] = [
                    {"input_index": term["index"], "head": term["index"] // 64,
                     "delta_q48": term["delta_q48"]}
                    for term in sorted(terms, key=lambda term: (-abs(term["delta_q48"]), term["index"]))[:16]
                ]
    write(out / "projection_coordinates.json", rows)
    # The already reviewed nonlinear downstream mapping is carried, not rerun or injected.
    for row in propagation["gate_up"]:
        expected_delta = Fraction(units(int(row["actual"]["bits"], 16))
                                  - units(int(row["reference"]["bits"], 16)), 1 << 24)
        require(Fraction(sum(row["component_deltas_q48"]), 1 << 48)
                + Fraction(row["rounding_and_local_delta"]) == expected_delta,
                "reviewed downstream weighted propagation does not close")
    a8 = decode_trace(inputs.read(Path(results[8]["output_hidden"]["path"])
                                 .with_name("trace.hex")), 3)
    r8 = hex_rows(inputs.read(prepared[8]["independent_stages"]["8"]["path"]))
    failed = [i for i, (x, y) in enumerate(zip(a8[8], r8, strict=True)) if not accepts(x, y)]
    require(failed == [13, 15], "retained layer08 failures changed")
    c = rows[5]
    down_delta = units(a[17][5]) - units(r[17][5])
    final_delta = units(a[18][5]) - units(r[18][5])
    require(residual(a[12][5], a[17][5]) == a[18][5]
            and residual(r[12][5], r[17][5]) == r[18][5], "channel5 final residual mismatch")
    channel5 = {
        "incoming_hidden": str(Fraction(c["incoming_hidden_delta_q24"], 1 << 24)),
        **{key: str(Fraction(c[key], 1 << 48)) for key in (
            "attention_input_delta_q48", "projection_rounding_delta_q48",
            "local_projection_delta_q48", "reference_recurrence_delta_q48")},
        "stage11_delta": str(Fraction(c["projection_delta_q24"], 1 << 24)),
        "stage12_rounding": str(Fraction(c["residual_rounding_delta_q24"], 1 << 24)),
        "stage12_delta": str(Fraction(c["stage12_delta_q24"], 1 << 24)),
        "stage17_delta": str(Fraction(down_delta, 1 << 24)),
        "stage18_rounding": str(Fraction(final_delta - c["stage12_delta_q24"] - down_delta, 1 << 24)),
        "stage18_delta": str(Fraction(final_delta, 1 << 24)),
        "stage18_actual": f"{a[18][5]:04x}", "stage18_reference": f"{r[18][5]:04x}",
    }
    write(out / "propagation.json", {
        "channel5": channel5, "all_coordinate_squared_residual_delta_q48":
        sum(row["squared_residual_delta_q48"] for row in rows),
        "reviewed_downstream": propagation, "layer08_actual_input_equals_stage18": True,
        "scope": "Exact boundary/weighted-term accounting, not an O-only causal score intervention.",
    })
    result = {
        "status": "LOCAL_PROJECTION_DISCREPANCY_NOT_REPAIRED" if local
                  else "RETAINED_O_PROJECTION_RECONSTRUCTED_NOT_REPAIRED",
        "authenticated_stage10_outputs": 896, "source_derived_stage11_inputs": 896,
        "reconstructed_actual_stage11_outputs": 896, "independent_oracle_outputs": 1792,
        "witnesses": witnesses, "witness_reduction_terms": len(witnesses) * 896,
        "actual_local_bit_differences": local, "reference_recurrence_bit_differences": recurrence,
        "trajectory_bit_differences": {
            "incoming_hidden": sum(x != y for x, y in zip(incoming_a, incoming_r, strict=True)),
            **{f"stage{s}": sum(x != y for x, y in zip(a[s], r[s], strict=True)) for s in (10, 11)},
        },
        "stage10_material_failure_indices": [row["index"] for row in rows if not row["stage10_within_gate"]],
        "stage11_material_failure_indices": [row["index"] for row in rows if not row["stage11_within_gate"]],
        "channel5": channel5,
        "original_layer08_failures": [
            {"index": i, "actual": f"{a8[8][i]:04x}", "reference": f"{r8[i]:04x}"}
            for i in failed],
        "next_upstream_boundary": {
            "producer": "layer07 stage09 probabilities and persistent V -> stage10 attention values",
            "coordinates": list(range(896)), "prioritized_input_contributors": contributors,
            "separate_branch": prior["next_upstream_boundary"]["separate_incoming_branch"],
            "question": "Reconstruct exact own-input AV sums; separate stage09 probability drift "
                        "from authenticated historical/current V drift, retaining the independent "
                        "layer06 hidden branch. Batch the connected score/softmax/value hypotheses.",
            "scope": "No unique upstream producer or causal repair established.",
        } if not local else None,
        "failure_taxonomy": "independently_propagated_FP16_trajectory_drift",
        "root_cause_hypothesis": "Stage10 operand drift propagates through exact O arithmetic; "
                                 "layer06 incoming hidden remains an independent additive branch."
                                 if not local else "Own-input local O reconstruction differs.",
        "regression": "All 896 exact own-input/reference-input projections, 64512 witness terms, "
                      "all residual/reduction identities and retained layer08 failures13/15.",
        "token_history": HISTORY, "comparison_policy": POLICY, "repair": None,
        "RTL_simulations": 0, "contract_compiles": 1, "binary64_evaluated": False,
        "third_token_selected": False, "independent_review": "pending normal Host Reviewer",
        "elapsed_seconds": time.monotonic() - started,
    }
    write(out / "result.json", result)
    write(out / "artifacts.json", [
        {"path": str(path), "bytes": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(out.iterdir()) if path.is_file()
    ])
    for path in out.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("next_upstream_boundary", "witnesses")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out.resolve())
