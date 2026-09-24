#!/usr/bin/env python3
"""Source-bound layer05 AV attribution on retained lineage A, without RTL replay."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback


ROOT = Path("/home/argustest/ace3-argus")
BASE = ROOT / "build/model24_selected_token_position3_continuations"
PARENT = BASE / "lineage_a_layer05_stage11_stage16_c939919c9508_attempt002"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
              "handoffs/c939919c9508/round-0001.json")
PREFIX = ROOT / "build/token358_position3_full_continuation_attempt003"
PINS = {
    "frozen.json": "41e9e4fee536cf84331514100adcc9994c59ddd7e9bb9fb43e0a73f438cf3d65",
    "result.json": "d37e7a8ec7780e376282e2e1f4c5038cc1a4f2d814a799c93132d8fd9a06cf6f",
}
REVIEW_PIN = "b5976a89c86bb17add370e2ff7cdd59510bc1825390fa43aa77343cb24844440"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    data = Path(path).read_bytes()
    return dict(path=str(Path(path).resolve()), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())


def write(path, value):
    with Path(path).open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def command(out, name, argv):
    text = shlex.join(argv) + "\n"
    with (out / (name + ".command.sh")).open("x", encoding="ascii") as stream:
        stream.write(text)
    with (out / (name + ".log")).open("xb") as stream:
        process = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                 cwd=ROOT, check=False)
    write(out / (name + ".status.json"), dict(returncode=process.returncode))
    require(process.returncode == 0, f"{name} failed; see its immutable log")
    return (out / (name + ".log")).read_text()


def bootstrap(out):
    require(out.parent == BASE and out.is_dir(), "pre-created in-scope attempt required")
    source_dir = out / "sources"
    source_dir.mkdir()
    model = ROOT / "ace3/model"
    pending = [
        model / Path(__file__).name,
        model / "diagnose_lineage_a_position3_boundary.py",
        model / "attention_oracle.py",
        model / "trace_lineage_a_layer07_attention_value.py",
        model / "trace_lineage_a_layer07_silu.py",
        model / "tests/test_trace_lineage_a_layer07_attention_value.py",
    ]
    sources, visited = [], set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        data = path.read_bytes()
        destination = source_dir / path.name
        with destination.open("xb") as stream:
            stream.write(data)
        sources.append(dict(origin=record(path), snapshot=record(destination)))
        for node in ast.walk(ast.parse(data)):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                candidate = model / (name.split(".")[0] + ".py")
                if candidate.is_file():
                    pending.append(candidate)
    for name in ("ace3_attention_value_core.sv", "ace3_fp16_fixed.sv",
                 "ace3_decoder_layer0_token_engine.sv", "ace3_fp16_kv_cache.sv"):
        origin, target = ROOT / "ace3/rtl" / name, source_dir / name
        with target.open("xb") as stream:
            stream.write(origin.read_bytes())
        sources.append(dict(origin=record(origin), snapshot=record(target)))
    write(out / "source_snapshots.json", sources)
    argv = [sys.executable, "-B", str(source_dir / Path(__file__).name),
            "--worker", "--out", str(out)]
    status = 1
    try:
        command(out, "measurement", argv)
        result = json.loads((out / "result.json").read_text())
        print(json.dumps({k: result[k] for k in (
            "status", "coordinates", "causal_product_terms", "local_AV_discrepancies",
            "ranked_coordinate_joins", "elapsed_seconds", "next_exact_upstream_boundary")},
            sort_keys=True))
        for position, summary in result["summaries"].items():
            print(json.dumps(dict(position=position, **{k: summary[k] for k in (
                "stage10_bit_differences", "stage10_material_failures",
                "stage09_probability_bit_differences", "V_bit_differences_by_causal_key",
                "component_nonzero_coordinates", "component_L1_q48")}), sort_keys=True))
        status = 0
    finally:
        write(out / "artifacts.json", [record(p) for p in sorted(out.rglob("*")) if p.is_file()])
        for p in out.rglob("*"):
            if p.is_file():
                p.chmod(0o444)
        source_dir.chmod(0o555)
        out.chmod(0o555)
    return status


def measure(out):
    sys.path.insert(0, str(out / "sources"))
    import bisect
    import numpy as np
    import diagnose_lineage_a_position3_boundary as boundary
    from attention_oracle import attention_value

    started = time.monotonic()
    boundary.ROOT, boundary.BUILD = ROOT, ROOT / "build"
    units, accepts = boundary.units, boundary.accepts
    source_dir = out / "sources"
    namespace = dict(require=require, units=units, accepts=accepts,
                     POSITIVE=boundary.POSITIVE, bisect=bisect, attention_value=attention_value)
    selected = {
        "trace_lineage_a_layer07_silu.py": ("rne_ratio",),
        "trace_lineage_a_layer07_attention_value.py": ("nearest_f16", "compose", "decompose"),
    }
    for filename, names in selected.items():
        tree = ast.parse((source_dir / filename).read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        require(len(nodes) == len(names), "missing frozen pure numerical helper")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    namespace["PARTS"] = (
        "probability", "historical_v", "current_v", "historical_interaction",
        "current_interaction", "av_rounding", "local_av", "reference_recurrence",
    )
    parts_order = (
        "probability", "historical_v", "current_v", "historical_interaction",
        "current_interaction", "av_q24_rounding", "av_fp16_rounding", "local_av",
        "reference_native_vs_direct_rounding", "reference_direct_recurrence",
    )
    inputs = boundary.Inputs()
    parent_roots = {}
    for name, pin in PINS.items():
        parent_roots[name] = inputs.load(PARENT / name, root=True)
        require(inputs.records[str(PARENT / name)]["sha256"] == pin, "reviewed root changed")
    review = inputs.load(REVIEW, root=True)
    require(inputs.records[str(REVIEW)]["sha256"] == REVIEW_PIN
            and review["producer_role"] == "reviewer"
            and review["mission_id"] == "c939919c9508"
            and review["review"]["status"] == "done", "genuine reviewed parent missing")
    prior = parent_roots["result.json"]
    retained = Path(prior["retained_numerical_attempt"]["path"]).parent
    parent_result = inputs.load(prior["retained_numerical_attempt"]["path"])
    pf = inputs.load(retained / "frozen.json")
    require(pf["token_history"] == boundary.HISTORY
            and pf["comparison_policy"] == boundary.POLICY, "foreign lineage/numerical policy")
    projections = inputs.load(retained / "projections.json")
    full = inputs.load(retained / "full_hidden_accounting.json")
    joins = inputs.load(retained / "ranked_joins.json")
    ranked = inputs.load(retained / "ranked_inputs.json")
    inherited_bytes = inputs.read(retained / "inherited_score_accounting.json")
    inherited = json.loads(inherited_bytes)
    require(inherited["positions01_status"] == "UNSPLIT_INHERITED_REMAINDER",
            "positions0/1 inherited remainder changed")
    rounding = inputs.load(PARENT / "native_contribution_rounding.json")
    norm6_record = next(r for r in pf["authenticated_inputs"] if r["path"].endswith("/rmsnorm.json"))
    norm6 = inputs.load(norm6_record["path"])
    coefficients = np.load(io.BytesIO(inputs.read(retained / "native_coefficients_q24.npz")),
                           allow_pickle=False)["self_attn.o_proj"]
    require(coefficients.shape == (896, 896) and coefficients.dtype == np.dtype("int64"),
            "native O coefficient geometry/dtype")
    factors = {
        p: np.load(io.BytesIO(inputs.read(PARENT / f"position{p}_factorized_inputs.npz")),
                   allow_pickle=False) for p in (2, 3)
    }
    inputs.load(PREFIX / "seal.json")
    traversal = inputs.load(PREFIX / "frozen.json")
    original = inputs.load(traversal["original_frozen"]["path"])
    policy = inputs.load(original["independent_policy"]["path"])
    require(policy["comparison"] == boundary.POLICY, "independent policy changed")
    prepared = inputs.load(PREFIX / "layer05/prepared.json")
    current = inputs.load(PREFIX / "layer05/result.json")
    kv = prepared["kv_parent"]
    history = inputs.load(kv["layer_result"]["path"])
    require(kv["layer_index"] == history["layer_index"] == current["layer"] == 5
            and kv["valid_positions"] == [0, 1, 2] and current["position"] == 3
            and kv["live_binary"] == prepared["binary"]
            and "model24_persistent_kv_selected_token_corrected_q_attempt005"
            in kv["state"]["path"], "foreign V-cache lineage/ABI")
    inputs.read(kv["state"]["path"])
    inputs.read(prepared["binary"]["path"])
    for rec in kv["position_states"]:
        inputs.read(rec["path"])
    actual = {
        p: boundary.decode_trace(inputs.read(rec["path"]), p)
        for p, rec in enumerate(kv["actual_kv_traces"])
    }
    current_trace = Path(current["output_hidden"]["path"]).with_name("trace.hex")
    inputs.register(dict(path=str(current_trace), sha256=current["exact"]["trace"]["actual_sha256"]))
    actual[3] = boundary.decode_trace(inputs.read(current_trace), 3)
    require(len(actual) == 4 and all(actual[p][3] == actual[p][7]
                                   and len(actual[p][7]) == 128 for p in range(4)),
            "V producer/cache-write or head geometry mismatch")
    ref_parent = inputs.load(prepared["independent_parent"]["path"])
    require(ref_parent["layer"] == 5 and ref_parent["history"] == boundary.HISTORY[:3]
            and ref_parent["own_cache"]["axes"] == ["position", "kv_head", "head_dim"],
            "independent history mismatch")
    tensor_identities = []

    def array(rec):
        inputs.register(dict(path=rec["path"], sha256=rec["file_sha256"]))
        data = inputs.read(rec["path"])
        value = np.load(io.BytesIO(data), allow_pickle=False)
        semantic = hashlib.sha256(value.tobytes()).hexdigest()
        require(value.dtype == np.dtype("<u2") and list(value.shape) == rec["shape"]
                and semantic == rec["semantic_sha256"], "reference tensor identity mismatch")
        tensor_identities.append(dict(file=inputs.records[rec["path"]],
                                      shape=list(value.shape), dtype=str(value.dtype),
                                      semantic_sha256=semantic))
        return value

    reference = {2: {}, 3: {}}
    sizes = {3: 128, 7: 128, 9: None, 10: 896}
    for p in (2, 3):
        for stage, size in sizes.items():
            size = 14 * (p + 1) if size is None else size
            if p == 2:
                matching = [r for r in ref_parent["stages"]
                            if r["path"].endswith(f"/stage{stage:02d}.npy")]
                require(len(matching) == 1, "missing/duplicate independent stage")
                bits = array(matching[0]).reshape(-1).tolist()
            else:
                rec = prepared["independent_stages"][str(stage)]
                data = inputs.read(rec["path"])
                lines = data.decode("ascii").splitlines()
                require(all(len(line) == 4 for line in lines), "FP16 hex framing")
                bits = [int(line, 16) for line in lines]
                semantic = hashlib.sha256(np.asarray(bits, dtype="<u2").tobytes()).hexdigest()
                require(rec.get("semantic_sha256", semantic) == semantic, "hex semantic mismatch")
                tensor_identities.append(dict(file=inputs.records[rec["path"]], shape=[len(bits)],
                                              dtype="uint16 FP16 bits", semantic_sha256=semantic))
            require(len(bits) == len(actual[p][stage]) == size, "stage geometry mismatch")
            reference[p][stage] = bits
        require(reference[p][3] == reference[p][7], "independent V/cache-write mismatch")
        require(factors[p]["stage10_actual"].tolist() == actual[p][10]
                and factors[p]["stage10_reference"].tolist() == reference[p][10],
                "reviewed stage10 coordinate join changed")
    vr = array(ref_parent["own_cache"]["v"])
    require(vr.shape == (3, 2, 64) and vr[2].reshape(-1).tolist() == reference[2][7],
            "position2 current V/history join mismatch")
    values_r = vr.reshape(3, 128).tolist() + [reference[3][7]]
    values_a = [actual[p][7] for p in range(4)]
    for vector in values_a + values_r:
        for bits in vector:
            units(bits)

    def source_bindings(value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= value.keys():
                yield value
            for child in value.values():
                yield from source_bindings(child)
        elif isinstance(value, list):
            for child in value:
                yield from source_bindings(child)

    historical_sources = list(source_bindings(original)) + list(source_bindings(pf))
    sources = json.loads((out / "source_snapshots.json").read_text())
    for item in sources:
        require(record(item["snapshot"]["path"]) == item["snapshot"], "snapshot changed")
        origin = item["origin"]["path"]
        if origin.endswith(".sv"):
            matches = [r for r in historical_sources if r["path"] == origin]
            require(matches and all(r["sha256"] == item["origin"]["sha256"] for r in matches),
                    "historical RTL source identity unavailable or changed: " + origin)
    controller = (source_dir / "ace3_decoder_layer0_token_engine.sv").read_text()
    for fragment in (
        ".probability_f16_i(probability_mem[key_position_q]),.value_f16_i(cache_v_w)",
        ".read_position_i({8'd0,key_position_q}),.read_head_i(mapped_kv_head_w)",
        ".write_v_f16_i(v_mem[kv_flat_index_w])",
        "probability_mem[context_index_q]<=sm_out_w",
        "attention_mem[q_flat_index_w]<=av_out_w",
        "(head_q<4'd7)?4'd0:4'd1",
    ):
        require(fragment in controller, "source-derived cache/AV wiring changed")
    compile_argv = ["iverilog", "-g2012", "-s", "ace3_attention_value_core",
                    "-o", str(out / "public_contract.vvp"),
                    str(source_dir / "ace3_fp16_fixed.sv"),
                    str(source_dir / "ace3_attention_value_core.sv")]
    version = command(out, "iverilog_version", ["iverilog", "-V"])
    write(out / "frozen.json", dict(
        scope="Layer05 positions2/3 lineage-A retained AV split; no model/RTL replay",
        parent_pins=PINS, parent_review=inputs.records[str(REVIEW)],
        token_history=boundary.HISTORY, comparison_policy=boundary.POLICY,
        independent_policy=policy, authenticated_inputs=list(inputs.records.values()),
        source_snapshots=sources, python=sys.version, numpy=np.__version__, iverilog=version,
        public_top="ace3_attention_value_core", parameters="Exact public defaults; no overrides",
        public_declaration=(source_dir / "ace3_attention_value_core.sv").read_text().split(");", 1)[0] + ");",
        compile_argv=compile_argv, selected_pure_functions=selected,
        coordinates={str(p): list(range(896)) for p in (2, 3)},
        causal_keys={"2": [0, 1, 2], "3": [0, 1, 2, 3]},
        component_names=parts_order, tensor_identities=tensor_identities,
        native_O_tensor_bindings=pf["tensor_bindings"],
        native_O_coefficient_semantic_sha256=hashlib.sha256(coefficients.tobytes()).hexdigest(),
        arithmetic="Exact Q48 products/sum -> RNE Q24 -> RNE FP16; positive exact zero",
        attribution="Exact bilinear reference-anchored P,V,interaction decomposition; "
                    "current/historical are relative to each query position. Through O all "
                    "896x896 contributions are factorized without top-N truncation. "
                    "Hidden/numerator/square telescopes are not nonlinear score interventions.",
        input_boundary="Authenticated V writes plus source-derived cache reads and bound serialized "
                       "state; no captured cache-read bus and no state import or execution.",
        positions01="UNSPLIT_INHERITED_REMAINDER; V operand measurements do not resolve upstream Q/K remainder",
        binary64_evaluated=False, numerical_policy_changed=False,
    ))
    command(out, "public_contract_compile", compile_argv)
    command(out, "helper_regression", [
        sys.executable, "-B", "-m", "unittest", "discover", "-s", str(source_dir),
        "-p", "test_trace_lineage_a_layer07_attention_value.py",
    ])
    phases = dict(authentication_freeze_compile_regression=time.monotonic() - started)
    numerical_start = time.monotonic()
    summaries, rows_by_position, outputs = {}, {}, {}
    new_joins = []
    for p in (2, 3):
        rows = []
        for index in range(896):
            head, dim = divmod(index, 64)
            lane = (head // 7) * 64 + dim
            pa = actual[p][9][head * (p + 1):(head + 1) * (p + 1)]
            pr = reference[p][9][head * (p + 1):(head + 1) * (p + 1)]
            va, rv = [x[lane] for x in values_a[:p + 1]], [x[lane] for x in values_r[:p + 1]]
            d = namespace["decompose"](pa, pr, va, rv, actual[p][10][index], reference[p][10][index])
            parts = d.pop("components_q48")
            total_delta = d["sums_q48"]["aa"] - d["sums_q48"]["rr"]
            q24_delta = (namespace["rne_ratio"](d["sums_q48"]["aa"], 1 << 24)
                         - namespace["rne_ratio"](d["sums_q48"]["rr"], 1 << 24))
            parts["av_q24_rounding"] = (q24_delta << 24) - total_delta
            parts["av_fp16_rounding"] = parts.pop("av_rounding") - parts["av_q24_rounding"]
            direct = int(d["direct_reference_rne"], 16)
            native_r = int(d["rounded"]["rr"], 16)
            parts["reference_native_vs_direct_rounding"] = (units(native_r) - units(direct)) << 24
            parts["reference_direct_recurrence"] = (units(direct) - units(reference[p][10][index])) << 24
            require(parts.pop("reference_recurrence") ==
                    parts["reference_native_vs_direct_rounding"] + parts["reference_direct_recurrence"],
                    "reference recurrence split does not close")
            require(set(parts) == set(parts_order)
                    and sum(parts.values()) == (units(actual[p][10][index]) - units(reference[p][10][index])) << 24,
                    "AV coordinate attribution does not close")
            rows.append(dict(index=index, head=head, kv_head=head // 7, dimension=dim,
                             probability_actual=[f"{b:04x}" for b in pa],
                             probability_reference=[f"{b:04x}" for b in pr],
                             v_actual=[f"{b:04x}" for b in va], v_reference=[f"{b:04x}" for b in rv],
                             actual=f"{actual[p][10][index]:04x}",
                             reference=f"{reference[p][10][index]:04x}",
                             components_q48=parts, **d))
        rows_by_position[p] = rows
        matrix = np.asarray([[row["components_q48"][key] for row in rows]
                             for key in parts_order], dtype=object)
        propagated = matrix @ coefficients.astype(object)
        o = projections[str(p)]["self_attn.o_proj"]
        delta_units = [units(a) - units(r) for a, r in zip(actual[p][10], reference[p][10], strict=True)]
        raw_delta = np.asarray(delta_units, dtype=object) @ coefficients.astype(object)
        require(raw_delta.tolist() == [a - r for a, r in zip(
            o["native_totals_q48"]["A"], o["native_totals_q48"]["R"], strict=True)],
            "complete native O coefficient/tensor identity join")
        priority = list(dict.fromkeys(j["input_index"] for j in joins if j["position"] == p))
        order = priority + [i for i in range(896) if i not in priority]
        output = [None] * 896
        for i in order:
            parent = full[str(p)][i]
            nr = norm6[str(p)]["all_coordinates"][i]
            require(parent["index"] == nr["index"] == i, "full denominator coordinate join")
            gamma = units(nr["weight"])
            endpoint_sum = units(nr["actual_hidden"]) + units(nr["reference_hidden"])
            parts = {key: int(propagated[j, i]) for j, key in enumerate(parts_order)}
            require(sum(parts.values()) == int(raw_delta[i]) << 24,
                    "all-factor O contribution closure")
            o_delta = o["components_q24"]["stage10_attention_value"][i]
            parts["O_rounding_difference"] = (o_delta << 48) - sum(parts.values())
            require(o_delta == units(o["native_bits"]["A"][i]) - units(o["native_bits"]["R"][i]),
                    "rounded O endpoints changed")
            orders = {}
            for name, old in parent["orders"].items():
                require(old["hidden_components_q48"]["O_stage10_attention_value"] == o_delta << 24,
                        "parent O/hidden join mismatch")
                hidden = {key: value << 24 for key, value in old["hidden_components_q48"].items()
                          if key != "O_stage10_attention_value"}
                hidden.update({"AV_" + key: value for key, value in parts.items()})
                numerator = {key: value * gamma for key, value in hidden.items()}
                square = {key: value * endpoint_sum for key, value in hidden.items()}
                for new_values, old_values in (
                    (hidden, old["hidden_components_q48"]),
                    (numerator, old["numerator_components_q72"]),
                    (square, old["square_components_q72"]),
                ):
                    require(sum(new_values.values()) == sum(old_values.values()) << 24,
                            "hidden/numerator/full denominator telescope does not close")
                require("incoming_layer04_stage18" in hidden
                        and hidden["incoming_layer04_stage18"] ==
                        old["hidden_components_q48"]["incoming_layer04_stage18"] << 24,
                        "separate residual branch lost")
                orders[name] = dict(hidden_components_q72=hidden,
                                    numerator_components_q96=numerator, square_components_q96=square)
            top_inputs = sorted(range(896), key=lambda j: -abs(delta_units[j] * int(coefficients[j, i])))[:16]
            output[i] = dict(index=i, retained_ranked=i in priority, orders=orders,
                             O_components_q72=parts, largest_AV_inputs=top_inputs)
        outputs[p] = output
        summaries[p] = dict(
            coordinates=len(rows), causal_product_terms=896 * (p + 1),
            local_AV_discrepancies=[r["index"] for r in rows if r["components_q48"]["local_av"]],
            stage10_bit_differences=sum(a != r for a, r in zip(actual[p][10], reference[p][10], strict=True)),
            stage10_material_failures=[i for i in range(896)
                                       if not accepts(actual[p][10][i], reference[p][10][i])],
            stage09_probability_bit_differences=sum(a != r for a, r in zip(
                actual[p][9], reference[p][9], strict=True)),
            V_bit_differences_by_causal_key=[sum(a != r for a, r in zip(
                values_a[k], values_r[k], strict=True)) for k in range(p + 1)],
            component_L1_q48={key: sum(abs(r["components_q48"][key]) for r in rows) for key in parts_order},
            component_nonzero_coordinates={key: sum(r["components_q48"][key] != 0 for r in rows)
                                           for key in parts_order},
            full_denominator_components_q96={
                order_name: {key: sum(row["orders"][order_name]["square_components_q96"][key]
                                     for row in output)
                             for key in output[0]["orders"][order_name]["square_components_q96"]}
                for order_name in output[0]["orders"]},
            retained_ranked_unique_coordinates=len(priority),
            reference_P_actual_V_material_failures=[r["index"] for r in rows if not r["v_only_within_gate"]],
            actual_P_reference_V_material_failures=[r["index"] for r in rows if not r["probability_only_within_gate"]],
        )
        with (out / f"position{p}_AV_factors.npz").open("xb") as stream:
            np.savez_compressed(stream, component_names=np.asarray(parts_order),
                                components_q48=np.asarray(matrix.tolist(), dtype=str),
                                coefficient_q24=coefficients,
                                input_indices=np.arange(896), output_indices=np.arange(896))
    for j in joins:
        p, i = j["position"], j["input_index"]
        rr = ranked[j["parent_row"]]
        entry = rr["retained_top_inputs_by_component"][j["component"]][j["rank"]]
        require(rr["position"] == p and rr["channel"] == j["channel"] and rr["kind"] == j["kind"]
                and entry["input_index"] == i and entry["coefficient_q24"] == j["coefficient_q24"]
                and outputs[p][i]["index"] == i, "retained ranked join mismatch")
        new_joins.append(dict(j, target=f"full_hidden_accounting.json/{p}/{i}",
                              inherited_target=j["target"]))
    require(len(new_joins) == prior["ranked_coordinate_joins"] == 35840,
            "retained/ranked coverage incomplete")
    for name, data in (("av_coordinates.json", rows_by_position),
                       ("full_hidden_accounting.json", outputs), ("ranked_joins.json", new_joins)):
        write(out / name, data)
    with (out / "inherited_score_accounting.json").open("xb") as stream:
        stream.write(inherited_bytes)
    phases["numerics_and_full_coordinate_accounting"] = time.monotonic() - numerical_start
    local = sum(len(s["local_AV_discrepancies"]) for s in summaries.values())
    result = dict(
        status="DIAGNOSTIC_COMPLETE_NO_RTL_REPLAY", review_status="PENDING_NORMAL_HOST_REVIEWER",
        summaries=summaries, coordinates=1792, causal_product_terms=6272,
        independent_integer_AV_comparisons=7168, full_O_weighted_coordinate_joins=2 * 896 * 896,
        ranked_coordinate_joins=len(new_joins), full_denominator_coordinates=1792,
        local_AV_discrepancies=local, positions01="UNSPLIT_INHERITED_REMAINDER_BYTE_IDENTICAL",
        residual_branch="Layer04/stage18 -> layer05/stage12 and inherited stage13/MLP terms remain separate",
        factorization="For EVERY AV input j and O output i, named components_q48[k,j] times "
                      "coefficient_q24[j,i] is exact Q72. Decimal integer strings in NPZ avoid "
                      "overflow/pickle. All outputs join both retained gate/up telescopes; "
                      "no component is promoted to an additive nonlinear score share.",
        failure_taxonomy="RETAINED_INDEPENDENT_FP16_TRAJECTORY_DRIFT",
        root_cause_hypothesis="Stage09 probability and stage03/cache V operand drift versus native "
                              "AV/reference rounding are discriminated; not a unique upstream producer claim.",
        regression="All AA/AR/RA/RR dots checked against independent integer AV oracle, all causal-key "
                   "bilinear identities, full O and hidden/numerator/denominator identities, all ranked joins",
        intervention_candidate_supported=False,
        next_exact_upstream_boundary=(
            "Resolve listed own-input layer05 AV discrepancies before intervention." if local else
            "Batch layer05 positions2/3 stage08 -> stage09 score/softmax discrimination with "
            "layer05 positions0-3 stage00 -> stage03 -> stage07 V producer/cache lineage. "
            "Use the measured nonzero per-key terms and retained ranked/full-denominator joins; "
            "retain layer04 stage18 -> layer05 stage12 as a separate upstream residual boundary. "
            "No supported general arithmetic repair follows from bilinear attribution alone."),
        contract_compiles=1, RTL_simulations=0, fresh_RTL_PASS=False,
        binary64_evaluated=False, numerical_policy_changed=False, production_source_edits=False,
        source_boundary="Frozen helper/RTL sources and parent tensors authenticated; conditional "
                        "reconstruction is not a cache-read bus capture or historical binary proof.",
        phase_seconds=phases, elapsed_seconds=time.monotonic() - started,
        parent_result=inputs.records[str(PARENT / "result.json")],
        artifacts={p.name: record(p) for p in sorted(out.iterdir())
                   if p.is_file() and p.name not in ("measurement.log", "result.json")},
    )
    write(out / "result.json", result)
    print(json.dumps({k: result[k] for k in (
        "status", "coordinates", "causal_product_terms", "local_AV_discrepancies",
        "ranked_coordinate_joins", "elapsed_seconds", "next_exact_upstream_boundary")}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        try:
            measure(args.out.resolve())
        except Exception as error:
            write(args.out / "failure.json", dict(
                taxonomy="DIAGNOSTIC_EXECUTION_OR_BINDING_FAILURE",
                root_cause_hypothesis=str(error),
                regression="Re-execute only a corrected fresh attempt; retain this failure",
                traceback=traceback.format_exc(), RTL_correctness_conclusion=None))
            raise
    else:
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.exit(bootstrap(args.out.resolve()))
