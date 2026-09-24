#!/usr/bin/env python3
"""Lossless retained-parent diagnosis; never supplies activations to execution."""

from __future__ import annotations

import argparse
import ast
import bisect
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import torch


ROOT = Path("/home/argustest/ace3-argus")
BASE = ROOT / "build/model24_selected_token_position3_continuations"
PARENT = BASE / "lineage_a_layer05_stage12_stage17_bb39b933f695_attempt002"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
              "handoffs/bb39b933f695/round-0001.json")
PREFIX = ROOT / "build/token358_position3_full_continuation_attempt003"
HIST = ROOT / "build/model24_persistent_kv_selected_token_corrected_q_attempt005"
REF = ROOT / "build/independent_fp16_trajectory_20260906_1133/recovery001"
PINS = {
    "frozen.json": "a85f9062084f94d8ec9f8b4ead55ea29fb082eaff78694e99380d44bfa2de1a0",
    "result.json": "745992de65041ffcd8a0a5f3db2ef9378e7884b348d555c20fc50b1ad0517385",
}
STAGES = (10, 11, 12, 13, 14, 15, 16, 17, 18)
NORM_PARTS = ("local_stage13", "norm_arithmetic_policy",
              "inherited_stage12", "reference_norm_recurrence")
PROJ_PARTS = ("local_projection", *NORM_PARTS, "reference_projection_recurrence")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path, data=None):
    path = Path(path).resolve()
    data = path.read_bytes() if data is None else data
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def write(path, value):
    with path.open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(out):
    started = time.monotonic()
    require(out.parent == BASE and not (out / "frozen.json").exists(),
            "a fresh in-scope attempt is required")
    torch.set_num_threads(2)
    source_dir = out / "sources"
    source_dir.mkdir()
    sources = []

    def snapshot(path, expected=None):
        path = Path(path)
        data = path.read_bytes()
        require(expected is None or hashlib.sha256(data).hexdigest() == expected,
                "source changed: " + str(path))
        target = source_dir / path.name
        with target.open("xb") as stream:
            stream.write(data)
        sources.append(dict(origin=record(path, data), snapshot=record(target)))
        return target

    parent_roots = {}
    for name, pin in PINS.items():
        data = (PARENT / name).read_bytes()
        require(hashlib.sha256(data).hexdigest() == pin, "reviewed parent pin mismatch")
        parent_roots[name] = json.loads(data)
    pf = parent_roots["frozen.json"]
    for item in pf["sources"]:
        snapshot(item["snapshot"]["path"], item["snapshot"]["sha256"])
    for name in ("trace_lineage_a_layer07_silu.py", "trace_lineage_a_layer07_rmsnorm.py",
                 "official_single_decoder_layer.py"):
        snapshot(ROOT / "ace3/model" / name)
    snapshot(PARENT / "measure.py",
             parent_roots["result.json"]["artifacts"]["measure.py"]["sha256"])
    snapshot(Path(__file__))
    for name in ("ace3_fp16_rmsnorm_core.sv", "ace3_fp16_silu_gate_core.sv",
                 "ace3_decoder_layer0_token_engine.sv"):
        snapshot(ROOT / "ace3/rtl" / name)
    sys.path.insert(0, str(source_dir))
    import diagnose_lineage_a_position3_boundary as boundary
    import fp16_adaptation_oracle as adaptation
    from projection_oracle import complete_projection_output

    boundary.ROOT, boundary.BUILD = ROOT, ROOT / "build"
    units = boundary.units
    inputs = boundary.Inputs()
    for name in PINS:
        inputs.load(PARENT / name, root=True)
    review = inputs.load(REVIEW, root=True)
    require(review["kind"] == "round_reviewed_handoff"
            and review["producer_role"] == "reviewer"
            and review["mission_id"] == "bb39b933f695"
            and review["review"]["status"] == "done", "independent parent review missing")
    require(pf["comparison_policy"] == boundary.POLICY
            and pf["token_history"] == boundary.HISTORY, "parent lineage/policy mismatch")
    parent_rows = inputs.load(PARENT / "producer_split.json")
    ranked = inputs.load(PARENT / "ranked_inputs.json")
    inherited = inputs.load(PARENT / "inherited_score_accounting.json")
    require(inherited["positions01_status"] == "UNSPLIT_INHERITED_REMAINDER",
            "positions0/1 must remain inherited")
    norm6 = inputs.load(next(r["path"] for r in pf["authenticated_inputs"]
                             if r["path"].endswith("/rmsnorm.json")))
    inputs.load(PREFIX / "seal.json")
    traversal = inputs.load(PREFIX / "frozen.json")
    original = inputs.load(traversal["original_frozen"]["path"])
    policy = inputs.load(original["independent_policy"]["path"])
    require(policy["comparison"] == boundary.POLICY
            and policy["silu"] == "binary64 silu(gate)*up then one FP16 boundary, as accepted helper",
            "reference policy changed")
    prepared = inputs.load(PREFIX / "layer05/prepared.json")
    p3 = inputs.load(PREFIX / "layer05/result.json")
    history = inputs.load(HIST / "layer05/result.json")
    p2 = history["positions"][2]
    ref5 = inputs.load(REF / "layer05_generation1.json")
    require(history["layer_index"] == p3["layer"] == ref5["layer"] == 5
            and p2["position"] == 2 and p3["position"] == 3
            and ref5["history"] == boundary.HISTORY[:3] and ref5["positions"] == [2],
            "layer/position/history mismatch")

    def bits_file(rec, size, npy=False):
        if npy:
            inputs.register(dict(path=rec["path"], sha256=rec["file_sha256"]))
            x = np.load(io.BytesIO(inputs.read(rec["path"])), allow_pickle=False)
            require(x.dtype == np.dtype("<u2") and list(x.shape) == rec["shape"] == [size],
                    "reference NPY dtype/geometry")
            bits = x.tolist()
        else:
            lines = inputs.read(rec["path"]).decode("ascii").splitlines()
            require(len(lines) == size and all(len(line) == 4 for line in lines),
                    "reference hex framing")
            bits = [int(line, 16) for line in lines]
        semantic = hashlib.sha256(np.asarray(bits, dtype="<u2").tobytes()).hexdigest()
        require(rec.get("semantic_sha256", semantic) == semantic, "semantic hash mismatch")
        for bit in bits:
            units(bit)
        return bits

    actual, reference = {}, {}
    for p, output, comparison in ((2, p2["output"], p2["exact_comparison"]["trace"]),
                                   (3, p3["output_hidden"], p3["exact"]["trace"])):
        path = Path(output["path"]).with_name("trace.hex")
        inputs.register(dict(path=str(path), sha256=comparison["actual_sha256"]))
        data = inputs.read(path)
        require(len(data.splitlines()) == comparison["actual_rows"], "trace row count")
        decoded = boundary.decode_trace(data, p)
        actual[p] = {s: decoded[s] for s in STAGES}
        reference[p] = {}
        for s in STAGES:
            size = 4864 if s in (14, 15, 16) else 896
            require(len(actual[p][s]) == size, "actual stage geometry")
            for bit in actual[p][s]:
                units(bit)
            if p == 2:
                records = [r for r in ref5["stages"] if r["path"].endswith(f"/stage{s:02d}.npy")]
                require(len(records) == 1, "independent stage coverage")
                reference[p][s] = bits_file(records[0], size, True)
            else:
                reference[p][s] = bits_file(prepared["independent_stages"][str(s)], size)
        for i, row in enumerate(parent_rows[str(p)]):
            require(row["index"] == i
                    and row["stage11"] == dict(actual=f"{actual[p][11][i]:04x}",
                                              reference=f"{reference[p][11][i]:04x}")
                    and row["stage12_cases"]["A"] == f"{actual[p][12][i]:04x}"
                    and row["stage12_cases"]["R"] == f"{reference[p][12][i]:04x}",
                    "reviewed producer coordinate join")
        retained = np.load(io.BytesIO(inputs.read(
            PARENT / f"position{p}_activation_contributions.npz")), allow_pickle=False)
        require(retained["actual_stage16"].tolist() == actual[p][16]
                and retained["reference_stage16"].tolist() == reference[p][16],
                "reviewed stage16 coordinate join")

    tensors, tensor_records = {}, []
    wanted = ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
              "post_attention_layernorm")
    for records in (p2["vectors"]["tensors"], prepared["vectors"]["tensors"]):
        current = {}
        for rec in records:
            meta = rec["checkpoint_tensor"]
            name = meta["name"].removeprefix("model.layers.5.")
            op, kind = name.rsplit(".", 1)
            if op not in wanted:
                continue
            lines = inputs.read(rec["serialized"]["path"]).decode("ascii").splitlines()
            width = 2 if meta["dtype"] == "F16" else 4
            require(meta["dtype"] in ("F16", "I32")
                    and all(len(line) == width * 2 for line in lines), "AWQ word framing")
            words = [int(line, 16) for line in lines]
            data = b"".join(word.to_bytes(width, "little") for word in words)
            require(len(data) == meta["bytes"]
                    and hashlib.sha256(data).hexdigest() == meta["sha256"]
                    and any(r["name"] == meta["name"] and r["sha256"] == meta["sha256"]
                            for r in ref5["checkpoint_tensor_hashes"]), "official tensor identity")
            current.setdefault(op, {})[kind] = words
            tensor_records.append(rec)
        require(set(current) == set(wanted), "required tensor coverage")
        require(not tensors or tensors == current, "position2/3 official tensor mismatch")
        tensors = current

    values = lambda bits: np.asarray(bits, dtype="<u2").view("<f2").astype(np.float64)
    differences = lambda left, right: [i for i, (a, b) in
                                        enumerate(zip(left, right, strict=True)) if a != b]
    namespace = dict(np=np, torch=torch, units=units, require=require, bisect=bisect,
                     Fraction=Fraction, Decimal=Decimal, POSITIVE=boundary.POSITIVE,
                     ONE=1 << 24, EPSILON_Q48=adaptation.EPSILON_Q48,
                     values=values, differences=differences)
    selected = {
        "trace_lineage_a_layer07_downproj.py": ("round_q48",),
        "trace_lineage_a_layer07_rmsnorm.py": ("floor_sqrt", "normalize", "reconstruct"),
        "official_single_decoder_layer.py": ("_torch_rmsnorm",),
        "trace_lineage_a_layer06_stage12_stage17_producers.py":
            ("exact_norm_round", "norm_boundary_measurement"),
        "trace_lineage_a_layer07_silu.py":
            ("rne_ratio", "accurate_silu", "mathematical_silu", "nearest_decimal", "decompose"),
        "measure.py": ("project_vectors",),
    }
    for name, names in selected.items():
        tree = ast.parse((source_dir / name).read_text(), filename=name)
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
        require(len(nodes) == len(names) and {n.name for n in nodes} == set(names),
                "selected pure helper coverage")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_dir / name), "exec"),
             namespace)

    priorities = {}
    for p in (2, 3):
        rows = parent_rows[str(p)]
        ranked_indices = [e["input_index"] for row in ranked if row["position"] == p
                          for entries in row["retained_top_inputs_by_component"].values()
                          for e in entries]
        denominator = sorted(range(896), key=lambda i:
                             -abs(sum(rows[i]["square_components_q48"].values())))
        channels = list(dict.fromkeys(denominator[:16] + ranked_indices))[:32]
        activation_indices = list(dict.fromkeys(term["input_index"] for i in channels
                                                for term in rows[i]["top_stage16_terms"]))[:32]
        require(len(channels) == len(activation_indices) == 32, "scalar priority coverage")
        priorities[p] = dict(output=channels, activation=activation_indices)
    tool = shutil.which("iverilog")
    require(tool is not None, "evaluator_no_execution: inspect declared local containers")
    rtl = [source_dir / name for name in (
        "ace3_fp16_fixed.sv", "ace3_awq_w4a16_g128_dot_lane.sv",
        "ace3_q47_48_to_f16_rne.sv", "ace3_awq_w4a16_projection_engine.sv",
        "ace3_fp16_rmsnorm_core.sv", "ace3_fp16_silu_gate_core.sv")]
    commands = []
    for size, label in ((896, "o"), (4864, "gate_up")):
        commands.append([tool, "-g2012", "-s", "ace3_awq_w4a16_projection_engine",
                         "-s", "ace3_fp16_rmsnorm_core", "-s", "ace3_fp16_silu_gate_core",
                         "-Pace3_awq_w4a16_projection_engine.IN_FEATURES=896",
                         f"-Pace3_awq_w4a16_projection_engine.OUT_FEATURES={size}",
                         "-Pace3_awq_w4a16_projection_engine.BIAS_ENABLE=0",
                         "-Pace3_awq_w4a16_projection_engine.SINGLE_ROUND_BIAS=0",
                         "-Pace3_fp16_rmsnorm_core.HIDDEN_SIZE=896",
                         "-Pace3_fp16_silu_gate_core.INTERMEDIATE_SIZE=4864",
                         "-Pace3_fp16_silu_gate_core.ACCURATE_SIGMOID=1",
                         "-o", str(out / f"public_{label}.vvp"), *map(str, rtl)])
    def command(argv, stem):
        with (out / f"{stem}.command.sh").open("x") as stream:
            stream.write(shlex.join(argv) + "\n")
        with (out / f"{stem}.log").open("x") as stream:
            proc = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        write(out / f"{stem}.status.json", dict(returncode=proc.returncode))
        require(proc.returncode == 0, "evaluator_no_execution: " + stem)

    command([tool, "-V"], "tool_version")
    freeze = dict(
        scope="Lineage A layer05 positions2/3 stage11 and stage13-stage16 retained diagnosis",
        parent_pins=PINS, parent_review=inputs.records[str(REVIEW)],
        authenticated_inputs=list(inputs.records.values()), sources=sources,
        program=record(__file__), launch=record(out / "run.sh"),
        python=sys.version, numpy=np.__version__, torch=torch.__version__,
        tool_version=record(out / "tool_version.log"), selected_pure_functions=selected,
        comparison_policy=boundary.POLICY, independent_policy=original["independent_policy"],
        token_history=boundary.HISTORY, tensor_bindings=tensor_records,
        projection_policy=pf["awq_policy"], scalar_priorities=priorities,
        coordinates={str(p): {str(s): len(actual[p][s]) for s in STAGES} for p in (2, 3)},
        compile_argv=commands,
        public_declarations={p.name: p.read_text().split("module ", 1)[1].split(");", 1)[0] + ");"
                             for p in rtl[-3:]},
        silu_policy="source Q24 exponential/sigmoid, Q72 product, Q24 RNE, FP16 RNE; "
                    "separate Decimal 80/100-digit mathematical diagnostics; no reference replacement",
        attribution="Two ordered gate/up paths, not unique causal shares. Lossless Q24 input "
                    "components times native Q24 coefficients define EVERY Q48 contribution. "
                    "Hidden Q48 times gamma/endpoint sum Q24 gives exact numerator/square Q72. "
                    "Full-denominator identity is not a nonlinear score intervention.",
        positions01="UNSPLIT_INHERITED_REMAINDER_BYTE_IDENTICAL",
        exclusions=["RTL replay", "reference injection", "binary64 evaluation", "new policy",
                    "trajectory admission", "unique root cause", "hardware"],
    )
    write(out / "frozen.json", freeze)
    for i, argv in enumerate(commands):
        command(argv, f"public_contract_{i}")
    command([sys.executable, "-B", "-m", "unittest", "discover", "-s", "ace3/model/tests",
             "-p", "test_trace_lineage_a_layer07_*.py"], "helper_regressions")
    phases = dict(authentication_freeze_compile_regressions=time.monotonic() - started)
    math_start = time.monotonic()

    def delta(left, right):
        return [units(a) - units(b) for a, b in zip(left, right, strict=True)]

    coefficients = {}
    for op in wanted[:-1]:
        t = tensors[op]
        require(set(t) == {"qweight", "qzeros", "scales"}, "unbiased native AWQ contract")
        n = 4864 if op == "mlp.down_proj" else 896
        m = 4864 if op in ("mlp.gate_proj", "mlp.up_proj") else 896
        ww = np.asarray(t["qweight"], dtype=np.uint32).reshape(n, m // 8)
        zz = np.asarray(t["qzeros"], dtype=np.uint32).reshape(n // 128, m // 8)
        scales = values(t["scales"]).reshape(n // 128, m)
        c = np.empty((n, m), dtype=np.float64)
        for lane, nibble in enumerate((0, 4, 1, 5, 2, 6, 3, 7)):
            w = ((ww >> (4 * nibble)) & 15).astype(np.int16)
            z = ((zz >> (4 * nibble)) & 15).astype(np.int16)
            c[:, lane::8] = (w - np.repeat(z, 128, axis=0)) * np.repeat(scales[:, lane::8], 128, axis=0)
        require(np.all(c * 2**24 == np.trunc(c * 2**24)), "inexact Q24 coefficients")
        coefficients[op] = (c * 2**24).astype(np.int64)
    with (out / "native_coefficients_q24.npz").open("xb") as stream:
        np.savez_compressed(stream, **coefficients)

    scalar_rows, full_oracles = [], 0

    def projection_pair(a, r, op, p, label):
        nonlocal full_oracles
        t, c = tensors[op], coefficients[op]
        bits, totals = namespace["project_vectors"](a, r, t)
        for kind, x in (("A", a), ("R", r)):
            independent = (values(x) @ (c.astype(np.float64) / 2**24)).astype("<f2").view("<u2").tolist()
            require(bits[kind] == independent, "full independent AWQ oracle disagreement")
            full_oracles += len(independent)
            for channel in priorities[p]["output" if op == "self_attn.o_proj" else "activation"]:
                words = c.shape[1] // 8
                total, b, invalid, saturated, groups = complete_projection_output(
                    x, t["qweight"][channel // 8::words], t["qzeros"][channel // 8::words],
                    t["scales"][channel::c.shape[1]], channel % 8)
                require((total, b, invalid, saturated) ==
                        (totals[kind][channel], bits[kind][channel], False, False),
                        "independent scalar native AWQ disagreement")
                scalar_rows.append(dict(position=p, op=op, case=label, operand=kind,
                                        channel=channel, total_q48=total, bits=b, groups_q48=groups))
        return bits, totals

    def silu(gates, ups):
        result = [namespace["accurate_silu"](g, u) for g, u in zip(gates, ups, strict=True)]
        require(all(adaptation.silu_gate_exp(g, u) == (row["bits"], False, False)
                    for g, u, row in zip(gates, ups, result, strict=True)),
                "independent integer SiLU disagreement")
        return [row["bits"] for row in result]

    summaries, refined_rows, norm_rows, projection_rows, silu_rows = {}, {}, {}, {}, {}
    for p in (2, 3):
        a, r = actual[p], reference[p]
        ob, ot = projection_pair(a[10], r[10], "self_attn.o_proj", p, "stage10")
        o_parts = dict(local_O=delta(a[11], ob["A"]), stage10_attention_value=delta(ob["A"], ob["R"]),
                       reference_O_recurrence=delta(ob["R"], r[11]))
        weights = tensors["post_attention_layernorm"]["weight"]
        native_norm, native_meta = namespace["reconstruct"](a[12], weights)
        independent_norm, mean, root = adaptation.rmsnorm(a[12], weights)
        require(independent_norm == [(b, False, False) for b in native_norm]
                and (mean, root) == (native_meta["mean_q48"], native_meta["rms_q24"]),
                "independent integer RMSNorm disagreement")
        math_a, norm_disagreements_a, math_meta_a = namespace["norm_boundary_measurement"](a[12], weights)
        math_r, norm_disagreements_r, math_meta_r = namespace["norm_boundary_measurement"](r[12], weights)
        norm_chain = [a[13], native_norm, math_a, math_r, r[13]]
        norm_parts = {key: delta(norm_chain[i], norm_chain[i + 1])
                      for i, key in enumerate(NORM_PARTS)}
        norm_rows[p] = dict(native=native_meta, mathematical_actual=math_meta_a,
                            mathematical_reference=math_meta_r,
                            independently_adjudicated_rounding_disagreements={
                                "actual": norm_disagreements_a, "reference": norm_disagreements_r},
                            chain_bits=norm_chain, components_q24=norm_parts,
                            stage12_denominator_components_q48={
                                key: sum(row["stage12_components_q24"][key]
                                         * (units(a[12][i]) + units(r[12][i]))
                                         for i, row in enumerate(parent_rows[str(p)]))
                                for key in parent_rows[str(p)][0]["stage12_components_q24"]})
        chains, projection_rows[p] = {}, {}
        for stage, op in ((14, "mlp.gate_proj"), (15, "mlp.up_proj")):
            projected, totals = [], []
            for i in range(0, 4, 2):
                pair, sums = projection_pair(norm_chain[i], norm_chain[i + 1], op, p, f"norm{i}/{i+1}")
                projected.extend([pair["A"], pair["R"]])
                totals.extend([sums["A"], sums["R"]])
            pair, sums = projection_pair(norm_chain[4], norm_chain[4], op, p, "reference_norm")
            projected.append(pair["A"])
            totals.append(sums["A"])
            chains[stage] = [a[stage], *projected, r[stage]]
            projection_rows[p][op] = dict(chain_bits=chains[stage], native_totals_q48=totals,
                                         components_q24={key: delta(chains[stage][i], chains[stage][i+1])
                                                         for i, key in enumerate(PROJ_PARTS)})
        projection_rows[p]["self_attn.o_proj"] = dict(native_bits=ob, native_totals_q48=ot,
                                                     components_q24=o_parts)
        g, u = chains[14], chains[15]
        gate_at_a = [silu(x, u[0]) for x in g]
        gate_at_r = [silu(x, u[-1]) for x in g]
        up_at_a = [silu(g[0], x) for x in u]
        up_at_r = [silu(g[-1], x) for x in u]
        require(gate_at_a[0] == up_at_a[0] and gate_at_r[-1] == up_at_r[-1],
                "gate/up factorial endpoint mismatch")
        math_bits, decimal_rows = [], []
        with localcontext() as ctx:
            ctx.prec = 80
            for i in range(4864):
                ideals, local_parts = {}, {}
                for label, gb, ub, retained in (
                        ("A", a[14][i], a[15][i], a[16][i]),
                        ("R", r[14][i], r[15][i], r[16][i])):
                    ideal = namespace["mathematical_silu"](gb, ub)
                    rec = namespace["accurate_silu"](gb, ub)
                    rounded = namespace["nearest_decimal"](ideal, (gb ^ ub) & 0x8000)
                    with localcontext() as higher:
                        higher.prec = 100
                        independent_ideal = namespace["mathematical_silu"](gb, ub)
                        require(rounded == namespace["nearest_decimal"](independent_ideal, (gb ^ ub) & 0x8000),
                                "unstable mathematical SiLU rounding")
                    ideals[label] = dict(value=str(ideal), fp16_bits=rounded)
                    local_parts[label] = namespace["decompose"](gb, ub, retained, rec, ideal)
                math_bits.append(ideals["R"]["fp16_bits"])
                decimal_rows.append(dict(index=i, mathematical=ideals, arithmetic_components=local_parts))
        common = dict(local_silu=delta(a[16], gate_at_a[0]),
                      reference_silu_arithmetic_policy=delta(gate_at_r[-1], math_bits),
                      reference_silu_recurrence=delta(math_bits, r[16]))
        orders = {}
        for order, gg, uu in (("gate_first", gate_at_a, up_at_r), ("up_first", gate_at_r, up_at_a)):
            parts = dict(common)
            for j, key in enumerate(PROJ_PARTS):
                parts["gate_" + key] = delta(gg[j], gg[j+1])
                parts["up_" + key] = delta(uu[j], uu[j+1])
            require([sum(v[i] for v in parts.values()) for i in range(4864)] == delta(a[16], r[16]),
                    "full stage16 ordered attribution closure")
            orders[order] = parts
        silu_rows[p] = dict(orders_q24=orders, per_coordinate=decimal_rows,
                            gate_first_vs_up_first="Both exact telescopes; differences are interaction, not unique shares")

        down = coefficients["mlp.down_proj"]
        dx = np.asarray(delta(a[16], r[16]), dtype=np.int64)
        require(max(map(abs, dx.tolist())) * int(np.max(np.abs(down))) < 2**63,
                "individual native contribution overflow")
        retained = np.load(io.BytesIO(inputs.read(
            PARENT / f"position{p}_activation_contributions.npz")), allow_pickle=False)
        require(np.array_equal(dx[:, None] * down, retained["delta_q48"]),
                "full 4864x896 reviewed contribution identity")
        sums_by_order = {
            order: {key: (np.asarray(vector, dtype=object) @ down.astype(object)).tolist()
                    for key, vector in parts.items()}
            for order, parts in orders.items()}
        refined_rows[p] = []
        for i, parent in enumerate(parent_rows[str(p)]):
            norm6_row = norm6[str(p)]["all_coordinates"][i]
            require(norm6_row["index"] == i and norm6_row["actual_hidden"] == a[18][i]
                    and norm6_row["reference_hidden"] == r[18][i], "layer06 full-denominator join")
            gamma, endpoint_sum = units(norm6_row["weight"]), units(a[18][i]) + units(r[18][i])
            orders_out = {}
            for order, sums in sums_by_order.items():
                parts = {key: value << 24 for key, value in parent["hidden_components_q24"].items()
                         if key not in ("stage11_O_projection", "stage16_activation")}
                parts.update({"O_" + key: vec[i] << 24 for key, vec in o_parts.items()})
                parts.update({"activation_" + key: vec[i] for key, vec in sums.items()})
                parts["down_rounding_difference"] = parent["down_rounding_delta_q48"]
                require(sum(parts.values()) == (units(a[18][i]) - units(r[18][i])) << 24,
                        "full hidden attribution closure")
                numerator = {key: value * gamma for key, value in parts.items()}
                squares = {key: value * endpoint_sum for key, value in parts.items()}
                require(sum(numerator.values()) == sum(parent["numerator_components_q48"].values()) << 24
                        and sum(squares.values()) == sum(parent["square_components_q48"].values()) << 24,
                        "full numerator/denominator closure")
                orders_out[order] = dict(hidden_components_q48=parts,
                                         numerator_components_q72=numerator, square_components_q72=squares)
            refined_rows[p].append(dict(index=i, orders=orders_out))
        with (out / f"position{p}_factorized_inputs.npz").open("xb") as stream:
            np.savez_compressed(stream, stage10_actual=np.asarray(a[10], dtype="<u2"),
                                stage10_reference=np.asarray(r[10], dtype="<u2"),
                                stage10_delta_q24=np.asarray(delta(a[10], r[10]), dtype=np.int64),
                                stage13_chain_bits=np.asarray(norm_chain, dtype="<u2"),
                                stage13_component_names=np.asarray(list(norm_parts)),
                                stage13_components_q24=np.asarray(list(norm_parts.values()), dtype=np.int64),
                                stage16_actual=np.asarray(a[16], dtype="<u2"),
                                stage16_reference=np.asarray(r[16], dtype="<u2"),
                                gate_first_component_names=np.asarray(list(orders["gate_first"])),
                                up_first_component_names=np.asarray(list(orders["up_first"])),
                                gate_first_components_q24=np.asarray(list(orders["gate_first"].values()), dtype=np.int64),
                                up_first_components_q24=np.asarray(list(orders["up_first"].values()), dtype=np.int64))
        summaries[p] = dict(
            stage_bit_differences={s: len(differences(a[s], r[s])) for s in STAGES},
            stage_material_failures={s: sum(not boundary.accepts(x, y)
                                            for x, y in zip(a[s], r[s], strict=True)) for s in STAGES},
            own_input_bit_differences={
                "stage11": len(differences(a[11], ob["A"])),
                "stage13": len(differences(a[13], native_norm)),
                "stage14": len(differences(a[14], g[1])),
                "stage15": len(differences(a[15], u[1])),
                "stage16": len(differences(a[16], gate_at_a[0]))},
            stage13_components_l1_q24={key: sum(map(abs, v)) for key, v in norm_parts.items()},
            stage16_components_l1_q24={order: {key: sum(map(abs, v)) for key, v in parts.items()}
                                      for order, parts in orders.items()},
            full_denominator_components_q72={
                order: {key: sum(row["orders"][order]["square_components_q72"][key] for row in refined_rows[p])
                        for key in refined_rows[p][0]["orders"][order]["square_components_q72"]}
                for order in orders})
    joins = []
    for row_index, row in enumerate(ranked):
        p = row["position"]
        require(p in (2, 3), "foreign ranked position")
        for component, entries in row["retained_top_inputs_by_component"].items():
            for rank, entry in enumerate(entries):
                i = entry["input_index"]
                require(0 <= i < 896 and refined_rows[p][i]["index"] == i, "ranked coordinate join")
                joins.append(dict(parent_row=row_index, component=component, rank=rank,
                                  position=p, input_index=i, channel=row["channel"], kind=row["kind"],
                                  coefficient_q24=entry["coefficient_q24"],
                                  target=f"full_hidden_accounting.json/{p}/{i}",
                                  warning="Exact endpoint attribution; no nonlinear score-share claim"))
    require(len(joins) == parent_roots["result.json"]["ranked_input_entries"], "rank join coverage")
    for name in ("ranked_inputs.json", "inherited_score_accounting.json"):
        with (out / name).open("xb") as stream:
            stream.write(inputs.read(PARENT / name))
    for name, data in (("stage13.json", norm_rows), ("projections.json", projection_rows),
                       ("stage16.json", silu_rows), ("full_hidden_accounting.json", refined_rows),
                       ("ranked_joins.json", joins), ("scalar_oracles.json", scalar_rows)):
        write(out / name, data)
    phases["numerics_full_accounting_and_artifacts"] = time.monotonic() - math_start
    local_differences = sum(sum(s["own_input_bit_differences"].values()) for s in summaries.values())
    result = dict(
        status="DIAGNOSTIC_COMPLETE_NO_RTL_REPLAY", review_status="PENDING_NORMAL_HOST_REVIEWER",
        positions=[2, 3], summaries=summaries, independent_full_projection_comparisons=full_oracles,
        independent_scalar_projection_comparisons=len(scalar_rows), output_coordinates=1792,
        stage16_coordinates=9728, full_retained_down_contributions_authenticated=2 * 4864 * 896,
        ranked_coordinate_joins=len(joins), positions01="UNSPLIT_INHERITED_REMAINDER_BYTE_IDENTICAL",
        factorization="native_coefficients_q24.npz[k][j,i] * positionP_factorized_inputs.npz "
                      "component[j] is the exact Q48 native contribution for EVERY coordinate; "
                      "explicit component_names arrays define row order and join to JSON keys. "
                      "No top-N truncation.",
        failure_taxonomy="RETAINED_TRAJECTORY_DRIFT_DIAGNOSTIC",
        root_cause_hypothesis="Inherited stage10 and stage12 drift versus local operator differences "
                              "are separated, not asserted to be uniquely causal for downstream score failures.",
        regression="Independent integer RMSNorm/SiLU, full matrix and scalar AWQ oracles, "
                   "both gate/up telescopes, exact all-coordinate hidden/numerator/square closure",
        intervention_candidate_supported=False,
        next_exact_upstream_boundary=(
            "Resolve nonzero own-input discrepancies at the explicitly listed stage/position coordinates "
            "before choosing a source change; no unchanged RTL replay." if local_differences else
            "Layer05 positions2/3 stage10: split stage09 softmax and stage07 V/cache weighted accumulation, "
            "including all causal keys and native AV rounding. In parallel with that connected split, "
            "stage12 inherited RMSNorm input remains the reviewed layer04 stage18 plus layer05 stage11 "
            "residual branch; retain full layer06 denominator coordinates and positions0/1 remainder. "
            "No general repair is supported merely by these exact conditional reconstructions."),
        source_boundary="Consumed trace and tensor bytes are authenticated to reviewed parents. "
                        "Reconstruction is conditional on frozen operator snapshots; compile is not simulation "
                        "or proof of historical binary equivalence.",
        RTL_simulations=0, fresh_RTL_PASS=False, binary64_evaluated=False,
        numerical_policy_changed=False, production_source_edits=False,
        phase_seconds=phases, elapsed_seconds=time.monotonic() - started,
        artifacts={p.name: record(p) for p in sorted(out.iterdir())
                   if p.is_file() and p.name not in ("measurement.log", "run.sh")})
    write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in (
        "status", "elapsed_seconds", "independent_full_projection_comparisons",
        "independent_scalar_projection_comparisons", "ranked_coordinate_joins",
        "next_exact_upstream_boundary")}, sort_keys=True))
    print(json.dumps({str(p): summaries[p]["own_input_bit_differences"] for p in (2, 3)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.out.resolve())
