#!/usr/bin/env python3
"""Retained A-lineage upstream accounting; never executes a model or RTL."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path("/home/argustest/ace3-argus")
PARENT = ROOT / "build/layer04_stage10_layer03_branches_948b8acd97ba_attempt002"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/948b8acd97ba")
OUT = ROOT / "build/retained_layer04_softmax_v_layer03_residual_gateup_3e750691b915_attempt002"
PREFIX = ROOT / "build/token358_position3_full_continuation_attempt003"
ROOTS = {
    "frozen.json": "104af0e6fd7dd53de590d1c6f5c9c324defc0090a2d4aae5af3111e287d8930c",
    "result.json": "bb8001f303ebac7098de3210f26a41a1b20fec5da4b5f1bf8cd1eec3be319fdc",
}
QUALIFICATION = (
    "Positions0/1 retained reference bytes and cache prefixes are content-bound, "
    "not original-reference-input provenance authenticated. Historical dependence "
    "on them remains conditional at positions2/3. Cache joins are source-derived, "
    "not captured read-bus observations. Newly read layer02 reference roots are "
    "identified separately, not promoted to original execution provenance."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def write(name, value):
    with (OUT / name).open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def command(name, argv):
    with (OUT / (name + ".command.sh")).open("x", encoding="ascii") as stream:
        stream.write(shlex.join(argv) + "\n")
    start = time.monotonic()
    with (OUT / (name + ".log")).open("xb") as stream:
        try:
            result = subprocess.run(argv, cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=110, check=False)
        except subprocess.TimeoutExpired:
            write(name + ".status.json", dict(returncode=None, timeout=110,
                                              seconds=time.monotonic() - start))
            raise
    write(name + ".status.json", dict(returncode=result.returncode,
                                      seconds=time.monotonic() - start))
    require(result.returncode == 0, name + " failed; inspect retained inner log")


def bootstrap():
    require({p.name for p in OUT.iterdir()} == {"launch.command.sh"},
            "fresh attempt required; no overwrite")
    receipt = json.loads((REVIEW / "round-0001.json").read_text())
    checkpoint = (REVIEW / "CHECKPOINT.md").read_text()
    require(receipt["kind"] == "round_reviewed_handoff"
            and receipt["producer_role"] == "reviewer"
            and receipt["mission_id"] == "948b8acd97ba"
            and receipt["review"]["status"] == "done"
            and "stage09" in receipt["review"]["next_action"],
            "genuine mission-scoped reviewed parent required")
    for name, digest in ROOTS.items():
        require(record(PARENT / name)["sha256"] == digest and digest in checkpoint
                and str(PARENT) in checkpoint, "parent reviewed root binding")
    manifest = json.loads((PARENT / "artifacts.json").read_text())
    for rec in manifest:
        require(record(rec["path"]) == rec, "parent artifact changed: " + rec["path"])
    write("parent_authentication.json", dict(
        roots={name: record(PARENT / name) for name in ROOTS},
        manifest=record(PARENT / "artifacts.json"), artifacts_authenticated=len(manifest),
        review=record(REVIEW / "round-0001.json"), checkpoint=record(REVIEW / "CHECKPOINT.md"),
        mission=record(REVIEW / "mission.json"), historical_pending_wording_is_not_missing_review=True))
    failed = OUT.with_name("retained_layer04_softmax_v_layer03_residual_gateup_3e750691b915_attempt001")
    write("repair_basis.json", dict(
        failure=record(failed / "failure.json"), source=record(failed / "failed_diagnostic.py"),
        command=record(failed / "launch.command.sh"),
        failure_taxonomy="diagnostic_python_parse_error",
        root_cause_hypothesis="Missing closing parenthesis on residual coordinate row.update.",
        intervention="Close the call; no mathematical recurrence, operands, or acceptance changes.",
        regression="Fresh parse and full frozen coordinate accounting with independent oracles."))
    sources = OUT / "sources"
    sources.mkdir()
    snapshots = []
    for path in sorted((PARENT / "sources").iterdir()):
        if path.suffix not in (".py", ".sv"):
            continue
        require(any(rec["path"] == str(path) and rec == record(path) for rec in manifest),
                "unmanifested parent source")
        target = sources / path.name
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        snapshots.append(dict(origin=record(path), snapshot=record(target)))
    original_path = ROOT / "build/token358_position3_full_continuation_attempt001/frozen.json"
    require(record(original_path)["sha256"] ==
            "f81181054c32a22ac303af8416f2b41e733ea265b90cdcba9c71201757d6b599",
            "historical source freeze changed")
    original = json.loads(original_path.read_text())

    def records(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                yield value
            for child in value.values():
                yield from records(child)
        elif isinstance(value, list):
            for child in value:
                yield from records(child)

    historical = list(records(original))
    for path in (Path(__file__).resolve(), ROOT / "ace3/rtl/ace3_attention_softmax_core.sv"):
        rec = record(path)
        if path.suffix == ".sv":
            matches = [r for r in historical if r["path"] == str(path)]
            require(matches and all(r["sha256"] == rec["sha256"] for r in matches),
                    "softmax historical source mismatch")
        target = sources / path.name
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        snapshots.append(dict(origin=rec, snapshot=record(target)))
    write("source_snapshots.json", snapshots)
    command("measurement", [sys.executable, "-B", str(sources / Path(__file__).name), "--measure"])


def measure():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(OUT / "sources"))
    import numpy as np
    import torch
    from decimal import localcontext
    from fractions import Fraction
    import diagnose_lineage_a_position3_boundary as boundary
    from fp16_adaptation_oracle import residual_add, silu_gate_exp
    from exact_projection import project_vectors
    from trace_lineage_a_layer07_downproj import framed_hidden, hex_rows, round_q48
    from trace_lineage_a_layer07_score_softmax_v import array, softmax_components, v_projection
    from trace_lineage_a_layer07_silu import accurate_silu, mathematical_silu, nearest_decimal

    start = time.monotonic()
    torch.set_num_threads(1)
    boundary.ROOT, boundary.BUILD, boundary.PREFIX = ROOT, ROOT / "build", PREFIX
    units, accepts = boundary.units, boundary.accepts
    inputs = boundary.Inputs()
    inputs.load(PARENT / "artifacts.json", root=True)
    parent_freeze = inputs.load(PARENT / "frozen.json")
    require(parent_freeze["comparison_policy"] == boundary.POLICY
            and parent_freeze["token_history"] == boundary.HISTORY
            and inputs.load(PARENT / "result.json")["exact_accounting_closed"],
            "parent policy/history/accounting mismatch")
    supplemental = []

    def load(path):
        path = Path(path).resolve()
        fresh = str(path) not in inputs.bindings
        value = inputs.load(path, root=fresh)
        if fresh:
            supplemental.append(inputs.records[str(path)])
        return value

    def from_array(rec):
        return array(inputs, rec).reshape(-1).tolist()

    def from_hex(rec, framed=False):
        inputs.register(rec)
        bits = (framed_hidden if framed else hex_rows)(inputs.read(rec["path"]))
        digest = hashlib.sha256(np.asarray(bits, dtype="<u2").tobytes()).hexdigest()
        require(rec.get("semantic_sha256", digest) == digest, "FP16 semantic identity")
        return bits

    def reference_history(layer, pre, stages):
        history = load(pre["independent_parent"]["path"])
        early = load(Path(pre["independent_parent"]["path"]).with_name(f"layer{layer:02d}_generation0.json"))
        require(history["history"] == boundary.HISTORY[:3] and history["positions"] == [2]
                and early["history"] == boundary.HISTORY[:2] and early["positions"] == [0, 1]
                and history["checkpoint_tensor_hashes"] == early["checkpoint_tensor_hashes"],
                "independent retained history mismatch")
        result = {p: {} for p in range(4)}
        for rec in early["stages"]:
            stage = int(Path(rec["path"]).stem[5:])
            if stage in stages:
                position = int(Path(rec["path"]).parent.name[8:])
                result[position][stage] = from_array(rec)
        for stage in stages:
            result[2][stage] = from_array(next(rec for rec in history["stages"]
                                              if rec["path"].endswith(f"/stage{stage:02d}.npy")))
            result[3][stage] = from_hex(pre["independent_stages"][str(stage)])
        for name in ("k", "v"):
            require(from_array(early["own_cache"][name]) == from_array(history["own_cache"][name])[:256],
                    "conditional historical prefix changed")
        return result, history

    actual, reference, prepared, histories, incoming = {}, {}, {}, {}, {}
    for layer, stages in ((3, (11, 12, 14, 15, 16, 17)), (4, (0, 3, 7, 8, 9))):
        pre = prepared[layer] = load(PREFIX / f"layer{layer:02d}/prepared.json")
        current = load(PREFIX / f"layer{layer:02d}/result.json")
        kv = pre["kv_parent"]
        old = load(kv["layer_result"]["path"])
        require(current["layer"] == old["layer_index"] == kv["layer_index"] == layer
                and current["position"] == 3 and kv["valid_positions"] == [0, 1, 2]
                and pre["binary"] == kv["live_binary"]
                and "/model24_persistent_kv_selected_token_corrected_q_attempt005/" in kv["state"]["path"],
                "A-lineage binary/state ABI mismatch")
        for rec in (pre["binary"], kv["state"], *kv["position_states"]):
            inputs.read(rec["path"])
        actual[layer] = {p: boundary.decode_trace(inputs.read(rec["path"]), p)
                         for p, rec in enumerate(kv["actual_kv_traces"])}
        trace = Path(current["output_hidden"]["path"]).with_name("trace.hex")
        inputs.register(dict(path=str(trace), sha256=current["exact"]["trace"]["actual_sha256"]))
        actual[layer][3] = boundary.decode_trace(inputs.read(trace), 3)
        reference[layer], histories[layer] = reference_history(layer, pre, stages)
        incoming[layer] = [from_hex(old["positions"][p]["vectors"]["input"] if p < 3
                                   else pre["input_hidden"], True) for p in range(4)]
    p2 = load(ROOT / "build/token358_position3_full_continuation_attempt001/layer02/prepared.json")
    r2, _ = reference_history(2, p2, (18,))
    parent_branches = {p: inputs.load(PARENT / f"position{p}_layer03_branch_coordinates.json")
                       for p in range(4)}
    parent_av = {p: inputs.load(PARENT / f"position{p}_av_coordinates.json") for p in range(4)}
    parent_terms = inputs.load(PARENT / "av_causal_terms.json")
    parent_keys = {(r["position"], r["av_index"], r["key_position"]): r["parent_join"]
                   for r in parent_terms}
    require(len(parent_keys) == len(parent_terms) == 8960, "parent causal key uniqueness")
    tensors, tensor_records = {}, []
    for rec in prepared[4]["vectors"]["tensors"]:
        meta = rec["checkpoint_tensor"]
        if not meta["name"].startswith("model.layers.4.self_attn.v_proj."):
            continue
        vals = hex_rows(inputs.read(rec["serialized"]["path"]))
        data = b"".join(v.to_bytes(2 if meta["dtype"] == "F16" else 4, "little") for v in vals)
        require(len(data) == meta["bytes"] and hashlib.sha256(data).hexdigest() == meta["sha256"]
                and any(r["name"] == meta["name"] and r["sha256"] == meta["sha256"]
                        for r in histories[4]["checkpoint_tensor_hashes"]), "official V tensor binding")
        tensors[meta["name"].rsplit(".", 1)[1]] = vals
        tensor_records.append(rec)
    require(set(tensors) == {"qweight", "qzeros", "scales", "bias"}, "native biased V contract")
    tool = shutil.which("iverilog")
    require(tool is not None, "host iverilog missing; inspect declared local containers before retry")
    command("tool_versions", [sys.executable, "-B", "-c",
            "import sys,numpy,torch;print(sys.version);print(numpy.__version__);print(torch.__version__)"])
    command("iverilog_version", [tool, "-V"])
    roots = ("ace3_attention_softmax_core", "ace3_awq_w4a16_projection_engine",
             "ace3_fp16_residual_add_core", "ace3_fp16_silu_gate_core", "ace3_fp16_kv_cache")
    write("frozen.json", dict(
        parent=record(PARENT / "frozen.json"), evaluator=record(Path(__file__)),
        source_snapshots=record(OUT / "source_snapshots.json"), comparison_policy=boundary.POLICY,
        token_history=boundary.HISTORY, positions=[0, 1, 2, 3],
        authenticated_inputs=list(inputs.records.values()), supplemental_content_roots=supplemental,
        checkpoint_tensors=tensor_records, public_tops=list(roots), parameter_policy="exact source defaults",
        case_counts=dict(softmax=140, v=512, residual=3584, gate_up=19456, causal_v_reads=8960),
        numerical_policy="Native G128 asymmetric INT4 GEMM lanes 0,4,1,5,2,6,3,7; no qzero +1. "
        "FP16 scales/activations/KV. Exact Q48 dot -> FP16 -> bias FP16. Softmax integer Q24 "
        "exp/normalization versus independently rounded Decimal80 and binary64 exp. Residual "
        "Q24 add -> FP16. SiLU polynomial Q24 sigmoid; Q72 product -> Q24 RNE -> FP16 RNE.",
        reference_policy="Retained independently propagated FP16-interstage arrays; Decimal80 ideal "
        "SiLU plus separately measured torch binary64 SiLU recurrence. No re-anchoring or injection. "
        "Binary64 layer-final v1 and legacy gates are not evaluated or changed.",
        provenance=QUALIFICATION, original_positions0_1_reference_provenance_authenticated=False,
        validation="one execution; existing helper tests plus every-coordinate exact telescoping, "
        "independent native V oracle for both operands, independent SiLU/residual integer oracles",
        RTL_simulations=0))
    argv = [tool, "-g2012"]
    for name in roots:
        argv.extend(("-s", name))
    command("public_contract_compile", argv + ["-o", str(OUT / "public_contracts.vvp")]
            + [str(p) for p in sorted((OUT / "sources").glob("*.sv"))])
    command("existing_helper_regression", [sys.executable, "-B", "-m", "unittest", "discover",
            "-s", str(OUT / "sources"), "-p", "test_trace_lineage_a_layer07_silu.py"])
    phases = dict(authentication_freeze_compile_seconds=time.monotonic() - start)
    summaries = {}
    upstream = {}

    def split(chain, names):
        require(len(chain) == len(names) + 1, "split geometry")
        result = {name: units(chain[i]) - units(chain[i + 1]) for i, name in enumerate(names)}
        require(sum(result.values()) == units(chain[0]) - units(chain[-1]), "accounting closure")
        return result

    def summary(rows, local_key="local_reconstructed"):
        return dict(coordinates=len(rows),
                    bit_differences=sum(r["actual"] != r["reference"] for r in rows),
                    material_failures=sum(not r["within_gate"] for r in rows),
                    own_input_local_discrepancies=sum(r["actual"] != r[local_key] for r in rows),
                    component_nonzero={key: sum(r["components_q24"][key] != 0 for r in rows)
                                       for key in rows[0]["components_q24"]})

    def row_base(p, i, a, r, local):
        return dict(position=p, index=i, actual=f"{a:04x}", reference=f"{r:04x}",
                    local_reconstructed=f"{local:04x}", within_gate=accepts(a, r),
                    conditional_original_reference_inputs=p < 2,
                    inherited_historical_provenance_qualified=True)

    soft_start = time.monotonic()
    for p in range(4):
        a, r = actual[4][p], reference[4][p]
        require(len(a[8]) == len(r[8]) == len(a[9]) == len(r[9]) == 14 * (p + 1),
                "all head/key softmax coverage")
        rows = []
        for head in range(14):
            lo, hi = head * (p + 1), (head + 1) * (p + 1)
            for rec in softmax_components(a[8][lo:hi], r[8][lo:hi], a[9][lo:hi], r[9][lo:hi]):
                k = rec["key_position"]
                rec.update(position=p, query_head=head, index=lo+k,
                           score_actual=f"{a[8][lo+k]:04x}", score_reference=f"{r[8][lo+k]:04x}",
                           conditional_original_reference_inputs=p < 2,
                           inherited_historical_provenance_qualified=True)
                rows.append(rec)
        write(f"position{p}_softmax_coordinates.json", rows)
        summaries[f"p{p}_softmax"] = summary(rows)
    phases["softmax_seconds"] = time.monotonic() - soft_start
    v_start = time.monotonic()
    vrows, causal = {}, []
    for p in range(4):
        a, r = actual[4][p], reference[4][p]
        require(a[3] == a[7] and r[3] == r[7] and len(a[7]) == len(r[7]) == 128,
                "V projection/cache-write identity")
        require(incoming[4][p] == actual[3][p][18], "actual layer03/04 hidden join")
        projected, totals = project_vectors(a[0], r[0], {k: v for k, v in tensors.items() if k != "bias"})
        rows = []
        for i in range(128):
            aa, single_a, detail_a = v_projection(a[0], tensors, i)
            rr, single_r, detail_r = v_projection(r[0], tensors, i)
            require(totals["A"][i] == detail_a["actual_q48"]
                    and totals["R"][i] == detail_r["actual_q48"], "full V independent integer oracle")
            require(aa == round_q48((units(projected["A"][i]) + units(tensors["bias"][i])) << 24),
                    "independent V bias oracle")
            row = row_base(p, i, a[7][i], r[7][i], aa)
            row.update(input_stage=0, producer_stage=3, cache_stage=7,
                       actual_sum_q48=totals["A"][i], reference_sum_q48=totals["R"][i],
                       reference_two_round=f"{rr:04x}", single_round_actual=f"{single_a:04x}",
                       single_round_reference=f"{single_r:04x}", bias=f"{tensors['bias'][i]:04x}",
                       components_q24=split([a[7][i], aa, single_a, single_r, r[7][i]],
                           ["local_V", "dot_then_bias_rounding", "stage00_input_drift",
                            "reference_recurrence"]))
            rows.append(row)
        vrows[p] = rows
        write(f"position{p}_v_coordinates.json", rows)
        summaries[f"p{p}_v"] = summary(rows)
    for p in range(4):
        for i in range(896):
            for k in range(p + 1):
                index = (i // 64 // 7) * 64 + i % 64
                joined = parent_keys[p, i, k]
                probability_index = (i // 64) * (p + 1) + k
                require(joined["actual_value"] == vrows[k][index]["actual"]
                        and joined["reference_value"] == vrows[k][index]["reference"]
                        and int(joined["actual_probability"], 16) == actual[4][p][9][probability_index]
                        and int(joined["reference_probability"], 16) == reference[4][p][9][probability_index],
                        "parent AV probability/V producer join")
                causal.append(dict(position=p, av_index=i, key_position=k, v_index=index,
                                   v_record=f"position{k}_v_coordinates.json#{index}",
                                   actual=vrows[k][index]["actual"], reference=vrows[k][index]["reference"],
                                   conditional_original_reference_inputs=min(p, k) < 2))
        require(len(parent_av[p]) == 896, "parent AV coordinate coverage")
    require(len(causal) == 8960, "complete historical/current V join")
    write("causal_v_joins.json", causal)
    phases["v_seconds"] = time.monotonic() - v_start
    residual_start = time.monotonic()
    for p in range(4):
        a, r = actual[3][p], reference[3][p]
        h, rh = incoming[3][p], r2[p][18]
        require(all(len(x) == 896 for x in (h, rh, a[11], r[11], a[12], r[12])),
                "residual geometry")
        upstream[p] = {
            name: dict(bit_differences=sum(x != y for x, y in zip(left, right, strict=True)),
                       material_failures=sum(not accepts(x, y) for x, y in zip(left, right, strict=True)),
                       coordinates=len(left))
            for name, left, right in (
                ("layer02_stage18_incoming_hidden", h, rh), ("layer03_stage11_O", a[11], r[11]),
                ("layer03_stage14_gate", a[14], r[14]), ("layer03_stage15_up", a[15], r[15]),
                ("layer04_stage00_V_input", actual[4][p][0], reference[4][p][0]),
                ("layer04_stage08_scores", actual[4][p][8], reference[4][p][8]))}
        rows = []
        for i in range(896):
            joined = parent_branches[p][i]
            require(joined["index"] == i and joined["position"] == p
                    and int(joined["stages"]["12"]["actual"], 16) == a[12][i]
                    and int(joined["stages"]["12"]["reference"], 16) == r[12][i]
                    and int(joined["stages"]["17"]["actual"], 16) == a[17][i]
                    and int(joined["stages"]["17"]["reference"], 16) == r[17][i]
                    and int(joined["stage12_actual_operands"]["incoming"], 16) == h[i]
                    and int(joined["stage12_actual_operands"]["attention_o"], 16) == a[11][i],
                    "parent residual/down coordinate join")
            cases = {}
            for label, left, right in (("AA", h[i], a[11][i]), ("AR", h[i], r[11][i]),
                                       ("RA", rh[i], a[11][i]), ("RR", rh[i], r[11][i])):
                bits = round_q48((units(left) + units(right)) << 24)
                oracle, invalid, saturated = residual_add(left, right)
                require(not (invalid or saturated) and bits == oracle, "independent residual oracle")
                cases[label] = bits
            row = row_base(p, i, a[12][i], r[12][i], cases["AA"])
            row.update(incoming_actual=f"{h[i]:04x}", incoming_reference=f"{rh[i]:04x}",
                       o_actual=f"{a[11][i]:04x}", o_reference=f"{r[11][i]:04x}",
                       cases={k: f"{v:04x}" for k, v in cases.items()},
                       incoming_delta_q24=units(h[i])-units(rh[i]),
                       o_delta_q24=units(a[11][i])-units(r[11][i]),
                       rounding_delta_q24=(units(a[12][i])-units(h[i])-units(a[11][i])))
            row["components_q24"] = split([a[12][i], cases["AA"], cases["AR"], cases["RR"], r[12][i]],
                ["local_residual", "O_after_incoming", "incoming_at_reference_O", "reference_recurrence"])
            row["opposite_order_components_q24"] = split(
                [a[12][i], cases["AA"], cases["RA"], cases["RR"], r[12][i]],
                ["local_residual", "incoming_after_O", "O_at_reference_incoming", "reference_recurrence"])
            rows.append(row)
        write(f"position{p}_residual_coordinates.json", rows)
        summaries[f"p{p}_residual"] = summary(rows)
    phases["residual_seconds"] = time.monotonic() - residual_start
    silu_start = time.monotonic()
    for p in range(4):
        a, r = actual[3][p], reference[3][p]
        require(all(len(x) == 4864 for x in (a[14], r[14], a[15], r[15], a[16], r[16])),
                "gate/up-to-down input geometry")
        rows = []
        tr = torch.tensor([units(v)/(1 << 24) for v in r[14]], dtype=torch.float64)
        ur = torch.tensor([units(v)/(1 << 24) for v in r[15]], dtype=torch.float64)
        recurrence = (torch.nn.functional.silu(tr)*ur).half().numpy().view(np.uint16).tolist()
        for i in range(4864):
            reconstructed = accurate_silu(a[14][i], a[15][i])
            oracle, invalid, saturated = silu_gate_exp(a[14][i], a[15][i])
            require(not (invalid or saturated) and reconstructed["bits"] == oracle,
                    "independent own-input SiLU integer oracle")
            ideal = {}
            with localcontext() as context:
                context.prec = 80
                for label, gate, up in (("AA", a[14][i], a[15][i]), ("AR", a[14][i], r[15][i]),
                                        ("RA", r[14][i], a[15][i]), ("RR", r[14][i], r[15][i])):
                    ideal[label] = nearest_decimal(mathematical_silu(gate, up),
                                                   (gate ^ up) & 0x8000)
            direct = round_q48(Fraction(reconstructed["product_q72"], 1 << 24))
            if direct & 0x7fff == 0:
                direct = (a[14][i] ^ a[15][i]) & 0x8000
            row = row_base(p, i, a[16][i], r[16][i], oracle)
            row.update(gate_actual=f"{a[14][i]:04x}", gate_reference=f"{r[14][i]:04x}",
                       up_actual=f"{a[15][i]:04x}", up_reference=f"{r[15][i]:04x}",
                       source_down_input_index=i, ideal_cases={k: f"{v:04x}" for k, v in ideal.items()},
                       reference_binary64_recurrence=f"{recurrence[i]:04x}", **reconstructed)
            chain = [a[16][i], oracle, direct, ideal["AA"], ideal["AR"], ideal["RR"],
                     recurrence[i], r[16][i]]
            names = ["local_silu", "Q72_to_Q24_rounding", "sigmoid_approximation",
                     "up_after_gate", "gate_at_reference_up", "ideal_vs_binary64", "reference_recurrence"]
            row["components_q24"] = split(chain, names)
            chain[4] = ideal["RA"]
            names[3:5] = ["gate_after_up", "up_at_reference_gate"]
            row["opposite_order_components_q24"] = split(chain, names)
            rows.append(row)
        require(len(parent_branches[p]) == 896, "parent downstream branch coordinate coverage")
        write(f"position{p}_gateup_coordinates.json", rows)
        summaries[f"p{p}_gateup"] = summary(rows)
    phases["gate_up_seconds"] = time.monotonic() - silu_start
    discrepancies = sum(s["own_input_local_discrepancies"] for s in summaries.values())
    incoming_drift = sum(x["layer02_stage18_incoming_hidden"]["bit_differences"] for x in upstream.values())
    decision = (
        "Drift is already present at layer02 stage18 -> layer03 incoming hidden; "
        "this is the earliest observed boundary in this retained scope, not its first historical producer."
        if incoming_drift else
        "No incoming layer02 stage18 bit drift was observed; inspect the measured O/gate/up and score/V branches."
    )
    if discrepancies:
        decision += " Own-input discrepancies are retained and prevent claiming local operator agreement."
    else:
        decision += " All four local operators reconstruct exactly on retained actual operands."
    write("upstream_boundaries.json", upstream)
    write("result.json", dict(
        status="DIAGNOSTIC_COMPLETE_NO_RTL_REPLAY", exact_accounting_closed=True,
        summaries=summaries, own_input_local_discrepancies=discrepancies,
        phases_seconds=phases, elapsed_seconds=time.monotonic()-start,
        original_positions0_1_reference_provenance_authenticated=False, provenance=QUALIFICATION,
        decision=decision, upstream_boundaries=upstream,
        next_exact_upstream_boundary="If own-input reconstruction closes, the earlier retained boundary "
        "is layer02 stage18 incoming hidden versus layer03 stage11 O, with layer03 stage14/15 "
        "projection inputs and layer04 stage00 V input / stage08 QK score producers. General "
        "accurate softmax/SiLU are sensitivity candidates only where their named terms are nonzero; "
        "downstream causal benefit is not established by this local split.",
        intervention_candidate_supported=False, policy_changed=False, reference_injected=False,
        RTL_simulations=0, fresh_RTL_PASS=False, binary64_admission_evaluated=False,
        third_token_claim=False, source_promoted=False, A_B_splice=False,
        review_status="PENDING_NORMAL_HOST_REVIEWER"))
    print(json.dumps(dict(status="DIAGNOSTIC_COMPLETE_NO_RTL_REPLAY",
                          own_input_local_discrepancies=discrepancies, summaries=summaries)))


if __name__ == "__main__":
    if sys.argv[1:] == ["--measure"]:
        measure()
    else:
        try:
            bootstrap()
        except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
            write("failure.json", dict(
                status="DIAGNOSTIC_NO_COMPLETED_MEASUREMENT", failure_taxonomy="diagnostic_execution",
                root_cause_hypothesis=str(error),
                regression="Fresh attempt must complete all frozen coordinates and independent oracles.",
                RTL_correctness_conclusion=None))
            raise
        finally:
            files = sorted(p for p in OUT.rglob("*") if p.is_file())
            write("artifacts.json", [record(p) for p in files])
            for path in files + [OUT / "artifacts.json"]:
                path.chmod(0o444)
            for path in sorted((p for p in OUT.rglob("*") if p.is_dir()), reverse=True):
                path.chmod(0o555)
            OUT.chmod(0o555)
