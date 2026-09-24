#!/usr/bin/env python3
"""Exact retained A-lineage producer accounting; no RTL or model replay."""
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
PARENT = ROOT / "build/retained_layer04_softmax_v_layer03_residual_gateup_3e750691b915_attempt002"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/3e750691b915")
OUT = ROOT / "build/retained_layer02_stage18_producer_split_01ce0c7e0c31_attempt001"
PREFIX = ROOT / "build/token358_position3_full_continuation_attempt001"
ROOTS = {
    "frozen.json": "2f112e48823d346fd0f53b2577d2f7168bf6e518dedb32723236b2e7c14d323f",
    "result.json": "15d277a95a9044f3f28b9599ae34d9d1a011715d8f4e610963e118e0b300555e",
    "artifacts.json": "f1c3b56dffeb974b99aef0fc1122d3620b87804f039010dfb4adfd3b1d490894",
}
QUALIFICATION = (
    "Positions0/1 retained independent-reference bytes and historical K/V prefixes "
    "are content-bound, not original-reference-input provenance authenticated. "
    "Positions2/3 inherit that conditional historical qualification. Supplemental "
    "layer01/02 roots are identified separately, not promoted to original execution "
    "provenance. Stage16-to-down joins are source-derived ordered vector accesses, "
    "not captured read-bus observations. No A/B splice or reference injection."
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


def command(name, argv, timeout=110):
    with (OUT / (name + ".command.sh")).open("x", encoding="ascii") as stream:
        stream.write(shlex.join(argv) + "\n")
    start = time.monotonic()
    with (OUT / (name + ".log")).open("xb") as stream:
        try:
            result = subprocess.run(argv, cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            write(name + ".status.json", dict(returncode=None, timeout=timeout,
                                              seconds=time.monotonic() - start))
            raise
    write(name + ".status.json", dict(returncode=result.returncode,
                                      seconds=time.monotonic() - start))
    require(result.returncode == 0, name + " failed; retained inner log is authoritative")


def bootstrap():
    receipt = json.loads((REVIEW / "round-0001.json").read_text())
    checkpoint = (REVIEW / "CHECKPOINT.md").read_text()
    require(receipt["kind"] == "round_reviewed_handoff"
            and receipt["producer_role"] == "reviewer"
            and receipt["mission_id"] == "3e750691b915"
            and receipt["review"]["status"] == "done"
            and "layer02 stage18" in receipt["review"]["next_action"],
            "mission-scoped normal reviewed parent required")
    for name, digest in ROOTS.items():
        require(record(PARENT / name)["sha256"] == digest and digest in checkpoint,
                "reviewed parent root changed: " + name)
    manifest = json.loads((PARENT / "artifacts.json").read_text())
    for rec in manifest:
        require(record(rec["path"]) == rec, "parent artifact changed: " + rec["path"])
    write("parent_authentication.json", dict(
        roots={name: record(PARENT / name) for name in ROOTS},
        artifacts_authenticated=len(manifest), review=record(REVIEW / "round-0001.json"),
        mission=record(REVIEW / "mission.json"), checkpoint=record(REVIEW / "CHECKPOINT.md"),
        historical_pending_wording_is_not_missing_review=True))
    sources = OUT / "sources"
    sources.mkdir()
    snapshots = []
    for path in sorted((PARENT / "sources").iterdir()) + [Path(__file__).resolve()]:
        if path.suffix not in (".py", ".sv"):
            continue
        target = sources / path.name
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        snapshots.append(dict(origin=record(path), snapshot=record(target)))
    write("source_snapshots.json", snapshots)
    command("measurement", [sys.executable, "-B", str(sources / Path(__file__).name), "--measure"])
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    write("artifacts.json", [record(p) for p in files])
    for path in files + [OUT / "artifacts.json"]:
        path.chmod(0o444)
    result = json.loads((OUT / "result.json").read_text())
    print(json.dumps({key: result[key] for key in (
        "status", "decision", "next_exact_upstream_boundary", "own_input_local_discrepancies",
        "summaries", "phases_seconds")}, sort_keys=True))
    print(json.dumps({name: record(OUT / name) for name in
                      ("frozen.json", "result.json", "artifacts.json")}, sort_keys=True))


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
    from exact_projection import project_vectors
    from fp16_adaptation_oracle import residual_add, silu_gate_exp
    from projection_oracle import complete_projection_output
    from trace_lineage_a_layer07_downproj import framed_hidden, hex_rows, round_q48
    from trace_lineage_a_layer07_score_softmax_v import array
    from trace_lineage_a_layer07_silu import accurate_silu, mathematical_silu, nearest_decimal

    start = time.monotonic()
    torch.set_num_threads(1)
    units, accepts = boundary.units, boundary.accepts
    boundary.ROOT, boundary.BUILD = ROOT, ROOT / "build"
    inputs = boundary.Inputs()
    inputs.load(PARENT / "artifacts.json", root=True)
    parent_freeze = inputs.load(PARENT / "frozen.json")
    require(parent_freeze["comparison_policy"] == boundary.POLICY
            and parent_freeze["token_history"] == boundary.HISTORY
            and inputs.load(PARENT / "result.json")["exact_accounting_closed"],
            "parent policy/history/accounting mismatch")
    inputs.register(parent_freeze)
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

    original = load(PREFIX / "frozen.json")
    require(inputs.records[str(PREFIX / "frozen.json")]["sha256"] ==
            "f81181054c32a22ac303af8416f2b41e733ea265b90cdcba9c71201757d6b599",
            "historical source freeze changed")
    recovery = load(ROOT / "build/token358_position3_full_continuation_attempt002/result.json")
    pre = load(PREFIX / "layer02/prepared.json")
    transaction = load(PREFIX / "layer02/transaction.json")
    kv = pre["kv_parent"]
    old = load(kv["layer_result"]["path"])
    require(recovery["layer"] == transaction["layer_index"] == old["layer_index"] == 2
            and recovery["position"] == transaction["position"] == 3
            and pre["binary"] == kv["live_binary"] == recovery["binary"]
            and kv["valid_positions"] == [0, 1, 2]
            and "/model24_persistent_kv_selected_token_corrected_q_attempt005/" in kv["state"]["path"],
            "retained A-lineage position/layer/binary/state mismatch")
    for rec in (pre["binary"], kv["state"], *kv["position_states"]):
        inputs.read(rec["path"])
    actual = {p: boundary.decode_trace(inputs.read(rec["path"]), p)
              for p, rec in enumerate(kv["actual_kv_traces"])}
    trace_path = Path(transaction["output"]["path"]).with_name("trace.hex")
    actual[3] = boundary.decode_trace(inputs.read(trace_path), 3)
    incoming = [from_hex(old["positions"][p]["vectors"]["input"] if p < 3
                         else pre["input_hidden"], True) for p in range(4)]
    require(transaction["input"]["sha256"] == pre["input_hidden"]["semantic_sha256"],
            "actual layer01-to-layer02 semantic join")
    require(from_hex(transaction["output"], True) == actual[3][18],
            "stage18 trace/output semantic join")
    history = load(pre["independent_parent"]["path"])
    early = load(Path(pre["independent_parent"]["path"]).with_name("layer02_generation0.json"))
    stages = (11, 12, 14, 15, 16, 17, 18)
    reference = {p: {} for p in range(4)}
    for rec in early["stages"]:
        stage = int(Path(rec["path"]).stem[5:])
        if stage in stages:
            p = int(Path(rec["path"]).parent.name[8:])
            reference[p][stage] = from_array(rec)
    for stage in stages:
        reference[2][stage] = from_array(next(r for r in history["stages"]
                                             if r["path"].endswith(f"/stage{stage:02d}.npy")))
        reference[3][stage] = from_hex(pre["independent_stages"][str(stage)])
    upstream_history = load(Path(pre["independent_parent"]["path"]).with_name("layer01_generation1.json"))
    upstream_early = load(Path(pre["independent_parent"]["path"]).with_name("layer01_generation0.json"))
    reference_incoming = {}
    for obj, positions, tokens in (
            (early, [0, 1], boundary.HISTORY[:2]), (history, [2], boundary.HISTORY[:3]),
            (upstream_early, [0, 1], boundary.HISTORY[:2]),
            (upstream_history, [2], boundary.HISTORY[:3])):
        require(obj["history"] == tokens and obj["positions"] == positions,
                "independent retained reference history/position mismatch")
    require(early["checkpoint_tensor_hashes"] == history["checkpoint_tensor_hashes"],
            "layer02 reference checkpoint identity mismatch")
    for obj in (upstream_early, upstream_history):
        for rec in obj["stages"]:
            if rec["path"].endswith("/stage18.npy"):
                reference_incoming[int(Path(rec["path"]).parent.name[8:])] = from_array(rec)
    upstream_package = ROOT / "build/token358_position3_layer1_preflight_attempt002/package"
    load(upstream_package / "frozen.json")
    load(upstream_package / "launch_package.json")
    upstream_ref = upstream_package / "independent/stage18.hex"
    fresh = str(upstream_ref) not in inputs.bindings
    upstream_data = inputs.read(upstream_ref, root=fresh)
    if fresh:
        supplemental.append(inputs.records[str(upstream_ref)])
    reference_incoming[3] = hex_rows(upstream_data)
    for first, later in ((early, history), (upstream_early, upstream_history)):
        for name in ("k", "v"):
            require(from_array(first["own_cache"][name]) == from_array(later["own_cache"][name])[:256],
                    "conditional historical reference cache prefix mismatch")
    parent_rows = {p: inputs.load(PARENT / f"position{p}_residual_coordinates.json") for p in range(4)}
    tensors, tensor_records = {}, []
    for rec in pre["vectors"]["tensors"]:
        meta = rec["checkpoint_tensor"]
        if not meta["name"].startswith("model.layers.2.mlp.down_proj."):
            continue
        vals = hex_rows(inputs.read(rec["serialized"]["path"]))
        data = b"".join(v.to_bytes(2 if meta["dtype"] == "F16" else 4, "little") for v in vals)
        require(len(data) == meta["bytes"] and hashlib.sha256(data).hexdigest() == meta["sha256"]
                and any(r["name"] == meta["name"] and r["sha256"] == meta["sha256"]
                        for r in history["checkpoint_tensor_hashes"]), "official AWQ tensor identity")
        tensors[meta["name"].rsplit(".", 1)[1]] = vals
        tensor_records.append(rec)
    require(set(tensors) == {"qweight", "qzeros", "scales"}, "native unbiased down projection")
    tool = shutil.which("iverilog")
    require(tool is not None, "host iverilog unavailable; container availability is unconfirmed")
    command("tool_versions", [sys.executable, "-B", "-c",
            "import sys,numpy,torch;print(sys.version);print(numpy.__version__);print(torch.__version__)"])
    command("iverilog_version", [tool, "-V"])
    tops = ("ace3_awq_w4a16_projection_engine", "ace3_fp16_residual_add_core", "ace3_fp16_silu_gate_core")
    parameters = {
        "ace3_awq_w4a16_projection_engine.IN_FEATURES": 4864,
        "ace3_awq_w4a16_projection_engine.OUT_FEATURES": 896,
        "ace3_awq_w4a16_projection_engine.BIAS_ENABLE": 0,
        "ace3_awq_w4a16_projection_engine.SINGLE_ROUND_BIAS": 0,
        "ace3_fp16_residual_add_core.VECTOR_SIZE": 896,
        "ace3_fp16_silu_gate_core.INTERMEDIATE_SIZE": 4864,
        "ace3_fp16_silu_gate_core.ACCURATE_SIGMOID": 1,
    }
    write("frozen.json", dict(
        parent_roots={name: record(PARENT / name) for name in ROOTS},
        evaluator=record(Path(__file__)), source_snapshots=record(OUT / "source_snapshots.json"),
        comparison_policy=boundary.POLICY, token_history=boundary.HISTORY,
        positions=[0, 1, 2, 3], layer=2, authenticated_inputs=list(inputs.records.values()),
        supplemental_content_roots=supplemental, checkpoint_tensors=tensor_records,
        public_tops={top: record(OUT / "sources" / (top + ".sv")) for top in tops},
        public_parameters=parameters, public_port_contract="Exact frozen source declarations; no aliases",
        case_counts=dict(stage18=3584, stage12=3584, stage16=19456, stage17=3584,
                         incoming_hidden=3584, source_derived_down_scalar_terms=17432576),
        numerical_policy="Native G128 asymmetric GEMM nibble lanes 0,4,1,5,2,6,3,7; no qzero +1; "
        "FP16 scales/activations/KV. Exact Q48 down accumulation with signed 96-bit group and "
        "102-bit cross-group bounds, one FP16 RNE. Residual Q24 exact add then FP16 RNE. "
        "SiLU source polynomial Q24 sigmoid; Q72 product -> Q24 RNE -> FP16 RNE; "
        "signed underflow zero follows frozen operator helpers.",
        reference_policy="Retained independently propagated FP16-interstage arrays, with separate "
        "Decimal80 mathematical SiLU and torch binary64 SiLU recurrence; neither replaces the reference.",
        accounting="Full coordinate coverage, both substitution orders. Every source-derived "
        "(position,input,output) down term is factorized by coefficient_q24.npy and stage16 "
        "coordinate components. Every output has exact integer Q48 sums; no sparsification.",
        validation="One semantic execution; existing SiLU regressions, all-coordinate independent "
        "integer residual/SiLU/projection oracles, exact telescoping and parent joins.",
        provenance=QUALIFICATION, original_positions0_1_reference_provenance_authenticated=False,
        binary64_admission_evaluated=False, RTL_simulations=0))
    argv = [tool, "-g2012"]
    for top in tops:
        argv.extend(("-s", top))
    for name, value in parameters.items():
        argv.extend(("-P", f"{name}={value}"))
    command("public_contract_compile", argv + ["-o", str(OUT / "public_contracts.vvp")]
            + [str(p) for p in sorted((OUT / "sources").glob("*.sv"))])
    command("existing_helper_regression", [sys.executable, "-B", "-m", "unittest", "discover",
            "-s", str(OUT / "sources"), "-p", "test_trace_lineage_a_layer07_silu.py"])
    phases = dict(authentication_freeze_compile_seconds=time.monotonic() - start)
    summaries, upstream = {}, {}

    def split(chain, names):
        require(len(chain) == len(names) + 1, "split geometry")
        result = {name: units(chain[i]) - units(chain[i + 1]) for i, name in enumerate(names)}
        require(sum(result.values()) == units(chain[0]) - units(chain[-1]), "Q24 split closure")
        return result

    def row_base(p, i, actual_bits, reference_bits, local):
        return dict(position=p, index=i, actual=f"{actual_bits:04x}",
                    reference=f"{reference_bits:04x}", local_reconstructed=f"{local:04x}",
                    within_gate=accepts(actual_bits, reference_bits),
                    conditional_original_reference_inputs=p < 2,
                    inherited_historical_provenance_qualified=True)

    def save_rows(p, name, rows):
        write(f"position{p}_{name}_coordinates.json", rows)
        summaries[f"p{p}_{name}"] = dict(
            coordinates=len(rows),
            bit_differences=sum(r["actual"] != r["reference"] for r in rows),
            material_failures=sum(not r["within_gate"] for r in rows),
            own_input_local_discrepancies=sum(r["actual"] != r["local_reconstructed"] for r in rows),
            max_abs_error_q24=max(abs(units(int(r["actual"], 16))-units(int(r["reference"], 16)))
                                 for r in rows))

    residual_start = time.monotonic()
    for p in range(4):
        a, r, h, rh = actual[p], reference[p], incoming[p], reference_incoming[p]
        require(len(h) == len(rh) == len(parent_rows[p]) == 896, "incoming geometry")
        upstream[p] = {}
        for name, left, right in (("layer01_stage18_incoming_hidden", h, rh),
                                 ("layer02_stage11_O", a[11], r[11]),
                                 ("layer02_stage14_gate", a[14], r[14]),
                                 ("layer02_stage15_up", a[15], r[15])):
            upstream[p][name] = dict(
                coordinates=len(left), bit_differences=sum(x != y for x, y in zip(left, right, strict=True)),
                material_failures=sum(not accepts(x, y) for x, y in zip(left, right, strict=True)))
        write(f"position{p}_incoming_coordinates.json", [
            dict(position=p, index=i, actual=f"{h[i]:04x}", reference=f"{rh[i]:04x}",
                 delta_q24=units(h[i])-units(rh[i]), within_gate=accepts(h[i], rh[i]),
                 inherited_historical_provenance_qualified=True) for i in range(896)])
        for stage, left_a, left_r, right_a, right_r, names in (
                (18, a[12], r[12], a[17], r[17], ("stage12_residual", "stage17_down")),
                (12, h, rh, a[11], r[11], ("incoming_hidden", "stage11_O"))):
            require(all(len(v) == 896 for v in (left_a, left_r, right_a, right_r, a[stage], r[stage])),
                    "residual geometry")
            rows = []
            for i in range(896):
                parent = parent_rows[p][i]
                require(parent["position"] == p and parent["index"] == i
                        and int(parent["incoming_actual"], 16) == a[18][i]
                        and int(parent["incoming_reference"], 16) == r[18][i],
                        "reviewed parent layer02-stage18 coordinate join")
                cases = {}
                for label, left, right in (
                        ("AA", left_a[i], right_a[i]), ("AR", left_a[i], right_r[i]),
                        ("RA", left_r[i], right_a[i]), ("RR", left_r[i], right_r[i])):
                    bits = round_q48((units(left) + units(right)) << 24)
                    oracle, invalid, saturated = residual_add(left, right)
                    require(not (invalid or saturated) and bits == oracle, "independent residual oracle")
                    cases[label] = bits
                row = row_base(p, i, a[stage][i], r[stage][i], cases["AA"])
                row.update(operand_names=list(names), left_actual=f"{left_a[i]:04x}",
                           left_reference=f"{left_r[i]:04x}", right_actual=f"{right_a[i]:04x}",
                           right_reference=f"{right_r[i]:04x}",
                           cases={k: f"{v:04x}" for k, v in cases.items()},
                           actual_FP16_rounding_q24=units(cases["AA"])-units(left_a[i])-units(right_a[i]),
                           reference_FP16_rounding_q24=units(cases["RR"])-units(left_r[i])-units(right_r[i]))
                row["components_q24"] = split(
                    [a[stage][i], cases["AA"], cases["AR"], cases["RR"], r[stage][i]],
                    ["local_residual", names[1]+"_after_left", names[0]+"_at_reference_right", "reference_recurrence"])
                row["opposite_order_components_q24"] = split(
                    [a[stage][i], cases["AA"], cases["RA"], cases["RR"], r[stage][i]],
                    ["local_residual", names[0]+"_after_right", names[1]+"_at_reference_left", "reference_recurrence"])
                rows.append(row)
            save_rows(p, f"stage{stage}", rows)
    phases["residual_seconds"] = time.monotonic() - residual_start
    silu_start = time.monotonic()
    gateup = {}
    for p in range(4):
        a, r = actual[p], reference[p]
        require(all(len(v) == 4864 for v in (a[14], a[15], a[16], r[14], r[15], r[16])),
                "stage16 input/output geometry")
        tr = torch.tensor([units(v)/(1 << 24) for v in r[14]], dtype=torch.float64)
        ur = torch.tensor([units(v)/(1 << 24) for v in r[15]], dtype=torch.float64)
        recurrence = (torch.nn.functional.silu(tr)*ur).half().numpy().view(np.uint16).tolist()
        rows = []
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
                    ideal[label] = nearest_decimal(mathematical_silu(gate, up), (gate ^ up) & 0x8000)
            direct = round_q48(Fraction(reconstructed["product_q72"], 1 << 24))
            if direct & 0x7fff == 0:
                direct = (a[14][i] ^ a[15][i]) & 0x8000
            row = row_base(p, i, a[16][i], r[16][i], oracle)
            row.update(gate_actual=f"{a[14][i]:04x}", gate_reference=f"{r[14][i]:04x}",
                       up_actual=f"{a[15][i]:04x}", up_reference=f"{r[15][i]:04x}",
                       source_down_input_index=i, ideal_cases={k: f"{v:04x}" for k, v in ideal.items()},
                       reference_binary64_recurrence=f"{recurrence[i]:04x}", **reconstructed)
            chain = [a[16][i], oracle, direct, ideal["AA"], ideal["AR"], ideal["RR"], recurrence[i], r[16][i]]
            names = ["local_silu", "Q72_to_Q24_rounding", "sigmoid_approximation",
                     "up_after_gate", "gate_at_reference_up", "ideal_vs_binary64", "reference_recurrence"]
            row["components_q24"] = split(chain, names)
            chain[4] = ideal["RA"]
            names[3:5] = ["gate_after_up", "up_at_reference_gate"]
            row["opposite_order_components_q24"] = split(chain, names)
            rows.append(row)
        gateup[p] = rows
        save_rows(p, "stage16", rows)
    phases["silu_seconds"] = time.monotonic() - silu_start
    down_start = time.monotonic()
    shifts = np.array([0, 16, 4, 20, 8, 24, 12, 28], dtype=np.uint32)
    qw = np.asarray(tensors["qweight"], dtype=np.uint32).reshape(4864, 112)
    qz = np.asarray(tensors["qzeros"], dtype=np.uint32).reshape(38, 112)
    weights = ((qw[:, :, None] >> shifts) & 15).reshape(4864, 896).astype(np.int64)
    zeros = ((qz[:, :, None] >> shifts) & 15).reshape(38, 896).astype(np.int64)
    scales = np.asarray([units(x) for x in tensors["scales"]], dtype=np.int64).reshape(38, 896)
    coefficients = (weights-np.repeat(zeros, 128, axis=0))*np.repeat(scales, 128, axis=0)
    with (OUT / "coefficient_q24.npy").open("xb") as stream:
        np.save(stream, coefficients, allow_pickle=False)
    coefficient_objects = coefficients.astype(object)
    for p in range(4):
        a, r = actual[p], reference[p]
        bits, totals = project_vectors(a[16], r[16], tensors)
        propagated = {}
        for key in ("components_q24", "opposite_order_components_q24"):
            names = list(gateup[p][0][key])
            matrix = np.asarray([[row[key][name] for row in gateup[p]] for name in names], dtype=object)
            propagated[key] = dict(zip(names, (matrix @ coefficient_objects).tolist(), strict=True))
        rows = []
        for i in range(896):
            group_sums = {}
            for label, activation in (("A", a[16]), ("R", r[16])):
                oracle = complete_projection_output(
                    activation, tensors["qweight"][i//8::112], tensors["qzeros"][i//8::112],
                    tensors["scales"][i::896], i % 8)
                require(not (oracle[2] or oracle[3]) and oracle[0] == totals[label][i]
                        and oracle[1] == bits[label][i]
                        and all(-(1 << 95) <= v < (1 << 95) for v in oracle[4])
                        and -(1 << 101) <= oracle[0] < (1 << 101), "independent all-coordinate down oracle")
                group_sums[label] = oracle[4]
            row = row_base(p, i, a[17][i], r[17][i], bits["A"][i])
            row.update(actual_sum_q48=totals["A"][i], reference_sum_q48=totals["R"][i],
                       reference_reconstructed=f"{bits['R'][i]:04x}", group_sums_q48=group_sums,
                       input_coordinates=4864, coefficient_column=i,
                       source_derived_not_observed_bus=True)
            for key, values in propagated.items():
                components = {name: values[name][i] for name in values}
                require(sum(components.values()) == totals["A"][i]-totals["R"][i],
                        "all 4864 stage16 terms close into down sum")
                components.update(
                    local_down=(units(a[17][i])-units(bits["A"][i])) << 24,
                    actual_down_FP16_rounding=(units(bits["A"][i]) << 24)-totals["A"][i],
                    negative_reference_down_FP16_rounding=totals["R"][i]-(units(bits["R"][i]) << 24),
                    down_reference_recurrence=(units(bits["R"][i])-units(r[17][i])) << 24)
                require(sum(components.values()) == (units(a[17][i])-units(r[17][i])) << 24,
                        "stage17 Q48 output split closure")
                row[key.replace("q24", "q48")] = components
            rows.append(row)
        save_rows(p, "stage17", rows)
    phases["down_projection_seconds"] = time.monotonic() - down_start
    discrepancies = sum(s["own_input_local_discrepancies"] for s in summaries.values())
    incoming_drift = sum(v["layer01_stage18_incoming_hidden"]["bit_differences"] for v in upstream.values())
    decision = (
        "Drift is already present at layer01 stage18 -> layer02 incoming hidden; this is "
        "the earliest observed boundary in this retained scope, not the first historical producer."
        if incoming_drift else
        "No incoming layer01 stage18 bit drift observed; the retained layer02 O and gate/up branches remain candidates."
    )
    decision += (" All four layer02 own-input operators reconstruct exactly." if not discrepancies else
                 " Own-input discrepancies remain and prevent local operator agreement.")
    write("upstream_boundaries.json", upstream)
    write("result.json", dict(
        status="DIAGNOSTIC_COMPLETE_NO_RTL_REPLAY", decision=decision, exact_accounting_closed=True,
        summaries=summaries, upstream_boundaries=upstream, own_input_local_discrepancies=discrepancies,
        next_exact_upstream_boundary="Batch layer01 stage18 residual/down, its incoming layer00 stage18 "
        "versus O-stage11, and gate/up-to-stage16 producers. Layer02 stage11 and stage14/15 "
        "remain parallel contributor boundaries; inherited drift is not a unique root cause. "
        "No downstream benefit of a general SiLU or rounding intervention is established here.",
        factorized_down_terms=4*4864*896, down_coefficients=record(OUT / "coefficient_q24.npy"),
        phases_seconds=phases, elapsed_seconds=time.monotonic()-start,
        provenance=QUALIFICATION, original_positions0_1_reference_provenance_authenticated=False,
        intervention_candidate_supported=False, policy_changed=False, reference_injected=False,
        RTL_simulations=0, fresh_RTL_PASS=False, binary64_admission_evaluated=False,
        third_token_claim=False, source_promoted=False, A_B_splice=False,
        review_status="PENDING_NORMAL_HOST_REVIEWER"))


if __name__ == "__main__":
    if sys.argv[1:] == ["--measure"]:
        measure()
    else:
        require({p.name for p in OUT.iterdir()} == {"launch.command.sh"},
                "fresh attempt required; existing files must not be replaced")
        try:
            bootstrap()
        except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
            if OUT.is_dir() and not (OUT / "failure.json").exists():
                write("failure.json", dict(
                    status="DIAGNOSTIC_NO_COMPLETED_MEASUREMENT",
                    failure_taxonomy="diagnostic_execution",
                    root_cause_hypothesis=str(error),
                    regression="Fresh attempt must close all frozen coordinates against independent oracles.",
                    RTL_correctness_conclusion=None))
                files = sorted(p for p in OUT.rglob("*") if p.is_file())
                if not (OUT / "artifacts.json").exists():
                    write("artifacts.json", [record(p) for p in files])
                for path in files + [OUT / "artifacts.json"]:
                    path.chmod(0o444)
            raise
