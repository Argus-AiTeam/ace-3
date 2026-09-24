#!/usr/bin/env python3
"""Authenticated retained layer00 producer accounting; no RTL/trajectory replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path("/home/argustest/ace3-argus")
PARENT = ROOT / "build/retained_layer01_stage18_producer_split_a947a562a897_attempt002"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/a947a562a897")
PARENT_HASHES = {
    "frozen.json": "63c02441386befce46e3609b0194bb46174138d015461abaa8b0cbaaffe5285a",
    "result.json": "730801c737ed9cc6a3bc61fcea1ec658640eb9922317845eb32213bb2a223df9",
    "artifacts.json": "cd0cdfccab679f4972b39a8137450779260bd4a81709a6dcbf2134adc3b7be02",
}
QUALIFICATION = (
    "Retained A-lineage only. Positions0/1 reference-stage bytes are authenticated, "
    "but their original reference-input provenance is unavailable; positions2/3 "
    "inherit that qualification. Official token embeddings are independently "
    "extracted from the authenticated checkpoint, not substituted for execution "
    "inputs. Equality with those roots does not retroactively authenticate original "
    "reference consumption. Projection accesses are source-derived, not observed "
    "read-bus transactions. No historical K/V regeneration or binary64 admission."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return dict(path=str(path), bytes=size, sha256=digest.hexdigest())


def write(name, value):
    with (OUT / name).open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def seal():
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    write("artifacts.json", [record(p) for p in files])
    for path in files + [OUT / "artifacts.json"]:
        path.chmod(0o444)


def bootstrap():
    receipt = json.loads((REVIEW / "round-0001.json").read_text())
    require(receipt["kind"] == "round_reviewed_handoff"
            and receipt["producer_role"] == "reviewer"
            and receipt["mission_id"] == "a947a562a897"
            and receipt["review"]["status"] == "done"
            and "layer00 stage18" in receipt["review"]["next_action"],
            "genuine mission-scoped parent review required")
    for name, digest in PARENT_HASHES.items():
        require(record(PARENT / name)["sha256"] == digest, "parent identity changed: " + name)
    manifest = json.loads((PARENT / "artifacts.json").read_text())
    for rec in manifest:
        require(record(rec["path"]) == rec, "parent artifact changed: " + rec["path"])
    write("parent_authentication.json", dict(
        roots={n: record(PARENT / n) for n in PARENT_HASHES},
        review=record(REVIEW / "round-0001.json"),
        mission=record(REVIEW / "mission.json"), authenticated_artifacts=len(manifest),
        historical_pending_field_is_not_missing_review=True))
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
    if ATTEMPT > 1:
        previous = OUT.with_name(
            f"retained_layer00_stage18_producer_split_dc9689d1352f_attempt{ATTEMPT-1:03d}")
        write("prior_attempt_diagnosis.json", dict(
            previous_files=[record(p) for p in sorted(previous.rglob("*")) if p.is_file()],
            failure_taxonomy="evaluator_no_execution",
            root_cause_hypothesis="Safe-path Python excluded the frozen script directory "
            "from sys.path; the retained helper import failed before input loading.",
            regression="Explicitly load helpers from this attempt's frozen source directory "
            "and execute the entire original oracle accounting once.",
            RTL_correctness_conclusion=None))
    os.execv(sys.executable, [sys.executable, "-B", str(sources / Path(__file__).name),
                             "--attempt", str(ATTEMPT), "--measure"])


def measure():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(OUT / "sources"))
    import numpy as np
    import torch
    from decimal import localcontext
    from fractions import Fraction
    from safetensors import safe_open
    import diagnose_lineage_a_position3_boundary as boundary
    import retained_layer01_stage18_producer_split_diagnostic as parent
    from exact_projection import project_vectors
    from fp16_adaptation_oracle import residual_add, rmsnorm, silu_gate_exp
    from projection_oracle import complete_projection_output
    from trace_lineage_a_layer07_downproj import framed_hidden, hex_rows, round_q48
    from trace_lineage_a_layer07_rmsnorm import reconstruct, mathematical_outputs
    from trace_lineage_a_layer07_score_softmax_v import array
    from trace_lineage_a_layer07_silu import accurate_silu, mathematical_silu, nearest_decimal

    start = time.monotonic()
    torch.set_num_threads(1)
    parent.OUT = OUT
    boundary.ROOT, boundary.BUILD = ROOT, ROOT / "build"
    units, accepts = boundary.units, boundary.accepts
    inputs = boundary.Inputs()
    inputs.load(PARENT / "artifacts.json", root=True)
    pf = inputs.load(PARENT / "frozen.json")
    inputs.register(pf)
    require(pf["comparison_policy"] == boundary.POLICY
            and pf["token_history"] == boundary.HISTORY
            and inputs.load(PARENT / "result.json")["exact_accounting_closed"],
            "parent numerical policy/history/accounting changed")
    supplemental = []

    def load(path):
        path = Path(path).resolve()
        fresh = str(path) not in inputs.bindings
        value = inputs.load(path, root=fresh)
        if fresh:
            supplemental.append(inputs.records[str(path)])
        return value

    def bits(rec, framed=False):
        inputs.register(rec)
        result = (framed_hidden if framed else hex_rows)(inputs.read(rec["path"]))
        digest = hashlib.sha256(np.asarray(result, dtype="<u2").tobytes()).hexdigest()
        require(rec.get("semantic_sha256", digest) == digest, "FP16 semantic hash mismatch")
        return result

    historical = ROOT / "build/model24_persistent_kv_selected_token_corrected_q_attempt005"
    current = ROOT / "build/token358_position3_layer0_attempt001"
    old, later = load(historical / "layer00/result.json"), load(current / "result.json")
    freeze = load(current / "frozen.json")
    historical_freeze = load(freeze["parent_frozen"]["path"])
    load(freeze["package"]["path"])
    transaction = load(later["transaction"]["path"])
    require(old["layer_index"] == later["layer"] == 0 and later["position"] == 3
            and later["token_history"] == boundary.HISTORY
            and later["selected_token_id"] == 358
            and freeze["parameters"] == {"ACCURATE_SILU": 1, "LAYER_INDEX": 0},
            "layer00 retained public execution identity")
    actual, reference, incoming = {}, {p: {} for p in range(4)}, {}
    for p in range(3):
        pos = old["positions"][p]
        require(pos["position"] == p and pos["layer_index"] == 0, "historical position")
        actual[p] = boundary.decode_trace(inputs.read(pos["raw"]["trace"]["path"]), p)
        incoming[p] = bits(pos["vectors"]["input"], True)
        require(bits(pos["output"], True) == actual[p][18], "historical trace/final join")
        require(hashlib.sha256(np.asarray(incoming[p], dtype="<u2").tobytes()).hexdigest()
                == pos["input"]["sha256"], "historical consumed input semantic identity")
    actual[3] = boundary.decode_trace(
        inputs.read(later["independent_comparisons"][0]["actual_trace"]["path"]), 3)
    incoming[3] = bits(freeze["embedding"]["file"], True)
    require(transaction["layer_index"] == 0 and transaction["position"] == 3
            and bits(transaction["vectors"]["input"], True) == incoming[3]
            and transaction["input"]["sha256"] == freeze["embedding"]["semantic_sha256"],
            "position3 authenticated simulator-input/token-root join")
    require(bits(later["output_hidden"], True) == actual[3][18], "position3 trace/final join")
    require(freeze["embedding"]["token_id"] == 358 and freeze["embedding"]["position"] == 3
            and hashlib.sha256(np.asarray(incoming[3], dtype="<u2").tobytes()).hexdigest()
            == freeze["embedding"]["semantic_sha256"], "position3 embedding semantic identity")
    ref_root = Path(freeze["independent_parent"]["path"]).parent
    early, history = load(ref_root / "layer00_generation0.json"), load(freeze["independent_parent"]["path"])
    require(early["checkpoint_tensor_hashes"] == history["checkpoint_tensor_hashes"],
            "independent reference checkpoint tensors changed")
    for obj, positions, tokens in ((early, [0, 1], boundary.HISTORY[:2]),
                                   (history, [2], boundary.HISTORY[:3])):
        require(obj["layer"] == 0 and obj["positions"] == positions and obj["history"] == tokens,
                "reference position/history identity")
        for rec in obj["stages"]:
            p = int(Path(rec["path"]).parent.name[8:])
            stage = int(Path(rec["path"]).stem[5:])
            reference[p][stage] = array(inputs, rec).reshape(-1).tolist()
    for name in ("k", "v"):
        require(array(inputs, early["own_cache"][name]).reshape(-1).tolist()
                == array(inputs, history["own_cache"][name]).reshape(-1).tolist()[:256],
                "historical independent K/V prefix identity")
    for comparison in later["independent_comparisons"]:
        require(comparison["position"] == 3 and comparison["layer"] == 0, "reference p3 identity")
        reference[3][comparison["stage"]] = bits(comparison["independent_reference"])
    for p in range(4):
        rows = inputs.load(PARENT / f"position{p}_incoming_coordinates.json")
        require(len(rows) == len(incoming[p]) == 896, "parent/input geometry")
        for i, row in enumerate(rows):
            require(row["position"] == p and row["index"] == i
                    and int(row["actual"], 16) == actual[p][18][i]
                    and int(row["reference"], 16) == reference[p][18][i],
                    "reviewed layer00-to-layer01 actual/reference join")
        for stage in range(19):
            require(len(actual[p][stage]) == len(reference[p][stage]), "stage geometry")
            for value in actual[p][stage] + reference[p][stage]:
                units(value)

    checkpoint = record(freeze["checkpoint"]["path"])
    require(checkpoint == freeze["checkpoint"], "official checkpoint content identity")
    tensors, embedding_rows = {}, {}
    with safe_open(checkpoint["path"], framework="np") as source:
        for rec in history["checkpoint_tensor_hashes"]:
            name = rec["name"]
            if not any(part in name for part in
                       ("mlp.", "self_attn.o_proj.", "layernorm.weight")):
                continue
            tensor = np.asarray(source.get_tensor(name))
            require(list(tensor.shape) == rec["shape"]
                    and tensor.dtype == np.dtype("<f2" if rec["dtype"] == "F16" else "<i4")
                    and hashlib.sha256(tensor.tobytes()).hexdigest() == rec["sha256"],
                    "official tensor identity: " + name)
            tensors[name] = tensor.view(np.uint16 if rec["dtype"] == "F16" else np.uint32).reshape(-1).tolist()
        embeddings = source.get_slice("model.embed_tokens.weight")
        for p, token in enumerate(boundary.HISTORY):
            embedding_rows[p] = np.asarray(embeddings[token], dtype="<f2").view(np.uint16).tolist()
    historical_vectors = load(historical / "layer00/vectors/manifest.json")
    for vector_set in (historical_vectors, transaction["vectors"]):
        authenticated_names = set()
        for rec in vector_set["tensors"]:
            meta = rec["checkpoint_tensor"]
            name = meta["name"]
            if name not in tensors:
                continue
            serialized = hex_rows(inputs.read(rec["serialized"]["path"]))
            require(serialized == tensors[name], "simulator/checkpoint tensor join: " + name)
            authenticated_names.add(name)
        require(authenticated_names == set(tensors), "incomplete simulator tensor accounting")
    write("checkpoint_tensors.json", history["checkpoint_tensor_hashes"])
    tops = ("ace3_decoder_layer0_token_engine", "ace3_awq_w4a16_projection_engine",
            "ace3_fp16_residual_add_core", "ace3_fp16_silu_gate_core", "ace3_fp16_rmsnorm_core")
    for top in tops:
        path = OUT / "sources" / (top + ".sv")
        for contract in (freeze, historical_freeze):
            rec = next(r for r in contract["sources"] if r["path"].endswith("/" + top + ".sv"))
            require(record(path)["sha256"] == rec["sha256"], "historical source mismatch: " + top)
    require(freeze["public_contract"] in
            (OUT / "sources/ace3_decoder_layer0_token_engine.sv").read_text(),
            "exact public top/port contract changed")
    tool = shutil.which("iverilog")
    require(tool is not None, "host iverilog missing; declared containers need inspection")
    parameters = dict(pf["public_parameters"])
    parameters["ace3_decoder_layer0_token_engine.LAYER_INDEX"] = 0
    parent.command("tool_versions", [sys.executable, "-B", "-c",
                   "import sys,numpy,torch,safetensors; print(sys.version); "
                   "print(numpy.__version__,torch.__version__,safetensors.__version__)"])
    parent.command("iverilog_version", [tool, "-V"])
    write("frozen.json", dict(
        evaluator=record(__file__), parent_authentication=record(OUT / "parent_authentication.json"),
        source_snapshots=record(OUT / "source_snapshots.json"),
        tools=[record(tool), record(sys.executable), record(OUT / "tool_versions.log"),
               record(OUT / "iverilog_version.log")],
        authenticated_inputs=list(inputs.records.values()), supplemental_content_roots=supplemental,
        checkpoint=checkpoint, layer=0, positions=list(range(4)), token_history=boundary.HISTORY,
        public_tops={t: record(OUT / "sources" / (t + ".sv")) for t in tops},
        public_parameters=parameters, public_port_contract=freeze["public_contract"],
        comparison_policy=boundary.POLICY, numerical_policy=pf["numerical_policy"],
        extra_numerical_policy="RMSNorm Q48 RNE mean, floor integer Q24 sqrt, "
        "Q24 quotient RNE, FP16 RNE. Mathematical RMS uses exact squared midpoint "
        "comparisons; SiLU Decimal80 is a diagnostic, not an adopted recurrence.",
        case_counts=dict(token_root=3584, stage00=3584, stage11=3584, stage12=3584,
                         stage13=3584, stage14=19456, stage15=19456, stage16=19456,
                         stage17=3584, stage18=3584),
        projection_terms_per_two_input_path=4*(896*896 + 3*896*4864),
        factorized_accounting="Every input/output coefficient and operand delta is retained, "
        "plus exact G128 and full-output Q48 sums. No pruning. Both residual and SiLU "
        "substitution orders are retained. Projections split local, input, FP16 rounding, "
        "and reference-recurrence terms; reference recurrence is explicit, not a hidden gap.",
        validation="One fresh semantic execution with independent scalar projection, residual, "
        "RMS and SiLU oracles; exact unchanged public contract compilation is not RTL PASS.",
        provenance=QUALIFICATION, RTL_simulations=0, binary64_admission_evaluated=False))
    argv = [tool, "-g2012"]
    for top in tops:
        argv.extend(("-s", top))
    for name, value in parameters.items():
        argv.extend(("-P", f"{name}={value}"))
    parent.command("public_contract_compile", argv + ["-o", str(OUT / "public_contracts.vvp")]
                   + [str(p) for p in sorted((OUT / "sources").glob("*.sv"))])
    parent.command("existing_helper_regression", [sys.executable, "-B", "-m", "unittest",
                   "discover", "-s", str(OUT / "sources"), "-p", "test_trace_lineage_a_layer07_*.py"])
    phases = dict(authentication_freeze_compile_seconds=time.monotonic()-start)
    summaries = {}

    def split(values, names):
        require(len(values) == len(names)+1, "component geometry")
        result = {name: units(values[i])-units(values[i+1]) for i, name in enumerate(names)}
        require(sum(result.values()) == units(values[0])-units(values[-1]), "exact Q24 closure")
        return result

    def row(p, i, a, r, local):
        return dict(position=p, index=i, actual=f"{a:04x}", reference=f"{r:04x}",
                    local_reconstructed=f"{local:04x}", within_gate=accepts(a, r))

    def save(p, name, rows):
        write(f"position{p}_{name}_coordinates.json", rows)
        summaries[f"p{p}_{name}"] = dict(
            coordinates=len(rows),
            bit_differences=sum(r["actual"] != r["reference"] for r in rows),
            material_failures=sum(not r["within_gate"] for r in rows),
            own_input_local_discrepancies=sum(r["actual"] != r["local_reconstructed"] for r in rows))

    phase = time.monotonic()
    for p in range(4):
        a, r = actual[p], reference[p]
        roots = embedding_rows[p]
        require(len(roots) == 896, "official token embedding geometry")
        save(p, "token_root", [
            dict(row(p, i, value, roots[i], value), token_id=boundary.HISTORY[p],
                 delta_q24=units(value)-units(roots[i]),
                 original_reference_consumption_authenticated=False)
            for i, value in enumerate(incoming[p])])
        for stage, la, lr, ra, rr, names in (
                (18, a[12], r[12], a[17], r[17], ("stage12_residual", "stage17_down")),
                (12, incoming[p], roots, a[11], r[11], ("token_root", "stage11_O"))):
            rows = []
            for i in range(896):
                cases = {}
                for label, left, right in (("AA", la[i], ra[i]), ("AR", la[i], rr[i]),
                                           ("RA", lr[i], ra[i]), ("RR", lr[i], rr[i])):
                    result = round_q48((units(left)+units(right)) << 24)
                    oracle, invalid, saturated = residual_add(left, right)
                    require(not (invalid or saturated) and result == oracle, "independent residual oracle")
                    cases[label] = result
                entry = row(p, i, a[stage][i], r[stage][i], cases["AA"])
                entry.update(left_actual=f"{la[i]:04x}", left_reference=f"{lr[i]:04x}",
                             right_actual=f"{ra[i]:04x}", right_reference=f"{rr[i]:04x}",
                             operand_names=names, cases={k: f"{v:04x}" for k, v in cases.items()},
                             actual_FP16_rounding_q24=units(cases["AA"])-units(la[i])-units(ra[i]),
                             reference_FP16_rounding_q24=units(cases["RR"])-units(lr[i])-units(rr[i]))
                entry["components_q24"] = split(
                    [a[stage][i], cases["AA"], cases["AR"], cases["RR"], r[stage][i]],
                    ["local_residual", names[1]+"_after_left", names[0]+"_at_reference_right", "reference_recurrence"])
                entry["opposite_order_components_q24"] = split(
                    [a[stage][i], cases["AA"], cases["RA"], cases["RR"], r[stage][i]],
                    ["local_residual", names[0]+"_after_right", names[1]+"_at_reference_left", "reference_recurrence"])
                rows.append(entry)
            save(p, f"stage{stage}", rows)
        for stage, aa, rr, norm in ((0, incoming[p], roots, "input_layernorm"),
                                     (13, a[12], r[12], "post_attention_layernorm")):
            weights = tensors[f"model.layers.0.{norm}.weight"]
            local_a, detail_a = reconstruct(aa, weights)
            local_r, detail_r = reconstruct(rr, weights)
            for activation, local in ((aa, local_a), (rr, local_r)):
                oracle, _, _ = rmsnorm(activation, weights)
                require(all(not invalid and not saturated and b == expected
                            for (b, invalid, saturated), expected in zip(oracle, local, strict=True)),
                        "independent RMS oracle")
            ideal_a, ideal_r = mathematical_outputs(aa, weights), mathematical_outputs(rr, weights)
            rows = []
            for i in range(896):
                entry = row(p, i, a[stage][i], r[stage][i], local_a[i])
                entry.update(input_actual=f"{aa[i]:04x}", input_reference=f"{rr[i]:04x}",
                             weight=f"{weights[i]:04x}", actual_reduction=detail_a,
                             reference_reduction=detail_r,
                             reference_reconstructed=f"{local_r[i]:04x}")
                entry["components_q24"] = split(
                    [a[stage][i], local_a[i], ideal_a[i], ideal_r[i], local_r[i], r[stage][i]],
                    ["local_rms", "actual_Q_rounding", "input_via_mathematical_RMS",
                     "negative_reference_Q_rounding", "reference_recurrence"])
                rows.append(entry)
            save(p, f"stage{stage:02d}", rows)
        rows = []
        tr = torch.tensor([units(v)/(1 << 24) for v in r[14]], dtype=torch.float64)
        ur = torch.tensor([units(v)/(1 << 24) for v in r[15]], dtype=torch.float64)
        recurrence = (torch.nn.functional.silu(tr)*ur).half().numpy().view(np.uint16).tolist()
        for i in range(4864):
            local = accurate_silu(a[14][i], a[15][i])
            oracle, invalid, saturated = silu_gate_exp(a[14][i], a[15][i])
            require(not (invalid or saturated) and local["bits"] == oracle, "independent SiLU oracle")
            ideal = {}
            with localcontext() as context:
                context.prec = 80
                for label, gate, up in (("AA", a[14][i], a[15][i]), ("AR", a[14][i], r[15][i]),
                                        ("RA", r[14][i], a[15][i]), ("RR", r[14][i], r[15][i])):
                    ideal[label] = nearest_decimal(mathematical_silu(gate, up), (gate ^ up) & 0x8000)
            direct = round_q48(Fraction(local["product_q72"], 1 << 24))
            if direct & 0x7fff == 0:
                direct = (a[14][i] ^ a[15][i]) & 0x8000
            entry = row(p, i, a[16][i], r[16][i], oracle)
            entry.update(gate_actual=f"{a[14][i]:04x}", gate_reference=f"{r[14][i]:04x}",
                         up_actual=f"{a[15][i]:04x}", up_reference=f"{r[15][i]:04x}",
                         ideal_cases={k: f"{v:04x}" for k, v in ideal.items()}, **local)
            chain = [a[16][i], oracle, direct, ideal["AA"], ideal["AR"], ideal["RR"], recurrence[i], r[16][i]]
            names = ["local_silu", "Q72_to_Q24_rounding", "sigmoid_approximation",
                     "up_after_gate", "gate_at_reference_up", "ideal_vs_binary64", "reference_recurrence"]
            entry["components_q24"] = split(chain, names)
            chain[4], names[3:5] = ideal["RA"], ["gate_after_up", "up_at_reference_gate"]
            entry["opposite_order_components_q24"] = split(chain, names)
            rows.append(entry)
        save(p, "stage16", rows)
    phases["root_residual_rms_silu_seconds"] = time.monotonic()-phase
    for stage, input_stage, projection in ((11, 10, "self_attn.o_proj"),
                                           (14, 13, "mlp.gate_proj"),
                                           (15, 13, "mlp.up_proj"),
                                           (17, 16, "mlp.down_proj")):
        phase = time.monotonic()
        ts = {name: tensors[f"model.layers.0.{projection}.{name}"]
              for name in ("qweight", "qzeros", "scales")}
        n, outputs = len(actual[0][input_stage]), len(actual[0][stage])
        groups, words = n//128, outputs//8
        shifts = np.array([0, 16, 4, 20, 8, 24, 12, 28], dtype=np.uint32)
        qw = np.asarray(ts["qweight"], dtype=np.uint32).reshape(n, words)
        qz = np.asarray(ts["qzeros"], dtype=np.uint32).reshape(groups, words)
        w = ((qw[:, :, None] >> shifts) & 15).reshape(n, outputs).astype(np.int64)
        z = ((qz[:, :, None] >> shifts) & 15).reshape(groups, outputs).astype(np.int64)
        scales = np.asarray([units(v) for v in ts["scales"]], dtype=np.int64).reshape(groups, outputs)
        coefficients = (w-np.repeat(z, 128, axis=0))*np.repeat(scales, 128, axis=0)
        with (OUT / f"stage{stage}_coefficient_q24.npy").open("xb") as stream:
            np.save(stream, coefficients, allow_pickle=False)
        for p in range(4):
            aa, rr = actual[p][input_stage], reference[p][input_stage]
            write(f"position{p}_stage{stage}_inputs.json", dict(
                input_stage=input_stage, actual=[f"{v:04x}" for v in aa],
                reference=[f"{v:04x}" for v in rr],
                delta_q24=[units(a)-units(r) for a, r in zip(aa, rr, strict=True)]))
            projected, totals = project_vectors(aa, rr, ts)
            rows = []
            for i in range(outputs):
                group_sums = {}
                for label, activation in (("A", aa), ("R", rr)):
                    oracle = complete_projection_output(
                        activation, ts["qweight"][i//8::words], ts["qzeros"][i//8::words],
                        ts["scales"][i::outputs], i % 8)
                    require(not (oracle[2] or oracle[3]) and oracle[0] == totals[label][i]
                            and oracle[1] == projected[label][i]
                            and all(-(1 << 95) <= v < (1 << 95) for v in oracle[4])
                            and -(1 << 101) <= oracle[0] < (1 << 101),
                            "independent native AWQ scalar oracle")
                    group_sums[label] = oracle[4]
                a, r = actual[p][stage][i], reference[p][stage][i]
                qa, qr = projected["A"][i], projected["R"][i]
                components = dict(
                    local_projection=(units(a)-units(qa)) << 24,
                    actual_FP16_rounding=(units(qa) << 24)-totals["A"][i],
                    input_producer=totals["A"][i]-totals["R"][i],
                    negative_reference_FP16_rounding=totals["R"][i]-(units(qr) << 24),
                    reference_recurrence=(units(qr)-units(r)) << 24)
                require(sum(components.values()) == (units(a)-units(r)) << 24, "projection Q48 closure")
                entry = row(p, i, a, r, qa)
                entry.update(reference_reconstructed=f"{qr:04x}", components_q48=components,
                             actual_sum_q48=totals["A"][i], reference_sum_q48=totals["R"][i],
                             group_sums_q48=group_sums, coefficient_column=i, input_coordinates=n,
                             source_derived_not_observed_bus=True)
                rows.append(entry)
            save(p, f"stage{stage}", rows)
        phases[f"stage{stage}_projection_seconds"] = time.monotonic()-phase
    census = {p: {s: dict(
        coordinates=len(actual[p][s]),
        bit_differences=sum(a != r for a, r in zip(actual[p][s], reference[p][s], strict=True)),
        material_failures=sum(not accepts(a, r) for a, r in zip(actual[p][s], reference[p][s], strict=True)))
        for s in range(19)} for p in range(4)}
    root_differences = sum(summaries[f"p{p}_token_root"]["bit_differences"] for p in range(4))
    local_differences = sum(s["own_input_local_discrepancies"] for s in summaries.values())
    first = {p: next((s for s in range(19) if census[p][s]["bit_differences"]), None) for p in range(4)}
    if root_differences:
        decision = "Token-root/input state differs from authenticated official embeddings."
    elif any(s is not None and s <= 12 for s in first.values()):
        decision = ("Token-root inputs match the official checkpoint. Earliest retained drift "
                    "is in the layer00 attention/O cone (including input RMSNorm stage00), "
                    "before the MLP. This is a boundary, not a unique defective operator.")
    else:
        decision = "No earlier retained attention/input drift; inspect layer00 MLP producers."
    write("result.json", dict(
        status="DIAGNOSTIC_COMPLETE_NO_RTL_REPLAY", decision=decision,
        exact_accounting_closed=True, unexplained_accounting_gaps=0,
        token_root_bit_differences=root_differences, own_input_local_discrepancies=local_differences,
        earliest_retained_differing_stage=first, summaries=summaries, all_stage_census=census,
        next_exact_upstream_boundary="Use the earliest-stage census and local/rounding components "
        "to select an attention-input/RMS or attention/V/O discriminating experiment. "
        "No measured downstream intervention benefit is established by this decomposition.",
        phases_seconds=phases, elapsed_seconds=time.monotonic()-start, provenance=QUALIFICATION,
        original_positions0_1_reference_provenance_authenticated=False,
        intervention_candidate_supported=False, policy_changed=False, reference_injected=False,
        RTL_simulations=0, fresh_RTL_PASS=False, binary64_admission_evaluated=False,
        third_token_claim=False, source_promoted=False, A_B_splice=False,
        review_status="PENDING_NORMAL_HOST_REVIEWER"))
    print(json.dumps(dict(decision=decision, own_input_local_discrepancies=local_differences,
                          token_root_bit_differences=root_differences,
                          earliest_retained_differing_stage=first, phases_seconds=phases)))


STAGE00_PARENT = ROOT / "build/retained_layer00_stage18_producer_split_dc9689d1352f_attempt002"
STAGE00_REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/dc9689d1352f/round-0001.json")
STAGE00_HASHES = {
    "frozen.json": "ed95ba4990bfc4669113a2c80a98ba0082b2a35c3865e21273f2143e38101fa4",
    "result.json": "313201705dc25f36012e768c61c1c6ec3ca615d4827c360a8bc8a63305d24020",
    "artifacts.json": "2c5d43595158227120e8927bd5abbceb05ae4b06f0156fba48cc4f29a09547fb",
}


def stage00_bootstrap():
    receipt = json.loads(STAGE00_REVIEW.read_text())
    require(receipt["kind"] == "round_reviewed_handoff"
            and receipt["producer_role"] == "reviewer"
            and receipt["mission_id"] == "dc9689d1352f"
            and receipt["review"]["status"] == "done"
            and "11 stage00 witnesses" in receipt["review"]["next_action"],
            "reviewed stage00 discriminator parent required")
    for name, digest in STAGE00_HASHES.items():
        require(record(STAGE00_PARENT / name)["sha256"] == digest, "stage00 parent changed")
    manifest = {r["path"]: r for r in json.loads((STAGE00_PARENT / "artifacts.json").read_text())}
    frozen = json.loads((STAGE00_PARENT / "frozen.json").read_text())
    records = []

    def authenticated_copy(path, destination, expected):
        rec = record(path)
        require(rec == expected, "discriminator input changed: " + str(path))
        with destination.open("xb") as stream:
            stream.write(Path(path).read_bytes())
        require(record(destination)["sha256"] == rec["sha256"], "snapshot copy changed")
        records.append(dict(origin=rec, snapshot=record(destination)))

    sources = OUT / "sources"
    sources.mkdir()
    for path in sorted((STAGE00_PARENT / "sources").iterdir()):
        if path.suffix in (".py", ".sv") and path.name != Path(__file__).name:
            authenticated_copy(path, sources / path.name, manifest[str(path)])
    evaluator = Path(__file__).resolve()
    authenticated_copy(evaluator, sources / evaluator.name, record(evaluator))
    data = OUT / "inputs"
    data.mkdir()
    for p in range(4):
        for suffix in ("stage00", "token_root"):
            path = STAGE00_PARENT / f"position{p}_{suffix}_coordinates.json"
            authenticated_copy(path, data / path.name, manifest[str(path)])
    bindings = {r["path"]: r for r in frozen["authenticated_inputs"]}
    current = ROOT / "build/token358_position3_layer0_attempt001/frozen.json"
    authenticated_copy(current, data / "position3_execution_frozen.json", bindings[str(current)])
    execution = json.loads((data / "position3_execution_frozen.json").read_text())
    for rec in execution["sources"]:
        if "/independent_fp16_trajectory_20260906_1133/source/" in rec["path"]:
            name = Path(rec["path"]).name
            if (sources / name).exists():
                require(record(sources / name)["sha256"] == rec["sha256"],
                        "reference helper differs from position3 frozen source")
                require(record(rec["path"]) == rec, "reference source origin changed")
                records.append(dict(origin=rec, snapshot=record(sources / name)))
            else:
                authenticated_copy(rec["path"], sources / name, rec)
    rec = execution["independent_policy"]
    authenticated_copy(rec["path"], data / "independent_policy.json", rec)
    for name in ("layer00_generation0.json", "layer00_generation1.json"):
        path = Path(execution["independent_parent"]["path"]).with_name(name)
        authenticated_copy(path, data / name, bindings[str(path)])
    write("parent_authentication.json", dict(
        parent_roots={n: record(STAGE00_PARENT / n) for n in STAGE00_HASHES},
        genuine_review=record(STAGE00_REVIEW), snapshots=records,
        authentication_scope="Reviewed retained coordinates and their source/consumption ancestry; "
        "no checkpoint reread, no reference-input provenance upgrade, no simulator replay."))
    os.execv(sys.executable, [
        sys.executable, "-B", str(sources / evaluator.name), "--stage00-discriminator",
        "--attempt", str(ATTEMPT), "--measure"])


def stage00_measure():
    import ast
    import math
    from fractions import Fraction

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(OUT / "sources"))
    import numpy as np
    import torch
    import retained_layer01_stage18_producer_split_diagnostic as parent
    from diagnose_lineage_a_position3_boundary import POLICY, POSITIVE, accepts, units
    from fp16_adaptation_oracle import EPSILON_Q48, rmsnorm
    from trace_lineage_a_layer07_downproj import round_q48
    from trace_lineage_a_layer07_rmsnorm import mathematical_outputs, reconstruct

    started = time.monotonic()
    torch.set_num_threads(1)
    parent.OUT = OUT
    frozen = json.loads((STAGE00_PARENT / "frozen.json").read_text())
    policy = json.loads((OUT / "inputs/independent_policy.json").read_text())
    require(frozen["comparison_policy"] == policy["comparison"] == POLICY, "policy changed")
    rows = [json.loads((OUT / f"inputs/position{p}_stage00_coordinates.json").read_text())
            for p in range(4)]
    witnesses = [[r["index"] for r in position if r["actual"] != r["reference"]]
                 for position in rows]
    require([len(w) for w in witnesses] == [3, 6, 1, 1], "retained witness set changed")
    reference_path = OUT / "sources/official_single_decoder_layer.py"
    nodes = [n for n in ast.parse(reference_path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == "_torch_rmsnorm"]
    require(len(nodes) == 1, "one frozen independent RMSNorm definition required")
    namespace = dict(np=np, torch=torch)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reference_path), "exec"), namespace)
    tool = shutil.which("iverilog")
    require(tool is not None, "host iverilog missing; inspect declared containers before retry")
    parent.command("tool_versions", [sys.executable, "-B", "-c",
                   "import sys,numpy,torch; print(sys.version); "
                   "print(numpy.__version__,torch.__version__)"])
    parent.command("iverilog_version", [tool, "-V"])
    top_source = OUT / "sources/ace3_decoder_layer0_token_engine.sv"
    require(frozen["public_port_contract"] in top_source.read_text(), "public contract changed")
    names = [
        "frozen_Q48", "no_Q24_quotient", "no_floor_sqrt", "no_mean_RNE",
        "binary64_epsilon_exact_math", "source_binary64", "retained_reference"]
    write("frozen.json", dict(
        evaluator=record(__file__), parent_authentication=record(OUT / "parent_authentication.json"),
        public_port_contract=frozen["public_port_contract"],
        public_parameters=frozen["public_parameters"], public_tops=frozen["public_tops"],
        tools=[record(tool), record(sys.executable), record(OUT / "tool_versions.log"),
               record(OUT / "iverilog_version.log")],
        positions=list(range(4)), token_history=frozen["token_history"],
        witness_indices=witnesses, all_coordinates=3584, comparison_policy=POLICY,
        frozen_recurrence="Exact Q48 sumsq; epsilon=281474977/2^48; RNE mean; "
        "floor sqrt in Q24; RNE quotient in Q24; FP16 RNE with signed zero.",
        independent_recurrence="Frozen _torch_rmsnorm: FP16 operands decoded to binary64; "
        "pow(2).mean, + binary64 literal 1e-6, rsqrt, x*inverse*weight; single FP16 RNE. "
        "propagated_reference.stage applies FP16 boundary; no normalized-activation FP16 cut.",
        ordered_diagnostic_chain=names,
        independent_oracles="Existing integer RMS oracle; exact rational squared-midpoint "
        "RNE for every sqrt variant; existing mathematical_outputs cross-check; separate "
        "scalar math.fsum/sqrt recurrence checked against frozen Torch source.",
        single_factor_controls=["remove_Q24_quotient_only", "remove_floor_sqrt_only"],
        numerical_policy_changed=False, RTL_simulations=0, provenance=QUALIFICATION))
    argv = [tool, "-g2012"]
    for top in frozen["public_tops"]:
        argv.extend(("-s", top))
    for name, value in frozen["public_parameters"].items():
        argv.extend(("-P", f"{name}={value}"))
    parent.command("public_contract_compile", argv + ["-o", str(OUT / "public_contracts.vvp")]
                   + [str(p) for p in sorted((OUT / "sources").glob("*.sv"))])
    phases = dict(freeze_compile_seconds=time.monotonic() - started)

    def signed_round(total, sign):
        result = round_q48(total)
        return result | sign if result & 0x7fff == 0 else result

    def sqrt_rne(product, mean, sign):
        # Compare squared FP16 midpoints, never a rounded floating-point sqrt.
        target = Fraction(product * product, 1) / mean
        require(target <= POSITIVE[-1] ** 2, "sqrt diagnostic overflow")
        low, high = 0, len(POSITIVE) - 1
        while low < high:
            middle = (low + high) // 2
            if POSITIVE[middle] ** 2 < target:
                low = middle + 1
            else:
                high = middle
        upper, lower = low, max(0, low - 1)
        midpoint = (POSITIVE[lower] + POSITIVE[upper]) ** 2
        bits = upper if 4 * target > midpoint else lower
        if 4 * target == midpoint:
            bits = lower if lower % 2 == 0 else upper
        return bits | sign

    def sqrt_q24_rne(product, mean, sign):
        target = Fraction(product * product, 1) / mean
        lower = math.isqrt(target.numerator // target.denominator)
        compare = 4 * target - (2 * lower + 1) ** 2
        magnitude = lower + int(compare > 0 or (compare == 0 and lower % 2 == 1))
        return signed_round((-magnitude if sign else magnitude) << 24, sign)

    phase = time.monotonic()
    all_rows, reductions = [], []
    for p, position in enumerate(rows):
        require(len(position) == 896, "stage00 geometry")
        roots = json.loads((OUT / f"inputs/position{p}_token_root_coordinates.json").read_text())
        activation, weights = [], []
        for i, (r, root) in enumerate(zip(position, roots, strict=True)):
            require(r["position"] == p and r["index"] == root["index"] == i
                    and root["position"] == p
                    and root["actual"] == root["reference"] == r["input_actual"] == r["input_reference"],
                    "reviewed consumed input/official embedding coordinate join")
            activation.append(int(r["input_actual"], 16))
            weights.append(int(r["weight"], 16))
        local, detail = reconstruct(activation, weights)
        oracle, mean, root = rmsnorm(activation, weights)
        require(all(not bad and not sat and b == expected
                    for (b, bad, sat), expected in zip(oracle, local, strict=True)),
                "independent frozen integer oracle disagrees")
        require(mean == detail["mean_q48"] and root == detail["rms_q24"], "reduction mismatch")
        x = np.asarray(activation, dtype="<u2").view("<f2").astype(np.float64)
        w = np.asarray(weights, dtype="<u2").view("<f2")
        tensor = namespace["_torch_rmsnorm"](torch.from_numpy(x.reshape(1, -1)), w)
        require(bool(torch.isfinite(tensor).all()), "nonfinite independent RMS output")
        reference = tensor.half().numpy().view(np.uint16).reshape(-1).tolist()
        variance = math.fsum(float(v) ** 2 for v in x) / len(x)
        inverse = 1.0 / math.sqrt(variance + 1e-6)
        scalar_values = np.asarray([float(v) * inverse * float(g) for v, g in zip(x, w)],
                                   dtype=np.float16).view(np.uint16).tolist()
        require(reference == scalar_values, "independent scalar/source recurrence disagreement")
        ideal = mathematical_outputs(activation, weights)
        exact_mean = Fraction(detail["sumsq_q48"], 896) + EPSILON_Q48
        reference_mean = (Fraction(detail["sumsq_q48"], 896)
                          + Fraction.from_float(1e-6) * (1 << 48))
        reductions.append(dict(
            position=p, **detail, exact_mean_q48=str(exact_mean),
            mean_RNE_error_q48=str(mean - exact_mean),
            floor_sqrt_remainder_q48=mean - root * root,
            reference_mean_q48=str(reference_mean),
            reference_variance_binary64_hex=variance.hex(),
            reference_inverse_binary64_hex=inverse.hex()))
        for i, r in enumerate(position):
            a, g = activation[i], weights[i]
            product = units(a) * units(g)
            sign = (a ^ g) & 0x8000
            quotient = Fraction(product, root)
            direct = signed_round(quotient * (1 << 24), sign)
            ideal_mean = sqrt_rne(product, Fraction(mean), sign)
            exact = sqrt_rne(product, exact_mean, sign)
            require(exact == ideal[i], "independent exact mathematical RMS disagreement")
            ideal_epsilon = sqrt_rne(product, reference_mean, sign)
            require(local[i] == int(r["actual"], 16) == int(r["local_reconstructed"], 16)
                    and detail == r["actual_reduction"] == r["reference_reduction"],
                    "retained own-input recurrence mismatch")
            retained = int(r["reference"], 16)
            values = [local[i], direct, ideal_mean, exact, ideal_epsilon, reference[i], retained]
            components = {f"{names[j]}_minus_{names[j+1]}": units(values[j])-units(values[j+1])
                          for j in range(len(values)-1)}
            require(sum(components.values()) == units(local[i])-units(retained), "accounting gap")
            row = dict(position=p, index=i, witness=i in witnesses[p],
                       activation=f"{a:04x}", weight=f"{g:04x}",
                       outputs={k: f"{v:04x}" for k, v in zip(names, values, strict=True)},
                       remove_floor_sqrt_only=f"{sqrt_q24_rne(product, Fraction(mean), sign):04x}",
                       components_q24=components, quotient_q24=str(quotient),
                       quotient_RNE_q24=round(quotient),
                       quotient_RNE_error_q24=str(round(quotient)-quotient),
                       retained_within_gate=accepts(local[i], retained),
                       source_matches_retained=reference[i] == retained)
            all_rows.append(row)
    phases["recurrence_discrimination_seconds"] = time.monotonic() - phase
    summaries = {}
    for name in names[:-1] + ["remove_floor_sqrt_only"]:
        selected = [r["remove_floor_sqrt_only"] if name == "remove_floor_sqrt_only"
                    else r["outputs"][name] for r in all_rows]
        summaries[name] = dict(
            witness_mismatches=sum(r["witness"] and v != r["outputs"]["retained_reference"]
                                   for r, v in zip(all_rows, selected, strict=True)),
            all_coordinate_mismatches=sum(v != r["outputs"]["retained_reference"]
                                         for r, v in zip(all_rows, selected, strict=True)),
            material_failures=sum(not accepts(int(v, 16), int(r["outputs"]["retained_reference"], 16))
                                  for r, v in zip(all_rows, selected, strict=True)))
    unresolved = sum(not r["source_matches_retained"] for r in all_rows)
    decision = (
        "All retained stage00 reference bytes are reproduced by the frozen independent "
        "binary64 recurrence on authenticated official-root operands. The ordered exact "
        "decomposition and single-factor controls quantify arithmetic-boundary differences, "
        "not an input-provenance repair or downstream causal benefit."
        if not unresolved else
        "The source-bound independent recurrence leaves explicit retained-reference residuals; "
        "original reference operand/producer provenance remains the next boundary.")
    write("reductions.json", reductions)
    write("coordinates.json", all_rows)
    write("witnesses.json", [r for r in all_rows if r["witness"]])
    write("result.json", dict(
        status="DIAGNOSTIC_COMPLETE_NO_RTL_REPLAY", decision=decision,
        accounted_witnesses=sum(len(w) for w in witnesses), coordinates=len(all_rows),
        summaries=summaries, exact_accounting_closed=True, unexplained_accounting_gaps=0,
        source_recurrence_retained_residuals=unresolved, provenance=QUALIFICATION,
        original_reference_input_consumption_authenticated=False,
        next_boundary="Use measured stage00 operator differences in a separately scoped "
        "downstream sensitivity experiment with compatible affected hidden/K/V dependencies; "
        "no production arithmetic change or full traversal is justified by this local comparison."
        if not unresolved else "Resolve the explicit reference recurrence/provenance residuals.",
        general_operator_candidate="Single-round mathematical RMSNorm is only a local candidate "
        "if its full-coordinate residual count is zero; it is not adopted or a demonstrated "
        "downstream repair. Frozen Q48 and all numerical gates remain unchanged.",
        downstream_intervention_benefit_measured=False, policy_changed=False,
        RTL_simulations=0, fresh_RTL_PASS=False, binary64_admission_evaluated=False,
        third_token_claim=False, reference_injected=False, source_promoted=False,
        phases_seconds=phases, elapsed_seconds=time.monotonic()-started,
        review_status="PENDING_NORMAL_HOST_REVIEWER"))
    print(json.dumps(dict(decision=decision, witnesses=11, summaries=summaries,
                          source_recurrence_retained_residuals=unresolved, phases_seconds=phases)))


SENSITIVITY_PARENT = ROOT / (
    "build/retained_layer00_stage00_rmsnorm_witness_discriminator_d004a401c237_attempt001")
SENSITIVITY_REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/d004a401c237/round-0001.json")
PROPAGATION_PARENT = ROOT / (
    "build/retained_layer00_rmsnorm_candidate_downstream_sensitivity_7b2e1e6dd113_attempt001")
PROPAGATION_REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/7b2e1e6dd113/round-0001.json")
LAYER02_PARENT = ROOT / (
    "build/retained_layer01_rmsnorm_candidate_propagation_228864847a63_attempt001")
LAYER02_BASELINE = ROOT / (
    "build/retained_layer02_stage18_producer_split_01ce0c7e0c31_attempt001")
LAYER02_REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/228864847a63/round-0001.json")


def layer02_bootstrap():
    receipt = json.loads(LAYER02_REVIEW.read_text())
    require(receipt["kind"] == "round_reviewed_handoff"
            and receipt["producer_role"] == "reviewer"
            and receipt["mission_id"] == "228864847a63"
            and receipt["review"]["status"] == "done"
            and "layer02 stage00" in receipt["review"]["next_action"],
            "reviewed paired layer01 propagation required")
    roots, counts = {}, {}
    for label, directory in (("paired_layer01", LAYER02_PARENT),
                             ("retained_layer02", LAYER02_BASELINE)):
        manifest = json.loads((directory / "artifacts.json").read_text())
        for rec in manifest:
            require(record(rec["path"]) == rec, label + " artifact changed: " + rec["path"])
        roots[label] = {name: record(directory / name) for name in
                        ("artifacts.json", "frozen.json", "result.json")}
        counts[label] = len(manifest)
    sources = OUT / "sources"
    sources.mkdir()
    snapshots = []
    retained_tops = json.loads((LAYER02_BASELINE / "frozen.json").read_text())["public_tops"]
    for path in sorted((LAYER02_PARENT / "sources").iterdir()):
        if path.suffix not in (".py", ".sv") or path.name == Path(__file__).name:
            continue
        if path.stem in retained_tops and path.suffix == ".sv":
            retained = LAYER02_BASELINE / "sources" / path.name
            require(record(path)["sha256"] == record(retained)["sha256"],
                    "layer02 retained arithmetic source mismatch: " + path.name)
        target = sources / path.name
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        snapshots.append(dict(origin=record(path), snapshot=record(target)))
    target = sources / Path(__file__).name
    with target.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    snapshots.append(dict(origin=record(__file__), snapshot=record(target)))
    write("parent_authentication.json", dict(
        review=record(LAYER02_REVIEW),
        parent_mission=record(LAYER02_REVIEW.with_name("mission.json")),
        mission=record(LAYER02_REVIEW.parent.parent / "2cbdd5f727db/mission.json"),
        roots=roots, authenticated_artifacts=counts, sources=snapshots,
        source_compatibility="Layer02's retained archive covers only its public producer "
        "tops; these are compared byte-for-byte. The remaining full source archive is "
        "inherited from reviewed layer01. Every reconstructed layer02 stage must additionally "
        "match retained RTL bit-for-bit; no source equivalence is inferred from missing files.",
        historical_pending_field_is_not_missing_review=True))
    os.execv(sys.executable, [sys.executable, "-B", str(target), "--layer02-propagation",
                             "--attempt", str(ATTEMPT), "--measure"])


def layer02_measure():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(OUT / "sources"))
    import numpy as np
    import torch
    import diagnose_lineage_a_position3_boundary as boundary
    from trace_lineage_a_layer07_downproj import framed_hidden, hex_rows
    from trace_lineage_a_layer07_score_softmax_v import array

    started = time.monotonic()
    torch.set_num_threads(1)
    boundary.ROOT, boundary.BUILD = ROOT, ROOT / "build"
    inputs = boundary.Inputs()
    inputs.load(OUT / "parent_authentication.json", root=True)
    inputs.load(LAYER02_PARENT / "artifacts.json")
    pf = inputs.load(LAYER02_PARENT / "frozen.json")
    reviewed = inputs.load(LAYER02_PARENT / "result.json")
    require(pf["layer"] == 1 and pf["positions"] == list(range(4))
            and pf["comparison_policy"] == boundary.POLICY
            and pf["token_history"] == boundary.HISTORY
            and reviewed["stage_comparisons"] == 76
            and reviewed["frozen_control_bit_differences"] == 0
            and reviewed["candidate_material_failures"] == 0
            and reviewed["frozen_material_failures"] == 0
            and not reviewed["source_promoted"], "reviewed layer01 scope/policy changed")
    inputs.load(LAYER02_BASELINE / "artifacts.json")
    baseline = inputs.load(LAYER02_BASELINE / "frozen.json")
    require(baseline["layer"] == 2 and baseline["comparison_policy"] == boundary.POLICY
            and baseline["token_history"] == boundary.HISTORY
            and inputs.load(LAYER02_BASELINE / "result.json")["exact_accounting_closed"],
            "retained layer02 accounting/policy changed")
    prefix = ROOT / "build/token358_position3_full_continuation_attempt001"
    original = inputs.load(prefix / "frozen.json")
    require(inputs.records[str(prefix / "frozen.json")]["sha256"] ==
            "f81181054c32a22ac303af8416f2b41e733ea265b90cdcba9c71201757d6b599",
            "retained layer02 source freeze changed")
    pre = inputs.load(prefix / "layer02/prepared.json")
    transaction = inputs.load(prefix / "layer02/transaction.json")
    kv = pre["kv_parent"]
    old = inputs.load(kv["layer_result"]["path"])
    require(transaction["layer_index"] == old["layer_index"] == kv["layer_index"] == 2
            and transaction["position"] == 3 and kv["valid_positions"] == [0, 1, 2]
            and pre["binary"] == kv["live_binary"]
            and "/model24_persistent_kv_selected_token_corrected_q_attempt005/" in kv["state"]["path"],
            "retained layer02 identity/profile mismatch")

    def bits(rec, framed=False):
        inputs.register(rec)
        result = (framed_hidden if framed else hex_rows)(inputs.read(rec["path"]))
        digest = hashlib.sha256(np.asarray(result, dtype="<u2").tobytes()).hexdigest()
        require(digest == rec.get("semantic_sha256", digest), "FP16 semantic input mismatch")
        return result

    actual = {p: boundary.decode_trace(inputs.read(rec["path"]), p)
              for p, rec in enumerate(kv["actual_kv_traces"])}
    actual[3] = boundary.decode_trace(inputs.read(
        Path(transaction["output"]["path"]).with_name("trace.hex")), 3)
    require(bits(transaction["output"], True) == actual[3][18]
            and transaction["input"]["sha256"] == pre["input_hidden"]["semantic_sha256"],
            "layer02 position3 trace/output/input join")
    incoming, candidate_incoming = {}, {}
    for p in range(4):
        upstream = inputs.load(LAYER02_PARENT / f"position{p}_stages.json")
        incoming[p] = upstream["frozen_Q48"]["18"]
        candidate_incoming[p] = upstream["single_round"]["18"]
        consumed = bits(old["positions"][p]["vectors"]["input"] if p < 3
                        else pre["input_hidden"], True)
        require(incoming[p] == consumed and len(candidate_incoming[p]) == 896,
                "reviewed layer01 output/layer02 consumed hidden join")
        if p < 3:
            require(bits(old["positions"][p]["output"], True) == actual[p][18],
                    "historical layer02 trace/final join")
    history = inputs.load(pre["independent_parent"]["path"])
    early = inputs.load(Path(pre["independent_parent"]["path"]).with_name(
        "layer02_generation0.json"))
    require(early["checkpoint_tensor_hashes"] == history["checkpoint_tensor_hashes"],
            "layer02 independent checkpoint identity changed")
    reference = {p: {} for p in range(4)}
    for obj, positions in ((early, [0, 1]), (history, [2])):
        require(obj["layer"] == 2 and obj["positions"] == positions
                and obj["history"] == boundary.HISTORY[:max(positions)+1],
                "layer02 independent reference identity")
        for rec in obj["stages"]:
            p = int(Path(rec["path"]).parent.name[8:])
            stage = int(Path(rec["path"]).stem[5:])
            reference[p][stage] = array(inputs, rec).reshape(-1).tolist()
    for stage in range(19):
        reference[3][stage] = bits(pre["independent_stages"][str(stage)])
    require(all(set(reference[p]) == set(actual[p]) == set(range(19)) for p in range(4)),
            "complete layer02 independent and retained stages required")
    tensor_hashes = {r["name"]: r for r in history["checkpoint_tensor_hashes"]}
    tensors = {}
    for rec in pre["vectors"]["tensors"]:
        meta = rec["checkpoint_tensor"]
        name = meta["name"]
        data = hex_rows(inputs.read(rec["serialized"]["path"]))
        payload = np.asarray(data, dtype="<u2" if meta["dtype"] == "F16" else "<u4")
        require(name.startswith("model.layers.2.")
                and all(meta[key] == tensor_hashes[name][key]
                        for key in ("name", "dtype", "shape", "sha256"))
                and payload.nbytes == meta["bytes"]
                and hashlib.sha256(payload.tobytes()).hexdigest() == meta["sha256"],
                "layer02 simulator/independent tensor join: " + name)
        tensors[name.removeprefix("model.layers.2.")] = data
    require(len(tensors) == 26, "complete layer02 tensors required")
    pf["public_parameters"] = dict(pf["public_parameters"])
    pf["public_parameters"]["ace3_decoder_layer0_token_engine.LAYER_INDEX"] = 2
    write("input_lineages.json", dict(
        frozen_Q48=incoming, single_round=candidate_incoming,
        parent=record(LAYER02_PARENT / "artifacts.json"),
        retained_execution_freeze=record(prefix / "frozen.json"),
        retained_public_interfaces=original["public_interfaces"],
        intervention="Inherit only the reviewed layer00 stage00 single-round change via "
        "each arm's own layer01 stage18. Both layer02 RMSNorm operators remain Q48/RNE.",
        reused="No layer00/layer01 recomputation. Authenticated layer02 weights, sources "
        "and independent retained references remain unchanged.",
        invalidated="Changed incoming hidden affects stage00 and stage12 directly. "
        "Only bit-changed stage dependencies propagate; changed stage06 K and stage07 V "
        "affect subsequent positions' score and AV respectively. All layer02 software "
        "K/V are arm-local. No later-layer cache or opaque simulator state is imported."))
    propagate_sensitivity(inputs, pf, actual, reference, incoming, candidate_incoming,
                          tensors, [], started, layer=2)


def layer01_bootstrap():
    receipt = json.loads(PROPAGATION_REVIEW.read_text())
    require(receipt["kind"] == "round_reviewed_handoff"
            and receipt["producer_role"] == "reviewer"
            and receipt["mission_id"] == "7b2e1e6dd113"
            and receipt["review"]["status"] == "done"
            and "layer01" in receipt["review"]["next_action"],
            "reviewed layer00 propagation parent required")
    manifest = json.loads((PROPAGATION_PARENT / "artifacts.json").read_text())
    for rec in manifest:
        require(record(rec["path"]) == rec, "reviewed propagation artifact changed")
    for name, digest in PARENT_HASHES.items():
        require(record(PARENT / name)["sha256"] == digest,
                "retained layer01 baseline changed: " + name)
    sources = OUT / "sources"
    sources.mkdir()
    snapshots = []
    for path in sorted((PROPAGATION_PARENT / "sources").iterdir()):
        if path.suffix not in (".py", ".sv") or path.name == Path(__file__).name:
            continue
        target = sources / path.name
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        snapshots.append(dict(origin=record(path), snapshot=record(target)))
    target = sources / Path(__file__).name
    with target.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    snapshots.append(dict(origin=record(__file__), snapshot=record(target)))
    write("parent_authentication.json", dict(
        review=record(PROPAGATION_REVIEW),
        mission=record(PROPAGATION_REVIEW.with_name("mission.json")),
        reviewed_parent={name: record(PROPAGATION_PARENT / name) for name in
                         ("artifacts.json", "frozen.json", "result.json")},
        baseline_roots={name: record(PARENT / name) for name in PARENT_HASHES},
        authenticated_parent_artifacts=len(manifest), sources=snapshots,
        historical_pending_field_is_not_missing_review=True))
    os.execv(sys.executable, [sys.executable, "-B", str(target), "--layer01-propagation",
                             "--attempt", str(ATTEMPT), "--measure"])


def layer01_measure():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(OUT / "sources"))
    import numpy as np
    import torch
    import diagnose_lineage_a_position3_boundary as boundary
    import retained_layer01_stage18_producer_split_diagnostic as parent
    from trace_lineage_a_layer07_downproj import framed_hidden, hex_rows
    from trace_lineage_a_layer07_score_softmax_v import array

    started = time.monotonic()
    torch.set_num_threads(1)
    boundary.ROOT, boundary.BUILD = ROOT, ROOT / "build"
    inputs = boundary.Inputs()
    inputs.load(OUT / "parent_authentication.json", root=True)
    inputs.load(PROPAGATION_PARENT / "artifacts.json")
    pf = inputs.load(PROPAGATION_PARENT / "frozen.json")
    reviewed = inputs.load(PROPAGATION_PARENT / "result.json")
    require(pf["layer"] == 0 and pf["positions"] == list(range(4))
            and pf["comparison_policy"] == boundary.POLICY
            and pf["token_history"] == boundary.HISTORY
            and reviewed["frozen_control_bit_differences"] == 0
            and reviewed["candidate_material_failures"] == 0
            and reviewed["frozen_material_failures"] == 0
            and not reviewed["source_promoted"], "reviewed layer00 scope/policy changed")
    inputs.load(PARENT / "artifacts.json")
    baseline = inputs.load(PARENT / "frozen.json")
    require(baseline["comparison_policy"] == boundary.POLICY
            and inputs.load(PARENT / "result.json")["exact_accounting_closed"],
            "retained layer01 accounting/policy changed")
    pre = inputs.load(parent.PACKAGE / "launch_package.json")
    kv = pre["kv_parent"]
    old = inputs.load(kv["layer_result"]["path"])
    require(pre["layer_index"] == old["layer_index"] == kv["layer_index"] == 1
            and pre["position"] == 3 and pre["token_history"] == boundary.HISTORY
            and pre["parameters"] == {"ACCURATE_SILU": 1, "LAYER_INDEX": 1}
            and kv["valid_positions"] == [0, 1, 2],
            "retained layer01 identity/profile mismatch")

    def bits(rec, framed=False):
        inputs.register(rec)
        result = (framed_hidden if framed else hex_rows)(inputs.read(rec["path"]))
        digest = hashlib.sha256(np.asarray(result, dtype="<u2").tobytes()).hexdigest()
        require(digest == rec.get("semantic_sha256", digest), "FP16 semantic input mismatch")
        return result

    execution = parent.PACKAGE.parent / "future_execution/position003"
    actual = {p: boundary.decode_trace(inputs.read(rec["path"]), p)
              for p, rec in enumerate(kv["actual_kv_traces"])}
    actual[3] = boundary.decode_trace(inputs.read(execution / "raw/trace.hex"), 3)
    require(bits(next(r for r in baseline["authenticated_inputs"]
                      if r["path"] == str(execution / "raw/final.hex")), True) == actual[3][18],
            "layer01 position3 retained trace/final join")
    incoming, candidate_incoming = {}, {}
    for p in range(4):
        upstream = inputs.load(PROPAGATION_PARENT / f"position{p}_stages.json")
        incoming[p] = upstream["frozen_Q48"]["18"]
        candidate_incoming[p] = upstream["single_round"]["18"]
        consumed = bits(old["positions"][p]["vectors"]["input"] if p < 3
                        else pre["input_hidden"], True)
        require(incoming[p] == consumed and len(candidate_incoming[p]) == 896,
                "reviewed layer00 output/layer01 consumed hidden join")
        if p < 3:
            require(bits(old["positions"][p]["output"], True) == actual[p][18],
                    "historical layer01 trace/final join")
    history = inputs.load(pre["independent_parent"]["path"])
    early = inputs.load(Path(pre["independent_parent"]["path"]).with_name(
        "layer01_generation0.json"))
    require(early["checkpoint_tensor_hashes"] == history["checkpoint_tensor_hashes"],
            "independent checkpoint identity changed")
    reference = {p: {} for p in range(4)}
    for obj, positions in ((early, [0, 1]), (history, [2])):
        require(obj["layer"] == 1 and obj["positions"] == positions
                and obj["history"] == boundary.HISTORY[:max(positions)+1],
                "layer01 independent reference identity")
        for rec in obj["stages"]:
            p = int(Path(rec["path"]).parent.name[8:])
            stage = int(Path(rec["path"]).stem[5:])
            reference[p][stage] = array(inputs, rec).reshape(-1).tolist()
    for stage in range(19):
        reference[3][stage] = bits(pre["independent_expected_stages"][str(stage)])
    require(all(set(reference[p]) == set(actual[p]) == set(range(19)) for p in range(4)),
            "complete layer01 independent and retained stages required")
    tensor_hashes = {r["name"]: r for r in history["checkpoint_tensor_hashes"]}
    tensors = {}
    for rec in pre["vectors"]["tensors"]:
        meta = rec["checkpoint_tensor"]
        name = meta["name"]
        data = hex_rows(inputs.read(rec["serialized"]["path"]))
        payload = np.asarray(data, dtype="<u2" if meta["dtype"] == "F16" else "<u4")
        require(all(meta[key] == tensor_hashes[name][key]
                    for key in ("name", "dtype", "shape", "sha256"))
                and payload.nbytes == meta["bytes"]
                and hashlib.sha256(payload.tobytes()).hexdigest() == meta["sha256"],
                "layer01 simulator/independent tensor join: " + name)
        tensors[name.removeprefix("model.layers.1.")] = data
    require(len(tensors) == 26, "complete layer01 tensors required")
    for top in pf["public_tops"]:
        source = next(r for r in pre["sources"] if r["path"].endswith("/" + top + ".sv"))
        require(record(OUT / "sources" / (top + ".sv"))["sha256"] == source["sha256"],
                "layer01 frozen public source mismatch: " + top)
    pf["public_parameters"] = dict(pf["public_parameters"])
    pf["public_parameters"]["ace3_decoder_layer0_token_engine.LAYER_INDEX"] = 1
    write("input_lineages.json", dict(
        frozen_Q48=incoming, single_round=candidate_incoming,
        parent=record(PROPAGATION_PARENT / "artifacts.json"),
        intervention="Inherit only the reviewed layer00 stage00 change. Layer01 stage00 "
        "and stage13 retain Q48/RNE; do not apply mathematical RMSNorm again.",
        reused="All layer00 arrays; no layer00 recomputation. Immutable layer01 weights, "
        "coefficients, arithmetic sources and independent retained reference.",
        invalidated="Bit-changed layer01 hidden operands invalidate their downstream "
        "operators and any changed K/V entries for subsequent fixed-history positions. "
        "No historical opaque simulator state or later-layer cache is imported."))
    propagate_sensitivity(inputs, pf, actual, reference, incoming, candidate_incoming,
                          tensors, [], started, layer=1)


def sensitivity_bootstrap():
    receipt = json.loads(SENSITIVITY_REVIEW.read_text())
    require(receipt["kind"] == "round_reviewed_handoff"
            and receipt["producer_role"] == "reviewer"
            and receipt["mission_id"] == "d004a401c237"
            and receipt["review"]["status"] == "done", "reviewed RMSNorm parent required")
    manifest = {r["path"]: r for r in
                json.loads((SENSITIVITY_PARENT / "artifacts.json").read_text())}
    sources = OUT / "sources"
    sources.mkdir()
    snapshots = []
    for path in sorted((SENSITIVITY_PARENT / "sources").iterdir()):
        if path.suffix not in (".py", ".sv") or path.name == Path(__file__).name:
            continue
        require(record(path) == manifest[str(path)], "reviewed helper changed")
        target = sources / path.name
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        snapshots.append(dict(origin=manifest[str(path)], snapshot=record(target)))
    evaluator = Path(__file__).resolve()
    target = sources / evaluator.name
    with target.open("xb") as stream:
        stream.write(evaluator.read_bytes())
    snapshots.append(dict(origin=record(evaluator), snapshot=record(target)))
    for name in ("result.json", "frozen.json", "parent_authentication.json", "coordinates.json"):
        require(record(SENSITIVITY_PARENT / name) == manifest[str(SENSITIVITY_PARENT / name)],
                "reviewed RMSNorm input changed: " + name)
    ancestry = json.loads((SENSITIVITY_PARENT / "parent_authentication.json").read_text())
    for rec in ancestry["parent_roots"].values():
        require(record(rec["path"]) == rec, "reviewed layer00 ancestry changed")
    write("parent_authentication.json", dict(
        review=record(SENSITIVITY_REVIEW),
        reviewed_parent={name: record(SENSITIVITY_PARENT / name) for name in
                         ("artifacts.json", "frozen.json", "result.json", "coordinates.json",
                          "parent_authentication.json")},
        ancestor_roots=ancestry["parent_roots"], sources=snapshots))
    os.execv(sys.executable, [sys.executable, "-B", str(target), "--rmsnorm-downstream",
                             "--attempt", str(ATTEMPT), "--measure"])


def sensitivity_measure():
    from decimal import Decimal, localcontext

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(OUT / "sources"))
    import numpy as np
    import torch
    import diagnose_lineage_a_position3_boundary as boundary
    import retained_layer01_stage18_producer_split_diagnostic as parent
    from exact_projection import project_vectors
    from fp16_adaptation_oracle import EPSILON_Q48, residual_add, rmsnorm, silu_gate_exp
    from projection_oracle import complete_projection_output
    from qwen2_rope_oracle import qwen2_coefficient, rotate_pair
    from attention_oracle import attention_score, attention_softmax, attention_value
    from trace_lineage_a_layer07_downproj import framed_hidden, hex_rows, round_q48
    from trace_lineage_a_layer07_rmsnorm import mathematical_outputs
    from trace_lineage_a_layer07_score_softmax_v import array
    from trace_lineage_a_layer07_silu import nearest_decimal

    started = time.monotonic()
    torch.set_num_threads(1)
    parent.OUT = OUT
    boundary.ROOT, boundary.BUILD = ROOT, ROOT / "build"
    inputs = boundary.Inputs()
    auth = inputs.load(OUT / "parent_authentication.json", root=True)
    pf = inputs.load(auth["ancestor_roots"]["frozen.json"]["path"])
    inputs.register(pf["authenticated_inputs"])
    reviewed = inputs.load(SENSITIVITY_PARENT / "result.json")
    require(reviewed["source_recurrence_retained_residuals"] == 0
            and reviewed["summaries"]["no_mean_RNE"]["all_coordinate_mismatches"] == 0,
            "reviewed single-round candidate changed")
    reviewed_coordinates = inputs.load(SENSITIVITY_PARENT / "coordinates.json")
    current = ROOT / "build/token358_position3_layer0_attempt001"
    historical = ROOT / "build/model24_persistent_kv_selected_token_corrected_q_attempt005"
    old = inputs.load(historical / "layer00/result.json")
    later = inputs.load(current / "result.json")
    execution = inputs.load(current / "frozen.json")
    transaction = inputs.load(later["transaction"]["path"])
    history = inputs.load(execution["independent_parent"]["path"])
    early = inputs.load(Path(execution["independent_parent"]["path"]).with_name(
        "layer00_generation0.json"))
    require(pf["comparison_policy"] == boundary.POLICY
            and pf["token_history"] == later["token_history"] == boundary.HISTORY
            and execution["parameters"] == {"ACCURATE_SILU": 1, "LAYER_INDEX": 0},
            "retained profile/history changed")

    def bits(rec, framed=False):
        inputs.register(rec)
        result = (framed_hidden if framed else hex_rows)(inputs.read(rec["path"]))
        digest = hashlib.sha256(np.asarray(result, dtype="<u2").tobytes()).hexdigest()
        require(digest == rec.get("semantic_sha256", digest), "semantic input mismatch")
        return result

    actual, reference, incoming = {}, {p: {} for p in range(4)}, {}
    for p in range(3):
        pos = old["positions"][p]
        require(pos["position"] == p and pos["layer_index"] == 0, "historical identity")
        actual[p] = boundary.decode_trace(inputs.read(pos["raw"]["trace"]["path"]), p)
        incoming[p] = bits(pos["vectors"]["input"], True)
        require(bits(pos["output"], True) == actual[p][18], "historical hidden join")
        require(hashlib.sha256(np.asarray(incoming[p], dtype="<u2").tobytes()).hexdigest()
                == pos["input"]["sha256"], "consumed embedding identity")
    incoming[3] = bits(transaction["vectors"]["input"], True)
    require(incoming[3] == bits(execution["embedding"]["file"], True),
            "position3 consumed embedding join")
    actual[3] = boundary.decode_trace(inputs.read(
        later["independent_comparisons"][0]["actual_trace"]["path"]), 3)
    require(actual[3][18] == bits(later["output_hidden"], True), "position3 hidden join")
    for obj, positions in ((early, [0, 1]), (history, [2])):
        require(obj["layer"] == 0 and obj["positions"] == positions
                and obj["history"] == boundary.HISTORY[:max(positions)+1],
                "independent reference position/history changed")
        for rec in obj["stages"]:
            p = int(Path(rec["path"]).parent.name[8:])
            stage = int(Path(rec["path"]).stem[5:])
            reference[p][stage] = array(inputs, rec).reshape(-1).tolist()
    for comparison in later["independent_comparisons"]:
        require(comparison["layer"] == 0 and comparison["position"] == 3, "reference identity")
        reference[3][comparison["stage"]] = bits(comparison["independent_reference"])
    tensors = {}
    tensor_hashes = {r["name"]: r for r in history["checkpoint_tensor_hashes"]}
    require(early["checkpoint_tensor_hashes"] == history["checkpoint_tensor_hashes"],
            "reference checkpoint identity changed")
    for rec in transaction["vectors"]["tensors"]:
        meta = rec["checkpoint_tensor"]
        name = meta["name"]
        data = hex_rows(inputs.read(rec["serialized"]["path"]))
        dtype = "<u2" if meta["dtype"] == "F16" else "<u4"
        payload = np.asarray(data, dtype=dtype)
        require(all(meta[key] == tensor_hashes[name][key]
                    for key in ("name", "dtype", "shape", "sha256"))
                and payload.nbytes == meta["bytes"]
                and hashlib.sha256(payload.tobytes()).hexdigest() == meta["sha256"],
                "simulator/independent tensor join: " + name)
        tensors[name.removeprefix("model.layers.0.")] = data
    require(len(tensors) == 26, "complete layer00 tensors required")
    for r in reviewed_coordinates:
        p, i = r["position"], r["index"]
        require(int(r["activation"], 16) == incoming[p][i]
                and int(r["weight"], 16) == tensors["input_layernorm.weight"][i]
                and int(r["outputs"]["frozen_Q48"], 16) == actual[p][0][i]
                and int(r["outputs"]["retained_reference"], 16) == reference[p][0][i],
                "reviewed candidate operand/control join")

    propagate_sensitivity(inputs, pf, actual, reference, incoming, incoming,
                          tensors, reviewed_coordinates, started, layer=0)


def propagate_sensitivity(inputs, pf, actual, reference, incoming, candidate_incoming,
                          tensors, reviewed_coordinates, started, *, layer):
    from decimal import Decimal, localcontext
    import diagnose_lineage_a_position3_boundary as boundary
    import retained_layer01_stage18_producer_split_diagnostic as parent
    from exact_projection import project_vectors
    from fp16_adaptation_oracle import EPSILON_Q48, residual_add, rmsnorm, silu_gate_exp
    from projection_oracle import complete_projection_output
    from qwen2_rope_oracle import qwen2_coefficient, rotate_pair
    from attention_oracle import attention_score, attention_softmax, attention_value
    from trace_lineage_a_layer07_downproj import round_q48
    from trace_lineage_a_layer07_rmsnorm import mathematical_outputs
    from trace_lineage_a_layer07_silu import nearest_decimal

    parent.OUT = OUT
    tool = shutil.which("iverilog")
    require(tool is not None, "host iverilog missing; inspect declared containers")
    parent.command("tool_versions", [sys.executable, "-B", "-c",
                   "import sys,numpy,torch; print(sys.version); print(numpy.__version__,torch.__version__)"])
    parent.command("iverilog_version", [tool, "-V"])
    require(pf["public_port_contract"] in
            (OUT / "sources/ace3_decoder_layer0_token_engine.sv").read_text(),
            "public module/port contract changed")
    projections = {1: (0, "self_attn.q_proj"), 2: (0, "self_attn.k_proj"),
                   3: (0, "self_attn.v_proj"), 11: (10, "self_attn.o_proj"),
                   14: (13, "mlp.gate_proj"), 15: (13, "mlp.up_proj"),
                   17: (16, "mlp.down_proj")}
    dependencies = {0: [], 1: [0], 2: [0], 3: [0], 4: [1], 5: [2], 6: [5], 7: [3],
                    8: [4, 6], 9: [8], 10: [9, 7], 11: [10], 12: [11], 13: [12],
                    14: [13], 15: [13], 16: [14, 15], 17: [16], 18: [12, 17]}
    write("frozen.json", dict(
        evaluator=record(__file__), parent_authentication=record(OUT / "parent_authentication.json"),
        authenticated_inputs=list(inputs.records.values()), tools=[record(tool),
            record(sys.executable), record(OUT / "tool_versions.log"), record(OUT / "iverilog_version.log")],
        layer=layer, positions=list(range(4)), stages=list(range(19)), token_history=boundary.HISTORY,
        public_port_contract=pf["public_port_contract"], public_tops=pf["public_tops"],
        public_parameters=pf["public_parameters"], comparison_policy=boundary.POLICY,
        causal_intervention="Only layer00 stage00: exact mean of squared FP16 operands plus "
        "epsilon=281474977/2^48; exact mathematical inverse RMS times activation and FP16 "
        "weight; one binary16 RNE with signed zero. No Q48 mean RNE, floor Q24 sqrt, or Q24 quotient. "
        "For later layers, inherit each reviewed upstream arm without repeating the intervention; "
        "both later-layer RMSNorm operators retain frozen Q48/RNE.",
        frozen_control="Retained A Q48/RNE, corrected Q/K single-round bias, V double-round bias, "
        "original softmax, FP16 RoPE and KV, accurate SiLU; stage13 RMS remains Q48/RNE.",
        dependencies=dependencies, cross_position_dependencies={"8": "all earlier stage06 K",
                                                               "10": "all earlier stage07 V"},
        regeneration_rule="Recompute only nodes with bit-changed direct operands, plus the "
        "stage00 intervention; reuse bit-identical nodes. Earlier candidate K/V, never old "
        "incompatible K/V, feed later positions. All positions keep their authenticated token roots.",
        endpoint=f"Layer{layer:02d} stage18 / layer{layer+1:02d} incoming hidden only. "
        "No later-layer cache is consumed "
        "or certified compatible; no layer08, binary64 trajectory, or whole-model benefit claim.",
        independent_oracles="All frozen-control stage coordinates must reproduce retained RTL; "
        "candidate RMS uses independent Decimal80 sqrt/RNE and reviewed exact midpoint results; "
        "exact native AWQ reductions cross-checked with scalar bit oracle at first/last/max-delta "
        "output per projection/position/arm. References remain the independent retained trajectory.",
        provenance="A lineage; original positions0/1 reference-input consumption remains unproven. "
        f"This software counterfactual regenerates layer{layer:02d} K/V only, not simulator state.",
        numerical_policy_changed=False, RTL_simulations=0, binary64_admission_evaluated=False))
    argv = [tool, "-g2012"]
    for top in pf["public_tops"]:
        argv.extend(("-s", top))
    for name, value in pf["public_parameters"].items():
        argv.extend(("-P", f"{name}={value}"))
    parent.command("public_contract_compile", argv + ["-o", str(OUT / "public_contracts.vvp")]
                   + [str(p) for p in sorted((OUT / "sources").glob("*.sv"))])
    phases = {"authentication_freeze_compile_seconds": time.monotonic() - started}
    caches = {arm: {"k": [], "v": []} for arm in ("frozen_Q48", "single_round")}
    summaries, regenerated, oracle_checks = [], [], 0
    halted = None

    def finite(outputs, label):
        require(all(not invalid and not saturated for _, invalid, saturated in outputs),
                label + " invalid/saturated")
        return [int(value) for value, _, _ in outputs]

    def operator(stage, stages, root, cache, p):
        if stage in (0, 13):
            suffix = "input_layernorm.weight" if stage == 0 else "post_attention_layernorm.weight"
            return finite(rmsnorm(root if stage == 0 else stages[12], tensors[suffix])[0], "RMS")
        if stage in (4, 5):
            source = stages[1 if stage == 4 else 2]
            result = list(source)
            for base in range(0, len(source), 64):
                for pair in range(32):
                    lo, hi, bad, sat = rotate_pair(source[base+pair], source[base+pair+32],
                                                  *qwen2_coefficient(p, pair))
                    require(not (bad or sat), "RoPE invalid/saturated")
                    result[base+pair], result[base+pair+32] = lo, hi
            return result
        if stage in (6, 7):
            return list(stages[5 if stage == 6 else 3])
        if stage == 8:
            result = []
            for h in range(14):
                for pos in range(p+1):
                    key = cache["k"][pos][(h//7)*64:(h//7+1)*64]
                    score = attention_score(stages[4][h*64:(h+1)*64], key, [True]*64, p, pos)
                    require(not (score.invalid or score.cache_miss or score.saturation), "score invalid")
                    result.append(score.score_f16)
            return result
        if stage == 9:
            result = []
            for h in range(14):
                row = attention_softmax(stages[8][h*(p+1):(h+1)*(p+1)],
                    list(range(p+1)), [True]*(p+1), [False]*(p+1), [False]*(p+1), p)
                require(not (row.invalid or row.cache_miss or row.row_error), "softmax invalid")
                result.extend(row.probabilities_f16)
            return result
        if stage == 10:
            result = []
            for h in range(14):
                for dim in range(64):
                    row = attention_value(stages[9][h*(p+1):(h+1)*(p+1)],
                        [cache["v"][pos][(h//7)*64+dim] for pos in range(p+1)],
                        [True]*(p+1), [False]*(p+1))
                    require(not (row.invalid or row.cache_miss or row.row_error or row.saturation),
                            "AV invalid")
                    result.append(row.value_f16)
            return result
        if stage in (12, 18):
            return finite([residual_add(a, b) for a, b in zip(
                root if stage == 12 else stages[12], stages[11 if stage == 12 else 17],
                strict=True)], "residual")
        require(stage == 16, "unsupported sensitivity stage")
        return finite([silu_gate_exp(g, u) for g, u in zip(stages[14], stages[15], strict=True)],
                      "SiLU")

    def compare(left, right):
        require(len(left) == len(right), "comparison geometry")
        errors = [abs(boundary.units(a)-boundary.units(b)) for a, b in zip(left, right, strict=True)]
        return dict(coordinates=len(left), bit_differences=sum(a != b for a, b in zip(left, right)),
                    material_failures=sum(not boundary.accepts(a, b) for a, b in zip(left, right)),
                    max_abs_error=max(errors, default=0)/(1 << 24), sum_abs_error_q24=sum(errors))

    for p in range(4):
        phase = time.monotonic()
        arms = {arm: {} for arm in caches}
        changed = {}
        for stage in range(19):
            control, candidate = arms["frozen_Q48"], arms["single_round"]
            changed_parents = [s for s in dependencies[stage] if changed[s]]
            changed_history = []
            if stage in (8, 10):
                key = "k" if stage == 8 else "v"
                changed_history = [pos for pos in range(p) if
                    caches["frozen_Q48"][key][pos] != caches["single_round"][key][pos]]
            changed_root = stage in (0, 12) and incoming[p] != candidate_incoming[p]
            intervention = layer == 0 and stage == 0
            affected = intervention or changed_root or bool(changed_parents or changed_history)
            if stage in projections:
                operand, suffix = projections[stage]
                ts = {name: tensors[suffix+"."+name] for name in ("qweight", "qzeros", "scales")}
                projected, totals = project_vectors(control[operand],
                    candidate[operand] if affected else control[operand], ts)
                bias = tensors.get(suffix+".bias")
                for arm, label in (("frozen_Q48", "A"), ("single_round", "R")):
                    result = projected[label]
                    if bias is not None:
                        if stage in (1, 2):
                            biased = [v + (boundary.units(b) << 24)
                                      for v, b in zip(totals[label], bias, strict=True)]
                            result = [round_q48(v) for v in biased]
                            result = [b | 0x8000 if b == 0 and v < 0 else b
                                      for b, v in zip(result, biased, strict=True)]
                        else:
                            result = finite([residual_add(v, b) for v, b in
                                             zip(result, bias, strict=True)], "projection bias")
                    arms[arm][stage] = (list(control[stage]) if arm == "single_round"
                                       and not affected else result)
                selected = {0, len(control[stage])-1, max(range(len(control[stage])),
                    key=lambda i: abs(boundary.units(control[stage][i])
                                      - boundary.units(candidate[stage][i])))}
                count, words = len(control[stage]), len(control[stage])//8
                for arm, label in (("frozen_Q48", "A"), ("single_round", "R")):
                    for i in sorted(selected):
                        oracle = complete_projection_output(arms[arm][operand],
                            ts["qweight"][i//8::words], ts["qzeros"][i//8::words],
                            ts["scales"][i::count], i % 8,
                            None if bias is None else bias[i], single_round_bias=stage in (1, 2))
                        require(not (oracle[2] or oracle[3]) and oracle[0] == totals[label][i]
                                and oracle[1] == arms[arm][stage][i], "scalar projection disagreement")
                        oracle_checks += 1
            else:
                control[stage] = operator(stage, control, incoming[p], caches["frozen_Q48"], p)
                if intervention:
                    weights = tensors["input_layernorm.weight"]
                    candidate[0] = mathematical_outputs(incoming[p], weights)
                    with localcontext() as context:
                        context.prec = 80
                        x = [Decimal(boundary.units(b))/(1 << 24) for b in incoming[p]]
                        mean = sum(v*v for v in x)/len(x) + Decimal(EPSILON_Q48)/(1 << 48)
                        inverse = 1/mean.sqrt()
                        independent = [nearest_decimal(v*inverse*Decimal(boundary.units(w))/(1 << 24),
                            (a ^ w) & 0x8000) for v, a, w in zip(x, incoming[p], weights, strict=True)]
                    require(candidate[0] == independent, "independent mathematical RMS disagreement")
                    expected = [int(r["outputs"]["no_mean_RNE"], 16)
                                for r in reviewed_coordinates if r["position"] == p]
                    require(candidate[0] == expected, "candidate differs from reviewed operation")
                else:
                    candidate[stage] = (operator(stage, candidate, candidate_incoming[p],
                        caches["single_round"], p) if affected else list(control[stage]))
            control_match = compare(control[stage], actual[p][stage])
            if control_match["bit_differences"]:
                write(f"control_mismatch_p{p}_stage{stage:02d}.json", dict(
                    actual=actual[p][stage], modeled=control[stage], comparison=control_match))
            require(control_match["bit_differences"] == 0,
                    f"frozen-control compatibility failure at p{p}/stage{stage:02d}")
            changed[stage] = control[stage] != candidate[stage]
            if stage in (6, 7):
                key = "k" if stage == 6 else "v"
                for arm in arms:
                    caches[arm][key].append(list(arms[arm][stage]))
            summaries.append(dict(position=p, stage=stage,
                frozen_vs_reference=compare(control[stage], reference[p][stage]),
                candidate_vs_reference=compare(candidate[stage], reference[p][stage]),
                candidate_vs_frozen=compare(candidate[stage], control[stage])))
            regenerated.append(dict(position=p, stage=stage, recomputed=affected,
                changed_direct_stages=changed_parents, changed_earlier_cache_positions=changed_history,
                changed_incoming_hidden=changed_root,
                bit_changed=changed[stage], reason="stage00 intervention" if intervention else
                "changed operands" if affected else "bit-identical operands; compatible reuse"))
            if layer == 2 and any(summaries[-1][arm]["material_failures"] for arm in
                                  ("frozen_vs_reference", "candidate_vs_reference")):
                halted = dict(position=p, stage=stage,
                    failure_taxonomy="independent_FP16_trajectory_mismatch",
                    root_cause_hypothesis="The retained or inherited changed hidden/K/V "
                    "trajectory exceeds the unchanged stage gate; no unique producer is established.",
                    regression="Recheck this first failing stage with its recorded arm-local "
                    "operands and unchanged independent reference before extending the cone.",
                    comparison=summaries[-1],
                    failing_coordinates={arm: [i for i, (a, r) in enumerate(zip(
                        arms[arm][stage], reference[p][stage], strict=True))
                        if not boundary.accepts(a, r)] for arm in arms})
                write("failure.json", halted)
                break
        write(f"position{p}_stages.json", dict(**arms, independent_retained=reference[p]))
        phases[f"position{p}_seconds"] = time.monotonic()-phase
        if halted:
            break
    write("comparisons.json", summaries)
    write("regenerated_dependencies.json", regenerated)
    write("software_kv.json", caches)
    endpoints = [s for s in summaries if s["stage"] == 18]
    stage00 = [s for s in summaries if s["stage"] == 0]
    endpoint_change = sum(s["candidate_vs_reference"]["sum_abs_error_q24"]
                          - s["frozen_vs_reference"]["sum_abs_error_q24"] for s in endpoints)
    candidate_failures = sum(s["candidate_vs_reference"]["material_failures"] for s in summaries)
    if layer == 2:
        next_boundary = (
            f"Layer02 position{halted['position']} stage{halted['stage']:02d}: localize the "
            "recorded first material failure before any extension." if halted else
            "Layer03 stage00 and only its genuinely affected hidden/K/V cone, inheriting "
            "these separate layer02 endpoints under unchanged gates. Aggregate improvement "
            "supports a bounded comparison, not production adoption or layer08/layer21 repair."
            if endpoint_change < 0 else
            "Layer02 earliest error-amplifying stage in comparisons.json: discriminate "
            "the measured endpoint tradeoff before extending toward layer08/layer21.")
    elif layer == 1:
        next_boundary = (
            "Stop at the first failing layer01 candidate stage; localize its changed operands "
            "before extending the cone." if candidate_failures else
            "Layer02 stage00 and its genuinely affected hidden/K/V cone under unchanged gates. "
            "The aggregate layer01 endpoint improvement supports another bounded comparison, "
            "not production adoption or a layer08/layer21 repair claim." if endpoint_change < 0 else
            "Layer01 earliest error-amplifying stage in comparisons.json; discriminate the "
            "measured endpoint regression before further propagation toward layer08/layer21.")
    else:
        next_boundary = (
            "Layer01 stage00 consuming these four software layer00 outputs. Any extension "
            "must regenerate its genuinely changed K/V/hidden cone; old later-layer caches are not "
            "compatible merely by shape. Judge measured endpoint tradeoffs before extending. "
            "No inference about layer08 witnesses or binary64-v1 from this layer00 experiment.")
    status = "DIAGNOSTIC_HALTED_MATERIAL_FAILURE" if halted else "DIAGNOSTIC_COMPLETE_NO_RTL_REPLAY"
    write("result.json", dict(
        status=status, first_material_failure=halted,
        **{f"layer{layer:02d}_endpoint_comparisons": endpoints},
        layer=layer, stage00_comparisons=stage00,
        stage_comparisons=len(summaries), frozen_control_bit_differences=0,
        scalar_projection_oracle_checks=oracle_checks,
        candidate_material_failures=candidate_failures,
        frozen_material_failures=sum(s["frozen_vs_reference"]["material_failures"] for s in summaries),
        regenerated_nodes=sum(r["recomputed"] for r in regenerated),
        bit_changed_nodes=sum(r["bit_changed"] for r in regenerated),
        downstream_intervention_benefit_measured=True,
        endpoint_sum_abs_error_q24_change=endpoint_change,
        supports_further_bounded_propagation=not halted and candidate_failures == 0
        and endpoint_change < 0,
        next_boundary=next_boundary,
        provenance="Independent retained FP16 stage references unchanged; original positions0/1 "
        "reference-input provenance remains unavailable. Fixed tokens [9707,1879,0,358], "
        "not candidate-native greedy history. Candidate arrays are software sensitivity only.",
        numerical_policy_changed=False, production_Q48_frozen=True, source_promoted=False,
        RTL_simulations=0, fresh_RTL_PASS=False, binary64_admission_evaluated=False,
        third_token_claim=False, reference_injected_into_execution=False,
        phases_seconds=phases, elapsed_seconds=time.monotonic()-started,
        review_status="PENDING_NORMAL_HOST_REVIEWER"))
    print(json.dumps(dict(status=status, endpoints=endpoints,
                          phases_seconds=phases), sort_keys=True))


def layer02_stage16_bootstrap():
    import subprocess
    import numpy as np
    import torch

    parent = ROOT / "build/retained_layer02_rmsnorm_candidate_propagation_2cbdd5f727db_attempt002"
    review = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
                  "handoffs/2cbdd5f727db/round-0001.json")
    roots = {
        "frozen.json": "5a71b8df19468a42a62f8ae632067c039d67abd2b32c6c9e82d5314000decf4c",
        "result.json": "4e25c8d3bd5f6c48fbfa167d267257c2992518cc1c9a85da5af5777f1f6ca1bc",
        "artifacts.json": "16747fe6f10fb902784175f01a4a7980c9d3fca735903158dd89c1f6fba0197e",
    }
    for name, digest in roots.items():
        require(record(parent / name)["sha256"] == digest, "reviewed parent changed: " + name)
    require(record(review)["sha256"] ==
            "f39fd9115de292c6df93a25df315e6dda3455c5dddf6acc51949952ca132013f",
            "parent reviewer receipt changed")
    receipt = json.loads(review.read_text())
    require(receipt["kind"] == "round_reviewed_handoff"
            and receipt["producer_role"] == "reviewer"
            and receipt["mission_id"] == "2cbdd5f727db"
            and receipt["review"]["status"] == "done"
            and "stage16 index813" in receipt["review"]["next_action"],
            "mission-scoped independent parent review required")
    manifest = {r["path"]: r for r in json.loads((parent / "artifacts.json").read_text())}
    names = [
        "fp16_adaptation_oracle.py", "diagnose_lineage_a_position3_boundary.py",
        "trace_lineage_a_layer07_silu.py", "trace_lineage_a_layer07_downproj.py",
        "projection_oracle.py", "awq_bit_oracle.py", "propagated_reference.py",
        "ace3_fp16_silu_gate_core.sv", "ace3_fp16_fixed.sv",
        "ace3_q47_48_to_f16_rne.sv",
    ]
    consumed = []
    for path in [parent / "position0_stages.json"] + [parent / "sources" / n for n in names]:
        rec = record(path)
        require(rec == manifest[str(path)], "consumed parent artifact changed: " + str(path))
        consumed.append(rec)
    sources = OUT / "sources"
    sources.mkdir()
    snapshots = []
    for path in [parent / "sources" / n for n in names] + [Path(__file__).resolve()]:
        target = sources / path.name
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        snapshots.append(dict(origin=record(path), snapshot=record(target)))
    contract = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/"
                    "handoffs/6eb2e0362261/latest.json")
    write("task_contract.json", json.loads(contract.read_text()))
    write("parent_authentication.json", dict(
        roots={n: record(parent / n) for n in roots}, review=record(review),
        consumed=consumed, snapshots=snapshots, task_contract=record(contract),
        historical_pending_field_is_not_missing_review=True,
        scope="Only selected stage14/15/16 coordinates are used; no trajectory replay."))
    stages = json.loads((parent / "position0_stages.json").read_text())
    indices = list(range(811, 816))
    selected = {}
    for arm in ("frozen_Q48", "single_round", "independent_retained"):
        selected[arm] = {}
        for stage in ("14", "15", "16"):
            require(len(stages[arm][stage]) == 4864, "retained stage geometry")
            selected[arm][stage] = {str(i): stages[arm][stage][i] for i in indices}
    write("operands.json", selected)
    parent_freeze = json.loads((parent / "frozen.json").read_text())
    compiler = shutil.which("iverilog")
    require(compiler is not None, "Icarus unavailable; inspect declared local containers")
    version = subprocess.run([compiler, "-V"], capture_output=True, text=True, check=False)
    write("tool_versions.json", dict(
        python=sys.version, executable=sys.executable, numpy=np.__version__,
        torch=torch.__version__, iverilog_path=compiler,
        iverilog_returncode=version.returncode, iverilog=version.stdout + version.stderr))
    require(version.returncode == 0, "compiler version probe failed")
    top = "ace3_fp16_silu_gate_core"
    command = [compiler, "-g2012", "-s", top,
               "-P", top + ".INTERMEDIATE_SIZE=4864",
               "-P", top + ".ACCURATE_SIGMOID=1",
               "-o", str(OUT / "public_contract.vvp")]
    command += [str(sources / n) for n in names if n.endswith(".sv")]
    write("frozen.json", dict(
        layer=2, position=0, stage=16, witness=813, indices=indices,
        arms=list(selected), gate_up_factorial="Every recorded gate arm x every recorded up arm",
        arithmetic_interventions=[
            "Native Q24 sigmoid and Q72->Q24->FP16 product",
            "Same Q24 sigmoid, direct exact Q72->FP16 product",
            "Decimal mathematical sigmoid rounded to Q24, original product boundaries",
            "Decimal80/120 mathematical SiLU*up with one final FP16 RNE",
            "Recorded binary64 torch SiLU*up recurrence with one final FP16 boundary"],
        comparison_policy=parent_freeze["comparison_policy"],
        reference_policy="Unchanged retained independent FP16 trajectory. Conditional "
        "reference-operand recurrence checks are not a re-anchored trajectory.",
        decimal_precision_crosscheck=[80, 120], selection="Witness and two adjacent indices each side",
        evaluator=record(sources / Path(__file__).name), operands=record(OUT / "operands.json"),
        parent_authentication=record(OUT / "parent_authentication.json"),
        tools=record(OUT / "tool_versions.json"), compile_command=command,
        public_top=top, public_parameters={"INTERMEDIATE_SIZE": 4864, "ACCURATE_SIGMOID": 1},
        public_port_contract=(sources / (top + ".sv")).read_text().split(");", 1)[0] + ");",
        provenance=parent_freeze["provenance"], RTL_simulations=0,
        numerical_policy_changed=False, source_promoted=False,
        regression="Reproduce the retained frozen PASS/candidate FAIL at index813; "
        "cross-check independent integer reconstructions and Decimal rounding; "
        "measure both operand-swap orders and arithmetic-only interventions."))
    for path in [*sources.iterdir(), *OUT.glob("*.json")]:
        path.chmod(0o444)
    compiled = subprocess.run(command, capture_output=True, text=True, check=False)
    write("public_contract_compile.json", dict(
        command=command, returncode=compiled.returncode,
        stdout=compiled.stdout, stderr=compiled.stderr, simulation=False))
    require(compiled.returncode == 0, "frozen public SiLU contract did not compile")
    os.execv(sys.executable, [sys.executable, "-B", str(sources / Path(__file__).name),
                             "--layer02-stage16-localization", "--attempt", str(ATTEMPT),
                             "--measure"])


def layer02_stage16_measure():
    from decimal import Decimal, localcontext, ROUND_HALF_EVEN
    from fractions import Fraction

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(OUT / "sources"))
    import numpy as np
    import torch
    from diagnose_lineage_a_position3_boundary import accepts, ordered, units
    from fp16_adaptation_oracle import silu_gate_exp
    from trace_lineage_a_layer07_downproj import round_q48
    from trace_lineage_a_layer07_silu import accurate_silu, mathematical_silu, nearest_decimal, rne_ratio

    started = time.monotonic()
    torch.set_num_threads(1)
    freeze = json.loads((OUT / "frozen.json").read_text())
    require(record(Path(__file__)) == freeze["evaluator"], "frozen evaluator changed")
    require(record(OUT / "operands.json") == freeze["operands"], "frozen operands changed")
    data = json.loads((OUT / "operands.json").read_text())

    def scalar(bits):
        return dict(bits=f"{bits:04x}", value=units(bits) / 2**24,
                    exact=str(Fraction(units(bits), 2**24)))

    def compare(bits, reference):
        error = abs(units(bits) - units(reference))
        denominator = max(abs(units(reference)), 1024)
        ulp = abs(ordered(bits) - ordered(reference))
        passed = error <= 2**21 or (1000 * error < denominator and ulp <= 1)
        require(passed == accepts(bits, reference), "unchanged gate disagreement")
        return dict(actual=scalar(bits), reference=scalar(reference),
                    signed_error_exact=str(Fraction(units(bits) - units(reference), 2**24)),
                    absolute_error_exact=str(Fraction(error, 2**24)),
                    absolute_error=error / 2**24, relative_error_exact=str(Fraction(error, denominator)),
                    relative_error=error / denominator, ordered_FP16_ULP=ulp, accepted=passed)

    def fp16_product(q72, zero_sign):
        result = round_q48(Fraction(q72, 2**24))
        return result | zero_sign if result & 0x7fff == 0 else result

    def delta(left, right):
        return str(Fraction(units(left) - units(right), 2**24))

    rows = []
    for index in freeze["indices"]:
        key = str(index)
        reference = data["independent_retained"]["16"][key]
        cases = {}
        for gate_arm in freeze["arms"]:
            for up_arm in freeze["arms"]:
                gate = data[gate_arm]["14"][key]
                up = data[up_arm]["15"][key]
                zero_sign = (gate ^ up) & 0x8000
                local = accurate_silu(gate, up)
                oracle, invalid, saturated = silu_gate_exp(gate, up)
                require(not (invalid or saturated) and local["bits"] == oracle,
                        "independent stage16 integer reconstruction disagreement")
                ideals = {}
                for precision in freeze["decimal_precision_crosscheck"]:
                    with localcontext() as context:
                        context.prec = precision
                        exact = mathematical_silu(gate, up)
                        ideals[precision] = nearest_decimal(exact, zero_sign)
                        if precision == 120:
                            g = Decimal(units(gate)) / 2**24
                            e = (-abs(g)).exp()
                            sigmoid = (e if g < 0 else Decimal(1)) / (1 + e)
                            sigmoid_q24 = int((sigmoid * 2**24).to_integral_value(
                                rounding=ROUND_HALF_EVEN))
                            ideal_decimal = str(exact)
                            sigmoid_decimal = str(sigmoid)
                require(ideals[80] == ideals[120], "Decimal80/120 FP16 rounding disagreement")
                direct = fp16_product(local["product_q72"], zero_sign)
                ideal_sigmoid_product = units(gate) * units(up) * sigmoid_q24
                ideal_sigmoid_q24 = rne_ratio(ideal_sigmoid_product, 2**48)
                ideal_sigmoid_bits = fp16_product(ideal_sigmoid_q24 << 48, zero_sign)
                g64 = torch.tensor(units(gate) / 2**24, dtype=torch.float64)
                u64 = torch.tensor(units(up) / 2**24, dtype=torch.float64)
                binary64 = (torch.nn.functional.silu(g64) * u64).half().numpy().view(np.uint16).item()
                outputs = dict(native=oracle, direct_product=direct,
                               mathematical_sigmoid_Q24=ideal_sigmoid_bits,
                               mathematical_single_round=ideals[120], binary64_recurrence=binary64)
                cases[gate_arm + "/" + up_arm] = dict(
                    gate=scalar(gate), up=scalar(up), integer_reconstruction=local,
                    gate_up_product_exact=str(Fraction(units(gate) * units(up), 2**48)),
                    native_product_exact=str(Fraction(local["product_q72"], 2**72)),
                    product_Q24_rounding_error_exact=str(Fraction(
                        (local["rounded_q24"] << 48) - local["product_q72"], 2**72)),
                    final_FP16_rounding_error_exact=str(Fraction(
                        units(oracle) - local["rounded_q24"], 2**24)),
                    mathematical_value_decimal120=ideal_decimal,
                    mathematical_sigmoid_decimal120=sigmoid_decimal,
                    mathematical_sigmoid_Q24=sigmoid_q24,
                    comparisons={name: compare(bits, reference) for name, bits in outputs.items()})
        retained = {arm: compare(data[arm]["16"][key], reference) for arm in freeze["arms"]}
        local_agreement = {
            arm: cases[arm + "/" + arm]["comparisons"]["native"]["actual"]["bits"]
            == retained[arm]["actual"]["bits"]
            for arm in ("frozen_Q48", "single_round")}
        cf = cases["single_round/frozen_Q48"]["comparisons"]["native"]["actual"]["bits"]
        fc = cases["frozen_Q48/single_round"]["comparisons"]["native"]["actual"]["bits"]
        f, c = data["frozen_Q48"]["16"][key], data["single_round"]["16"][key]
        # Both telescoping orders expose gate/up interaction without assigning it arbitrarily.
        components = dict(
            gate_then_up=[delta(int(cf, 16), f), delta(c, int(cf, 16))],
            up_then_gate=[delta(int(fc, 16), f), delta(c, int(fc, 16))],
            candidate_minus_frozen=delta(c, f))
        for order in ("gate_then_up", "up_then_gate"):
            require(sum(map(Fraction, components[order])) ==
                    Fraction(components["candidate_minus_frozen"]), "operand delta accounting")
        rows.append(dict(index=index, retained=retained, local_agreement=local_agreement,
                         candidate_vs_frozen=compare(c, f), cases=cases,
                         operand_components_exact=components))
    write("interventions.json", rows)
    witness = next(row for row in rows if row["index"] == freeze["witness"])
    require(witness["retained"]["frozen_Q48"]["accepted"]
            and not witness["retained"]["single_round"]["accepted"],
            "recorded first-failure regression changed")
    cases = witness["cases"]
    candidate = cases["single_round/single_round"]["comparisons"]
    reference_case = cases["independent_retained/independent_retained"]["comparisons"]
    swaps = {name: cases[name]["comparisons"]["native"] for name in (
        "single_round/frozen_Q48", "independent_retained/single_round",
        "single_round/independent_retained")}
    reference_matches = all(
        reference_case[name]["actual"]["bits"] == witness["retained"]["independent_retained"]["actual"]["bits"]
        for name in ("mathematical_single_round", "binary64_recurrence"))
    if not all(witness["local_agreement"].values()):
        classification = "stage16_retained_vs_own_operand_arithmetic_discrepancy"
        next_boundary = "Reconcile retained stage16 output with its authenticated own-operand reconstruction."
    elif not reference_matches:
        classification = "oracle_reference_recurrence_discrepancy"
        next_boundary = "Resolve recorded reference recurrence discrepancy without replacing the reference."
    elif (not candidate["mathematical_single_round"]["accepted"]
          and all(not row["accepted"] for row in candidate.values())
          and all(row["accepted"] for row in swaps.values())):
        classification = "inherited_gate_up_operand_drift"
        next_boundary = (
            "Layer02 position0 stage15 index813 candidate up-projection drift and the "
            "shared stage14 gate offset from reference: next measure their stage13 "
            "operand/projection contributions. No later-stage extension or replay is justified yet.")
    else:
        classification = "unresolved_mixed_operand_arithmetic_interaction"
        next_boundary = "Use the recorded factorial to select a further bounded discriminator, not propagation."
    write("result.json", dict(
        status="DIAGNOSTIC_COMPLETE", classification=classification,
        failure_taxonomy="independent_FP16_trajectory_mismatch",
        root_cause_hypothesis=classification, next_boundary=next_boundary,
        frozen_vs_reference=witness["retained"]["frozen_Q48"],
        candidate_vs_reference=witness["retained"]["single_round"],
        candidate_vs_frozen=witness["candidate_vs_frozen"],
        operand_swap_interventions=swaps, arithmetic_only_interventions=candidate,
        reference_policy_control=reference_case, reference_recurrence_matches=reference_matches,
        witness_local_agreement=witness["local_agreement"],
        neighbor_local_agreement={str(r["index"]): r["local_agreement"] for r in rows},
        coordinates=len(rows), factorial_cases=sum(len(r["cases"]) for r in rows),
        regression="Recorded frozen PASS/candidate FAIL reproduced; independent integer "
        "reconstruction, Decimal80/120 RNE and both operand-swap accounting orders checked once.",
        supports_repair=False, production_adoption=False, cone_extended=False,
        RTL_simulations=0, fresh_RTL_PASS=False, binary64_admission_evaluated=False,
        numerical_policy_changed=False, source_promoted=False, reference_injected_into_execution=False,
        provenance=freeze["provenance"], elapsed_seconds=time.monotonic() - started,
        scope="Retained software sensitivity only; no unique upstream producer or "
        "full-stage/model correctness established. Reference input-consumption qualification persists.",
        review_status="PENDING_NORMAL_HOST_REVIEWER"))
    print(json.dumps(dict(status="DIAGNOSTIC_COMPLETE", classification=classification,
                          output=str(OUT)), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--stage00-discriminator", action="store_true")
    parser.add_argument("--rmsnorm-downstream", action="store_true")
    parser.add_argument("--layer01-propagation", action="store_true")
    parser.add_argument("--layer02-propagation", action="store_true")
    parser.add_argument("--layer02-stage16-localization", action="store_true")
    args = parser.parse_args()
    require(sum((args.stage00_discriminator, args.rmsnorm_downstream,
                 args.layer01_propagation, args.layer02_propagation,
                 args.layer02_stage16_localization)) <= 1, "select one experiment")
    ATTEMPT = args.attempt
    require(ATTEMPT > 0, "positive fresh attempt required")
    OUT = ROOT / f"build/retained_layer00_stage18_producer_split_dc9689d1352f_attempt{ATTEMPT:03d}"
    if args.stage00_discriminator:
        OUT = ROOT / ("build/retained_layer00_stage00_rmsnorm_witness_discriminator_"
                      f"d004a401c237_attempt{ATTEMPT:03d}")
    if args.rmsnorm_downstream:
        OUT = ROOT / ("build/retained_layer00_rmsnorm_candidate_downstream_sensitivity_"
                      f"7b2e1e6dd113_attempt{ATTEMPT:03d}")
    if args.layer01_propagation:
        OUT = ROOT / ("build/retained_layer01_rmsnorm_candidate_propagation_"
                      f"228864847a63_attempt{ATTEMPT:03d}")
    if args.layer02_propagation:
        OUT = ROOT / ("build/retained_layer02_rmsnorm_candidate_propagation_"
                      f"2cbdd5f727db_attempt{ATTEMPT:03d}")
    if args.layer02_stage16_localization:
        OUT = ROOT / ("build/retained_layer02_stage16_index813_localization_"
                      f"6eb2e0362261_attempt{ATTEMPT:03d}")
    if not args.measure:
        OUT.mkdir()
    else:
        require(Path(__file__).resolve().parent == OUT / "sources", "run only frozen source")
    try:
        if args.measure:
            if args.layer02_stage16_localization:
                layer02_stage16_measure()
            elif args.layer02_propagation:
                layer02_measure()
            elif args.layer01_propagation:
                layer01_measure()
            elif args.rmsnorm_downstream:
                sensitivity_measure()
            elif args.stage00_discriminator:
                stage00_measure()
            else:
                measure()
            seal()
        elif args.layer02_stage16_localization:
            layer02_stage16_bootstrap()
        elif args.layer02_propagation:
            layer02_bootstrap()
        elif args.layer01_propagation:
            layer01_bootstrap()
        elif args.rmsnorm_downstream:
            sensitivity_bootstrap()
        elif args.stage00_discriminator:
            stage00_bootstrap()
        else:
            bootstrap()
    except (ValueError, KeyError, OSError, ArithmeticError, ImportError) as error:
        write("failure.json", dict(
            status="DIAGNOSTIC_NO_COMPLETED_MEASUREMENT", failure_taxonomy="diagnostic_execution",
            root_cause_hypothesis=str(error),
            regression="A fresh attempt must execute all unchanged coordinate/oracle accounting.",
            RTL_correctness_conclusion=None))
        seal()
        raise
