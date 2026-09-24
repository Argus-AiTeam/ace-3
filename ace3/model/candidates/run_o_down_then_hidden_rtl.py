#!/usr/bin/env python3
"""Isolated, actual-output-fed residual association cone; never production admission."""
import argparse
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "build"
SOFTWARE = BUILD / "general_b_hidden_drift_l0_s18_state_choice_8b12d6a4aea7_attempt001"
ARM = SOFTWARE / "o_down_then_hidden"
B = BUILD / "layer08_position2_q_only_rope_rtl_86938809f0a2_attempt010"
CLOSURE = BUILD / "layer08_position3_softmax_exp_replay_attempt003/dependency_closure"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/8b12d6a4aea7/round-0001.json")
RTL = ROOT / "ace3/rtl/candidates/o_down_then_hidden_v1/ace3_fp16_o_down_sum.sv"
TB = ROOT / "ace3/tb/ace3_o_down_then_hidden_tb.sv"
TOP = "ace3_decoder_layer0_token_engine"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module unavailable: {path}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"source integration contract changed: {old}")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != BUILD or not out.name.startswith("o_down_then_hidden_"):
        raise RuntimeError("output outside isolated namespace")
    out.mkdir(exist_ok=False)
    self_snapshot = out / "runner.py"
    shutil.copyfile(Path(__file__), self_snapshot)
    d = module("association_execution", CLOSURE / "replay.py")
    started = time.monotonic()
    phase, active = "authentication", None
    bound, transactions, gates, phases = {}, [], [], []

    def bind(rec):
        d.authenticate(rec)
        bound[rec["path"]] = rec
        return Path(rec["path"])

    def read_bound(rec):
        return d.load(bind(rec))

    def timed(argv, directory, label, timeout):
        t = time.monotonic()
        d.run_command(argv, directory, label, timeout)
        phases.append({"phase": label, "node": active, "seconds": time.monotonic() - t})

    try:
        review = read_bound(d.record(REVIEW))
        d.require(review["producer_role"] == "reviewer" and
                  review["kind"] == "round_reviewed_handoff" and
                  review["mission_id"] == "8b12d6a4aea7" and
                  review["review"]["status"] == "done", "software review unavailable")
        sm = read_bound(d.record(SOFTWARE / "output_manifest.json"))
        artifacts = {r["path"]: r for r in sm["artifacts"]}
        sf = read_bound(artifacts[str(SOFTWARE / "freeze.json")])
        read_bound(artifacts[str(SOFTWARE / "result.json")])
        inherited = {r["path"]: r for r in sf["bindings"]}
        bf = read_bound(inherited[str(B / "frozen.json")])
        old = read_bound(d.record(B / "result.json"))
        d.require(old["status"] == "PASS_BOUNDED_RTL_CANDIDATE",
                  "retained B execution did not pass its bounded gate")
        graph_path = BUILD / "general_b_hidden_drift_d7f5d9101813_attempt003/retained_graph.npz"
        bind(inherited[str(graph_path)])
        for rec in bf["source_closure"]:
            bind(rec)
        checkpoint = bind(bf["checkpoint"])
        helper_path = BUILD / "general_b_l8p3s1_index223_5a41bc443f05_attempt002/run.py"
        helper = module("association_fraction_oracle", bind(inherited[str(helper_path)]))
        half, rne = lru_cache(None)(helper.half), lru_cache(None)(helper.rne)
        prior, unused_oracle = d.imports()
        np, traversal = prior.np, prior.traversal
        graph = np.load(graph_path, allow_pickle=False)
        software = {}
        for layer in range(1, 8):
            for position in range(4):
                p = ARM / f"layer{layer:02d}_position{position}_stages.npz"
                with np.load(bind(artifacts[str(p)]), allow_pickle=False) as data:
                    software[layer, position] = {k: data[k].copy() for k in data.files}
        endpoint_path = ARM / "layer08_position3_endpoint.npz"
        with np.load(bind(artifacts[str(endpoint_path)]), allow_pickle=False) as data:
            endpoint = {k: data[k].copy() for k in data.files}
        l0 = []
        for position in range(4):
            p = BUILD / f"general_b_hidden_drift_e866008ca948_attempt002/layer00_position{position}_stages.npz"
            with np.load(bind(inherited[str(p)]), allow_pickle=False) as data:
                l0.append({k: data[k].copy() for k in data.files})
        cf = read_bound(bf["reference_source"])
        for position in range(4):
            embedding = bind(cf["embeddings"][position]["input"])
            d.require(np.array_equal(traversal.load_hidden_bits(embedding),
                                     l0[position]["input_hidden"]),
                      f"L0/P{position} reviewed token embedding differs")

        retained = {}
        templates = {}
        for row in old["transactions"]:
            if row["layer"] != 0:
                continue
            p = row["position"]
            result = read_bound(row["result"])
            prepared = read_bound(result["prepared"])
            tx = result["transaction"]
            payload = bind(tx["raw"]["trace"]).read_bytes()
            bind(tx["output_state"])
            bind(prepared["binary"])
            templates[p] = [(int(line[6:8], 16), int(line[8:12], 16))
                            for line in payload.decode("ascii").splitlines()]
            for stage in range(18):
                actual = prior.frontier.trace_stage(
                    payload, p, stage, l0[p][f"stage{stage:02d}"].size,
                    p + 1 if stage in (8, 9) else None)
                d.require(np.array_equal(actual, l0[p][f"stage{stage:02d}"]),
                          f"retained L0 RTL/software differs P{p}/S{stage}")
            retained[p] = {"transaction": tx, "prepared": prepared}

        def stages(state, stop=18):
            return {s: state[f"stage{s:02d}"] for s in range(stop + 1)}

        def trace_for(arrays, position):
            # Reuse the authenticated controller schedule, not its arithmetic values.
            trace, cursors, previous = [], {8: 0, 9: 0}, None
            for stage, index in templates[min(position, 2)]:
                if stage not in arrays:
                    previous = stage
                    continue
                if stage in (8, 9):
                    if stage != previous:
                        for key in range(position + 1):
                            trace.append((stage, key, int(arrays[stage][cursors[stage]]), position))
                            cursors[stage] += 1
                else:
                    trace.append((stage, index, int(arrays[stage][index]), position))
                previous = stage
            d.require(len(trace) == sum(a.size for a in arrays.values()), "trace schedule coverage")
            return trace

        source = out / "source"
        source.mkdir()
        for rec in bf["source_closure"]:
            path = Path(rec["path"])
            if path.parent == B / "source":
                shutil.copyfile(path, source / path.name)
        shutil.copyfile(RTL, source / RTL.name)
        shutil.copyfile(TB, source / TB.name)
        engine = source / f"{TOP}.sv"
        original = engine.read_text("ascii")
        injected = """    wire [15:0] o_down_inner_w;
    wire o_down_invalid_w, o_down_saturation_w;
    ace3_fp16_o_down_sum residual_inner (
      .attention_o_i(o_mem[hidden_index_q]), .mlp_down_i(down_mem[hidden_index_q]),
      .sum_o(o_down_inner_w), .invalid_o(o_down_invalid_w),
      .saturation_o(o_down_saturation_w));
"""
        changed = replace_once(original, "    ace3_fp16_residual_add_core #(.VECTOR_SIZE(896)) res2 (",
                               injected + "    ace3_fp16_residual_add_core #(.VECTOR_SIZE(896)) res2 (")
        changed = replace_once(
            changed,
            ".projection_f16_i(down_mem[hidden_index_q]),.residual_f16_i(res1_mem[hidden_index_q])",
            ".projection_f16_i((o_down_invalid_w || o_down_saturation_w) ? 16'h7e00 : o_down_inner_w),.residual_f16_i(activation_mem[hidden_index_q])")
        d.require(d.header(changed) == d.header(original), "public decoder interface changed")
        engine.write_text(changed, encoding="ascii")
        main_cpp = source / f"{TOP}_main.cpp"
        cpp = main_cpp.read_text("ascii")
        prefix_branch = """
        if (prefix_only) {
            if (active_layer_index != 8 || transaction_position != 0 ||
                !state_in.empty() || h.expected_trace.size() != 1792)
                throw std::runtime_error("prefix-only contract");
            h.idle(); h.reset(); h.load(1,0); h.load(2,0); h.load(0,0);
            checking=true; h.start(0,0,0);
            const uint64_t deadline=cycles+40000000;
            while (trace_count<h.expected_trace.size()) {
                if(cycles>deadline) throw std::runtime_error("prefix timeout");
                h.tick();
            }
            if (failures || final_count || done_count)
                throw std::runtime_error("prefix mismatch or overrun");
            h.top.final(); h.close_raw();
            std::cout<<"ASSOCIATION_PREFIX_PASS stages=0,1 records="<<trace_count<<"\\n";
            return 0;
        }
"""
        cpp = replace_once(cpp, "        bool fail_after_raw=false;",
                           "        bool fail_after_raw=false;\n        bool prefix_only=false;")
        cpp = replace_once(cpp, '            else if(argument=="--fail-after-raw")',
                           '            else if(argument=="--prefix-only") prefix_only=true;\n            else if(argument=="--fail-after-raw")')
        cpp = replace_once(cpp, "        if (transaction_mode) {", prefix_branch + "        if (transaction_mode) {")
        main_cpp.write_text(cpp, encoding="ascii")
        tools = {name: d.tool(name, flag) for name, flag in
                 (("verilator", "--version"), ("iverilog", "-V"), ("vvp", "-V"),
                  ("make", "--version"), ("g++", "--version"))}
        d.SOURCE = source
        commands = {}
        for layer in range(1, 9):
            command = d.decoder_command(tools, layer, out / f"layer{layer:02d}/obj", True)
            command.insert(-1, str(source / RTL.name))
            commands[layer] = command
        d.write(out / "freeze.json", {
            "review": d.record(REVIEW), "software_manifest": d.record(SOFTWARE / "output_manifest.json"),
            "input_bindings": list(bound.values()),
            "source": d.record(self_snapshot),
            "source_closure": [d.record(p) for p in sorted(source.iterdir())],
            "imported_python": [d.record(Path(m.__file__)) for m in list(sys.modules.values())
                                if getattr(m, "__file__", None) and
                                Path(m.__file__).suffix == ".py" and
                                Path(m.__file__).is_relative_to(ROOT)],
            "public_top_contract": d.header(original), "parameters": bf["parameters"],
            "compile_commands": commands, "tools": tools, "python": sys.version,
            "executable": sys.executable, "numpy": np.__version__,
            "arithmetic": "RNE16(RNE16(attention_o + mlp_down) + incoming_hidden)",
            "unchanged": "Stage12/MLP, native asymmetric G128 AWQ, Q-only fused-Q48 RoPE, general exponential, FP16 activation/KV",
            "tokens": [9707, 1879, 0, 358], "official_attempt": False,
            "gate": "finite AND (abs<=0.125 OR (rel<0.001 AND ordered_FP16_ULP<=1)); denominator=max(abs(reference),2^-14)",
            "reference_policy": sf["reference_policy"],
            "scope": "L0/P0-2 authenticated RTL stages0-17 reused; first L0/P3 B RTL transaction uses B's own P2 state; only its unchanged stages0-17 feed changed residual. New L1-7 binaries start empty and restore only own states.",
            "endpoint": "L8 S0/S1 consume actual L7/P3 hidden. These stages do not use position/RoPE/KV; physical start_position=0 in a fresh empty engine. Stop after Q, no L8 S2-18 or candidate L8 KV claim.",
            "stop": "First local exact or unchanged independent gate failure, never advance a failing transaction.",
            "claims_excluded": ["production promotion", "binary64-v1 admission", "tail",
                                "greedy dialogue", "synthesis", "PPA", "FPGA", "model admission"],
        })
        traversal.run_logged = lambda argv, log: timed(list(argv), log.parent, log.stem, 1800)
        tensor_metadata = {r["name"]: r for r in bf["checkpoint_tensors"]}

        def prepare(model, layer, position, hidden, arrays, directory):
            vectors = traversal.materialize_transaction_vectors(
                model, layer, position, hidden, directory / "vectors")
            for tensor in vectors["tensors"]:
                rec = tensor["checkpoint_tensor"]
                d.require(rec["sha256"] == tensor_metadata[rec["name"]]["sha256"],
                          "official tensor binding changed")
            final = arrays[18] if 18 in arrays else np.zeros(896, dtype="<u2")
            vectors.update(traversal.materialize_runtime_vector_contract(
                layer, position, final, trace_for(arrays, position), directory / "vectors"))
            for rec in [vectors["input"], vectors["rope_coefficients"], vectors["trace"],
                        vectors["final_hidden"], *[t["serialized"] for t in vectors["tensors"]]]:
                bind(rec)
            return vectors

        def compare(layer, position, stage, actual):
            ref = graph[f"l{layer}_p{position}_s{stage}_reference"].reshape(-1)
            d.require(actual.shape == ref.shape, "independent stage geometry")
            row = {"node": [layer, position, stage],
                   **prior.comparison.comparison(actual, ref)}
            gates.append(row)
            d.write(out / f"gate_{len(gates):04d}.json", row)
            d.require(row["failure_count"] == 0, "unchanged independent FP16 gate failure")
            return row

        # Compile the unchanged public top/port/parameter contract before any execution.
        phase, active = "public_top_compile", [1, None]
        first_directory = out / "layer01"
        first_directory.mkdir()
        timed(commands[1], first_directory, "compile", 1800)
        from safetensors import safe_open
        with safe_open(str(checkpoint), framework="np") as model:
            active, phase = [0, 3], "first_L0_P3_RTL"
            td = out / "l0_position3_unchanged_prefix"
            td.mkdir()
            arrays = {s: graph[f"l0_p3_s{s}_actual"].reshape(-1) for s in range(19)}
            vectors = prepare(model, 0, 3, l0[3]["input_hidden"], arrays, td)
            parent = retained[2]
            binary = bind(parent["prepared"]["binary"])
            state = bind(parent["transaction"]["output_state"])
            d.write(td / "prepared.json", {
                "binary": d.record(binary), "input_state": d.record(state),
                "vectors": vectors, "retained_arithmetic": "B unchanged stages0-17 only"})
            tx, _ = traversal.execute_transaction(
                binary, 0, 3, l0[3]["input_hidden"], vectors, td / "vectors",
                td / "runtime", td / "unchanged_B.state", state)
            payload = Path(tx["raw"]["trace"]["path"]).read_bytes()
            for stage in range(18):
                actual = prior.frontier.trace_stage(payload, 3, stage, arrays[stage].size,
                                                    4 if stage in (8, 9) else None)
                d.require(np.array_equal(actual, l0[3][f"stage{stage:02d}"]),
                          f"first L0/P3 RTL differs from reviewed operands S{stage}")
                l0[3][f"stage{stage:02d}"] = actual
            d.write(td / "result.json", {"transaction": tx, "stages0_17_exact": True,
                                       "stage18_not_candidate_state": True})

            phase, active = "operator_cases", None

            def add(a, b):
                value = half(int(a)) + half(int(b))
                d.require(abs(value) <= 65504, "residual case overflow")
                return 0x8000 if value == 0 and int(a) == int(b) == 0x8000 else rne(value)

            cases = out / "operator_cases.hex"
            count = 0
            with cases.open("x", encoding="ascii") as stream:
                all_states = l0 + [software[key] for key in sorted(software)]
                for state in all_states:
                    for h, o, down in zip(state["input_hidden"], state["stage11"],
                                          state["stage17"], strict=True):
                        inner = add(o, down)
                        final = add(inner, h)
                        stream.write(f"{int(h):04x} {int(o):04x} {int(down):04x} {inner:04x} {final:04x}\n")
                        count += 1
                for h, o, down in [(0, 0, 0x8000), (0x8000, 0x8000, 0x8000),
                                    (1, 1, 1), (0, 0x3ff, 1), (0, 0x3c00, 0x1000),
                                    (0, 0x3c01, 0x1000), (1, 0x7bff, 0xfbff),
                                    (0, 0x8400, 1), (0, 0xbc00, 0x9000),
                                    (0x662c, 0xb752, 0x3a1f)]:
                    inner = add(o, down)
                    final = add(inner, h)
                    stream.write(f"{h:04x} {o:04x} {down:04x} {inner:04x} {final:04x}\n")
                    count += 1
            d.write(out / "operator_freeze.json", {
                "cases": d.record(cases), "count": count,
                "oracle": d.record(helper_path), "rounding": "exact Fraction, nearest FP16 ties-even",
                "ordering": "L0/P0-3 first 3584 cases; then reviewed L1-7/P0-3 operands; ten directed cases",
                "protocol": "reset; per-result three-cycle backpressure; index/last/flags/busy; invalid, overflow, X, clear"})
            primitive = out / "operator.vvp"
            phase = "operator_compile"
            timed([tools["iverilog"]["executable"]["path"], "-g2012", "-s",
                   "ace3_o_down_then_hidden_tb", "-o", str(primitive),
                   *[str(source / name) for name in
                     ("ace3_fp16_fixed.sv", "ace3_fp16_residual_add_core.sv", RTL.name, TB.name)]],
                  out, "operator_compile", 120)
            phase = "operator_simulation"
            timed([tools["vvp"]["executable"]["path"], str(primitive),
                   f"+CASES={cases}", f"+OUTPUT={out / 'operator_actual.hex'}"],
                  out, "operator_simulation", 120)
            results = np.asarray([[int(x, 16) for x in line.split()]
                                  for line in (out / "operator_actual.hex").read_text().splitlines()],
                                 dtype="<u2")
            d.require(results.shape == (count, 2), "operator output cardinality")
            d.write(out / "operator_result.json", {
                "status": "PASS", "cases": count, "binary": d.record(primitive),
                "actual": d.record(out / "operator_actual.hex"),
                "freeze": d.record(out / "operator_freeze.json")})
            hidden = {}
            for p in range(4):
                hidden[p] = results[p * 896:(p + 1) * 896, 1].copy()
                compare(0, p, 18, hidden[p])
            hidden_parents = {p: d.record(out / "operator_actual.hex") for p in range(4)}

            for layer in range(1, 8):
                directory = out / f"layer{layer:02d}"
                if layer != 1:
                    directory.mkdir()
                    phase, active = "decoder_compile", [layer, None]
                    timed(commands[layer], directory, "compile", 1800)
                binary = directory / f"obj/V{TOP}"
                binary_rec = d.record(binary)
                d.write(directory / "binary.json", {
                    "binary": binary_rec, "freeze": d.record(out / "freeze.json"),
                    "generated_abi": [d.record(p) for p in sorted(binary.parent.iterdir())
                                      if p.suffix in (".h", ".cpp")]})
                next_hidden, next_parents = {}, {}
                state, keys, values = None, [], []
                for position in range(4):
                    phase, active = "preparation", [layer, position]
                    sw = software[layer, position]
                    d.require(np.array_equal(hidden[position], sw["input_hidden"]),
                              "actual hidden no longer matches reviewed oracle input")
                    for cache, field in ((keys, "input_cache_k"), (values, "input_cache_v")):
                        d.require(np.array_equal(np.asarray(cache, dtype="<u2").reshape(-1, 128),
                                                 sw[field]), "own actual KV differs from oracle input")
                    td = directory / f"position{position:03d}"
                    td.mkdir()
                    arrays = stages(sw)
                    vectors = prepare(model, layer, position, hidden[position], arrays, td)
                    bind(binary_rec)
                    if state:
                        bind(state)
                    d.write(td / "prepared.json", {
                        "binary": binary_rec, "input_state": state, "input_hidden": hidden_parents[position],
                        "hidden_position_slice": position if layer == 1 else None,
                        "vectors": vectors, "oracle": artifacts[str(
                            ARM / f"layer{layer:02d}_position{position}_stages.npz")]})
                    phase = "simulation"
                    tx, actual_final = traversal.execute_transaction(
                        binary, layer, position, hidden[position], vectors, td / "vectors",
                        td / "runtime", td / "candidate.state",
                        Path(state["path"]) if state else None)
                    payload = Path(tx["raw"]["trace"]["path"]).read_bytes()
                    phase = "comparison"
                    decoded = {
                        stage: prior.frontier.trace_stage(
                            payload, position, stage, arrays[stage].size,
                            position + 1 if stage in (8, 9) else None)
                        for stage in range(19)}
                    with (td / "actual_operands.npz").open("xb") as stream:
                        np.savez(stream, input_hidden=hidden[position],
                                 input_cache_k=np.asarray(keys, dtype="<u2").reshape(-1, 128),
                                 input_cache_v=np.asarray(values, dtype="<u2").reshape(-1, 128),
                                 **{f"stage{s:02d}": a for s, a in decoded.items()})
                    for stage, actual in decoded.items():
                        d.require(np.array_equal(actual, arrays[stage]),
                                  f"local exact stage mismatch L{layer}/P{position}/S{stage}")
                        compare(layer, position, stage, actual)
                    d.require(np.array_equal(actual_final, decoded[18]), "final/trace mismatch")
                    if (layer, position) == (7, 0):
                        d.write(td / "index62.json", {
                            "hidden": f"{int(hidden[position][62]):04x}",
                            "o": f"{int(decoded[11][62]):04x}",
                            "down": f"{int(decoded[17][62]):04x}",
                            "inner": f"{add(decoded[11][62], decoded[17][62]):04x}",
                            "actual": f"{int(actual_final[62]):04x}",
                            "reference": f"{int(graph['l7_p0_s18_reference'].reshape(-1)[62]):04x}"})
                    d.write(td / "result.json", {
                        "transaction": tx, "stage_gates": gates[-19:], "all_local_stages_exact": True,
                        "input_state": state, "actual_operands": d.record(td / "actual_operands.npz")})
                    transactions.append({"layer": layer, "position": position,
                                         "result": d.record(td / "result.json")})
                    keys.append(decoded[6].tolist())
                    values.append(decoded[7].tolist())
                    state = tx["output_state"]
                    next_hidden[position], next_parents[position] = actual_final, tx["output"]
                    print(f"completed L{layer}/P{position} gates={len(gates)}", flush=True)
                hidden, hidden_parents = next_hidden, next_parents

            active, phase = [8, 3], "endpoint_compile"
            directory = out / "layer08"
            directory.mkdir()
            timed(commands[8], directory, "compile", 1800)
            binary = directory / f"obj/V{TOP}"
            d.require(np.array_equal(hidden[3], endpoint["input_hidden"]), "endpoint actual parent mismatch")
            vectors = prepare(model, 8, 0, hidden[3], stages(endpoint, 1), directory)
            raw = directory / "raw"
            raw.mkdir()
            d.write(directory / "prepared.json", {
                "binary": d.record(binary), "vectors": vectors, "input_hidden": hidden_parents[3],
                "logical_position": 3, "physical_position": 0, "input_state": None,
                "no_position_dependent_stages": True})
            phase = "endpoint_simulation"
            timed([str(binary), "--layer-index", "8", "--vector-dir", str(directory / "vectors"),
                   "--tensor-dir", str(directory / "vectors"), "--raw-dir", str(raw),
                   "--prefix-only"], directory, "simulation", 1800)
            payload = (raw / "trace.hex").read_bytes()
            phase = "endpoint_comparison"
            for stage in range(2):
                actual = prior.frontier.trace_stage(payload, 0, stage, 896)
                d.require(np.array_equal(actual, endpoint[f"stage{stage:02d}"]),
                          "endpoint local exact mismatch")
                compare(8, 3, stage, actual)
            d.write(out / "result.json", {
                "status": "PASS_BOUNDED_RTL_CANDIDATE", "normal_host_review": "PENDING",
                "transactions": transactions, "stage_comparisons": len(gates),
                "prior_474_prefix": "PRESERVED", "L7_P0_S18_index62": d.record(
                    out / "layer07/position000/index62.json"),
                "GENERAL_B_L8_P3_S1": gates[-1], "endpoint_raw": d.record(raw / "trace.hex"),
                "endpoint_has_no_L8_KV_or_position_dependent_execution": True,
                "elapsed_seconds": time.monotonic() - started, "phase_times": phases,
                "binary64_v1_admission": False, "production_promotion": False})
    except (RuntimeError, OSError, ValueError, KeyError, ArithmeticError,
            subprocess.SubprocessError) as error:
        failed_gate = gates[-1] if gates and gates[-1]["failure_count"] else None
        d.write(out / "failure.json", {
            "phase": phase, "active": active, "error": str(error), "transactions": transactions,
            "stage_gates": gates, "phase_times": phases, "elapsed_seconds": time.monotonic() - started,
            "first_failed_gate": failed_gate,
            "failure_taxonomy": "changed_residual_trajectory_gate" if failed_gate else
                "numerical_or_protocol" if phase in (
                "comparison", "endpoint_comparison", "operator_simulation") else "evaluator_no_completion",
            "root_cause_hypothesis": (
                "Two-add residual association changes the propagated FP16 hidden/KV trajectory beyond the unchanged independent gate; not evidence of a local RNE defect."
                if failed_gate else
                f"The {phase} boundary did not complete its exact arithmetic/protocol/dependency contract: {error}. Cause is not established by compile or process status."),
            "regression": "Preserve this attempt and its first failing boundary; any repair gets a fresh ID with the same independent gate and input lineage.",
            "no_admission_or_promotion": True})
        raise
    finally:
        d.write(out / "output_manifest.json", {
            "artifacts": [d.record(p) for p in sorted(out.rglob("*"))
                          if p.is_file() and "obj" not in p.parts],
            "normal_host_review": "PENDING"})


if __name__ == "__main__":
    main()
