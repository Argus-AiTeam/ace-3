#!/usr/bin/env python3
"""Conditional state cuts of the retained compensated GENERAL B failure."""
import csv
from fractions import Fraction
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "build"
OUT = BUILD / "general_b_hidden_drift_6c53311f8a6a_attempt001"
PARENT = BUILD / "general_b_hidden_drift_d7f5d9101813_attempt003"
SCREEN = BUILD / "general_b_hidden_drift_865a56e59eaf_attempt001"
FAILED = BUILD / "general_b_hidden_drift_e866008ca948_attempt004"
BOUND = {}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def record(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def bind(rec):
    path = Path(rec["path"])
    if str(path) not in BOUND:
        require(record(path) == rec, f"retained binding changed: {path}")
        BOUND[str(path)] = rec
    return path


def write(name, value):
    with (OUT / name).open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "missing module loader")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    started = time.monotonic()
    require(not (OUT / "freeze.json").exists(), "attempt already consumed")
    compile(Path(__file__).read_text(), __file__, "exec")
    manifests = {}
    for directory in (PARENT, SCREEN, FAILED):
        manifest = directory / "output_manifest.json"
        bind(record(manifest))
        manifests[directory] = {r["path"]: r for r in read(manifest)["artifacts"]}
        for name in ("result.json", "freeze.json"):
            bind(manifests[directory][str(directory / name)])
    for name in ("retained_graph.npz", "producer_accounting.json"):
        bind(manifests[PARENT][str(PARENT / name)])
    for name in ("compiled_source.py", "adapter_repair.json", "comparisons.json"):
        bind(manifests[FAILED][str(FAILED / name)])
    prior = read(FAILED / "result.json")
    require(prior["stage_comparisons"] == 475
            and prior["first_failure"]["node"] == [7, 0, 18],
            "retained failure contract changed")
    reviews = []
    review_root = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
    for mission in ("d7f5d9101813", "865a56e59eaf", "e866008ca948"):
        path = review_root / mission / "round-0001.json"
        review = read(bind(record(path)))
        require(review["producer_role"] == "reviewer"
                and review["review"]["status"] == "done"
                and review["mission_id"] == mission, "missing genuine predecessor review")
        reviews.append({"path": str(path), "created_at": review["created_at"],
                        "review": review["review"]})
    inherited = read(PARENT / "freeze.json")
    sources = {r["path"]: r for r in inherited["bindings"]}
    for rec in sources.values():
        if rec["path"].endswith(".py"):
            bind(rec)
    tools = module("retained_compensation_tools", FAILED / "compiled_source.py")
    split = module("boundary_split", bind(sources[str(tools.SPLIT)]))
    native = module("boundary_native", bind(sources[str(tools.NATIVE)]))
    helper_path = BUILD / "general_b_l8p3s1_index223_5a41bc443f05_attempt002/run.py"
    helper = module("boundary_exact", bind(sources[str(helper_path)]))
    helper.half = lru_cache(maxsize=None)(helper.half)
    half, rne = helper.half, lru_cache(maxsize=None)(helper.rne)
    independent_path = BUILD / "independent_fp16_trajectory_20260906_1133/source/official_single_decoder_layer.py"
    independent = module("boundary_independent_awq", bind(sources[str(independent_path)]))
    np, torch, d = split.np, split.torch, split.d
    torch.set_num_threads(1)
    from safetensors import safe_open
    bfreeze = read(bind(sources[str(tools.B / "frozen.json")]))
    checkpoint = bind(bfreeze["checkpoint"])
    layer0 = read(FAILED / "adapter_repair.json")["reused_inputs"]
    states = {}
    for position, rec in enumerate(layer0):
        with np.load(bind(rec), allow_pickle=False) as data:
            states[0, position] = {k: data[k].copy() for k in data.files}
    for tx in prior["transactions"]:
        path = FAILED / tx["arrays"]
        with np.load(bind(manifests[FAILED][str(path)]), allow_pickle=False) as data:
            states[tx["layer"], tx["position"]] = {k: data[k].copy() for k in data.files}
    spec = read(OUT / "preregistered.json")
    write("freeze.json", {
        "bindings": list(BOUND.values()),
        "sources": [record(Path(__file__)), record(OUT / "execution.command.sh"),
                    record(OUT / "preregistered.json")],
        "reviews": reviews, "specification": spec,
        "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
        "public_interfaces_unchanged": inherited["public_interfaces"],
        "public_contract_action": "Preserved declarations only; no RTL compilation or execution.",
        "reference_policy_unchanged": inherited["reference_policy"],
    })
    graph = np.load(PARENT / "retained_graph.npz", allow_pickle=False)

    def retained(layer, stage, side="actual", position=0):
        return graph[f"l{layer}_p{position}_s{stage}_{side}"].reshape(-1)

    def f64(bits):
        return np.asarray(bits, dtype="<u2").view("<f2").astype(np.float64)

    def q24(bits):
        return (f64(bits) * (1 << 24)).astype(np.int64)

    def scalar(bits):
        return {"bits": f"{int(bits):04x}", "exact": str(half(bits))}

    def compare(actual, reference):
        actual, reference = np.asarray(actual, dtype="<u2"), np.asarray(reference, dtype="<u2")
        require(actual.shape == reference.shape, "gate shape mismatch")
        require(np.all((actual & 0x7c00) != 0x7c00)
                and np.all((reference & 0x7c00) != 0x7c00), "nonfinite gate operand")
        a, r = q24(actual), q24(reference)
        delta = np.abs(a - r)
        ordered = lambda b: np.where(b & 0x8000, 0x8000 - (b & 0x7fff).astype(np.int64),
                                    0x8000 + b.astype(np.int64))
        ulp = np.abs(ordered(actual) - ordered(reference))
        accepted = (delta <= (1 << 21)) | (
            (delta * 1000 < np.maximum(np.abs(r), 1024)) & (ulp <= 1))
        vector = d.compare(actual, reference)
        require(bool(np.all(accepted)) == vector["within_tolerance"],
                "independent exact-integer/vector gate disagreement")
        return {"within_tolerance": bool(np.all(accepted)),
                "failed": [{"index": int(i), **helper.scalar(actual[i], reference[i])}
                           for i in np.flatnonzero(~accepted)]}

    phases = []
    phase = time.monotonic()
    comparisons = []
    for old in read(FAILED / "comparisons.json"):
        layer, position, stage = old["node"]
        gate = compare(states[layer, position][f"stage{stage:02d}"],
                       retained(layer, stage, "reference", position))
        comparisons.append({"node": old["node"], **gate})
        require(gate["within_tolerance"] == old["comparison"]["within_tolerance"],
                "retained comparison result changed")
    require(len(comparisons) == 475 and all(r["within_tolerance"] for r in comparisons[:474])
            and comparisons[-1]["failed"][0]["index"] == 62, "first-failure regression")
    write("retained_gate_comparisons.json", comparisons)
    phases.append({"phase": "retained_475_gate_comparisons", "seconds": time.monotonic() - phase})

    phase = time.monotonic()
    accounting = []
    oracle_coordinates = 0
    first_perturbation = None
    for (layer, position), state in sorted(states.items()):
        require(np.array_equal(state["input_hidden"], states[layer - 1, position]["stage18"]
                               if layer else retained(-1, 18, position=position)),
                "candidate hidden continuity changed")
        if position:
            for field, stage in (("input_cache_k", 6), ("input_cache_v", 7)):
                expected = np.asarray([states[layer, p][f"stage{stage:02d}"]
                                       for p in range(position)], dtype="<u2")
                require(np.array_equal(state[field], expected), "candidate cache continuity changed")
        else:
            require(state["input_cache_k"].size == 0 and state["input_cache_v"].size == 0,
                    "P0 unexpectedly consumes historical cache")
        for stage in range(19):
            changed = np.flatnonzero(state[f"stage{stage:02d}"] != retained(layer, stage, position=position))
            if first_perturbation is None and changed.size:
                first_perturbation = {
                    "node": [layer, position, stage], "changed_scalars": int(changed.size),
                    "coordinates": [
                        {"index": int(i), "candidate": scalar(state[f"stage{stage:02d}"][i]),
                         "retained_B": scalar(retained(layer, stage, position=position)[i]),
                         "delta": str(half(state[f"stage{stage:02d}"][i])
                                      - half(retained(layer, stage, position=position)[i]))}
                        for i in changed],
                    "interpretation": "First intervention-created state boundary, not a unique cause of the later failure.",
                }
        if position:
            continue
        h, o, s, down = (state[k] for k in ("input_hidden", "stage11", "stage12", "stage17"))
        require(np.all(state["stage09"] == 0x3c00), "single-context softmax not unity")
        grouped_v = np.repeat(state["stage07"].reshape(2, 64), 7, axis=0).reshape(-1)
        require(np.array_equal(grouped_v, state["stage10"]), "P0 value/GQA identity failed")
        require(np.array_equal(state["stage03"], state["stage07"]), "V cache write mismatch")
        for i in range(896):
            exact = half(h[i]) + half(o[i]) + half(down[i])
            require(rne(exact) == int(state["stage18"][i]), "independent compensated residual RNE mismatch")
            require(half(state["companion_fp16"][i]) == half(h[i]) + half(o[i]) - half(s[i]),
                    "independent companion identity mismatch")
            oracle_coordinates += 1
        i = 62
        rows = {}
        for side in ("actual", "reference"):
            operands = [retained(layer - 1, 18, side)[i], retained(layer, 11, side)[i],
                        retained(layer, 12, side)[i], retained(layer, 17, side)[i]]
            rows[side] = {
                "incoming_hidden": scalar(operands[0]), "attention_o": scalar(operands[1]),
                "residual1": scalar(operands[2]), "mlp_down": scalar(operands[3]),
                "final": scalar(retained(layer, 18, side)[i]),
                "incoming_hidden_delta": str(half(h[i]) - half(operands[0])),
                "attention_o_delta": str(half(o[i]) - half(operands[1])),
                "mlp_down_delta": str(half(down[i]) - half(operands[3])),
            }
        accounting.append({
            "node": [layer, 0, 18, i], "incoming_hidden": scalar(h[i]),
            "attention_o": scalar(o[i]), "residual1": scalar(s[i]),
            "mlp_down": scalar(down[i]), "companion": scalar(state["companion_fp16"][i]),
            "native_final": scalar(state["native_final"][i]),
            "candidate_final": scalar(state["stage18"][i]),
            "exact_sum": str(half(h[i]) + half(o[i]) + half(down[i])),
            "rounding_error": str(half(state["stage18"][i]) - half(h[i]) - half(o[i]) - half(down[i])),
            "comparators": rows,
            "p0_cache": {"historical_kv_elements": 0, "softmax_unity_heads": 14,
                         "current_K_changed_vs_B": int(np.count_nonzero(state["stage06"] != retained(layer, 6))),
                         "current_V_changed_vs_B": int(np.count_nonzero(state["stage07"] != retained(layer, 7))),
                         "attention_exactly_grouped_current_V": True},
        })
    write("state_accounting.json", {"first_perturbation": first_perturbation,
                                   "position0_residual_chain": accounting})
    phases.append({"phase": "exact_residual_and_state_accounting", "seconds": time.monotonic() - phase})

    phase = time.monotonic()
    values, tensors = {}, {}
    metadata = {r["name"]: r for r in bfreeze["checkpoint_tensors"]}
    prefix = "model.layers.7."
    with safe_open(str(checkpoint), framework="numpy") as model:
        for name, meta in metadata.items():
            if name.startswith(prefix):
                data = model.get_tensor(name)
                require(hashlib.sha256(data.tobytes()).hexdigest() == meta["sha256"],
                        f"official model tensor mismatch: {name}")
                tensors[name[len(prefix):]] = data
                require(data.dtype.itemsize in (2, 4), "unsupported AWQ tensor ABI")
                values[name.replace(prefix, "model.layers.0.") + ":"] = (
                    data.view("<u2").reshape(-1).tolist() if data.dtype.itemsize == 2
                    else data.view("<u4").reshape(-1))
    project = split.operators.projection(values, False)
    ns = native.run_token.__globals__
    base = states[7, 0]

    def norm(vector):
        return np.asarray(ns["_expect_finite"](ns["rmsnorm"](
            vector.tolist(), values["model.layers.0.post_attention_layernorm.weight:"])[0],
            "conditional norm2"), dtype="<u2")

    def projection(name, vector, outputs):
        return np.asarray(project(values, name, vector.tolist(), outputs), dtype="<u2")

    def residual(left, right):
        output = []
        for a, b in zip(left, right, strict=True):
            value, invalid, saturated = ns["residual_add"](int(a), int(b))
            require(not invalid and not saturated, "conditional residual invalid")
            output.append(value)
        return np.asarray(output, dtype="<u2")

    arms = []
    cuts = spec["conditional_cuts"]
    for cut in cuts:
        s = {i: base[f"stage{i:02d}"].copy() for i in range(19)}
        h = base["input_hidden"].copy()
        if cut == "incoming_hidden":
            h = retained(6, 18).copy()
            calls = {"q": 0, "k": 0}
            runner = native.bind(native.run_token, {
                "_module": project,
                "_rope": split.split_rope(ns["_rope"], 14, 2, True, False, calls),
            })
            keys, vals = [], []
            _, trace = runner(values, h.tolist(), 0, keys, vals, accurate_silu=True)
            groups = {}
            for stage, index, value, position in trace:
                require(position == 0, "conditional trace position mismatch")
                groups.setdefault(stage, []).append((index, value))
            s = {i: d.decode_stage_records(rows, i, 0) for i, rows in groups.items()}
            require(set(s) == set(range(19)) and calls == {"q": 1, "k": 1},
                    "conditional public software trace ABI changed")
        else:
            stage = {"current_v": 7, "attention": 10, "o": 11, "residual1": 12,
                     "norm2": 13, "gate": 14, "up": 15, "silu": 16, "down": 17}[cut]
            s[stage] = retained(7, stage).copy()
            if stage == 7:
                s[3] = s[7].copy()
                s[10] = np.repeat(s[7].reshape(2, 64), 7, axis=0).reshape(-1)
            if stage <= 10:
                s[11] = projection("self_attn.o_proj", s[10], 896)
            if stage <= 11:
                s[12] = residual(s[11], h)
            if stage <= 12:
                s[13] = norm(s[12])
            if stage <= 13:
                s[14] = projection("mlp.gate_proj", s[13], 4864)
                s[15] = projection("mlp.up_proj", s[13], 4864)
            if stage <= 15:
                silu = []
                for g, u in zip(s[14], s[15], strict=True):
                    value, invalid, saturated = ns["silu_gate_exp"](int(g), int(u))
                    require(not invalid and not saturated, "conditional SiLU invalid")
                    silu.append(value)
                s[16] = np.asarray(silu, dtype="<u2")
            if stage <= 16:
                s[17] = projection("mlp.down_proj", s[16], 896)
        total = f64(h) + f64(s[11]) + f64(s[17])
        require(np.isfinite(total).all() and np.max(np.abs(total)) <= 65504,
                "conditional final overflow")
        s[18] = total.astype("<f2").view("<u2")
        for i in range(896):
            require(int(s[18][i]) == rne(half(h[i]) + half(s[11][i]) + half(s[17][i])),
                    "conditional final independent RNE mismatch")
            oracle_coordinates += 1
        gates = [{"node": [7, 0, i], **compare(s[i], retained(7, i, "reference"))}
                 for i in range(19)]
        first = next((g for g in gates if not g["within_tolerance"]), None)
        with (OUT / f"conditional_{cut}.npz").open("xb") as stream:
            np.savez(stream, input_hidden=h, **{f"stage{i:02d}": a for i, a in s.items()})
        arms.append({"cut": cut, "source": "retained_B_actual_full_vector",
                     "scope": "L7/P0 scratch conditional suffix only; not a realizable intervention",
                     "witness": helper.scalar(s[18][62], retained(7, 18, "reference")[62]),
                     "exact_sum62": str(half(h[62]) + half(s[11][62]) + half(s[17][62])),
                     "first_local_gate_failure": first, "stage_gates": gates,
                     "prior_474_changed_trajectory_gates": "NOT_EVALUATED",
                     "original_GENERAL_B_path": "NOT_REACHED"})
    write("conditional_cuts.json", arms)
    phases.append({"phase": "conditional_local_suffixes", "seconds": time.monotonic() - phase})

    phase = time.monotonic()
    dots = []
    for output_stage, input_stage, name in ((11, 10, "self_attn.o_proj"), (17, 16, "mlp.down_proj")):
        q = independent._torch_unpack(tensors[name + ".qweight"]).numpy().astype(np.int64)[:, 62]
        z = independent._torch_unpack(tensors[name + ".qzeros"]).numpy().astype(np.int64)[:, 62]
        scales = tensors[name + ".scales"].view("<u2")[:, 62]
        totals = {"candidate": Fraction(0), "retained_B": Fraction(0), "reference": Fraction(0)}
        terms = []
        path = OUT / f"stage{output_stage:02d}_index62_exact_terms.csv"
        with path.open("x", newline="", encoding="ascii") as stream:
            writer = csv.writer(stream)
            writer.writerow(["input_index", "qweight", "qzero", "scale_bits", "weight_exact",
                             "candidate_input", "B_input", "reference_input", "candidate_term",
                             "B_term", "reference_term", "candidate_minus_B", "candidate_minus_reference"])
            for i, quantized in enumerate(q):
                weight = (int(quantized) - int(z[i // 128])) * half(scales[i // 128])
                a, b, r = (base[f"stage{input_stage:02d}"][i], retained(7, input_stage)[i],
                           retained(7, input_stage, "reference")[i])
                at, bt, rt = (half(v) * weight for v in (a, b, r))
                totals["candidate"] += at
                totals["retained_B"] += bt
                totals["reference"] += rt
                writer.writerow([i, int(quantized), int(z[i // 128]), f"{int(scales[i // 128]):04x}",
                                 str(weight), f"{int(a):04x}", f"{int(b):04x}", f"{int(r):04x}",
                                 str(at), str(bt), str(rt), str(at - bt), str(at - rt)])
                terms.append({"index": i, "candidate_input": scalar(a), "retained_B_input": scalar(b),
                              "reference_input": scalar(r), "weight": str(weight),
                              "candidate_minus_B": str(at - bt)})
        actual_output = int(base[f"stage{output_stage:02d}"][62])
        dots.append({"node": [7, 0, output_stage, 62], "input_stage": input_stage,
                     "exact_dot": {k: str(v) for k, v in totals.items()},
                     "exact_dot_candidate_minus_B": str(totals["candidate"] - totals["retained_B"]),
                     "own_operand_direct_RNE": f"{rne(totals['candidate']):04x}",
                     "actual_output": scalar(actual_output),
                     "matches_direct_RNE": rne(totals["candidate"]) == actual_output,
                     "output_minus_exact_dot": str(half(actual_output) - totals["candidate"]),
                     "all_terms": path.name,
                     "largest_absolute_B_deltas": sorted(
                         terms, key=lambda row: abs(Fraction(row["candidate_minus_B"])), reverse=True)[:16]})
    write("exact_projection_accounting.json", dots)
    phases.append({"phase": "independent_native_AWQ_dot_accounting", "seconds": time.monotonic() - phase})
    graph.close()
    result = {
        "status": "NO_SUPPORTED_CAUSAL_INTERVENTION",
        "retained_first_failure": comparisons[-1], "prior_gate_passes": 474,
        "fresh_retained_comparisons": 475,
        "first_intervention_created_state_boundary": first_perturbation["node"],
        "conditional_cut_count": len(arms),
        "conditional_witness_clear_cuts": [a["cut"] for a in arms if a["witness"]["accepted"]],
        "conditional_all_local_stage_gates_clear": [
            a["cut"] for a in arms if a["first_local_gate_failure"] is None],
        "independent_exact_residual_coordinates": oracle_coordinates,
        "p0_historical_KV_contribution": "Absent: all eight P0 input histories are empty.",
        "p0_current_K_contribution_to_attention_value": "Zero for the retained finite single-context rows; all softmax probabilities are exactly one.",
        "supported_intervention": None,
        "scientific_blocker": "Full-vector B state cuts are conditional diagnostics, not a general mathematical repair. "
                              "They do not establish regenerated P1-3 hidden/KV gates or re-entry at L8/P3/S1/S8. "
                              "No tested implementable mechanism preserves the affected 474-comparison prefix.",
        "failure_taxonomy": "conditional_upstream_state_trajectory_divergence",
        "root_cause_hypothesis": "The rejected companion changes an early layer-final hidden boundary; "
                                 "inherited hidden and the current-V/MLP branches propagate rounded state differences. "
                                 "Exact residual and AWQ term accounting bounds this statement without claiming a unique producer.",
        "regression": "Authenticated prior 475 comparisons; hidden and cache continuity; P0 singleton attention; "
                      "exact residual RNE; full-vector local suffix cuts against the unchanged independent reference.",
        "binary64_v1_status": "FAIL_PRESERVED_NOT_REEVALUATED",
        "original_GENERAL_B_L8_P3_S1_and_S8": "NOT_REACHED",
        "implementation_adopted": False, "rtl_runs": 0,
        "independent_review": 1, "self_maintenance": 0,
        "review_status": "PENDING_NORMAL_HOST_REVIEWER",
        "phase_timings": phases, "elapsed_seconds": time.monotonic() - started,
    }
    write("result.json", result)
    write("output_manifest.json", {"artifacts": [
        record(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "execution.log"
    ]})
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
