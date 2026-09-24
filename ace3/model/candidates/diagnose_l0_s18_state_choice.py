#!/usr/bin/env python3
"""Bounded FP16 residual-association diagnostics; never imports candidate state into RTL."""
import argparse
import csv
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
FRONTIER = BUILD / "general_b_hidden_drift_6c53311f8a6a_attempt002"
FAILED = BUILD / "general_b_hidden_drift_e866008ca948_attempt004"
PARENT = BUILD / "general_b_hidden_drift_d7f5d9101813_attempt003"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/6c53311f8a6a/round-0001.json")
ARMS = {"o_down_then_hidden": (1, 2, 0), "hidden_down_then_o": (0, 2, 1)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "missing diagnostic module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    with path.open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.parent == BUILD and out.name.startswith("general_b_hidden_drift_l0_s18_state_choice_"),
            "output outside bounded writable namespace")
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    tools = load_module("state_choice_bindings", FRONTIER / "compiled_source.py")
    read, record, bind = tools.read, tools.record, tools.bind
    manifests = {}
    for directory in (FRONTIER, FAILED, PARENT):
        path = directory / "output_manifest.json"
        manifests[directory] = {r["path"]: r for r in read(bind(record(path)))["artifacts"]}
    for directory, names in (
        (FRONTIER, ("compiled_source.py", "freeze.json", "result.json", "state_accounting.json")),
        (FAILED, ("compiled_source.py", "freeze.json", "adapter_repair.json", "result.json")),
        (PARENT, ("freeze.json", "retained_graph.npz")),
    ):
        for name in names:
            bind(manifests[directory][str(directory / name)])
    review = read(bind(record(REVIEW)))
    require(review["producer_role"] == "reviewer" and review["review"]["status"] == "done"
            and review["mission_id"] == "6c53311f8a6a", "reviewed frontier unavailable")
    inherited = read(PARENT / "freeze.json")
    sources = {r["path"]: r for r in inherited["bindings"]}
    for rec in sources.values():
        if rec["path"].endswith(".py"):
            bind(rec)
    previous = load_module("state_choice_previous", FAILED / "compiled_source.py")
    split = load_module("state_choice_split", bind(sources[str(previous.SPLIT)]))
    native = load_module("state_choice_native", bind(sources[str(previous.NATIVE)]))
    helper_path = BUILD / "general_b_l8p3s1_index223_5a41bc443f05_attempt002/run.py"
    helper = load_module("state_choice_exact", bind(sources[str(helper_path)]))
    helper.half = lru_cache(maxsize=None)(helper.half)
    half, rne = helper.half, lru_cache(maxsize=None)(helper.rne)
    np, torch, d = split.np, split.torch, split.d
    torch.set_num_threads(1)
    from safetensors import safe_open
    bf = read(bind(sources[str(previous.B / "frozen.json")]))
    checkpoint = bind(bf["checkpoint"])
    retained_l0 = []
    for rec in read(FAILED / "adapter_repair.json")["reused_inputs"]:
        with np.load(bind(rec), allow_pickle=False) as data:
            retained_l0.append({k: data[k].copy() for k in data.files})
    prior_failure = read(FAILED / "result.json")
    require(prior_failure["stage_comparisons"] == 475
            and prior_failure["first_failure"]["node"] == [7, 0, 18], "first-failure contract")
    spec = {
        "operation": "two source-level FP16 pairwise residual associations",
        "operand_order": ["incoming_hidden", "stage11_attention_o", "stage17_mlp_down"],
        "arms": {name: list(order) for name, order in ARMS.items()},
        "formula": "RNE16(RNE16(operand[i] + operand[j]) + operand[k])",
        "rationale": "Enumerate the two other pairings of the three existing FP16 residual operands. "
                     "Stage12 and its MLP input remain unchanged. Two finite FP16 adds suffice; "
                     "no companion, wider persistent state, reference operand, coordinate selector, "
                     "or learned constant. Apply each fixed association at every layer/position.",
        "scope": "Software state-choice diagnostic, not a changed production arithmetic contract.",
        "frozen_history": [9707, 1879, 0, 358],
        "reuse": "Authenticated B L0 stages0-17 and original independent reference arrays only. "
                "Recompute L0 S18; regenerate L1+ own hidden/K/V in layer-position order.",
        "stop": "First unchanged finite/absolute/relative/ULP gate failure; no later transaction. "
                "A native transaction computes its full trace before stage-ordered comparison.",
        "maximum_endpoint": "L8/P3/S1, reached only after all L0-7/P0-3 gates clear",
        "rejected_mechanisms": "No direct-RNE or compensated residual-companion trajectory rerun.",
        "support": "Clear the authenticated 474-comparison prefix, comparison475, remaining L7 "
                   "history and GENERAL B L8/P3/S1; otherwise no supported choice in this family.",
        "gate": "finite AND (abs<=0.125 OR (rel<0.001 AND ordered_FP16_ULP<=1)); "
                "relative denominator=max(abs(reference),2^-14)",
        "binary64_v1": "All existing FAILs retained, not reevaluated or waived.",
        "review": "Normal independent Host Reviewer required.",
    }
    write(out / "freeze.json", {
        "specification": spec, "bindings": list(tools.BOUND.values()),
        "source": record(Path(__file__)), "public_interfaces": inherited["public_interfaces"],
        "public_contract": "No RTL/top/port/parameter edits or simulator execution.",
        "reference_policy": inherited["reference_policy"],
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "numpy": np.__version__, "torch": torch.__version__},
    })
    compile(Path(__file__).read_text(), str(Path(__file__)), "exec")
    graph = np.load(PARENT / "retained_graph.npz", allow_pickle=False)
    counts = {"exact_additions": 0, "stage_gates": 0}

    def retained(layer, position, stage, side="actual"):
        return graph[f"l{layer}_p{position}_s{stage}_{side}"].reshape(-1)

    def scalar(bits):
        return {"bits": f"{int(bits):04x}", "exact": str(half(bits))}

    def add(left, right):
        require(left.shape == right.shape, "residual operand shape")
        a = left.view("<f2").astype(np.float64)
        b = right.view("<f2").astype(np.float64)
        total = a + b
        require(np.isfinite(total).all() and (np.abs(total) <= 65504).all(),
                "invalid/overflowing residual intermediate")
        result = total.astype("<f2").view("<u2")
        for i, (x, y) in enumerate(zip(left, right, strict=True)):
            exact = half(x) + half(y)
            expected = rne(exact)
            if exact == 0 and int(x) == 0x8000 and int(y) == 0x8000:
                expected = 0x8000
            require(int(result[i]) == expected, "independent Fraction RNE disagreement")
        counts["exact_additions"] += len(result)
        return result

    def choice(incoming, stages, order):
        operands = (incoming, stages[11], stages[17])
        i, j, k = order
        inner = add(operands[i], operands[j])
        return add(inner, operands[k]), inner

    def compare(layer, position, stage, actual):
        reference = retained(layer, position, stage, "reference")
        require(actual.shape == reference.shape, "gate shape mismatch")
        require(((actual & 0x7c00) != 0x7c00).all()
                and ((reference & 0x7c00) != 0x7c00).all(), "nonfinite gate")
        aq = (actual.view("<f2").astype(np.float64) * (1 << 24)).astype(np.int64)
        rq = (reference.view("<f2").astype(np.float64) * (1 << 24)).astype(np.int64)
        ordered = lambda bits: np.where(bits & 0x8000,
                                        0x8000 - (bits & 0x7fff).astype(np.int64),
                                        0x8000 + bits.astype(np.int64))
        delta = np.abs(aq - rq)
        ulp = np.abs(ordered(actual) - ordered(reference))
        accepted = (delta <= (1 << 21)) | ((delta * 1000 < np.maximum(np.abs(rq), 1024)) & (ulp <= 1))
        gate = d.compare(actual, reference)
        require(bool(accepted.all()) == gate["within_tolerance"], "exact/vector gate disagreement")
        counts["stage_gates"] += 1
        return {"node": [layer, position, stage], "within_tolerance": bool(accepted.all()),
                "failed": [{"index": int(i), **helper.scalar(actual[i], reference[i]),
                            "retained_B": scalar(retained(layer, position, stage)[i])}
                           for i in np.flatnonzero(~accepted)]}

    directed = [(0, 0x8000), (0x8000, 0x8000), (1, 1), (0x3ff, 1),
                (0x3c00, 0x1000), (0x3c01, 0x1000), (0x4000, 0xbc00),
                (0x7bff, 0xfbff), (0x8400, 1), (0xbc00, 0x9000)]
    add(np.asarray([a for a, _ in directed], dtype="<u2"),
        np.asarray([b for _, b in directed], dtype="<u2"))
    metadata = {r["name"]: r for r in bf["checkpoint_tensors"]}

    def load_values(model, layer):
        values = {}
        prefix = f"model.layers.{layer}."
        for name, meta in metadata.items():
            if name.startswith(prefix):
                data = model.get_tensor(name)
                require(hashlib.sha256(data.tobytes()).hexdigest() == meta["sha256"],
                        f"model tensor changed: {name}")
                require(data.dtype.itemsize in (2, 4), "unsupported native AWQ tensor")
                values[name.replace(prefix, "model.layers.0.") + ":"] = (
                    data.view("<u2").reshape(-1).tolist() if data.dtype.itemsize == 2
                    else data.view("<u4").reshape(-1))
        require(values, "missing layer tensors")
        return values

    boundary = read(FRONTIER / "state_accounting.json")["first_perturbation"]
    require(boundary["node"] == [0, 0, 18] and boundary["changed_scalars"] == 42,
            "reviewed first perturbation changed")
    changed = np.flatnonzero(retained_l0[0]["stage18"] != retained(0, 0, 18)).tolist()
    require(changed == [row["index"] for row in boundary["coordinates"]],
            "42-coordinate retained boundary mismatch")
    outcomes = []
    for name, order in ARMS.items():
        directory = out / name
        directory.mkdir()
        hidden, comparisons, transactions, phases = [], [], [], []
        l0_comparisons, failure, witness = [], None, None
        for position, state in enumerate(retained_l0):
            stages = {s: state[f"stage{s:02d}"] for s in range(19)}
            require(all(np.array_equal(stages[s], retained(0, position, s)) for s in range(18)),
                    "reused L0 prefix is not authenticated B")
            result, inner = choice(state["input_hidden"], stages, order)
            if position == 0:
                require(not np.array_equal(result, state["stage18"])
                        and not np.array_equal(result, retained(0, 0, 18)),
                        "choice duplicates rejected companion or unchanged B at first boundary")
                with (directory / "l0_p0_operands.csv").open("x", newline="", encoding="ascii") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["index", "in_reviewed_42", "hidden_bits", "hidden_exact",
                                     "o_bits", "o_exact", "down_bits", "down_exact",
                                     "native_B_bits", "rejected_companion_bits", "inner_bits",
                                     "inner_exact", "candidate_bits", "candidate_exact",
                                     "candidate_minus_B", "candidate_minus_companion"])
                    for i in range(896):
                        h, o, down = (int(state["input_hidden"][i]), int(stages[11][i]), int(stages[17][i]))
                        b, c = int(retained(0, 0, 18)[i]), int(state["stage18"][i])
                        writer.writerow([i, i in changed, f"{h:04x}", str(half(h)),
                                         f"{o:04x}", str(half(o)), f"{down:04x}", str(half(down)),
                                         f"{b:04x}", f"{c:04x}", f"{int(inner[i]):04x}",
                                         str(half(inner[i])), f"{int(result[i]):04x}", str(half(result[i])),
                                         str(half(result[i]) - half(b)), str(half(result[i]) - half(c))])
                boundary_changes = {
                    "vs_B": np.flatnonzero(result != retained(0, 0, 18)).tolist(),
                    "vs_rejected_companion": np.flatnonzero(result != state["stage18"]).tolist(),
                    "reviewed_42": changed,
                }
            with (directory / f"layer00_position{position}_state.npz").open("xb") as stream:
                np.savez(stream, input_hidden=state["input_hidden"], inner=inner, stage18=result)
            row = compare(0, position, 18, result)
            l0_comparisons.append(row)
            if not row["within_tolerance"]:
                failure = row
                break
            hidden.append(result)
        with safe_open(str(checkpoint), framework="numpy") as model:
            for layer in range(1, 8):
                if failure is not None:
                    break
                phase = time.monotonic()
                values = load_values(model, layer)
                project = split.operators.projection(values, False)
                next_hidden, keys, vals = [], [], []
                for position in range(4):
                    incoming = hidden[position]
                    cache_k = np.asarray(keys, dtype="<u2").reshape(-1, 128).copy()
                    cache_v = np.asarray(vals, dtype="<u2").reshape(-1, 128).copy()
                    require(cache_k.shape[0] == position and cache_v.shape[0] == position,
                            "candidate history length mismatch")
                    calls = {"q": 0, "k": 0}
                    runner = native.bind(native.run_token, {
                        "_module": project,
                        "_rope": split.split_rope(native.run_token.__globals__["_rope"],
                                                 14, 2, True, False, calls),
                    })
                    final, trace = runner(values, incoming.tolist(), position, keys, vals, accurate_silu=True)
                    groups = {}
                    for stage, index, value, token in trace:
                        require(token == position, "trace token mismatch")
                        groups.setdefault(stage, []).append((index, value))
                    stages = {s: d.decode_stage_records(rows, s, position) for s, rows in groups.items()}
                    require(set(stages) == set(range(19)) and calls == {"q": 1, "k": 1},
                            "public software stage/operator ABI changed")
                    require(np.array_equal(final, stages[18]), "native final trace mismatch")
                    require(np.array_equal(keys[-1], stages[6]) and np.array_equal(vals[-1], stages[7])
                            and np.array_equal(np.asarray(keys[:-1], dtype="<u2").reshape(-1, 128), cache_k)
                            and np.array_equal(np.asarray(vals[:-1], dtype="<u2").reshape(-1, 128), cache_v),
                            "persistent candidate cache mismatch")
                    native_final = stages[18].copy()
                    stages[18], inner = choice(incoming, stages, order)
                    filename = f"layer{layer:02d}_position{position}_stages.npz"
                    with (directory / filename).open("xb") as stream:
                        np.savez(stream, input_hidden=incoming, input_cache_k=cache_k,
                                 input_cache_v=cache_v, native_final=native_final, inner=inner,
                                 **{f"stage{s:02d}": a for s, a in stages.items()})
                    transactions.append({"layer": layer, "position": position, "arrays": filename,
                                         "hidden_parent": f"L{layer-1}/P{position}/S18",
                                         "kv_parent": f"own L{layer}/P0-{position-1}" if position else "empty"})
                    for stage in range(19):
                        row = compare(layer, position, stage, stages[stage])
                        row["affected_cone_comparison"] = len(comparisons) + 1
                        comparisons.append(row)
                        if [layer, position, stage] == [7, 0, 18]:
                            witness = {"gate": helper.scalar(stages[18][62], retained(7, 0, 18, "reference")[62]),
                                       "hidden": scalar(incoming[62]), "o": scalar(stages[11][62]),
                                       "down": scalar(stages[17][62]), "inner": scalar(inner[62]),
                                       "native_final": scalar(native_final[62])}
                        if not row["within_tolerance"]:
                            failure = row
                            break
                    if failure is not None:
                        failure["operand_arrays"] = filename
                        break
                    next_hidden.append(stages[18])
                phases.append({"layer": layer, "seconds": time.monotonic() - phase})
                if failure is None:
                    hidden = next_hidden
            if failure is None:
                values = load_values(model, 8)
                ns = native.run_token.__globals__
                n1 = ns["_expect_finite"](ns["rmsnorm"](
                    hidden[3].tolist(), values["model.layers.0.input_layernorm.weight:"])[0], "norm1")
                q = split.operators.projection(values, False)(
                    values, "self_attn.q_proj", n1, 896,
                    values["model.layers.0.self_attn.q_proj.bias:"], accepted_q_projection=True)
                endpoint = [np.asarray(n1, dtype="<u2"), np.asarray(q, dtype="<u2")]
                with (directory / "layer08_position3_endpoint.npz").open("xb") as stream:
                    np.savez(stream, input_hidden=hidden[3], stage00=endpoint[0], stage01=endpoint[1])
                for stage, actual in enumerate(endpoint):
                    row = compare(8, 3, stage, actual)
                    comparisons.append(row)
                    if not row["within_tolerance"]:
                        failure = row
                        failure["operand_arrays"] = "layer08_position3_endpoint.npz"
                        break
        write(directory / "comparisons.json", {"layer0": l0_comparisons, "affected_cone": comparisons})
        outcome = {
            "name": name, "association": list(order), "l0_p0_changed_coordinates": boundary_changes,
            "L0_stages0_17": "BIT_IDENTICAL_AUTHENTICATED_REUSE",
            "prior_474_gate_prefix_preserved": len(comparisons) >= 474
                and all(r["within_tolerance"] for r in comparisons[:474]),
            "prior_474_gate_prefix_passes": sum(r["within_tolerance"] for r in comparisons[:474]),
            "comparison475": comparisons[474] if len(comparisons) >= 475 else "NOT_REACHED",
            "L7_P0_S18_index62": witness if witness is not None else "NOT_REACHED_STOPPED_EARLIER",
            "GENERAL_B_L8_P3_S1": next((r for r in comparisons if r["node"] == [8, 3, 1]), "NOT_REACHED"),
            "first_failure": failure, "transactions": transactions, "phase_timings": phases,
            "supported_choice": failure is None,
            "failure_taxonomy": "changed_residual_state_trajectory_gate_failure" if failure else None,
            "root_cause_hypothesis": "A different general FP16 residual association changes rounded "
                "hidden states and their downstream hidden/K/V cone; no unique producer is inferred.",
            "regression": "Exact two-add oracle, L0 prefix reuse, own hidden/K/V continuity, "
                          "independent unchanged stage gate; halt before the next transaction.",
        }
        write(directory / "result.json", outcome)
        outcomes.append(outcome)
    graph.close()
    result = {
        "status": ("SUPPORTED_BOUNDED_SOFTWARE_CHOICE" if any(r["supported_choice"] for r in outcomes)
                   else "NO_SUPPORTED_REALIZABLE_STATE_CHOICE"),
        "scope": "Only the two predeclared uniform two-add association choices, not an impossibility proof.",
        "arms": outcomes, "independent_oracle_counts": counts,
        "directed_add_cases": len(directed), "elapsed_seconds": time.monotonic() - start,
        "binary64_v1": "FAIL_PRESERVED_NOT_REEVALUATED", "rtl_runs": 0,
        "implementation_adopted": False, "review_status": "PENDING_NORMAL_HOST_REVIEWER",
        "claim_limit": "Software diagnostic only; no precision policy, model admission or token claim.",
    }
    write(out / "result.json", result)
    write(out / "output_manifest.json", {"artifacts": [
        record(path) for path in sorted(out.rglob("*")) if path.is_file()
    ]})
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                      "arms": [{"name": r["name"], "first_failure": r["first_failure"],
                                "witness": r["L7_P0_S18_index62"]} for r in outcomes]}, sort_keys=True))


if __name__ == "__main__":
    main()
