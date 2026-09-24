#!/usr/bin/env python3
"""Retained layer0/1 reconstruction and dependency-pruned numerical sensitivity."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as functional

from diagnose_position3_upstream_hidden import (
    A, REFERENCE, ROOT, Evidence, compared, half_bits, record, require, values, write_json,
)
from decoder_layer0_oracle import _expect_finite, _rope
from fp16_adaptation_oracle import rmsnorm, silu_gate_exp
from attention_oracle import attention_score, attention_softmax, attention_value


PRIOR = ROOT / "build/token358_position3_upstream_hidden_trajectory_localization_attempt001/measurement002"
PROJECTIONS = {
    1: (0, "self_attn.q_proj", True), 2: (0, "self_attn.k_proj", True),
    3: (0, "self_attn.v_proj", True), 11: (10, "self_attn.o_proj", False),
    14: (13, "mlp.gate_proj", False), 15: (13, "mlp.up_proj", False),
    17: (16, "mlp.down_proj", False),
}
DEPS = {
    0: (-1,), 1: (0,), 2: (0,), 3: (0,), 4: (1,), 5: (2,), 6: (5,), 7: (3,),
    8: (4, 6, -2), 9: (8,), 10: (9, 7, -3), 11: (10,), 12: (-1, 11), 13: (12,),
    14: (13,), 15: (13,), 16: (14, 15), 17: (16,), 18: (12, 17),
}


def namespace(evidence):
    scope = {"np": np, "torch": torch, "GROUP_SIZE": 128, "_require": require,
             "AWQ_REVERSE_ORDER": (0, 4, 1, 5, 2, 6, 3, 7)}
    source = REFERENCE / "official_single_decoder_layer.py"
    names = {"_torch_unpack", "_torch_linear", "_torch_rmsnorm"}
    nodes = [node for node in ast.parse(evidence.read(source)).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in nodes} == names, "independent operator source incomplete")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(source), "exec"), scope)
    return scope


class Operators:
    def __init__(self, layer, tensors, scope):
        self.prefix = f"model.layers.{layer}."
        self.tensors, self.scope = tensors, scope
        self.weights = {}

    def tensor(self, bits):
        return torch.from_numpy(values(bits))

    def linear(self, stage, stages, dot_only=False):
        operand, suffix, bias = PROJECTIONS[stage]
        name = self.prefix + suffix
        if name not in self.weights:
            unpack = self.scope["_torch_unpack"]
            q = unpack(self.tensors[name + ".qweight"]).to(torch.float64)
            z = unpack(self.tensors[name + ".qzeros"]).to(torch.float64)
            scale = torch.from_numpy(self.tensors[name + ".scales"].astype(np.float64))
            self.weights[name] = ((q - z.repeat_interleave(128, dim=0))
                                  * scale.repeat_interleave(128, dim=0))
        result = self.tensor(stages[operand]).reshape(1, -1) @ self.weights[name]
        if bias and not dot_only:
            result = result + torch.from_numpy(self.tensors[name + ".bias"].astype(np.float64))
        return result.reshape(-1)

    def independent_raw(self, stage, stages, incoming, caches):
        if stage in (0, 13):
            suffix = "input_layernorm.weight" if stage == 0 else "post_attention_layernorm.weight"
            return self.scope["_torch_rmsnorm"](
                self.tensor(incoming if stage == 0 else stages[12]).reshape(1, -1),
                self.tensors[self.prefix + suffix]).reshape(-1)
        if stage in PROJECTIONS:
            return self.linear(stage, stages)
        if stage in (4, 5):
            x = self.tensor(stages[1 if stage == 4 else 2]).reshape(-1, 64)
            frequency = 1.0 / (1_000_000.0 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
            angle = torch.outer(torch.arange(3, 4, dtype=torch.float64), frequency)
            low, high = x[:, :32], x[:, 32:]
            return torch.cat((low * angle.cos() - high * angle.sin(),
                              high * angle.cos() + low * angle.sin()), dim=1).reshape(-1)
        if stage in (6, 7):
            return self.tensor(stages[5 if stage == 6 else 3])
        if stage == 8:
            q, k = self.tensor(stages[4]).reshape(14, 64), self.tensor(caches["k"])
            return torch.stack([q[h] @ k[:, h // 7].T / 8 for h in range(14)]).reshape(-1)
        if stage == 9:
            return torch.softmax(self.tensor(stages[8]).reshape(14, 4), dim=-1).reshape(-1)
        if stage == 10:
            p, v = self.tensor(stages[9]).reshape(14, 4), self.tensor(caches["v"])
            return torch.stack([p[h] @ v[:, h // 7] for h in range(14)]).reshape(-1)
        if stage == 12:
            return self.tensor(stages[11]) + self.tensor(incoming)
        if stage == 16:
            return functional.silu(self.tensor(stages[14])) * self.tensor(stages[15])
        require(stage == 18, f"unknown stage {stage}")
        return self.tensor(stages[12]) + self.tensor(stages[17])

    def independent(self, stage, stages, incoming, caches):
        return self.independent_raw(stage, stages, incoming, caches).to(
            torch.float16).numpy().view("<u2").reshape(-1)

    def rtl_model(self, stage, stages, incoming, caches):
        if stage in (0, 13):
            suffix = "input_layernorm.weight" if stage == 0 else "post_attention_layernorm.weight"
            weight = self.tensors[self.prefix + suffix].view("<u2").reshape(-1).tolist()
            operand = incoming if stage == 0 else stages[12]
            return np.asarray(_expect_finite(rmsnorm(operand.tolist(), weight)[0], "norm"),
                              dtype="<u2")
        if stage in PROJECTIONS:
            if stage == 3:
                dot = half_bits(self.linear(stage, stages, dot_only=True).numpy())
                bias = self.tensors[self.prefix + "self_attn.v_proj.bias"].astype(np.float64)
                return half_bits(values(dot) + bias)
            return half_bits(self.linear(stage, stages).numpy())
        if stage in (4, 5):
            operand = stages[1 if stage == 4 else 2]
            return np.asarray(_rope(operand.tolist(), operand.size // 64, 3), dtype="<u2")
        if stage in (6, 7):
            return stages[5 if stage == 6 else 3].copy()
        if stage == 8:
            q = stages[4].reshape(14, 64)
            result = []
            for h in range(14):
                for p in range(4):
                    score = attention_score(q[h].tolist(), caches["k"][p, h // 7].tolist(),
                                            [True] * 64, 3, p)
                    require(not score.invalid and not score.cache_miss, "score invalid")
                    result.append(score.score_f16)
            return np.asarray(result, dtype="<u2")
        if stage == 9:
            result = []
            for row in stages[8].reshape(14, 4):
                softmax = attention_softmax(row.tolist(), list(range(4)), [True] * 4,
                                            [False] * 4, [False] * 4, 3)
                require(not softmax.invalid and not softmax.cache_miss and not softmax.row_error,
                        "softmax invalid")
                result.extend(softmax.probabilities_f16)
            return np.asarray(result, dtype="<u2")
        if stage == 10:
            result = []
            for h, p in enumerate(stages[9].reshape(14, 4)):
                for dim in range(64):
                    value = attention_value(p.tolist(), caches["v"][:, h // 7, dim].tolist(),
                                            [True] * 4, [False] * 4)
                    require(not value.invalid and not value.cache_miss and not value.row_error,
                            "context invalid")
                    result.append(value.value_f16)
            return np.asarray(result, dtype="<u2")
        if stage == 16:
            return np.asarray(_expect_finite(
                [silu_gate_exp(int(g), int(u)) for g, u in zip(stages[14], stages[15], strict=True)],
                "silu"), dtype="<u2")
        return half_bits(self.independent_raw(stage, stages, incoming, caches).numpy())


def load_tensors(evidence, prepared, ref):
    hashes = {item["name"]: item["sha256"] for item in ref["checkpoint_tensor_hashes"]}
    tensors = {}
    for item in prepared["vectors"]["tensors"]:
        meta = item["checkpoint_tensor"]
        dtype = "<u2" if meta["dtype"] == "F16" else "<u4"
        data = np.asarray([int(row, 16) for row in
                           evidence.read(item["serialized"]["path"]).splitlines()], dtype=dtype)
        require(data.nbytes == meta["bytes"] and hashlib.sha256(data.tobytes()).hexdigest()
                == meta["sha256"] == hashes[meta["name"]], f"tensor binding {meta['name']}")
        tensors[meta["name"]] = data.view("<f2" if dtype == "<u2" else "<i4").reshape(meta["shape"])
    require(len(tensors) == 26, "missing tensors")
    return tensors


def plans(evidence):
    roots = [record(PRIOR / "seal.json"), record(A / "seal.json")]
    for root in roots:
        evidence.register(root)
        evidence.load(root["path"])
    previous = evidence.load(PRIOR / "result.json")
    evidence.register(previous["authenticated_inputs"])
    frozen = evidence.load(A / "frozen.json")
    original = evidence.load(frozen["original_frozen"]["path"])
    package = evidence.load(original["launch_package"]["path"])
    evidence.load(package["layer1_review"]["path"])
    layer1 = evidence.load(package["layer1_result"]["path"])
    f1 = evidence.load(layer1["frozen"]["path"])
    p1 = evidence.load(f1["launch_package"]["path"])
    evidence.load(p1["layer0_review"]["path"])
    evidence.load(p1["layer0_seal"]["path"])
    r0 = evidence.load(p1["layer0_result"]["path"])
    d0 = ROOT / "build/token358_position3_layer0_attempt001"
    f0, p0 = evidence.load(d0 / "frozen.json"), evidence.load(d0 / "prepared.json")
    require(f0["comparison_policy"] == p1["comparison_policy"] == frozen["comparison_policy"],
            "early-layer numerical gate changed")
    require(p1["token_history"] == r0["token_history"] == [9707, 1879, 0, 358],
            "early-layer token history")
    p0 = {**p0, "input_hidden": f0["embedding"]["file"], "kv_parent": r0["kv_parent"],
          "independent_parent": f0["independent_parent"]}
    p1 = {**p1, "independent_stages": p1["independent_expected_stages"]}
    rows = [(0, r0, p0), (1, evidence.load(layer1["numerical_result"]["path"]), p1)]
    for layer in range(2, 9):
        directory = (ROOT / "build/token358_position3_full_continuation_attempt001/layer02"
                     if layer == 2 else A / f"layer{layer:02d}")
        receipt = (evidence.load(frozen["recovered_layer2"]["path"]) if layer == 2
                   else evidence.load(directory / "result.json"))
        rows.append((layer, receipt, evidence.load(directory / "prepared.json")))
    return rows, roots, frozen, previous


def run(out, command, sweep_from=None):
    started = time.monotonic()
    torch.set_num_threads(1)
    evidence = Evidence()
    rows, roots, frozen, previous = plans(evidence)
    scope = namespace(evidence)
    proof = {}
    if sweep_from is not None:
        proof_root = record(sweep_from / "seal.json")
        evidence.register(proof_root)
        evidence.load(proof_root["path"])
        proof_frozen = evidence.load(sweep_from / "frozen.json")
        old_source = evidence.read(sweep_from / "source.py")
        def operator_definition(source):
            return next(ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                        if isinstance(n, ast.ClassDef) and n.name == "Operators")
        require(operator_definition(old_source) == operator_definition(Path(__file__).read_bytes()),
                "cannot reuse retained reconstruction after operator implementation change")
        for helper in proof_frozen["helpers"]:
            require(record(helper["path"]) == helper, "retained-proof helper changed")
        proof = {layer: evidence.load(sweep_from / f"layer{layer:02d}.json") for layer in range(9)}
        roots.append(proof_root)
    write_json(out / "frozen.json", {
        "kind": "retained_input_and_changed_cone_software_diagnostic_not_RTL_acceptance",
        "roots": roots, "source": record(__file__), "command": record(command),
        "helpers": [record(ROOT / "ace3/model" / name) for name in (
            "diagnose_position3_upstream_hidden.py", "diagnose_layer08_position3_stage08.py",
            "decoder_layer0_oracle.py", "fp16_adaptation_oracle.py", "attention_oracle.py",
            "qwen2_rope_oracle.py")],
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "comparison_policy": frozen["comparison_policy"],
        "scope": "Position3 layers0/1 retained operators; first differing operator changed-cone only to layer8 scores",
        "counterfactuals": [
            "Replace first local differing stage by independent policy on actual operands; RTL-model suffix, actual historical KV",
            "Insert that actual stage into independent trajectory; independent-policy suffix, independent historical KV",
        ],
        "stop_rule": "Stop each branch when bit-identical to its retained baseline; never propagate past layer8 stage08",
        "incremental_sweep": (
            "All remaining local layer0/1 policy substitutions plus each early historical K/V cut; "
            "reuse unchanged operator proofs, do not repeat the stage0 null-effect experiment"
            if proof else None),
        "prefix_RTL_replays": 0, "official_attempts": 0, "policy_changes": 0,
    })
    prior_actual = prior_independent = None
    cases = {}
    earliest = None
    layer_paths = []
    operator_counts = {"early_independent_reconstruction": 0, "retained_rtl_model": 0,
                       "counterfactual": 0}
    for layer, receipt, prepared in rows:
        layer_start = time.monotonic()
        transaction = evidence.load(receipt["transaction"]["path"])
        ref = evidence.load(prepared["independent_parent"]["path"])
        require(receipt["layer"] == ref["layer"] == layer and receipt["position"] == 3
                and ref["history"] == [9707, 1879, 0], f"layer identity {layer}")
        actual = evidence.trace(transaction["raw"]["trace"]["path"], 3)
        independent = {s: evidence.bits(prepared["independent_stages"][str(s)]["path"])
                       for s in range(19)}
        hidden = evidence.bits(prepared["vectors"]["input"]["path"], True)
        require(evidence.read(prepared["vectors"]["input"]["path"])
                == evidence.read(prepared["input_hidden"]["path"]), f"input binding {layer}")
        require(hashlib.sha256(hidden.tobytes()).hexdigest() == transaction["input"]["sha256"],
                f"input semantic binding {layer}")
        ihidden = hidden if layer == 0 else prior_independent
        if layer:
            require(np.array_equal(hidden, prior_actual), f"actual boundary {layer}")
        require(np.array_equal(actual[18], evidence.bits(receipt["output_hidden"]["path"], True)),
                f"trace/final binding {layer}")
        exact_trace = (prepared["exact_oracle"] if layer == 0 else prepared["vectors"])["trace"]
        require(evidence.read(transaction["raw"]["trace"]["path"]) == evidence.read(exact_trace["path"]),
                f"retained exact-local trace {layer}")
        fields = dict(item.split("=", 1) for item in
                      evidence.read(transaction["raw"]["terminal"]["path"]).decode().split())
        require(all(fields[k] == v for k, v in {
            "natural_terminal": "1", "exit_code": "0", "done_count": "1", "final_count": "896",
            "position": "3", "layer_index": str(layer)}.items()), f"terminal {layer}")
        parent = prepared["kv_parent"]
        require(parent["layer_index"] == layer and parent["valid_positions"] == [0, 1, 2],
                f"KV parent identity {layer}")
        old = [evidence.trace(item["path"], p, 7)
               for p, item in enumerate(parent["actual_kv_traces"])]
        ac, ic = {}, {}
        for kind, stage in (("k", 6), ("v", 7)):
            ac[kind] = np.stack([x[stage].reshape(2, 64) for x in old]
                                + [actual[stage].reshape(2, 64)])
            independent_old = evidence.array(ref["own_cache"][kind]["path"])
            require(independent_old.shape == (3, 2, 64), f"independent parent geometry {layer}")
            ic[kind] = np.concatenate((independent_old, independent[stage].reshape(1, 2, 64)))
            if "actual_kv" in receipt:
                require(np.array_equal(ac[kind], evidence.array(receipt["actual_kv"][kind]["path"])),
                        f"actual consumed KV parent {layer}/{kind}")
        require(all(np.array_equal(x[5], x[6]) and np.array_equal(x[3], x[7])
                    for x in [*old, actual]), f"cache write order {layer}")
        ops = Operators(layer, load_tensors(evidence, prepared, ref), scope)
        stage_rows = []
        validated_actual = {s: True for s in proof.get(layer, {}).get(
            "rtl_model_reconstructed_stages", [])}

        def check_actual(stage):
            if stage not in validated_actual:
                reconstructed = ops.rtl_model(stage, actual, hidden, ac)
                require(np.array_equal(reconstructed, actual[stage]),
                        f"RTL-model retained reconstruction {layer}/{stage}: "
                        f"{compared(actual[stage], reconstructed)}")
                validated_actual[stage] = True
                operator_counts["retained_rtl_model"] += 1

        if layer < 2 and proof:
            stage_rows = proof[layer]["stages"]
            require(all(x["independent_retained_exact"] and x["rtl_model_retained_exact"]
                        for x in stage_rows), "retained operator proof incomplete")
            for entry in stage_rows:
                stage = entry["stage"]
                if not entry["independent_on_actual_operands"]["different_count"]:
                    continue
                if layer == 0 and stage == 0:
                    continue
                if earliest is None:
                    earliest = {"layer": layer, "stage": stage,
                                "scope": "first remaining local difference; stage0 null effect retained separately",
                                "same_operand_difference": entry["independent_on_actual_operands"]}
                for mode in ("actual", "independent"):
                    stages, incoming, cache = ((actual, hidden, ac) if mode == "actual"
                                               else (independent, ihidden, ic))
                    method = ops.independent if mode == "actual" else ops.rtl_model
                    name = f"layer{layer:02d}_stage{stage:02d}_{mode}"
                    cases[name] = {
                        "mode": mode, "seed": method(stage, stages, incoming, cache),
                        "seed_layer": layer, "seed_stage": stage, "hidden": incoming.copy(),
                        "layers": [],
                    }
            for kind in ("k", "v"):
                for mode in ("actual", "independent"):
                    incoming = hidden if mode == "actual" else ihidden
                    other_cache = ic if mode == "actual" else ac
                    name = f"layer{layer:02d}_historical_{kind}_{mode}"
                    cases[name] = {
                        "mode": mode, "seed_layer": layer, "seed_stage": None,
                        "historical_kind": kind, "historical_seed": other_cache[kind][:3].copy(),
                        "hidden": incoming.copy(), "layers": [],
                    }
        elif layer < 2:
            for stage in range(19):
                reconstructed = ops.independent(stage, independent, ihidden, ic)
                require(np.array_equal(reconstructed, independent[stage]),
                        f"independent retained reconstruction {layer}/{stage}")
                operator_counts["early_independent_reconstruction"] += 1
                conditional = ops.independent(stage, actual, hidden, ac)
                check_actual(stage)
                comparison = compared(actual[stage], conditional)
                stage_rows.append({
                    "stage": stage, "trajectory": compared(actual[stage], independent[stage]),
                    "independent_on_actual_operands": comparison,
                    "independent_retained_exact": True, "rtl_model_retained_exact": True,
                    "actual_trace": transaction["raw"]["trace"]["path"],
                    "independent_stage": prepared["independent_stages"][str(stage)]["path"],
                })
                if earliest is None and comparison["different_count"]:
                    require(all(np.array_equal(actual[s], independent[s]) for s in range(stage)),
                            "earliest local stage already has a separate preceding trajectory difference")
                    earliest = {"layer": layer, "stage": stage,
                                "same_operand_difference": comparison,
                                "actual_trace": transaction["raw"]["trace"],
                                "independent_stage": prepared["independent_stages"][str(stage)],
                                "scope": "earliest current-position local stage, not historical KV producer"}
                    cases = {
                        "repair_first_on_actual": {"mode": "actual", "seed": conditional.copy(),
                                                  "seed_layer": layer, "seed_stage": stage,
                                                  "hidden": hidden.copy(), "layers": []},
                        "inject_first_on_independent": {"mode": "independent",
                                                       "seed_layer": layer, "seed_stage": stage,
                                                       "seed": actual[stage].copy(),
                                                       "hidden": ihidden.copy(), "layers": []},
                    }
            require(sum(len(x["trajectory"]["failure_indices"]) for x in stage_rows)
                    == receipt["material_mismatches"] == 0, f"unchanged early gate {layer}")
        layer_cases = {}
        for name, case in cases.items():
            baseline, base_hidden, cache = ((actual, hidden, ac) if case["mode"] == "actual"
                                             else (independent, ihidden, ic))
            state = {s: x.copy() for s, x in baseline.items()}
            counter_cache = {k: x.copy() for k, x in cache.items()}
            changed = {-1: not np.array_equal(case["hidden"], base_hidden)}
            if layer == case["seed_layer"] and "historical_kind" in case:
                kind = case["historical_kind"]
                counter_cache[kind][:3] = case["historical_seed"]
                changed[-2 if kind == "k" else -3] = not np.array_equal(
                    counter_cache[kind][:3], cache[kind][:3])
            evaluated, changes = [], []
            for stage in range(9 if layer == 8 else 19):
                is_seed = layer == case["seed_layer"] and stage == case["seed_stage"]
                if is_seed:
                    state[stage] = case["seed"].copy()
                elif any(changed.get(dep, False) for dep in DEPS[stage]):
                    check_actual(stage)
                    method = ops.rtl_model if case["mode"] == "actual" else ops.independent
                    state[stage] = method(stage, state, case["hidden"], counter_cache)
                    evaluated.append(stage)
                    operator_counts["counterfactual"] += 1
                changed[stage] = not np.array_equal(state[stage], baseline[stage])
                if stage in (6, 7):
                    counter_cache["k" if stage == 6 else "v"][3] = state[stage].reshape(2, 64)
                if changed[stage]:
                    changes.append({"stage": stage, "vs_retained_baseline": compared(
                        state[stage], baseline[stage])})
            last = 8 if layer == 8 else 18
            saved = out / f"{name}_layer{layer:02d}_stage{last:02d}.npy"
            with saved.open("xb") as stream:
                np.save(stream, state[last], allow_pickle=False)
            row = {"layer": layer, "evaluated_stages": evaluated, "changed_stages": changes,
                   "output": record(saved), "suffix_died_here": not changed[last]}
            if layer == case["seed_layer"]:
                row["seed"] = {"stage": case["seed_stage"],
                               "historical_kind": case.get("historical_kind"),
                               "vs_baseline": compared(
                                   case["historical_seed"], cache[case["historical_kind"]][:3])
                               if "historical_kind" in case else compared(
                                   case["seed"], baseline[case["seed_stage"]])}
            if layer == 8:
                row["scores_vs_original_independent"] = compared(state[8], independent[8])
                row["witnesses"] = [
                    {"index": i, "actual": float(values(actual[8])[i]),
                     "independent": float(values(independent[8])[i]),
                     "counterfactual": float(values(state[8])[i]),
                     "actual_bits": f"{int(actual[8][i]):04x}",
                     "independent_bits": f"{int(independent[8][i]):04x}",
                     "counterfactual_bits": f"{int(state[8][i]):04x}"}
                    for i in (13, 15)]
                require(compared(actual[8], independent[8])["failure_indices"] == [13, 15],
                        "original layer08 failure changed")
            else:
                case["hidden"] = state[18]
            case["layers"].append(row)
            layer_cases[name] = row
        layer_row = {
            "layer": layer, "actual_input": prepared["input_hidden"],
            "independent_input": (prepared["input_hidden"] if layer == 0
                                  else rows[layer - 1][2]["independent_stages"]["18"]),
            "incoming_hidden": compared(hidden, ihidden), "stages": stage_rows,
            "prior_measurement": None if layer < 2 else str(PRIOR / f"layer{layer:02d}.json"),
            "historical_kv": {k: compared(ac[k][:3], ic[k][:3]) for k in ("k", "v")},
            "actual_kv_parents": parent["actual_kv_traces"],
            "independent_kv_parents": ref["own_cache"],
            "current_kv": {k: compared(ac[k][3], ic[k][3]) for k in ("k", "v")},
            "counterfactuals": layer_cases,
            "rtl_model_reconstructed_stages": sorted(validated_actual),
            "seconds": time.monotonic() - layer_start,
        }
        path = out / f"layer{layer:02d}.json"
        write_json(path, layer_row)
        layer_paths.append(record(path))
        prior_actual, prior_independent = actual[18], independent[18]
        print(json.dumps({"layer": layer, "local_differing_stages": [
            x["stage"] for x in stage_rows if x["independent_on_actual_operands"]["different_count"]],
            "active_cases": sum(not r["suffix_died_here"] for r in layer_cases.values()),
            "cases": len(layer_cases),
            "seconds": layer_row["seconds"]}), flush=True)
    require(earliest is not None, "no early differing operator to isolate")
    result = {
        "status": "RETAINED_EARLY_OPERATORS_AND_CONDITIONAL_SENSITIVITY_COMPLETE",
        "earliest_current_position_local_difference": earliest,
        "historical_causal_origin": "Not localized: historical independently owned KV differences are held fixed per branch",
        "counterfactuals": {name: {"mode": case["mode"], "endpoint": case["layers"][-1],
                                  "layers": case["layers"]} for name, case in cases.items()},
        "operator_counts": operator_counts, "layers": layer_paths,
        "reused_early_operator_proofs": 38 if proof else 0,
        "authenticated_inputs": list(evidence.consumed.values()),
        "previous_measurement": record(PRIOR / "result.json"),
        "claim_boundary": (
            "Software changed-cone sensitivity, not fresh RTL or a replacement independent acceptance "
            "trajectory. A nonzero effect is not proof of a unique historical cause. Current-position "
            "seed replacement does not repair earlier-position caches or change production reference policy."
        ),
        "prefix_RTL_replays": 0, "official_attempts": 0, "acceptance_policy_changes": 0,
        "review": "pending normal Host Reviewer", "seconds": time.monotonic() - started,
    }
    write_json(out / "result.json", result)
    write_json(out / "seal.json", {"files": [record(p) for p in sorted(out.iterdir())]})
    print(json.dumps({"status": result["status"], "earliest": earliest,
                      "operator_counts": operator_counts, "seconds": result["seconds"]}), flush=True)
    for name, case in cases.items():
        endpoint = case["layers"][-1]
        print(json.dumps({"case": name, "scores13_15": [
            x["counterfactual"] for x in endpoint["witnesses"]],
            "failure_indices": endpoint["scores_vs_original_independent"]["failure_indices"]}),
            flush=True)


def repair_source(out, command, retained):
    """Discriminate the two retained successful cuts without replaying either cone."""
    from decimal import Decimal, localcontext, ROUND_HALF_EVEN
    from fractions import Fraction
    from attention_oracle import (
        decode_f16_q24, exp_approx_q24, q24_to_f16, round_div_even_unsigned,
    )
    from projection_oracle import complete_projection_output

    started = time.monotonic()
    torch.set_num_threads(1)
    evidence = Evidence()
    root = record(retained / "seal.json")
    evidence.register(root)
    evidence.load(root["path"])
    prior = evidence.load(retained / "result.json")
    evidence.register(prior["authenticated_inputs"])
    old_frozen = evidence.load(retained / "frozen.json")
    require(hashlib.sha256(evidence.read(retained / "source.py")).hexdigest()
            == old_frozen["source"]["sha256"], "retained source snapshot mismatch")
    for helper in old_frozen["helpers"]:
        evidence.read(helper["path"])
    rows, roots, frozen, _ = plans(evidence)
    require(frozen["comparison_policy"] == old_frozen["comparison_policy"],
            "retained numerical gate changed")
    sources = [record(ROOT / path) for path in (
        "ace3/model/projection_oracle.py", "ace3/model/awq_bit_oracle.py",
        "ace3/rtl/ace3_attention_softmax_core.sv",
        "ace3/rtl/ace3_awq_w4a16_projection_engine.sv",
        "ace3/contracts/awq_w4a16_projection_engine.json")]
    write_json(out / "frozen.json", {
        "kind": "retained_two_operator_source_discrimination_not_RTL_acceptance",
        "source": record(__file__), "command": record(command),
        "roots": [root, *roots], "production_sources": sources,
        "helpers": old_frozen["helpers"], "comparison_policy": frozen["comparison_policy"],
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "torch_device": "cpu", "torch_threads": 1, "decimal_precision": 80,
        "scope": "Layer0 stage09 rows and layer1 stage15 only; retained layer08 endpoints",
        "public_RTL_contract": "No RTL module, port, parameter, or arithmetic change; no RTL attempt",
        "prefix_replays": 0, "RTL_invocations": 0, "official_attempts": 0,
    })
    scope = namespace(evidence)
    local = {}
    for layer, stage in ((0, 9), (1, 15)):
        _, receipt, prepared = rows[layer]
        transaction = evidence.load(receipt["transaction"]["path"])
        ref = evidence.load(prepared["independent_parent"]["path"])
        actual = evidence.trace(transaction["raw"]["trace"]["path"], 3)
        proof = evidence.load(retained / f"layer{layer:02d}.json")
        stage_proof = next(row for row in proof["stages"] if row["stage"] == stage)
        require(stage_proof["actual_trace"] == transaction["raw"]["trace"]["path"],
                "candidate trace binding")
        ops = Operators(layer, load_tensors(evidence, prepared, ref), scope)
        raw = ops.independent_raw(stage, actual, None, {})
        conditional = raw.to(torch.float16).numpy().view("<u2").reshape(-1)
        direct = half_bits(raw.numpy())
        reconstructed = ops.rtl_model(stage, actual, None, {})
        require(np.array_equal(reconstructed, actual[stage]), "candidate RTL-model reconstruction")
        require(compared(actual[stage], conditional)
                == stage_proof["independent_on_actual_operands"], "candidate substitution changed")
        changed = np.flatnonzero(actual[stage] != conditional).tolist()
        row = {
            "layer": layer, "stage": stage, "actual_trace": transaction["raw"]["trace"],
            "independent_stage": prepared["independent_stages"][str(stage)],
            "conditional_vs_actual": compared(actual[stage], conditional),
            "numpy_vs_torch_conversion": compared(direct, conditional),
            "witnesses": [],
        }
        if stage == 15:
            name = ops.prefix + "mlp.up_proj"
            tensors = ops.tensors
            for index in changed:
                packed, lane = divmod(index, 8)
                shift = 4 * (0, 4, 1, 5, 2, 6, 3, 7)[lane]
                weights = tensors[name + ".qweight"][:, packed]
                zeros = tensors[name + ".qzeros"][:, packed]
                scales = tensors[name + ".scales"][:, index]
                exact = sum((
                    Fraction(float(x)) * Fraction(float(scales[i // 128]))
                    * (((int(weights[i]) >> shift) & 15)
                       - ((int(zeros[i // 128]) >> shift) & 15))
                    for i, x in enumerate(values(actual[13]))), Fraction())
                a, c = int(actual[stage][index]), int(conditional[index])
                av, cv = (Fraction(float(x)) for x in values([a, c]))
                require(abs(a - c) == 1 and min(av, cv) <= exact <= max(av, cv),
                        "projection witness does not bracket exact value")
                nearest = min((a, c), key=lambda b: (
                    abs(exact - Fraction(float(values([b])[0]))), b & 1))
                accumulator, integer_bits, invalid, saturation, groups = complete_projection_output(
                    actual[13].tolist(), weights.tolist(), zeros.tolist(),
                    scales.view("<u2").tolist(), lane)
                require(not invalid and not saturation and Fraction(accumulator, 1 << 48) == exact
                        and integer_bits == nearest, "exact rational/integer projection disagreement")
                value = float(raw[index])
                via_fp32 = int(half_bits(np.asarray([value], dtype=np.float32))[0])
                row["witnesses"].append({
                    "index": index, "operand_stage": 13, "tensor_prefix": name,
                    "packed_column": packed, "logical_lane": lane, "physical_shift": shift,
                    "exact_dot": str(exact), "exact_dot_decimal": float(exact),
                    "exact_q48_accumulator": accumulator, "group_accumulators": groups,
                    "midpoint": str((av + cv) / 2),
                    "exact_minus_midpoint": str(exact - (av + cv) / 2),
                    "binary64": value, "binary64_hex": value.hex(),
                    "binary64_minus_exact": str(Fraction(value) - exact),
                    "actual_bits": f"{a:04x}", "torch_bits": f"{c:04x}",
                    "numpy_bits": f"{int(direct[index]):04x}",
                    "via_fp32_bits": f"{via_fp32:04x}", "exact_RNE_bits": f"{nearest:04x}",
                    "actual_matches_exact_RNE": a == nearest,
                    "torch_matches_exact_RNE": c == nearest,
                    "torch_matches_via_fp32": c == via_fp32,
                })
        else:
            with localcontext() as context:
                context.prec = 80
                for head in sorted({index // 4 for index in changed}):
                    bits = actual[8].reshape(14, 4)[head]
                    scores = [Decimal.from_float(float(x)) for x in values(bits)]
                    maximum = max(scores)
                    ideal_exp = [(x - maximum).exp() for x in scores]
                    total = sum(ideal_exp)
                    exact_exp_q24 = [int((x * (1 << 24)).to_integral_value(
                        rounding=ROUND_HALF_EVEN)) for x in ideal_exp]
                    scores_q24 = [decode_f16_q24(int(b))[0] for b in bits]
                    approx_exp = [exp_approx_q24(max(scores_q24) - x) for x in scores_q24]
                    for column in range(4):
                        index = head * 4 + column
                        probability = ideal_exp[column] / total
                        a, c = int(actual[9][index]), int(conditional[index])
                        candidates = {a, c, int(direct[index])}
                        nearest = min(candidates, key=lambda b: (
                            abs(probability - Decimal.from_float(float(values([b])[0]))), b & 1))
                        low = Decimal.from_float(float(values([nearest - 1])[0]))
                        mid = Decimal.from_float(float(values([nearest])[0]))
                        high = Decimal.from_float(float(values([nearest + 1])[0]))
                        require((low + mid) / 2 < probability < (mid + high) / 2,
                                "Decimal probability rounding is not strictly bracketed")
                        ideal_exp_only = q24_to_f16(round_div_even_unsigned(
                            exact_exp_q24[column] << 24, sum(exact_exp_q24)))[0]
                        exact_division_only = int(half_bits(
                            float(Fraction(approx_exp[column], sum(approx_exp))))[0])
                        row["witnesses"].append({
                            "index": index, "scores_bits": [f"{int(b):04x}" for b in bits],
                            "scores": [str(x) for x in scores], "probability_decimal80": str(probability),
                            "approx_exp_q24": approx_exp, "ideal_exp_q24": exact_exp_q24,
                            "actual_bits": f"{a:04x}", "torch_bits": f"{c:04x}",
                            "numpy_bits": f"{int(direct[index]):04x}",
                            "decimal_RNE_bits": f"{nearest:04x}",
                            "ideal_exp_only_bits": f"{ideal_exp_only:04x}",
                            "exact_division_only_bits": f"{exact_division_only:04x}",
                            "rounding_boundary_margin": str(min(
                                probability - (low + mid) / 2, (mid + high) / 2 - probability)),
                        })
        local[f"layer{layer:02d}_stage{stage:02d}"] = row

    _, receipt, prepared = rows[8]
    transaction = evidence.load(receipt["transaction"]["path"])
    actual = evidence.trace(transaction["raw"]["trace"]["path"], 3)[8]
    independent = evidence.bits(prepared["independent_stages"]["8"]["path"])
    require(compared(actual, independent)["failure_indices"] == [13, 15],
            "original layer08 failure changed")
    other = ROOT / "build/token358_position3_full_continuation_attempt004"
    other_root = record(other / "seal.json")
    evidence.register(other_root)
    evidence.load(other_root["path"])
    other_receipt = evidence.load(other / "layer08/result.json")
    other_transaction = evidence.load(other_receipt["transaction"]["path"])
    require(evidence.read(transaction["raw"]["trace"]["path"])
            == evidence.read(other_transaction["raw"]["trace"]["path"]),
            "attempt003/004 retained traces differ")
    endpoints = {}
    for name in ("layer00_stage09_actual", "layer01_stage15_actual",
                 "layer00_stage09_independent", "layer01_stage15_independent"):
        endpoint = prior["counterfactuals"][name]["endpoint"]
        scores = evidence.array(endpoint["output"]["path"])
        comparison = compared(scores, independent)
        require(comparison == endpoint["scores_vs_original_independent"],
                f"retained conditional endpoint changed: {name}")
        endpoints[name] = {"output": endpoint["output"], "comparison": comparison,
                           "witnesses": endpoint["witnesses"]}
    projection = local["layer01_stage15"]["witnesses"]
    conversion_artifact = bool(projection) and all(
        w["actual_matches_exact_RNE"] and not w["torch_matches_exact_RNE"]
        and w["torch_matches_via_fp32"] and w["binary64_minus_exact"] == "0"
        for w in projection)
    softmax = local["layer00_stage09"]["witnesses"]
    exp_cut = all(w["ideal_exp_only_bits"] == w["decimal_RNE_bits"] == w["torch_bits"]
                  for w in softmax)
    result = {
        "status": "PRODUCTION_REPAIR_NOT_UNIQUELY_JUSTIFIED",
        "local_diagnostics": local, "retained_endpoints": endpoints,
        "original_failure": compared(actual, independent),
        "original_trace": transaction["raw"]["trace"],
        "attempt004_root": other_root, "attempt004_trace": other_transaction["raw"]["trace"],
        "stage15_conversion_artifact_demonstrated": conversion_artifact,
        "softmax_ideal_exp_cut_matches_on_examined_rows": exp_cut,
        "intervention_boundary": (
            "The stage15 cut is not a projection RTL repair: exact rational accumulation and integer "
            "RNE agree with actual RTL; the conditional Torch conversion disagrees and matches the "
            "FP32-intermediate result. Correct that diagnostic conversion explicitly before using "
            "it to recommend projection edits; preserve the old substitution as a failed causal hypothesis."
            if conversion_artifact else
            "The exact dot/conversion witnesses do not establish the FP32-intermediate hypothesis."
        ),
        "remaining_ambiguity": (
            "The layer0 softmax cut changes the approximation policy on retained scores. It demonstrates "
            "sufficiency in one frozen software suffix, not necessity or a unique whole-trajectory "
            "producer. Historical K/V and other upstream differences remain held fixed. No concrete "
            "synthesizable replacement has been tested; a score-only conditional endpoint does not "
            "establish all affected-stage gates. These records do not justify a fresh official attempt."
        ),
        "hypothesis_regression": {
            "taxonomy": "conditional_substitution_misattributed_to_production",
            "hypothesis": "The stage15 apparent repair is a conversion perturbation, not incorrect AWQ accumulation",
            "regression": "Layer1 stage15 index791: exact rational dot and integer FP16 RNE versus retained RTL and both casts",
        },
        "authenticated_inputs": list(evidence.consumed.values()),
        "prefix_replays": 0, "RTL_invocations": 0, "official_attempts": 0,
        "acceptance_policy_changes": 0, "repaired_path_claim": False,
        "review": "pending normal Host Reviewer", "seconds": time.monotonic() - started,
    }
    write_json(out / "result.json", result)
    write_json(out / "seal.json", {"files": [record(p) for p in sorted(out.iterdir())]})
    print(json.dumps({k: result[k] for k in (
        "status", "stage15_conversion_artifact_demonstrated",
        "softmax_ideal_exp_cut_matches_on_examined_rows", "seconds")}), flush=True)


def softmax_kv_discrimination(out, command, retained, source_retained):
    """Factor only layer0 exponential policy and independently owned historical K/V."""
    from decimal import Decimal, localcontext, ROUND_HALF_EVEN
    from fractions import Fraction
    import math
    from attention_oracle import (
        decode_f16_q24, exp_approx_q24, q24_to_f16, round_div_even_unsigned,
    )

    started = time.monotonic()
    torch.set_num_threads(1)
    evidence = Evidence()
    roots = []
    for directory in (retained, source_retained):
        root = record(directory / "seal.json")
        evidence.register(root)
        evidence.load(root["path"])
        roots.append(root)
    prior = evidence.load(retained / "result.json")
    source_result = evidence.load(source_retained / "result.json")
    # Source snapshots have different historical hashes at the same editable path.
    # Authenticate the frozen documents without treating those paths as current inputs.
    old_frozen = json.loads(evidence.read(retained / "frozen.json"))
    source_frozen = json.loads(evidence.read(source_retained / "frozen.json"))
    for result in (prior, source_result):
        evidence.register(result["authenticated_inputs"])
    evidence.register(source_frozen["production_sources"])
    old_source = evidence.read(retained / "source.py")
    require(hashlib.sha256(old_source).hexdigest() == old_frozen["source"]["sha256"],
            "retained source snapshot mismatch")
    require(hashlib.sha256(evidence.read(source_retained / "source.py")).hexdigest()
            == source_frozen["source"]["sha256"], "source diagnostic snapshot mismatch")
    def operator_definition(source):
        return next(ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                    if isinstance(n, ast.ClassDef) and n.name == "Operators")
    require(operator_definition(old_source) == operator_definition(Path(__file__).read_bytes()),
            "retained operator proof cannot be reused after an operator change")
    for helper in old_frozen["helpers"]:
        evidence.read(helper["path"])
    for source in source_frozen["production_sources"]:
        evidence.read(source["path"])
    require(source_result["stage15_conversion_artifact_demonstrated"],
            "retained projection conversion distinction missing")
    rows, plan_roots, frozen, _ = plans(evidence)
    require(frozen["comparison_policy"] == old_frozen["comparison_policy"]
            == source_frozen["comparison_policy"], "frozen gate changed")
    cases = {
        f"exp{exp}_k{k}_v{v}": {"ideal_exp": bool(exp), "independent_k": bool(k),
                               "independent_v": bool(v), "layers": []}
        for exp in range(2) for k in range(2) for v in range(2)
    }
    write_json(out / "frozen.json", {
        "kind": "layer0_softmax_historical_kv_factorial_software_diagnostic",
        "source": record(__file__), "command": record(command), "roots": roots + plan_roots,
        "helpers": old_frozen["helpers"], "production_sources": source_frozen["production_sources"],
        "comparison_policy": frozen["comparison_policy"],
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "torch_device": "cpu", "torch_threads": 1,
        "candidate": (
            "Layer0 stage09 only: e_i=RNE_integer(2^24*exp(s_i-max(s))); "
            "p_i=FP16_RNE(RNE_integer(2^24*e_i/sum(e))/2^24). "
            "Decimal precision80 exponential; original Q24 division and FP16 conversion. "
            "No LUT/interpolation, no changed scores, no changed gate."
        ),
        "independent_candidate_oracle": (
            "math.exp binary64 must agree at exponential Q24 rounding; Fraction round "
            "must agree at division RNE; exact Q24 dyadic numpy FP16 must agree at output. "
            "Decimal unquantized softmax FP16 neighbors are a separate numerical comparison."
        ),
        "factors": {name: {k: v for k, v in case.items() if k != "layers"}
                    for name, case in cases.items()},
        "held_fixed": (
            "Layer0 incoming hidden/current K/V and stages0-7, all layers1-8 historical K/V, "
            "all other RTL-model operators, original independently propagated stage expectations. "
            "Only positions0-2 of layer0 K and/or V are exchanged diagnostically."
        ),
        "stop": "Dependency-pruned actual-model suffix, ending at layer08 stage08",
        "public_RTL_contract": (
            "No module/port/parameter/source change; software-only, no RTL compilation or attempt. "
            "Historical mixed-cache branches are counterfactuals, not admissible production state."
        ),
        "authorization_rule": (
            "No production intervention without all affected-stage evidence and a concrete "
            "production implementation with independently reviewed affected-prefix requirements."
        ),
        "RTL_invocations": 0, "official_attempts": 0, "prefix_replays": 0,
    })
    scope = namespace(evidence)
    softmax_rows = {}
    previous_actual = None
    stage_outputs = {}
    layer_reports = []
    candidate_matches_retained_seed = False
    for layer, receipt, prepared in rows:
        layer_started = time.monotonic()
        transaction = evidence.load(receipt["transaction"]["path"])
        ref = evidence.load(prepared["independent_parent"]["path"])
        proof = evidence.load(retained / f"layer{layer:02d}.json")
        require(receipt["layer"] == ref["layer"] == proof["layer"] == layer
                and receipt["position"] == 3 and ref["history"] == [9707, 1879, 0],
                f"layer identity {layer}")
        actual = evidence.trace(transaction["raw"]["trace"]["path"], 3)
        last = 8 if layer == 8 else 18
        independent = {s: evidence.bits(prepared["independent_stages"][str(s)]["path"])
                       for s in range(last + 1)}
        hidden = evidence.bits(prepared["vectors"]["input"]["path"], True)
        require(evidence.read(prepared["vectors"]["input"]["path"])
                == evidence.read(prepared["input_hidden"]["path"]), f"hidden binding {layer}")
        require(hashlib.sha256(hidden.tobytes()).hexdigest() == transaction["input"]["sha256"],
                f"hidden semantic binding {layer}")
        if layer:
            require(np.array_equal(previous_actual, hidden), f"actual boundary {layer}")
        parent = prepared["kv_parent"]
        require(parent["layer_index"] == layer and parent["valid_positions"] == [0, 1, 2],
                f"KV identity {layer}")
        require(proof["actual_kv_parents"] == parent["actual_kv_traces"]
                and proof["independent_kv_parents"] == ref["own_cache"],
                f"retained cache proof binding {layer}")
        old = [evidence.trace(item["path"], p, 7)
               for p, item in enumerate(parent["actual_kv_traces"])]
        caches = {kind: np.stack([x[stage].reshape(2, 64) for x in old]
                                + [actual[stage].reshape(2, 64)])
                  for kind, stage in (("k", 6), ("v", 7))}
        independent_old = {kind: evidence.array(ref["own_cache"][kind]["path"])
                           for kind in ("k", "v")} if layer == 0 else {}
        require(all(x.shape == (3, 2, 64) for x in independent_old.values()),
                "historical cache geometry")
        ops = Operators(layer, load_tensors(evidence, prepared, ref), scope)
        reconstructed = set(proof["rtl_model_reconstructed_stages"])
        exact_trace = (prepared["exact_oracle"] if layer == 0 else prepared["vectors"])["trace"]
        require(evidence.read(transaction["raw"]["trace"]["path"])
                == evidence.read(exact_trace["path"]), f"retained local trace binding {layer}")
        saved = {f"actual_{s:02d}": actual[s] for s in range(last + 1)}
        saved.update({f"independent_{s:02d}": independent[s] for s in range(last + 1)})
        saved["actual_incoming"] = hidden
        saved.update({f"actual_cache_{k}": v for k, v in caches.items()})
        saved.update({f"independent_old_{k}": v for k, v in independent_old.items()})
        layer_cases = {}
        for name, case in cases.items():
            state = {s: x.copy() for s, x in actual.items()}
            cache = {k: x.copy() for k, x in caches.items()}
            incoming = hidden if layer == 0 else case["hidden"]
            changed = {-1: not np.array_equal(incoming, hidden)}
            if layer == 0:
                for kind, dependency in (("k", -2), ("v", -3)):
                    if case[f"independent_{kind}"]:
                        cache[kind][:3] = independent_old[kind]
                    changed[dependency] = not np.array_equal(cache[kind], caches[kind])
            stage_rows = []
            for stage in range(last + 1):
                candidate = layer == 0 and stage == 9 and case["ideal_exp"]
                affected = candidate or any(changed.get(d, False) for d in DEPS[stage])
                if candidate:
                    key = str(int(case["independent_k"]))
                    if key not in softmax_rows:
                        probabilities, witnesses = [], []
                        with localcontext() as context:
                            context.prec = 80
                            for head, bits in enumerate(state[8].reshape(14, 4)):
                                scores = [Decimal.from_float(float(x)) for x in values(bits)]
                                exps = [(x - max(scores)).exp() for x in scores]
                                quantized = [int((x * (1 << 24)).to_integral_value(
                                    rounding=ROUND_HALF_EVEN)) for x in exps]
                                independent_exp = [round(math.exp(float(x - max(scores)))
                                                         * (1 << 24)) for x in scores]
                                require(quantized == independent_exp, "exponential oracle disagreement")
                                q = [decode_f16_q24(int(b))[0] for b in bits]
                                approx = [exp_approx_q24(max(q) - x) for x in q]
                                for column, exp in enumerate(quantized):
                                    p_q24 = round_div_even_unsigned(exp << 24, sum(quantized))
                                    require(p_q24 == round(Fraction(exp << 24, sum(quantized))),
                                            "division oracle disagreement")
                                    p_bits, invalid = q24_to_f16(p_q24)
                                    require(not invalid and p_bits == int(half_bits(
                                        p_q24 / (1 << 24))[0]), "FP16 oracle disagreement")
                                    ideal = exps[column] / sum(exps)
                                    approximate_bits = int(half_bits(float(ideal))[0])
                                    neighbors = range(max(0, approximate_bits - 1),
                                                      min(0x3c00, approximate_bits + 1) + 1)
                                    ideal_bits = min(neighbors, key=lambda b: (
                                        abs(ideal - Decimal.from_float(float(values([b])[0]))), b & 1))
                                    probabilities.append(p_bits)
                                    witnesses.append({
                                        "index": head * 4 + column,
                                        "scores_bits": [f"{int(b):04x}" for b in bits],
                                        "exponentials_decimal80": [str(x) for x in exps],
                                        "ideal_exp_q24": quantized, "lut_exp_q24": approx,
                                        "probability_q24": p_q24, "candidate_bits": f"{p_bits:04x}",
                                        "ideal_probability_decimal80": str(ideal),
                                        "ideal_softmax_RNE_bits": f"{ideal_bits:04x}",
                                    })
                        softmax_rows[key] = {"bits": probabilities, "witnesses": witnesses,
                                             "scores_bits": state[8].tolist()}
                    require(softmax_rows[key]["scores_bits"] == state[8].tolist(),
                            "V history unexpectedly affected scores")
                    state[9] = np.asarray(softmax_rows[key]["bits"], dtype="<u2")
                elif affected:
                    require(stage in reconstructed, f"missing retained operator proof {layer}/{stage}")
                    state[stage] = ops.rtl_model(stage, state, incoming, cache)
                changed[stage] = not np.array_equal(state[stage], actual[stage])
                if stage in (6, 7):
                    cache["k" if stage == 6 else "v"][3] = state[stage].reshape(2, 64)
                if affected:
                    stage_rows.append({
                        "stage": stage, "changed": changed[stage],
                        "vs_actual": compared(state[stage], actual[stage]),
                        "vs_original_independent": compared(state[stage], independent[stage]),
                    })
                saved[f"{name}_{stage:02d}"] = state[stage]
            saved[f"{name}_incoming"] = incoming
            case["hidden"] = state[last].copy()
            row = {"layer": layer, "affected_stages": stage_rows,
                   "endpoint_vs_original_independent": compared(state[last], independent[last])}
            case["layers"].append(row)
            layer_cases[name] = row
            if layer == 0 and name == "exp1_k0_v0":
                conditional = ops.independent(9, actual, hidden, caches)
                candidate_matches_retained_seed = np.array_equal(state[9], conditional)
                saved["retained_conditional_stage09"] = conditional
                for witness in source_result["local_diagnostics"]["layer00_stage09"]["witnesses"]:
                    require(int(state[9][witness["index"]]) == int(witness["ideal_exp_only_bits"], 16),
                            "retained Decimal witness disagreement")
            retained_name = {
                "exp1_k0_v0": "layer00_stage09_actual",
                "exp0_k1_v0": "layer00_historical_k_actual",
                "exp0_k0_v1": "layer00_historical_v_actual",
            }.get(name)
            if retained_name and (name != "exp1_k0_v0" or candidate_matches_retained_seed):
                old_endpoint = prior["counterfactuals"][retained_name]["layers"][layer]["output"]
                require(np.array_equal(state[last], evidence.array(old_endpoint["path"])),
                        f"retained conditional regression {name}/{layer}")
        require(all(np.array_equal(saved[f"exp0_k0_v0_{s:02d}"], actual[s])
                    for s in range(last + 1)), f"unchanged control {layer}")
        path = out / f"layer{layer:02d}_stages.npz"
        with path.open("xb") as stream:
            np.savez(stream, **saved)
        stage_outputs[layer] = saved
        report = {"layer": layer, "arrays": record(path),
                  "array_keys": "actual_SS, independent_SS, expE_kK_vV_SS; SS is decimal stage",
                  "actual_trace": transaction["raw"]["trace"],
                  "operator_proof": record(retained / f"layer{layer:02d}.json"),
                  "cases": layer_cases, "seconds": time.monotonic() - layer_started}
        report_path = out / f"layer{layer:02d}.json"
        write_json(report_path, report)
        layer_reports.append(record(report_path))
        previous_actual = actual[18]
        print(json.dumps({"layer": layer, "seconds": report["seconds"]}), flush=True)
    require(compared(stage_outputs[8]["actual_08"],
                     stage_outputs[8]["independent_08"])["failure_indices"] == [13, 15],
            "original failure changed")
    effects = []
    for factor, bit in (("exp", 0), ("k", 1), ("v", 2)):
        for exp in range(2):
            for k in range(2):
                for v in range(2):
                    flags = [exp, k, v]
                    if flags[bit]:
                        continue
                    low = f"exp{exp}_k{k}_v{v}"
                    flags[bit] = 1
                    high = f"exp{flags[0]}_k{flags[1]}_v{flags[2]}"
                    effects.append({
                        "factor": factor, "low": low, "high": high,
                        "stage_effects": [
                            {"layer": layer, "stage": stage,
                             "high_vs_low": compared(arrays[f"{high}_{stage:02d}"],
                                                     arrays[f"{low}_{stage:02d}"])}
                            for layer, arrays in stage_outputs.items()
                            for stage in range(9 if layer == 8 else 19)
                            if not np.array_equal(arrays[f"{high}_{stage:02d}"],
                                                  arrays[f"{low}_{stage:02d}"])
                        ],
                    })
    summary = {}
    for name, case in cases.items():
        failures = [{"layer": row["layer"], "stage": stage["stage"],
                     "indices": stage["vs_original_independent"]["failure_indices"]}
                    for row in case["layers"] for stage in row["affected_stages"]
                    if stage["vs_original_independent"]["failure_indices"]]
        summary[name] = {
            "layer08_stage08": case["layers"][-1]["endpoint_vs_original_independent"],
            "affected_stage_failures": failures,
            "all_observed_affected_stages_within_policy": not failures,
        }
    result = {
        "status": "BOUNDED_SOFTMAX_KV_DISCRIMINATION_COMPLETE_PRODUCER_NOT_UNIQUELY_IDENTIFIED",
        "cases": summary, "factor_effects": effects, "softmax_oracle_rows": softmax_rows,
        "candidate_bit_identical_to_retained_softmax_cut": candidate_matches_retained_seed,
        "layers": layer_reports, "authenticated_inputs": list(evidence.consumed.values()),
        "production_intervention_authorized": False, "repaired_path_claim": False,
        "reason": (
            "The factorial isolates a specified layer0 stage09 exponential-policy effect from "
            "layer0 historical K and V substitutions on retained operands, with all changed-cone "
            "stages through layer08 scores preserved. It does not localize historical cache producers "
            "or other upstream differences; later-layer historical caches remain actual. A "
            "conditional endpoint is not a whole-trajectory repair. No synthesizable replacement "
            "or affected-prefix state rebuild has been implemented or executed."
        ),
        "regressions": {
            "original_failure": "unchanged layer08 stage08 indices13,15",
            "unchanged_control": "all retained stages bit-identical",
            "retained_single_factor_endpoints": {
                "K_only_and_V_only": "all 9 layers bit-identical",
                "softmax_only": ("all 9 layers bit-identical" if candidate_matches_retained_seed
                                 else "different specified seed; endpoint equality not required"),
            },
            "specified_exp_oracle": "all 28 rows checked by Decimal, math.exp and rational RNE",
        },
        "hypothesis_regression": {
            "taxonomy": "conditional_endpoint_insufficient_for_production_attribution",
            "hypothesis": "A passing softmax-only endpoint need not preserve every affected-stage gate",
            "regression": "exp1_k0_v0: all affected stages vs frozen independent trajectory",
        },
        "boundary": "Controlled software numerical diagnostic, not fresh RTL acceptance or a third token",
        "RTL_invocations": 0, "official_attempts": 0, "prefix_replays": 0,
        "acceptance_policy_changes": 0, "review": "pending normal Host Reviewer",
        "seconds": time.monotonic() - started,
    }
    write_json(out / "result.json", result)
    write_json(out / "seal.json", {"files": [record(p) for p in sorted(out.iterdir())]})
    print(json.dumps({"status": result["status"], "cases": {
        name: {"endpoint_failures": row["layer08_stage08"]["failure_indices"],
               "affected_stage_failures": row["affected_stage_failures"]}
        for name, row in summary.items()}, "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--sweep-from", type=Path)
    parser.add_argument("--repair-source-from", type=Path)
    parser.add_argument("--softmax-kv-from", type=Path)
    parser.add_argument("--source-diagnostic-from", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    require(sum(x is not None for x in (
        args.sweep_from, args.repair_source_from, args.softmax_kv_from)) <= 1,
        "diagnostic modes are exclusive")
    require((args.softmax_kv_from is None) == (args.source_diagnostic_from is None),
            "softmax/KV requires the retained source diagnostic")
    if args.softmax_kv_from:
        require(out.parent == ROOT / "build/token358_position3_softmax_kv_discrimination_attempt001",
                "softmax/KV output namespace")
    elif args.repair_source_from:
        require(out.parent == ROOT / "build/token358_position3_upstream_hidden_trajectory_repair_source_attempt001",
                "repair-source output namespace")
    else:
        require(out.parent.parent == ROOT / "build" and out.parent.name.startswith(
            "token358_position3_upstream_hidden_trajectory_localization"), "output namespace")
    out.mkdir(exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    try:
        if args.softmax_kv_from:
            softmax_kv_discrimination(out, args.command.resolve(), args.softmax_kv_from.resolve(),
                                     args.source_diagnostic_from.resolve())
        elif args.repair_source_from:
            repair_source(out, args.command.resolve(), args.repair_source_from.resolve())
        else:
            run(out, args.command.resolve(),
                args.sweep_from.resolve() if args.sweep_from is not None else None)
    except (RuntimeError, ValueError, OSError, KeyError, ArithmeticError) as error:
        write_json(out / "failure.json", {
            "taxonomy": "retained_operand_diagnostic_reconstruction_failure",
            "hypothesis": str(error),
            "regression": "The named retained operand/operator must reconstruct exactly before any counterfactual is interpreted",
            "RTL_invocations": 0,
        })
        raise
