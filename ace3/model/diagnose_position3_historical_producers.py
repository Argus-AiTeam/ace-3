#!/usr/bin/env python3
"""Discriminate later historical K/V dependence using a sealed position-3 cone."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import diagnose_position3_early_hidden_cone as cone
from diagnose_position3_upstream_hidden import Evidence, compared, record, require, write_json


def specifications():
    cases = {}
    for exp in (0, 1):
        for k in (0, 1):
            for v in (0, 1):
                cases[f"exp{exp}_later_k{k}_v{v}"] = {
                    "exp": exp, "layers": list(range(1, 9)), "positions": [0, 1, 2],
                    "kinds": [kind for kind, enabled in (("k", k), ("v", v)) if enabled],
                }
        for kind in ("k", "v"):
            for layer in range(1, 9):
                cases[f"exp{exp}_layer{layer:02d}_{kind}"] = {
                    "exp": exp, "layers": [layer], "positions": [0, 1, 2], "kinds": [kind],
                }
            for position in range(3):
                cases[f"exp{exp}_position{position}_{kind}"] = {
                    "exp": exp, "layers": list(range(1, 9)), "positions": [position],
                    "kinds": [kind],
                }
    return cases


def arrays(evidence, path):
    with np.load(io.BytesIO(evidence.read(path)), allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    require(all(x.dtype == np.dtype("<u2") for x in result.values()), "non-FP16 archive")
    return result


def run(out, command, retained):
    started = time.monotonic()
    torch.set_num_threads(1)
    evidence = Evidence()
    root = record(retained / "seal.json")
    evidence.register(root)
    seal = evidence.load(root["path"])
    for item in seal["files"]:
        evidence.read(item["path"])
    prior = evidence.load(retained / "result.json")
    old_frozen = json.loads(evidence.read(retained / "frozen.json"))
    require(hashlib.sha256(evidence.read(retained / "source.py")).hexdigest()
            == old_frozen["source"]["sha256"], "retained source snapshot mismatch")
    require(record(cone.__file__)["sha256"] == old_frozen["source"]["sha256"],
            "live cone differs from retained operator implementation")
    evidence.register(prior["authenticated_inputs"])
    for key in ("helpers", "production_sources"):
        evidence.register(old_frozen[key])
        for item in old_frozen[key]:
            evidence.read(item["path"])
    rows, plan_roots, frozen, _ = cone.plans(evidence)
    require(frozen["comparison_policy"] == old_frozen["comparison_policy"], "gate drift")
    require(prior["RTL_invocations"] == prior["official_attempts"] == 0,
            "unexpected prior evidence boundary")
    sources = [record(module.__file__) for module in tuple(sys.modules.values())
               if getattr(module, "__file__", None)
               and Path(module.__file__).resolve().is_relative_to(cone.ROOT / "ace3/model")
               and Path(module.__file__).suffix == ".py"]
    cases = specifications()
    write_json(out / "frozen.json", {
        "question": (
            "Does layer0 stage09 candidate sufficiency survive replacing later-layer historical "
            "K/V, and can an individually localized later history substitution also clear the "
            "original endpoint? This expands the prior factorial's fixed-history boundary."
        ),
        "source": record(__file__), "command": record(command), "sources": sources,
        "roots": [root, *plan_roots], "comparison_policy": frozen["comparison_policy"],
        "cases": cases, "candidate": old_frozen["candidate"],
        "held_fixed": (
            "All layer0 historical/current K/V and incoming hidden; original independently "
            "propagated position3 expectations; unchanged production operators except the "
            "retained layer0 stage09 software candidate seed. Positions0-2 only are "
            "substituted in layers1-8; current-position K/V follow each changed cone."
        ),
        "source_boundary": (
            "Reuses authenticated operator reconstruction proofs, not new RTL evidence. "
            "Independent-history substitution is a diagnostic intervention, not a producer "
            "implementation. Layer1 projection conversion artifacts are not repair evidence. "
            "Retained seals authenticate bytes; no pending Reviewer verdict is assumed."
        ),
        "public_RTL_contract": (
            "No RTL module, port, parameter or arithmetic source changes; no RTL compile "
            "or execution requested by this software-only diagnostic."
        ),
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "device": "cpu", "threads": 1, "stop": "layer08 stage08",
        "RTL_invocations": 0, "official_attempts": 0, "prefix_replays": 0,
    })
    scope = cone.namespace(evidence)
    incoming_by_case = {}
    summaries = {name: {"changed_stages": [], "affected_stage_failures": []} for name in cases}
    reports, history_differences, source_gaps = [], [], []
    previous_actual = None
    for layer, receipt, prepared in rows:
        tick = time.monotonic()
        retained_report = evidence.load(retained / f"layer{layer:02d}.json")
        saved = arrays(evidence, retained_report["arrays"]["path"])
        proof = evidence.load(retained_report["operator_proof"]["path"])
        ref = evidence.load(prepared["independent_parent"]["path"])
        transaction = evidence.load(receipt["transaction"]["path"])
        require(proof["layer"] == ref["layer"] == receipt["layer"] == layer
                and receipt["position"] == 3 and ref["history"] == [9707, 1879, 0],
                f"layer identity {layer}")
        actual = evidence.trace(transaction["raw"]["trace"]["path"], 3)
        last = 8 if layer == 8 else 18
        independent = {s: evidence.bits(prepared["independent_stages"][str(s)]["path"])
                       for s in range(last + 1)}
        hidden = evidence.bits(prepared["vectors"]["input"]["path"], True)
        require(np.array_equal(hidden, saved["actual_incoming"]), f"incoming binding {layer}")
        require(all(np.array_equal(actual[s], saved[f"actual_{s:02d}"])
                    and np.array_equal(independent[s], saved[f"independent_{s:02d}"])
                    for s in range(last + 1)), f"retained stage binding {layer}")
        if layer:
            require(np.array_equal(previous_actual, hidden), f"hidden lineage {layer}")
        parent = prepared["kv_parent"]
        require(parent["layer_index"] == layer and parent["valid_positions"] == [0, 1, 2]
                and proof["actual_kv_parents"] == parent["actual_kv_traces"]
                and proof["independent_kv_parents"] == ref["own_cache"],
                f"historical parent binding {layer}")
        old = [evidence.trace(item["path"], position, 7)
               for position, item in enumerate(parent["actual_kv_traces"])]
        cache = {kind: np.stack([x[stage].reshape(2, 64) for x in old]
                               + [actual[stage].reshape(2, 64)])
                 for kind, stage in (("k", 6), ("v", 7))}
        independent_old = {kind: evidence.array(ref["own_cache"][kind]["path"])
                           for kind in ("k", "v")}
        require(all(x.shape == (3, 2, 64) for x in independent_old.values()),
                f"historical geometry {layer}")
        require(all(np.array_equal(cache[kind], saved[f"actual_cache_{kind}"])
                    for kind in ("k", "v")), f"cache archive binding {layer}")
        for kind in ("k", "v"):
            for position in range(3):
                delta = compared(cache[kind][position], independent_old[kind][position])
                history_differences.append({
                    "layer": layer, "position": position, "kind": kind, "comparison": delta,
                    "actual_trace": parent["actual_kv_traces"][position],
                    "independent_cache": ref["own_cache"][kind],
                })
        ops = cone.Operators(layer, cone.load_tensors(evidence, prepared, ref), scope)
        reconstructed = set(proof["rtl_model_reconstructed_stages"])
        layer_cases, output_arrays = {}, {
            **{f"actual_{s:02d}": actual[s] for s in range(last + 1)},
            **{f"independent_{s:02d}": independent[s] for s in range(last + 1)},
            **{f"actual_cache_{kind}": x for kind, x in cache.items()},
            **{f"independent_old_{kind}": x for kind, x in independent_old.items()},
        }
        for name, spec in cases.items():
            exp = spec["exp"]
            if layer == 0:
                state = {s: saved[f"exp{exp}_k0_v0_{s:02d}"].copy() for s in range(19)}
                # The candidate seed is authenticated and independently recomputed, not inferred
                # from whether it eventually clears a selected downstream witness.
                if exp:
                    candidate = []
                    from fractions import Fraction
                    for bits in state[8].reshape(14, 4):
                        scores = cone.values(bits).tolist()
                        exps = [round(math.exp(x - max(scores)) * (1 << 24)) for x in scores]
                        q24 = [round(Fraction(x << 24, sum(exps))) for x in exps]
                        candidate.extend(np.asarray(q24, dtype=np.float64)
                                         .__truediv__(1 << 24).astype("<f2").view("<u2"))
                    require(np.array_equal(np.asarray(candidate, dtype="<u2"), state[9]),
                            "independent retained exponential seed regression")
                affected = [x["stage"] for x in retained_report["cases"][
                    f"exp{exp}_k0_v0"]["affected_stages"]]
                incoming = hidden
            else:
                state = {s: actual[s].copy() for s in range(last + 1)}
                incoming = incoming_by_case[name]
                current_cache = {kind: x.copy() for kind, x in cache.items()}
                if layer in spec["layers"]:
                    for kind in spec["kinds"]:
                        current_cache[kind][spec["positions"]] = independent_old[kind][spec["positions"]]
                changed = {
                    -1: not np.array_equal(incoming, hidden),
                    -2: not np.array_equal(current_cache["k"], cache["k"]),
                    -3: not np.array_equal(current_cache["v"], cache["v"]),
                }
                affected = []
                for stage in range(last + 1):
                    if any(changed.get(dep, False) for dep in cone.DEPS[stage]):
                        require(stage in reconstructed, f"missing operator proof {layer}/{stage}")
                        state[stage] = ops.rtl_model(stage, state, incoming, current_cache)
                        affected.append(stage)
                    changed[stage] = not np.array_equal(state[stage], actual[stage])
                    if stage in (6, 7):
                        current_cache["k" if stage == 6 else "v"][3] = state[stage].reshape(2, 64)
                for kind in ("k", "v"):
                    output_arrays[f"{name}_cache_{kind}"] = current_cache[kind]
            compared_stages = []
            for stage in affected:
                comparison = compared(state[stage], independent[stage])
                compared_stages.append({"stage": stage, "vs_original_independent": comparison,
                                        "vs_actual": compared(state[stage], actual[stage])})
                if not np.array_equal(state[stage], actual[stage]):
                    summaries[name]["changed_stages"].append([layer, stage])
                if comparison["failure_indices"]:
                    summaries[name]["affected_stage_failures"].append({
                        "layer": layer, "stage": stage, "indices": comparison["failure_indices"],
                    })
            if not spec["kinds"]:
                require(all(np.array_equal(state[s], saved[f"exp{exp}_k0_v0_{s:02d}"])
                            for s in range(last + 1)), f"retained control regression {name}/{layer}")
            output_arrays.update({f"{name}_{s:02d}": state[s] for s in range(last + 1)})
            output_arrays[f"{name}_incoming"] = incoming
            incoming_by_case[name] = state[last].copy()
            layer_cases[name] = {
                "affected_stages": compared_stages,
                "endpoint_vs_original_independent": compared(state[last], independent[last]),
            }
            if layer == 8:
                summaries[name]["layer08_stage08"] = layer_cases[name]["endpoint_vs_original_independent"]
                summaries[name]["all_affected_stage_gates_clear"] = (
                    not summaries[name]["affected_stage_failures"])
                summaries[name]["endpoint_and_affected_gates_clear"] = (
                    not summaries[name]["layer08_stage08"]["failure_indices"]
                    and not summaries[name]["affected_stage_failures"])
        if layer == 8:
            require(compared(actual[8], independent[8])["failure_indices"] == [13, 15],
                    "original failure witnesses changed")
        path = out / f"layer{layer:02d}_stages.npz"
        with path.open("xb") as stream:
            np.savez(stream, **output_arrays)
        report_path = out / f"layer{layer:02d}.json"
        write_json(report_path, {
            "layer": layer, "arrays": record(path), "retained_report": record(
                retained / f"layer{layer:02d}.json"), "cases": layer_cases,
            "seconds": time.monotonic() - tick,
        })
        reports.append(record(report_path))
        previous_actual = actual[18]
        print(json.dumps({"layer": layer, "seconds": time.monotonic() - tick}), flush=True)
    passing = [name for name, summary in summaries.items()
               if summary["endpoint_and_affected_gates_clear"]]
    source_gaps.extend([
        "Historical K/V substitutions exchange independently propagated values, not one "
        "source-bound production operator. They do not distinguish the earlier hidden input, "
        "RMSNorm, K/V projection rounding and K RoPE producers of those histories.",
        "The exponential candidate is applied at layer0 position3 only. A general rollout "
        "also affects earlier-position outputs and potentially later-layer cached states, "
        "which these mixed-history branches do not regenerate.",
    ])
    result = {
        "status": "BOUNDED_HISTORICAL_FACTORISATION_COMPLETE_NONUNIQUE_PRODUCER",
        "cases": summaries, "passing_conditional_cases": passing, "layers": reports,
        "historical_differences": history_differences, "unresolved_producer_boundaries": source_gaps,
        "production_intervention_authorized": False,
        "authorization_reason": (
            "Changed-cone software sufficiency and history sensitivity do not identify an "
            "implemented, independently reviewed production intervention or its valid state "
            "rebuild boundary. This does not veto Planner selecting the isolated general "
            "exponential RTL candidate after genuine review of its bounded evidence."
        ),
        "hypothesis_regression": {
            "taxonomy": "historical_state_and_operator_policy_confounding",
            "hypothesis": "The prior stage09 sufficiency depended on fixed later-layer histories",
            "regression": (
                "52 preregistered branches cross the two retained layer0 seeds with later "
                "K/V joint, individual-layer and individual-position substitutions."
            ),
        },
        "regressions": {
            "original_failure": "layer08 stage08 indices13,15 preserved",
            "retained_controls": "all stages of both no-history-change controls bit-identical",
            "candidate_seed": "independent binary64 exponential plus rational Q24 RNE and FP16",
        },
        "authenticated_inputs": list(evidence.consumed.values()),
        "source_end_bindings": sources,
        "RTL_invocations": 0, "official_attempts": 0, "prefix_replays": 0,
        "production_changes": 0, "acceptance_policy_changes": 0,
        "review": "pending normal Host Reviewer; not invoked by Engineer",
        "boundary": "Software counterfactuals only; no repaired trajectory or third token",
        "seconds": time.monotonic() - started,
    }
    for source in sources:
        require(record(source["path"]) == source, f"source changed during run: {source['path']}")
    write_json(out / "result.json", result)
    write_json(out / "seal.json", {"files": [record(p) for p in sorted(out.iterdir())]})
    print(json.dumps({
        "status": result["status"], "cases": len(cases),
        "passing_conditional_cases": passing, "seconds": result["seconds"],
        "production_intervention_authorized": False, "official_attempts": 0,
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.parent.parent == cone.ROOT / "build"
            and out.parent.name.startswith("token358_position3_historical_producer_"),
            "output outside bounded diagnostic namespace")
    out.mkdir(exist_ok=False)
    with (out / "source.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    try:
        run(out, args.command.resolve(), args.retained.resolve())
    except (RuntimeError, ValueError, OSError, KeyError, ArithmeticError) as error:
        write_json(out / "failure.json", {
            "taxonomy": "historical_cone_evidence_or_reconstruction_failure",
            "hypothesis": str(error),
            "regression": "Authenticate the named binding and both retained controls before inference",
            "RTL_invocations": 0, "official_attempts": 0,
        })
        raise
